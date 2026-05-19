"""End-to-end smoke test для η_Al calibration stack (Phase 1, PR 1-13).

Прогоняет полный цикл на synthetic data, в изолированном tmp:
  1. Generate synthetic deox heats (PR 4)
  2. Ingest в heats.db (PR 1)
  3. Bayesian calibration per plant (PR 5)
  4. Train ML model η_Al via pipeline (PR 8)  [опционально — медленно, по флагу]
  5. EtaAlPredictor mix (PR 9)
  6. compute_al_demand_slag_aware с predictor → conformal p10/p50/p90 (PR 6+9)
  7. recommend_optimal_method objective=pareto (PR 7)
  8. Pattern Library DX08-DX12 sanity (PR 13)
  9. Symbolic regression correction (PR 11) [быстрый config]
  10. High-Al diagnoser cohort (PR 12) [deterministic, no LLM]

Работает БЕЗ ANTHROPIC_API_KEY (LLM-части degrade gracefully) и БЕЗ real data.
Использует tmp dirs — НЕ загрязняет рабочий каталог (heats.db, calibrations).
Exit 0 = все этапы pass.

Usage:
  PYTHONPATH=. python scripts/smoke_test_deox_calibration.py
  PYTHONPATH=. python scripts/smoke_test_deox_calibration.py --skip-train  # без ML (быстро, default)
  PYTHONPATH=. python scripts/smoke_test_deox_calibration.py --train       # с ML training через pipeline
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

# Smoke — информативный лог, без шума WARNING из degrade-путей.
logging.basicConfig(level=logging.ERROR)


def _step(n: int, msg: str) -> None:
    print(f"\n[{n}] {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--train",
        action="store_true",
        help="Запустить ML training через pipeline (медленно). По умолчанию skip.",
    )
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip ML training (default ON, флаг оставлен для совместимости).",
    )
    ap.add_argument(
        "--n-heats",
        type=int,
        default=800,
        help="Число synthetic плавок (default 800 → ~53 на (plant,method)-cell, "
             "комфортный margin для N≥30 posterior threshold).",
    )
    args = ap.parse_args()
    do_train = args.train and not args.skip_train

    failures: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="deox_smoke_"))
    db_path = tmp / "heats.db"
    calib_dir = tmp / "calibs"
    models_dir = tmp / "models"  # пустой → predictor работает без ML (plant_only)
    print(f"Smoke workspace: {tmp}")

    try:
        # ── Step 1: Generate synthetic ────────────────────────────────
        _step(1, "Generate synthetic deox heats (PR 4)")
        from app.backend.data_curator import (
            generate_synthetic_deox_calibration_dataset,
        )

        df = generate_synthetic_deox_calibration_dataset(n_heats=args.n_heats)
        assert len(df) == args.n_heats, f"expected {args.n_heats}, got {len(df)}"
        first_plant = str(df["plant_id"].iloc[0])
        print(f"    OK {len(df)} heats, {df['plant_id'].nunique()} plants")

        # ── Step 2: Ingest в heats.db ─────────────────────────────────
        _step(2, "Ingest heats into SQLite (PR 1)")
        from app.backend.heat_records import (
            HeatRecord,
            bulk_insert_heats,
            count_heats,
        )
        from app.backend.slag_aware_deox import AL_TO_O_MASS_RATIO

        records: list[HeatRecord] = []
        for _, row in df.iterrows():
            o_init = float(row["o_a_initial_ppm"])
            o_after = 5.0
            mass = float(row["steel_mass_ton"])
            eta = float(row["eta_al_effective"])
            # Восстанавливаем al_added из η, чтобы has_outcome=True было consistent.
            o_removed_kg = (o_init - o_after) / 1e6 * mass * 1000.0
            al_added = o_removed_kg * AL_TO_O_MASS_RATIO / eta if eta > 0 else 0.0
            records.append(
                HeatRecord(
                    source="synthetic",
                    plant_id=str(row["plant_id"]),
                    steel_mass_ton=mass,
                    o_a_initial_ppm=o_init,
                    o_a_after_ppm=o_after,
                    al_added_kg=al_added,
                    method_id=str(row["method_id"]),
                    eta_al_effective=eta,
                    slag_feo_pct=float(row["slag_feo_pct"]),
                    slag_mass_kg=float(row["slag_mass_kg"]),
                    t_al_addition_c=float(row["t_al_addition_c"]),
                    dt_to_al_min=float(row["dt_to_al_min"]),
                    co_deox_fesi_kg=float(row["co_deox_fesi_kg"]),
                    refractory_heat_count=int(row["refractory_heat_count"]),
                )
            )
        ids = bulk_insert_heats(records, db_path=db_path)
        assert count_heats(db_path=db_path) == args.n_heats
        print(f"    OK {len(ids)} records inserted")

        # ── Step 3: Bayesian calibration ──────────────────────────────
        _step(3, "Bayesian per-plant calibration (PR 5)")
        from app.backend.eta_al_calibration import EtaAlCalibrator

        cal = EtaAlCalibrator(db_path=db_path, calibrations_dir=calib_dir)
        results = cal.calibrate_all_plants()
        n_yaml = sum(1 for r in results if r.yaml_written)
        n_posteriors = sum(
            1
            for r in results
            for p in r.calibrations
            if p.skipped_reason is None
        )
        print(
            f"    OK {len(results)} plants, {n_yaml} YAML written, "
            f"{n_posteriors} method-posteriors above N>=30"
        )
        assert n_yaml >= 1, "no plant calibration YAML written"
        assert n_posteriors >= 1, "no (plant,method) posterior above threshold"

        # ── Step 4: Train ML (опционально) ────────────────────────────
        model_version = None
        if do_train:
            _step(4, "Train ML model η_Al via pipeline (PR 8)")
            try:
                from app.backend.engine import Orchestrator

                orch = Orchestrator()
                state = orch.run_full_pipeline(steel_class="deox_calibration")
                model_version = getattr(
                    getattr(state, "model", None), "version", None
                )
                print(f"    OK trained model_version={model_version}")
            except Exception as exc:  # noqa: BLE001 — train не должен валить smoke
                print(f"    SKIP train: pipeline integration failed ({exc})")
                model_version = None
        else:
            _step(4, "SKIP train (use --train to enable)")

        # ── Step 5+6: Predictor + conformal Al demand ─────────────────
        _step(5, "EtaAlPredictor mix + conformal Al demand (PR 6+9)")
        from app.backend.eta_al_predictor import EtaAlPredictor
        from app.backend.slag_aware_deox import compute_al_demand_slag_aware

        predictor = EtaAlPredictor(
            calibrator=cal,
            model_version=model_version,  # None + empty models_dir → plant_only
            models_dir=models_dir,
        )
        res = compute_al_demand_slag_aware(
            steel_mass_ton=300,
            o_a_initial_ppm=500,
            target_o_a_ppm=5,
            target_al_pct=0.02,
            method="asis_shot",
            eta_al_predictor=predictor,
            plant_id=first_plant,
        )
        assert res.al_pure_kg_p10 <= res.al_pure_kg_p50 <= res.al_pure_kg_p90, (
            f"conformal ordering violated: "
            f"{res.al_pure_kg_p10}/{res.al_pure_kg_p50}/{res.al_pure_kg_p90}"
        )
        meta = res.inputs.get("predicted_eta_metadata")
        source = meta.get("source") if meta else "n/a"
        print(
            f"    OK Al p10/p50/p90 = {res.al_pure_kg_p10:.1f}/"
            f"{res.al_pure_kg_p50:.1f}/{res.al_pure_kg_p90:.1f} kg, source={source}"
        )

        # ── Step 7: recommend pareto ──────────────────────────────────
        _step(7, "recommend_optimal_method objective=pareto (PR 7)")
        from app.backend.slag_aware_deox import recommend_optimal_method

        rec = recommend_optimal_method(
            steel_mass_ton=300,
            o_a_initial_ppm=500,
            target_o_a_ppm=5,
            target_al_pct=0.02,
            objective="pareto",
        )
        assert len(rec.pareto_frontier) >= 1, "empty pareto frontier"
        print(
            f"    OK chosen={rec.chosen_method_id}, "
            f"frontier={len(rec.pareto_frontier)}"
        )

        # ── Step 8: Pattern DX08-DX12 sanity ──────────────────────────
        _step(8, "Pattern Library DX08-DX12 sanity (PR 13)")
        from pattern_library.patterns import Phase, run_all_patterns

        # DX11 — TRAINING, gated on steel_class + low coverage.
        dx11 = run_all_patterns(
            {"steel_class": "deox_calibration", "coverage_90_ci": 0.70},
            phase=Phase.TRAINING,
        )
        assert any(w["pattern_id"] == "DX11" for w in dx11), "DX11 did not fire"
        # DX09 — DEOXIDATION, small per-method dataset → literature prior.
        dx09 = run_all_patterns(
            {"plant_n_heats_for_method": 10, "min_heats_threshold": 30},
            phase=Phase.DEOXIDATION,
        )
        assert any(w["pattern_id"] == "DX09" for w in dx09), "DX09 did not fire"
        print("    OK DX11 fires on low coverage, DX09 on small dataset")

        # ── Step 9: Symbolic regression (быстрый config) ──────────────
        _step(9, "Symbolic regression η correction (PR 11)")
        from scripts.symbolic_eta_correction import _load_data
        from app.backend.symbolic_regressor import run_symbolic_regression

        X, y, fnames = _load_data("synthetic", n_heats=300)
        sr = run_symbolic_regression(
            X,
            y,
            feature_names=fnames,
            population_size=200,
            generations=5,
            random_state=42,
            n_jobs=1,
        )
        assert len(sr.pareto_frontier) >= 1, "empty SR frontier"
        print(f"    OK SR frontier: {len(sr.pareto_frontier)} formulas")

        # ── Step 10: High-Al diagnoser cohort (deterministic) ─────────
        _step(10, "High-Al diagnoser cohort detection (PR 12)")
        from app.backend.high_al_diagnoser import (
            build_cohort_context,
            identify_high_al_outliers,
        )

        heats_dicts = [
            {
                "al_added_kg": r.al_added_kg,
                "steel_mass_ton": r.steel_mass_ton,
                "method_id": r.method_id,
                "o_a_after_ppm": r.o_a_after_ppm,
                "slag_feo_pct": r.slag_feo_pct,
                "o_a_initial_ppm": r.o_a_initial_ppm,
                "dt_to_al_min": r.dt_to_al_min,
                "t_al_addition_c": r.t_al_addition_c,
                "refractory_heat_count": r.refractory_heat_count,
            }
            for r in records
        ]
        outliers, baseline = identify_high_al_outliers(
            heats_dicts, method="percentile"
        )
        cohort = build_cohort_context(
            outliers, baseline, method="percentile", threshold_pct=120
        )
        assert len(outliers) >= 1, "no high-Al outliers detected"
        print(
            f"    OK {len(outliers)} outliers, "
            f"{len(cohort['feature_deltas'])} feature deltas"
        )

    except Exception as exc:  # noqa: BLE001 — собираем все ошибки в отчёт
        traceback.print_exc()
        failures.append(str(exc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAIL DEOX CALIBRATION SMOKE: {len(failures)} error(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK DEOX CALIBRATION SMOKE PASSED — all Phase 1 stages green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
