"""E2E smoke для shadow validation pipeline (Phase 2 S1-S4).

Прогон на synthetic в tmp:
  1. Generate synthetic deox heats + ingest
  2. Calibrate plants (S5 использует existing posteriors)
  3. run_shadow_comparison (S1)
  4. compute_shadow_stats (S2)
  5. generate_shadow_report HTML+JSON (S3)

Synthetic тавтологичен (AI re-predicts свою η + residual budget) — smoke проверяет
МЕХАНИКУ (пайплайн отрабатывает, отчёт честен), НЕ подтверждение гипотезы.
Реальная −10% валидация — на real heats_db.

Usage: PYTHONPATH=. python scripts/smoke_test_shadow.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def _step(n, msg):
    print(f"\n[{n}] {msg}")


def main() -> int:
    failures = []
    tmp = Path(tempfile.mkdtemp(prefix="shadow_smoke_"))
    print(f"Smoke workspace: {tmp}")
    try:
        from app.backend.data_curator import (
            generate_synthetic_deox_calibration_dataset,
        )
        from app.backend.eta_al_calibration import EtaAlCalibrator
        from app.backend.eta_al_predictor import EtaAlPredictor
        from app.backend.heat_records import (
            HeatRecord,
            bulk_insert_heats,
            count_heats,
        )
        from app.backend.shadow_reporter import generate_shadow_report
        from app.backend.shadow_stats import compute_shadow_stats
        from app.backend.shadow_validation import run_shadow_comparison
        from app.backend.slag_aware_deox import AL_TO_O_MASS_RATIO

        db = tmp / "heats.db"

        _step(1, "Generate + ingest synthetic heats")
        df = generate_synthetic_deox_calibration_dataset(n_heats=500)
        records = []
        for _, row in df.iterrows():
            o_init = float(row["o_a_initial_ppm"])
            o_after = 5.0
            mass = float(row["steel_mass_ton"])
            eta = float(row["eta_al_effective"])
            # back-calc actual Al charge: O removed (kg) → Al demand / η
            o_removed = (o_init - o_after) / 1e6 * mass * 1000
            al_added = o_removed * AL_TO_O_MASS_RATIO / eta if eta > 0 else 0.0
            records.append(HeatRecord(
                source="synthetic", plant_id=str(row["plant_id"]),
                steel_mass_ton=mass, o_a_initial_ppm=o_init,
                o_a_after_ppm=o_after, al_added_kg=al_added,
                method_id=str(row["method_id"]), eta_al_effective=eta,
                al_residual_pct=0.025, slag_feo_pct=float(row["slag_feo_pct"]),
                slag_mass_kg=float(row["slag_mass_kg"]),
                t_al_addition_c=float(row["t_al_addition_c"]),
            ))
        bulk_insert_heats(records, db_path=db)
        assert count_heats(db_path=db) == 500
        print("    OK 500 heats")

        _step(2, "Calibrate plants")
        cal = EtaAlCalibrator(db_path=db, calibrations_dir=tmp / "calibs")
        cal_results = cal.calibrate_all_plants()
        n_cal = sum(1 for r in cal_results if r.yaml_written)
        print(f"    OK {n_cal} plants calibrated")

        _step(3, "run_shadow_comparison (S1)")
        pred = EtaAlPredictor(calibrator=cal, model_version="NONEXISTENT")
        comps = run_shadow_comparison(records, pred)
        compared = [c for c in comps if c.skip_reason is None]
        assert len(compared) > 0, "no heats compared"
        print(f"    OK {len(compared)}/{len(comps)} compared")

        _step(4, "compute_shadow_stats (S2)")
        stats = compute_shadow_stats(comps)
        assert stats.n_compared == len(compared)
        # НЕ ассертим hypothesis_met (synthetic тавтологичен)
        print(f"    OK median delta={stats.median_delta_pct_all:.1f}% (all), "
              f"quality-pass median={stats.median_delta_pct:.1f}%, "
              f"CI=[{stats.ci_low:.1f},{stats.ci_high:.1f}], "
              f"wilcoxon_p={stats.wilcoxon_p}, hypothesis_met={stats.hypothesis_met}")
        print("      (synthetic тавтологичен — hypothesis_met не проверяется)")

        _step(5, "generate_shadow_report HTML+JSON (S3)")
        html, jsn = generate_shadow_report(
            comps, stats, is_synthetic=True, out_dir=tmp,
        )
        assert html.exists() and jsn.exists()
        html_text = html.read_text(encoding="utf-8")
        assert "Retrospective" in html_text, "disclaimer missing"
        assert "synthetic" in html_text.lower(), "synthetic warning missing"
        import json
        data = json.loads(jsn.read_text(encoding="utf-8"))
        assert data["is_synthetic"] is True
        assert "stats" in data
        print("    OK HTML+JSON, disclaimers + synthetic warning present")

    except Exception as e:  # noqa: BLE001 — smoke собирает все ошибки
        import traceback
        traceback.print_exc()
        failures.append(str(e))
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAIL SHADOW SMOKE: {len(failures)} errors")
        return 1
    print("PASS SHADOW VALIDATION SMOKE — S1->S2->S3 pipeline OK (synthetic mechanics)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
