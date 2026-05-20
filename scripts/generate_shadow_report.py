"""CLI: shadow validation report (Phase 2 S3).

Usage:
  python scripts/generate_shadow_report.py --source synthetic --n-heats 500
  python scripts/generate_shadow_report.py --source heats_db
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


def _build_synthetic_records(n_heats: int):
    """Synthetic deox heats → list[HeatRecord] (target O_a = 5 ppm)."""
    from app.backend.data_curator import generate_synthetic_deox_calibration_dataset
    from app.backend.deoxidation import AL_TO_O_MASS_RATIO
    from app.backend.heat_records import HeatRecord

    df = generate_synthetic_deox_calibration_dataset(n_heats=n_heats)
    records: list[HeatRecord] = []
    for _, row in df.iterrows():
        o_init = float(row["o_a_initial_ppm"])
        o_after = 5.0
        mass = float(row["steel_mass_ton"])
        eta = float(row["eta_al_effective"])
        # Back-out фактической Al-дозы из η: O_removed[kg] × stoich / η
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
                al_residual_pct=0.025,
                slag_feo_pct=float(row["slag_feo_pct"]),
                slag_mass_kg=float(row["slag_mass_kg"]),
                t_al_addition_c=float(row["t_al_addition_c"]),
            )
        )
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Shadow validation report (Phase 2 S3)")
    ap.add_argument(
        "--source", choices=["synthetic", "heats_db"], default="heats_db"
    )
    ap.add_argument("--n-heats", type=int, default=500)
    ap.add_argument("--al-price", type=float, default=2.40)
    args = ap.parse_args()

    from app.backend.eta_al_calibration import EtaAlCalibrator
    from app.backend.eta_al_predictor import EtaAlPredictor
    from app.backend.shadow_reporter import generate_shadow_report
    from app.backend.shadow_stats import compute_shadow_stats
    from app.backend.shadow_validation import run_shadow_comparison

    is_synthetic = args.source == "synthetic"
    if is_synthetic:
        from app.backend.heat_records import bulk_insert_heats

        td = Path(tempfile.mkdtemp(prefix="shadow_synth_"))
        db = td / "heats.db"
        records = _build_synthetic_records(args.n_heats)
        bulk_insert_heats(records, db_path=db)
        cal = EtaAlCalibrator(db_path=db, calibrations_dir=td / "calib")
        cal.calibrate_all_plants()
        # model_version=None → predictor uses plant posterior + literature prior
        pred = EtaAlPredictor(calibrator=cal)
        comps = run_shadow_comparison(records, pred)
    else:
        cal = EtaAlCalibrator()
        cal.calibrate_all_plants()
        pred = EtaAlPredictor(calibrator=cal)
        comps = run_shadow_comparison(None, pred)  # heats_db

    stats = compute_shadow_stats(comps)
    html, jsn = generate_shadow_report(
        comps,
        stats,
        al_price_eur_per_kg=args.al_price,
        is_synthetic=is_synthetic,
    )
    print(f"n_compared={stats.n_compared}, n_qpass={stats.n_quality_pass}")
    print(
        f"median Δ={stats.median_delta_pct:.1f}%, "
        f"hypothesis_met={stats.hypothesis_met}"
    )
    print(
        f"Al saved={stats.total_al_saved_kg:.0f}kg, "
        f"€{stats.total_al_saved_kg * args.al_price:.0f}"
    )
    print(f"HTML: {html}")
    print(f"JSON: {jsn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
