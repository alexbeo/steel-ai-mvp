"""CLI: запустить Bayesian калибровку η_Al для одного plant'а или всех.

Usage:
  python scripts/calibrate_eta_al.py --plant-id PLANT_A
  python scripts/calibrate_eta_al.py --all
  python scripts/calibrate_eta_al.py --plant-id PLANT_A --dry-run --min-heats 20
"""
from __future__ import annotations

import argparse
import sys

from app.backend.eta_al_calibration import EtaAlCalibrator, MIN_HEATS_DEFAULT


def main() -> int:
    ap = argparse.ArgumentParser(description="Bayesian per-plant η_Al калибровка")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plant-id", help="Plant identifier")
    g.add_argument("--all", action="store_true", help="Calibrate all plants in DB")
    ap.add_argument("--min-heats", type=int, default=MIN_HEATS_DEFAULT)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute и print, без YAML write и Decision Log",
    )
    args = ap.parse_args()

    cal = EtaAlCalibrator(min_heats_for_posterior=args.min_heats)

    # Optional monkey-patch для dry-run (no side effects)
    eta_mod = None
    original_write = None
    dl_mod = None
    original_log = None
    if args.dry_run:
        from app.backend import eta_al_calibration as eta_mod
        original_write = eta_mod._write_yaml_atomic
        eta_mod._write_yaml_atomic = lambda *a, **kw: None
        try:
            from decision_log import logger as dl_mod
            original_log = dl_mod.log_decision
            dl_mod.log_decision = lambda *a, **kw: 0
        except ImportError:
            dl_mod = None

    try:
        if args.all:
            results = cal.calibrate_all_plants()
        else:
            results = [cal.calibrate_plant(args.plant_id)]
    finally:
        if args.dry_run:
            if eta_mod is not None and original_write is not None:
                eta_mod._write_yaml_atomic = original_write
            if dl_mod is not None and original_log is not None:
                dl_mod.log_decision = original_log

    print(f"\nCalibrated {len(results)} plant(s):")
    any_calibrated = False
    for r in results:
        print(f"\n=== {r.plant_id} (n_total={r.n_total_heats}) ===")
        if r.yaml_written:
            print(f"  YAML: {r.yaml_path}")
        elif args.dry_run:
            print("  (dry-run — no YAML write)")
        else:
            print("  YAML NOT written (no method passed threshold)")
        for p in r.calibrations:
            if p.skipped_reason:
                print(f"  {p.method_id:>20s}: SKIPPED ({p.skipped_reason})")
            else:
                print(
                    f"  {p.method_id:>20s}: n={p.n_heats_used:3d}, "
                    f"η={p.posterior_eta_mean:.3f} [{p.posterior_eta_q05:.3f}, "
                    f"{p.posterior_eta_q95:.3f}] (prior={p.prior_eta_mean:.3f})"
                )
                any_calibrated = True
    return 0 if any_calibrated or args.dry_run else 2


if __name__ == "__main__":
    sys.exit(main())
