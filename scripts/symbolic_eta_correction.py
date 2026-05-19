"""CLI: symbolic regression поправки η_Al = method_baseline × correction(features).

Находит closed-form correction factor (data-driven аналог Excel k=0.89).
Опциональный LLM-критик оценивает physical plausibility.

Usage:
  python scripts/symbolic_eta_correction.py --source synthetic --n-heats 500
  python scripts/symbolic_eta_correction.py --source heats_db --critic
  python scripts/symbolic_eta_correction.py --no-critic --save-report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_data(source: str, n_heats: int):
    """Return (X DataFrame, y Series, feature_names) for SR.

    B2 target: correction = eta_al_effective / method_eta_baseline.
    Features = numeric feature_set минус baseline/offset/categorical/meta.
    """
    import pandas as pd

    from app.backend.slag_aware_deox import load_addition_methods

    if source == "synthetic":
        from app.backend.data_curator import (
            generate_synthetic_deox_calibration_dataset,
        )
        df = generate_synthetic_deox_calibration_dataset(n_heats=n_heats)
    else:  # heats_db
        from app.backend.heat_records import list_heats
        heats = list_heats(has_outcome=True, limit=100000)
        rows = []
        methods = load_addition_methods()
        for h in heats:
            if h.eta_al_effective is None or h.method_id is None:
                continue
            if h.method_id not in methods:
                continue
            d = {
                "eta_al_effective": h.eta_al_effective,
                "method_eta_baseline": methods[h.method_id].eta_al_typical,
                "slag_feo_pct": h.slag_feo_pct,
                "slag_mass_kg": h.slag_mass_kg,
                "t_al_addition_c": h.t_al_addition_c,
                "dt_to_al_min": h.dt_to_al_min,
                "co_deox_fesi_kg": h.co_deox_fesi_kg,
                "refractory_heat_count": h.refractory_heat_count,
            }
            rows.append(d)
        df = pd.DataFrame(rows).dropna()
        if len(df) < 100:
            raise SystemExit(
                f"Недостаточно плавок с outcome (N={len(df)} < 100). "
                f"Используйте --source synthetic."
            )

    # B2 target: correction = eta / baseline
    df = df[df["method_eta_baseline"] > 0].copy()
    y = df["eta_al_effective"] / df["method_eta_baseline"]
    y.name = "correction"
    # Features: numeric, excl target + baseline + offset + categorical + meta
    exclude = {
        "eta_al_effective", "method_eta_baseline", "plant_offset_baseline",
        "heat_id", "heat_date", "campaign_id", "plant_id", "method_id",
        "addition_timing", "carrier_gas", "vacuum_treatment",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[feature_cols].reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y, feature_cols


def _find_knee(frontier):
    """Pick knee = max R² per unit sqrt(complexity) (simple heuristic)."""
    if not frontier:
        return None
    return max(frontier, key=lambda f: f.r2 / max(f.complexity, 1) ** 0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description="Symbolic regression η_Al correction")
    ap.add_argument("--source", choices=["synthetic", "heats_db"], default="synthetic")
    ap.add_argument("--n-heats", type=int, default=500)
    ap.add_argument("--population", type=int, default=2000)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--parsimony", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--critic", dest="critic", action="store_true", default=True)
    ap.add_argument("--no-critic", dest="critic", action="store_false")
    ap.add_argument("--save-report", action="store_true")
    args = ap.parse_args()

    X, y, feature_names = _load_data(args.source, args.n_heats)
    print(f"Loaded {len(X)} rows, {len(feature_names)} features: {feature_names}")
    print(f"Target correction: mean={y.mean():.3f}, std={y.std():.3f}")

    from app.backend.symbolic_regressor import run_symbolic_regression
    run = run_symbolic_regression(
        X, y, feature_names=feature_names,
        population_size=args.population, generations=args.generations,
        parsimony_coefficient=args.parsimony, random_state=args.seed,
    )

    print(f"\nPareto frontier ({len(run.pareto_frontier)} formulas):")
    for f in run.pareto_frontier:
        print(f"  complexity={f.complexity:3d} | R²={f.r2:+.3f} | RMSE={f.rmse:.4f}")
        print(f"    {f.formula_infix}")

    knee = _find_knee(run.pareto_frontier)
    best = run.best_overall

    # LLM критик для knee + best
    if args.critic:
        from app.backend.symbolic_eta_critic import make_symbolic_eta_critic
        critic = make_symbolic_eta_critic()
        if critic is None:
            print("\n[critic skipped — no ANTHROPIC_API_KEY or prompt]")
        else:
            to_review = []
            if knee:
                to_review.append(("knee", knee))
            if best and best is not knee:
                to_review.append(("best", best))
            for label, formula in to_review:
                print(f"\n=== Critic review [{label}] ===")
                verdict = critic.review_formula(
                    formula_infix=formula.formula_infix,
                    r2=formula.r2, complexity=formula.complexity,
                    feature_names=feature_names,
                )
                if verdict:
                    print(f"  Verdict: {verdict.verdict} (conf {verdict.confidence})")
                    print(f"  {verdict.summary}")
                    for chk in verdict.physical_checks:
                        print(f"    {chk.term}: expected {chk.expected_sign}, "
                              f"found {chk.found_sign} → {chk.verdict}")

    # Decision Log
    try:
        from decision_log.logger import log_decision
        log_decision(
            phase="deoxidation",
            decision=(
                f"Symbolic η_Al correction: best R²={best.r2:.3f} "
                f"at complexity={best.complexity}"
                if best else "No formula found"
            ),
            reasoning=(
                f"source={args.source}, N={len(X)}, "
                f"gplearn pop={args.population} gen={args.generations}"
            ),
            context={
                "feature_names": feature_names,
                "frontier": [
                    {"complexity": f.complexity, "r2": f.r2,
                     "formula": f.formula_infix}
                    for f in run.pareto_frontier
                ],
            },
            author="symbolic_eta_correction",
            tags=["symbolic_regression", "eta_correction", "gplearn"],
        )
    except Exception as e:
        print(f"[Decision Log save failed: {e}]")

    if args.save_report:
        import json
        from datetime import datetime
        rpt_dir = Path("reports")
        rpt_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rpt_path = rpt_dir / f"eta_correction_{ts}.json"
        rpt_path.write_text(json.dumps({
            "source": args.source, "n_rows": len(X),
            "feature_names": feature_names,
            "frontier": [
                {"complexity": f.complexity, "r2": f.r2, "rmse": f.rmse,
                 "formula": f.formula_infix}
                for f in run.pareto_frontier
            ],
        }, indent=2, ensure_ascii=False))
        print(f"\nReport → {rpt_path}")

    return 0 if run.pareto_frontier else 1


if __name__ == "__main__":
    sys.exit(main())
