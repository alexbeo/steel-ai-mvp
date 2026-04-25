"""
B1 verification runner.

For a given trained model:
  1. Load class dataset (real Agrawal NIMS / synthetic HSLA / synthetic Q&T)
  2. Run symbolic regression on (model.feature_list, model.target)
  3. Report Pareto frontier of (complexity, R²) and the best closed-form
  4. Persist summary to docs/b1_formulas_<version>.json + Decision Log

Symbolic regression value: not predictive uplift (A1 already covers
that) but interpretable structure. Even on a saturated baseline a
short formula with R²=0.7-0.85 has scientific value if it matches a
known empirical law (or surprises us by not matching).

Run:
    PYTHONPATH=. .venv/bin/python scripts/symbolic_regression_for_model.py
    PYTHONPATH=. .venv/bin/python scripts/symbolic_regression_for_model.py \\
        --model fatigue_fatigue_strength_xgb_20260424_233914 \\
        --population 2000 --generations 15
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"


def _latest_model_for_class(class_id: str) -> str | None:
    candidates = []
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        if meta.get("steel_class") == class_id:
            candidates.append(d.name)
    return candidates[-1] if candidates else None


def _load_class_dataset(steel_class: str) -> pd.DataFrame | None:
    try:
        from app.backend.steel_classes import (
            compute_features_for_class,
            get_synthetic_generator,
            load_steel_class,
        )
        profile = load_steel_class(steel_class)
        gen = get_synthetic_generator(profile.synthetic_generator_name)
        return compute_features_for_class(gen(), steel_class)
    except Exception as e:
        logger.warning("can't load dataset for %s: %s", steel_class, e)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--class-id", default="fatigue_carbon_steel")
    parser.add_argument("--population", type=int, default=1500)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--parsimony", type=float, default=0.01,
                        help="penalty on tree length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-features", type=int, default=10,
                        help="restrict SR to top-N features by importance")
    args = parser.parse_args()

    model_version = args.model or _latest_model_for_class(args.class_id)
    if not model_version:
        logger.error("no model for class %s", args.class_id)
        sys.exit(1)
    logger.info("model: %s", model_version)

    from app.backend.model_trainer import load_model
    bundle = load_model(model_version)
    meta = bundle["meta"]
    target = meta["target"]
    steel_class = meta.get("steel_class", "unknown")

    df = _load_class_dataset(steel_class)
    if df is None or target not in df.columns:
        logger.error("can't load dataset / target missing")
        sys.exit(1)

    importance = meta["feature_importance"]
    top_features = [
        f for f, _ in sorted(importance.items(), key=lambda kv: -kv[1])
        [: args.top_features]
        if f in df.columns
    ]
    logger.info(
        "running symbolic regression on %d top features: %s",
        len(top_features), top_features,
    )

    X = df[top_features]
    y = df[target]
    logger.info("dataset: n=%d, target=%s", len(X), target)
    logger.info(
        "GP config: pop=%d, gen=%d, parsimony=%s",
        args.population, args.generations, args.parsimony,
    )

    from app.backend.symbolic_regressor import run_symbolic_regression
    result = run_symbolic_regression(
        X, y,
        feature_names=top_features,
        population_size=args.population,
        generations=args.generations,
        parsimony_coefficient=args.parsimony,
        random_state=args.seed,
    )

    DOCS_DIR.mkdir(exist_ok=True)
    out_path = DOCS_DIR / f"b1_formulas_{model_version}.json"
    out_path.write_text(json.dumps({
        "model_version": model_version,
        "steel_class": steel_class,
        "target": target,
        "feature_names": top_features,
        "n_train": result.n_train,
        "best_overall": asdict(result.best_overall) if result.best_overall else None,
        "pareto_frontier": [asdict(f) for f in result.pareto_frontier],
    }, indent=2, ensure_ascii=False))
    logger.info("formulas → %s", out_path.relative_to(PROJECT_ROOT))

    from decision_log.logger import log_decision
    log_decision(
        phase="training",
        decision=(
            f"SymbolicRegression: {len(result.pareto_frontier)} формул "
            f"на Pareto, лучшая R²={result.best_overall.r2:.3f} "
            f"(complexity {result.best_overall.complexity})"
            if result.best_overall else "SymbolicRegression: 0 формул"
        ),
        reasoning=(
            f"model={model_version}, n_features={len(top_features)}, "
            f"pop={args.population}, gen={args.generations}"
        ),
        context={
            "model_version": model_version,
            "feature_names": top_features,
            "best_overall": asdict(result.best_overall) if result.best_overall else None,
            "frontier_size": len(result.pareto_frontier),
        },
        author="symbolic_regressor",
        tags=["symbolic_regression", "gplearn"],
    )

    print()
    print("=" * 70)
    print(f"Symbolic regression — Pareto frontier для {model_version}")
    print(f"target = {target}, признаки = {len(top_features)}, n = {result.n_train}")
    print("=" * 70)
    print(f"\n{'Сложность':>10} {'R²':>7} {'RMSE':>9}  Формула")
    print("-" * 70)
    for f in result.pareto_frontier:
        line = f"{f.complexity:>10}  {f.r2:>+5.3f}  {f.rmse:>8.2f}  {f.formula_infix}"
        if len(line) > 200:
            line = line[:197] + "..."
        print(line)

    if result.best_overall:
        b = result.best_overall
        print()
        print("=" * 70)
        print(f"Лучшая по R² (complexity {b.complexity}):")
        print("=" * 70)
        print(f"R²    = {b.r2:.4f}")
        print(f"RMSE  = {b.rmse:.2f}")
        print(f"MAE   = {b.mae:.2f}")
        print(f"\nФормула:\n  {target} ≈ {b.formula_infix}")


if __name__ == "__main__":
    main()
