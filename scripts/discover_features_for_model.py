"""
A1 verification runner: discover new features and measure R² uplift.

For a given trained model:
  1. Load class dataset (real Agrawal / synthetic HSLA / synthetic Q&T)
  2. Build context with current feature_set + feature_importance + ranges
  3. Ask Claude FeatureDiscoverer for 3-5 new feature proposals
  4. For each proposal:
       a. Apply formula to dataset via pandas.DataFrame.eval
       b. Train baseline XGBoost (current feature_set, fixed seed)
       c. Train extended XGBoost (current + new feature, same seed)
       d. Compare R² and MAE on test fold
  5. Persist results to docs/a1_uplift_<version>.json + Decision Log

Without ANTHROPIC_API_KEY: prints prompt context (dry-run) and exits.

Run:
    PYTHONPATH=. ANTHROPIC_API_KEY=... \
        .venv/bin/python scripts/discover_features_for_model.py [--model VERSION]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
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


def build_context(model_version: str) -> tuple[dict, pd.DataFrame | None]:
    from app.backend.model_trainer import load_model

    bundle = load_model(model_version)
    meta = bundle["meta"]
    metrics = meta["metrics"]
    steel_class = meta.get("steel_class", "unknown")
    target = meta["target"]

    df = _load_class_dataset(steel_class)
    available_columns = list(df.columns) if df is not None else []

    ctx = {
        "model_version": model_version,
        "steel_class": steel_class,
        "target": target,
        "available_columns": available_columns,
        "current_feature_set": meta["feature_list"],
        "feature_importance": meta["feature_importance"],
        "training_ranges": meta["training_ranges"],
        "r2_test": metrics["r2_test"],
        "mae_test": metrics["mae_test"],
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
    }
    return ctx, df


def _train_compact(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    seed: int = 42,
) -> dict:
    """Fixed-seed compact XGBoost training for fair before/after comparison.
    No Optuna, no quantile heads — just main + R²/MAE on a deterministic split.
    """
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor

    X = df[features].values
    y = df[target].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed,
    )
    m = XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        random_state=seed, tree_method="hist", n_jobs=-1,
    )
    m.fit(X_tr, y_tr, verbose=False)
    pred = m.predict(X_te)
    return {
        "r2": float(r2_score(y_te, pred)),
        "mae": float(mean_absolute_error(y_te, pred)),
        "n_features": len(features),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--class-id", default="fatigue_carbon_steel")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_version = args.model or _latest_model_for_class(args.class_id)
    if not model_version:
        logger.error("no model for class %s", args.class_id)
        sys.exit(1)
    logger.info("model: %s", model_version)

    ctx, df = build_context(model_version)
    if df is None or ctx["target"] not in df.columns:
        logger.error("can't load dataset / target missing")
        sys.exit(1)

    if args.dry_run or not os.environ.get("ANTHROPIC_API_KEY"):
        from app.backend.feature_discoverer import _build_user_payload
        print(_build_user_payload(ctx))
        if not args.dry_run:
            print("\nset ANTHROPIC_API_KEY to run live")
        return

    from app.backend.feature_discoverer import (
        apply_formula,
        FormulaError,
        make_feature_discoverer,
    )
    from decision_log.logger import log_decision

    fd = make_feature_discoverer()
    if fd is None:
        logger.error("FeatureDiscoverer unavailable")
        sys.exit(1)

    logger.info("calling Claude (model=%s)…", fd.model)
    proposals = fd.discover(ctx)
    logger.info("got %d proposals", len(proposals))
    if not proposals:
        sys.exit(2)

    target = ctx["target"]
    base_features = ctx["current_feature_set"]
    logger.info("training baseline (n_features=%d)…", len(base_features))
    baseline = _train_compact(df, target, base_features)
    logger.info(
        "  baseline: R²=%.4f, MAE=%.2f", baseline["r2"], baseline["mae"]
    )

    results = []
    for p in proposals:
        try:
            new_col = apply_formula(df, p.formula, p.name)
        except FormulaError as e:
            logger.warning("✗ %s — formula failed: %s", p.name, e)
            results.append({
                "proposal": asdict(p),
                "applied": False,
                "error": str(e),
                "delta_r2": None,
                "delta_mae": None,
            })
            continue

        df_ext = df.assign(**{p.name: new_col})
        ext_features = base_features + [p.name]
        ext = _train_compact(df_ext, target, ext_features)

        delta_r2 = ext["r2"] - baseline["r2"]
        delta_mae = ext["mae"] - baseline["mae"]
        sign = "+" if delta_r2 >= 0 else "−"
        logger.info(
            "  [%s] %s  ΔR²=%s%.4f  ΔMAE=%+.2f  (%s)",
            p.mechanism_class, p.name, sign, abs(delta_r2),
            delta_mae, p.formula,
        )
        results.append({
            "proposal": asdict(p),
            "applied": True,
            "ext_r2": ext["r2"],
            "ext_mae": ext["mae"],
            "delta_r2": delta_r2,
            "delta_mae": delta_mae,
        })

    summary = {
        "model_version": model_version,
        "steel_class": ctx["steel_class"],
        "target": target,
        "baseline": baseline,
        "results": results,
    }

    DOCS_DIR.mkdir(exist_ok=True)
    out_path = DOCS_DIR / f"a1_uplift_{model_version}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("metrics → %s", out_path.relative_to(PROJECT_ROOT))

    helpful = sum(1 for r in results if r.get("applied") and r["delta_r2"] > 0)
    log_decision(
        phase="training",
        decision=(
            f"FeatureDiscoverer cycle: {len(proposals)} proposals, "
            f"{helpful} с положительным ΔR²"
        ),
        reasoning=(
            f"baseline R²={baseline['r2']:.4f}, "
            f"helpful={helpful}/{len(results)}"
        ),
        context={
            "model_version": model_version,
            "summary": summary,
        },
        author="ui",
        tags=["feature_discovery_cycle", "sonnet-4-6"],
    )

    print()
    print("=" * 70)
    print(f"Сводка для {model_version}")
    print("=" * 70)
    print(f"baseline: R²={baseline['r2']:.4f}, MAE={baseline['mae']:.2f}")
    for r in results:
        p = r["proposal"]
        if not r.get("applied"):
            print(f"\n✗ {p['name']:30s} ({p['mechanism_class']:13s}) — {r.get('error', '?')}")
            continue
        sign = "+" if r["delta_r2"] >= 0 else "−"
        verdict = (
            "ПОЛЕЗНА" if r["delta_r2"] > 0 else
            "БЕСПОЛЕЗНА" if abs(r["delta_r2"]) < 1e-4 else
            "ВРЕДНА"
        )
        print(
            f"\n[{verdict}] {p['name']:30s} ({p['mechanism_class']:13s}) "
            f"ΔR²={sign}{abs(r['delta_r2']):.4f} ΔMAE={r['delta_mae']:+.2f}"
        )
        print(f"  formula: {p['formula']}")
        print(f"  expected: {p['expected_uplift']}")


if __name__ == "__main__":
    main()
