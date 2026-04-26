"""
A2 verification runner.

Takes a trained-model version (default: latest fatigue_carbon_steel),
loads its meta + dataset, builds hypothesis-generation context, calls
HypothesisGenerator (when ANTHROPIC_API_KEY is set), prints hypotheses
and persists them to the Decision Log.

Without API key the script prints the prompt context and exits — useful
for dry-run inspection of what would be sent to Claude.

Run:
    PYTHONPATH=. ANTHROPIC_API_KEY=sk-ant-... \\
        .venv/bin/python scripts/generate_hypotheses_for_model.py
    PYTHONPATH=. ANTHROPIC_API_KEY=sk-ant-... \\
        .venv/bin/python scripts/generate_hypotheses_for_model.py \\
        --model hsla_yield_strength_xgb_20260424_232852
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def latest_model_for_class(class_id: str) -> str | None:
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
    """Get the dataset for a steel class via the registry, with feature
    engineering applied. Returns None if unknown class or load fails.

    Routes through `get_synthetic_generator` + `compute_features_for_class`
    — the same path the train tab uses, so a model trained from the UI
    sees the same columns at inference time. Avoids hardcoded parquet
    paths that drift when new classes are added.
    """
    try:
        from app.backend.steel_classes import (
            compute_features_for_class,
            get_synthetic_generator,
            load_steel_class,
        )
        profile = load_steel_class(steel_class)
        gen = get_synthetic_generator(profile.synthetic_generator_name)
        df_raw = gen()
        return compute_features_for_class(df_raw, steel_class)
    except Exception as e:
        logger.warning("can't load dataset for class %s: %s", steel_class, e)
        return None


def build_context(model_version: str) -> dict:
    from app.backend.model_trainer import load_model

    bundle = load_model(model_version)
    meta = bundle["meta"]
    metrics = meta["metrics"]

    sample_predictions = []
    target = meta["target"]
    steel_class = meta.get("steel_class", "unknown")
    target_distribution = {}

    df = _load_class_dataset(steel_class)
    if df is not None and target in df.columns:
        target_distribution = {
            "min": float(df[target].min()),
            "max": float(df[target].max()),
            "mean": float(df[target].mean()),
            "std": float(df[target].std()),
            "n": int(len(df)),
        }
        missing = [f for f in meta["feature_list"] if f not in df.columns]
        if missing:
            logger.warning(
                "model expects features absent from dataset (%d missing); "
                "skipping sample_predictions: %s", len(missing), missing[:5]
            )
        else:
            feat = meta["feature_list"]
            sample = df[feat + [target]].sample(
                n=min(5, len(df)), random_state=42,
            )
            X = sample[feat]
            q_correction = meta.get("conformal_correction_mpa", 0.0)
            preds = bundle["main"].predict(X)
            lo = bundle["q05"].predict(X) - q_correction
            hi = bundle["q95"].predict(X) + q_correction
            sample_predictions = [
                {
                    "actual": float(sample[target].iloc[i]),
                    "pred": float(preds[i]),
                    "lower_90": float(lo[i]),
                    "upper_90": float(hi[i]),
                }
                for i in range(len(sample))
            ]

    return {
        "model_version": model_version,
        "steel_class": steel_class,
        "target": target,
        "data_source": meta.get("data_source"),
        "data_source_doi": meta.get("data_source_doi"),
        "r2_train": metrics["r2_train"],
        "r2_val": metrics["r2_val"],
        "r2_test": metrics["r2_test"],
        "mae_test": metrics["mae_test"],
        "rmse_test": metrics["rmse_test"],
        "coverage_90_ci": metrics["coverage_90_ci"],
        "conformal_correction_mpa": meta.get("conformal_correction_mpa", 0.0),
        "n_train": metrics["n_train"],
        "n_val": metrics["n_val"],
        "n_test": metrics["n_test"],
        "feature_importance": meta["feature_importance"],
        "training_ranges": meta["training_ranges"],
        "target_distribution": target_distribution,
        "sample_predictions": sample_predictions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model version (default: latest fatigue_carbon_steel)")
    parser.add_argument("--class-id", default="fatigue_carbon_steel",
                        help="steel class id when --model not given")
    parser.add_argument("--dry-run", action="store_true",
                        help="print context payload without calling LLM")
    args = parser.parse_args()

    model_version = args.model or latest_model_for_class(args.class_id)
    if not model_version:
        logger.error("No model found for class %s", args.class_id)
        sys.exit(1)
    logger.info("Using model: %s", model_version)

    ctx = build_context(model_version)

    if args.dry_run or not os.environ.get("ANTHROPIC_API_KEY"):
        from app.backend.hypothesis_generator import _build_user_payload
        print("=" * 70)
        print("DRY RUN — payload that would be sent to Claude:")
        print("=" * 70)
        print(_build_user_payload(ctx))
        if not args.dry_run:
            print()
            print("Set ANTHROPIC_API_KEY in env to run live.")
        return

    from app.backend.hypothesis_generator import make_hypothesis_generator
    gen = make_hypothesis_generator()
    if gen is None:
        logger.error("HypothesisGenerator unavailable (no key / SDK missing).")
        sys.exit(1)

    logger.info("Calling Claude generator (model: %s)...", gen.model)
    hypotheses = gen.generate(ctx)
    logger.info("Got %d hypotheses", len(hypotheses))

    reviews_by_id: dict[str, dict] = {}
    if hypotheses:
        from app.backend.hypothesis_critic import make_hypothesis_critic
        crit = make_hypothesis_critic()
        if crit is None:
            logger.warning("HypothesisCritic unavailable — skipping peer review")
        else:
            from dataclasses import asdict as _asdict
            logger.info("Calling Claude critic for adversarial review...")
            verdicts = crit.review(
                ctx,
                [_asdict(h) for h in hypotheses],
            )
            logger.info("Got %d reviews", len(verdicts))
            for v in verdicts:
                reviews_by_id[v.hypothesis_id] = _asdict(v)

    print()
    print("=" * 70)
    print(f"Гипотезы для {model_version}")
    print("=" * 70)
    for i, h in enumerate(hypotheses, 1):
        rv = reviews_by_id.get(h.id)
        print(
            f"\n[{i}/{len(hypotheses)}] "
            f"[новизна={h.novelty}, стоимость={h.experiment_cost_estimate}] "
            f"{h.statement}"
        )
        print(f"  Обоснование: {h.rationale}")
        print(f"  Эксперимент: fix {h.proposed_experiment.get('fix')}")
        print(f"               sweep {h.proposed_experiment.get('sweep')}")
        print(f"  Ожидание: {h.expected_outcome}")
        print(f"  Сравнение с классикой: {h.economic_impact.vs_classical_baseline}")
        print(f"  Оценка экономии: {h.economic_impact.estimated_saving}")
        print(f"  Метод проверки: {h.economic_impact.measurement_method}")
        print(f"  id={h.id}, теги={h.tags}")
        if rv:
            print(
                f"  ── Рецензия: {rv['verdict']} "
                f"(уверенность {rv['confidence']})"
            )
            print(f"     {rv['summary']}")
            for s in rv.get("strengths", []):
                print(f"     + {s}")
            for w in rv.get("weaknesses", []):
                print(f"     − {w}")
            if rv.get("suggested_revision"):
                print(f"     ⤳ Правка: {rv['suggested_revision']}")


if __name__ == "__main__":
    main()
