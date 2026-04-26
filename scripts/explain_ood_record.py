"""
A3 verification runner — Sonnet объясняет OOD-кейс на Agrawal модели.

Берёт hand-crafted рецепт с одной-двумя композиционными аномалиями
(например, Mn выше верхней границы training, или C ниже нижней),
запускает GMM OOD-detector + ML predict, передаёт всё Sonnet'у —
получает structured diagnosis.

Run:
    PYTHONPATH=. ANTHROPIC_API_KEY=... \\
        .venv/bin/python scripts/explain_ood_record.py
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def _latest_model(class_id: str) -> str | None:
    for d in sorted(MODELS_DIR.iterdir(), reverse=True):
        if not d.is_dir(): continue
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("steel_class") == class_id:
            return d.name
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--class-id", default="fatigue_carbon_steel")
    args = parser.parse_args()

    from app.backend.anomaly_explainer import make_anomaly_explainer
    from app.backend.data_curator import load_real_agrawal_fatigue_dataset
    from app.backend.model_trainer import load_model, predict_with_uncertainty

    model_version = args.model or _latest_model(args.class_id)
    bundle = load_model(model_version)
    meta = bundle["meta"]
    feature_list = meta["feature_list"]
    training_ranges = meta["training_ranges"]

    df = load_real_agrawal_fatigue_dataset()
    if "sub_class" in df.columns:
        sub = df[df["sub_class"] == "carbon_low_alloy"]
    else:
        sub = df
    baseline = sub.select_dtypes(include=[np.number]).median()

    # Construct a deliberately OOD recipe: push Mn above training max,
    # push C below training min. Both happen often in spring-steel
    # families that look weird from Agrawal's carbon_low_alloy lens.
    ood_recipe = baseline.copy()
    mn_hi = training_ranges["mn_pct"][1]
    ood_recipe["mn_pct"] = mn_hi * 1.15  # 15% выше верхней границы
    c_lo = training_ranges["c_pct"][0]
    ood_recipe["c_pct"] = c_lo * 0.7    # 30% ниже нижней границы

    X_ood = pd.DataFrame(
        [[float(ood_recipe[f]) for f in feature_list]], columns=feature_list,
    )
    pp = predict_with_uncertainty(bundle, X_ood).iloc[0]

    out_of_range = []
    for f in feature_list:
        if f not in training_ranges:
            continue
        lo, hi = training_ranges[f]
        v = float(ood_recipe[f])
        if v < lo or v > hi:
            out_of_range.append({
                "feature": f,
                "value": v,
                "training_range": [lo, hi],
                "deviation_pct": (
                    (v - hi) / hi * 100 if v > hi else
                    (lo - v) / max(abs(lo), 1e-9) * 100
                ),
            })

    logger.info(
        "OOD recipe: Mn=%.2f (training %s), C=%.3f (training %s)",
        ood_recipe["mn_pct"], training_ranges["mn_pct"],
        ood_recipe["c_pct"], training_ranges["c_pct"],
    )
    logger.info(
        "ML predict: σf=%.0f МПа [%.0f, %.0f], OOD flag=%s, log_density=%.2f",
        pp["prediction"], pp["lower_90"], pp["upper_90"],
        pp["ood_flag"], pp["log_density"],
    )

    explainer = make_anomaly_explainer()
    if explainer is None:
        logger.error("AnomalyExplainer недоступен"); sys.exit(1)

    medians = {
        c: float(baseline[c]) for c in feature_list
        if c in training_ranges
    }
    ctx = {
        "model_version": model_version,
        "steel_class": meta.get("steel_class"),
        "target": meta["target"],
        "recipe": {f: float(ood_recipe[f]) for f in feature_list if f in training_ranges},
        "training_ranges": training_ranges,
        "training_medians": medians,
        "ml_prediction": {
            "predicted": float(pp["prediction"]),
            "lower_90": float(pp["lower_90"]),
            "upper_90": float(pp["upper_90"]),
            "ci_width": float(pp["upper_90"] - pp["lower_90"]),
        },
        "ood_flag": bool(pp["ood_flag"]),
        "ood_score": float(pp["log_density"]),
        "out_of_range_features": out_of_range,
    }

    logger.info("вызываю Sonnet AnomalyExplainer...")
    exp = explainer.explain(ctx)
    if exp is None:
        logger.error("explainer вернул None"); sys.exit(2)

    print()
    print("=" * 78)
    print(f"Anomaly explanation для OOD-рецепта на {model_version}")
    print("=" * 78)
    print(f"Severity: {exp.severity}")
    print()
    print(f"Резюме: {exp.summary}")
    print()
    print("Аномальные параметры:")
    for af in exp.anomalous_features:
        print(
            f"  • {af.feature} = {af.value:.4f}  (training [{af.training_range[0]:.4f}, "
            f"{af.training_range[1]:.4f}], тип: {af.deviation_kind})"
        )
        print(f"    {af.note}")
    print()
    print("Возможные металлургические проблемы:")
    for c in exp.mechanism_concerns:
        print(f"  • {c}")
    print()
    print(f"Производственные риски: {exp.production_risks}")
    print()
    print(f"Рекомендуемая правка: {exp.suggested_correction}")


if __name__ == "__main__":
    main()
