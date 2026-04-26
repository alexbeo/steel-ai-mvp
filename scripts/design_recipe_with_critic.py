"""
RD3 — Полный цикл подбора рецепта: Sonnet designer + ML/cost truth gate
+ Sonnet PhD critic с evidence fact-check.

Цепочка для одной design-сессии:
  1. build_context из обученной модели + dataset baseline
  2. RecipeDesigner.design → 3-4 рецепта с PhD reasoning + evidence
  3. ML+cost truth gate: для каждого рецепта вычисляем predicted
     property (с conformal CI) + ferroalloy cost через cost_model
  4. RecipeCritic.review → ACCEPT/REVISE/REJECT + evidence_check + ...
  5. Persist combined cycle to Decision Log under tag "recipe_cycle"
  6. Print human-readable Russian report

Run:
    PYTHONPATH=. ANTHROPIC_API_KEY=... \\
        .venv/bin/python scripts/design_recipe_with_critic.py
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

DESIGN_COMPOSITION = ["si_pct", "mn_pct", "ni_pct", "cr_pct", "cu_pct", "mo_pct"]
DESIGN_PROCESS = [
    "normalizing_temp_c",
    "carburizing_temp_c",
    "carburizing_time_min",
    "tempering_temp_c",
    "tempering_time_min",
    "through_hardening_cooling_rate_c_per_s",
]


def _latest_model(class_id: str) -> str | None:
    for d in sorted(MODELS_DIR.iterdir(), reverse=True):
        if not d.is_dir(): continue
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("steel_class") == class_id:
            return d.name
    return None


def _load_baseline(df: pd.DataFrame, sub_class: str = "carbon_low_alloy") -> pd.Series:
    if "sub_class" in df.columns:
        subset = df[df["sub_class"] == sub_class]
        if len(subset) < 50:
            subset = df
            sub_class = "all"
    else:
        subset = df; sub_class = "all"
    logger.info("baseline: sub_class=%s, n=%d", sub_class, len(subset))
    return subset.select_dtypes(include=[np.number]).median()


def _composition_dict(row: pd.Series) -> dict[str, float]:
    return {k: float(v) for k, v in row.items() if k.endswith("_pct")}


def _verify_recipe(
    recipe_dict: dict, baseline: pd.Series, feature_list: list[str],
    bundle: dict, snapshot,
) -> dict:
    """Apply recipe to baseline vector, run XGBoost predict + cost compute."""
    from app.backend.cost_model import compute_cost
    from app.backend.model_trainer import predict_with_uncertainty

    row = baseline.copy()
    for k, v in recipe_dict.get("composition", {}).items():
        if k in row.index:
            row[k] = float(v)
    for k, v in recipe_dict.get("process_params", {}).items():
        if k in row.index:
            row[k] = float(v)

    X = pd.DataFrame(
        [[float(row[f]) for f in feature_list]], columns=feature_list,
    )
    pred_df = predict_with_uncertainty(bundle, X)
    p = pred_df.iloc[0]

    comp_dict = _composition_dict(row)
    try:
        cb = compute_cost(comp_dict, snapshot, mode="full")
        cost = cb.total_per_ton
        ferroalloy = [
            {
                "material": c.material_id,
                "kg_per_ton": round(c.mass_kg_per_ton_steel, 2),
                "eur_per_ton": round(c.contribution_per_ton, 2),
            }
            for c in cb.contributions if c.material_id != "scrap"
        ]
    except Exception as e:
        logger.warning("cost compute failed for recipe %s: %s",
                       recipe_dict.get("name"), e)
        cost = float("nan"); ferroalloy = []

    return {
        "predicted_property": float(p["prediction"]),
        "lower_90": float(p["lower_90"]),
        "upper_90": float(p["upper_90"]),
        "ood_flag": bool(p["ood_flag"]),
        "cost_per_ton": float(cost) if cost == cost else None,
        "ferroalloy_breakdown": ferroalloy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--class-id", default="fatigue_carbon_steel")
    parser.add_argument("--task", default=(
        "Снизить ferroalloy cost vs baseline при сохранении или улучшении "
        "fatigue strength. Целевой компромисс: каждый −€10/т имеет ценность "
        "если |Δσf| ≤ 30 МПа."
    ))
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY не задан")
        sys.exit(1)

    model_version = args.model or _latest_model(args.class_id)
    if not model_version:
        logger.error("нет модели для %s", args.class_id)
        sys.exit(1)
    logger.info("model: %s", model_version)

    from app.backend.cost_model import compute_cost, seed_snapshot
    from app.backend.data_curator import load_real_agrawal_fatigue_dataset
    from app.backend.model_trainer import load_model
    from app.backend.recipe_designer import make_recipe_designer
    from app.backend.recipe_critic import make_recipe_critic

    bundle = load_model(model_version)
    meta = bundle["meta"]
    target = meta["target"]
    training_ranges = meta["training_ranges"]

    df = load_real_agrawal_fatigue_dataset()
    baseline = _load_baseline(df)
    snapshot = seed_snapshot()
    baseline_comp = _composition_dict(baseline)
    baseline_cost = compute_cost(baseline_comp, snapshot, mode="full").total_per_ton

    feature_list = meta["feature_list"]
    X_base = pd.DataFrame(
        [[float(baseline[f]) for f in feature_list]], columns=feature_list,
    )
    from app.backend.model_trainer import predict_with_uncertainty
    base_pred = predict_with_uncertainty(bundle, X_base).iloc[0]
    baseline_predicted = float(base_pred["prediction"])

    available_composition = [c for c in DESIGN_COMPOSITION if c in feature_list]
    available_process = [p for p in DESIGN_PROCESS if p in feature_list]
    baseline_recipe = {
        **{k: float(baseline[k]) for k in available_composition},
        **{k: float(baseline[k]) for k in available_process},
    }

    logger.info(
        "baseline: σf=%.0f МПа, cost=%.2f €/т",
        baseline_predicted, baseline_cost,
    )

    designer_ctx = {
        "task": args.task,
        "steel_class": meta.get("steel_class"),
        "target": target,
        "data_source": meta.get("data_source"),
        "model_version": model_version,
        "r2_test": meta["metrics"]["r2_test"],
        "mae_test": meta["metrics"]["mae_test"],
        "coverage_90_ci": meta["metrics"]["coverage_90_ci"],
        "conformal_correction_mpa": meta.get("conformal_correction_mpa", 0),
        "feature_importance": meta["feature_importance"],
        "training_ranges": training_ranges,
        "target_distribution": {
            "min": float(df[target].min()),
            "max": float(df[target].max()),
            "mean": float(df[target].mean()),
            "std": float(df[target].std()),
            "n": int(len(df)),
        },
        "baseline_recipe": baseline_recipe,
        "baseline_predicted_property": baseline_predicted,
        "baseline_cost_per_ton": float(baseline_cost),
        "available_composition": available_composition,
        "available_process": available_process,
    }

    designer = make_recipe_designer()
    if designer is None:
        logger.error("designer недоступен"); sys.exit(1)
    logger.info("вызываю RecipeDesigner...")
    recipes = designer.design(designer_ctx)
    logger.info("получено %d рецептов", len(recipes))
    if not recipes:
        sys.exit(2)

    logger.info("ML+cost верификация %d рецептов...", len(recipes))
    recipes_with_verification = []
    for r in recipes:
        ml = _verify_recipe(asdict(r), baseline, feature_list, bundle, snapshot)
        ml["delta_property"] = ml["predicted_property"] - baseline_predicted
        ml["delta_cost"] = (
            ml["cost_per_ton"] - baseline_cost
            if ml["cost_per_ton"] is not None else None
        )
        d = asdict(r); d["ml_verification"] = ml
        recipes_with_verification.append(d)

    critic = make_recipe_critic()
    if critic is None:
        logger.error("critic недоступен"); sys.exit(1)
    logger.info("вызываю RecipeCritic...")
    verdicts = critic.review(designer_ctx, recipes_with_verification)
    logger.info("получено %d рецензий", len(verdicts))

    verdicts_by_id = {v.recipe_id: asdict(v) for v in verdicts}
    counts = {"ACCEPT": 0, "REVISE": 0, "REJECT": 0}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1

    from decision_log.logger import log_decision
    log_decision(
        phase="inverse_design",
        decision=(
            f"Recipe cycle: {len(recipes)} рецептов, "
            f"A={counts['ACCEPT']} R={counts['REVISE']} X={counts['REJECT']}"
        ),
        reasoning=f"model={model_version}, baseline σf={baseline_predicted:.0f} МПа, cost={baseline_cost:.2f} €/т",
        context={
            "model_version": model_version,
            "baseline": {
                "recipe": baseline_recipe,
                "predicted_property": baseline_predicted,
                "cost_per_ton": float(baseline_cost),
            },
            "recipes": recipes_with_verification,
            "reviews": [asdict(v) for v in verdicts],
            "verdict_counts": counts,
        },
        author="ui",
        tags=["recipe_cycle", "sonnet-4-6"],
    )

    print()
    print("=" * 78)
    print(f"Recipe design cycle на {model_version}")
    print("=" * 78)
    print(f"Baseline: σf={baseline_predicted:.0f} МПа, cost={baseline_cost:.2f} €/т")
    print(f"Verdicts: ACCEPT={counts['ACCEPT']}, REVISE={counts['REVISE']}, "
          f"REJECT={counts['REJECT']}")

    for i, r_dict in enumerate(recipes_with_verification, 1):
        ml = r_dict["ml_verification"]
        rv = verdicts_by_id.get(r_dict["id"])
        verdict_str = (
            f"{rv['verdict']} (увер. {rv['confidence']})" if rv else "—"
        )
        delta_p = ml["delta_property"]
        delta_c = ml["delta_cost"]
        print()
        print(f"[{i}/{len(recipes)}] [{r_dict['novelty']}] {r_dict['name']}")
        print(f"    Δσf = {delta_p:+.0f} МПа, Δcost = "
              f"{delta_c:+.2f} €/т"
              if delta_c is not None else
              f"    Δσf = {delta_p:+.0f} МПа, cost ошибка")
        print(f"    Прогноз: {ml['predicted_property']:.0f} МПа "
              f"[{ml['lower_90']:.0f},{ml['upper_90']:.0f}], "
              f"OOD={'ДА' if ml['ood_flag'] else 'нет'}")
        print(f"    Ожидание автора: {r_dict['expected_outcome']}")
        print(f"    Вердикт PhD-критика: {verdict_str}")
        if rv:
            print(f"      Резюме: {rv['summary']}")
            for ec in rv.get("evidence_check", []):
                mark = {"VALID": "✓", "INVALID": "✗", "UNVERIFIABLE": "?"}[ec["verdict"]]
                print(f"      {mark} {ec['claim']} — {ec['note']}")
            for s in rv.get("strengths", []):
                print(f"      + {s}")
            for w in rv.get("weaknesses", []):
                print(f"      − {w}")
            if rv.get("suggested_revision"):
                print(f"      ⤳ Правка: {rv['suggested_revision']}")


if __name__ == "__main__":
    main()
