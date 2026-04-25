"""
B2 runner — top-K next experiments для Agrawal модели по cost-weighted EI.

Output: markdown report + Decision Log запись с тагом `active_learning`.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--class-id", default="fatigue_carbon_steel")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from app.backend.active_learner import propose_next_experiments
    from app.backend.cost_model import compute_cost, seed_snapshot
    from app.backend.data_curator import load_real_agrawal_fatigue_dataset
    from app.backend.model_trainer import load_model, predict_with_uncertainty
    from decision_log.logger import log_decision

    model_version = args.model or _latest_model(args.class_id)
    if not model_version:
        logger.error("нет модели для %s", args.class_id); sys.exit(1)
    logger.info("model: %s", model_version)

    bundle = load_model(model_version)
    meta = bundle["meta"]
    feature_list = meta["feature_list"]
    target = meta["target"]
    training_ranges = meta["training_ranges"]

    df = load_real_agrawal_fatigue_dataset()
    if "sub_class" in df.columns:
        sub = df[df["sub_class"] == "carbon_low_alloy"]
        if len(sub) < 50: sub = df
    else:
        sub = df
    baseline = sub.select_dtypes(include=[np.number]).median()

    f_star = float(df[target].max())
    snapshot = seed_snapshot()

    X_base = pd.DataFrame(
        [[float(baseline[f]) for f in feature_list]], columns=feature_list,
    )
    base_pred = float(predict_with_uncertainty(bundle, X_base).iloc[0]["prediction"])
    baseline_comp = {k: float(v) for k, v in baseline.items() if k.endswith("_pct")}
    baseline_cost = compute_cost(baseline_comp, snapshot, mode="full").total_per_ton

    logger.info(
        "baseline: σ_pred=%.0f МПа, cost=%.2f €/т | f*=%.0f МПа (max in dataset)",
        base_pred, baseline_cost, f_star,
    )

    decision_vars = [
        v for v in DESIGN_COMPOSITION + DESIGN_PROCESS
        if v in feature_list and v in training_ranges
    ]
    bounds = {v: tuple(training_ranges[v]) for v in decision_vars}

    def cost_fn(comp):
        return compute_cost(comp, snapshot, mode="full").total_per_ton

    logger.info(
        "запускаю LHS-скан n=%d по %d decision vars, top_k=%d…",
        args.n_samples, len(decision_vars), args.top_k,
    )
    proposals = propose_next_experiments(
        model_bundle=bundle,
        baseline_row=baseline,
        feature_list=feature_list,
        decision_vars=decision_vars,
        bounds=bounds,
        f_star=f_star,
        cost_fn=cost_fn,
        baseline_cost=baseline_cost,
        baseline_property=base_pred,
        n_samples=args.n_samples,
        top_k=args.top_k,
        seed=args.seed,
    )

    DOCS_DIR.mkdir(exist_ok=True)
    out_lines = [
        f"# Active learning — top-{args.top_k} следующих экспериментов",
        "",
        f"**Модель:** `{model_version}`",
        f"**Acquisition:** Expected Improvement / cost (cost-weighted EI)",
        f"**Baseline:** carbon_low_alloy median",
        f"  - σ_pred = {base_pred:.0f} МПа, cost = {baseline_cost:.2f} €/т",
        f"  - f* (max in training) = {f_star:.0f} МПа",
        "",
        "## Топ кандидатов",
        "",
        f"| # | EI/cost | EI, МПа·кг | σ_pred (90% CI) | cost €/т | Δσ vs base | Δcost vs base | OOD |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, p in enumerate(proposals, 1):
        out_lines.append(
            f"| {i} | {p.acquisition_score:.4f} | {p.expected_improvement:.1f} | "
            f"{p.predicted_property:.0f} [{p.lower_90:.0f},{p.upper_90:.0f}] | "
            f"{p.cost_per_ton:.2f} | {p.delta_vs_baseline_property:+.0f} | "
            f"{p.delta_vs_baseline_cost:+.2f} | "
            f"{'⚠️' if p.ood_flag else 'нет'} |"
        )
    out_lines.extend(["", "## Состав топ-1 candidate", ""])
    if proposals:
        p = proposals[0]
        out_lines.append("**Composition:**")
        for k, v in p.composition.items():
            base_v = float(baseline[k])
            out_lines.append(f"- `{k}`: {base_v:.4f} → {v:.4f} ({v-base_v:+.4f})")
        out_lines.append("")
        out_lines.append("**Process:**")
        for k, v in p.process_params.items():
            base_v = float(baseline[k])
            out_lines.append(f"- `{k}`: {base_v:.1f} → {v:.1f} ({v-base_v:+.1f})")
    out_lines.extend([
        "",
        "## Caveats",
        "",
        "- EI это математическое ожидание выгоды относительно `f*` "
        "(max-observed-train), считается из conformal-corrected 90% CI "
        "через нормальное приближение. Реальное распределение может "
        "быть skewed.",
        "- Cost-weighted ranking предпочитает дешёвые candidates даже с "
        "умеренной EI. Если приоритет — максимум σ независимо от стоимости, "
        "сортируйте по чистому EI вместо EI/cost.",
        "- Кандидаты с OOD-флагом ⚠️ — composition outside training "
        "distribution; ML predictions ненадёжны. Reviewer должен либо "
        "убрать их, либо явно принять OOD-риск.",
    ])

    out_path = DOCS_DIR / f"b2_next_experiments_{model_version}.md"
    out_path.write_text("\n".join(out_lines))
    logger.info("отчёт → %s", out_path.relative_to(PROJECT_ROOT))

    log_decision(
        phase="inverse_design",
        decision=f"ActiveLearner: top-{args.top_k} next experiments",
        reasoning=(
            f"model={model_version}, n_samples={args.n_samples}, "
            f"f*={f_star:.0f}, top1 EI/cost={proposals[0].acquisition_score:.4f} "
            f"if proposals else 'none'"
        ),
        context={
            "model_version": model_version,
            "baseline": {
                "predicted_property": base_pred,
                "cost_per_ton": float(baseline_cost),
            },
            "f_star": f_star,
            "n_samples": args.n_samples,
            "decision_vars": decision_vars,
            "proposals": [asdict(p) for p in proposals],
        },
        author="active_learner",
        tags=["active_learning"],
    )

    print()
    print("=" * 78)
    print(f"Active learning — top-{args.top_k} экспериментов")
    print("=" * 78)
    print(f"Baseline: σ={base_pred:.0f} МПа, cost={baseline_cost:.2f} €/т")
    print(f"f* (target к улучшению) = {f_star:.0f} МПа")
    print()
    print(f"{'#':>3} {'EI/cost':>9} {'EI':>7} {'σ pred':>7} "
          f"{'cost €/т':>9} {'Δσ':>7} {'Δcost':>8} {'OOD':>4}")
    for i, p in enumerate(proposals, 1):
        print(
            f"{i:>3} {p.acquisition_score:>9.4f} {p.expected_improvement:>7.1f} "
            f"{p.predicted_property:>7.0f} {p.cost_per_ton:>9.2f} "
            f"{p.delta_vs_baseline_property:>+7.0f} "
            f"{p.delta_vs_baseline_cost:>+8.2f} "
            f"{'⚠️' if p.ood_flag else '—':>4}"
        )
    print()
    print(f"полный отчёт: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
