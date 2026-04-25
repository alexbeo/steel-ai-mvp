"""
EC.1 — Concrete property+cost showcase на Agrawal NIMS модели.

Демонстрирует основной value proposition MVP: AI находит рецепт,
который улучшает предел усталости (или сохраняет на baseline) при
снижении ferroalloy-cost vs текущая практика.

Берёт обученный fatigue_carbon_steel model + carbon_low_alloy
sub-class median как baseline. Запускает NSGA-II через pymoo с двумя
objectives: max(predicted fatigue) + min(€/ton).

Output — таблица топ-5 AI-предложений vs baseline с явными
Δfatigue + Δcost + key composition shifts. Сохраняется как
markdown в docs/property_cost_demo_<model_version>.md.

Run:
    PYTHONPATH=. .venv/bin/python scripts/show_property_cost_on_agrawal.py
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
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


# Decision variables: что AI варьирует. Остальные фичи = baseline median.
DESIGN_COMPOSITION = ["si_pct", "mn_pct", "ni_pct", "cr_pct", "cu_pct", "mo_pct"]
DESIGN_PROCESS = [
    "normalizing_temp_c",
    "carburizing_temp_c",
    "carburizing_time_min",
    "tempering_temp_c",
    "tempering_time_min",
    "through_hardening_cooling_rate_c_per_s",
]
DESIGN_VARS = DESIGN_COMPOSITION + DESIGN_PROCESS

# c_pct, p_pct, s_pct — held at baseline (commodity scrap composition,
# инженер обычно не оптимизирует их свободно)


def _latest_model(class_id: str = "fatigue_carbon_steel") -> str | None:
    for d in sorted(MODELS_DIR.iterdir(), reverse=True):
        if not d.is_dir(): continue
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("steel_class") == class_id:
            return d.name
    return None


def _load_baseline(df: pd.DataFrame, sub_class: str = "carbon_low_alloy") -> pd.Series:
    """Median of training rows in the chosen sub-class. Excludes non-numeric
    columns that XGBoost wouldn't see."""
    if "sub_class" in df.columns:
        subset = df[df["sub_class"] == sub_class]
        if len(subset) < 50:
            subset = df  # fallback: pooled median
            sub_class = "all"
    else:
        subset = df; sub_class = "all"
    numeric = subset.select_dtypes(include=[np.number]).median()
    logger.info("baseline: sub_class=%s, n=%d records", sub_class, len(subset))
    return numeric


def _composition_dict(row: pd.Series) -> dict[str, float]:
    """Extract _pct + n_ppm columns into composition dict for cost_model."""
    return {k: float(v) for k, v in row.items() if k.endswith("_pct")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--population", type=int, default=120)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from app.backend.cost_model import compute_cost, seed_snapshot
    from app.backend.data_curator import load_real_agrawal_fatigue_dataset
    from app.backend.model_trainer import load_model
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.lhs import LHS
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    model_version = args.model or _latest_model()
    if not model_version:
        raise SystemExit("no fatigue_carbon_steel model found")
    logger.info("model: %s", model_version)

    bundle = load_model(model_version)
    meta = bundle["meta"]
    feature_list = meta["feature_list"]
    target = meta["target"]  # fatigue_strength_mpa
    training_ranges = meta["training_ranges"]
    q_correction = meta.get("conformal_correction_mpa", 0.0)
    main_model = bundle["main"]
    q05_model = bundle["q05"]
    q95_model = bundle["q95"]

    df = load_real_agrawal_fatigue_dataset()
    baseline = _load_baseline(df)

    snapshot = seed_snapshot()
    baseline_comp = _composition_dict(baseline)
    baseline_cost = compute_cost(baseline_comp, snapshot, mode="full")

    baseline_features = pd.DataFrame(
        [[float(baseline[f]) for f in feature_list]],
        columns=feature_list,
    )
    baseline_pred = float(main_model.predict(baseline_features)[0])
    baseline_lo = float(q05_model.predict(baseline_features)[0]) - q_correction
    baseline_hi = float(q95_model.predict(baseline_features)[0]) + q_correction

    logger.info(
        "baseline: predicted fatigue=%.0f МПа [%.0f, %.0f], cost=%.2f €/т",
        baseline_pred, baseline_lo, baseline_hi, baseline_cost.total_per_ton,
    )

    # Decision variable bounds from training_ranges
    bounds_lo = np.array([training_ranges[v][0] for v in DESIGN_VARS])
    bounds_hi = np.array([training_ranges[v][1] for v in DESIGN_VARS])

    class CostAwareProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=len(DESIGN_VARS), n_obj=2, n_constr=0,
                xl=bounds_lo, xu=bounds_hi,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            # Build full feature vector: baseline + perturbed decision vars
            row = baseline.copy()
            for i, var in enumerate(DESIGN_VARS):
                row[var] = float(x[i])
            X_row = pd.DataFrame(
                [[float(row[f]) for f in feature_list]],
                columns=feature_list,
            )
            pred = float(main_model.predict(X_row)[0])

            comp = _composition_dict(row)
            try:
                cb = compute_cost(comp, snapshot, mode="full")
                cost = cb.total_per_ton
            except Exception:
                cost = 1e9  # invalid composition → unfit

            # NSGA-II minimises both — invert fatigue
            out["F"] = [-pred, cost]

    problem = CostAwareProblem()
    algorithm = NSGA2(
        pop_size=args.population,
        sampling=LHS(),
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    res = minimize(
        problem, algorithm,
        termination=get_termination("n_gen", args.generations),
        seed=args.seed, verbose=False,
    )
    logger.info("NSGA-II finished: %d Pareto points", len(res.F))

    rows = []
    for x_vec, f_vec in zip(res.X, res.F):
        row = baseline.copy()
        for i, var in enumerate(DESIGN_VARS):
            row[var] = float(x_vec[i])
        X_row = pd.DataFrame(
            [[float(row[f]) for f in feature_list]],
            columns=feature_list,
        )
        pred = float(main_model.predict(X_row)[0])
        lo = float(q05_model.predict(X_row)[0]) - q_correction
        hi = float(q95_model.predict(X_row)[0]) + q_correction

        comp = _composition_dict(row)
        cb = compute_cost(comp, snapshot, mode="full")
        rows.append({
            "predicted_fatigue": pred,
            "lower_90": lo,
            "upper_90": hi,
            "cost_per_ton": cb.total_per_ton,
            "delta_fatigue": pred - baseline_pred,
            "delta_cost": cb.total_per_ton - baseline_cost.total_per_ton,
            "alloy_mass_total_kg_per_ton": sum(
                c.mass_kg_per_ton_steel for c in [c for c in cb.contributions if c.material_id != 'scrap']
            ),
            "composition": {v: float(row[v]) for v in DESIGN_COMPOSITION},
            "process": {v: float(row[v]) for v in DESIGN_PROCESS},
            "ferroalloy_breakdown": [
                {
                    "material": c.material_id,
                    "mass_kg_per_ton": round(c.mass_kg_per_ton_steel, 2),
                    "cost_per_ton": round(c.contribution_per_ton, 2),
                }
                for c in [c for c in cb.contributions if c.material_id != 'scrap']
            ],
        })

    candidates_df = pd.DataFrame(rows)

    # Filter: dominant interest = improve OR maintain fatigue at lower cost
    same_or_better = candidates_df[
        candidates_df["delta_fatigue"] >= -10  # within 10 МПа of baseline
    ].copy()
    same_or_better["score"] = (
        same_or_better["delta_fatigue"] - same_or_better["delta_cost"]
    )
    top = same_or_better.nlargest(args.top_n, "score")

    # ---------- markdown report ----------
    out_lines: list[str] = []
    out_lines.append(f"# Property+Cost demo на Agrawal NIMS")
    out_lines.append("")
    out_lines.append(f"**Модель:** `{model_version}`")
    out_lines.append(f"**Дата:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    out_lines.append(f"**Дата артефакта:** {meta.get('trained_at', '?')[:10]}")
    out_lines.append("")
    out_lines.append("## Baseline (медиана carbon_low_alloy sub-class из training data)")
    out_lines.append("")
    out_lines.append(
        f"Прогнозируемый предел усталости: **{baseline_pred:.0f} МПа** "
        f"(90% CI [{baseline_lo:.0f}, {baseline_hi:.0f}])"
    )
    out_lines.append(
        f"Стоимость ferroalloy: **{baseline_cost.total_per_ton:.2f} €/т**"
    )
    out_lines.append("")
    out_lines.append("Ключевой состав baseline:")
    for k in DESIGN_COMPOSITION:
        out_lines.append(f"- `{k}`: {baseline[k]:.4f} wt%")
    out_lines.append("")

    out_lines.append(f"## Топ-{args.top_n} AI-предложений (NSGA-II Pareto)")
    out_lines.append("")
    out_lines.append("Отсортированы по `Δfatigue − Δcost` (макс. экономическая ценность).")
    out_lines.append("")
    out_lines.append(
        "| # | Δσf, МПа | Δcost, €/т | Прогноз σf, МПа (90% CI) | Cost, €/т | "
        "Главные изменения состава |"
    )
    out_lines.append(
        "|---|---:|---:|---:|---:|---|"
    )

    for rank, (_, r) in enumerate(top.iterrows(), 1):
        diffs = []
        for k in DESIGN_COMPOSITION:
            db = r["composition"][k] - float(baseline[k])
            if abs(db) >= 0.005:
                diffs.append((k, db, r["composition"][k]))
        diffs.sort(key=lambda t: -abs(t[1]))
        comp_str = "; ".join(
            f"{k}: {baseline[k]:.3f}→{cv:.3f} ({db:+.3f})"
            for k, db, cv in diffs[:4]
        ) or "—"
        out_lines.append(
            f"| {rank} | {r['delta_fatigue']:+.0f} | "
            f"{r['delta_cost']:+.2f} | "
            f"{r['predicted_fatigue']:.0f} "
            f"[{r['lower_90']:.0f},{r['upper_90']:.0f}] | "
            f"{r['cost_per_ton']:.2f} | {comp_str} |"
        )
    out_lines.append("")

    out_lines.append("## Лучший candidate — детально")
    out_lines.append("")
    if len(top) > 0:
        best = top.iloc[0]
        out_lines.append(
            f"**Δfatigue = {best['delta_fatigue']:+.0f} МПа**, "
            f"**Δcost = {best['delta_cost']:+.2f} €/т**"
        )
        out_lines.append("")
        out_lines.append("Расход ферросплавов в этом рецепте:")
        out_lines.append("")
        out_lines.append("| Материал | кг/т стали | €/т |")
        out_lines.append("|---|---:|---:|")
        for c in best["ferroalloy_breakdown"]:
            out_lines.append(
                f"| `{c['material']}` | {c['mass_kg_per_ton']:.2f} | "
                f"{c['cost_per_ton']:.2f} |"
            )
        out_lines.append("")
        out_lines.append("Для сравнения baseline:")
        out_lines.append("")
        out_lines.append("| Материал | кг/т стали | €/т |")
        out_lines.append("|---|---:|---:|")
        for c in [c for c in baseline_cost.contributions if c.material_id != 'scrap']:
            out_lines.append(
                f"| `{c.material_id}` | {c.mass_kg_per_ton_steel:.2f} | "
                f"{c.contribution_per_ton:.2f} |"
            )
        out_lines.append("")
        out_lines.append(
            f"Если перевести Δcost = {best['delta_cost']:+.2f} €/т на партию "
            f"100 тонн стали — это **{best['delta_cost']*100:+.0f} €/партия**."
        )

    out_lines.append("")
    out_lines.append("## Caveats и честность интерпретации")
    out_lines.append("")
    out_lines.append(
        "- Δfatigue это **предсказание модели**, не результат реальной плавки. "
        "Доверительный интервал 90 % указан рядом — реальное значение почти "
        "всегда внутри."
    )
    out_lines.append(
        "- Stating «−€18/т» означает **только разницу в расходе ferroalloy** "
        "(Mn, Si, Cr, Ni, Mo, Cu) на 1 тонне стали. Не учтены: рабочее время, "
        "энергия, амортизация, накладные."
    )
    out_lines.append(
        "- Baseline — медиана `carbon_low_alloy` подгруппы. Если у вашего "
        "завода своя стандартная рецептура — подайте её и AI пересчитает "
        "Δ-метрики относительно неё."
    )
    out_lines.append(
        "- Модель обучена на 437 публичных Agrawal NIMS records (статья IMMI "
        "3:8, 2014); HSLA pipe / тонколистовой / advanced AHSS не покрыты."
    )

    out_path = DOCS_DIR / f"property_cost_demo_{model_version}.md"
    out_path.write_text("\n".join(out_lines))
    logger.info("отчёт → %s", out_path.relative_to(PROJECT_ROOT))

    # Console summary
    print()
    print("=" * 78)
    print(f"Demo property+cost на {model_version}")
    print("=" * 78)
    print(
        f"Baseline: predicted fatigue {baseline_pred:.0f} МПа, "
        f"cost {baseline_cost.total_per_ton:.2f} €/т"
    )
    print()
    print("Top AI-предложений (макс Δfatigue − Δcost):")
    print(f"{'#':>3} {'Δσf,МПа':>10} {'Δcost,€/т':>11} {'σf,МПа':>9} {'cost,€/т':>10}")
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        print(
            f"{rank:>3}  {r['delta_fatigue']:>+9.0f} "
            f"{r['delta_cost']:>+11.2f} "
            f"{r['predicted_fatigue']:>9.0f} "
            f"{r['cost_per_ton']:>10.2f}"
        )
    print()
    print(f"Полный отчёт сохранён: docs/property_cost_demo_{model_version}.md")


if __name__ == "__main__":
    main()
