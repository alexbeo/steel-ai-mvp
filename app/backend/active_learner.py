"""
Active Learner — B2 cost-aware Bayesian acquisition for next-experiment
selection.

Стандартная Expected Improvement (EI) acquisition function, weighted
by 1/cost: предпочитает эксперименты, которые одновременно (a) скорее
всего улучшат целевое свойство относительно best-observed-train,
(b) дешёвые в исполнении.

Не требует LLM — это pure-numerical layer над existing XGBoost model
+ cost_model. Дополняет recipe_designer (Sonnet PhD) и hypothesis_
generator (Sonnet research): здесь — стохастический скан с
аналитической acquisition, вместо LLM reasoning.

EI formulation (Jones et al. 1998):
    z = (μ - f*) / σ
    EI(x) = (μ - f*) * Φ(z) + σ * φ(z)  if σ > 0 else max(μ - f*, 0)
где μ = predicted property, σ = (upper_90 - lower_90) / 2 / 1.6449
(нормальное приближение из conformal-corrected 90% CI), f* = max
наблюдаемой target в training data.

Cost-aware ranking: acquisition_score = EI(x) / cost(x). Кандидат с
большим EI и низкой стоимостью получает наибольший score.
"""
from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExperimentProposal:
    id: str
    composition: dict[str, float]
    process_params: dict[str, float]
    predicted_property: float
    lower_90: float
    upper_90: float
    uncertainty_width: float
    ood_flag: bool
    cost_per_ton: float
    expected_improvement: float
    acquisition_score: float
    delta_vs_baseline_property: float
    delta_vs_baseline_cost: float
    tags: list[str] = field(default_factory=list)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def expected_improvement(
    mu: float, ci_lower: float, ci_upper: float, f_star: float,
) -> float:
    """EI from a calibrated 90% CI assumed-normal."""
    sigma = max((ci_upper - ci_lower) / (2.0 * 1.6449), 1e-9)
    z = (mu - f_star) / sigma
    return (mu - f_star) * _norm_cdf(z) + sigma * _norm_pdf(z)


def _sample_lhs(n: int, lo: np.ndarray, hi: np.ndarray, seed: int = 42) -> np.ndarray:
    """Latin Hypercube Sampling within bounds — even coverage."""
    rng = np.random.default_rng(seed)
    d = len(lo)
    cuts = np.linspace(0, 1, n + 1)
    u = rng.uniform(size=(n, d))
    a = cuts[:n].reshape(-1, 1)
    b = cuts[1:].reshape(-1, 1)
    raw = a + (b - a) * u
    for j in range(d):
        rng.shuffle(raw[:, j])
    return lo + (hi - lo) * raw


def propose_next_experiments(
    model_bundle: dict,
    baseline_row: pd.Series,
    feature_list: list[str],
    decision_vars: list[str],
    bounds: dict[str, tuple[float, float]],
    f_star: float,
    cost_fn,
    baseline_cost: float,
    baseline_property: float,
    n_samples: int = 1000,
    top_k: int = 5,
    seed: int = 42,
) -> list[ExperimentProposal]:
    """Generate LHS-distributed candidates over decision_vars within
    bounds, score by EI / cost, return top_k ranked.

    Args:
        model_bundle: from model_trainer.load_model — has main, q05, q95,
            ood, meta with conformal_correction_mpa.
        baseline_row: median or specific recipe; non-decision features held
            at these values.
        feature_list: model input feature names in correct order.
        decision_vars: subset of feature_list that LHS will vary.
        bounds: {var: (lo, hi)} for each decision_var.
        f_star: best observed target in training (for EI).
        cost_fn: callable composition_dict -> float €/ton (or raises).
        baseline_cost / baseline_property: for delta computation.
    """
    from app.backend.model_trainer import predict_with_uncertainty

    lo = np.array([bounds[v][0] for v in decision_vars])
    hi = np.array([bounds[v][1] for v in decision_vars])
    samples = _sample_lhs(n_samples, lo, hi, seed=seed)

    rows = pd.DataFrame(
        np.tile(
            [float(baseline_row[f]) for f in feature_list],
            (n_samples, 1),
        ),
        columns=feature_list,
    )
    for j, var in enumerate(decision_vars):
        if var in feature_list:
            rows[var] = samples[:, j]

    pred_df = predict_with_uncertainty(model_bundle, rows)

    proposals: list[ExperimentProposal] = []
    for i in range(n_samples):
        mu = float(pred_df["prediction"].iloc[i])
        lo_i = float(pred_df["lower_90"].iloc[i])
        hi_i = float(pred_df["upper_90"].iloc[i])
        ood_i = bool(pred_df["ood_flag"].iloc[i])

        comp_dict = {
            k: float(rows[k].iloc[i])
            for k in feature_list if k.endswith("_pct")
        }
        try:
            cost_i = float(cost_fn(comp_dict))
        except Exception:
            continue
        if cost_i <= 0:
            continue

        ei = expected_improvement(mu, lo_i, hi_i, f_star)
        acq = ei / cost_i

        composition = {
            v: float(rows[v].iloc[i]) for v in decision_vars
            if v.endswith("_pct") and v in feature_list
        }
        process = {
            v: float(rows[v].iloc[i]) for v in decision_vars
            if not v.endswith("_pct") and v in feature_list
        }

        proposals.append(ExperimentProposal(
            id=str(uuid.uuid4())[:8],
            composition=composition,
            process_params=process,
            predicted_property=mu,
            lower_90=lo_i,
            upper_90=hi_i,
            uncertainty_width=hi_i - lo_i,
            ood_flag=ood_i,
            cost_per_ton=cost_i,
            expected_improvement=ei,
            acquisition_score=acq,
            delta_vs_baseline_property=mu - baseline_property,
            delta_vs_baseline_cost=cost_i - baseline_cost,
        ))

    proposals.sort(key=lambda p: -p.acquisition_score)
    return proposals[:top_k]
