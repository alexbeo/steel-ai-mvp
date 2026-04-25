"""Tests for active_learner — EI math + LHS coverage + ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backend.active_learner import (
    ExperimentProposal,
    _norm_cdf,
    _norm_pdf,
    _sample_lhs,
    expected_improvement,
    propose_next_experiments,
)


def test_norm_helpers_match_scipy():
    """Sanity-check our analytic Φ, φ against known values."""
    # φ(0) = 1/sqrt(2π) ≈ 0.3989
    assert abs(_norm_pdf(0.0) - 0.3989422804014327) < 1e-10
    # Φ(0) = 0.5; Φ(1.96) ≈ 0.975
    assert abs(_norm_cdf(0.0) - 0.5) < 1e-10
    assert abs(_norm_cdf(1.96) - 0.975) < 1e-3


def test_ei_zero_when_uncertainty_zero_and_below_best():
    """Без неопределённости и ниже best — EI=0."""
    ei = expected_improvement(mu=400, ci_lower=400, ci_upper=400, f_star=500)
    assert abs(ei) < 1e-6


def test_ei_positive_when_mu_above_best():
    """Когда predicted выше f*, EI > 0."""
    ei = expected_improvement(mu=600, ci_lower=550, ci_upper=650, f_star=500)
    assert ei > 50  # roughly mu - f*


def test_ei_positive_with_uncertainty_even_below_best():
    """Широкая неопределённость может дать EI>0 даже когда μ < f*."""
    ei_low_unc = expected_improvement(mu=480, ci_lower=475, ci_upper=485, f_star=500)
    ei_high_unc = expected_improvement(mu=480, ci_lower=300, ci_upper=660, f_star=500)
    assert ei_high_unc > ei_low_unc
    assert ei_high_unc > 0


def test_lhs_coverage_within_bounds():
    """LHS должен покрывать диапазон равномерно, остаться внутри."""
    lo = np.array([0.0, 1.0])
    hi = np.array([1.0, 5.0])
    samples = _sample_lhs(50, lo, hi, seed=0)
    assert samples.shape == (50, 2)
    assert (samples >= lo).all()
    assert (samples <= hi).all()
    # каждая колонка должна покрывать большую часть диапазона
    for j in range(2):
        col = samples[:, j]
        assert (col.max() - col.min()) > 0.8 * (hi[j] - lo[j])


def test_lhs_deterministic_with_seed():
    s1 = _sample_lhs(20, np.array([0.0]), np.array([1.0]), seed=42)
    s2 = _sample_lhs(20, np.array([0.0]), np.array([1.0]), seed=42)
    assert np.array_equal(s1, s2)


def test_propose_next_experiments_ranks_by_acquisition():
    """End-to-end: маленькая модель + cost_fn → топ-кандидаты по EI/cost."""
    from xgboost import XGBRegressor

    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "mn_pct": rng.uniform(0.5, 1.5, 200),
        "si_pct": rng.uniform(0.2, 0.8, 200),
    })
    y = 100 + 200 * X["mn_pct"] + 50 * X["si_pct"] + rng.normal(0, 5, 200)

    main_m = XGBRegressor(n_estimators=80, max_depth=4, random_state=0)
    main_m.fit(X.values, y.values)
    q05 = XGBRegressor(
        n_estimators=80, max_depth=4, random_state=0,
        objective="reg:quantileerror", quantile_alpha=0.05,
    )
    q05.fit(X.values, y.values)
    q95 = XGBRegressor(
        n_estimators=80, max_depth=4, random_state=0,
        objective="reg:quantileerror", quantile_alpha=0.95,
    )
    q95.fit(X.values, y.values)

    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=2, random_state=0).fit(X.values)
    bundle = {
        "main": main_m, "q05": q05, "q95": q95,
        "ood": {"gmm": gmm, "comp_cols": ["mn_pct", "si_pct"]},
        "meta": {
            "feature_list": ["mn_pct", "si_pct"],
            "conformal_correction_mpa": 0.0,
        },
    }

    baseline = pd.Series({"mn_pct": 1.0, "si_pct": 0.5})

    def cost_fn(comp):
        # Mn 2 €/wt%, Si 0.5 €/wt% (per ton steel) — favours low Mn
        return 100 + 200 * comp.get("mn_pct", 0) + 50 * comp.get("si_pct", 0)

    proposals = propose_next_experiments(
        model_bundle=bundle,
        baseline_row=baseline,
        feature_list=["mn_pct", "si_pct"],
        decision_vars=["mn_pct", "si_pct"],
        bounds={"mn_pct": (0.5, 1.5), "si_pct": (0.2, 0.8)},
        f_star=float(y.max()),
        cost_fn=cost_fn,
        baseline_cost=cost_fn({"mn_pct": 1.0, "si_pct": 0.5}),
        baseline_property=float(main_m.predict([[1.0, 0.5]])[0]),
        n_samples=200,
        top_k=5,
        seed=42,
    )

    assert len(proposals) == 5
    assert all(isinstance(p, ExperimentProposal) for p in proposals)
    # отсортировано по acquisition_score убыв.
    scores = [p.acquisition_score for p in proposals]
    assert scores == sorted(scores, reverse=True)
    # all bounded
    for p in proposals:
        assert 0.5 <= p.composition["mn_pct"] <= 1.5
        assert 0.2 <= p.composition["si_pct"] <= 0.8
        assert p.cost_per_ton > 0
        assert p.expected_improvement >= 0
