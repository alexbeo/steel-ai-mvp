"""Unit tests for split-conformal correction in train_model.

Verifies:
- `conformal_correction_mpa` is non-negative (undercovered quantiles widen,
  overcovered ones collapse to ~0).
- Post-correction coverage ≥ raw coverage (strictly equal only when raw is
  already ≥ target, in which case Q may still be positive but small).
- `predict_with_uncertainty` applies the correction: `upper - lower` widens
  by 2Q compared to raw quantiles.
- A model bundle without `conformal_correction_mpa` in meta (old artifacts)
  degrades gracefully to raw quantiles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.backend.data_curator import generate_synthetic_hsla_dataset
from app.backend.feature_eng import compute_hsla_features
from app.backend.model_trainer import (
    load_model,
    predict_with_uncertainty,
    train_model,
)
from app.backend.steel_classes import load_steel_class


def _small_hsla_training(n_samples: int = 300, trials: int = 5) -> dict:
    df_raw = generate_synthetic_hsla_dataset(n_samples=n_samples, random_seed=0)
    df = compute_hsla_features(df_raw)
    profile = load_steel_class("pipe_hsla")
    feat = [f for f in profile.feature_set if f in df.columns]
    trained = train_model(
        df_features=df,
        target="yield_strength_mpa",
        feature_list=feat,
        n_optuna_trials=trials,
        steel_class="pipe_hsla",
    )
    return {"trained": trained, "feat": feat, "df": df}


def test_conformal_correction_is_non_negative():
    res = _small_hsla_training()
    m = res["trained"].metrics
    assert m.conformal_correction_mpa >= 0.0
    assert 0.0 <= m.coverage_90_ci_raw <= 1.0
    assert 0.0 <= m.coverage_90_ci <= 1.0


def test_conformal_coverage_not_worse_than_raw():
    """Widening intervals with Q ≥ 0 can only increase or preserve coverage."""
    res = _small_hsla_training()
    m = res["trained"].metrics
    assert m.coverage_90_ci >= m.coverage_90_ci_raw - 1e-9


def test_conformal_widens_predicted_interval():
    """predict_with_uncertainty should return intervals wider than raw q05/q95
    by exactly 2 * conformal_correction_mpa."""
    res = _small_hsla_training()
    bundle = load_model(res["trained"].version)
    q = float(bundle["meta"]["conformal_correction_mpa"])

    df_sample = res["df"].iloc[:5].copy()
    out = predict_with_uncertainty(bundle, df_sample)

    raw_lo = bundle["q05"].predict(df_sample[res["feat"]])
    raw_hi = bundle["q95"].predict(df_sample[res["feat"]])
    np.testing.assert_allclose(out["lower_90"].values, raw_lo - q, atol=1e-6)
    np.testing.assert_allclose(out["upper_90"].values, raw_hi + q, atol=1e-6)


def test_quantile_crossing_rate_metric_present_and_low():
    """M10 (R-006 audit D-1): quantile crossing rate must be tracked in
    TrainingMetrics and must be < 1% on healthy training. XGBoost q05 and q95
    are independent — verifying they don't routinely cross is an integrity check."""
    res = _small_hsla_training()
    m = res["trained"].metrics
    assert hasattr(m, "quantile_crossing_rate"), \
        "TrainingMetrics must expose quantile_crossing_rate (M10 input)"
    assert 0.0 <= m.quantile_crossing_rate <= 1.0
    # On healthy synthetic HSLA, crossings should be rare (well under 1%).
    assert m.quantile_crossing_rate < 0.01, (
        f"Crossing rate = {m.quantile_crossing_rate:.1%} ≥ 1% on baseline "
        f"synthetic HSLA — model has UQ integrity issue"
    )


def test_pattern_m10_triggers_on_high_crossing_rate():
    """M10 pattern fires if crossing_rate > 1%, doesn't fire if ≤ 1%."""
    from pattern_library.patterns import _check_m10_quantile_crossing
    # No data → no trigger
    assert not _check_m10_quantile_crossing({}).triggered
    # Healthy: 0.5% < 1% threshold
    assert not _check_m10_quantile_crossing({"quantile_crossing_rate": 0.005}).triggered
    # Borderline: exactly 1% (strict >, not ≥) — no trigger
    assert not _check_m10_quantile_crossing({"quantile_crossing_rate": 0.01}).triggered
    # Triggers: 5% crossing
    res = _check_m10_quantile_crossing({"quantile_crossing_rate": 0.05})
    assert res.triggered
    assert "5.0%" in res.message
    assert res.details["crossing_rate"] == 0.05


def test_pattern_i04_pcm_gate_triggers_on_high_pcm():
    """I04 (R-006 D-2): Pcm > limit on Pareto candidate must trigger HIGH."""
    from pattern_library.patterns import _check_i04_recipe_pcm_within_limit
    # No data → no trigger
    assert not _check_i04_recipe_pcm_within_limit({}).triggered
    # Healthy: Pcm ~0.17 (HSLA range)
    res = _check_i04_recipe_pcm_within_limit({
        "pareto_candidates": [{"composition": {"c_pct": 0.08, "mn_pct": 1.5, "si_pct": 0.3}}],
        "pcm_limit": 0.22,
    })
    assert not res.triggered
    # Bad: Pcm > 0.22
    res = _check_i04_recipe_pcm_within_limit({
        "pareto_candidates": [{"composition": {"c_pct": 0.15, "mn_pct": 2.0}}],
        "pcm_limit": 0.22,
    })
    assert res.triggered
    assert "Pcm" in res.message


def test_pattern_i05_andrews_ms_triggers_on_low_ms():
    """I05 (R-006 D-3): Andrews Ms < 200°C must trigger HIGH (retained austenite risk)."""
    from pattern_library.patterns import (
        _check_i05_andrews_ms_above_threshold,
        _andrews_ms_from_recipe,
    )
    # Verify Andrews formula exact (no Si term per VERIFIED 2026-04-26)
    ms = _andrews_ms_from_recipe({
        "c_pct": 0.4, "mn_pct": 1.5, "ni_pct": 0.5, "cr_pct": 0.3, "mo_pct": 0.1,
    })
    assert 310 < ms < 312, f"Andrews returned {ms} — formula drift"
    # Healthy: low-C HSLA recipe Ms > 200
    res = _check_i05_andrews_ms_above_threshold({
        "pareto_candidates": [{"composition": {"c_pct": 0.08, "mn_pct": 1.0}}],
    })
    assert not res.triggered
    # Bad: very-alloyed recipe brings Ms below 200
    res = _check_i05_andrews_ms_above_threshold({
        "pareto_candidates": [{"composition": {
            "c_pct": 0.6, "mn_pct": 2.0, "ni_pct": 2.0, "cr_pct": 2.0,
        }}],
    })
    assert res.triggered
    assert "Ms" in res.message


def test_pattern_c05_element_content_sum():
    """C05 (R-006 D-4): element_content sum outside [0.90, 1.15] triggers MEDIUM."""
    from pattern_library.patterns import _check_c05_element_content_sum
    # Healthy: sum ~1.0
    res = _check_c05_element_content_sum({
        "snapshot_materials": [
            {"id": "FeMn-80", "kind": "ferroalloy", "element_content": {"Mn": 0.80, "Fe": 0.20}},
        ],
    })
    assert not res.triggered
    # Bad: sum 1.35 (data-entry error)
    res = _check_c05_element_content_sum({
        "snapshot_materials": [
            {"id": "BadAlloy", "kind": "ferroalloy",
             "element_content": {"Mn": 0.78, "Fe": 0.50, "C": 0.07}},
        ],
    })
    assert res.triggered
    assert "1.350" in res.message
    # Bad: sum 0.85 (missing element)
    res = _check_c05_element_content_sum({
        "snapshot_materials": [
            {"id": "Incomplete", "kind": "ferroalloy",
             "element_content": {"Cr": 0.65, "Fe": 0.20}},
        ],
    })
    assert res.triggered


def test_predict_falls_back_to_raw_when_meta_lacks_correction():
    """Old model artifacts (before this change) have no conformal_correction_mpa —
    predict_with_uncertainty must still work, treating Q as 0."""
    res = _small_hsla_training()
    bundle = load_model(res["trained"].version)
    bundle["meta"].pop("conformal_correction_mpa", None)

    df_sample = res["df"].iloc[:5].copy()
    out = predict_with_uncertainty(bundle, df_sample)
    raw_lo = bundle["q05"].predict(df_sample[res["feat"]])
    raw_hi = bundle["q95"].predict(df_sample[res["feat"]])
    np.testing.assert_allclose(out["lower_90"].values, raw_lo, atol=1e-6)
    np.testing.assert_allclose(out["upper_90"].values, raw_hi, atol=1e-6)
