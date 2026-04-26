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
