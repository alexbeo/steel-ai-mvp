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
