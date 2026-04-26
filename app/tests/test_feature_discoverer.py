"""Unit tests for feature_discoverer (mock Anthropic client + real pandas eval)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.backend.feature_discoverer import (
    FeatureDiscoverer,
    FeatureProposal,
    FormulaError,
    _build_user_payload,
    apply_formula,
    make_feature_discoverer,
)


def test_factory_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_feature_discoverer() is None


def test_factory_builds_client_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-dummy")
    fd = make_feature_discoverer()
    assert fd is not None
    assert isinstance(fd, FeatureDiscoverer)


def test_build_user_payload_has_columns_and_metrics():
    ctx = {
        "steel_class": "fatigue_carbon_steel",
        "target": "fatigue_strength_mpa",
        "available_columns": ["c_pct", "mn_pct"],
        "current_feature_set": ["c_pct"],
        "feature_importance": {"c_pct": 0.5},
        "training_ranges": {"c_pct": [0.17, 0.63]},
        "r2_test": 0.978,
        "mae_test": 16.6,
        "n_train": 290,
        "n_test": 87,
    }
    payload = _build_user_payload(ctx)
    body = payload.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
    parsed = json.loads(body)
    assert parsed["available_columns"] == ["c_pct", "mn_pct"]
    assert parsed["metrics"]["r2_test"] == 0.978
    assert "c_pct" in parsed["feature_importance_top10"]


# ---------------------------------------------------------------------------
# apply_formula safety contract
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "c_pct": [0.20, 0.30, 0.50],
        "mn_pct": [0.80, 1.20, 0.60],
        "reduction_ratio": [500, 1500, 3000],
        "carburizing_temp_c": [30, 900, 920],
    })


def test_apply_formula_simple_ratio(sample_df):
    out = apply_formula(sample_df, "mn_pct / (c_pct + 0.001)", "mn_over_c")
    assert isinstance(out, pd.Series)
    assert out.name == "mn_over_c"
    assert len(out) == 3


def test_apply_formula_log_transform(sample_df):
    out = apply_formula(sample_df, "log(reduction_ratio + 1)", "log_reduction")
    assert (out > 0).all()


def test_apply_formula_comparison_binarization(sample_df):
    """pandas.DataFrame.eval doesn't support where() but does support
    comparison * 1 — verify the documented pattern actually works."""
    out = apply_formula(
        sample_df, "(carburizing_temp_c > 800) * 1", "is_carburized",
    )
    assert list(out.values) == [0, 1, 1]


def test_apply_formula_aggregate(sample_df):
    out = apply_formula(sample_df, "c_pct + mn_pct / 6", "cev_like")
    assert len(out) == 3


def test_apply_formula_unknown_column_raises(sample_df):
    with pytest.raises(FormulaError, match="failed"):
        apply_formula(sample_df, "ni_pct + 1", "bad")


def test_apply_formula_division_by_zero_raises():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 1.0]})
    with pytest.raises(FormulaError, match="inf"):
        apply_formula(df, "a / b", "ratio")


def test_apply_formula_nan_propagation_raises():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [1.0, 1.0]})
    with pytest.raises(FormulaError, match="NaN"):
        apply_formula(df, "a + b", "sum_with_nan")


def test_apply_formula_arbitrary_python_blocked(sample_df):
    """pandas.DataFrame.eval doesn't allow arbitrary Python — verify."""
    with pytest.raises(FormulaError):
        apply_formula(sample_df, "__import__('os').system('echo pwned')", "evil")


# ---------------------------------------------------------------------------
# discover() parsing
# ---------------------------------------------------------------------------

def _make_mock_response(features: list[dict]) -> MagicMock:
    resp = MagicMock()
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"features": features}
    text_block = MagicMock(); text_block.type = "text"
    resp.content = [text_block, tool_block]
    resp.usage = MagicMock(
        input_tokens=2000, output_tokens=800,
        cache_read_input_tokens=1500, cache_creation_input_tokens=0,
    )
    resp.model = "claude-sonnet-4-6"
    resp.stop_reason = "tool_use"
    return resp


def _full_proposal_dict(**overrides):
    base = {
        "name": "mn_over_c",
        "formula": "mn_pct / (c_pct + 0.001)",
        "mechanism_class": "ratio",
        "rationale": "Mn:C stoichiometry controls free Mn for MnS precipitation",
        "expected_uplift": "slight (R² +0.005-0.01)",
        "risk_notes": "min(c_pct)=0.17 in training, no division by zero",
    }
    base.update(overrides)
    return base


def test_discover_parses_well_formed_response(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response([_full_proposal_dict()])

    import app.backend.feature_discoverer as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)

    fd = FeatureDiscoverer(client=mock_client)
    out = fd.discover({"steel_class": "fatigue_carbon_steel", "target": "fatigue_strength_mpa"})
    assert len(out) == 1
    p = out[0]
    assert isinstance(p, FeatureProposal)
    assert p.name == "mn_over_c"
    assert p.mechanism_class == "ratio"
    assert "steel_class:fatigue_carbon_steel" in p.tags
    assert len(p.id) == 8


def test_discover_returns_empty_on_api_failure():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("network down")
    out = FeatureDiscoverer(client=mock_client).discover({"steel_class": "x"})
    assert out == []


def test_discover_skips_malformed_entries(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response([
        _full_proposal_dict(),
        {"name": "incomplete"},
    ])
    import app.backend.feature_discoverer as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)

    out = FeatureDiscoverer(client=mock_client).discover({"steel_class": "x"})
    assert len(out) == 1
    assert out[0].name == "mn_over_c"
