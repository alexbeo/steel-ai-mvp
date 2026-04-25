"""Unit tests for hypothesis_generator (mock Anthropic client — no real API calls)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from app.backend.hypothesis_generator import (
    Hypothesis,
    HypothesisGenerator,
    _build_user_payload,
    make_hypothesis_generator,
)


def test_factory_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_hypothesis_generator() is None


def test_factory_builds_client_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-dummy")
    gen = make_hypothesis_generator()
    assert gen is not None
    assert isinstance(gen, HypothesisGenerator)
    assert gen.model == "claude-sonnet-4-6"


def test_build_user_payload_includes_all_expected_sections():
    ctx = {
        "steel_class": "fatigue_carbon_steel",
        "target": "fatigue_strength_mpa",
        "data_source": "Agrawal et al. 2014",
        "r2_train": 0.99, "r2_val": 0.94, "r2_test": 0.978,
        "mae_test": 16.6, "rmse_test": 21.0,
        "coverage_90_ci": 0.92, "conformal_correction_mpa": 15.9,
        "n_train": 290, "n_val": 60, "n_test": 87,
        "feature_importance": {
            "carburizing_temp_c": 0.475,
            "normalizing_temp_c": 0.406,
            "through_hardening_cooling_rate_c_per_s": 0.060,
            "cr_pct": 0.016,
            "through_hardening_temp_c": 0.010,
        },
        "training_ranges": {"c_pct": [0.17, 0.63]},
        "target_distribution": {"min": 225, "max": 1190, "mean": 553},
        "sample_predictions": [
            {"actual": 480, "pred": 472, "lower": 455, "upper": 495},
        ],
    }
    payload_str = _build_user_payload(ctx)
    body = payload_str.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
    parsed = json.loads(body)
    assert parsed["steel_class"] == "fatigue_carbon_steel"
    assert parsed["metrics"]["coverage_90_ci"] == 0.92
    assert parsed["metrics"]["conformal_correction_mpa"] == 15.9
    assert "carburizing_temp_c" in parsed["feature_importance_top10"]
    assert parsed["target_distribution"]["max"] == 1190
    assert len(parsed["sample_predictions"]) == 1


def _make_mock_response(hypotheses: list[dict]) -> MagicMock:
    resp = MagicMock()
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"hypotheses": hypotheses, "summary": "ok"}
    other_block = MagicMock()
    other_block.type = "text"
    resp.content = [other_block, tool_block]
    resp.usage = MagicMock(
        input_tokens=1500, output_tokens=400,
        cache_read_input_tokens=1200, cache_creation_input_tokens=0,
    )
    resp.model = "claude-sonnet-4-6"
    return resp


def _full_hypothesis_dict(**overrides):
    base = {
        "statement": "Carburizing presence dominates fatigue strength prediction.",
        "rationale": "carburizing_temp_c importance 0.475 — by far the largest.",
        "proposed_experiment": {
            "fix": {"c_pct": 0.20, "mn_pct": 0.85},
            "sweep": {"variable": "carburizing_temp_c",
                      "range": [800, 950], "step": 25},
        },
        "expected_outcome": "Fatigue rises by ~150 МПа above 870 °C.",
        "novelty": "MEDIUM",
        "experiment_cost_estimate": "MEDIUM",
        "economic_impact": {
            "vs_classical_baseline": "Trial-and-error: 5-10 melts at €5-15k each",
            "estimated_saving": "saves 4-6 melts (~€30-90k) per recipe iteration",
            "measurement_method": "compare 6 melt pairs at fixed and swept carb temp",
        },
    }
    base.update(overrides)
    return base


def test_generate_parses_well_formed_response(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response([_full_hypothesis_dict()])

    import app.backend.hypothesis_generator as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)

    gen = HypothesisGenerator(client=mock_client)
    out = gen.generate({
        "steel_class": "fatigue_carbon_steel",
        "target": "fatigue_strength_mpa",
        "feature_importance": {"carburizing_temp_c": 0.475},
    })
    assert len(out) == 1
    h = out[0]
    assert isinstance(h, Hypothesis)
    assert h.novelty == "MEDIUM"
    assert h.proposed_experiment["sweep"]["variable"] == "carburizing_temp_c"
    assert "steel_class:fatigue_carbon_steel" in h.tags
    assert len(h.id) == 8
    assert h.experiment_cost_estimate == "MEDIUM"
    assert h.economic_impact.vs_classical_baseline.startswith("Trial-and-error")
    assert "€" in h.economic_impact.estimated_saving


def test_generate_returns_empty_on_api_failure():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("network down")
    gen = HypothesisGenerator(client=mock_client)
    out = gen.generate({"steel_class": "fatigue_carbon_steel"})
    assert out == []


def test_generate_returns_empty_when_no_tool_block_in_response():
    mock_client = MagicMock()
    resp = MagicMock()
    text_block = MagicMock(); text_block.type = "text"
    resp.content = [text_block]
    resp.usage = MagicMock(input_tokens=10, output_tokens=10,
                           cache_read_input_tokens=0,
                           cache_creation_input_tokens=0)
    resp.model = "claude-sonnet-4-6"
    mock_client.messages.create.return_value = resp
    gen = HypothesisGenerator(client=mock_client)
    out = gen.generate({"steel_class": "fatigue_carbon_steel"})
    assert out == []


def test_generate_skips_malformed_hypothesis_entries(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response([
        _full_hypothesis_dict(statement="good one"),
        {"statement": "missing fields"},
    ])

    import app.backend.hypothesis_generator as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)

    gen = HypothesisGenerator(client=mock_client)
    out = gen.generate({"steel_class": "fatigue_carbon_steel"})
    assert len(out) == 1
    assert out[0].statement == "good one"
