"""Unit tests for hypothesis_critic (mock Anthropic client — no real API calls)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from app.backend.hypothesis_critic import (
    CriticVerdict,
    HypothesisCritic,
    _build_user_payload,
    make_hypothesis_critic,
)


def test_factory_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_hypothesis_critic() is None


def test_factory_builds_client_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-dummy")
    crit = make_hypothesis_critic()
    assert crit is not None
    assert isinstance(crit, HypothesisCritic)
    assert crit.model == "claude-sonnet-4-6"


def test_build_user_payload_contains_artifact_and_hypotheses():
    artifact_ctx = {
        "steel_class": "fatigue_carbon_steel",
        "target": "fatigue_strength_mpa",
        "r2_test": 0.978, "mae_test": 16.6,
        "rmse_test": 21.0,
        "coverage_90_ci": 0.92,
        "conformal_correction_mpa": 15.9,
        "n_train": 290, "n_val": 60, "n_test": 87,
        "feature_importance": {
            "carburizing_temp_c": 0.475,
            "normalizing_temp_c": 0.41,
        },
        "training_ranges": {"c_pct": [0.17, 0.63]},
        "target_distribution": {"min": 225, "max": 1190, "mean": 553},
        "sample_predictions": [{"actual": 480, "pred": 472}],
    }
    hypotheses = [{
        "id": "abc12345",
        "statement": "test claim",
        "rationale": "because data",
        "proposed_experiment": {"fix": {}, "sweep": {"variable": "x", "range": [0, 1]}},
        "expected_outcome": "y",
        "novelty": "MEDIUM",
        "experiment_cost_estimate": "MEDIUM",
        "economic_impact": {
            "vs_classical_baseline": "trial-and-error",
            "estimated_saving": "qualitative",
            "measurement_method": "A/B",
        },
    }]
    payload = _build_user_payload(artifact_ctx, hypotheses)
    assert "Сводка артефакта модели" in payload
    assert "Гипотезы на review" in payload
    blocks = payload.split("```json")
    artifact_block = json.loads(blocks[1].split("```")[0])
    hyp_block = json.loads(blocks[2].split("```")[0])
    assert artifact_block["metrics"]["coverage_90_ci"] == 0.92
    assert artifact_block["metrics"]["conformal_correction_mpa"] == 15.9
    assert "carburizing_temp_c" in artifact_block["feature_importance_top10"]
    assert hyp_block[0]["id"] == "abc12345"
    # Provenance fields stripped: no model_version / tags in cleaned hypothesis
    assert "model_version" not in hyp_block[0]
    assert "tags" not in hyp_block[0]


def _make_mock_response(reviews: list[dict]) -> MagicMock:
    resp = MagicMock()
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"reviews": reviews}
    text_block = MagicMock(); text_block.type = "text"
    resp.content = [text_block, tool_block]
    resp.usage = MagicMock(
        input_tokens=2000, output_tokens=800,
        cache_read_input_tokens=1500, cache_creation_input_tokens=0,
    )
    resp.model = "claude-sonnet-4-6"
    resp.stop_reason = "tool_use"
    return resp


def _full_review_dict(**overrides):
    base = {
        "hypothesis_id": "abc12345",
        "verdict": "REVISE",
        "confidence": "MEDIUM",
        "summary": "Solid grounding but the experiment confounds two effects.",
        "strengths": ["опирается на feature_importance"],
        "weaknesses": ["sweep смешивает T и cooling rate"],
        "suggested_revision": "разделить sweep на 2 серии",
    }
    base.update(overrides)
    return base


def test_review_parses_well_formed_response(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response([_full_review_dict()])

    import app.backend.hypothesis_critic as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)

    crit = HypothesisCritic(client=mock_client)
    out = crit.review(
        artifact_ctx={"steel_class": "x", "target": "y"},
        hypotheses=[{"id": "abc12345", "statement": "test"}],
    )
    assert len(out) == 1
    v = out[0]
    assert isinstance(v, CriticVerdict)
    assert v.verdict == "REVISE"
    assert v.confidence == "MEDIUM"
    assert v.suggested_revision == "разделить sweep на 2 серии"
    assert "feature_importance" in v.strengths[0]


def test_review_handles_accept_with_null_revision(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response([
        _full_review_dict(verdict="ACCEPT", suggested_revision=None, weaknesses=[]),
    ])
    import app.backend.hypothesis_critic as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)

    out = HypothesisCritic(client=mock_client).review(
        artifact_ctx={}, hypotheses=[{"id": "abc12345"}]
    )
    assert len(out) == 1
    assert out[0].verdict == "ACCEPT"
    assert out[0].suggested_revision is None
    assert out[0].weaknesses == []


def test_review_returns_empty_for_empty_hypotheses_list():
    crit = HypothesisCritic(client=MagicMock())
    assert crit.review(artifact_ctx={}, hypotheses=[]) == []


def test_review_returns_empty_on_api_failure():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("network down")
    out = HypothesisCritic(client=mock_client).review(
        artifact_ctx={}, hypotheses=[{"id": "x"}]
    )
    assert out == []


def test_review_skips_malformed_entries(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response([
        _full_review_dict(),
        {"hypothesis_id": "incomplete"},  # missing required fields
    ])
    import app.backend.hypothesis_critic as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)

    out = HypothesisCritic(client=mock_client).review(
        artifact_ctx={}, hypotheses=[{"id": "abc12345"}, {"id": "incomplete"}]
    )
    assert len(out) == 1
    assert out[0].hypothesis_id == "abc12345"
