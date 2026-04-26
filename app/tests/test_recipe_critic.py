"""Unit tests for recipe_critic (mocked Anthropic)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from app.backend.recipe_critic import (
    EvidenceCheck,
    RecipeCritic,
    RecipeVerdict,
    _build_user_payload,
    make_recipe_critic,
)


def test_factory_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_recipe_critic() is None


def test_factory_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert isinstance(make_recipe_critic(), RecipeCritic)


def test_payload_includes_artifact_and_recipes_with_verification():
    artifact_ctx = {
        "steel_class": "fatigue_carbon_steel",
        "target": "fatigue_strength_mpa",
        "r2_test": 0.978,
        "feature_importance": {"carb_temp": 0.475},
        "baseline_recipe": {"mn_pct": 0.75},
        "baseline_predicted_property": 530,
    }
    recipes_with = [{
        "id": "abc12345",
        "name": "test",
        "composition": {"mn_pct": 0.4},
        "evidence": ["artifact: ...", "mechanism: ..."],
        "ml_verification": {
            "predicted_property": 600, "lower_90": 500, "upper_90": 700,
            "cost_per_ton": 430, "delta_property": 70, "delta_cost": -22,
        },
    }]
    payload = _build_user_payload(artifact_ctx, recipes_with)
    blocks = payload.split("```json")
    art = json.loads(blocks[1].split("```")[0])
    rec = json.loads(blocks[2].split("```")[0])
    assert art["metrics"]["r2_test"] == 0.978
    assert art["baseline_predicted_property"] == 530
    assert rec[0]["ml_verification"]["delta_cost"] == -22


def _mock_response(reviews):
    resp = MagicMock()
    tb = MagicMock(); tb.type = "tool_use"
    tb.input = {"reviews": reviews}
    txt = MagicMock(); txt.type = "text"
    resp.content = [txt, tb]
    resp.usage = MagicMock(
        input_tokens=3000, output_tokens=1200,
        cache_read_input_tokens=2500, cache_creation_input_tokens=0,
    )
    resp.model = "claude-sonnet-4-6"
    resp.stop_reason = "tool_use"
    return resp


def _full_review_dict(**overrides):
    base = {
        "recipe_id": "abc12345",
        "verdict": "REVISE",
        "confidence": "HIGH",
        "summary": "Evidence solid, но ML predicted +5 МПа vs claimed +50.",
        "evidence_check": [
            {"claim": "feature_importance[ni_pct] not top-10",
             "verdict": "VALID", "note": "matches payload"},
            {"claim": "Pickering при низком C",
             "verdict": "VALID", "note": "applicable in this range"},
        ],
        "strengths": ["оба источника evidence корректны"],
        "weaknesses": ["expected_outcome завышено vs ML truth-gate"],
        "suggested_revision": "снизить expected_outcome до +5 МПа или объяснить gap",
    }
    base.update(overrides)
    return base


def test_review_parses_well_formed(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response([_full_review_dict()])
    import app.backend.recipe_critic as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)
    out = RecipeCritic(client=mock_client).review(
        artifact_ctx={"steel_class": "x"},
        recipes_with_verification=[{"id": "abc12345"}],
    )
    assert len(out) == 1
    v = out[0]
    assert isinstance(v, RecipeVerdict)
    assert v.verdict == "REVISE"
    assert len(v.evidence_check) == 2
    assert all(isinstance(ec, EvidenceCheck) for ec in v.evidence_check)
    assert v.evidence_check[0].verdict == "VALID"


def test_review_handles_accept_with_null_revision(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response([
        _full_review_dict(verdict="ACCEPT", suggested_revision=None, weaknesses=[]),
    ])
    import app.backend.recipe_critic as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)
    out = RecipeCritic(client=mock_client).review({}, [{"id": "abc12345"}])
    assert out[0].verdict == "ACCEPT"
    assert out[0].suggested_revision is None


def test_review_returns_empty_for_empty_input():
    assert RecipeCritic(client=MagicMock()).review({}, []) == []


def test_review_returns_empty_on_api_failure():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("network")
    assert RecipeCritic(client=mock_client).review({}, [{"id": "x"}]) == []
