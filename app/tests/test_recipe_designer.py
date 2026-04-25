"""Unit tests for recipe_designer (mocked Anthropic)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from app.backend.recipe_designer import (
    CompositionRecipe,
    RecipeDesigner,
    _build_user_payload,
    make_recipe_designer,
)


def test_factory_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_recipe_designer() is None


def test_factory_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert isinstance(make_recipe_designer(), RecipeDesigner)


def test_payload_contains_baseline_and_top_features():
    ctx = {
        "task": "снизить cost при сохранении σf",
        "steel_class": "fatigue_carbon_steel",
        "target": "fatigue_strength_mpa",
        "feature_importance": {"carburizing_temp_c": 0.475, "mn_pct": 0.05},
        "training_ranges": {"mn_pct": [0.37, 1.6]},
        "baseline_recipe": {"mn_pct": 0.75, "si_pct": 0.26},
        "baseline_predicted_property": 530,
        "baseline_cost_per_ton": 452.66,
        "available_composition": ["mn_pct", "si_pct"],
        "available_process": ["normalizing_temp_c"],
    }
    payload = _build_user_payload(ctx)
    body = payload.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
    parsed = json.loads(body)
    assert parsed["baseline_recipe"]["mn_pct"] == 0.75
    assert parsed["baseline_predicted_property"] == 530
    assert "carburizing_temp_c" in parsed["feature_importance_top10"]


def _mock_response(recipes):
    resp = MagicMock()
    tb = MagicMock(); tb.type = "tool_use"
    tb.input = {"recipes": recipes}
    txt = MagicMock(); txt.type = "text"
    resp.content = [txt, tb]
    resp.usage = MagicMock(
        input_tokens=2000, output_tokens=1500,
        cache_read_input_tokens=1500, cache_creation_input_tokens=0,
    )
    resp.model = "claude-sonnet-4-6"
    resp.stop_reason = "tool_use"
    return resp


def _full_recipe_dict(**overrides):
    base = {
        "name": "Low-Ni cost-saver",
        "composition": {"mn_pct": 0.4, "si_pct": 0.4, "ni_pct": 0.01},
        "process_params": {"normalizing_temp_c": 880, "tempering_temp_c": 200},
        "rationale": "Снижение Ni до 0.01 wt% даёт основную экономию...",
        "evidence": [
            "artifact: feature_importance[ni_pct] не входит в top-10",
            "mechanism: Ni в carbon steel <1% — marginal hardenability",
        ],
        "expected_outcome": "Δσf ≈ +50, Δcost ≈ −20 €/т",
        "risk_notes": "при Mn<0.4 — риск hot-shortness через S",
        "novelty": "MEDIUM",
    }
    base.update(overrides)
    return base


def test_design_parses_well_formed_response(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response([_full_recipe_dict()])
    import app.backend.recipe_designer as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)
    rd = RecipeDesigner(client=mock_client)
    out = rd.design({"steel_class": "fatigue_carbon_steel", "target": "fatigue_strength_mpa"})
    assert len(out) == 1
    r = out[0]
    assert isinstance(r, CompositionRecipe)
    assert r.name == "Low-Ni cost-saver"
    assert r.composition["ni_pct"] == 0.01
    assert len(r.evidence) == 2
    assert r.evidence[0].startswith("artifact:")
    assert r.evidence[1].startswith("mechanism:")
    assert "steel_class:fatigue_carbon_steel" in r.tags


def test_design_skips_malformed(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response([
        _full_recipe_dict(),
        {"name": "incomplete"},
    ])
    import app.backend.recipe_designer as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)
    out = RecipeDesigner(client=mock_client).design({})
    assert len(out) == 1


def test_design_returns_empty_on_api_failure():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("net down")
    assert RecipeDesigner(client=mock_client).design({"steel_class": "x"}) == []
