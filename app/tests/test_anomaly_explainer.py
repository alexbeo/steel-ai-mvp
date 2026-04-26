"""Tests for anomaly_explainer (mocked Anthropic)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from app.backend.anomaly_explainer import (
    AnomalousFeature,
    AnomalyExplainer,
    AnomalyExplanation,
    _build_user_payload,
    make_anomaly_explainer,
)


def test_factory_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_anomaly_explainer() is None


def test_factory_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert isinstance(make_anomaly_explainer(), AnomalyExplainer)


def test_payload_contains_recipe_and_ranges():
    ctx = {
        "steel_class": "fatigue_carbon_steel",
        "target": "fatigue_strength_mpa",
        "recipe": {"mn_pct": 1.8, "c_pct": 0.15},
        "training_ranges": {"mn_pct": [0.37, 1.6], "c_pct": [0.17, 0.63]},
        "out_of_range_features": ["mn_pct", "c_pct"],
        "ml_prediction": {"mu": 480, "lower_90": 200, "upper_90": 850},
        "ood_flag": True,
    }
    payload = _build_user_payload(ctx)
    body = payload.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
    parsed = json.loads(body)
    assert parsed["recipe"]["mn_pct"] == 1.8
    assert parsed["training_ranges"]["mn_pct"] == [0.37, 1.6]
    assert parsed["ood_flag"] is True


def _mock_response(payload):
    resp = MagicMock()
    tb = MagicMock(); tb.type = "tool_use"; tb.input = payload
    txt = MagicMock(); txt.type = "text"
    resp.content = [txt, tb]
    resp.usage = MagicMock(
        input_tokens=1500, output_tokens=800,
        cache_read_input_tokens=1200, cache_creation_input_tokens=0,
    )
    resp.model = "claude-sonnet-4-6"
    resp.stop_reason = "tool_use"
    return resp


def test_explain_parses_well_formed(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response({
        "summary": "Mn=1.8 wt% выходит за training_ranges; риск austenite retention.",
        "anomalous_features": [
            {
                "feature": "mn_pct", "value": 1.8,
                "training_range": [0.37, 1.6],
                "deviation_kind": "out_of_range_high",
                "note": "На 12.5% выше верхней границы training",
            },
        ],
        "mechanism_concerns": [
            "Mn>1.6 wt% при низком C ведёт к austenite retention на отпуске",
        ],
        "production_risks": "Pищ retained austenite → нестабильная hardness.",
        "suggested_correction": "Снизить Mn до 1.5 или повысить C до 0.30.",
        "severity": "MEDIUM",
    })
    import app.backend.anomaly_explainer as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)
    out = AnomalyExplainer(client=mock_client).explain({
        "steel_class": "fatigue_carbon_steel", "recipe": {"mn_pct": 1.8},
    })
    assert out is not None
    assert isinstance(out, AnomalyExplanation)
    assert out.severity == "MEDIUM"
    assert len(out.anomalous_features) == 1
    assert isinstance(out.anomalous_features[0], AnomalousFeature)
    assert out.anomalous_features[0].deviation_kind == "out_of_range_high"


def test_explain_returns_none_on_api_failure():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("net down")
    assert AnomalyExplainer(client=mock_client).explain({"steel_class": "x"}) is None


def test_explain_returns_none_on_malformed_payload(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response({
        "summary": "ok",
        # missing required fields
    })
    import app.backend.anomaly_explainer as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)
    out = AnomalyExplainer(client=mock_client).explain({"steel_class": "x"})
    assert out is None
