"""Tests for deoxidation_critic (mocked Anthropic)."""
from __future__ import annotations

from unittest.mock import MagicMock

from app.backend.deoxidation_critic import (
    AdvisoryVerdict,
    DeoxidationCritic,
    EvidenceCheck,
    make_deoxidation_critic,
)


def test_factory_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_deoxidation_critic() is None


def test_factory_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert isinstance(make_deoxidation_critic(), DeoxidationCritic)


def _mock_response(payload):
    resp = MagicMock()
    tb = MagicMock(); tb.type = "tool_use"; tb.input = payload
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
        "advisory_id": "abc12345",
        "verdict": "ACCEPT",
        "confidence": "HIGH",
        "summary": "Числа в норме, recovery реалистичный, форма Al обоснована.",
        "evidence_check": [
            {"claim": "thermo HY=16.1 кг", "verdict": "VALID",
             "note": "соответствует thermo_estimates"},
        ],
        "strengths": ["recovery factor 72% реалистичен для wire+1580°C"],
        "weaknesses": [],
        "suggested_revision": None,
    }
    base.update(overrides)
    return base


def test_review_parses_well_formed(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response(_full_review_dict())
    import app.backend.deoxidation_critic as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)
    out = DeoxidationCritic(client=mock_client).review({}, {"id": "abc12345"})
    assert out is not None
    assert isinstance(out, AdvisoryVerdict)
    assert out.verdict == "ACCEPT"
    assert len(out.evidence_check) == 1
    assert isinstance(out.evidence_check[0], EvidenceCheck)


def test_review_returns_none_on_failure():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("net")
    assert DeoxidationCritic(client=mock_client).review({}, {"id": "x"}) is None


def test_review_returns_none_on_malformed(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response({"summary": "ok"})
    import app.backend.deoxidation_critic as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)
    assert DeoxidationCritic(client=mock_client).review({}, {"id": "x"}) is None
