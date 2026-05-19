"""Tests for symbolic_eta_critic (mocked Anthropic)."""
from __future__ import annotations

from unittest.mock import MagicMock

from app.backend.symbolic_eta_critic import (
    EtaFormulaVerdict,
    PhysicalCheck,
    SymbolicEtaCritic,
    make_symbolic_eta_critic,
)


def test_make_critic_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_symbolic_eta_critic() is None


def test_make_critic_no_prompt(monkeypatch):
    import app.backend.symbolic_eta_critic as mod
    monkeypatch.setattr(mod, "_SYSTEM_PROMPT_TEXT", None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert make_symbolic_eta_critic() is None


def _mock_response(payload):
    resp = MagicMock()
    tb = MagicMock()
    tb.type = "tool_use"
    tb.input = payload
    txt = MagicMock()
    txt.type = "text"
    resp.content = [txt, tb]
    resp.usage = MagicMock(
        input_tokens=2000, output_tokens=800,
        cache_read_input_tokens=1500, cache_creation_input_tokens=0,
    )
    resp.model = "claude-sonnet-4-6"
    resp.stop_reason = "tool_use"
    return resp


def _full_review_dict(**overrides):
    base = {
        "formula": "1.0 - 0.01 * slag_feo_pct",
        "verdict": "ACCEPT",
        "confidence": "HIGH",
        "summary": "FeO имеет отрицательный знак — физично, форма простая.",
        "physical_checks": [
            {"term": "slag_feo_pct", "expected_sign": "-",
             "found_sign": "-", "verdict": "matches"},
        ],
        "concerns": [],
    }
    base.update(overrides)
    return base


def test_critic_parses_well_formed(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response(_full_review_dict())
    import app.backend.symbolic_eta_critic as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)
    out = SymbolicEtaCritic(client=mock_client).review_formula(
        formula_infix="1.0 - 0.01 * slag_feo_pct",
        r2=0.62, complexity=7, feature_names=["slag_feo_pct"],
    )
    assert out is not None
    assert isinstance(out, EtaFormulaVerdict)
    assert out.verdict == "ACCEPT"
    assert len(out.physical_checks) == 1
    assert isinstance(out.physical_checks[0], PhysicalCheck)


def test_critic_malformed(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response({"summary": "ok"})
    import app.backend.symbolic_eta_critic as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)
    out = SymbolicEtaCritic(client=mock_client).review_formula(
        formula_infix="x", r2=0.1, complexity=3, feature_names=["x"],
    )
    assert out is None


def test_critic_api_failure():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("net")
    out = SymbolicEtaCritic(client=mock_client).review_formula(
        formula_infix="x", r2=0.1, complexity=3, feature_names=["x"],
    )
    assert out is None
