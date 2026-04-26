"""Tests for deoxidation_advisor (mocked Anthropic)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from app.backend.deoxidation_advisor import (
    DeoxidationAdvisor,
    DeoxidationAdvisory,
    _build_user_payload,
    make_deoxidation_advisor,
)


def test_factory_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_deoxidation_advisor() is None


def test_factory_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert isinstance(make_deoxidation_advisor(), DeoxidationAdvisor)


def test_payload_includes_heat_and_thermo():
    ctx = {
        "heat_context": {"o_a_init_ppm": 280, "target_o_a_ppm": 5,
                         "temp_c": 1580, "mass_t": 100,
                         "composition": {"c_pct": 0.20, "mn_pct": 1.2}},
        "thermo_estimates": {"fruehan_kg": 15.2, "sigworth_elliott_kg": 12.8,
                             "hayashi_kg": 16.1},
    }
    payload = _build_user_payload(ctx)
    body = payload.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
    parsed = json.loads(body)
    assert parsed["heat_context"]["o_a_init_ppm"] == 280
    assert parsed["thermo_estimates"]["fruehan_kg"] == 15.2


def _mock_response(payload):
    resp = MagicMock()
    tb = MagicMock(); tb.type = "tool_use"; tb.input = payload
    txt = MagicMock(); txt.type = "text"
    resp.content = [txt, tb]
    resp.usage = MagicMock(
        input_tokens=2000, output_tokens=1500,
        cache_read_input_tokens=1500, cache_creation_input_tokens=0,
    )
    resp.model = "claude-sonnet-4-6"
    resp.stop_reason = "tool_use"
    return resp


def _full_advisory_dict(**overrides):
    base = {
        "summary": "Standard wire-feeding 16 кг при 1580 °C, recovery 72%.",
        "al_addition_kg": 16.0,
        "al_form": "wire",
        "addition_strategy": "Подать 16 кг проволоки за 90 секунд, 8 мин argon stirring.",
        "expected_recovery_pct": 72.0,
        "kinetic_timing_min": [6.0, 10.0],
        "risk_flags": ["Mn/S ratio = 33 — пограничное"],
        "inclusion_forecast": "Доминирует Al2O3, Ca-treatment рекомендован.",
        "pre_actions": ["Замерить O_a", "Pre-deox SiMn если slag FeO>3%"],
        "post_actions": ["Sample на residual Al", "Ca-treat 0.18 кг/т"],
        "model_convergence_note": "Модели сходятся ±15%, выбран HY.",
        "evidence": [
            "thermo: HY=16.1 кг, recovery 70-75% при 1580°C",
            "mechanism: Turkdogan ladle data — Al recovery 70%",
            "process: addition 90s ≈ bath turnover 60s",
        ],
        "confidence": "HIGH",
    }
    base.update(overrides)
    return base


def test_advise_parses_well_formed(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response(_full_advisory_dict())
    import app.backend.deoxidation_advisor as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)
    out = DeoxidationAdvisor(client=mock_client).advise({
        "heat_context": {"o_a_init_ppm": 280},
        "thermo_estimates": {"fruehan_kg": 15.2},
    })
    assert out is not None
    assert isinstance(out, DeoxidationAdvisory)
    assert out.al_form == "wire"
    assert out.al_addition_kg == 16.0
    assert out.confidence == "HIGH"
    assert len(out.evidence) == 3
    assert out.kinetic_timing_min == [6.0, 10.0]


def test_advise_returns_none_on_api_failure():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("net down")
    assert DeoxidationAdvisor(client=mock_client).advise({}) is None


def test_advise_returns_none_on_malformed(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response({"summary": "ok"})
    import app.backend.deoxidation_advisor as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)
    assert DeoxidationAdvisor(client=mock_client).advise({}) is None
