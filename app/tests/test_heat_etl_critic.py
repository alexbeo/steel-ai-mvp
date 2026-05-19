"""Unit tests for heat_etl_critic (mocked Anthropic). Standard LLM-quintet."""
from __future__ import annotations

from unittest.mock import MagicMock

import app.backend.heat_etl_critic as etl_critic_mod
from app.backend.heat_etl_critic import (
    EtlMappingProposal,
    HeatEtlCritic,
    make_heat_etl_critic,
)


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    tb = MagicMock()
    tb.type = "tool_use"
    tb.input = payload
    txt = MagicMock()
    txt.type = "text"
    resp.content = [txt, tb]
    resp.usage = MagicMock(
        input_tokens=1500,
        output_tokens=400,
        cache_read_input_tokens=1200,
        cache_creation_input_tokens=0,
    )
    resp.model = "claude-sonnet-4-6"
    resp.stop_reason = "tool_use"
    return resp


def test_make_critic_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Prompt may be loaded or not — factory must still return None
    assert make_heat_etl_critic() is None


def test_make_critic_returns_none_without_prompt(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(etl_critic_mod, "_SYSTEM_PROMPT_TEXT", None)
    assert make_heat_etl_critic() is None


def test_make_critic_returns_instance_when_key_and_prompt(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(etl_critic_mod, "_SYSTEM_PROMPT_TEXT", "fake prompt")
    critic = make_heat_etl_critic()
    assert isinstance(critic, HeatEtlCritic)


def test_critic_parses_well_formed_response(monkeypatch):
    monkeypatch.setattr(etl_critic_mod, "_SYSTEM_PROMPT_TEXT", "fake prompt")
    monkeypatch.setattr(etl_critic_mod, "_log_usage", lambda *a, **kw: None)
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response(
        {
            "mappings": [
                {
                    "excel_column": "weird_col",
                    "heat_record_field": "o_a_initial_ppm",
                    "confidence": "HIGH",
                    "reasoning": "значения в типичном диапазоне 400-700 ppm",
                },
                {
                    "excel_column": "metoda",
                    "heat_record_field": "method_id",
                    "confidence": "MEDIUM",
                    "reasoning": "славянская транслитерация",
                },
            ],
            "unmapped": [
                {
                    "excel_column": "operator_signature",
                    "reason": "не является полем HeatRecord",
                },
            ],
        }
    )
    out = HeatEtlCritic(client=mock_client).propose_mappings(
        [
            {"excel_column": "weird_col", "sample_values": ["500", "600"]},
            {"excel_column": "metoda", "sample_values": ["asis_shot"]},
            {"excel_column": "operator_signature", "sample_values": ["Иванов"]},
        ]
    )
    assert isinstance(out, EtlMappingProposal)
    assert len(out.mappings) == 2
    assert out.mappings[0].excel_column == "weird_col"
    assert out.mappings[0].heat_record_field == "o_a_initial_ppm"
    assert out.mappings[0].confidence == "HIGH"
    assert len(out.unmapped) == 1
    assert out.unmapped[0].excel_column == "operator_signature"
    assert out.usage["input_tokens"] == 1500


def test_critic_handles_malformed_response(monkeypatch):
    monkeypatch.setattr(etl_critic_mod, "_SYSTEM_PROMPT_TEXT", "fake prompt")
    monkeypatch.setattr(etl_critic_mod, "_log_usage", lambda *a, **kw: None)
    mock_client = MagicMock()
    # Missing required keys
    mock_client.messages.create.return_value = _mock_response(
        {"mappings": [{"excel_column": "x"}], "unmapped": []}
    )
    out = HeatEtlCritic(client=mock_client).propose_mappings(
        [{"excel_column": "x", "sample_values": []}]
    )
    assert out is None


def test_critic_handles_no_tool_use_block(monkeypatch):
    monkeypatch.setattr(etl_critic_mod, "_SYSTEM_PROMPT_TEXT", "fake prompt")
    mock_client = MagicMock()
    resp = MagicMock()
    txt = MagicMock()
    txt.type = "text"
    resp.content = [txt]
    resp.usage = MagicMock(input_tokens=0, output_tokens=0)
    mock_client.messages.create.return_value = resp
    out = HeatEtlCritic(client=mock_client).propose_mappings(
        [{"excel_column": "x", "sample_values": []}]
    )
    assert out is None


def test_critic_handles_api_failure(monkeypatch):
    monkeypatch.setattr(etl_critic_mod, "_SYSTEM_PROMPT_TEXT", "fake prompt")
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("network")
    out = HeatEtlCritic(client=mock_client).propose_mappings(
        [{"excel_column": "x", "sample_values": []}]
    )
    assert out is None


def test_critic_empty_input_returns_empty_proposal(monkeypatch):
    monkeypatch.setattr(etl_critic_mod, "_SYSTEM_PROMPT_TEXT", "fake prompt")
    mock_client = MagicMock()
    out = HeatEtlCritic(client=mock_client).propose_mappings([])
    assert isinstance(out, EtlMappingProposal)
    assert out.mappings == []
    assert out.unmapped == []
    # API must not be called for empty input
    mock_client.messages.create.assert_not_called()
