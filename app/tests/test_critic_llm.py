"""Unit tests for critic_llm (mock Anthropic client — no real API calls)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from app.backend.critic_llm import (
    LLMCritic,
    LLMObservation,
    _build_user_payload,
    make_llm_critic,
)


def test_factory_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_llm_critic() is None


def test_factory_builds_client_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-dummy")
    critic = make_llm_critic()
    assert critic is not None
    assert isinstance(critic, LLMCritic)
    assert critic.model == "claude-sonnet-4-6"


def test_build_user_payload_has_expected_keys():
    ctx = {
        "r2_train": 0.95, "r2_val": 0.88, "r2_test": 0.86,
        "mae_test": 12.3, "rmse_test": 16.1, "coverage_90_ci": 0.91,
        "n_train": 1666, "n_val": 334, "n_test": 500,
        "split_strategy": "time_based", "cv_strategy": "group_kfold",
        "feature_importance": {
            "c_pct": 0.25, "mn_pct": 0.18, "nb_pct": 0.11, "ti_pct": 0.08,
            "rolling_finish_temp": 0.07, "cooling_rate_c_per_s": 0.05,
            "si_pct": 0.04, "cr_pct": 0.03, "ni_pct": 0.025, "cu_pct": 0.02,
            "noise_1": 0.01, "noise_2": 0.005,
        },
        "training_ranges": {"c_pct": [0.04, 0.12], "mn_pct": [0.9, 1.75]},
        "steel_class": "pipe_hsla",
        "target": "yield_strength_mpa",
    }
    payload_str = _build_user_payload(ctx)
    assert "json" in payload_str.lower()
    body = payload_str.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
    parsed = json.loads(body)
    assert parsed["metrics"]["r2_test"] == 0.86
    assert parsed["dataset_size"]["n_train"] == 1666
    assert len(parsed["feature_importance_top10"]) == 10
    assert list(parsed["feature_importance_top10"])[0] == "c_pct"
    assert "noise_2" not in parsed["feature_importance_top10"]


def _make_mock_response(observations: list[dict]) -> MagicMock:
    """Build a mock Anthropic response with a tool_use block."""
    resp = MagicMock()
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"observations": observations, "summary": "ok"}
    resp.content = [tool_block]
    resp.model = "claude-sonnet-4-6"
    resp.usage = MagicMock(
        input_tokens=1200,
        output_tokens=180,
        cache_read_input_tokens=800,
        cache_creation_input_tokens=0,
    )
    return resp


def test_review_training_happy_path():
    client = MagicMock()
    client.messages.create.return_value = _make_mock_response([
        {"severity": "HIGH", "category": "model",
         "message": "Top feature cu_pct подозрителен",
         "rationale": "Для HSLA медь не должна доминировать"},
        {"severity": "MEDIUM", "category": "physics",
         "message": "rolling_finish_temp диапазон 720-900°C шире обычного",
         "rationale": "Для pipe-HSLA оптимум 780-820°C"},
    ])
    critic = LLMCritic(client=client)

    with patch("app.backend.critic_llm._log_usage") as mock_log:
        result = critic.review_training({
            "r2_train": 0.95, "r2_val": 0.71,
            "feature_importance": {"cu_pct": 0.35, "c_pct": 0.10},
            "training_ranges": {"rolling_finish_temp": [720, 900]},
        })

    assert len(result) == 2
    assert isinstance(result[0], LLMObservation)
    assert result[0].severity == "HIGH"
    assert result[0].category == "model"
    assert result[1].severity == "MEDIUM"
    mock_log.assert_called_once()


def test_system_prompt_has_cache_control():
    client = MagicMock()
    client.messages.create.return_value = _make_mock_response([])
    critic = LLMCritic(client=client)

    critic.review_training({"r2_train": 0.9, "r2_val": 0.85})

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["max_tokens"] == 1200
    assert isinstance(kwargs["system"], list)
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["tool_choice"] == {"type": "tool", "name": "report_observations"}


def test_api_error_returns_empty_list(caplog):
    client = MagicMock()
    client.messages.create.side_effect = ConnectionError("network down")
    critic = LLMCritic(client=client)

    with caplog.at_level("WARNING", logger="app.backend.critic_llm"):
        result = critic.review_training({"r2_train": 0.9, "r2_val": 0.85})

    assert result == []
    assert any("API call failed" in r.message for r in caplog.records)


def test_response_without_tool_use_returns_empty_list(caplog):
    client = MagicMock()
    resp = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    resp.content = [text_block]
    client.messages.create.return_value = resp
    critic = LLMCritic(client=client)

    with caplog.at_level("WARNING", logger="app.backend.critic_llm"):
        result = critic.review_training({"r2_train": 0.9, "r2_val": 0.85})

    assert result == []
    assert any("no tool_use" in r.message for r in caplog.records)


def test_bad_payload_shape_returns_empty_list(caplog):
    client = MagicMock()
    resp = MagicMock()
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"wrong_key": "no observations here"}
    resp.content = [tool_block]
    resp.model = "claude-sonnet-4-6"
    resp.usage = MagicMock(input_tokens=10, output_tokens=5,
                           cache_read_input_tokens=0,
                           cache_creation_input_tokens=0)
    client.messages.create.return_value = resp
    critic = LLMCritic(client=client)

    with caplog.at_level("WARNING", logger="app.backend.critic_llm"):
        result = critic.review_training({"r2_train": 0.9, "r2_val": 0.85})

    assert result == []
    assert any("bad payload shape" in r.message for r in caplog.records)
