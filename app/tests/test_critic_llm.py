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

    with patch("app.backend.critic_llm._log_usage"):
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


def test_log_usage_writes_decision_log(monkeypatch, tmp_path):
    """_log_usage pipes token counts and observations into log_decision."""
    from app.backend.critic_llm import _log_usage
    from decision_log import logger as dl_module

    tmp_db = tmp_path / "test_decisions.db"
    monkeypatch.setattr(dl_module, "DEFAULT_DB_PATH", tmp_db)
    # log_decision's db_path default is bound at function-definition time,
    # so the setattr above doesn't affect it. Rebind the function's
    # __defaults__ to route the write into tmp_db and isolate the real DB.
    orig_defaults = dl_module.log_decision.__defaults__
    new_defaults = tuple(
        tmp_db if isinstance(d, type(tmp_db)) else d
        for d in orig_defaults
    )
    monkeypatch.setattr(dl_module.log_decision, "__defaults__", new_defaults)

    resp = MagicMock()
    resp.model = "claude-sonnet-4-6"
    resp.usage = MagicMock(
        input_tokens=1200,
        output_tokens=180,
        cache_read_input_tokens=800,
        cache_creation_input_tokens=0,
    )
    observations = [
        LLMObservation(severity="HIGH", category="model",
                       message="test msg", rationale="test why"),
    ]
    _log_usage(resp, elapsed_s=2.34, observations=observations)

    rows = dl_module.query_decisions(tag="llm_critic", db_path=tmp_db)
    assert len(rows) == 1
    r = rows[0]
    assert r["author"] == "llm_critic"
    assert "llm_critic" in r["tags"]
    assert "sonnet-4-6" in r["tags"]
    assert "1200" in r["reasoning"]
    assert "800" in r["reasoning"]
    assert r["context"]["usage"]["output_tokens"] == 180
    assert r["context"]["usage"]["cache_read"] == 800
    assert r["context"]["usage"]["latency_s"] == 2.34
    assert len(r["context"]["observations"]) == 1


def test_engine_critic_delegates_to_llm_critic_on_training():
    """engine.Critic.review('training', ctx) forwards to LLMCritic.review_training."""
    from app.backend.engine import Critic

    mock_llm_critic = MagicMock()
    mock_llm_critic.review_training.return_value = [
        LLMObservation(severity="MEDIUM", category="physics",
                       message="msg", rationale="why"),
    ]

    critic = Critic(use_llm=True, llm_critic=mock_llm_critic)
    report = critic.review("training", {
        "feature_importance": {"c_pct": 0.3, "mn_pct": 0.2},
        "r2_train": 0.9, "r2_val": 0.85, "coverage_90_ci": 0.88,
        "has_time_column": True, "has_groups": True,
        "split_strategy": "time_based", "cv_strategy": "group_kfold",
    })

    mock_llm_critic.review_training.assert_called_once()
    assert len(report.exploratory_observations) == 1
    assert report.exploratory_observations[0]["severity"] == "MEDIUM"
    assert report.exploratory_observations[0]["category"] == "physics"


def test_engine_critic_does_not_call_llm_outside_training():
    """Phases other than training must not invoke LLMCritic."""
    from app.backend.engine import Critic

    mock_llm_critic = MagicMock()
    critic = Critic(use_llm=True, llm_critic=mock_llm_critic)

    critic.review("inverse_design", {"pareto_size": 10})

    mock_llm_critic.review_training.assert_not_called()


def test_engine_critic_without_use_llm_does_not_call_llm_critic():
    """use_llm=False → LLMCritic never invoked even if instance provided."""
    from app.backend.engine import Critic

    mock_llm_critic = MagicMock()
    critic = Critic(use_llm=False, llm_critic=mock_llm_critic)

    critic.review("training", {"r2_train": 0.9, "r2_val": 0.85})

    mock_llm_critic.review_training.assert_not_called()
