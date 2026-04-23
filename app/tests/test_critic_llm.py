"""Unit tests for critic_llm (mock Anthropic client — no real API calls)."""
from __future__ import annotations

import json

from app.backend.critic_llm import (
    LLMCritic,
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
