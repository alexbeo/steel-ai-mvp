# Critic v2 — LLM-review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Заменить заглушку `Critic._llm_review` в `engine.py` на работающий Claude Sonnet 4.6-вызов, который на фазе `training` добавляет информационные observations, не блокируя pipeline.

**Architecture:** Новый модуль `app/backend/critic_llm.py` изолирует всю работу с Anthropic SDK. `LLMCritic.review_training(context)` собирает промпт с cached system prompt, вызывает API с `tool_use` structured output, fail-open на любые ошибки. Активация авто через `ANTHROPIC_API_KEY`. Интеграция в `engine.py:Critic` — одна замена блока; UI получает новую секцию «🤖 LLM-Critic» после Pattern Library warnings.

**Tech Stack:** `anthropic>=0.18` SDK (уже в requirements.txt), Python 3.12, pytest + `unittest.mock.MagicMock` для API без реальных вызовов. Streamlit UI.

**Spec:** `docs/superpowers/specs/2026-04-23-critic-v2-llm-review-design.md`

---

## File Structure

### Новые
- `app/backend/critic_llm.py` — `LLMCritic` class, `make_llm_critic()` factory, `_SYSTEM_PROMPT_TEXT`, `_TOOL_SCHEMA`, `_build_user_payload`, `_log_usage`, `LLMObservation` dataclass.
- `app/tests/test_critic_llm.py` — 6 тестов на mock'ах.

### Модифицируемые
- `app/backend/engine.py`: `Critic.__init__` принимает `llm_critic`, `CriticReport.exploratory_observations: list[dict]`, блок `_llm_review` удаляется и заменяется inline-вызовом `LLMCritic.review_training`.
- `scripts/run_pipeline.py`: `Critic(use_llm=True)` вместо `use_llm=False`.
- `app/frontend/app.py`: sidebar status «🤖 LLM-Critic», блок observations в tab_train после Pattern Library warnings, `critic_ctx` обогащается `training_ranges` и `n_train/val/test`.
- `CLAUDE.md`: строка про LLM-Critic в секции «Critic и Pattern Library» + env var требование.

---

## Task 1: Scaffold critic_llm.py — types, factory, constants

**Files:**
- Create: `app/backend/critic_llm.py`

- [ ] **Step 1: Создать модуль с типами и фабрикой, БЕЗ логики API**

Create `app/backend/critic_llm.py`:

```python
"""
LLM-Critic v2 — Claude Sonnet 4.6 as exploratory reviewer on training phase.

Activated via ANTHROPIC_API_KEY env var; returns an empty observation
list on any API failure so the pipeline continues as if LLM-Critic
were not configured. Observations are informational — they do not
affect Verdict (Pattern Library remains the sole gate).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Literal

logger = logging.getLogger(__name__)

Severity = Literal["HIGH", "MEDIUM", "LOW"]
Category = Literal["data", "model", "physics", "process"]


@dataclass
class LLMObservation:
    severity: Severity
    category: Category
    message: str
    rationale: str


_SYSTEM_PROMPT_TEXT = """\
Ты senior ML-инженер с 10-летним опытом в металлургии HSLA-сталей
(трубопроводные, K60-K65). Тебе на review поступает training-артефакт
от XGBoost-пайплайна: метрики, feature importance, training_ranges,
split/CV strategy, размеры выборок.

Твоя задача — выступить вторым мнением после детерминированной
Pattern Library. Ищи то, что правила не видят:

1. METRICS & CALIBRATION
   - Коэффициенты R² правдоподобны для текущего размера датасета?
   - Gap r2_train − r2_val / r2_test указывает на overfitting?
   - Coverage 90% CI в ожидаемом диапазоне 85-95%?
     Под-confidence (<85%) / сверх-confidence (>95%)?

2. FEATURE IMPORTANCE PHYSICS
   - Для pipe-HSLA с target σт / σв / δ ожидаются в top-10:
     c_pct, mn_pct, nb_pct, ti_pct, v_pct, rolling_finish_temp,
     cooling_rate, cev_iiw, pcm, microalloying_sum.
   - Если в top-5 «экзотика» (cu_pct, s_pct, n_ppm) без Nb/Ti —
     подозрение на spurious correlation или data leakage.
   - Суммарная доля одной фичи > 50% — возможна утечка target'а.

3. DATA LEAKAGE VIA SPLIT
   - Если split_strategy != "time_based" на данных с временной
     колонкой — high risk leakage.
   - Если cv_strategy != "group_kfold" на данных с groups —
     оптимистичный CV-score.

4. TRAINING_RANGES PHYSICAL SANITY
   - Диапазоны должны быть в типичных для pipe-HSLA пределах:
     C 0.03-0.15 %; Mn 0.8-1.8; Nb 0-0.06; Ti 0-0.03; Si 0.1-0.6;
     rolling_finish_temp 740-860 °C; cooling_rate 5-30 °C/s.
   - Выход за эти пределы → либо другой класс стали, либо ошибка
     генерации данных.

ФОРМАТ ОТВЕТА — через tool report_observations:
- До 5 observations (выбирай самые важные).
- severity: HIGH (стоп-сигнал для senior'а), MEDIUM (нужно
  выяснить), LOW (к сведению).
- category: data | model | physics | process.
- message и rationale на русском.
- Если всё чисто — верни пустой список. Не придумывай проблемы.
"""


_TOOL_SCHEMA = {
    "name": "report_observations",
    "description": "Report observations about training artifact quality",
    "input_schema": {
        "type": "object",
        "properties": {
            "observations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string",
                                     "enum": ["HIGH", "MEDIUM", "LOW"]},
                        "category": {"type": "string",
                                     "enum": ["data", "model",
                                              "physics", "process"]},
                        "message":   {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["severity", "category", "message", "rationale"],
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["observations"],
    },
}


class LLMCritic:
    MODEL_ID = "claude-sonnet-4-6"
    MAX_TOKENS = 1200
    TIMEOUT_S = 30.0

    def __init__(self, client: Any, model: str | None = None):
        self.client = client
        self.model = model or self.MODEL_ID

    def review_training(self, context: dict) -> list[LLMObservation]:
        """Stub — implemented in Task 3."""
        return []


def make_llm_critic() -> LLMCritic | None:
    """Return LLMCritic if ANTHROPIC_API_KEY is set, else None."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.warning("anthropic package missing — LLM-Critic disabled")
        return None
    return LLMCritic(client=Anthropic())
```

- [ ] **Step 2: Verify module imports cleanly**

Run: `PYTHONPATH=. .venv/bin/python -c "from app.backend.critic_llm import LLMCritic, LLMObservation, make_llm_critic; print('OK')"`
Expected: `OK`.

Run: `.venv/bin/ruff check app/backend/critic_llm.py`
Expected: `All checks passed!`

- [ ] **Step 3: Commit**

```bash
git add app/backend/critic_llm.py
git commit -m "feat(critic_llm): scaffold LLMCritic + factory + system prompt + tool schema"
```

---

## Task 2: _build_user_payload + first factory tests

**Files:**
- Modify: `app/backend/critic_llm.py`
- Create: `app/tests/test_critic_llm.py`

- [ ] **Step 1: Write failing tests for factory and payload builder**

Create `app/tests/test_critic_llm.py`:

```python
"""Unit tests for critic_llm (mock Anthropic client — no real API calls)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.backend.critic_llm import (
    LLMCritic, LLMObservation, make_llm_critic, _build_user_payload,
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
    # JSON block should parse
    body = payload_str.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
    parsed = json.loads(body)
    assert parsed["metrics"]["r2_test"] == 0.86
    assert parsed["dataset_size"]["n_train"] == 1666
    # Top-10 only (sorted desc by importance)
    assert len(parsed["feature_importance_top10"]) == 10
    assert list(parsed["feature_importance_top10"])[0] == "c_pct"
    # noise_2 (0.005) should be outside top-10
    assert "noise_2" not in parsed["feature_importance_top10"]
```

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_critic_llm.py -v`
Expected: `ImportError: cannot import name '_build_user_payload'`.

- [ ] **Step 2: Implement `_build_user_payload` in critic_llm.py**

Add at module level (after `_TOOL_SCHEMA`, before `class LLMCritic`):

```python
def _build_user_payload(ctx: dict) -> str:
    """Compose the user message for Claude — a JSON snapshot of training context."""
    importance = ctx.get("feature_importance") or {}
    top10 = dict(sorted(importance.items(), key=lambda kv: -kv[1])[:10])
    payload = {
        "metrics": {
            "r2_train": ctx.get("r2_train"),
            "r2_val": ctx.get("r2_val"),
            "r2_test": ctx.get("r2_test"),
            "mae_test": ctx.get("mae_test"),
            "rmse_test": ctx.get("rmse_test"),
            "coverage_90_ci": ctx.get("coverage_90_ci"),
        },
        "dataset_size": {
            "n_train": ctx.get("n_train"),
            "n_val": ctx.get("n_val"),
            "n_test": ctx.get("n_test"),
        },
        "split_strategy": ctx.get("split_strategy"),
        "cv_strategy": ctx.get("cv_strategy"),
        "feature_importance_top10": top10,
        "training_ranges": ctx.get("training_ranges") or {},
        "steel_class": ctx.get("steel_class", "pipe_hsla"),
        "target": ctx.get("target", "yield_strength_mpa"),
    }
    return (
        "Training артефакт для review:\n```json\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
        + "\n```"
    )
```

- [ ] **Step 3: Run tests**

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_critic_llm.py -v`
Expected: `3 passed`.

Run: `.venv/bin/ruff check app/backend/critic_llm.py app/tests/test_critic_llm.py`
Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add app/backend/critic_llm.py app/tests/test_critic_llm.py
git commit -m "feat(critic_llm): _build_user_payload + factory tests"
```

---

## Task 3: review_training happy path + cache_control + tool_use parsing

**Files:**
- Modify: `app/backend/critic_llm.py`
- Modify: `app/tests/test_critic_llm.py`

- [ ] **Step 1: Write failing test for valid response**

Append to `app/tests/test_critic_llm.py`:

```python
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

    # Inspect kwargs passed to messages.create
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["max_tokens"] == 1200
    assert isinstance(kwargs["system"], list)
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    # tool_use choice forces our schema
    assert kwargs["tool_choice"] == {"type": "tool", "name": "report_observations"}
```

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_critic_llm.py -v -k "happy_path or cache_control"`
Expected: both fail (review_training returns []; _log_usage missing).

- [ ] **Step 2: Implement `review_training` and `_log_usage`**

In `app/backend/critic_llm.py`:

Replace the stub `review_training` method body with:

```python
    def review_training(self, context: dict) -> list[LLMObservation]:
        """Query Claude and return observations; [] on any failure."""
        user_payload = _build_user_payload(context)
        start = time.monotonic()
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.MAX_TOKENS,
                system=[{
                    "type": "text",
                    "text": _SYSTEM_PROMPT_TEXT,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=[_TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": "report_observations"},
                messages=[{"role": "user", "content": user_payload}],
                timeout=self.TIMEOUT_S,
            )
        except Exception as e:
            logger.warning("LLM-Critic API call failed: %s", e)
            return []

        elapsed = time.monotonic() - start

        tool_block = next(
            (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            logger.warning("LLM-Critic: no tool_use block in response")
            return []

        try:
            raw_obs = tool_block.input["observations"]
            observations = [LLMObservation(**o) for o in raw_obs]
        except (KeyError, TypeError) as e:
            logger.warning("LLM-Critic: bad payload shape: %s", e)
            return []

        _log_usage(resp, elapsed, observations)
        return observations
```

Add a stub `_log_usage` at module level (full impl in Task 5):

```python
def _log_usage(resp: Any, elapsed_s: float, observations: list[LLMObservation]) -> None:
    """Persist LLM-Critic metrics to Decision Log — full impl in Task 5."""
    logger.debug(
        "LLM-Critic: %d observations, %.2fs elapsed",
        len(observations), elapsed_s,
    )
```

- [ ] **Step 3: Run tests**

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_critic_llm.py -v`
Expected: `5 passed` (3 from Task 2 + 2 new).

- [ ] **Step 4: Commit**

```bash
git add app/backend/critic_llm.py app/tests/test_critic_llm.py
git commit -m "feat(critic_llm): review_training happy path + prompt caching"
```

---

## Task 4: Error-path tests — API errors, malformed responses

**Files:**
- Modify: `app/tests/test_critic_llm.py`

- [ ] **Step 1: Write failing tests for error paths**

Append to `app/tests/test_critic_llm.py`:

```python
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
```

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_critic_llm.py -v`
Expected: `8 passed` — all three new error paths are already handled by the `try/except` blocks in Task 3's `review_training`.

- [ ] **Step 2: Commit (tests only — implementation already passes)**

```bash
git add app/tests/test_critic_llm.py
git commit -m "test(critic_llm): cover API errors, missing tool_use, bad payload shape"
```

---

## Task 5: _log_usage — Decision Log integration

**Files:**
- Modify: `app/backend/critic_llm.py`
- Modify: `app/tests/test_critic_llm.py`

- [ ] **Step 1: Write test for log_usage writing into Decision Log**

Append to `app/tests/test_critic_llm.py`:

```python
def test_log_usage_writes_decision_log(monkeypatch, tmp_path):
    """_log_usage pipes token counts and observations into log_decision."""
    from app.backend.critic_llm import _log_usage
    from decision_log import logger as dl_module

    # Redirect Decision Log to a temp db
    tmp_db = tmp_path / "test_decisions.db"
    monkeypatch.setattr(dl_module, "DEFAULT_DB_PATH", tmp_db)

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
    assert "1200" in r["reasoning"]        # input_tokens
    assert "800" in r["reasoning"]         # cache_read
    assert r["context"]["usage"]["output_tokens"] == 180
    assert r["context"]["usage"]["cache_read"] == 800
    assert r["context"]["usage"]["latency_s"] == 2.34
    assert len(r["context"]["observations"]) == 1
```

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_critic_llm.py::test_log_usage_writes_decision_log -v`
Expected: FAIL — the stub only calls `logger.debug`, writes nothing.

- [ ] **Step 2: Replace `_log_usage` stub with the real implementation**

Replace the `_log_usage` body in `app/backend/critic_llm.py`:

```python
def _log_usage(resp: Any, elapsed_s: float, observations: list[LLMObservation]) -> None:
    """Persist LLM-Critic metrics to Decision Log (input/output/cache tokens)."""
    from decision_log.logger import log_decision

    usage = resp.usage
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    cache_read = getattr(usage, "cache_read_input_tokens", 0)
    cache_create = getattr(usage, "cache_creation_input_tokens", 0)
    model = getattr(resp, "model", "unknown")

    log_decision(
        phase="training",
        decision=f"LLM-Critic review: {len(observations)} observations",
        reasoning=(
            f"Model={model}, "
            f"input={input_tokens} (cache_read={cache_read}, "
            f"cache_create={cache_create}), "
            f"output={output_tokens}, latency={elapsed_s:.2f}s"
        ),
        context={
            "observations": [asdict(o) for o in observations],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read": cache_read,
                "cache_create": cache_create,
                "latency_s": round(elapsed_s, 2),
            },
        },
        author="llm_critic",
        tags=["llm_critic", "sonnet-4-6"],
    )
```

- [ ] **Step 3: Run all critic_llm tests**

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_critic_llm.py -v`
Expected: `9 passed`.

Run: `.venv/bin/ruff check app/backend/critic_llm.py app/tests/test_critic_llm.py`
Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add app/backend/critic_llm.py app/tests/test_critic_llm.py
git commit -m "feat(critic_llm): log usage + observations to Decision Log"
```

---

## Task 6: engine.py — replace `_llm_review` stub with LLMCritic delegation

**Files:**
- Modify: `app/backend/engine.py`
- Modify: `app/tests/test_critic_llm.py`

- [ ] **Step 1: Read current engine.py to identify exact lines**

Use `grep -n "_llm_review\|exploratory_observations\|use_llm" app/backend/engine.py` to confirm line numbers before editing. Expected matches around lines 47, 73, 106-108, 125-128, 148-179.

- [ ] **Step 2: Write integration test first**

Append to `app/tests/test_critic_llm.py`:

```python
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
```

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_critic_llm.py -v -k "engine_critic"`
Expected: FAIL — engine.Critic does not yet accept `llm_critic`, and `exploratory_observations` is `list[str]` not `list[dict]`.

- [ ] **Step 3: Modify `app/backend/engine.py` — `CriticReport` field type**

Find the `CriticReport` dataclass (line ~73) and change the field type:

```python
@dataclass
class CriticReport:
    phase: str
    verdict: Verdict
    warnings: list[dict] = field(default_factory=list)
    exploratory_observations: list[dict] = field(default_factory=list)   # was list[str]
    requires_human_review: bool = False
    recommended_action: str = ""
```

- [ ] **Step 4: Modify `Critic.__init__` to accept `llm_critic`**

Find `class Critic` (line ~101) and replace its `__init__` and `review` method. Replace the whole class body up to (but not including) the legacy `_llm_review` method:

```python
class Critic:
    """
    Runs Pattern Library checks; when use_llm=True and an LLMCritic is
    configured (or ANTHROPIC_API_KEY is in env), also calls Claude Sonnet 4.6
    on the `training` phase for exploratory observations.
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_client=None,
        llm_critic=None,
    ):
        self.use_llm = use_llm
        self.llm_client = llm_client          # legacy param, retained for API compat
        self.llm_critic = llm_critic
        if self.use_llm and self.llm_critic is None:
            from app.backend.critic_llm import make_llm_critic
            self.llm_critic = make_llm_critic()

    def review(self, phase: str, context: dict) -> CriticReport:
        phase_enum = Phase(phase) if phase in {p.value for p in Phase} else None

        pattern_warnings = run_all_patterns(context, phase=phase_enum)

        high_count = sum(1 for w in pattern_warnings if w["severity"] == "HIGH")
        medium_count = sum(1 for w in pattern_warnings if w["severity"] == "MEDIUM")

        if high_count > 0:
            verdict = Verdict.BLOCK
        elif medium_count > 0:
            verdict = Verdict.PASS_WITH_WARNINGS
        else:
            verdict = Verdict.PASS

        exploratory_raw: list[dict] = []
        if self.use_llm and self.llm_critic and phase == "training":
            from dataclasses import asdict
            observations = self.llm_critic.review_training(context)
            exploratory_raw = [asdict(o) for o in observations]

        requires_human = high_count > 0 or any(
            not w.get("auto_fixable", True) for w in pattern_warnings
        )

        recommendations = [w["suggestion"] for w in pattern_warnings]
        recommended_action = "; ".join(recommendations[:3])

        return CriticReport(
            phase=phase,
            verdict=verdict,
            warnings=pattern_warnings,
            exploratory_observations=exploratory_raw,
            requires_human_review=requires_human,
            recommended_action=recommended_action,
        )
```

- [ ] **Step 5: Remove the legacy `_llm_review` method**

Find and **delete** the entire `_llm_review` method (lines ~148-179 in the current file). It should no longer be referenced anywhere.

- [ ] **Step 6: Run integration tests**

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_critic_llm.py -v`
Expected: `12 passed` (9 existing + 3 new engine integration).

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/ -q -m "not integration"`
Expected: all pre-existing tests still pass (22 cost_model + 9 critic_llm + 3 engine_critic = 34 — actually count may differ; the key is: no regressions).

- [ ] **Step 7: Run smoke_test — legacy path still works without key**

```bash
unset ANTHROPIC_API_KEY
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py
```
Expected: SMOKE TEST PASSED (LLM-Critic silently disabled because no key).

- [ ] **Step 8: Commit**

```bash
git add app/backend/engine.py app/tests/test_critic_llm.py
git commit -m "feat(engine): Critic delegates training-phase review to LLMCritic"
```

---

## Task 7: run_pipeline.py + UI integration

**Files:**
- Modify: `scripts/run_pipeline.py`
- Modify: `app/frontend/app.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Flip `use_llm` default in run_pipeline.py**

In `scripts/run_pipeline.py`, find the Orchestrator construction. Change:
```python
critic=Critic(use_llm=False),
```
to:
```python
critic=Critic(use_llm=True),
```

- [ ] **Step 2: Add LLM-Critic status in Streamlit sidebar**

In `app/frontend/app.py`, find the sidebar section (line ~42 — where Decision Log metric lives). After the Decision Log block, add:

```python
# LLM-Critic status
import os as _os_llm
_llm_ok = bool(_os_llm.environ.get("ANTHROPIC_API_KEY"))
st.sidebar.metric(
    "🤖 LLM-Critic",
    "✓ активен" if _llm_ok else "— нет ключа",
)
```

- [ ] **Step 3: Call LLMCritic + render observations in tab_train**

In `app/frontend/app.py`, find the tab_train block (around line 177). After the existing "⚠️ Отчёт Critic" rendering loop, add the LLM-Critic call and render. First, enrich `critic_ctx` to include training_ranges and n_train/val/test (needed by the LLM payload):

Find the existing `critic_ctx = {...}` in tab_train and extend it:

```python
            critic_ctx = {
                "r2_train": trained.metrics.r2_train,
                "r2_val": trained.metrics.r2_val,
                "r2_test": trained.metrics.r2_test,
                "mae_test": trained.metrics.mae_test,
                "rmse_test": trained.metrics.rmse_test,
                "coverage_90_ci": trained.metrics.coverage_90_ci,
                "n_train": trained.metrics.n_train,
                "n_val": trained.metrics.n_val,
                "n_test": trained.metrics.n_test,
                "prediction_has_ci": True,
                "has_time_column": True,
                "has_groups": True,
                "split_strategy": "time_based",
                "cv_strategy": "group_kfold",
                "feature_importance": trained.feature_importance,
                "training_ranges": trained.training_ranges,
                "steel_class": "pipe_hsla",
                "ood_detector_configured": True,
                "target": target_col,
            }
```

After the existing Pattern Library rendering loop and its `else: st.success("✓ Critic не нашёл проблем")`, append:

```python
            # LLM-Critic (Claude Sonnet 4.6) — only runs with ANTHROPIC_API_KEY
            from app.backend.critic_llm import make_llm_critic
            _llm = make_llm_critic()
            if _llm is not None:
                with st.spinner("🤖 LLM-Critic проверяет..."):
                    from dataclasses import asdict as _asdict
                    llm_obs = _llm.review_training(critic_ctx)
                    st.session_state["llm_observations"] = [
                        _asdict(o) for o in llm_obs
                    ]

            llm_obs_rendered = st.session_state.get("llm_observations", [])
            if llm_obs_rendered:
                st.subheader("🤖 LLM-Critic (Claude Sonnet 4.6)")
                for o in llm_obs_rendered:
                    sev = o["severity"]
                    msg = (f"**[{sev}] {o['category']}:** {o['message']}\n\n"
                           f"💡 {o['rationale']}")
                    if sev == "HIGH":
                        st.error(msg)
                    elif sev == "MEDIUM":
                        st.warning(msg)
                    else:
                        st.info(msg)
            elif _llm is not None:
                st.caption("🤖 LLM-Critic: проблем не обнаружено")
```

- [ ] **Step 4: Update CLAUDE.md**

Find the section «Critic и Pattern Library — главный защитный механизм» (line ~70). After its existing bullet list, add:

```markdown
- **LLM-Critic (v2)** — опциональный второй слой review на фазе `training`. Активируется через `ANTHROPIC_API_KEY` в env; при отсутствии ключа — тихий fallback на Pattern Library-only. Использует `claude-sonnet-4-6` через `app/backend/critic_llm.py`, prompt caching (`cache_control="ephemeral"`), structured output через `tool_use`. Observations информационные: попадают в `CriticReport.exploratory_observations` (`list[dict]`), отображаются в UI после Pattern Library warnings, **не влияют** на `Verdict`.
```

- [ ] **Step 5: Run smoke test without API key (regression check)**

```bash
unset ANTHROPIC_API_KEY
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py
```
Expected: SMOKE TEST PASSED (both legacy and EUR-cost paths).

- [ ] **Step 6: Run Streamlit and verify 200 OK**

```bash
lsof -ti:8501 | xargs kill 2>/dev/null
PYTHONPATH=. .venv/bin/streamlit run app/frontend/app.py --server.port 8501 --server.headless true &
STREAMLIT_PID=$!
sleep 5
curl -sI http://localhost:8501 | head -1
lsof -ti:8501 | xargs kill 2>/dev/null
wait $STREAMLIT_PID 2>/dev/null
```
Expected: `HTTP/1.1 200 OK`.

- [ ] **Step 7: Run all tests**

```bash
PYTHONPATH=. .venv/bin/pytest app/tests/ -q -m "not integration"
```
Expected: all pass, no regressions.

- [ ] **Step 8: Commit**

```bash
git add scripts/run_pipeline.py app/frontend/app.py CLAUDE.md
git commit -m "feat(ui/cli): enable LLM-Critic by default; sidebar status; training-tab observations"
```

---

## Task 8: Final verification + tag

**Files:** None (verification only).

- [ ] **Step 1: Full pytest including cost_model + critic_llm**

```bash
PYTHONPATH=. .venv/bin/pytest app/tests/ -v -m "not integration"
```
Expected: all pass. Count should be previous (22 cost_model with Patterns C01-C04) + 12 critic_llm = 34.

- [ ] **Step 2: Ruff on all new/modified code**

```bash
.venv/bin/ruff check app/backend/critic_llm.py app/tests/test_critic_llm.py
```
Expected: `All checks passed!`

- [ ] **Step 3: Smoke test without key**

```bash
unset ANTHROPIC_API_KEY
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py
```
Expected: SMOKE TEST PASSED.

- [ ] **Step 4: (Optional, requires real key) Smoke test with key**

If the user has a real `ANTHROPIC_API_KEY` available:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py
```
Expected: SMOKE TEST PASSED + `decision_log/decisions.db` contains at least one row with tag `llm_critic`. Verify with:

```bash
PYTHONPATH=. .venv/bin/python -c "from decision_log.logger import query_decisions; rs = query_decisions(tag='llm_critic', limit=5); print(len(rs), 'records'); print(rs[0]['reasoning'] if rs else 'none')"
```

If no key available, skip this step — it will be validated manually by the user.

- [ ] **Step 5: Streamlit manual smoke**

```bash
PYTHONPATH=. .venv/bin/streamlit run app/frontend/app.py --server.headless true
```

Open http://localhost:8501. Verify:
- Sidebar: `🤖 LLM-Critic: ✓ активен` if key present, `— нет ключа` otherwise.
- Tab «🤖 Обучение модели» → train a model → Pattern Library «⚠️ Отчёт Critic» appears as before.
- If key present: «🤖 LLM-Critic (Claude Sonnet 4.6)» section appears with 0-5 observations.
- If no key: no LLM-Critic section.

Stop: `lsof -ti:8501 | xargs kill`

- [ ] **Step 6: Tag the release**

```bash
git log --oneline feature/cost-optimization..HEAD | head -15
git tag -a v0.3-llm-critic -m "MVP v0.3: LLM-Critic (Claude Sonnet 4.6) on training phase"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §4.1 `LLMCritic` class | 1, 3 |
| §4.2 System prompt | 1 |
| §4.3 User payload builder | 2 |
| §4.4 Usage logging | 5 |
| §4.5 `engine.py:Critic` integration | 6 |
| §4.6 UI changes | 7 |
| §7 Tests 1-6 | 2, 3, 4, 5 (also 6 for engine tests) |
| §9 Acceptance criteria | 8 |

All spec sections covered.

**2. Placeholder scan**

No `TBD`/`TODO`/vague "add error handling" phrases. Every step has either exact code, exact command with expected output, or exact commit message.

**3. Type consistency**

- `LLMObservation` fields: `severity`, `category`, `message`, `rationale` — used identically in Tasks 1, 2, 3, 5, 6 and tests.
- `CriticReport.exploratory_observations: list[dict]` — declared in Task 6 Step 3; used in Tasks 6 (engine tests) and 7 (UI).
- `make_llm_critic() → LLMCritic | None` — returned None path tested in Task 2, used in Task 7 UI.
- `LLMCritic.review_training(context) → list[LLMObservation]` — signature stable across Tasks 1 (stub), 3 (impl), 6 (integration mock), 7 (UI call).
- `_log_usage(resp, elapsed_s, observations)` — 3-arg signature stable in Tasks 3 (stub call), 5 (impl), 5 (test).

No drift.

**4. Ambiguity**

- Task 6 Step 5 says "delete the entire `_llm_review` method". Concrete: grep shows it's between lines 148-179; the replacement in Step 4 does not call it, so removal is safe.
- Task 7 Step 3 modifies `critic_ctx` — previous contents are listed in full to avoid "merge" guessing.
- Task 8 Step 4 is optional (real API key) — explicitly flagged.

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-23-critic-v2-llm-review-plan.md`.**
