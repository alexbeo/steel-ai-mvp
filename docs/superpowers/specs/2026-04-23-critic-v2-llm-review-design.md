# Critic v2 — LLM-exploratory review на фазе training

**Дата:** 2026-04-23
**Статус:** Design (готов к implementation plan)
**Baseline:** коммит `6cd5ed1` (EUR cost-optimization закрыта)

---

## 1. Цель и мотивация

Pattern Library ловит ~60-70% типовых ошибок ML-pipeline (известные анти-паттерны: overfitting, leakage, calibration). Остаются «длинные хвосты» ошибок, которые трудно закодировать правилом: физическая неадекватность feature importance, подозрительные паттерны в training_ranges, несоответствие split-стратегии структуре данных. Senior-инженер замечает такое интуитивно.

**Цель:** заменить заглушку `Critic._llm_review()` в `engine.py:148-179` на работающую интеграцию с Claude Sonnet 4.6, который выступает как **второе мнение** на фазе `training`. LLM получает тот же контекст, что Pattern Library, и возвращает структурированные наблюдения — информационно, не блокирует pipeline.

---

## 2. Scope

### В MVP входит
- LLM-Critic вызывается **только** на фазе `training`. Фазы `data_acquisition`, `preprocessing`, `inverse_design`, `validation`, `reporting` остаются с Pattern Library-only (экономия + фокус).
- Модель: Claude Sonnet 4.6 (hard-coded `"claude-sonnet-4-6"`).
- **Всегда** вызывается при наличии ключа — независимо от того, нашла ли Pattern Library warnings (policy: дополнительное мнение ценнее всего именно когда уже что-то найдено, см. §3).
- Авто-активация через `ANTHROPIC_API_KEY` env var. Нет ключа → тихий fallback на Pattern Library-only. Нет UI-toggle в MVP.
- Observations информационные: попадают в `CriticReport.exploratory_observations`, показываются в UI, **не влияют** на `Verdict` (BLOCK/PASS) — Pattern Library остаётся единственным гарантом остановки pipeline.
- Prompt caching на system prompt (`cache_control="ephemeral"`) — снижает стоимость повторных вызовов в 10 раз.
- Structured output через Anthropic `tool_use` — гарантированно валидный JSON, без regex-парсинга.
- Decision Log аудит: input/output/cache tokens, latency, observations, tags `llm_critic`.

### Явно вне MVP
- LLM-Critic на других фазах (inverse_design, validation). Следующая итерация.
- Конфигурация модели через env/config (всегда Sonnet 4.6 в v1).
- Explicit UI-toggle «Включить LLM-Critic» (в v1 только env-var).
- Влияние LLM-observations на Verdict (потенциально v2 через consensus-логику).
- Кеширование результатов между идентичными запусками (runtime dedup).
- Streaming ответа.
- Residuals analysis в промпте (оставлено на v3, см. §5).
- Self-consistency / N-sample sampling.

---

## 3. Архитектурные решения (из brainstorm)

| Решение | Выбор | Почему |
|---|---|---|
| Scope фаз | Только `training` | High-value, быстрая окупаемость, минимум зависимостей |
| Модель | Sonnet 4.6 (hard-coded) | Sweet-spot reasoning/cost для структурированной задачи; Opus — переплата для MVP |
| Trigger | Всегда при `training` | Второе мнение ценнее при уже найденных warnings; предсказуемая цена (1 вызов = 1 тренинг) |
| Активация | Env var auto | Ноль конфигурации для демо; dev без ключа работает по-старому |
| Влияние на Verdict | Информационно | Pattern Library детерминирована; LLM ещё не проверен в prod; риск false-positive block неприемлем в MVP |
| Содержимое промпта | metrics + feature_importance + training_ranges + split/CV + dataset size | Достаточно для physics-aware reasoning; ~1.5k токенов |
| Output format | JSON через `tool_use` | Гарантированная валидность, не нужен defensive парсинг |
| Prompt caching | Ephemeral cache на system prompt | 10× дешевле повторные запросы в течение 5 минут TTL |

---

## 4. Компоненты

### 4.1 Новый модуль `app/backend/critic_llm.py`

```python
"""
LLM-Critic v2 — Claude Sonnet 4.6 as exploratory reviewer
on training phase. Informational only — does not affect Verdict.

Activated via ANTHROPIC_API_KEY env var. Fails open: any API error
returns an empty observation list so the pipeline continues as if
LLM-Critic were not configured.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Literal

logger = logging.getLogger(__name__)

Severity = Literal["HIGH", "MEDIUM", "LOW"]
Category = Literal["data", "model", "physics", "process"]


@dataclass
class LLMObservation:
    severity: Severity
    category: Category
    message: str          # Russian, surfaced to UI
    rationale: str        # short "why", also Russian


class LLMCritic:
    MODEL_ID = "claude-sonnet-4-6"
    MAX_TOKENS = 1200
    TIMEOUT_S = 30.0
    SYSTEM_PROMPT = _SYSTEM_PROMPT_TEXT    # see §4.2

    TOOL_SCHEMA = {
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
                                          "enum": ["data", "model", "physics", "process"]},
                            "message":  {"type": "string"},
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

    def __init__(self, client, model: str | None = None):
        self.client = client
        self.model = model or self.MODEL_ID

    def review_training(self, context: dict) -> list[LLMObservation]:
        """Returns observations from Claude or [] on any failure."""
        user_payload = _build_user_payload(context)
        start = time.monotonic()
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.MAX_TOKENS,
                system=[{
                    "type": "text",
                    "text": self.SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=[self.TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": "report_observations"},
                messages=[{"role": "user", "content": user_payload}],
                timeout=self.TIMEOUT_S,
            )
        except Exception as e:
            logger.warning("LLM-Critic API call failed: %s", e)
            return []

        elapsed = time.monotonic() - start
        tool_block = next(
            (b for b in resp.content if b.type == "tool_use"), None
        )
        if tool_block is None:
            logger.warning("LLM-Critic: no tool_use in response")
            return []

        try:
            raw = tool_block.input["observations"]
            observations = [LLMObservation(**o) for o in raw]
        except (KeyError, TypeError) as e:
            logger.warning("LLM-Critic: bad payload shape: %s", e)
            return []

        _log_usage(resp, elapsed, observations)
        return observations


def make_llm_critic() -> LLMCritic | None:
    """Factory: returns LLMCritic if ANTHROPIC_API_KEY is set, else None."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.warning("anthropic package missing — LLM-Critic disabled")
        return None
    return LLMCritic(client=Anthropic())
```

### 4.2 System prompt (cached, ~800 токенов)

Полный текст в `critic_llm.py` как константа `_SYSTEM_PROMPT_TEXT`. Содержимое:

```
Ты senior ML-инженер с 10-летним опытом в металлургии HSLA-сталей
(trubопроводные, K60-K65). Тебе на review поступает training-артефакт
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
   - Если `split_strategy != "time_based"` на данных с временной
     колонкой — high risk leakage.
   - Если `cv_strategy != "group_kfold"` на данных с groups —
     оптимистичный CV-score.

4. TRAINING_RANGES PHYSICAL SANITY
   - Диапазоны должны быть в типичных для pipe-HSLA пределах:
     C 0.03-0.15 %; Mn 0.8-1.8; Nb 0-0.06; Ti 0-0.03; Si 0.1-0.6;
     rolling_finish_temp 740-860 °C; cooling_rate 5-30 °C/s.
   - Выход за эти пределы → либо другой класс стали, либо ошибка
     генерации данных.

ФОРМАТ ОТВЕТА — через tool `report_observations`:
- До 5 observations (выбирай самые важные).
- severity: HIGH (стоп-сигнал для senior'а), MEDIUM (нужно
  выяснить), LOW (к сведению).
- category: data | model | physics | process.
- message и rationale на русском.
- Если всё чисто — верни пустой список. Не придумывай проблемы.
```

### 4.3 User payload (dynamic, ~400-600 токенов)

```python
def _build_user_payload(ctx: dict) -> str:
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
        "feature_importance_top10": dict(sorted(
            (ctx.get("feature_importance") or {}).items(),
            key=lambda kv: -kv[1],
        )[:10]),
        "training_ranges": ctx.get("training_ranges") or {},
        "steel_class": ctx.get("steel_class", "pipe_hsla"),
        "target": ctx.get("target", "yield_strength_mpa"),
    }
    return "Training артефакт для review:\n```json\n" + \
           json.dumps(payload, indent=2, ensure_ascii=False) + "\n```"
```

### 4.4 Usage logging

```python
def _log_usage(resp, elapsed_s: float, observations: list[LLMObservation]):
    usage = resp.usage
    from decision_log.logger import log_decision
    log_decision(
        phase="training",
        decision=f"LLM-Critic review: {len(observations)} observations",
        reasoning=(
            f"Model={resp.model}, "
            f"input={usage.input_tokens} (cache_read={getattr(usage, 'cache_read_input_tokens', 0)}, "
            f"cache_create={getattr(usage, 'cache_creation_input_tokens', 0)}), "
            f"output={usage.output_tokens}, "
            f"latency={elapsed_s:.2f}s"
        ),
        context={
            "observations": [asdict(o) for o in observations],
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read": getattr(usage, "cache_read_input_tokens", 0),
                "cache_create": getattr(usage, "cache_creation_input_tokens", 0),
                "latency_s": round(elapsed_s, 2),
            },
        },
        author="llm_critic",
        tags=["llm_critic", "sonnet-4-6"],
    )
```

### 4.5 Интеграция в `engine.py:Critic`

Изменяется `__init__`:
```python
def __init__(self, use_llm: bool = False, llm_client=None, llm_critic=None):
    self.use_llm = use_llm
    self.llm_client = llm_client      # legacy param, unused
    self.llm_critic = llm_critic      # NEW; LLMCritic instance or None
    if self.use_llm and self.llm_critic is None:
        from app.backend.critic_llm import make_llm_critic
        self.llm_critic = make_llm_critic()   # reads env
```

Заменяется блок `_llm_review` (строки 148-179). Новая логика в `review`:
```python
exploratory_raw: list[dict] = []
if self.use_llm and self.llm_critic and phase == "training":
    observations = self.llm_critic.review_training(context)
    exploratory_raw = [asdict(o) for o in observations]
```

`_llm_review` как метод удаляется (вся логика в `LLMCritic.review_training`). `exploratory_observations` поле `CriticReport` меняется с `list[str]` на `list[dict]` — содержит сериализованные `LLMObservation`.

`run_pipeline.py` и UI-код меняют дефолт: `Critic(use_llm=True)` вместо `False`. Если ключа нет, `make_llm_critic()` возвращает None → полный fallback.

### 4.6 UI изменения (`app/frontend/app.py`)

**Sidebar**, после Decision Log metric:
```python
import os
llm_status = "✓ активен" if os.environ.get("ANTHROPIC_API_KEY") else "— нет ключа"
st.sidebar.metric("🤖 LLM-Critic", llm_status)
```

**Tab «🤖 Обучение модели»**. Сейчас после `train_model()` вызывается `run_all_patterns(ctx, phase=Phase.TRAINING)` напрямую. Не ломаем это — добавляем **параллельный** вызов LLMCritic и сохраняем observations в session_state. Код после существующего блока «⚠️ Отчёт Critic»:

```python
# LLM-Critic — работает только при наличии ANTHROPIC_API_KEY
from app.backend.critic_llm import make_llm_critic
_llm = make_llm_critic()
if _llm is not None:
    with st.spinner("🤖 LLM-Critic review..."):
        llm_obs = _llm.review_training(critic_ctx)
    st.session_state["llm_observations"] = [asdict(o) for o in llm_obs]

# Render (session_state persists across Streamlit reruns)
llm_obs = st.session_state.get("llm_observations", [])
if llm_obs:
    st.subheader("🤖 LLM-Critic (Claude Sonnet 4.6)")
    for o in llm_obs:
        sev = o["severity"]
        msg = (f"**[{sev}] {o['category']}:** {o['message']}\n\n"
               f"💡 {o['rationale']}")
        if sev == "HIGH":
            st.error(msg)
        elif sev == "MEDIUM":
            st.warning(msg)
        else:
            st.info(msg)
```

`critic_ctx` — уже существующая переменная в tab_train, содержит metrics/feature_importance/split_strategy и т.д. Добавим недостающие ключи (`training_ranges`, `n_train/val/test`) при её построении.

---

## 5. Out of scope / следующие итерации

- **v2: LLM-Critic на `inverse_design`** — смотрит на топ-кандидатов, OOD-флаги, Pareto front.
- **v3: Residuals в промпт** — mean/std/skewness ошибок на test, топ-5 outliers.
- **v4: Влияние на Verdict** — consensus-логика (BLOCK только при согласии Pattern Library + LLM).
- **v5: Multi-model ensemble** — Sonnet + Opus параллельно, консолидация observations.
- **v6: Feature-engineering assistant** — Claude предлагает новые derived features на основе физики.

---

## 6. Файлы, изменяемые в реализации

### Новые
- `app/backend/critic_llm.py` — LLMCritic класс, фабрика, system prompt, tool schema.
- `app/tests/test_critic_llm.py` — 4 теста (см. §7).
- `docs/superpowers/specs/2026-04-23-critic-v2-llm-review-design.md` — этот документ.

### Изменяемые
- `app/backend/engine.py` — `Critic.__init__` принимает `llm_critic`, удаляется `_llm_review`, `review()` вызывает `LLMCritic.review_training`, `CriticReport.exploratory_observations` — `list[dict]`.
- `scripts/run_pipeline.py` — `Critic(use_llm=True)`.
- `app/frontend/app.py` — sidebar status + observations block в tab_train.
- `CLAUDE.md` — строка про LLM-Critic в секции «Critic и Pattern Library» + env var.

---

## 7. Тесты (`app/tests/test_critic_llm.py`)

1. **`test_observation_schema_parses_valid_response`** — mock Anthropic client возвращает `tool_use` блок с валидным JSON (2 observations) → получаем `[LLMObservation, LLMObservation]` с правильными severity/category.
2. **`test_api_error_returns_empty_list`** — mock кидает `anthropic.APIConnectionError` → `review_training(ctx) == []`, warning в логи.
3. **`test_bad_payload_shape_returns_empty_list`** — mock возвращает tool_use без поля `observations` → `[]`.
4. **`test_system_prompt_has_cache_control`** — проверка, что `messages.create` вызывается с `system[0].cache_control == {"type": "ephemeral"}` (инспектируем kwargs через mock).
5. **`test_factory_returns_none_without_api_key`** — `ANTHROPIC_API_KEY` снят через `monkeypatch.delenv` → `make_llm_critic() is None`.
6. **`test_factory_builds_client_with_key`** — `ANTHROPIC_API_KEY=dummy` → `make_llm_critic()` возвращает `LLMCritic` instance.

Все тесты используют `unittest.mock.MagicMock` для anthropic client — никаких реальных API-запросов.

---

## 8. Стоимость и производительность

**Sonnet 4.6 pricing (на 2026-04-23):**
- input: $3 / M tokens
- cache read: $0.30 / M tokens (10× дешевле)
- cache create: $3.75 / M tokens
- output: $15 / M tokens

**Типичный запрос:**
- System prompt: ~800 токенов. Первый вызов — создаёт cache (~$0.003). Последующие в течение 5 мин — cache read (~$0.0003).
- User payload: ~500 токенов → $0.0015.
- Output: ~300 токенов → $0.0045.

**Итого первый вызов:** ~$0.009 (€0.0082). **Последующие (cached):** ~$0.006 (€0.0055). В демо-сценарии с 5-10 запусками обучения подряд — €0.04-0.06 всего.

**Latency:** 2-4 секунды на запрос (Sonnet 4.6 + 1200 max_tokens). Не блокирует UI — вкладка «Обучение» уже использует `st.spinner` во время train_model().

---

## 9. Acceptance criteria

1. `pytest app/tests/test_critic_llm.py -v` — 6 тестов проходят (все с mock, без реальных API-запросов).
2. `PYTHONPATH=. .venv/bin/pytest app/tests/ -q` — все предыдущие тесты продолжают проходить (23 из cost_model + 6 новых = 29).
3. `PYTHONPATH=. .venv/bin/python scripts/smoke_test.py` без ANTHROPIC_API_KEY — проходит как раньше (тихий fallback).
4. `PYTHONPATH=. ANTHROPIC_API_KEY=<real> .venv/bin/python scripts/smoke_test.py` — smoke test проходит + в Decision Log появляется запись с tag `llm_critic`.
5. Streamlit UI без ключа — sidebar `🤖 LLM-Critic: — нет ключа`, секция observations в tab_train не появляется.
6. Streamlit UI с ключом — sidebar `🤖 LLM-Critic: ✓ активен`, после обучения появляется блок с observations (0-5 штук).
7. Pipeline никогда не падает из-за LLM-ошибки: disconnect/timeout/invalid JSON → логи `warning`, observations=[], UI просто не показывает секцию.
8. Decision Log запись содержит `cache_read_input_tokens > 0` на втором и последующих запусках в течение 5 мин (доказательство prompt caching).

---

## 10. Риски и допущения

- **Галлюцинации LLM**: observations могут быть «убедительными, но неверными». Mitigation: severity «информационно» (не блокирует pipeline), `Verdict` по-прежнему Pattern Library-only. Пользователь видит «это второе мнение» и сам решает.
- **API недоступность** (rate limit, outage): `fail-open` — observations=[], pipeline продолжается. Detection по отсутствию записи в Decision Log.
- **Prompt caching TTL 5 минут**: если между тренингами >5 мин, каждый раз пересоздаётся кэш. Не критично для MVP (train-цикл обычно <1 мин).
- **Версия модели**: Anthropic может депрекировать `claude-sonnet-4-6`. Mitigation: при 404 от API → `logger.error` + observations=[]. Обновить `MODEL_ID` в patch.
- **Контекст window** (200k у Sonnet 4.6): наш промпт ~1.5k — 0.75% от лимита. Зазор огромный.
- **Стоимость на больших тренингах**: 100 тренингов = ~€0.8. Для MVP ничтожно; для продакшена включить runtime dedup (v4).
- **Русский язык в output**: Anthropic SDK возвращает то, что модель сгенерировала. Sonnet 4.6 отлично владеет русским. Если output вдруг на английском — `message`/`rationale` отображаются как есть (не критично для UI).
