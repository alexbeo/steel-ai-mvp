# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Команды запуска

Все команды требуют `PYTHONPATH=.` (проект не упакован как пакет, импорты идут от корня).

```bash
# Установка
pip install -r requirements.txt

# End-to-end smoke test (~1-2 мин, обязательная проверка после изменений)
PYTHONPATH=. python scripts/smoke_test.py

# Streamlit UI
PYTHONPATH=. streamlit run app/frontend/app.py           # localhost:8501

# CLI-pipeline через Orchestrator + Critic (с human-in-the-loop)
PYTHONPATH=. python scripts/run_pipeline.py --full
PYTHONPATH=. python scripts/run_pipeline.py --step data      # только data
PYTHONPATH=. python scripts/run_pipeline.py --step train     # только train
PYTHONPATH=. python scripts/run_pipeline.py --step design --target-min 485 --target-max 580

# Docker (одна команда — поднимает Streamlit с volumes для data/models/reports/decision_log)
docker-compose up --build

# Одиночный модуль как demo (большинство backend-файлов имеют __main__ dry-run)
PYTHONPATH=. python app/backend/engine.py
PYTHONPATH=. python pattern_library/patterns.py
PYTHONPATH=. python decision_log/logger.py

# Тесты / линт (pytest и ruff в requirements.txt, каталог app/tests/ сейчас пуст)
pytest app/tests -v
ruff check .
```

## Архитектура — ключевое для продуктивности

Это **упрощённый** вариант системы из плана продукта. Вместо формальных "skills" с LLM-прослойкой — детерминированный Python-код в `app/backend/`, который напрямую вызывается агентами.

### Важное расхождение со структурой

README описывает идеальную структуру с агентами-LLM и skills-модулями. Реальный MVP устроен иначе — **ориентируйтесь на код, а не на README**:

- `agents/` — **только** `SYSTEM_PROMPT.md` для Orchestrator и Critic (документация, не код). Большинство подпапок пустые.
- `app/backend/*.py` — **реальные executive-агенты** (классы `*Agent`), каждый оборачивает детерминированный Python над scikit-learn/XGBoost/pymoo. Никакие LLM-вызовы в production-пути не делаются.
- `skills/*/scripts/` — **все пусты**. Вся логика скиллов живёт в `app/backend/`.
- `pattern_library/{data,model,production}_issues/` — **пустые каталоги**. Все проверки находятся в одном файле `pattern_library/patterns.py` (список `PATTERNS` + `run_all_patterns()`).
- Critic использует LLM **опционально** — по умолчанию только детерминированные проверки из Pattern Library. LLM вызывается, только если `Critic(use_llm=True, llm_client=...)` и Pattern Library ничего не нашла.

### Поток выполнения pipeline

`app/backend/engine.py` — ядро. Orchestrator проходит фазы в строгом порядке и после **каждой** фазы вызывает Critic:

```
data_acquisition → preprocessing → feature_engineering →
training → inverse_design → validation → reporting
```

Для каждой фазы Orchestrator:
1. Собирает `task` через `_build_task_for_phase()`.
2. Вызывает `agent.run(state, task)` — возвращает `AgentResult`.
3. Мержит output в `PipelineState` через `_merge_result_into_state()`.
4. Собирает контекст для Critic через `_build_critic_context()` и зовёт `Critic.review(phase, ctx)`.
5. Если вердикт `BLOCK` — бросает `HumanInTheLoopRequired` наверх. CLI/UI передаёт ответ пользователя через callback `on_human_checkpoint` и перезапускает фазу с `skip_checkpoint=True`.

`PipelineState` передаётся по ссылке между фазами — **один источник истины** для dataset/features/model/candidates.

### Critic и Pattern Library — главный защитный механизм

- Паттерны закодированы в `pattern_library/patterns.py`: каждый — `Pattern(id, phase, severity, check: (ctx) -> CheckResult, suggestion)`.
- ID-префиксы: `D*` (data), `M*` (model), `I*` (inverse design), `V*` (validation), `P*` (production).
- `Severity.HIGH` → `Verdict.BLOCK` → human checkpoint. `MEDIUM` → `PASS_WITH_WARNINGS`. Отсутствие триггеров → `PASS`.
- Новые проверки добавляются **только** правкой `patterns.py` и дополнением списка `PATTERNS`. README `pattern_library/README.md` — это прозаическое описание ~20 паттернов для MVP, используйте его как справочник при расширении (D01–V01 там описаны подробно, не все из них реализованы в коде).
- `_build_critic_context()` в `engine.py` — **единственное** место, где решается, какие ключи попадут в `ctx` для проверок. Если добавляете новый паттерн, зависящий от новых данных, расширяйте этот метод.
- **LLM-Critic (v2)** — опциональный второй слой review на фазе `training`. Активируется через `ANTHROPIC_API_KEY` в env; при отсутствии ключа — тихий fallback на Pattern Library-only. Использует `claude-sonnet-4-6` через `app/backend/critic_llm.py`, prompt caching (`cache_control="ephemeral"`), structured output через `tool_use`. Observations информационные: попадают в `CriticReport.exploratory_observations` (`list[dict]`), отображаются в UI после Pattern Library warnings, **не влияют** на `Verdict`.

### Decision Log — долгая память проекта

`decision_log/logger.py` — SQLite в `decision_log/decisions.db` (gitignored). Любое решение с последствиями (выбор модели, strategy split, отказ от фичи) должно попадать в лог через `log_decision(phase, decision, reasoning, alternatives_considered=..., context=..., author=...)`. Это компенсирует отсутствие persistent memory у LLM-сессий — при новом запуске Orchestrator в начале читает `summarize_project_history()`.

### ML-конвенции, заложенные в коде

Эти решения уже приняты и проверяются Critic — не ломайте их без явного обновления Pattern Library:

- **Split strategy = time-based**, CV = **GroupKFold** (M07, D06). Синтетический датасет содержит `campaign_id` и год — смотрите `data_curator.py`.
- **Uncertainty обязательна** (M04): quantile regression q05/q95 поверх XGBoost. `TrainedModel.has_uncertainty = True`.
- **Calibration target: 85–95% coverage** для 90% CI (M02). При нарушении — conformal prediction.
- **OOD detector обязателен** перед inverse design (M06): `GaussianMixture` по training composition.
- **Inverse design bounds** не выходят за training range более чем на ±10% (I01). Запрашивайте `training_ranges` из модели.
- **Multi-objective — нормализуем** (I02). Pareto size < 5 считается проблемой (I03).
- **Физические границы** на composition проверяются в `patterns.py` `_check_d07_physical_bounds` (жёсткие HSLA-диапазоны).
- **Cost objective использует ferroalloy pricing** — `app/backend/cost_model.py` с `PriceSnapshot(date, currency, materials)`. Legacy `ELEMENT_PRICES_EUR_PER_KG` остаётся только как fallback при `price_snapshot=None`. Seed-прайс — `data/prices/seed_2026-04-23.yaml` (EUR, 11 позиций, покрывает весь `PIPE_HSLA_FEATURE_SET`). Каждый run с прайсом сохраняет snapshot в `decision_log/price_snapshots/<ts>.yaml` (gitignored) + запись в Decision Log с tag `cost_optimization`. Pattern Library проверяет C01–C04.

### Target feature set

`PIPE_HSLA_FEATURE_SET` в `app/backend/feature_eng.py` — канонический список фичей для pipe-HSLA (химия + CEV/Pcm/CEN и ratios + процессные параметры). При смене класса стали или клиента его надо обновлять вместе с physical_bounds в `data_curator.py` и ожидаемыми top-features в `_check_m05_feature_importance_sanity`.

### UI и API

- `app/frontend/app.py` — Streamlit с 4 вкладками (Дизайн, Обучение, Прогноз, История). Сам импортирует функции напрямую из `app/backend/*` — не через Orchestrator. Для быстрого UX обучение/дизайн запускаются синхронно с прогресс-баром.
- `app/backend/` содержит намёки на FastAPI, но отдельного `api.py` сейчас нет — FastAPI-слой не реализован.
- Streamlit опирается на наличие обученных моделей в `models/<version>/`. Если моделей нет — сначала вкладка «Обучение модели».

### Данные — синтетика, а не NIMS

`data_curator.generate_synthetic_hsla_dataset()` создаёт физически правдоподобный датасет для демо. Реальная загрузка NIMS MatNavi **не реализована** — замена синтетики на загрузчик реальных данных указана в QUICKSTART.md как work item. Датасет сохраняется в `data/hsla_synthetic.parquet` (gitignored).

## Языковая конвенция

Код, комментарии, docstrings, логи и UI написаны на смеси русского и английского — русский для объяснений/бизнес-терминов, английский для технических идентификаторов. Сохраняйте этот стиль. Пользовательские сообщения (UI, checkpoint-вопросы, HTML-отчёты) — **русский**. Сообщения ошибок уровня парсера/валидации данных (не поднимающиеся в UI) — **английский**, потому что их читает разработчик или data-author. Пример водораздела: `PriceSnapshotIncomplete("Нет цен для: Nb")` (surfaced в UI) vs `ValueError("FeMn-80: element_content sum = 0.9, must be ≈ 1.0")` (parser-level).

## Gitignored артефакты

`data/*.parquet`, `models/*/`, `reports/*.html`, `decision_log/*.db`. После `smoke_test.py` эти файлы появятся, но коммитить их не нужно.
