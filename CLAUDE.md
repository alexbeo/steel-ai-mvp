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

# Тесты / линт (pytest и ruff в requirements.txt; в .venv 132+ тестов)
pytest app/tests -v
ruff check .

# AI-capabilities runners (требуют ANTHROPIC_API_KEY в .env)
PYTHONPATH=. .venv/bin/python scripts/generate_hypotheses_for_model.py    # A2
PYTHONPATH=. .venv/bin/python scripts/discover_features_for_model.py      # A1
PYTHONPATH=. .venv/bin/python scripts/symbolic_regression_for_model.py    # B1
PYTHONPATH=. .venv/bin/python scripts/show_property_cost_on_agrawal.py    # EC.1
PYTHONPATH=. .venv/bin/python scripts/design_recipe_with_critic.py        # recipe pair
PYTHONPATH=. .venv/bin/python scripts/propose_next_experiments.py         # B2
PYTHONPATH=. .venv/bin/python scripts/explain_ood_record.py               # A3
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

### Target feature set / multi-class

Классы стали описаны YAML-профилями в `data/steel_classes/<id>.yaml` (`id, name, standard, feature_set, physical_bounds, target_properties, expected_top_features, synthetic_generator_name, process_params`). Реестр — `app/backend/steel_classes.py:AVAILABLE_CLASS_IDS`; сейчас: `pipe_hsla` (API 5L) и `en10083_qt` (EN 10083-2 Q&T carbon steels). **Номенклатура европейская** — советских марок нет.

Класс — атрибут каждой обученной модели: `TrainedModel.steel_class` сохраняется в `models/<version>/meta.json`. Downstream-UI (prediction, design) читает активный класс из meta и ведёт себя соответственно: поля ввода в «Прогноз» подстраиваются под `feature_set`, target-label читается из профиля, вкладка «Дизайн» показывает banner для Q&T (inverse design остаётся HSLA-only в этой итерации).

Physical bounds и expected top-features для Critic проверок `D07` / `M05` читаются из профиля через `_build_critic_context` (fallback на HSLA-константы для старых моделей без `meta["steel_class"]`). Synthetic-генераторы живут в `data_curator.py` (`generate_synthetic_hsla_dataset`, `generate_synthetic_en10083_qt_dataset`) и регистрируются в `steel_classes.get_synthetic_generator(name)`.

### Al-deoxidation advisory (on-line LF)

`app/backend/deoxidation.py` — physics-only калькулятор раскисления жидкой стали алюминием на фазе ladle furnace. Три термодинамические модели в registry (`THERMO_MODELS`): Fruehan 1985 (дефолт), Sigworth-Elliott 1974, Hayashi-Yamamoto 2013. Две функции: `compute_al_demand` (forward — сколько Al подать) и `compute_al_quality` (inverse — эффективная чистота Al по факту плавки) + `compare_all_models` для сравнения 3 формул.

UI — вкладка «🔥 Раскисление» с 3 sub-tabs (Forward / Inverse / Compare). Target O_a читается из активного `SteelClassProfile.target_o_activity_ppm` (HSLA=5, Q&T=15). Pattern Library имеет фазу `Phase.DEOXIDATION` + паттерны `DX01`/`DX02`/`DX03`. Decision Log — **опт-ин** (кнопка «Сохранить») во избежание спама БД на производственном темпе 50-200 плавок/день.

**Не входит в MVP**: кинетика растворения Al, баланс FeO в шлаке, комбинированное раскисление (Al+FeSi+Ca), ML, feedback loop, интеграция с анализаторами O. Это фазы v0.6+.

### UI и API

- `app/frontend/app.py` — Streamlit с **8 вкладками** (Дизайн, Обучение, Прогноз, Раскисление, Гипотезы, Подбор рецепта, Следующие эксперименты, История). Сам импортирует функции напрямую из `app/backend/*` — не через Orchestrator. Для быстрого UX обучение/дизайн запускаются синхронно с прогресс-баром.
- `app/backend/` содержит намёки на FastAPI, но отдельного `api.py` сейчас нет — FastAPI-слой не реализован.
- Streamlit опирается на наличие обученных моделей в `models/<version>/`. Если моделей нет — сначала вкладка «Обучение модели».
- AI-вкладки требуют `ANTHROPIC_API_KEY` в `.env` (gitignored). Без ключа отображают warning и degrade gracefully.

### AI integration roadmap — все 7 capabilities закрыты (2026-04-25)

Project pivoted from sales-tool framing to MVP focused on AI-driven pattern discovery (см. `docs/discussions/2026-04-25_project_purpose_reframe.md`). Целевые пользователи — R&D engineer + Materials scientist (academic). Все 7 AI capabilities верифицированы живьём на real Agrawal NIMS data:

| ID | Capability | Module | Verified |
|---|---|---|---|
| A2 | Hypothesis generator + PhD critic | `app/backend/hypothesis_generator.py` + `hypothesis_critic.py` | 5 hypotheses, 3 HIGH novelty |
| A1 | LLM feature discovery + retrain truth gate | `feature_discoverer.py` | 5/5 формул применены, null uplift на saturated baseline (architecture works) |
| B1 | Symbolic regression (gplearn) | `symbolic_regressor.py` | Pareto frontier R²=0.825 при complexity=70 |
| EC.1 | NSGA-II property+cost showcase | `scripts/show_property_cost_on_agrawal.py` | −€28/т при +181 МПа |
| Recipe pair | Sonnet PhD designer + critic с evidence-base | `recipe_designer.py` + `recipe_critic.py` | 1 ACCEPT, 2 REVISE, 1 REJECT — найдены mechanism inversion + cost error на порядок |
| B2 | Active learning (cost-weighted EI) | `active_learner.py` | Top-5 ranked в 150 мс, no LLM call |
| A3 | Anomaly explainer | `anomaly_explainer.py` | PhD diagnosis с named formulas (Calver, Andrews Ms, Pickering) |

**Промпты — gitignored intellectual property** (`prompts/*.md`, см. `prompts/README.md`). Каждый LLM-модуль грузит свой prompt через `app.backend.prompt_loader.load_prompt(name)`. На свежем клоне без prompts/ модули raise `PromptNotFoundError`.

**Strategic discussions** живут в `docs/discussions/` (15+ файлов от 2026-04-25). Это persistent context для будущих сессий — project purpose, target users, AI roadmap, verification reports каждой capability. См. `docs/discussions/README.md` как индекс.

### Данные — синтетика для HSLA/Q&T, реальные NIMS для fatigue

`data_curator.generate_synthetic_hsla_dataset()` создаёт физически правдоподобный синтетический датасет для HSLA-демо. Q&T (`generate_synthetic_en10083_qt_dataset`) — тоже синтетика.

**Класс `fatigue_carbon_steel`** обучается на **реальных 437 records из Agrawal NIMS 2014** (DOI 10.1186/2193-9772-3-8). Loader — `load_real_agrawal_fatigue_dataset()`. Это первая (и сейчас единственная) production-модель на real-world data; все AI-capabilities верифицированы именно на ней.

## Языковая конвенция

Код, комментарии, docstrings, логи и UI написаны на смеси русского и английского — русский для объяснений/бизнес-терминов, английский для технических идентификаторов. Сохраняйте этот стиль. Пользовательские сообщения (UI, checkpoint-вопросы, HTML-отчёты) — **русский**. Сообщения ошибок уровня парсера/валидации данных (не поднимающиеся в UI) — **английский**, потому что их читает разработчик или data-author. Пример водораздела: `PriceSnapshotIncomplete("Нет цен для: Nb")` (surfaced в UI) vs `ValueError("FeMn-80: element_content sum = 0.9, must be ≈ 1.0")` (parser-level).

## Gitignored артефакты

`data/*.parquet`, `models/*/`, `reports/*.html`, `decision_log/*.db`. После `smoke_test.py` эти файлы появятся, но коммитить их не нужно.
