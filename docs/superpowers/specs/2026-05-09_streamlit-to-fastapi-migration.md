# Design: Миграция UI Steel AI MVP со Streamlit на FastAPI + статический JS-frontend

> **Регламент:** R-001 Feature Development, Step 2 (Architect).
> **Автор:** Architect (sub-agent).
> **Дата:** 2026-05-09.
> **Статус:** черновик для Step 3 (Orchestrator surfaces open questions to user).

## Context

Текущий UI — Streamlit (`app/frontend/app.py`, 2577 строк, 8 вкладок), который **напрямую импортирует** функции из `app/backend/*.py` и рендерит результаты через `st.*` примитивы. Это удобно для прототипа, но мешает: (а) интеграции с внешними системами (REST), (б) кастомной visual identity (Streamlit «AI-default» эстетика), (в) deploy в нестандартные среды.

Пользователь предоставил визуальный дизайн `Steel-AI-MVP/index.html` (2135 строк HTML/CSS, **0 JS**, статический mockup): dark dashboard, oklch palette, Inter + JetBrains Mono, nav-rail 224px слева, topbar 52px, KPI strip × 5, 3-col grid (Targets / Pareto+candidates / PhD critic), lower-band (Deox + AI advisor), Decision Log strip, thermal footer. Этот дизайн становится новым "ground truth" для фронта.

Цель миграции — **поэтапно** заменить Streamlit на пару `FastAPI (тонкие обёртки) + static JS (визуал из index.html)`, **не трогая** backend бизнес-логику, тесты, Pattern Library, Decision Log, обученные модели.

### Текущая фактическая архитектура (verified чтением кода)

- `app/frontend/app.py` импортирует `train_model`, `run_inverse_design`, `compute_al_demand`, `make_recipe_designer` etc. **синхронно**. Long-running вызовы (Optuna 1-5 мин, NSGA-II ~40 с, Sonnet 30-180 с) блокируют сессию, Streamlit показывает spinner.
- **Engine.Orchestrator** (`engine.py`) с HITL-checkpoints используется **только** в `scripts/run_pipeline.py` CLI, **не в Streamlit**. Streamlit показывает Critic warnings post-hoc, не блокирует. Это важная находка: **миграция не требует state-machine для HITL** в первой итерации.
- `app/backend/` уже содержит `fastapi>=0.104, uvicorn>=0.24` в requirements.txt, но **`app/backend/api.py` не существует** — FastAPI слой нужно создать с нуля.
- Streamlit использует Altair для 2 чартов (Pareto scatter, comparison bar) + `st.bar_chart` для feature importance. Hand-crafted SVG из index.html — это статичные mockups; реальные данные приходят из NSGA-II / training_meta.

## Constraints

### Жёсткие
- **Backend бизнес-логика неприкосновенна**: `engine.py`, `model_trainer.py`, `inverse_designer.py`, `hypothesis_generator.py`, `recipe_designer.py`, `deoxidation.py`, `active_learner.py`, `anomaly_explainer.py`, `feature_discoverer.py`, `symbolic_regressor.py`, `cost_model.py`, Pattern Library, Decision Log, тесты в `app/tests/` — не меняем (R-006/R-008-style discipline).
- **Streamlit живёт параллельно** до конца миграции. Каждая вкладка = отдельный PR. После каждого этапа `pytest app/tests -v` (132+ тестов) и `scripts/smoke_test.py` должны проходить.
- **`.env`/API-keys/prompt-loader** — не трогаем. FastAPI читает те же `ANTHROPIC_API_KEY`/`prompts/` через существующие `make_*` фабрики.
- **Multi-class профили** — UI должен подстраиваться под `feature_set` активной модели через `meta.json` (как Streamlit сейчас).
- **`PYTHONPATH=.` convention** — стартовая команда `uvicorn` тоже идёт от корня, импорты вида `from app.backend.* import …`.
- **Декларативные YAML-профили** (`data/steel_classes/*.yaml`) — single source of truth для targets/feature_set/physical_bounds, frontend получает их через endpoint.
- **Solo dev, MVP темп** — простота >> архитектурная мощность. Никаких build-pipeline монстров, никаких SPA-роутеров на 4 уровня.

### Мягкие
- Cost: предпочесть инструменты, которые **не требуют node_modules в репо** (или минимально — только для dev). Production-сервер должен запускаться `uvicorn app.api.main:app` без `npm install`.
- Сложность: **не вводить framework если 80% задачи решает vanilla JS + DOM API**.
- Performance: long-running (training, NSGA-II, LLM) — UX не должен зависать; страница остаётся responsive.
- Reproducibility: каждый run сохраняется в Decision Log (как сейчас в Streamlit) — frontend это вызывает, не bypass'ит.

### Зависимости от регламентов
- **R-001 Feature Development** — это сам триггер.
- **R-005 Pattern Library** — никаких изменений; frontend читает warnings из существующего `run_all_patterns()`.
- **R-006 Domain Compliance** — class-aware UI требует чтения `meta.json["steel_class"]` + YAML профилей через endpoint.
- **R-008 Price Snapshot** — endpoint для price snapshot upload/edit/save должен соблюдать существующий PriceSnapshot YAML-формат.

## Options considered

Решений несколько — рассматриваю по группам.

---

### Group 1 — JS framework для frontend

**Option A: Vanilla JS + ES modules (recommended)**
- **Подход:** один `app.js` плюс несколько модулей (`api.js`, `router.js`, `views/*.js`, `components/*.js`). Никакого build-pipeline. Загружается напрямую через `<script type="module">`. State через `URLSearchParams` + локальные closures. DOM-rendering через `<template>` элементы и `cloneNode`/`querySelector`.
- **Pros:**
  - 0 зависимостей, 0 build, 0 node_modules в production.
  - Дизайн `index.html` уже vanilla — adoption нулевой.
  - Отладка в браузерных DevTools без sourcemaps.
  - Решает 100% задач: 8 экранов, формы, чарты, polling, file upload — всё это standard DOM API.
  - Долгоживуще: vanilla JS работает через 10 лет; React/Svelte версии устаревают.
- **Cons:**
  - Нет JSX/template-shorthand, ручное `createElement`/`innerHTML` для repeating elements (mitigated через `<template>` + `cloneNode`).
  - Нет реактивности; обновление DOM при `state.candidates[]` change — explicit re-render of section.
  - Чуть больше boilerplate для table-render (но это <100 строк на проект).
- **Touch surface:** новый `app/web/static/js/{app,api,router,views/,components/}.js` + `app/web/static/css/`.

**Option B: HTMX + Jinja2 server-templates**
- **Подход:** FastAPI рендерит Jinja2 partials. Frontend = HTMX-attributes (`hx-post`, `hx-target`, `hx-swap`) на статической HTML-разметке. Минимальный JS только для charts.
- **Pros:**
  - Server-driven; frontend — 0 JS-логики, фронт-разработка = HTML+CSS+attrs.
  - Идеально для `index.html` aesthetic — он и так server-rendered look.
  - Forms, tables, partial updates "from the box".
- **Cons:**
  - **Plotly/D3 charts требуют JS** — придётся либо server-side SVG (Vega-Lite Python → SVG), либо инлайнить JS для chart-init, что ломает HTMX-чистоту.
  - Long-running ops (training 5 мин) на HTMX = SSE-stream + ручной partial-replace; non-trivial.
  - Связка тяжелее объяснить будущему контрибьютору, чем "тут vanilla".
  - Adds Jinja2 + jinja2-extra dependency; Streamlit не использует Jinja сейчас.
- **Touch surface:** `app/web/templates/*.html` (Jinja) + `app/web/static/`.

**Option C: Alpine.js (CDN, no build)**
- **Подход:** vanilla HTML с `x-data`, `x-show`, `x-on:click`, `x-bind`. Реактивность ~ Vue 2 без билдов.
- **Pros:**
  - Реактивность из коробки, ~15 KB gzipped, 1 `<script src="alpine.min.js">`.
  - Шаблоны привычны Vue/React-разработчикам.
- **Cons:**
  - Лишняя абстракция при solo dev и одном dashboard. 8 экранов не дают value-density Alpine-реактивности.
  - Alpine не идеален для **табличных данных** (Top-5 candidates, ferroalloy breakdown, decision log) — нужен ручной loop через `<template x-for>`, мало выигрыша против `<template>` + cloneNode в vanilla.
  - +1 зависимость с CDN-pinning.
- **Touch surface:** идентично vanilla, но с inline `x-*` атрибутами в HTML.

**Option D: React/Svelte + Vite build**
- **Подход:** SPA с router, state-management, build на Vite, `npm run build` → static.
- **Pros:**
  - Component model масштабируется.
  - Богатая ecosystem (recharts, react-hook-form).
- **Cons:**
  - **Overkill для MVP:** 8 относительно независимых страниц, нет shared state кроме "active model".
  - Build-pipeline в Python-проекте = два toolchain'а в CI, два package manager'а, deploy ломается, если нет node на target.
  - "Framework slop" риск — easy сделать generic-Material-UI/Tailwind look, ломая `index.html` aesthetic.
- **Touch surface:** `frontend/` (отдельная папка), `package.json`, build artifacts → `app/web/static/`.

**Option E: Lit + Web Components**
- **Подход:** typed Web Components, можно без билда (lit через CDN) или с билдом.
- **Pros:** standards-based, encapsulation.
- **Cons:** при 8 экранах web components оверкилл; mental overhead shadow DOM vs необходимость интегрировать chart-libs (которые часто плохо работают со shadow DOM).
- **Touch surface:** аналогично Option A + lit dependency.

---

### Group 2 — Chart library

**Option A: Plotly.js (CDN minified-basic build, ~700 KB) — recommended**
- **Подход:** загрузить `plotly-basic-2.x.min.js` через CDN; рендер scatter (Pareto), bar (feature importance, model compare), thermal heatmap-like.
- **Pros:**
  - Interactive (zoom, pan, hover) "из коробки" — точнее, чем dead SVG в index.html.
  - Темы поддерживаются: можно настроить layout с oklch-equivalent hex (`#0d1517` etc) под matched palette.
  - Уже фактически stack: Streamlit Altair → Vega-Lite, переход на Plotly даёт interactive parity.
  - **Хорошо** работает с long-running data updates (`Plotly.react(elem, data, layout)` для re-render).
- **Cons:**
  - 700 KB — заметный bundle, но один раз cached. Можно lazy-load только на тех вкладках, где нужно.
  - Нет sub-pixel control над styling axis labels — но index.html SVG axes тоже простые.

**Option B: Chart.js (CDN, ~280 KB)**
- **Подход:** простой Canvas-based.
- **Pros:** легче Plotly; быстрее load; популярен.
- **Cons:**
  - Canvas (не SVG) — текст менее crisp, не вписывается в SVG-aesthetic index.html.
  - Pareto scatter с marker-shape variations (in-spec/OOD/rejected/Top-5 ring) делается **через plugins**, не из коробки.

**Option C: D3.js + custom SVG**
- **Подход:** ручной D3-код для каждого чарта в стиле index.html.
- **Pros:**
  - 100% pixel-control, точная репродукция эстетики index.html.
  - Чистый SVG, можно выгружать как PNG/PDF.
- **Cons:**
  - **D3 = низкоуровневый;** на 4-5 чартов уйдёт ~600 строк; solo dev должен поддерживать это.
  - Каждое изменение data shape = переписывание D3-pipeline.

**Option D: ECharts (CDN, ~1 MB)**
- **Подход:** Apache ECharts.
- **Pros:** мощный, темы, экспорт.
- **Cons:** размер; китайская доминанта в community (доки иногда машинно-переведённые); меньше adoption в US/EU.

**Option E: Server-side SVG via Vega-Lite Python**
- **Подход:** backend генерирует SVG через `altair_saver` / `vega-lite-cli` и отправляет HTML-blob.
- **Pros:** консистентно со Streamlit; charts даже без JS.
- **Cons:**
  - **Не interactive;** теряем zoom/hover/tooltip, который ожидает modern user.
  - Vega-Lite Python → SVG требует node.js на server (через `vega-cli`) или дополнительные `kaleido`/`Orca` пакеты — плюс одна зависимость, которую сложно держать.

---

### Group 3 — Long-running operations (training / NSGA-II / LLM)

**Option A: Background tasks + polling (recommended)**
- **Подход:** POST `/api/train` создаёт `job_id`, запускает background `asyncio.create_task` (или `concurrent.futures.ThreadPoolExecutor` для CPU-bound XGBoost). Frontend polls GET `/api/jobs/{job_id}` каждые 2 сек, показывает progress + статус. Job-state — in-memory dict `{job_id: {status, progress, result, error}}`.
- **Pros:**
  - Простой; никаких WebSocket/SSE infrastructure.
  - Resilient к network glitches (polling возобновляется).
  - Подходит и для training (5 мин), и для NSGA-II (40 с), и для LLM (3 мин).
  - Solo-process FastAPI достаточно — Optuna держит свой `tqdm`-progress, но мы можем mock'нуть progress=10%/40%/95% по этапам как Streamlit делает сейчас.
- **Cons:**
  - In-memory dict теряется при `uvicorn --reload` (dev) или restart. Mitigated: production-режим запускается один раз, reload off; dev — пользователь повторит.
  - Нет real-time token streaming для LLM — пользователь видит "идёт..." до конца, как сейчас в Streamlit. Acceptable для MVP.

**Option B: Server-Sent Events (SSE)**
- **Подход:** GET `/api/train/stream` возвращает `text/event-stream` с прогресс-events.
- **Pros:** real-time updates, "правильно" архитектурно.
- **Cons:**
  - Требует генерации событий в backend — у `train_model()` нет hooks; нужно либо patch Optuna callback, либо background-thread + Queue → SSE-pump. **Усложнение без явной пользы для UX в MVP.**
  - Для LLM-стриминга: Anthropic SDK поддерживает `with stream` — можно потенциально стримить токены; **но** текущие генераторы возвращают `list[Recipe]` через `tool_use`, не raw текст. Token streaming не применим без переписывания generator-API. Скоуп non-goal.

**Option C: WebSocket**
- **Подход:** двунаправленный websocket для realtime.
- **Pros:** richest UX.
- **Cons:** оверкилл — нет client→server во время run; только server→client. Все usecase покрывает SSE, а SSE — polling.

---

### Group 4 — Human-in-the-loop checkpoints

> Реальное состояние: **Streamlit это не использует**. HITL живёт только в `scripts/run_pipeline.py` CLI. Frontend на FastAPI наследует тот же дизайн — Critic-warnings возвращаются вместе с результатом, frontend рендерит их, user читает.

**Option A: Сохранить текущее (no HITL в UI) — recommended**
- Status quo. На FastAPI `/api/inverse-design` возвращает result + critic_warnings вместе. UI показывает warnings в банере, но не блокирует. Это **точная парность с Streamlit**.

**Option B: Continuation-token state machine**
- POST `/api/pipeline/start` → 202 + `pipeline_id`. Если Critic блокирует, GET возвращает `{status: "checkpoint_required", question, context, pipeline_id}`. POST `/api/pipeline/{id}/respond {answer}` продолжает.
- Pros: правильно для CLI-equivalent UX в web.
- Cons: требует engine.Orchestrator integration; в текущем UI этого нет → out of scope для миграции (отдельная фича).

---

### Group 5 — Build pipeline

**Option A: Нет билда (recommended)**
- ES modules через `<script type="module">`, CSS `<link>`, чарт-libs через CDN. FastAPI сервит `app/web/static/` через `StaticFiles`. Развёртывание = `uvicorn` + папка static.
- Cons: нет minification, нет tree-shaking. Acceptable: vanilla code мал, CDN-libs уже minified.

**Option B: esbuild (один-shot bundler)**
- Опциональный `npm run build` создаёт bundle.
- Cons: вводит npm в проект; для solo MVP overkill.

**Option C: Vite (full SPA toolchain)**
- См. Option D в Group 1.

---

## Recommendation

**Stack:**
1. **Backend:** новый модуль `app/api/` с FastAPI (uvicorn). Тонкие обёртки над существующими backend-функциями. Никакой бизнес-логики не дублируется.
2. **Frontend framework:** **vanilla JS + ES modules** (Group 1, Option A).
3. **Chart library:** **Plotly.js basic build** через CDN (Group 2, Option A). Lazy-load на нужных вкладках.
4. **Long-running ops:** **background task + polling** (Group 3, Option A).
5. **HITL checkpoints:** **сохранить текущее поведение** (Group 4, Option A) — Critic warnings возвращаются inline; реальный state-machine HITL — отдельная фича (out of scope).
6. **Build:** **никакого build pipeline** (Group 5, Option A). FastAPI сервит static как есть.

**Почему такой выбор:**

- **Solo dev + MVP темп.** Vanilla + CDN убирает целый класс проблем (npm/yarn lockfiles, deps drift, build cache). Лидер-критерий из CLAUDE.md ("follow code, not README; не вводить абстракции без 3+ users") выполнен.
- **`index.html` уже vanilla.** Мы наследуем его CSS/HTML 1:1, добавляем JS только для интерактивности. React переписал бы CSS-classes в JSX — разрушительно.
- **Polling-based progress** парирует UX-разрыв со Streamlit (spinner) и при этом не требует SSE infrastructure. Все 7 long-running операций (train, inverse, recipe, hypothesis, deox-AI, anomaly, symbolic_regression) укладываются в одну паттерн-обёртку `run_as_job(fn, *args, **kwargs)`.
- **Plotly.js** — мост между existing Streamlit Altair (Vega-Lite) и interactive frontend. Минимум переучивания (графики Pareto/feature importance/comparison концептуально те же).
- **Парность с Streamlit поддерживается** — каждая вкладка переносится 1:1, можно сравнивать pixel-by-pixel.

**Что осознанно не делаем:**
- Не пишем Pydantic-models для каждого ответа — используем `dict` returns from backend как есть, FastAPI авто-сериализует через `default=str`. Если pydantic-validation в каком-то месте окажется ценен (например, для price upload), добавим точечно. Не превращать в bureaucracy.
- Не делаем auth — это отдельная фича; нет требования.
- Не делаем real-time SSE — UX и polling достаточно для текущих 7 операций.

---

## Endpoint map — full coverage всех 8 вкладок и 4 sub-tabs «Раскисление»

Нотация: `path` → `backend function` (file:line) → краткий request/response shape. Все ответы JSON если не сказано иначе.

### Common / system

| Endpoint | Backend | Notes |
|---|---|---|
| `GET /api/health` | — | `{status: "ok", version, llm_ready: bool}` |
| `GET /api/system` | filesystem scan | `{models: [...], price_snapshot_seed: {...}, llm_available: bool, decision_log_count: int, active_class_default: "pipe_hsla"}` для topbar |
| `GET /api/steel-classes` | `steel_classes.available_steel_classes()` | `[{id, name, standard, target_properties[], feature_set[], physical_bounds, expected_top_features, target_o_activity_ppm}]` — для UI form schema |
| `GET /api/steel-classes/{id}` | `steel_classes.load_steel_class(id)` | full profile YAML as dict |
| `GET /api/models` | scan `models/` + read `meta.json` | `[{version, steel_class, target, metrics: {r2_test, mae_test, coverage_90_ci}, created_at}]` |
| `GET /api/models/{version}/meta` | read `models/{v}/meta.json` | full meta (training_ranges, feature_importance, conformal_correction, etc) |

### Tab «История» (самая простая — начать миграцию здесь)

| Endpoint | Backend | Shape |
|---|---|---|
| `GET /api/decisions?phase=&limit=&tag=` | `decision_log.logger.query_decisions` | `[{id, timestamp, phase, decision, reasoning, alternatives_considered, context, tags, author, outcome}]` |
| `GET /api/decisions/summary` | `summarize_project_history()` | `{summary_md: "..."}` для topbar |

### Tab «Прогноз» (вторая по простоте — single sync call)

| Endpoint | Backend | Shape |
|---|---|---|
| `POST /api/predict` | `model_trainer.load_model` + `predict_with_uncertainty` + `compute_features_for_class` | req: `{model_version, composition: {feature: value}, n_ppm?}`. resp: `{prediction, lower_90, upper_90, ci_half_width, ood_flag, derived: {cev_iiw?, pcm?, cen?}, target, target_label}` |
| `POST /api/anomaly-explain` (long-running, polled) | `anomaly_explainer.make_anomaly_explainer().explain` | req: `{model_version, recipe, target, prediction, lower_90, upper_90, ood_flag}`. resp: `{job_id}` |

### Tab «Дизайн сплава» (HSLA only, NSGA-II, ~40 с)

| Endpoint | Backend | Shape |
|---|---|---|
| `GET /api/price-snapshot` | `cost_model.seed_snapshot` (default) | full snapshot dict |
| `POST /api/price-snapshot` | parse YAML upload / в-memory сохр. | save user snapshot to session-keyed storage |
| `POST /api/price-snapshot/yaml-export` | yaml.safe_dump | text/yaml download |
| `POST /api/inverse-design` (job) | `inverse_designer.run_inverse_design` + `validator.validate_batch` | req: `{model_version, targets: {yt: {min, max}}, hard_constraints: {cev_iiw: {max}, pcm: {max}}, population_size, n_generations, price_snapshot_id (optional), cost_mode, use_cost}`. resp: `{job_id}` |
| `GET /api/jobs/{job_id}` | in-memory job-store | `{status: queued|running|done|failed, progress: 0-100, message, result?, error?}` |
| `POST /api/jobs/{job_id}/breakdown-export?cand_idx=` | session result | `text/csv` blob |

### Tab «Обучение» (training, ~1-5 мин, polled)

| Endpoint | Backend | Shape |
|---|---|---|
| `POST /api/train` (job) | `data_curator.generate_synthetic_*` + `compute_features_for_class` + `model_trainer.train_model` + critic patterns + `make_llm_critic` | req: `{steel_class, target, n_optuna_trials}`. resp: `{job_id}` |
| `GET /api/jobs/{job_id}` | (общий) | result содержит `{version, metrics, feature_importance, training_ranges, critic_warnings, llm_observations}` |

### Tab «Раскисление» (4 sub-tabs)

| Endpoint | Backend | Shape |
|---|---|---|
| `POST /api/deox/forward` | `deoxidation.compute_al_demand` + `pattern_library.run_all_patterns(Phase.DEOXIDATION)` | req: `{o_a_initial_ppm, target_o_a_ppm, temperature_C, steel_mass_ton, al_purity_pct, burn_off_pct, model_id, heat_id?}`. resp: `{result: AlDemandResult, warnings: [...]}` |
| `POST /api/deox/inverse` | `deoxidation.compute_al_quality` | req: `{o_a_before_ppm, o_a_after_ppm, al_added_kg, temperature_C, steel_mass_ton, burn_off_pct, model_id}`. resp: `{result, warnings}` |
| `POST /api/deox/compare` | `deoxidation.compare_all_models` | resp: `{models: [...3 results...], spread_pct}` |
| `POST /api/deox/ai-advisor` (long-running, ~3 мин) | `compare_all_models` + `make_deoxidation_advisor.advise` + `make_deoxidation_critic.review` + log_decision | req: `{heat_context, llm_options}`. resp: `{job_id}` → result `{advisory, critic_verdict, log_id}` |
| `POST /api/decisions/save` | `log_decision` | optional persistence для forward/inverse |
| `GET /api/deox/models` | `THERMO_MODELS` registry | `[{id, name, citation, applicability_note}]` |

### Tab «Гипотезы» (LLM, ~3 мин)

| Endpoint | Backend | Shape |
|---|---|---|
| `POST /api/hypotheses/run` (job) | `scripts.generate_hypotheses_for_model.build_context` + `make_hypothesis_generator.generate` + `make_hypothesis_critic.review` + `log_decision` | req: `{model_version}`. resp: `{job_id}` |
| `GET /api/hypotheses/last?model_version=` | `query_decisions(phase=training, tag=hypothesis_cycle)` | last cycle's hypotheses + reviews |

### Tab «Подбор рецепта» (LLM, ~3 мин, fatigue_carbon_steel only)

| Endpoint | Backend | Shape |
|---|---|---|
| `POST /api/recipes/run` (job) | `make_recipe_designer.design` + ML prediction + cost compute + `make_recipe_critic.review` + `log_decision` | req: `{model_version, task_text}`. resp: `{job_id}` |
| `GET /api/recipes/last?model_version=` | `query_decisions(phase=inverse_design, tag=recipe_cycle)` | latest cycle |

### Tab «Следующие эксперименты» (active learning, ~2 с)

| Endpoint | Backend | Shape |
|---|---|---|
| `POST /api/active-learning/propose` | `active_learner.propose_next_experiments` + `log_decision` | req: `{model_version, n_samples, top_k, seed}`. resp: `{baseline, f_star, proposals: [...]}`. **Synchronous** — операция дешёвая (150 ms-3 s), polling не нужен. |
| `GET /api/active-learning/last?model_version=` | `query_decisions(tag=active_learning)` | latest run |

### Topbar / KPI strip — agreggates

KPI strip в `index.html` показывает: «Плавок за смену», «In-spec rate», «Себестоимость», «Перерасход Al», «PhD-рецензий за нед.». Это **бизнес-метрики, которых сейчас нет в backend**. См. Open Question #5 ниже.

---

## File layout

```
app/
  api/                          # NEW
    __init__.py
    main.py                     # FastAPI app, mounts routers, StaticFiles
    deps.py                     # shared dependencies (current model, llm-availability)
    jobs.py                     # in-memory job-store + run_as_job() helper
    routers/
      system.py                 # /api/health, /api/system, /api/steel-classes, /api/models
      decisions.py              # /api/decisions
      predict.py                # /api/predict, /api/anomaly-explain
      design.py                 # /api/inverse-design, /api/price-snapshot
      train.py                  # /api/train
      deox.py                   # /api/deox/{forward,inverse,compare,ai-advisor}, /api/deox/models
      hypotheses.py             # /api/hypotheses/{run,last}
      recipes.py                # /api/recipes/{run,last}
      active_learning.py        # /api/active-learning/{propose,last}
    schemas.py                  # Pydantic models — только если нужны для validation; начать с минимума
  web/                          # NEW (static frontend)
    static/
      index.html                # entry, derived from Steel-AI-MVP/index.html (CSS preserved)
      css/
        app.css                 # extracted from <style> block in index.html
        overrides.css           # tab-specific tweaks added during migration
      js/
        app.js                  # bootstrap, router, common services
        api.js                  # fetch wrappers, job polling
        router.js               # hash-based routing (#/design, #/train, ...)
        components/
          topbar.js
          nav.js
          job-progress.js       # universal progress bar for jobs
          critic-card.js        # PhD critic card (used in 4 tabs)
          decision-log-row.js   # used in History + lower-band of Дизайн
          price-editor.js       # ferroalloy table editor
        views/
          design.js             # Tab 01 — Дизайн сплава
          recipe.js             # Tab 02 — Подбор рецепта
          hypotheses.js         # Tab 03 — Гипотезы
          deox.js               # Tab 04 — Раскисление (4 sub-tabs внутри)
          predict.js            # Tab 05 — Прогноз
          experiments.js        # Tab 06 — Следующие эксперименты
          train.js              # Tab 07 — Обучение
          history.js            # Tab 08 — История решений
        charts/
          pareto.js             # Plotly scatter с marker-shape variants
          feature-importance.js # Plotly horizontal bar
          deox-compare.js       # Plotly grouped bar
          spark.js              # KPI sparklines (тонкий SVG, без Plotly)
        utils/
          format.js             # number/date formatting, units
          dom.js                # query/template helpers
  frontend/                     # OLD — Streamlit UI, остаётся до конца миграции
    app.py                      # untouched
  backend/                      # untouched
  tests/                        # untouched

app/tests/
  test_api_*.py                 # NEW per-router tests using FastAPI TestClient
```

**Запуск:**

```bash
# Старый Streamlit (продолжает работать)
PYTHONPATH=. streamlit run app/frontend/app.py

# Новый FastAPI + static frontend
PYTHONPATH=. uvicorn app.api.main:app --reload --port 8000
# UI: http://localhost:8000/  (FastAPI mount StaticFiles на "/")
# API: http://localhost:8000/api/*
# OpenAPI docs: http://localhost:8000/docs (free benefit от FastAPI)
```

**Docker update (optional, см. Open Question #2):**
- Сейчас `docker-compose.yml` поднимает Streamlit (port 8501).
- На время co-existence можно держать **оба сервиса** в compose (streamlit на 8501, fastapi на 8000) — или разные порты в одном `Dockerfile.api`/`Dockerfile.streamlit`.

---

## Migration plan — последовательность вкладок

Принципы:
- **Один PR / коммит = одна вкладка** или одна инфраструктурная подзадача.
- После каждого PR: `pytest app/tests -v` зелёный, `scripts/smoke_test.py` зелёный, Streamlit продолжает работать.
- Frontend-shell строится один раз заранее (PR 1), затем вкладки добавляются по одной.

| PR # | Scope | Затронутые файлы | DoD |
|---|---|---|---|
| **1** | **Infrastructure**: создать `app/api/main.py` (минимальный FastAPI с health + StaticFiles на `/`), скопировать `Steel-AI-MVP/index.html` → `app/web/static/index.html`, вырезать CSS в `app.css`, написать `app.js` skeleton с router (8 пустых вкладок). Добавить `app/api/jobs.py` (job-store + `run_as_job`). Запуск через `uvicorn` работает, открывается dashboard, навигация переключает заглушки. | `app/api/main.py`, `app/api/jobs.py`, `app/web/static/{index.html,css,js}/`, `app/tests/test_api_health.py` | `pytest test_api_health.py` ✓; `curl /api/health` ✓; ручная проверка UI: nav кликается. Streamlit не сломан. |
| **2** | **Tab «История»** (самая простая, нет mutation, нет async). | `app/api/routers/decisions.py`, `app/web/static/js/views/history.js`, `app/web/static/js/components/decision-log-row.js`, `app/tests/test_api_decisions.py` | API возвращает то же, что Streamlit показывает; UI рендерит фильтр по фазе + список + раскрытие записи. |
| **3** | **Tab «Прогноз»** (synchronous, средняя сложность — class-aware форма). | `app/api/routers/predict.py`, `app/api/routers/system.py` (steel-classes, models), `app/web/static/js/views/predict.js`, `app/tests/test_api_predict.py` | Composition form подстраивается под `feature_set` активной модели; `/api/predict` совпадает 1:1 со Streamlit (cross-tested на одном `selected_model`). |
| **4** | **Tab «Раскисление» — sub-tabs Forward / Inverse / Compare** (sync, без LLM). | `app/api/routers/deox.py`, `app/web/static/js/views/deox.js` (без AI sub-tab), `app/tests/test_api_deox.py` | Все три sub-tabs работают; результаты эквивалентны Streamlit. Pattern Library warnings (DX01/DX02/DX03) рендерятся. |
| **5** | **Tab «Следующие эксперименты»** (sync, ~2 с, простой). | `app/api/routers/active_learning.py`, `app/web/static/js/views/experiments.js`, `app/tests/test_api_active_learning.py` | Top-K кандидатов с EI, OOD-флагами, delta vs baseline. |
| **6** | **Job infrastructure**: расширить `jobs.py` для long-running ops (training, NSGA-II, LLM). Реализовать polling component `job-progress.js` + lazy Plotly loader. | `app/api/jobs.py`, `app/web/static/js/components/job-progress.js`, `app/web/static/js/charts/pareto.js`, `app/tests/test_api_jobs.py` | Polling работает, progress отображается, Plotly загружается только когда нужен chart. |
| **7** | **Tab «Дизайн сплава»** (NSGA-II, ~40 с, polled). Включает price-editor, Pareto chart, candidate table, breakdown export. | `app/api/routers/design.py`, `app/web/static/js/views/design.js`, `app/web/static/js/components/price-editor.js`, `app/tests/test_api_design.py` | Pareto chart соответствует Streamlit Altair визуально (interactive Plotly version); price upload работает; breakdown CSV скачивается. |
| **8** | **Tab «Обучение»** (training, ~1-5 мин, polled). | `app/api/routers/train.py`, `app/web/static/js/views/train.js`, `app/web/static/js/charts/feature-importance.js`, `app/tests/test_api_train.py` (smoke с малым n_trials) | Полный cycle обучения через UI; new model появляется в /api/models; критик warnings + LLM observations. |
| **9** | **Tab «Раскисление» — AI advisor sub-tab** (LLM, ~3 мин). | `app/api/routers/deox.py` (расширение), `app/web/static/js/views/deox.js` (4-я subtab) | Полный operator protocol + PhD critic, decision log saved. |
| **10** | **Tab «Гипотезы»** (LLM, ~3 мин). | `app/api/routers/hypotheses.py`, `app/web/static/js/views/hypotheses.js`, `app/tests/test_api_hypotheses.py` | Cycle через UI, рендер 5 гипотез с verdict-badges. |
| **11** | **Tab «Подбор рецепта»** (LLM, ~3 мин, fatigue_carbon_steel only). | `app/api/routers/recipes.py`, `app/web/static/js/views/recipe.js`, `app/tests/test_api_recipes.py` | Recipe pair работает, evidence + fact-check рендерятся. |
| **12** | **Anomaly Explainer hookup** (внутри Прогноза, появляется при OOD). | `app/api/routers/predict.py` (расширение), `app/web/static/js/views/predict.js` | Кнопка «Объяснить почему рискованно» вызывает /api/anomaly-explain как job. |
| **13** | **KPI strip + topbar live data** (зависит от Open Question #5). | `app/api/routers/system.py` (расширение), `app/web/static/js/components/topbar.js` | KPI карточки заполнены реальными данными или явно помечены как «mock» с pinned numbers. |
| **14** | **Cleanup & docs**: обновить CLAUDE.md (как теперь стартовать), README, добавить `Dockerfile.api`. | `CLAUDE.md`, `README.md`, `Dockerfile.api`, `docker-compose.yml` | Документация описывает оба варианта; tests все зелёные; smoke OK. |
| **15** | **Streamlit decommission** (после approve пользователем). Удалить `app/frontend/app.py`, обновить scripts/smoke_test.py если он вызывает Streamlit. | `app/frontend/app.py` (delete), CLAUDE.md | Streamlit удалён; репо собирается; единственный UI — FastAPI. |

**Ожидаемая длительность:** PR 1 — 1 день; PR 2-5 — по полдня каждый; PR 6 — 1 день (job infra); PR 7-12 — по 1-1.5 дня каждый; PR 13-15 — по полдня. Итого ~12-14 рабочих дней solo.

---

## Tradeoffs accepted

- **Теряем `st.dataframe` editor.** Заменяем `<table>` + ручной inline edit (price-editor component). Менее удобно, но нет блокеров.
- **Теряем `st.tabs` мгновенное переключение.** Hash-routing с lazy view loading даёт тот же UX feel, но первый клик на вкладку = ~50 ms задержка из-за `import()`. Незначительно.
- **Теряем `st.rerun()` + auto-rerun.** В vanilla JS любое state-change = explicit re-render of section. Чуть больше кода, но проще для отладки.
- **Теряем mid-process LLM token streaming.** В Streamlit его и не было — спиннер до конца. На FastAPI polling c 2 сек интервалом даёт «пульсацию» индикатора, тоже acceptable.
- **In-memory job-store не переживает restart.** Acceptable для MVP — solo dev запускает один раз; в production-mode (`uvicorn` без --reload) сессия живёт долго. Если станет проблемой — переход на Redis/SQLite job-store позже.
- **Не принимаем reactive framework.** Сложные виды (price-editor с inline edit и derived totals) делаются вручную; ~50 строк vanilla кода. Если в будущем UI существенно вырастет (новые dashboards, drag-n-drop), обоснуем переход на Lit/Svelte отдельным design doc.

---

## Open questions for user (для Step 3)

1. **Deploy target?** Если планируется Hugging Face Spaces, оно ожидает Streamlit/Gradio из коробки. Для FastAPI на HF Spaces нужен docker-template, не SDK-template. Если deploy — на собственный VPS / docker-compose, никаких ограничений нет. **Где будет крутиться production?**

2. **Docker update сейчас или потом?** Текущий `docker-compose.yml` поднимает Streamlit (8501). Делать ли **в той же миграции** обновление до варианта «два сервиса (Streamlit 8501 + FastAPI 8000)» во время co-existence? Или оставить compose для Streamlit, а FastAPI пока запускать локально?

3. **Auth/login нужен?** Index.html показывает «А. Крылов · Технолог ЦЗЛ» в topbar. Это mock или планируется multi-user? Если multi-user — это отдельный регламент (R-XXX) и блокер для Tab 13 (KPI). Я предлагаю **trail-step:** вместо auth положить имя текущего user в `.env` (`USER_NAME=`, `USER_ROLE=`) и читать через `/api/system`.

4. **Streamlit decommission timing.** После какого PR пользователь готов удалить Streamlit? PR 15 — это формальный последний шаг, но фактически после PR 12 функциональная парность достигнута. **Хотим ли держать Streamlit как fallback ещё месяц после миграции?**

5. **KPI strip — реальные данные или mock?** Пять метрик в `index.html` («Плавок за смену 14/18», «In-spec 94.2%», «Себестоимость 741 €/т», «Перерасход Al +8.3%», «PhD-рецензий 37шт») — НЕ существуют в текущем backend. Варианты:
   - **(a)** Mock-numbers, помеченные «demo data» в UI до создания production-data pipeline. **Default рекомендация для MVP.**
   - **(b)** Mapping на существующее: «PhD-рецензий за нед» = `count(decisions where tag in {recipe_cycle, hypothesis_cycle, deoxidation_cycle} and timestamp > now-7d)` — реализуемо. Остальные 4 — нет источника.
   - **(c)** Скрыть KPI strip совсем до появления business-data ingestion.

6. **«Активный класс» в topbar** — фиксированный (`pipe_hsla`) как у index.html, или должен переключаться через dropdown? Сейчас в Streamlit активный класс **выводится** из выбранной модели. Предлагаю сохранить логику: «активный класс» = `meta.steel_class` выбранной модели + dropdown для переключения модели в topbar.

7. **Как обрабатывать models/<version> которые имеют старый `meta.json` (без `steel_class`)?** Streamlit fallback на `pipe_hsla`. Сохранить тот же fallback в API? **Default рекомендация: да.**

---

## Files to touch (детальный список)

**Создаются:**
- `app/api/__init__.py`
- `app/api/main.py` — FastAPI app, CORS (если нужен), StaticFiles mount, router includes
- `app/api/deps.py` — dependencies для get-current-llm, get-job-store
- `app/api/jobs.py` — `JobStore`, `run_as_job(coro_or_fn) -> job_id`, `get_job(id)`
- `app/api/schemas.py` — Pydantic models (минимально, по мере необходимости)
- `app/api/routers/system.py`
- `app/api/routers/decisions.py`
- `app/api/routers/predict.py`
- `app/api/routers/design.py`
- `app/api/routers/train.py`
- `app/api/routers/deox.py`
- `app/api/routers/hypotheses.py`
- `app/api/routers/recipes.py`
- `app/api/routers/active_learning.py`
- `app/web/static/index.html` — entry (адаптация Steel-AI-MVP/index.html)
- `app/web/static/css/app.css` — извлечён из <style> блока index.html
- `app/web/static/css/overrides.css` — для tab-specific tweaks
- `app/web/static/js/app.js`, `api.js`, `router.js`
- `app/web/static/js/components/{topbar,nav,job-progress,critic-card,decision-log-row,price-editor}.js`
- `app/web/static/js/views/{design,recipe,hypotheses,deox,predict,experiments,train,history}.js`
- `app/web/static/js/charts/{pareto,feature-importance,deox-compare,spark}.js`
- `app/web/static/js/utils/{format,dom}.js`
- `app/tests/test_api_health.py`
- `app/tests/test_api_decisions.py`
- `app/tests/test_api_predict.py`
- `app/tests/test_api_design.py` (включая тест что NSGA-II запуск создаёт job, returns ID, polling возвращает result)
- `app/tests/test_api_deox.py`
- `app/tests/test_api_train.py` (smoke, малое n_trials)
- `app/tests/test_api_hypotheses.py` (mock LLM)
- `app/tests/test_api_recipes.py` (mock LLM)
- `app/tests/test_api_active_learning.py`
- `app/tests/test_api_jobs.py` (job-store unit tests)
- `Dockerfile.api` — отдельный image для FastAPI
- `docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md` — этот документ

**Меняются:**
- `requirements.txt` — НЕТ изменений (FastAPI/uvicorn уже там; **Streamlit оставляем до PR 15**)
- `CLAUDE.md` — раздел «Команды запуска»: добавить `uvicorn app.api.main:app`; раздел «UI и API» — переписать
- `docker-compose.yml` — добавить второй сервис (PR 14)
- `.gitignore` — добавить `app/web/static/js/.cache/` если будет lazy-import cache (вряд ли)
- `scripts/smoke_test.py` — если вызывает Streamlit-specific API, обновить

**Удаляются (только в PR 15, после approve):**
- `app/frontend/app.py`
- (не сейчас) `Steel-AI-MVP/` — сохранить как design-reference в `docs/design-references/`

---

## Skill activations для Step 4 (Developer)

При реализации Developer должен активировать (через `steel-domain-expert` dispatcher или напрямую):

1. **`mlops-reproducibility-senior`** — для убедиться, что job-store / decision_log integration не ломает audit trail; для секции про «in-memory job loss on restart» — есть ли здесь reproducibility-риск (мнение: нет, потому что результаты успешных jobs persist в decision_log).
2. **`frontend-design`** (этот skill) — для финальной полировки эстетики на этапах PR 1, 7, 13. Не для рекомендаций — они приняты в этом design doc — а для реализации сложных компонентов (Pareto chart, KPI sparklines).
3. **`steel-ai-add-pattern`** — НЕ нужен (мы не добавляем новые паттерны).
4. **`steel-ai-add-llm-capability`** — НЕ нужен (мы не добавляем новые LLM-модули).
5. **PhD/domain skills** — НЕ нужны (миграция чисто инфраструктурная, без изменения domain-логики).
6. **`spec-driven-feature` / `superpowers:writing-plans`** — для разбивки PR'ов на atomic tasks при старте Step 4.
7. **`superpowers:test-driven-development`** — для каждого нового router'а: тест на TestClient before код роутера.

---

## User decisions (Step 3 approval — 2026-05-09)

User одобрил старт по этому плану. Финальные ответы на open questions:

1. **Deploy target → Hugging Face Spaces (Docker SDK).** PR 14 = `Dockerfile.api` под Docker SDK template HF Spaces, uvicorn на порту 7860.
2. **KPI strip → реальные метрики из active model + decision_log.** Переинтерпретируем 5 mockup-карточек:
   - "Плавок за смену" → "Активная модель" (`meta.steel_class` + версия)
   - "In-spec rate" → "Model R²" (test метрика из meta.metrics)
   - "Себестоимость" → "Coverage 90% CI" (calibration метрика, %)
   - "Перерасход Al" → "Pareto size" (последний design-run из decision_log, fallback "—")
   - "PhD-рецензий за нед" → `count(decisions where tag in {recipe_cycle, hypothesis_cycle, deoxidation_cycle, llm_critic} and timestamp > now-7d)`
   Все 5 — реальные данные, без mock. Sparklines — placeholder до PR 7.
3. **Streamlit decommission → сразу после PR 12.** PR 13 удаляет `app/frontend/app.py`. PR 14-15 — Docker/HF Spaces.
4. **Активный класс в topbar** → переключение через model-dropdown (как сейчас в Streamlit sidebar). Default рекомендация принята.
5. **Старые `meta.json` без `steel_class`** → fallback на `pipe_hsla`. Default рекомендация принята.
6. **Auth/login** → не нужен (HF Spaces public, MVP, solo dev). Out of scope.

Незакрытые но не блокирующие вопросы (решаем во время соответствующего PR):
- Точный layout 5 KPI-карточек после переинтерпретации — финализируем в PR 7.
- Pareto sparkline стиль (Plotly minimal vs custom inline SVG) — PR 7.

---

## Risks (top 3)

1. **`uvicorn --reload` теряет in-memory job-store при изменении файла во время training.** Митигация: дев-режим без `--reload` для долгих сессий; либо опциональный SQLite-backed JobStore (опция, в PR 6+).
2. **Visual fidelity index.html ↔ реальные данные.** Hand-crafted SVG'и в mockup нарисованы под красивый "demo" data shape. Plotly с реальным NSGA-II может выглядеть менее аккуратно (расположение точек, плотность кластера). Митигация: PR 7 включает task на pixel-perfect tuning Plotly layout под NSGA-II output на realистичных данных.
3. **Backend-функции возвращают dataclass'ы и numpy-типы;** FastAPI авто-сериализация может ломаться (`np.float32` не JSON-serializable). Митигация: общий `_json_default` encoder в `app/api/main.py` (numpy scalar/array → native, dataclass → asdict, datetime → isoformat) + `SafeJSONResponse`. **Важно (gotcha обнаружен в PR 1):** в FastAPI+pydantic v2 default ответ-сериализация идёт ДО `Response.render()`, поэтому routers возвращающие dict/dataclass с numpy внутри ДОЛЖНЫ декларироваться так:
   ```python
   @router.get("/foo", response_class=SafeJSONResponse, response_model=None)
   def foo() -> dict: ...
   ```
   либо возвращать `SafeJSONResponse(content=payload)` напрямую. Иначе будет `PydanticSerializationError`. Endpoints с примитивами (`/api/health`) от этого не страдают.
