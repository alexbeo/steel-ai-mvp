# Спецификация: AI для снижения расхода Al при раскислении (Tier 1 + Tier 2)

## Статус

Phase 1 (MVP) — **завершён 2026-05-20**. PR 1-14 (R-001 Feature Development, Block V плана
`snug-roaming-seahorse`). Все 13 implementation-PR прошли цикл Architect → Developer →
Reviewer → QA; PR 14 — capstone (E2E smoke + документация).

Регрессия на момент закрытия: 620/620 тестов зелёные, ~120 новых unit-тестов добавлено
за PR 1-13.

## Контекст и цель

**Проблема.** При раскислении жидкой стали алюминием на ladle furnace эффективность усвоения
η_Al сильно варьируется между цехами и плавками (реокисление от FeO шлака, поздняя подача,
высокая температура, состояние футеровки). На практике это ведёт к **систематическому
overconsumption Al** — операторы закладывают запас «на всякий случай», потому что
литературное η_typical не отражает специфику конкретного цеха.

**Цель Phase 1.** Дать advisory-инструмент, который оценивает η_Al *plant-specific* и
*feature-aware*, выдаёт доверительные интервалы расхода (а не точечную оценку) и помогает
выбрать метод подачи по multi-objective критерию. Целевой эффект — **−10..−15% расхода Al**
без потери качества (целевая [O]_a и остаточный [Al] выдержаны).

**Что осталось вне Phase 1** (см. раздел «Известные ограничения / Phase 2+»): кинетика,
shadow-валидация на реальном производстве, closed-loop с анализаторами кислорода.

## Архитектура (реализованная)

Слоистая, поверх существующего physics-only калькулятора `slag_aware_deox.py`:

1. **Хранилище плавок** (`heat_records.py` + `data/heats/schema.sql`) — SQLite, gitignored.
   Гибкий ingestion: ручной ввод (UI), Excel ETL, CSV bulk, synthetic generator.
2. **2-уровневая оценка η_Al**:
   - **Tier 1 — Bayesian posterior** (`eta_al_calibration.py`): Normal-Normal conjugate
     update в logit-space над literature prior, отдельно per `(plant_id, method_id)`.
     N≥30 threshold для commit posterior; иначе остаёмся на literature prior.
   - **Tier 2 — global ML модель** (`eta_al_predictor.py` + pipeline через `engine.py`):
     XGBoost + conformal по всем процесс-факторам.
   - **Mixture-of-experts** (`eta_al_predictor.py`): смешивает plant posterior и global ML
     в logit-space; вес `w = sigmoid((N_plant − 30)/10)` — гладкий переход, дисперсия
     включает штраф за рассогласование источников.
3. **Conformal интервалы** (`slag_aware_deox.py`): η-uncertainty (logit μ, σ)
   распространяется в массу Al → **p10/p50/p90** (`al_pure_kg_p10/_p50/_p90`).
4. **Multi-objective оптимизация** (`recommend_optimal_method`): objective ∈
   `cost / al_mass / pareto`; pareto — non-dominated frontier по (al_pure_kg, cost),
   chosen = knee point.
5. **Discovery-инструменты (LLM, опциональные)**: symbolic regression η-коррекции и
   high-Al anomaly explainer — degrade gracefully без `ANTHROPIC_API_KEY`.
6. **Качество-гейты** (`pattern_library/patterns.py`): DX08-DX12.

## Реализованные компоненты (PR 1-14)

| PR | Компонент | Ключевые файлы | Тесты |
|----|-----------|----------------|-------|
| 1  | Хранилище плавок (Pydantic + SQLite CRUD) | `app/backend/heat_records.py`, `data/heats/schema.sql` | unit CRUD |
| 2  | API + UI sub-tab «История плавок» | `app/api/routers/heats.py`, `app/web/static/*` | API |
| 3  | Excel ETL + critic + CLI импорт | `app/backend/heat_history_etl.py`, `heat_etl_critic.py`, `scripts/import_heats_from_excel.py` | ETL |
| 4  | Synthetic deox dataset + virtual class | `data/steel_classes/deox_calibration.yaml`, `data_curator.generate_synthetic_deox_calibration_dataset` | generator |
| 5  | Bayesian per-plant×method калибровка | `app/backend/eta_al_calibration.py`, `scripts/calibrate_eta_al.py` | posterior math |
| 6  | Conformal p10/p50/p90 в demand | `app/backend/slag_aware_deox.py` | propagation |
| 7  | objective=cost/al_mass/pareto + UI scatter | `app/backend/slag_aware_deox.py`, UI | optimize |
| 8  | Pipeline тренирует deox_calibration | `app/backend/engine.py`, `scripts/run_pipeline.py --class deox_calibration` | pipeline |
| 9  | EtaAlPredictor (mix global + plant) | `app/backend/eta_al_predictor.py` | mixture |
| 10 | UI «Калибровка η_Al» + 3 endpoints + optimize wiring | `app/api/routers/*`, `app/web/static/*` | API |
| 11 | Symbolic regression η-коррекция | `scripts/symbolic_eta_correction.py`, `symbolic_eta_critic.py` | SR |
| 12 | High-Al anomaly diagnoser (cohort) | `app/backend/high_al_diagnoser.py`, `scripts/explain_high_al_heats.py` | cohort |
| 13 | Pattern Library DX08-DX12 | `pattern_library/patterns.py` | patterns |
| 14 | E2E smoke + spec + CLAUDE.md | `scripts/smoke_test_deox_calibration.py`, этот файл, `CLAUDE.md` | smoke |

## Capabilities (Tier 1 + Tier 2)

- **A — plant-specific Bayesian калибровка η_Al** (PR 5). Closed-form Normal-Normal
  posterior в logit-space; per `(plant_id, method_id)`; YAML-снапшоты в
  `data/deox_methods/calibrations/<plant_id>.yaml`.
- **B — conformal interval p10/p50/p90** (PR 6). η-uncertainty → масса Al; p90 —
  safety-граница, p10 — оптимистичная.
- **D — objective=al_mass/pareto** (PR 7). Multi-objective выбор метода; pareto frontier
  + knee point.
- **E — ML модель η_Al** (PR 8). XGBoost + conformal через стандартный pipeline
  (`engine.py`), класс `deox_calibration`.
- **F — symbolic regression η-коррекции** (PR 11). gplearn; target = η/method_baseline;
  Pareto complexity-vs-R²; LLM-critic опционален.
- **H — anomaly explainer high-Al** (PR 12). Детерминированная детекция cohort'а
  (percentile/prediction) + Sonnet root-cause (опционально).

## Data flow

```
synthetic / Excel / manual / CSV
        │
        ▼
   heats.db (SQLite, gitignored)
        │
        ├──► EtaAlCalibrator (Tier 1) ──► calibrations/<plant>.yaml
        │                                        │
        └──► pipeline train (Tier 2) ──► models/deox_eta_al_effective_xgb_<ts>/
                                                 │
                                                 ▼
                              EtaAlPredictor (mix plant posterior + global ML)
                                                 │
                ┌────────────────────────────────┼────────────────────────────────┐
                ▼                                 ▼                                 ▼
   compute_al_demand_slag_aware       recommend_optimal_method        high_al_diagnoser /
   (p10/p50/p90)                      (cost/al_mass/pareto)           symbolic_eta_correction
```

## Verification

- **E2E smoke**: `scripts/smoke_test_deox_calibration.py` прогоняет все 10 этапов на
  synthetic data в изолированном tmp (не загрязняет рабочий каталог). Работает без
  `ANTHROPIC_API_KEY` и без real data. Verified 2026-05-20:
  `--skip-train` → все этапы зелёные (800 heats → 15 posteriors above N≥30, conformal
  p10≤p50≤p90, pareto frontier=2, DX09/DX11 fire, SR frontier=3, 162 high-Al outliers).
- **Unit**: ~120 тестов за PR 1-13 (CRUD, posterior math, mixture, conformal propagation,
  pareto, DX08-DX12).
- **Regression**: 620/620 на момент закрытия Phase 1.

## Известные ограничения / Phase 2+

- **DX10 dormant** — basicity-экстраполяция не wired в ctx-builder (ключи
  `current_slag_basicity` / `historical_basicity_range` не передаются); check молчит,
  активируется автоматически когда ключи появятся.
- **DX12 partial-dormant** — конфликт plant-posterior↔global ML требует
  reconstruction `global_eta_logit_mu`; не всегда доступен в ctx.
- **ML branch редко срабатывает из optimize** — в `optimize`-запросе нет полного
  composition для feature-vector, поэтому predictor чаще идёт plant_only / literature.
- **Synthetic feature_deltas частично тавтологичны** — synthetic-генератор задаёт η
  закрытой формулой, поэтому high-Al cohort deltas на synthetic коррелируют с генератором;
  на heats.db (real) сигнал честный.
- **σ_likelihood фиксирована = 0.5** (logit-space) — Phase 2: оценивать из данных.
- **default-arg-binding в `heat_records`** — функции берут `db_path=DEFAULT_DB_PATH` как
  default-аргумент (binding на момент def), поэтому monkeypatch модульной константы не
  работает; везде нужно передавать `db_path=` явно. Refactor-ticket на Phase 2.
- **Phase 2** — shadow-валидация на реальных плавках (predicted vs actual без влияния на
  операцию), оценка σ_likelihood, basicity wiring (DX10).
- **Phase 3** — production closed-loop с анализаторами кислорода и feedback.

## Регуляторные триггеры

- **R-001 Feature Development** — основной регламент Phase 1 (catch-all для новой
  функциональности; вся последовательность PR 1-14).
- **R-003 Add AI Capability** — паттерн для LLM-модулей PR 11 (symbolic critic) и PR 12
  (high-Al diagnoser): `load_prompt` try/except, factory gate на `ANTHROPIC_API_KEY`,
  structured `tool_use`, `cache_control=ephemeral`.
- **R-004 Add Steel Class** — virtual class `deox_calibration` (PR 4).
- **R-005 Pattern Library Extension** — DX08-DX12 (PR 13).
