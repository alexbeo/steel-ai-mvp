# Спецификация: Phase 2 — Shadow Validation (η_Al calibration)

## Статус

Phase 2 — завершён 2026-05-20. PR S1-S5 (R-001).

## Контекст и цель

Phase 1 (η_Al calibration stack) дал per-plant Bayesian-постериоры η_Al и
AI-рекомендацию Al-дозы (`EtaAlPredictor` + `compute_al_demand_slag_aware`).
Phase 2 проверяет ключевую гипотезу проекта:

> AI снижает расход Al на ≥10% без потери качества раскисления.

Метод — **retrospective shadow validation** на `heats.db`: для каждой исторической
плавки вычисляем, сколько Al рекомендовал бы AI, и сравниваем с фактически поданным
`al_added_kg`. Гипотеза подтверждается **только** на real heats; на synthetic
демо-данных проверяется лишь механика пайплайна.

## Решения (discovery)

- **Retrospective replay** `heats.db` (фактический `al_added_kg`), не A/B и не
  prospective trial.
- **Полный стек** в одном эпике: comparison engine + stats + report + UI.
- **Quality gate**: AI-рекомендация засчитывается только если она не раскисляет
  хуже (тот же target O_a) + фактический residual [Al] в spec + p90-доза достаточна.
  Экономия считается **только** по quality-pass плавкам — честная экономия от
  высокого прогнозного η, не от недодозировки.
- **Статистика**: per-heat paired Wilcoxon signed-rank + percentile bootstrap CI
  на median Δ%, стратификация plant × method.
- **Comparison в pure-Al-эквиваленте** (charge-форма → pure через `al_content_pct`
  метода добавки).

## Архитектура / data flow

```
heats.db (has_outcome=True)
  → AI recommendation per heat
      (EtaAlPredictor + compute_al_demand_slag_aware → p50/p90 pure-Al)
  → ShadowComparison (per-heat delta + quality gate)        [S1]
  → Wilcoxon + bootstrap CI + стратификация
      → ShadowStats (hypothesis_met)                         [S2]
  → HTML/JSON report (savings kg + €, disclaimers)           [S3]
  → UI dashboard sub-tab «🌓 Shadow-валидация» + endpoints    [S4]
```

`PipelineState` не задействован — shadow validation работает напрямую с `heats.db`
и калибровками, вне основного train/design пайплайна.

## Компоненты (S1-S5)

| PR | Компонент | Файлы | Тесты |
|---|---|---|---|
| S1 | Comparison engine: per-heat AI vs actual + quality gate | `app/backend/shadow_validation.py` | unit (ShadowComparison, skip-reasons, quality gate) |
| S2 | Statistics: Wilcoxon + bootstrap CI, `hypothesis_met` (4 условия), per-stratum | `app/backend/shadow_stats.py` | unit (edge cases, CI, hypothesis logic) |
| S3 | Report: HTML + JSON, savings kg + €, disclaimers, synthetic warning | `app/backend/shadow_reporter.py` + `scripts/generate_shadow_report.py` | unit (render, NaN-safe, escaping) |
| S4 | API endpoints + UI sub-tab «🌓 Shadow-валидация» | `app/api/routers/deox.py` + `app/web/static/` | endpoint/router tests |
| S5 | E2E smoke + docs | `scripts/smoke_test_shadow.py` + этот spec + CLAUDE.md | smoke chain S1→S2→S3 |

Итого по `pytest -k shadow`: **41 теста** проходят.

## hypothesis_met definition

`ShadowStats.hypothesis_met = True` ⟺ одновременно:

1. `median_delta_pct ≤ −target_reduction_pct` (default −10%, по quality-pass), И
2. `wilcoxon_p < 0.05` (значимый сдвиг от нуля), И
3. `ci_high < 0` (весь bootstrap 95% CI ниже нуля), И
4. `n_quality_pass ≥ min_hypothesis_n` (default 30 — выборка достаточна).

Aggregate-тест confirmatory; per-stratum p-values — exploratory (без коррекции на
multiple comparisons).

## Честный framing (критично)

- **Retrospective ≠ counterfactual.** Оператор при фактической плавке мог видеть
  информацию, недоступную модели.
- **Экономия — от высокого прогнозного η_Al** (меньше потерь в шлак/угар), НЕ от
  недодозировки: AI-доза бюджетирует тот же target O_a и residual [Al].
- **Synthetic тавтологичен**: AI re-predicts свою же η-формулу + добавляет residual
  budget, которого нет в back-calculated actual → на synthetic median Δ ≈ +95…+118%,
  `hypothesis_met=False`. Реальная −10% валидация — только на real `heats.db`.
- **CI = between-heat вариативность**; per-heat conformal p10/p90 — отдельный канал
  неопределённости (см. S1).
- **Comparison в pure-Al-эквиваленте** (charge-форма → pure через `al_content_pct`).

## Verification

- `scripts/smoke_test_shadow.py` — E2E chain S1→S2→S3 на synthetic в tmp, БЕЗ
  `ANTHROPIC_API_KEY`. Проверяет механику (пайплайн отрабатывает, отчёт честен:
  disclaimers + `is_synthetic`), НЕ ассертит `hypothesis_met`. PASS.
- `pytest app/tests -k shadow` → 41 passed.
- Full regression `pytest app/tests -q` (без `test_dockerfile_smoke`).
- `ruff check` — clean.

## Известные ограничения / Phase 3

- Synthetic demo не показывает positive savings (тавтология) — нужны real heats.
- Quality gate residual-проверка использует фактический `al_residual_pct`, не
  predicted (на synthetic predicted тоже тавтологичен).
- Sync endpoint limit ~2000 heats (больше → CLI `generate_shadow_report.py`).
- Per-stratum p-values exploratory (без MC коррекции).
- **Phase 3** = production closed-loop (AI-рекомендация = то, что оператор реально
  подаёт), change-management, prospective validation вместо retrospective replay.

## Регуляторика

S1-S5 — каждый под R-001 Feature Development:
architect → developer → reviewer → qa → mlops. Decision Log пишется на калибровке
(tag `eta_al_calibration`); shadow validation report — опт-ин артефакт в `reports/`.
