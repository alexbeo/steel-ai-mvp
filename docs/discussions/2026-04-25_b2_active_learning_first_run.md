---
title: B2 active learning — first live run на Agrawal
date: 2026-04-25
status: decided
verdict: PASSED architecturally, honest signal про trade-off на этом датасете
---

## Контекст

B2 четвёртый пункт AI integration roadmap. Реализация: `app/backend/active_learner.py` + `scripts/propose_next_experiments.py`.

Acquisition function: **cost-weighted Expected Improvement** (Jones et al. 1998), σ из conformal-corrected 90% CI через нормальное приближение, ranking по `EI / cost`.

Это **третий путь** к property+cost оптимизации в проекте — стохастический математический скан, комплементарный к Sonnet PhD pair (recipe_designer/critic) и NSGA-II Pareto (EC.1 demo).

## Параметры запуска

- **Модель:** `fatigue_fatigue_strength_xgb_20260424_233914`
- **Baseline:** carbon_low_alloy median (338 records), σ_pred=530 МПа, cost=452.66 €/т
- **f\*** (max-observed-training): **1190 МПа** (карбюризованная плавка из dataset)
- **LHS sampling:** 2000 точек по 12 decision_vars (6 priced compositions + 6 ключевых process)
- **Top-K:** 5

## Результат

| # | EI/cost | EI | σ_pred | cost €/т | Δσ vs base | Δcost |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.0004 | 0.3 | 731 | 650.27 | +200 | +197 |
| 2 | 0.0004 | 0.3 | 744 | 837.60 | +213 | +385 |
| 3 | 0.0004 | 0.3 | 718 | 762.98 | +188 | +310 |
| 4 | 0.0004 | 0.2 | 715 | 643.63 | +184 | +191 |
| 5 | 0.0003 | 0.3 | 734 | 970.86 | +203 | +518 |

Все 5 кандидатов **non-OOD**.

## Что показывает результат

### EI малый (~0.2-0.3 МПа·кг)

Это потому что **f\* выбран как максимум в training data (1190 МПа)** — крайнее значение карбюризованной плавки. От baseline 530 МПа до f\* — 660 МПа разрыв. LHS-скан почти не находит точек, чьё (μ + σ) приближается к 1190 МПа без специфической карбюризации.

Это **honest signal**: модель не считает что произвольная композиция в feasible space может приблизиться к экстремальному правому хвосту. **Карбюризация — главный driver** (мы уже видели в feature_importance: 0.475). Без неё расти выше ~750 МПа сложно.

### Все top-5 предлагают +200 МПа за счёт роста cost +€200..+€500/т

Кандидаты с большим EI имеют большую μ — то есть модель предсказывает существенный gain. Но рост свойства требует усиленного легирования, что повышает cost. Сортировка по `EI/cost` баланcирует, но в этой зоне space наклон trade-off `Δσ : Δcost ≈ 1 : 1.5` (за каждый +1 МПа модель просит +€1.5/т).

Это **другой trade-off** чем в EC.1 demo (где cost-saving преобладал) или recipe-pair (где critic нашёл рецепт −€18/т при −6 МПа). Active learner — exploratory, идёт **вверх по property**, не вниз по cost. Это управляется через выбор f\*: если задать f\* = baseline + 30 МПа (small target), top-K сместится в зону умеренных улучшений с экономией.

### Calibrated CI confirms predictions

Все top-5 имеют узкий 90% CI (±50-80 МПа вокруг μ) — модель уверена в своих прогнозах в этих точках. Это благодаря conformal-corrected calibration.

## Verification gate из roadmap

> «на Agrawal с искусственно retained 50-record subset, active learning должен достигать R² ≥ 0.95 за меньшее число итераций чем random sampling»

Этот benchmark **не выполнен в текущей итерации** — он требует отдельной симуляции (retain → AL pick → retrain → measure). Это backlog. Текущий B2 проверил **architecture** (LHS + EI + cost ranking + persistence работает), но не **benchmark vs random**.

Architecture verification: PASSED.
Empirical AL-vs-random benchmark: backlog.

## Operational

- **Cost:** $0 (no LLM)
- **Latency:** ~150 мс (LHS sample 2000 + 2000 model predictions + 2000 cost computes)
- **126/126 unit tests** проходят
- **Persistence:** Decision Log тэг `active_learning` с full proposals + baseline + f\* + n_samples

## Ограничения

1. **EI с extreme f\*** даёт малые значения. Можно использовать `min(observed)` вместо `max` для cost-minimization scenarios, или quantile-based f\* (95-th percentile) для balanced.
2. **LHS не учитывает constraint relationships** между decision_vars (например, при carburizing_temp=30 значит нет carburizing → carburizing_time должен быть тоже 0). Это можно добавить через rejection sampling или custom sampler.
3. **Без Sonnet-объяснения** для каждого кандидата — это чисто numerical. Можно дополнить: топ-3 по EI/cost запросить у Sonnet rationale-narrative. Это будет AL+LLM-hybrid вариант.

## Развилки

- **Tighter f\* selection:** позволять пользователю выбирать f\* как `target_property` (не max-observed). Это даёт «направленный поиск» — «найди эксперимент чтобы достичь 600 МПа дёшево».
- **Per-experiment Sonnet rationale:** для top-K кандидатов по AL вызвать recipe_designer chain — получить metallurgical narrative + critic. Это даёт AL+PhD hybrid.
- **AL benchmark vs random:** retained-records симуляция как verification от roadmap.

Backlog. Текущая архитектура работает.

## Verdict

B2 работает как третий numerical путь к property+cost оптимизации. Дополняет, не заменяет, recipe pair и NSGA-II Pareto. Каждый из трёх путей даёт ценность для разной аудитории / контекста:

| Метод | Сильная сторона | Аудитория |
|---|---|---|
| Recipe pair (Sonnet PhD) | Reasoning + evidence + critique | R&D engineer, single recipe |
| EC.1 NSGA-II Pareto | Trade-off frontier | ML scientist, full landscape |
| B2 Active Learning | Ranked next experiments | R&D, queue planning |
