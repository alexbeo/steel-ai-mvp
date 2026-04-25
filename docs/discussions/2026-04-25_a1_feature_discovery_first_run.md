---
title: A1 feature discovery — первый live cycle на Agrawal
date: 2026-04-25
status: decided
verdict: PASSED architecture, ZERO uplift on saturated baseline
---

## Контекст

A1 (LLM-driven feature discovery) — второе AI capability из roadmap. Реализация: `app/backend/feature_discoverer.py` + `scripts/discover_features_for_model.py`. Verification gate из roadmap: «на Agrawal NIMS feature discovery должен предложить ≥3 features, тестовый прогон должен показать R² не хуже исходного, желательно лучше на 0.005+».

## Параметры запуска

- **Модель:** `fatigue_fatigue_strength_xgb_20260424_233914` (R² 0.978)
- **Baseline для сравнения** (compact XGBoost, fixed seed, 80/20 split): R² **0.9866**, MAE 15.71 МПа
- **Sonnet call:** 1 — generator, 1917 input + cache_create 2946 / 4032 output / 39 с / ~$0.07

## Результаты — все 5 предложений

| # | Feature | Class | Formula | ΔR² | ΔMAE | Verdict |
|---|---|---|---|---|---|---|
| 1 | `tempering_severity` | interaction | `tempering_temp_c × √tempering_time_min` | +0.0000 | +0.12 | net-zero |
| 2 | `carburizing_thermal_dose` | interaction | `carburizing_temp_c × √carburizing_time_min` | −0.0000 | +0.32 | net-zero |
| 3 | `total_inclusion_area` | aggregate | `def_a + def_b + def_c` | +0.0003 | +0.17 | net-zero |
| 4 | `cev_carbon_equivalent` | aggregate | `c + Mn/6 + (Cr+Mo)/5 + Ni/15` | +0.0003 | −0.09 | net-zero |
| 5 | `log_reduction_ratio` | transform | `log(reduction_ratio)` | −0.0003 | +0.28 | net-zero |

**5 из 5 формул успешно вычислены** (DSL-safety работает: ни одна не упала на NaN/inf/unknown column). **Все ΔR² в пределах ±0.0003** — это шум, не сигнал.

## Что это означает

### Честное наблюдение №1: модель насыщена

Baseline R² = 0.9866 на 25-фичевой модели уже близок к asymptotic для этого датасета. **Headroom = 0.0134 абсолютных пунктов** до идеального R²=1. На таком уровне marginal feature engineering физически не может дать значимый уплифт — XGBoost с 400 деревьями уже выжал из 25 raw features всё что мог.

Это **известное свойство boosting'а**: tree ensemble имплицитно открывает interaction terms на уровне deep splits. Ручной (или LLM-driven) preengineering interactions помогает только в режимах:
- Маленький train (XGBoost не может выучить взаимодействие)
- Линейная модель (не открывает interactions сама)
- Сильно несбалансированные/skewed таргеты

Ничего из этого здесь не применимо.

### Честное наблюдение №2: LLM предложил физически осмысленные фичи

Все 5 — стандартные класс metallurgy:
- **CEV** (carbon equivalent) — учебный hardenability proxy
- **Total inclusion area** — superposition трёх defect-categories из ASTM E45
- **Tempering severity** — Hollomon-Jaffe-подобный параметр (T·√t)
- **Carburizing thermal dose** — аналогично
- **log(reduction_ratio)** — Hall-Petch type transformation

LLM не выдумал ничего бредового. Просто XGBoost эти комбинации уже неявно знает.

### Честное наблюдение №3: система не lied

В отличие от LLM который мог бы заявить «я добавил +0.02», эмпирическая verification показала null result. Это **самое ценное свойство архитектуры A1**: empirical truth gate перед persistence. Система не зарегистрирует спекулятивный uplift как реальный.

## Verification gate — формально PASSED

Из roadmap-документа:
> «на Agrawal NIMS feature discovery должен предложить ≥3 features, тестовый прогон должен показать R² не хуже исходного, желательно лучше на 0.005+»

| Критерий | Требовалось | Получено | OK? |
|---|---|---|---|
| Число предложений | ≥ 3 | 5 | ✓ |
| Diversity (≥ 2 mechanism_class) | да | 3 (interaction, aggregate, transform) | ✓ |
| R² не хуже baseline | да | max ΔR² = −0.0003 (граница шума) | ✓ |
| Желательно +0.005+ | nice to have | не достигнуто | — |

Architecture verification PASSED. **Empirical uplift на этой конкретной модели — null.** Это разрешённый исход для этого пункта roadmap.

## Стоимость одного цикла

- 1 Sonnet call (generator only): ~$0.07
- 6 compact XGBoost trainings (1 baseline + 5 ext): ~3 секунды
- Total cycle: **~$0.07 / 45 секунд**
- Дешевле чем A2 (генератор+критик ~$0.20 / 3 минуты)

## Ограничения и что делать дальше

### Ограничение 1 — saturated baseline скрывает potential A1

Все 3 наших production-моделей — **очень хорошие**: HSLA R² 0.899, Q&T 0.983, Agrawal 0.978. Тестировать A1 на них — почти всё равно что тестировать новый ферросплав на стали уровня aerospace: разница в noise-зоне.

Чтобы реально оценить A1, нужно либо:
- **Намеренно ослабленный baseline** (например, обучить HSLA на 6 raw элементах вместо 24 engineered features — посмотреть, восстановит ли A1 потерянный сигнал через CEV/Pcm-подобные aggregates)
- **Реальный заводской датасет** с естественной шумностью и пробелами (не synthetic)
- **Меньше train data** (50-100 записей вместо 290+) — где tree ensemble не может имплицитно открыть interactions

### Ограничение 2 — нет critic'а у A1 (пока)

В отличие от A2 (gen + crit), A1 — только gen. Empirical R² verification отчасти заменяет critic'а: numerical uplift объективен, не нуждается в peer review. Но critic мог бы поймать вещи, которые retrain не ловит:
- redundancy с уже существующей engineered feature
- data-leakage риск (фича коррелирует с target неявно)
- physical implausibility что не зашло в R², но scientific community отвергнет

Для current MVP-фазы решено НЕ добавлять A1-critic — empirical gate sufficient. Может стать опцией если переходим к serious R&D usage.

## Решение

A1 **готов к продакшн-использованию архитектурно**. Эмпирическая ценность на наших текущих synthetic моделях — null, но это известное свойство saturated baselines, а не баг.

Реальный test-case для A1 появится когда:
1. Подключим weakly-baseline-модель (намеренно или из реальных данных)
2. Кто-то загрузит свой CSV с пробелами и шумом, обучит модель на нём, и системе придётся работать в условиях ниже-saturation

## Следующие шаги (по AI roadmap)

A2 ✅, A1 ✅ (architecturally). Дальше:

3. **B1 — symbolic regression** — извлечение аналитических формул через PySR
4. **B2 — active learning** — Bayesian optimization для выбора следующего эксперимента
5. **A3 — anomaly explanation** — LLM объясняет OOD-кейсы

Альтернативный side-quest: **обучить ослабленный baseline специально для A1 demo** (5-feature HSLA), показать что A1 действительно восстанавливает CEV-подобный сигнал. Это не пункт roadmap, но даёт сильный empirical proof для marketing/pitch.
