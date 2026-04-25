---
title: Property+Cost optimization showcase — что есть, что не видно, что развить
date: 2026-04-25
status: in-progress  (EC.1 выполнено, EC.2 / EC.3 — backlog)
---

## Контекст

Пользователь зафиксировал что в продукте «не видел улучшения физических свойств за счёт изменения химического состава стали и оптимизации расходов за счёт легирующих элементов». Это **главный value proposition** оригинального MVP-замысла, и он сейчас не зримо проявлен в UI / отчётах. Идея добавлена в roadmap для рассмотрения и развития.

## Что **уже есть** в коде

| Модуль | Назначение | Готовность |
|---|---|---|
| `app/backend/cost_model.py` | `compute_cost(composition, snapshot)` — €/тонна стали для произвольного состава, через ferroalloy mapping | ✓ работает; покрытые элементы: Mn, Si, Cr, Ni, Mo, V, Nb, Ti, Cu, Al; non-priced: C, P, S, N |
| `data/prices/seed_2026-04-23.yaml` | Snapshot цен ферросплавов (11 материалов, EUR) | ✓ |
| `app/backend/inverse_designer.py` | NSGA-II через pymoo, multi-objective: max property, min cost. **Class HSLADesignProblem — HSLA-only by name and bounds** | частично (заточено под HSLA pipe) |
| Pattern Library C01-C04 | Проверки cost_optimization (устаревший прайс, физически невозможный ferroalloy, некорректный CostBreakdown, элемент не в snapshot) | ✓ |
| Streamlit вкладка «🎯 Дизайн сплава» | UI inverse design с целевым диапазоном свойств + cost mode | ✓ работает на HSLA |
| Decision Log price snapshots | Каждый run сохраняет копию snapshot для audit trail | ✓ |

**Вывод:** инфраструктура для property+cost оптимизации есть, она работает на HSLA, и Pattern Library защищает от типичных ошибок (плохие цены / невозможные смеси).

## Что **не показано** пользователю явно

1. **Нет «before/after» сравнения.** Дизайн tab выдаёт Pareto-фронт кандидатов с σт + cost, но **не сравнивает с какой-то baseline-практикой**. Пользователь не видит «сейчас вы делаете вот так и платите €X, AI предлагает рецепт Y и платит €X−Δ при том же или лучшем свойстве».
2. **Не работает на Agrawal NIMS** (real-data модели). `inverse_designer.py` параметризован под `VARIABLE_BOUNDS_HSLA` и предполагает HSLA-feature-space. Карбюризованные / пружинные стали не покрыты.
3. **Cost showcase не виден в pitch / документации.** Pitch упоминает CBAM и €-savings, но в коде нет deliverable который рендерил бы конкретные числа «−€18/т vs baseline».
4. **Mechanism transparency слабая.** Когда AI предлагает рецепт, неочевидно «какой именно ferroalloy substitute дал экономию» — пользователь видит итоговый cost но не диаграмму вкладов. Это есть в `CostBreakdown` (структура с пер-материал contributions), но в UI не выводится понятно.

## Почему это критично

Это **не cosmetics, а core product proof**. Без явной демонстрации property+cost trade-off вся система выглядит как «ML-калькулятор», а не как «инструмент экономической оптимизации рецептов». Все остальные AI-фичи (A2 hypothesis generator, B1 symbolic regression и т. п.) — **дополнительные** к этой основной обещанной возможности. Если её не видно — pitch неправдив.

## Предлагаемое развитие — три уровня

### Уровень EC.1 — быстрая демонстрация (DONE)

**Standalone скрипт** `scripts/show_property_cost_on_agrawal.py`. Берёт обученную Agrawal-модель, определяет baseline как медиану training data (или sub-class median для carbon_low_alloy), запускает NSGA-II через pymoo с двумя целями:
- максимизировать predicted fatigue_strength_mpa
- минимизировать €/т через `cost_model.compute_cost`

Над композиционным пространством из 9 элементов + 3-4 ключевых процесс-параметра, в bounds из `meta.json.training_ranges`.

Output — таблица «Топ-5 AI-предложений vs baseline» с колонками: Δfatigue (+МПа), Δcost (−€/т), что изменилось в составе, какой ferroalloy substitute. Сохраняется в `docs/property_cost_demo_<model>.md` (gitignored).

**Это закрывает «не видел нигде»** — даёт конкретные числа на Agrawal real-data модели, которые можно показать.

### Реальные числа от первого запуска EC.1

Модель: `fatigue_fatigue_strength_xgb_20260424_233914` (Agrawal NIMS 437 records, R²=0.978, conformal-coverage 0.92).

**Baseline (медиана carbon_low_alloy sub-class из 338 records):**
- Прогноз σf: **530 МПа** (90% CI [485, 559])
- Стоимость ferroalloy: **452.66 €/т**
- Состав: Si 0.26 / Mn 0.75 / Ni 0.06 / Cr 0.74 / Cu 0.06 / Mo 0.00

**Топ-5 AI-предложений по `Δσf − Δcost`:**

| # | Δσf, МПа | Δcost, €/т | Прогноз σf (90% CI) |
|---|---:|---:|---:|
| 1 | +181 | −28.00 | 712 [464, 989] |
| 2 | +188 | −20.71 | 719 [470, 989] |
| 3 | +189 | −19.85 | 719 [471, 989] |
| 4 | +187 | −21.22 | 718 [464, 989] |
| 5 | +183 | −25.61 | 713 [471, 989] |

**Лучший candidate (#1):** Δσf = +181 МПа, Δcost = −28 €/т → **−2800 €/партия 100 т**.

Что AI меняет (ranked by economic impact):
- **Ni: 0.06 → 0.01 wt%** → −€19.84 экономия (FeNi с €12/кг — самый дорогой здесь)
- **Mn: 0.75 → 0.38 wt%** → −€8.30 (FeMn-80, средняя цена)
- **Cu: 0.06 → 0.01 wt%** → −€4.21
- **Si: 0.26 → 0.37 wt%** → +€3.12 (доплата за раскисление)

**Чистый сдвиг рецепта:** уменьшить дорогие легирующие (Ni/Cu) которые в этом диапазоне дают marginal hardenability boost; уменьшить избыточный Mn (промотирует MnS inclusions, вред для усталости); компенсировать дешёвым Si (solid-solution strengthening, raise hardenability via Si-effect on tempering).

### Honest caveats

- **CI на predicted 712 широкий** — [464, 989], размах 525 МПа. Это conformal-corrected calibrated interval, и он честно сигнализирует: AI вышел в редко-сэмплированный угол композиционного пространства, модель не уверена. В production-flow здесь должен сработать OOD-детектор и снизить confidence явно.
- **Δcost учитывает только ferroalloy** (Mn, Si, Cr, Ni, Mo, Cu). Не входит: рабочее время, энергия, накладные. Реальная маржа на партии может отличаться в обе стороны.
- **Baseline = sub-class median**, не «обычная заводская практика». При работе с реальным заводом нужно подавать их собственный recipe как baseline.

### Уровень EC.2 — UI integration (~1 день)

Расширить вкладку «🎯 Дизайн сплава» так, чтобы она:
1. Принимала **baseline recipe** (или auto-derive из training data centroid)
2. Запускала inverse design параметризованно для **любого SteelClassProfile** (не только HSLA)
3. Рендерила **табличный результат** с Δproperty + Δcost + ferroalloy-mass-shift per candidate
4. Включала **CostBreakdown waterfall chart** — где видно «−€10 на FeMn (увеличили), +€3 на FeNb (уменьшили), итого −€7/т»

### Уровень EC.3 — генерализация inverse_designer на любой класс (~2-3 дня)

`HSLADesignProblem` рефакторится в `SteelClassDesignProblem(profile, model, cost_snapshot)` где bounds, target, и feature engineering берутся из активного `SteelClassProfile`. Это превращает существующий HSLA-only дизайн в class-agnostic фичу. После этого вкладка «Дизайн» работает для всех 3 production-моделей одинаково.

## Связь с B2 active learning из основного roadmap

Изначально B2 формулировался как «Bayesian optimization для выбора эксперимента с максимальным information gain». Идея пользователя по property+cost фактически переплетается с B2: правильное active learning должно искать не просто info-gain, а **info-gain в зоне максимальной economic value** (где Δproperty/Δcost наибольший). Это называется **cost-aware Bayesian optimization** — стандартный подход в materials informatics последних 3-4 лет.

**Предложение:** объединить B2 и эту идею. Конечный B2-deliverable будет: «следующий эксперимент с максимальным ожидаемым (Δproperty − λ·Δcost) при текущей uncertainty модели». Это автоматически даёт `(Δproperty, Δcost)` как explicit output, и закрывает обе цели.

## Решение по последовательности

1. **Сейчас:** EC.1 — standalone демо на Agrawal. ~30 минут. Закрывает «не видел нигде».
2. **После approval демонстрации:** B2 в объединённой формулировке (cost-aware Bayesian active learning).
3. **EC.2 + EC.3** (UI integration + generalization inverse_designer) — отдельный work item, делается когда начнётся работа с реальным внешним пользователем.

## Открытые вопросы

- На Agrawal target — fatigue strength, не yield. Является ли «cost-saving для carbon/spring/cementation steel» столь же интересным сценарием как для HSLA pipe? Скорее да: shafts/gears/springs — большие производственные объёмы, оптимизация состава при сохранении fatigue performance даёт реальный €-возврат. Будет проверено в EC.1.
- Baseline definition: median training data ИЛИ centroid одного sub-class (`carbon_low_alloy` 338 records, `spring` 51, `carburizing` 48). Первый вариант общий, второй более точный per use-case. EC.1 попробует оба.
