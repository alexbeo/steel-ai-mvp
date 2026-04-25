---
title: Deoxidation Advisor + PhD Critic — first live cycle
date: 2026-04-25
status: decided
verdict: PASSED — критик поймал арифметическую ошибку в evidence + неверный recovery factor
---

## Контекст

Per user request «улучшить Раскисление жидкой стали алюминием с использованием Anthropic Sonnet до уровня в 10 раз лучше текущего». Реализация: `app/backend/deoxidation_advisor.py` + `deoxidation_critic.py` + UI sub-tab «🤖 AI-советник + критик» в вкладке «🔥 Раскисление».

**Что было до:** 3 thermodynamic числа (Fruehan / Sigworth-Elliott / Hayashi-Yamamoto) и сравнительный график.
**Что стало:** complete operator protocol с PhD peer review.

## Параметры live запуска

Реалистичный кейс carbon-steel ковша:
- **Heat:** 100 т, T=1575°C, O_a 320→5 ppm
- **Composition:** C=0.20, Mn=0.65, Si=0.30, S=0.018, P=0.020 wt%
- **Slag FeO:** 3.5% (borderline)
- **Grade:** S355 carbon construction steel
- **Mn/S ratio:** 36 (ниже безопасного порога 50)

## Thermo baseline

| Модель | Al kg |
|---|---|
| Fruehan 1985 | 44.4 |
| Sigworth-Elliott 1974 | 44.4 |
| Hayashi-Yamamoto 2013 | 44.4 |

Spread 0% — модели сходятся, Mn=0.65 ниже high-Mn correction threshold.

## Advisor output (Sonnet PhD ladle metallurgist)

| Параметр | Значение |
|---|---|
| Al recommendation | **63 kg** wire (vs 44.4 kg theoretical) |
| Recovery factor | 70% |
| Confidence | HIGH |
| Kinetic timing | 6-12 мин |
| Risk flags | 6 штук |
| Pre-actions | 5 шагов |
| Post-actions | 5 шагов |
| Evidence | 4 строки с цитатами Turkdogan, Cramb, Cicutti, Ghosh |

**6 risk flags выявлено:**
1. Slag FeO=3.5% — пограничное (требует мониторинга, при >4% pre-deox SiMn)
2. **Mn/S=36 ниже безопасного 50** — риск MnS-стрингеров, нужен Ca-treatment
3. O_a=320 ppm высокая — обязательна двухэтапная подача
4. T=1575°C умеренная — кинетика замедлена, holding ≥8 мин
5. C=0.20 wt% — учесть Fe-coating wire
6. Mass 100 т большой — проверить wire feeder скорость

**Inclusion forecast:** Al2O3 2-10 мкм с gamma→alpha переходом; для S355J2+N вариант рекомендует Ca-treatment 0.15-0.20 kg/т CaSi для перевода в 12CaO·7Al2O3.

**Pre-actions** включают: измерение O_a, оценку slag FeO с pre-deox SiMn если >4%, проверку wire feeder, argon stirring 0.4 м³/мин, контроль T.

**Post-actions** включают: argon stirring 8-10 мин, sampling [Al]sol target 0.025-0.045 wt%, conditional Ca-treatment для S355J2+N grade, снижение stirring перед разливкой, SEN с Ar-продувкой против re-oxidation.

## Critic verdict: REVISE (HIGH confidence)

Критик выполнил 9 evidence-checks: **6 VALID, 2 INVALID, 1 UNVERIFIABLE.**

### Поймана #1: арифметическая ошибка в advisor evidence

> «35.4 + 30 = 65.4, а не ~44 kg. Этот evidence-расчёт не сходится с итоговым значением 44.4 kg из моделей — внутреннее противоречие в тексте advisory.»

Advisor написал:
- стехиометрия: 315 ppm O × (54/48) × 100 т = 35.4 kg Al
- residual: [Al]sol 0.030 wt% × 100 т = 30 kg
- итого: 35.4 + 30 = **65.4** (но advisor написал «~44»)

Критик корректно нашёл: чтобы итог был 44.4 kg, target [Al]sol должен быть ~0.009 wt% (≈9 ppm), не 0.030 wt%. Это **самопротиворечивое** evidence в advisor output.

### Поймана #2: recovery factor занижен

Advisor применил поправку «−3.5 п.п. на slag», получив 70%. Критик нашёл:
- Cramb (ISIJ 2004): снижение 3-4 п.п. **за каждый 1% FeO сверх 2% базового**, т.е. при FeO=3.5% коррекция (3.5−2.0)×3.5 = **−5.25 п.п.**, не −3.5
- Однако wire penetrates slag → snижение менее чувствительно
- Реалистичная оценка: **72-76%** (центр 74%), не 70%

Перерасход: 44.4/0.70 = 63.4 kg vs 44.4/0.74 = 60.0 kg. Разница 3.4 kg.

> «Создаёт риск [Al]sol > 0.045 wt% (выше target-диапазона 0.025-0.045 wt%).»

Suggested revision: «Пересмотреть recovery до 73-75% и снизить рекомендацию до 59-61 kg.»

### Поймана #3: UNVERIFIABLE citation

Advisor написал: «при FeO>4% recovery <65% для wire». Критик: claim требует конкретной ссылки которой нет в advisor evidence — flagged как UNVERIFIABLE.

### Что критик подтвердил (VALID)

- 3 thermo моделей сходятся 0% spread, Mn<1.0 порог HY-коррекции корректно опознан
- Подача 0.375 kg/с ≤ Cicutti et al. (2001) предел 0.4 kg/с
- Kinetic timing 6 мин по Ghosh (2001) — реалистично
- Mn/S=36 ниже EN 10025-2 практического порога 40
- Inclusion morphology + Ca/Al ratio chemistry — расчётно согласуется с CaO-Al2O3 диаграммой

## Operational

- **Cost:** ~$0.20-0.25 за полный цикл (advisor + critic)
- **Latency:** ~3 минуты
- **143/143 unit-тестов** проходят (было 132; +11 новых для двух модулей)
- **Persistence:** Decision Log тэг `deoxidation_cycle` с full snapshot heat + thermo + advisory + review

## Что демонстрирует — реальный «10x better»

| Аспект | До (3 thermo числа) | После (advisor + critic) |
|---|---|---|
| Output | 1 число (Al kg) | Complete operator protocol (13 полей) |
| Evidence | Нет | Цитаты на Turkdogan, Cramb, Cicutti, Ghosh + arithmetic |
| Recovery factor | Не учитывается | 70% (skoректирован критиком до 74%) |
| Risk identification | Только D01-D03 паттерны | 6 конкретных рисков плавки |
| Inclusion forecast | Нет | Al2O3 morphology + Ca-treatment grade-specific |
| Pre/post actions | Нет | 10 actionable шагов |
| Math fact-check | Нет | Critic поймал 65.4 ≠ 44.4 ошибку |
| Audit trail | Decision Log entry | Full advisor + critic verdict snapshot |

## Применимость в реальной работе

**Целевая аудитория** (per traditional steel plant roles):
- Senior ladle metallurgist на крупных заводах класса voestalpine Linz, ArcelorMittal Florange
- Технолог цеха внепечной обработки
- Process engineer responsible for ladle treatment cycle

**Use case в смене:**
1. Operator замеряет O_a → 320 ppm
2. Запускает AI-советник во время holding'а ladle
3. За 3 минуты получает полный protocol с PhD-рецензией
4. Если REVISE — читает suggested_revision и корректирует
5. ACCEPT/REVISE'ed protocol используется для смены

Без AI-советника тот же analysis занял бы 30+ минут совещания со старшим металлургом + math review — а критика на ошибку «35+30=44» вообще можно не заметить в живом обсуждении.

## Verdict

**Deoxidation pair PASSED.** Advisor + critic превратили утилитарный калькулятор в complete decision-support систему уровня ladle metallurgy lead'а с математически проверяемой evidence base.

## Открытые улучшения

- **Auto-revision loop:** при REVISE автоматически перезапустить advisor с suggested_revision'ом до ACCEPT-конвергенции (cost ×2-3, качество выше)
- **Slag chemistry deeper:** ввод полного slag analysis (CaO/Al2O3/MgO ratios) даст ещё более точный recovery prediction
- **Combined deoxidation:** Al + FeSi + Ca в один цикл
- **Calcium treatment dedicated module:** post-Al Ca optimization с inclusion modification physics

Все — backlog. Текущая система production-ready как UI-инструмент.
