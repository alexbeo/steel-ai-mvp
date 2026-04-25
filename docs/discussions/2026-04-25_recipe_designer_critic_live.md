---
title: Recipe Designer + PhD Critic — first live cycle на Agrawal
date: 2026-04-25
status: decided
verdict: PASSED — критик ловит mechanism inversions и cost errors на порядок
---

## Контекст

Реализация запроса пользователя «для подбора химии использовался Anthropic Sonnet с оппонентом и критиком уровня PhD через промпт и доказательной базой». Архитектура: `recipe_designer` (proposer) → ML+cost numerical truth gate → `recipe_critic` (PhD adversarial review with explicit evidence fact-check).

## Параметры запуска

- **Модель:** `fatigue_fatigue_strength_xgb_20260424_233914` (Agrawal NIMS 437 records)
- **Baseline:** медиана `carbon_low_alloy` sub-class из 338 записей
- **Sonnet calls:** 2 (designer + critic, оба claude-sonnet-4-6)

| Метрика | Designer | Critic |
|---|---|---|
| Latency | 81 с | 104 с |
| Total cycle | — | **3 минуты 5 секунд** |
| Cost (приближённо) | ~$0.10 | ~$0.10-0.13 |
| Total cycle cost | — | **~$0.20-0.25** |

## 4 рецепта от designer'а — что предложил Sonnet

| # | Стратегия | Novelty | Composition shifts |
|---|---|---|---|
| 1 | Ni-Cu обнуление + Mn-буст | LOW | Mn 0.75→1.05, Si 0.26→0.40, Ni→0, Cu→0 |
| 2 | Cr↓ + Mo↑ trade | MEDIUM | Cr 0.74→0.40, Mo 0→0.12 |
| 3 | T_norm uplift + Ni/Cu/Mo zero | MEDIUM | Ni→0, Cu→0, T_norm 870→910 |
| 4 | Si-buster + cost-min | MEDIUM | Si 0.26→0.55, Mn→0.90, T_temp ↓ |

## ML+Cost truth gate result

| # | predicted σf | 90% CI | Δσf | Δcost | OOD |
|---|---|---|---|---|---|
| 1 | 524 МПа | [477, 557] | **−6** | **−18.10** | нет |
| 2 | 510 МПа | [481, 552] | −20 | **+55.63** | нет |
| 3 | 529 МПа | [489, 560] | −2 | **−23.43** | нет |
| 4 | 591 МПа | [482, 688] | **+60** | −14.11 | нет |

## Critic verdicts — что критик нашёл

### #1 ACCEPT (HIGH) — производственный кандидат

Все 4 строки evidence получили **VALID** в evidence_check:
- artifact-cite по feature_importance — точное совпадение с payload (0.01607 = rank #4)
- mechanism-cite по Pickering Mn-coefficient (~32 МПа/wt%) — стандартный для феррито-перлитных сталей
- mechanism-cite по Grossmann DI multipliers — корректные числа

ML-верификация: delta_cost=−18.10 (обещано 12-18) и delta_property=−6 (обещано −10..+15) — оба попадают в заявленные авторские диапазоны.

Strengths критика:
- «Все параметры внутри training_ranges; ood_flag=false; CI [477-557] не аномально широкий»
- «Риск MnS и рост CEV явно упомянуты и количественно ограничены (S≤0.008 wt%)»

Единственное замечание: «полезно добавить референсные марки (30Mn2 / SAE 1330) для индустриальной валидации позиции».

### #2 REJECT (HIGH) — пойманы 3 ошибки

**Cost calculation error на порядок по знаку.** Автор обещал −4..−8 €/т экономии. Критик через ML cost_model насчитал **+55.6 €/т** (рецепт **на 55 евро дороже** baseline). Конкретно: FeMo при +0.12 wt% Mo даёт +68 €/т (FeMo ≈ €34/кг — самый дорогой ferroalloy), тогда как экономия от снижения FeCr-HC всего −13.6 €/т. Автор не учёл реальную стоимость FeMo.

**Mechanism inversion поймана:** «Mo2C secondary hardening при T_tempering=600°C даёт +20-35 МПа» — корректно для high-alloy tool steel (Cr>4%, Mo>2%), но **неприменимо** при Mo=0.12 wt% в нормализованной (без закалки) carbon steel. ML-предсказание (−20.4 МПа) эмпирически опровергает.

**Argument irrelevance:** Grossmann DI применён к рецепту, в котором отсутствует через-закалочный цикл (carburizing_temp_c=30, carburizing_time_min=0). Прокаливаемость irrelevant для нормализация+отпуск термомаршрута.

«Рецепт доминируемо хуже baseline по обоим KPI» — точный вывод.

### #3 REVISE (HIGH) — self-contradiction в evidence

Critic нашёл: автор в evidence ссылается на «Hall-Petch: рост T_norm → более мелкое зерно → +15-25 МПа», но **в собственном risk_notes** упоминает риск роста зерна при отсутствии TiN/AlN-пиннинга. Это explicit self-contradiction.

ML truth gate: автор обещал +20..+45 МПа, ML дал **−1.6 МПа** (essentially нейтральная прочность). При этом cost economy −23 €/т максимальная среди всех — рецепт остаётся ценным.

Suggested revision: «Скорректировать expected_outcome: убрать обещание σf≥baseline и сформулировать как нейтральная прочность при экономии −23 €/т. В evidence снять Hall-Petch как source of gain. Если цель именно Hall-Petch — добавить микролегирование Ti (0.020-0.030 wt%) для пиннинга зёрен. Заменить novelty на LOW.»

### #4 REVISE (MEDIUM) — статистический и методологический stress-test

Critic нашёл несколько проблем:

**CI uncertainty:** [482, 688] = размах 205 МПа, в 3-4 раза шире остальных рецептов. «Точечное предсказание +60.3 МПа нельзя считать надёжным» — рецепт требует физической валидации малой серией перед production.

**Pickering coefficient misapplied:** автор написал «+37 МПа на 0.1 wt% Si» применённое к σ_y феррита, потом транслировал на σ_fatigue напрямую. Critic: «Прямая трансляция Δσ_y феррита = Δσ_f методологически некорректна… Сталь fatigue_carbon_steel в нормализованном состоянии не является феррито-перлитной мягкой сталью.»

**Numerical error в HJP:** автор посчитал ΔHJP ~400 единиц, корректное значение ~315. Не критическая ошибка, но reviewer #2 такое замечает.

**Importance contradiction:** si_pct importance = 0.00168 (наименьший в top-10), что эмпирически противоречит заявленному большому механистическому вкладу Si.

Suggested revision: «Явно указать широкий CI [482-688] как ограничение… Исправить механизм Si: убрать прямую трансляцию Pickering σ_y → σ_f… Уточнить HJP до 315… Добавить в risk_notes контроль S≤0.008 wt% при Mn=0.90 wt%.»

## Что демонстрирует — реальный PhD-уровень

Critic показал семь компетенций реального journal reviewer:

| Компетенция | Доказательство в этом цикле |
|---|---|
| Cost-model fact-check | #2: пойман +55.6 vs обещано −4..−8 (ошибка на порядок) |
| Mechanism inversion catch | #2: Mo2C secondary hardening только для tool steel |
| Mechanism applicability bounds | #2: Grossmann DI irrelevant без закалочного цикла |
| Self-contradiction detection | #3: Hall-Petch claim contra собственный risk_notes |
| Statistical confidence calibration | #4: CI 205 МПа flagged как red |
| Methodological precision | #4: Pickering σ_y → σ_f некорректная трансляция |
| Numerical fact-checking | #4: HJP 315 ≠ автор-claimed 400 |

Каждое замечание — **конкретная цитата из артефакта или известного закона**, не вкусовое мнение.

## Что демонстрирует — рецепт #1 как готовый production-кандидат

После критика остался один ACCEPT-рецепт с **прозрачной доказательной базой**:

- **Сдвиги:** Mn 0.75→1.05 wt%, Si 0.26→0.40, Ni 0.06→0.01, Cu 0.06→0.01
- **Δσf = −6 МПа** (predicted 524, baseline 530, в допуске ±30)
- **Δcost = −18.10 €/т** (cost 434.56 vs baseline 452.66)
- **На партии 100 т = −1800 €**
- Все 4 строки evidence verified VALID
- OOD safe, CI узкий [477-557]
- Risk notes явно покрывают MnS-control при S≤0.008 wt%

Это и есть запрошенная пользователем **«максимально точная рекомендация уровня PhD через промпт и доказательной базой»**: композиция, evidence, ML-truth-gate подтверждение, и независимое adversarial review всё видно одним пакетом.

## Operational metrics

- Total cycle: **3 минуты 5 секунд** (designer 81с + ML/cost 0с + critic 104с)
- Cost: **~$0.20-0.25** при первом запуске; на повторных с прогретым cache ~$0.13
- Persistence: единая запись в Decision Log с тагом `recipe_cycle`, плюс отдельные `recipe_design` и `recipe_review` для audit
- Tests: 13 unit-тестов с моками, полный suite 119/119 (was 106; +13)

## Verdict

Архитектура **достигла цели пользователя**. Sonnet PhD pair даёт:
1. Расчёт-обоснованных рецептов с двойной evidence (artifact + mechanism)
2. ML+cost numerical truth gate ловит галлюцинации в ожиданиях
3. Independent PhD adversarial review с **построчным fact-check evidence**

Каждый production-рецепт сопровождается прозрачной картиной слабых мест — readers может оценить риски сами, не на доверии к LLM.

## Открытые улучшения

- **Revision loop**: автоматический re-run designer с suggested_revision'ами от critic'а до ACCEPT-конвергенции. Cost ×2-3, но качество финальных рецептов выше.
- **UI integration**: Streamlit-вкладка «🧪 Подбор рецепта» с baseline input + run-button + рендером карточек рецептов с evidence-check colored marks и финальной recommendation.
- **Multi-seed ensemble**: запустить designer N раз с разными seed, собрать union recipes, ранжировать по доле ACCEPT vs REJECT (proxy качества предложений).

Все три — backlog. Текущая архитектура production-ready как CLI-инструмент.
