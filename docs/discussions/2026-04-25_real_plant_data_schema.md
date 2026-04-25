---
title: Что подаётся СЕЙЧАС vs что просить у реального завода
date: 2026-04-25
status: decided
---

## Контекст

Перед outreach к R&D-инженеру / metallurgy-academic нужно чётко знать, какие данные мы имеем право просить — чтобы не запрашивать то, чего на заводе физически нет, и не упускать то, что есть.

## Что подаётся в систему СЕЙЧАС

В проекте живут три `SteelClassProfile` (data/steel_classes/*.yaml), каждый со своей schema.

### `pipe_hsla` (синтетика, демо)
- **Признаки (16):** 14 элементов состава wt% (C, Si, Mn, P, S, Cr, Ni, Mo, Cu, Al, V, Nb, Ti) + N в ppm + 2 параметра прокатки (`rolling_finish_temp`, `cooling_rate_c_per_s`)
- **Targets:** `yield_strength_mpa, tensile_strength_mpa, elongation_pct, kcv_neg60_j_cm2`
- **Служебные:** `heat_date, campaign_id`

### `en10083_qt` (синтетика, демо)
- **Признаки (10):** 6 элементов (C, Si, Mn, P, S, Cr) + 4 параметра термообработки (`austenitizing_temp, tempering_temp, tempering_time_min, section_thickness_mm`)
- **Targets:** `hardness_hrc, tensile_strength_mpa`

### `fatigue_carbon_steel` (РЕАЛЬНЫЙ Agrawal NIMS, 437 плавок)
- **Признаки (25):**
  - 9 элементов: C, Si, Mn, P, S, Ni, Cr, Cu, Mo
  - 12 параметров термообработки: normalizing/through-hardening/carburizing/diffusion/quenching/tempering — каждый с T + time + cooling_rate
  - 1 параметр прокатки: `reduction_ratio`
  - 3 неметаллических включения: `inclusion_area_defect_a/b/c` (по NMI rating)
- **Target:** `fatigue_strength_mpa`
- **Служебные:** `heat_id, heat_date, campaign_id, sub_class, source, source_doi`

### Что подаётся в Hypothesis Generator (LLM-вход)
Не raw плавки, а **сводка обученной модели**: метрики, feature_importance top-10, training_ranges по каждому признаку, 5 sample predictions, распределение target. Производное от training data — отдельно собирать не нужно.

## Что реально есть у металлургического завода

### Tier 1 — ВСЕГДА есть

**Сертификаты плавок** (EN 10204 3.1/3.2 / Werkszeugnis / Product Certificate):
- Heat ID + дата плавки/смена
- Химия из спектрометра — 8-15 элементов: C, Si, Mn, P, S всегда; плюс Cr, Ni, Mo, Cu, Al, V, Nb, Ti если применимо
- Marka стали (стандартное обозначение)
- Стандартные mech-тесты — один комплект на плавку:
  - Предел текучести (Rp0.2 / ReH), MPa
  - Предел прочности (Rm), MPa
  - Удлинение (A%)
  - Ударная вязкость Charpy (KV at temperature)
  - Твёрдость (HV/HRC/HB)

**Формат поставки:**
- **PDF-сертификаты** — самый частый случай для старых заводов. Требует парсинга/OCR.
- **CSV/Excel из MES** — одна строка = одна плавка. Лучший случай.
- **Прямой DB-доступ** к ERP (Oracle / SQL Server) — редкий, только при серьёзном engagement.

### Tier 2 — есть, но требует усилий

**Параметры прокатки и термообработки:**
- Reheat / roughing / finishing temp, cooling strategy, reduction ratio
- Quench medium + temperature, tempering profile

Хранится в Level-2 process historians (PIMS, IBA, Honeywell PHD, OSIsoft PI) как time-series. Для ML нужны агрегаты — отдельный extraction job.

**Неметаллические включения (NMI rating):**
- ASTM E45 method A/D, DIN 50602 K-method
- Есть для ответственных применений (валы, рельсы, aerospace). Для строительной стали — часто нет.

**Микроструктура:**
- Optical metallography (по запросу)
- Phase fractions, grain size (ASTM grain number)
- EBSD/XRD — только если был research-проект на этой плавке

### Tier 3 — обычно нет, но иногда есть

- Continuous time-series: температурные кривые ladle→casting→rolling→cooling, dilatometry
- Fatigue data (S-N curves) — НЕ рутинно, только спец-заказы
- Fracture toughness (K_IC, J_IC) — спец-applications
- Creep — для energy/aerospace
- Список отбракованных плавок и причины (sensitive — заводы делятся неохотно)

### Tier 4 — этого нет

- **Per-heat economics:** расход сырья, energy cost, time-to-recipe в € на плавку. Заводы считают per-shift/per-month. **Восстанавливать самим** через типовые ставки.
- **In-service performance:** как стальная труба прослужила 10 лет — у end-customer, не у завода.
- **Готовый ML-ready dataset «прямо сейчас отдадим»:** никто не выгружает 10000 плавок одним кликом. ~1-2 недели data engineering на их стороне.

## Минимальный viable schema для первого pilot

Если завод спрашивает «что вам нужно?» — реалистичный запрос:

**500-2000 плавок одной марки стали из последних 1-3 лет**, в Excel/CSV:

| Колонка | Что | Обязательно? |
|---|---|---|
| `heat_id` | уникальный идентификатор плавки | да |
| `heat_date` | дата плавки (для time-based split) | да |
| `grade` | марка стали (для фильтрации) | да |
| `c_pct, si_pct, mn_pct, p_pct, s_pct` + по марке | состав wt% | да |
| process params (T_reheat, T_finish, cooling, Q_medium, temper) | минимум 2-3 параметра | да |
| `yield_strength_mpa` или `tensile_strength_mpa` или `hardness_*` | один target | да |
| `section_thickness_mm` или `coil_id` | геометрия | желательно |
| `inclusion_rating` | если есть | optional |

Этого хватает на: первое demo + pilot model + первый Hypothesis Generator run + cost-optimization tab. Всё остальное — opt-in после.

## Подводные камни при интеграции реальных данных

### Naming convention drift
`c_pct` придёт как: `C`, `C%`, `C [%]`, `C_wt%`, `Carbon`, `pct_C`, `ELEMENT_C`. На русских заводах — кириллицей. Японские — иероглифы. **Нужен mapping wizard в UI** — пользователь сопоставляет свои колонки нашим именам.

### Единицы
- Yield в ksi, не MPa
- Температура в °F
- Размеры в inches
- Нужны converters.

### Missing values
Не каждая плавка имеет все 25 признаков. Какие-то без Charpy, какие-то без full chemistry. **Pipeline сейчас строго требует все feature_set** → нужна graceful imputation или per-feature optional.

### Multi-specimen heats
Одна плавка → несколько проб разной толщины → несколько mech-результатов. Сейчас один `heat_id` = одна строка. Реальный завод даст несколько строк на плавку. Решение: либо агрегация (mean / median по плавке), либо переход на specimen-level grain (heat_id остаётся группой для GroupKFold).

### Confidentiality
- Завод может дать анонимизированные heat_id, но без grade designation (раскрыло бы product mix)
- Может дать химию + tests, но без process параметров (раскрыло бы know-how)
- Нужно работать с любой комбинацией, gracefully degrading model quality по доступности фичей

## Что это значит для продукта (next 1-3 коммитов вперёд)

В рамках текущей AI roadmap (см. `2026-04-25_ai_integration_roadmap.md`):
- **A1 (LLM feature discovery)** должен генерировать предложения features из доступных колонок (не предполагать что user даст все 25 как у Agrawal)
- **B2 (active learning)** должен учитывать что каждый experiment = одна реальная плавка ≈ €5-15k
- **A2.3 UI tab** уже сейчас готов работать с любым активным `SteelClassProfile`, но в будущем нужна вкладка «Загрузить свой dataset» с mapping wizard

Это explicit feature backlog, а не текущая работа. Текущая — A2.3 UI на existing моделях.

## Сводка для outreach

**Просить:** CSV с 500-2000 плавок одной марки. Столбцы = `heat_id, date, grade, состав wt%, 2-3 параметра процесса, минимум 1 mech target`. Формат Excel/CSV. **Это есть у всех.**

**Не просить (без доп. оснований):**
- per-heat economics в €
- in-service performance конечного продукта
- полные time-series процессов
- "готовый clean ML-ready dataset"

**После первого pilot — opt-in:**
- inclusion ratings
- micrography
- process telemetry агрегаты
- failure / reject data (sensitive — обсуждать отдельно с NDA)
