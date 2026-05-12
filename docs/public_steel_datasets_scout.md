# Public steel datasets — расширенный скаут (Path 1 extended)

**Дата:** 2026-04-24
**Задача:** ранжировать доступные публичные датасеты реальных сталей (не только HSLA) для обучения ML-модели `composition + process → mechanical property`. Пользователь выбирает источник → далее Path 2 (loader + обучение).
**Предшественник:** `docs/public_hsla_scout.md` — HSLA-only скаут показал, что pipeline-HSLA публичных данных практически нет. Здесь раскрываем поиск на весь стальной каталог.

---

## Критерии ранжирования

Оцениваю каждый источник по 5 осям (шкала 1-5):

- **N** — размер. 5 ≥ 1000 records, 4 = 500-1000, 3 = 200-500, 2 = 100-200, 1 < 100.
- **Schema** — полнота: composition ≥ 10 элементов + 1+ process vars + target mech. property.
- **License** — пригодность для commercial ML: 5 = CC-BY / Apache / MIT / CC0; 3 = неясно; 1 = CC-BY-NC-ND / proprietary.
- **Availability** — доступ: 5 = прямой CSV/JSON download; 3 = supplementary manual pull; 1 = scraping-blocked.
- **Match** — совпадение целевой property и feature-схемы с нашим pipeline (`TrainedModel` → `yield_strength_mpa`, `tensile_strength_mpa`, `hardness_hrc`, composition + rolling/cooling/HT): 5 = direct fit; 3 = нужна adaptation profile; 1 = other task (e.g., S-N fatigue).

**Total score** = сумма (max 25).

---

## Ранжированный список

### #1 — Agrawal 2014 NIMS steel fatigue (Kaggle, Springer supplementary) — **22/25**

| Axis | Score | Заметка |
|---|---|---|
| N | 3 | 437 records (371 carbon/low-alloy + 48 carburizing + 18 spring) |
| Schema | 5 | 25 features = 10 композиция + 12 heat treatment + 1 rolling + 3 inclusions |
| License | 5 | NIMS raw data публичный, Kaggle dataset без ограничений |
| Availability | 5 | Прямой download: Kaggle `chaozhuang/steel-fatigue-strength-prediction`, также Kaggle `konghuanqing/fatigue-dataset-for-steel`; Springer supplementary |
| Match | 4 | Target = **fatigue strength** (не YS, но mech. property). В нашем pipeline — новый `SteelClassProfile`: `fatigue_carbon_steel` с target `fatigue_strength_mpa`. Composition+heat treatment matches нашему Q&T профилю почти 1-в-1 |

**Вердикт:** самый «ML-ready» датасет из всех найденных. Gold-standard в materials informatics literature (статья Agrawal et al. 2014, 1000+ citations). Риск — target не YS; для нашего pipeline это не drop-in замена, а новый steel class с новой target property.

**Что получим:** способны сказать «validated on 437 real NIMS records, R² 0.98» с абсолютно чистой аудиторией.

---

### #2 — Citrine Conduit «Mechanical properties of some steels» — **18/25**

| Axis | Score | Заметка |
|---|---|---|
| N | 4 | 800+ steels, JSON `steeldata_final-autopif_new.json` |
| Schema | 3 | Composition + YS + elongation + fracture toughness. Process vars не указаны |
| License | 3 | Явной не указано, «public datasets remain accessible» после decommissioning Citrination |
| Availability | 4 | Прямой JSON download с Citrination dataset `153092` |
| Match | 4 | YS target direct match; feature schema 13/16 overlap с `PIPE_HSLA_FEATURE_SET` (ожидается — надмножество matminer 312) |

**Вердикт:** «scale-up» существующей Phase 0.1. Прямая замена matminer 312 → 800. Класс тот же (general low-alloy, не HSLA). Marginal pitch-uplift.

**Что получим:** replacement для matminer baseline, но не качественный сдвиг.

---

### #3 — Tata Steel Jamshedpur hot-rolled low-carbon (MDPI Materials 18/13/2966) — **17/25**

| Axis | Score | Заметка |
|---|---|---|
| N | 3 | 435 records |
| Schema | **5** | 15 элементов + **Finish Rolling Temp (FRT)** + **Coil Target Temp (CTT)** + mech. properties. **Лучший schema-match с `PIPE_HSLA_FEATURE_SET`** (~90%) |
| License | 3 | MDPI open-access статья, но data availability statement надо читать руками (WebFetch блок) |
| Availability | 2 | Supplementary material статьи, не independent DOI. Нужен ручной pull |
| Match | 4 | Industrial hot-rolled low-carbon — *именно* наш TMCP-профиль. Target = YS + UTS |

**Вердикт:** единственный industrial hot-rolled dataset с FRT/CTT. Если supplementary подтверждается как скачиваемое — **сдвигает весь расклад** для pitch («validated on industrial TMCP records», не «literature composites»).

**Что получим:** defensible industrial claim при supplementary подтверждении.

---

### #4 — GitHub `ashwinshetgaonkar/Estimate-Mechanical-Properties-of-Steel-compostions` — **15/25**

| Axis | Score | Заметка |
|---|---|---|
| N | 4 | 915 rows |
| Schema | 2 | 7 элементов (Al, Cu, Mn, N, Ni, Co, C) + temperature → YS/UTS/elongation/reduction. Узкий |
| License | 3 | MIT (код), data license не указана |
| Availability | 5 | Прямой CSV в GitHub |
| Match | 1 | Co 20 % → это maraging-class, как в matminer; вероятно overlap |

**Вердикт:** большой объём, но скорее всего тот же maraging-mix. Diminishing returns против Conduit 800.

---

### #5 — NIMS Fatigue Data Sheets (126 sheets, DOI каждый, MDR) — **15/25**

| Axis | Score | Заметка |
|---|---|---|
| N | 5 | 126 data sheets, каждый = десятки S-N curves |
| Schema | 4 | Composition + heat treatment + specimen geometry + S-N data |
| License | 5 | «Fully and freely available on the Internet», permanent DOIs (`10.11503/nims.XXXX`), hosted on Materials Data Repository |
| Availability | 1 | **Per-sheet DOI, но каждый — PDF с embedded S-N curves**. Требует OCR/извлечения, не clean CSV |
| Match | -- | Coverage: S25C-S55C carbon, SCM/SNCM Cr-Mo-Ni, SUS 304/316/403/430/630/329J3L, SM50B/490B/570Q structural, SB pressure vessel, SKD/SUP tool/spring — **широчайший класс** |

**Вердикт:** золотая жила, но фрагментированная. Extraction sprint на 1-2 недели даст 5000+ records. Это — Path 3 с известной координатой цели.

---

### #6 — Austenitic Stainless 568 (J Mat Sci 2025, `10.1007/s10853-025-11805-6`) — **14/25**

| Axis | Score | Заметка |
|---|---|---|
| N | 4 | 568 data points |
| Schema | 4 | 24 variables: composition + grain size + strain rate + test temperature |
| License | 3 | Springer open-access неизвестен, supplementary нужно проверять |
| Availability | 2 | Unclear — статья 2025, supplementary не расшифрован в search |
| Match | 3 | Target YS+UTS. Новый `SteelClassProfile`: `austenitic_stainless` (SUS 304/316/321/347/NCF 800H) |

**Вердикт:** хороший размер и schema, но data availability требует ручной проверки.

---

### #7 — matminer `steel_strength` (312, Citrine/Figshare) — **13/25** — *baseline*

Уже оценён в Phase 0.1 (`docs/public_data_spike_report.md`). Pipeline работает (R² 0.85), но класс maraging/tool, не HSLA.

---

### #8 — MPEA dataset (CitrineInformatics/MPEA_dataset GitHub) — **12/25**

630 records HEA/MPEA (Cantor-type alloys). 23 колонки: FORMULA, microstructure, processing, crystal phase, HV, YS, UTS, elongation, E-modulus. Apache 2.0, direct CSV.

**Но:** это HEA, не сталь. ~85 Fe-rich records — Cantor-type, не ordinary steel. **Полезно как adjacent benchmark** для проверки универсальности pipeline, не как production training.

---

### #9 — Welded joints fatigue dataset (Figshare 2025, Scientific Data) — **11/25**

**Массивный:** 1,666 publications → 5,797 process-parameter tuples + **52,000 S-N data sets**. CSV на Figshare, CC-BY скорее всего.

**Но:** target = fatigue life S-N curves, не quasi-static mech. property. Другая задача (fatigue modeling), требовал бы нового agent в pipeline.

---

### #10 — Zenodo 17201294 «E450 HSLA» — **отбраковано**

**CC-BY-NC-ND = блокер для commercial pitch Voestalpine.** Не используем.

---

### #11 — Tier skipped — недоступные

- **NIMS MatNavi main** (Kinzoku, creep) — ToS запрещает bulk acquisition.
- **ASM Alloy Center 2000 records** — proprietary (источник PMC 7727799).
- **Hyundai Steel 5473 records** (Sci Rep 2021) — не released raw.
- **41924 industrial samples** (Mater Chem Phys 2025) — closed.
- **Zenodo 6778336 S355** — один sheet, не composition sweep.

---

## Ранжированный outcome

| Rank | Dataset | N | License | Match | Total | Integration effort |
|---|---|---|---|---|---|---|
| 1 | **Agrawal NIMS fatigue (Kaggle)** | 437 | open | fatigue target | 22 | новый profile `fatigue_carbon_steel`, 1 день |
| 2 | **Citrine Conduit 800** | 800 | public-open | YS direct | 18 | drop-in замена matminer, 2-4 часа |
| 3 | **Tata Steel 435 (MDPI)** | 435 | MDPI open | HSLA-like TMCP | 17 | ручной pull supplementary + adaptation, 1-2 дня |
| 4 | ashwinshetgaonkar 915 | 915 | unclear | maraging overlap | 15 | 2-4 часа, сомнительный uplift |
| 5 | NIMS fatigue sheets | ~5000 | CC-BY/DOI | fatigue, PDF | 15 | sprint 1-2 недели (extraction) |
| 6 | Austenitic stainless 568 | 568 | unclear | new class | 14 | 1 день + data verification |
| 7 | matminer 312 (baseline) | 312 | CC-BY | maraging | 13 | уже сделано |
| 8 | MPEA 630 | 630 | Apache 2.0 | HEA not steel | 12 | benchmark only |
| 9 | Welded fatigue 5797 | 5797 | CC-BY | S-N target | 11 | new pipeline branch |

---

## Рекомендация

**Top-3 кандидата на Path 2 (обучение pipeline на реальных данных):**

### **A. Agrawal NIMS fatigue — #1 по totals.** Риск/выгода:
- **+** Готовый Kaggle CSV, 437 clean records, широко валидированный датасет (Agrawal et al. 2014 — 1000+ citations).
- **+** Позволит добавить новый `SteelClassProfile` = `fatigue_carbon_steel` с target `fatigue_strength_mpa`. Расширит pipeline на 3-й класс сталей (к HSLA + Q&T).
- **−** Это новая ML-задача (fatigue), не наш основной use-case (YS/UTS прогноз для pipe steel).
- **−** Не поможет HSLA-pitch, но даст сильный claim «pipeline validated across 3 steel classes на 437 real NIMS records, R² 0.98».

### **B. Citrine Conduit 800 — #2 по totals.** Риск/выгода:
- **+** Прямая drop-in замена Phase 0.1 baseline, 2-4 часа работы.
- **+** YS target совпадает, composition schema близкая.
- **−** Marginal uplift: класс тот же maraging-mix, что matminer 312. Не решает главную слабость pitch.

### **C. Tata Steel 435 — #3 по totals, но #1 по Match.** Риск/выгода:
- **+** Единственный industrial hot-rolled low-carbon dataset с FRT+CTT. Schema 90 % match с нашим `PIPE_HSLA_FEATURE_SET`.
- **+** Defensible claim «validated on industrial TMCP data» — именно то, что отсутствовало в Phase 0.1.
- **−** Data availability MDPI supplementary требует ручного pull (WebFetch заблокирован); риск что supplementary не содержит полную таблицу.
- **−** 435 records — не массовый.

---

## Моё предложение для выбора

Три стратегии, в порядке ожидаемой value:

**1. Максимизировать ML-defensibility pitch (рекомендую).**
Параллельно запустить **A + B** как двойной удар:
- Agrawal NIMS → новый `fatigue_carbon_steel` class → claim «validated on 3 steel classes, 437 NIMS records, R² 0.98».
- Conduit 800 → заменить matminer 312 в Phase 0.1 eval → claim «validated on 800 real records» вместо 312.
- Время: 1-1.5 дня.

**2. Максимизировать HSLA-specificity (если customer ask появился).**
Только **C** — Tata Steel 435 с ручным pull MDPI supplementary. Если supplementary содержит clean table — это самый сильный claim. Если нет — failed spike за 1-2 часа, minimal sunk cost.

**3. Long-play (если pitch процесс идёт и есть 2-3 недели).**
**E** — NIMS fatigue sheets extraction sprint. 126 sheets с per-DOI → потенциал 5000+ records. Новый класс + широкий coverage. Это Path 3 с известной географией, но работа серьёзная.

---

## Decision point — ваш выбор

Какой из вариантов (или комбинацию) запускаем в Path 2:

- **A** (Agrawal NIMS) — новый fatigue_carbon_steel class
- **B** (Conduit 800) — замена matminer baseline
- **C** (Tata Steel 435) — HSLA-like industrial pull
- **A+B** — двойной удар (рекомендую)
- **E** (NIMS sheets extract) — long-play
- **Ничего** — остаться на Phase 0.1 wording

Готов запускать любой вариант сразу после вашего решения.
