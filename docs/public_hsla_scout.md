# Public HSLA data — feasibility scout (Path 1)

**Дата:** 2026-04-24
**Задача:** честно оценить, какие публичные источники реальных данных HSLA/pipeline-сталей существуют и пригодны для обучения production-модели `pipe_hsla` (API 5L X60-X70, `PIPE_HSLA_FEATURE_SET`).
**Формат:** feasibility brief без скачивания и кода. Следующий шаг (Path 2 или Path 3) определяет пользователь.

Контекст — это расширение Phase 0.1 data-spike (`docs/public_data_spike_report.md`, commit `186675f`), который уже показал, что matminer `steel_strength` (312 records) не-HSLA и даёт R² 0.85 только «on wrong steel class».

---

## Результаты обхода

### Tier 1 — пригодно «из коробки» для повторения Phase 0.1 на бóльшем N

| Источник | Размер | Схема | Лицензия | Вердикт |
|---|---|---|---|---|
| **matminer `steel_strength`** (Figshare DOI `10.6084/m9.figshare.7250453`) | 312 | 14 элементов + YS/UTS | Open, CC-BY | Уже использовано в Phase 0.1. Не HSLA (maraging/tool). |
| **Citrine Conduit «Mechanical properties of some steels»** (Citrination dataset `153092`, v3 от 2017-06-29, `steeldata_final-autopif_new.json`) | **800+** | composition + YS + elongation + fracture toughness | **Не указана явно**, платформа «public datasets remain accessible» | **Надмножество matminer 312** с высокой вероятностью (те же автор-цепочки). HSLA не помечен — это общий mix low-alloy. Реалистичный upgrade: 312 → ~800 records. Дельта к уже сделанному в 0.1 — умеренная. |
| **GitHub `ashwinshetgaonkar/Estimate-Mechanical-Properties-of-Steel-compostions`** | **915** | Al, Cu, Mn, N, Ni, Co, C + temperature → YS / UTS / elongation / reduction | MIT (код), лицензия данных не указана | Композиция включает Co 20 % → это тот же maraging-класс, что в matminer, но с бóльшим N. Не HSLA. Скорее всего пересекается с Conduit 800. |
| **MDPI `Materials` 18/13/2966 — Tata Steel Jamshedpur** (*Mechanical Property Prediction of Industrial Low-Carbon Hot-Rolled Steels Using ANN*) | **435** | 15 элементов + FRT + CTT + mech properties | MDPI open-access (проверить data availability) | **Самая близкая к HSLA запись** — industrial hot-rolled low-carbon + finish rolling temp + coil target temp = ровно наш `PIPE_HSLA_FEATURE_SET`. Прямой fetch статьи упал 403; нужен ручной заход через аккаунт или supplementary-извлечение. |

### Tier 2 — HSLA-специфично, но с фатальными ограничениями

| Источник | Размер | Проблема |
|---|---|---|
| **Zenodo `10.5281/zenodo.17201294`** (Kumar et al., 2025-09, «micro alloyed high strength low carbon structural steels», E450) | небольшой: xlsx-файлы «per figure» | **Лицензия CC-BY-NC-ND** — non-commercial + no derivatives = блокер для ML-работы в коммерческом pitch Voestalpine. Плюс формат «данные по каждой фигуре» — не унифицированная таблица; нужна ручная агрегация. |
| **Mendeley DOI `10.17632/mfhvdv4b8z.1`** (Korotaev, *Database for Steels Classification*) | не указан | Только trademark + состав + heat treatment + класс (аустенит/феррит/перлит). **Нет mechanical properties** — не пригодно для `yield_strength_mpa` обучения. |
| **Papers с supplementary tables** (Springer «literature-assisted pipeline Charpy» 2023, Acta Mater 2022, JMEP 2020, и т. п.) | 5-200 heats на статью | Рассеяны, heterogeneous units, часть полей отсутствует (rolling schedule есть не везде), paywall-блок для прямого fetch. Это Path 3. |

### Tier 3 — недоступно для bulk ML

| Источник | Причина отказа |
|---|---|
| **NIMS MatNavi / Kinzoku / Creep DS** | Бесплатная регистрация, но ToS: *«acquisition of large amounts of data, whether by manual or mechanical means, is prohibited»*. Web-scraping запрещён. Для ML-обучения класс блокера. |
| **ASM Alloy Center Database** (источник 2000-record работы PMC `PMC7727799`) | Проприетарный, licensing per-seat. |
| **Hyundai Steel 5473 records** (*Scientific Reports* 2021) | Корпоративная. В статье — только агрегированная статистика, raw не выложен. |
| **Industrial 41924 samples** (*Mater Chem Phys* 2025 паперу про UTS) | Tata / неидентифицированный mill, closed. |

---

## Схема compatibility с `PIPE_HSLA_FEATURE_SET`

Наш feature set (из `app/backend/steel_classes.py` профиля `pipe_hsla`) — 16 composition columns + rolling_finish_temp + cooling_rate, target `yield_strength_mpa`.

| Источник | Composition overlap | Process vars overlap | Target match | Полнота схемы |
|---|---|---|---|---|
| matminer 312 | **13/16** (нет p/s/cu, есть лишние Co/W) | **0/2** | YS ✓ | 60 % |
| Citrine Conduit 800+ | **~13/16** (ожидается то же) | **0/2** | YS ✓ | 60 % |
| ashwinshetgaonkar 915 | **7/16** (нет Si/P/S/Cr/Mo/V/Nb/Ti) | 1/2 (temperature) | YS ✓ | 40 % |
| **Tata Steel 435** | **~15/16** (15 элементов заявлено) | **2/2** (FRT + CTT) | YS ✓ | **90 %** |
| Zenodo 17201294 | частично | частично | лицензия блокирует | — |

---

## Вывод — что изменилось со времени Phase 0.1

**Не изменилось:**

- Единственный крупный открытый «ML-ready» tabular dataset для low-alloy steel — всё тот же матmiмер/Citrine (312 → 800 records в Conduit). Класс — maraging/tool/high-strength mix, не HSLA. Запуск Path 2 на Conduit 800 вместо matminer 312 даст пропорциональное улучшение pitch-claim (с «312 records R² 0.85» на «800 records R² ≈ 0.85») — но всё ещё NOT HSLA, и Critic OOD-детектор всё ещё будет отвергать synthetic-prediction на этом данные как в Phase 0.1 Config D (R² −8.5).
- NIMS остаётся hard-blocker.

**Изменилось — одна новая цель для Path 2:**

- **Tata Steel Jamshedpur 435-record dataset** (MDPI *Materials* 18/13/2966) — первый публичный industrial hot-rolled low-carbon набор со схемой, совместимой с нашим `PIPE_HSLA_FEATURE_SET` на 90 %. 435 record, 15 compositions + FRT + CTT + mech properties. Open-access журнал MDPI, но точная форма data availability требует ручного чтения статьи (прямой fetch блокируется CloudFlare).
- Если supplementary этого dataset подтверждается как CC-BY — это **единственная находка, которая меняет расклад** относительно Phase 0.1 отчёта.

---

## Рекомендация

**Default — остаться на Path 3 backlog (как в `186675f`).**

Phase 0.1 wording в pitch (*«ML-Pipeline validated on 312 real peer-reviewed records — R² 0.85; open dataset covers high-strength/tool steels, HSLA-specific accuracy comes from Phase 0 benchmark audit»*) — честный, и сканирование не нашло источника, который бы делал его сильнее без существенных инвестиций.

**Исключение — если появится customer ask:**

1. **Мини-спайк на Citrine Conduit 800** (2-4 часа): скачать JSON, harmonize к нашему schema, прогнать те же 4 config как в Phase 0.1. Ожидаемый результат: «pipeline validated on 800 real records» вместо 312, R² аналогичный. Marginal pitch-uplift.
2. **Tata Steel 435-spike** (1-2 дня с ручным извлечением supplementary): если data availability подтверждается — это **первый реальный industrial hot-rolled dataset со схемой ~90 % overlap**. Запустить train+hold-out на нём, сравнить feature_importance с нашей synthetic-моделью (check соответствия микролегирования и TMCP-рычагов). Если coverage and feature_importance сходятся — это **defensible claim уровня «validated on industrial TMCP data»**, не просто «real data».
3. **Paper-mining sprint (1-3 недели)** остаётся единственным способом построить HSLA-specific reference corpus. Без concrete customer-ask — избыточно.

**Что НЕ делать:**

- Не подключать Zenodo 17201294 (CC-BY-NC-ND = legal risk для commercial pitch).
- Не пытаться bulk-собрать MatNavi (нарушение ToS).
- Не повторять Phase 0.1 на ashwinshetgaonkar 915 — это другой класс (Co-rich maraging), как и matminer.

---

## Decision log

Скаут длился ~30 мин, пройдено: Figshare, Zenodo, Mendeley Data, Citrine, Citrination, Kaggle, GitHub (sedaoturak list, awesome-industrial-datasets, socoolblue, ashwinshetgaonkar, batiukmaks), NIMS ToS, MDPI, Springer, ScienceDirect (несколько — частично заблокированы).

Следующий decision-point у пользователя. Значимая новая информация vs Phase 0.1 отчёт: появилась одна потенциально-высокоценная цель (Tata Steel 435), всё остальное укладывается в уже принятое Path 3 решение.
