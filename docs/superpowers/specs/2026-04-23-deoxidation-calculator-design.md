# Al Deoxidation Calculator — physics-based advisory for ladle furnace

**Дата:** 2026-04-23
**Статус:** Design (готов к implementation plan)
**Baseline:** tag `v0.4-multi-class`

---

## 1. Цель и мотивация

MVP v0.4 покрывает offline alloy design (HSLA + EN 10083 Q&T). Эта фича расширяет систему в **смежный, но отдельный домен** — on-line decision support при **вторичной металлургии** (ladle furnace), а именно **раскисление алюминием**. Металлург-плавильщик перед выпуском из электропечи получает замер активного кислорода O_a (обычно 300-700 ppm) и должен добавить нужное количество Al для снижения O_a до уровня, заданного классом стали (5-15 ppm).

**Цель:** **physics-based advisory calculator** на базе 3 общепринятых термодинамических моделей Al-O равновесия. Без ML, без feedback loop, без интеграции с сенсорами. На каждый расчёт — мгновенный ответ с breakdown по массе, стоимости и сравнение моделей. Ценность: **замена Excel + справочника у плавильщика** + audit-trail по опт-ин.

Две функции (симметричные через одну физическую модель):
1. **Forward**: «сколько кг Al подать, чтобы снизить O_a с X до Y?»
2. **Inverse**: «по факту плавки (Al подали, O_a замерили до/после) — какое эффективное содержание активного Al было в поставке?»

---

## 2. Scope

### В MVP входит (scope A из brainstorm)
- Новый модуль `app/backend/deoxidation.py` с registry **3 термодинамических моделей**:
  - **Fruehan 1985**: `log₁₀K = 64000/T − 20.57` (классика, дефолт).
  - **Sigworth-Elliott 1974 (JSPS)**: `log₁₀K = 62680/T − 20.54`.
  - **Hayashi-Yamamoto 2013**: `log₁₀K = −62780/T + 19.18` + Al-Al коррекция при `[Al] > 0.05%`.
- Forward-режим: `compute_al_demand(...)` — вход O_a, T, mass, target_O_a, purity, burn_off, model_id → DeoxidationResult.
- Inverse-режим: `compute_al_quality(...)` — вход pre/post O_a, Al_added, T, mass, model_id → effective purity %.
- **Compare-mode**: запуск всех 3 моделей на одних входах и сведение в таблицу — демонстрация ±10-15% разброса между моделями.
- Новая вкладка Streamlit «🔥 Раскисление» (между «📊 Прогноз» и «📚 История»).
- Target O_a как опциональное поле в `SteelClassProfile` (`target_o_activity_ppm`). UI предзаполняет из профиля активной модели.
- Cost-оценка через существующий `cost_model` (читает `Al` из текущего price snapshot).
- Decision Log **опт-ин**: кнопка «💾 Сохранить в Decision Log» — по умолчанию расчёт не пишется, оператор решает сам. Исключает спам БД в режиме демо.
- Pattern Library фаза `Phase.DEOXIDATION` + 3 паттерна DX01-DX03.

### Явно вне MVP
- Кинетика (скорость растворения Al, stirring-зависимость — Fischer/Janke).
- Баланс FeO в шлаке (`3(FeO) + 2[Al] = 3[Fe] + Al₂O₃`).
- Комбинированное раскисление (Al + FeSi / Al + Ca-wire) — задача #3 из исходного ТЗ, отложена на v0.6.
- Реальные heat logs / online learning / feedback loop.
- Интеграция с анализаторами O-активности (Celox / Heraeus).
- Multi-element simultaneous deox (Al+Si+Mn).
- LLM-Critic для deoxidation phase (Critic v2 остаётся на `training`).

---

## 3. Архитектурные решения (из brainstorm)

| Решение | Выбор | Почему |
|---|---|---|
| Scope первой итерации | Задачи #1 + #2 (Al-only forward + inverse) | Одна физ-модель обслуживает обе |
| Глубина физики | Только термодинамика Al-O | Честно, минимум входов, измеримая точность |
| Термодинамические модели | 3 параллельно + compare-mode | Даёт clients интуицию о ±15% неопределённости без ML-обещаний |
| UI размещение | Новая вкладка «🔥 Раскисление» | Разделение offline design vs on-line advisory |
| Target O_a привязка | Поле `target_o_activity_ppm` в SteelClassProfile | Согласуется с multi-class архитектурой v0.4 |
| Decision Log | Опт-ин (кнопка «Сохранить») | Защита от спама БД при 50-200 расчётов/день |
| Cost integration | Переиспользуем `cost_model.PriceSnapshot` | Al уже в seed-прайсе (2.40 €/кг) |
| Задача #3 (комбинир. раскисление) | **Не в этой итерации** | Отдельный Pareto-pipeline, v0.6 |
| LLM-Critic на deox phase | Не подключается | Scope A — physics-only |

---

## 4. Физическая модель

### 4.1 Равновесие Al-O

Реакция: `2[Al] + 3[O] = Al₂O₃(s)`. Константа равновесия:

```
K = a(Al₂O₃) / (a[Al]² × a[O]³)
```

В разбавленных растворах `a(Al₂O₃) = 1` (чистый оксид), `a[Al] = [%Al]·f_Al`, `a[O] = [%O]·f_O`. Коэффициенты активности `f` из Wagner interaction parameters — в первом приближении `e_O^Al ≈ −1.17`, `e_Al^O ≈ −6.6`. Log-уравнение:

```
log₁₀(K) = −2·log₁₀([Al]·f_Al) − 3·log₁₀([O]·f_O)
```

Где `log₁₀(K)` задаётся термодинамической моделью из registry.

### 4.2 Три модели в registry

```python
THERMO_MODELS: dict[str, ThermoModel] = {
    "fruehan_1985": ThermoModel(
        name="Fruehan 1985",
        citation="Fruehan R., Ladle Metallurgy, ISS 1985",
        log_k=lambda T_K: 64000.0 / T_K - 20.57,
        valid_T_range=(1773, 1923),    # K, т.е. 1500-1650°C
        expected_accuracy_ppm=40.0,
    ),
    "sigworth_elliott_1974": ThermoModel(
        name="Sigworth-Elliott 1974",
        citation="JSPS Steelmaking Data Sourcebook, 1988 (based on Sigworth & Elliott 1974)",
        log_k=lambda T_K: 62680.0 / T_K - 20.54,
        valid_T_range=(1773, 1923),
        expected_accuracy_ppm=30.0,
    ),
    "hayashi_2013": ThermoModel(
        name="Hayashi-Yamamoto 2013",
        citation="Hayashi M., Yamamoto T., ISIJ Intl. 53, 2013",
        log_k=lambda T_K: -62780.0 / T_K + 19.18,
        valid_T_range=(1823, 1973),    # Hayashi fits narrower high-T range
        expected_accuracy_ppm=20.0,
        al_al_correction=True,
    ),
}

DEFAULT_MODEL_ID = "fruehan_1985"
```

`ThermoModel.log_k(T_K) -> float`. Для Hayashi опционально добавляется `al_al_correction` (квадратичный член при `[Al] > 0.05%` — +0.02 к log_k, эмпирика).

### 4.3 Forward расчёт

```python
@dataclass
class DeoxidationResult:
    al_total_kg: float              # полная навеска Al
    al_active_kg: float             # реагирующая часть (учёт purity)
    al_burn_off_kg: float           # угар
    o_a_expected_ppm: float         # ожидаемый остаточный O_a
    al_per_ton: float               # кг/т
    cost_eur: float                 # стоимость при текущем snapshot
    model_id: str
    inputs: dict                    # для audit
    warnings: list[str]             # physical / range warnings


def compute_al_demand(
    o_a_initial_ppm: float,
    temperature_C: float,
    steel_mass_ton: float,
    target_o_a_ppm: float,
    al_purity_pct: float = 100.0,
    burn_off_pct: float = 20.0,
    model_id: str = DEFAULT_MODEL_ID,
    al_price_per_kg: float = 2.40,
    currency: str = "EUR",
) -> DeoxidationResult:
    ...
```

Алгоритм:

1. Sanity checks: `o_a_initial ∈ [50, 800]`, `target < o_a_initial`, `T ∈ model.valid_T_range`. Warnings при нарушении.
2. Δ[O] в %: `delta_o_pct = (o_a_initial - target) / 1e6 × 100`.
3. Стехиометрия: 2 Al + 3 O = Al₂O₃. Molar masses: Al=26.98, O=16.00. Соотношение масс Al к O: `2·26.98 / (3·16.00) = 1.1242`. Но в пересчёте на растворённый O в стали: на 1 кг O нужно 1.1242 кг Al активного.
   - `delta_o_kg = delta_o_pct/100 × steel_mass_ton × 1000` (кг O).
   - `al_active_required_kg = delta_o_kg × 1.1242`.
4. Поправка через равновесную модель: используем `log_k(T_K)` для проверки, что target_O_a достижим при данной [Al]_eq; iteratively корректируем `al_active_required` если модель говорит «при такой [Al] равновесный [O] будет выше target» (Ньютон-Рапсон 3-5 итераций).
5. Учёт purity и burn-off: `al_total_kg = al_active_required × 100/purity_pct / (1 − burn_off_pct/100)`.
6. Cost: `al_total_kg × al_price_per_kg`.
7. Возвращаем `DeoxidationResult` + warnings если были.

### 4.4 Inverse (задача #2 — quality estimation)

```python
@dataclass
class AlQualityResult:
    effective_purity_pct: float     # эффективный процент активного Al
    effective_active_kg: float      # сколько Al реально реагировало
    expected_active_kg: float       # сколько должно было при 100% purity
    assumed_burn_off_pct: float
    model_id: str
    warnings: list[str]


def compute_al_quality(
    o_a_before_ppm: float,
    o_a_after_ppm: float,
    al_added_kg: float,
    temperature_C: float,
    steel_mass_ton: float,
    burn_off_pct: float = 20.0,
    model_id: str = DEFAULT_MODEL_ID,
) -> AlQualityResult:
    ...
```

Алгоритм:
1. Сколько O было связано: `delta_o_kg = (o_a_before - o_a_after) / 1e6 × steel_mass_ton × 1000`.
2. Активный Al, который реально сработал: `actual_active_kg = delta_o_kg × 1.1242`.
3. Ожидаемый при 100% purity: `expected_active_kg = al_added_kg × (1 − burn_off_pct/100)`.
4. Эффективная чистота: `effective_purity_pct = actual_active_kg / expected_active_kg × 100`.
5. Warning если `< 70%` — «подозрительно низкая чистота; проверьте поставку или увеличьте burn_off допущение».

### 4.5 Compare-mode

```python
def compare_all_models(
    o_a_initial_ppm: float,
    temperature_C: float,
    steel_mass_ton: float,
    target_o_a_ppm: float,
    al_purity_pct: float = 100.0,
    burn_off_pct: float = 20.0,
) -> list[DeoxidationResult]:
    """Return results for all 3 thermo models — used in 'Compare' UI mode."""
    return [
        compute_al_demand(..., model_id=mid)
        for mid in ("fruehan_1985", "sigworth_elliott_1974", "hayashi_2013")
    ]
```

UI отображает как таблицу с колонками: модель / Al (кг) / Al (кг/т) / O_a expected / стоимость. Разброс обычно ±10-15%.

---

## 5. SteelClassProfile расширение

Добавляется **опциональное** поле:

```yaml
# data/steel_classes/pipe_hsla.yaml
target_o_activity_ppm: 5.0    # критичная ударная вязкость при -60°C

# data/steel_classes/en10083_qt.yaml
target_o_activity_ppm: 15.0   # менее требовательная к чистоте
```

В dataclass:

```python
@dataclass
class SteelClassProfile:
    ...
    target_o_activity_ppm: float | None = None   # NEW
```

`load_steel_class` читает через `data.get("target_o_activity_ppm")`. None → UI требует ручной ввод; число → UI предзаполняет, но оставляет поле editable.

---

## 6. UI вкладки «🔥 Раскисление»

Новый `tab_deox` добавляется в список вкладок **между** `tab_predict` и `tab_history`:

```python
tab_design, tab_train, tab_predict, tab_deox, tab_history = st.tabs([
    "🎯 Дизайн сплава",
    "🤖 Обучение модели",
    "📊 Прогноз",
    "🔥 Раскисление",
    "📚 История",
])
```

### 6.1 Layout

```
🔥 Раскисление жидкой стали алюминием
Physics-based advisory. Без ML. На каждую плавку.

┌ Контекст плавки ──────────────────────────────────────┐
│ Активная модель: 🔩 Pipe HSLA                         │
│ Target O_a (из профиля): 5 ppm    [override: ___]     │
│ Термодинамическая модель: [Fruehan 1985 ▼]            │
└──────────────────────────────────────────────────────┘

[Сколько Al нужно] [Качество Al по факту] [Сравнить модели]  ← st.tabs sub

— Tab 1: Forward —
Измеренные параметры плавки:
  O_a измерено, ppm:     [450.00]   T расплава, °C: [1620.00]
  Масса стали, т:        [180.00]   Heat ID (опц.): [_______]

Параметры Al-присадки:
  % активного Al:        [100.00]   Угар, %:        [20.00]

[🧮 Рассчитать]

─── Результат (Fruehan 1985) ──────────────────────────
💊 Навеска Al: 54.8 кг (0.305 кг/т)
   ├─ Активный на реакцию: 45.7 кг
   ├─ Угар: 9.1 кг (20%)
   └─ Ожидаемый остаточный O_a: 5.0 ppm (= target)

💰 Стоимость: 131.5 EUR (при 2.40 EUR/кг, seed 2026-04-23)

(График Altair: «Al (кг) → остаточный O_a (ppm)», вертикальная красная линия на 54.8 кг)

[💾 Сохранить в Decision Log]   [📋 Экспорт в CSV]

— Tab 2: Inverse (качество Al по факту) —
Плавка уже прошла — оценим качество поставки:
  O_a до,  ppm:          [450]     O_a после, ppm: [8]
  Al добавлено, кг:      [65.0]    T, °C:          [1620]
  Масса стали, т:        [180]     Угар (допущение, %): [20]

[🔍 Оценить качество]

─── Результат ──────────────────────────────────────────
Эффективное активное Al: 82.4 %
(вместо ожидаемых 100% при декларируемой чистоте чушки)

⚠️ Warning: чистота ниже 90% — проверьте поставку или
   пересмотрите допущение о burn_off.

— Tab 3: Сравнить модели —
(ввод как в Tab 1)
[⚖️ Сравнить все 3 модели]

Результат — таблица:
┌─────────────────────┬────────┬──────────┬───────────┬──────────┐
│ Модель              │ Al, кг │ Al, кг/т │ O_a, ppm  │ Цена, €  │
├─────────────────────┼────────┼──────────┼───────────┼──────────┤
│ Fruehan 1985        │ 54.8   │ 0.305    │ 5.0       │ 131.5    │
│ Sigworth-Elliott 74 │ 51.2   │ 0.284    │ 5.0       │ 122.9    │
│ Hayashi 2013        │ 58.1   │ 0.323    │ 5.0       │ 139.4    │
└─────────────────────┴────────┴──────────┴───────────┴──────────┘

Разброс: ±6.3%. Для advisory это ожидаемая неопределённость
между академическими термодинамическими моделями.

Altair-график bar chart сравнивающий навески.
```

### 6.2 Session state

- `st.session_state["last_deox_result"]` — результат последнего Forward-расчёта (для Export CSV).
- `st.session_state["deox_model_id"]` — выбранная модель (persistent между запусками).
- `st.session_state["deox_heat_id"]` — Heat ID (опционально, очищается после сохранения в Decision Log).

---

## 7. Pattern Library

Новая фаза `Phase.DEOXIDATION = "deoxidation"` в enum.

3 новых паттерна:

| ID | Severity | Условие | Сообщение |
|---|---|---|---|
| **DX01** | HIGH | `o_a_initial_ppm > 800` OR `< 50` | «Измеренный O_a вне физически осмысленного диапазона 50-800 ppm для LF — проверьте датчик.» |
| **DX02** | MEDIUM | `target_o_a >= o_a_initial` | «Целевой O_a не меньше измеренного — раскисление не нужно или ошибка ввода.» |
| **DX03** | MEDIUM | `effective_purity < 70%` (inverse only) | «Эффективная чистота Al < 70% — поставка подозрительна. Проверьте чушку/чистоту лигатуры.» |

Patterns запускаются **автоматически** при каждом расчёте через `run_all_patterns(ctx, phase=Phase.DEOXIDATION)`. Результат добавляется в UI как блок над основным результатом (красный/жёлтый alerts, как в tab_train).

---

## 8. Decision Log — опт-ин

Кнопка «💾 Сохранить в Decision Log» в результате Forward-режима. При нажатии:

```python
log_decision(
    phase="deoxidation",
    decision=f"Al-deox для плавки {heat_id or 'без ID'}: "
             f"{result.al_total_kg:.1f} кг на {steel_mass_ton} т "
             f"({result.al_per_ton:.3f} кг/т)",
    reasoning=f"Model={result.model_id}, "
              f"O_a {o_a_initial}→{target_o_a} ppm @ {T}°C, "
              f"purity={purity}%, burn_off={burn_off}%. "
              f"Cost={result.cost_eur:.2f} {currency}",
    context={"inputs": result.inputs, "result": asdict(result)},
    author="deox_calculator",
    tags=["deoxidation", "al_deox", steel_class_id, heat_id or "no_id"],
)
```

Inverse режим — аналогично, с phase="deoxidation", tags=["deoxidation", "al_quality", heat_id].

Compare-mode записывает **один ряд** суммарно (все 3 результата в context).

---

## 9. Файлы

### Новые
- `app/backend/deoxidation.py` — `ThermoModel`, `THERMO_MODELS` registry, `DeoxidationResult`, `AlQualityResult`, `compute_al_demand`, `compute_al_quality`, `compare_all_models`, physical constants (`M_Al = 26.98`, `M_O = 16.00`, `AL_TO_O_MASS_RATIO = 1.1242`, Wagner coefficients).
- `app/tests/test_deoxidation.py` — 7-8 тестов.
- `docs/superpowers/specs/2026-04-23-deoxidation-calculator-design.md` (этот).

### Изменяемые
- `app/backend/steel_classes.py` — `SteelClassProfile.target_o_activity_ppm: float | None = None`.
- `data/steel_classes/pipe_hsla.yaml` — `target_o_activity_ppm: 5.0`.
- `data/steel_classes/en10083_qt.yaml` — `target_o_activity_ppm: 15.0`.
- `app/frontend/app.py` — новая `tab_deox` между predict и history, импорт deoxidation.
- `pattern_library/patterns.py` — `Phase.DEOXIDATION`, `_check_dx01/02/03`.
- `CLAUDE.md` — параграф про deox-калькулятор (2-3 строки).

---

## 10. Тесты — `app/tests/test_deoxidation.py`

1. **`test_fruehan_log_k_at_1873K`** — `log_k(1873) ≈ 64000/1873 - 20.57 ≈ 13.60`. Числовая проверка формулы.
2. **`test_compute_al_demand_typical_hsla`** — O_a=450, T=1620°C, mass=180, target=5 ppm, purity=100%, burn=20% → Al_total ≈ 54-60 кг (допуск ±10% от модели-дефолта).
3. **`test_compute_al_demand_with_lower_purity`** — purity=80% → Al_total больше на ~25% (100/80).
4. **`test_compare_all_models_returns_three_results`** — `compare_all_models(...)` возвращает list из 3 DeoxidationResult, все с одинаковыми inputs но разными `al_total_kg`. Разброс max/min < 25%.
5. **`test_inverse_roundtrip`** — forward с purity=85% → inverse на том же Al_added → возвращаемый effective_purity ≈ 85 ± 1%.
6. **`test_compute_al_demand_target_exceeds_initial_raises`** — target=100, initial=50 → `ValueError` или flag в warnings.
7. **`test_compute_al_demand_out_of_temperature_range_warns`** — T=1450°C (ниже Fruehan 1500) → warning в results.warnings.
8. **`test_compute_al_demand_cost_uses_price_argument`** — `al_price_per_kg=3.0` при 50 кг Al → cost = 150.

---

## 11. Acceptance criteria

1. `pytest app/tests/test_deoxidation.py -v` — 8 тестов проходят.
2. Полный suite `pytest app/tests/ -q` — без регрессий (47 предыдущих тестов + 8 новых = 55).
3. Streamlit вкладка «🔥 Раскисление» открывается, формы рендерятся, 3 sub-tabs работают.
4. Forward: O_a=500, T=1620, mass=180, target=5, purity=100 → получаем Al ≈ 55-62 кг в зависимости от модели, график рендерится.
5. Inverse: O_a_before=500, O_a_after=10, Al=65, T=1620, mass=180 → effective_purity близко к 85-95%.
6. Compare: таблица с 3 моделями, разброс в пределах ±15%.
7. DX01-DX03 срабатывают на соответствующих входах (O_a=900 → DX01 HIGH; target>initial → DX02; inverse с purity=50% → DX03).
8. Кнопка «Сохранить в Decision Log» создаёт запись с tag `deoxidation`.
9. Target O_a автоматически заполняется из профиля активной модели (5 ppm для HSLA, 15 ppm для Q&T).

---

## 12. Out of scope (roadmap)

- **v0.5.2** — задача #3: комбинированное раскисление (Al + FeSi + Ca-wire) с Pareto-оптимизацией «O_a × стоимость × время» через NSGA-II (переиспользуется архитектура `inverse_designer.py`).
- **v0.6** — кинетика растворения (Fischer/Janke), time-to-result график.
- **v0.7** — баланс FeO в шлаке, дополнительные входы: масса шлака, %FeO.
- **v0.8** — интеграция с реальными heat logs клиента (CSV upload или API), online re-calibration burn_off и purity эмпирически.
- **v0.9** — MES/L2 интеграция через OPC-UA, real-time inference от сенсора O-активности.
- **v1.0** — ML на клиентских heat logs с reservoir sampling и drift detection (classic ML for process control).

---

## 13. Риски и допущения

- **Burn-off как константа 20%** — реально зависит от температуры, интенсивности продувки Ar, геометрии ковша. Для advisory это приемлемо, но **обязательно** надо указать в UI «это grayscale допущение». Точное значение для конкретного ковша определяется после первого inverse-расчёта.
- **Тепловой баланс не моделируется** — добавление холодной Al-чушки снижает T расплава на 2-4°C на тонну Al/100т стали. Advisory даёт результат «в изотерме» — оператор знает, что в реальности нужно учесть термокомпенсацию.
- **Wagner interaction coefficients постоянные** — зависят от композиции (особенно [C]). Первое приближение достаточно для ±40 ppm точности.
- **Compare-mode разброс не означает «лучшая модель»** — это показатель неопределённости академических формул. В UI подчёркивается: выбор модели — калибруется на реальных heat logs клиента.
- **target_o_a — эмпирический параметр** — в YAML указаны типичные значения для advisory; реальный target зависит от ТЗ на конкретную марку.
- **Нет учёта дезоксидации Si, Mn, Ca** — все они также связывают O, формула недооценивает количество Al на 5-10% если в плавке есть предварительный добавленный Si (но для чистого Al-deox это работает корректно).
