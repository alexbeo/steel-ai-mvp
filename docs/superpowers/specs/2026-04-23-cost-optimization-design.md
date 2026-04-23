# Cost-optimization в inverse design HSLA-сталей

**Дата:** 2026-04-23
**Статус:** Design (готов к переходу в implementation plan)
**Связанные коммиты:** baseline `919bf6b` (initial commit)

---

## 1. Цель и мотивация

В текущем MVP `app/backend/inverse_designer.py` уже использует 3-objective NSGA-II (distance_to_target, alloying_cost, prediction_uncertainty), но `alloying_cost` считается через **хардкод** `ELEMENT_PRICES_EUR_PER_KG` — цены за «чистый элемент» в EUR. Это:
- не отражает реальную закупку (никто не покупает чистый Nb, покупают FeNb-65);
- не привязано к дате принятия решения (цены зашиты в код);
- не учитывает базовую шихту (лом / слябы);
- не имеет UI для обновления цен без правки кода.

**Цель:** заменить примитивную модель на полноценный **ferroalloy-based price snapshot** с датой, валютой, редактируемым UI и сохранением в Decision Log; довести demo до уровня «смотрите, на ценах сегодня выгоднее этот состав, а на ценах прошлого квартала — был бы другой».

---

## 2. Scope

### В MVP входит
- Только **химическая** себестоимость: базовая шихта + легирующие через ферросплавы.
- **Одновалютный** snapshot цен (RUB / USD / EUR — атрибут прайса).
- Ручное редактирование цен в UI перед запуском (`st.data_editor`) с сохранением snapshot в Decision Log.
- **Hard-fail** pre-check: если в прайсе нет цены на элемент из design space — дизайн не запускается.
- Два режима отображения cost: **full** (включая базу, ₽/т) и **incremental** (только стоимость легирования, ₽/т).
- Pareto plot «σт × cost» и per-candidate breakdown «материал → масса → цена → вклад».

### Явно вне MVP
- Процессные энергозатраты (нагрев, прокатка, термообработка).
- Вспомогательные материалы (раскислители Al/Ca, шлакообразующие, модификаторы).
- Курсовые коэффициенты и мульти-валютные прайсы (разные позиции в разных валютах).
- Онлайн-цены (LME, биржевые котировки).
- Tenant-specific capabilities (какие ферросплавы реально доступны клиенту).
- Сравнение «текущая vs перспективная плавка» с ROI-дельтой.

---

## 3. Архитектурные решения (принятые при brainstorming)

| Решение | Выбор | Почему отвергнуты альтернативы |
|---|---|---|
| Что входит в cost | Только химия | Энергия и вспомогательные — усложнение модели без выигрыша для первой демо |
| Как cost влияет на оптимизацию | **Заменяем** существующий `alloying_cost` objective (остаётся 3 objectives: target, cost, uncertainty) | Дополнительный objective превратил бы оптимизацию в 4D — сложнее визуализировать Pareto |
| Модель цен | Ферросплавы (FeNb-65, FeMn-80 и т.д.) | Чистоэлементная модель звучит непрофессионально для металлургов |
| Базовая шихта | Два режима: full / incremental, переключатель в UI | Скрывать нельзя, но в инкрементальном режиме контраст между кандидатами виднее |
| Ввод цен | `st.data_editor` в Streamlit + snapshot в Decision Log | CSV-only менее эффектно; для демо нужна возможность «смотрите, Nb подорожал на 15%» |
| Валюта | Атрибут прайса (RUB / USD / EUR), один на snapshot | Курсы и мульти-валюта — переусложнение MVP |
| UI результата | Pareto plot (σт × cost) + per-candidate breakdown | Чистая таблица недостаточно убедительна |
| Seed-прайс | ~10 позиций (scrap + 8 ферросплавов + 1 чистый Cu) | Покрывает весь `PIPE_HSLA_FEATURE_SET` |
| Отсутствующая цена | Hard fail перед запуском | Soft warning создаёт риск «молчаливых нулей» на демо |

---

## 4. Data model

### 4.1 Core типы — `app/backend/cost_model.py` (новый модуль)

```python
from dataclasses import dataclass, field
from datetime import date
from typing import Literal
from pathlib import Path

Kind = Literal["base", "ferroalloy", "pure"]
Currency = Literal["RUB", "USD", "EUR"]
CostMode = Literal["full", "incremental"]


@dataclass(frozen=True)
class Material:
    id: str                              # "FeNb-65", "scrap", "Cu"
    kind: Kind
    price_per_kg: float                  # в валюте snapshot
    element_content: dict[str, float]    # {"Nb": 0.65, "Fe": 0.35}, сумма ≈ 1.0
    # Для "base" (scrap) — {"Fe": 1.0}
    # Для "pure" (Cu, Al) — {"Cu": 1.0}
    # Для "ferroalloy" — главный элемент + Fe-остаток


@dataclass
class PriceSnapshot:
    date: date
    currency: Currency
    materials: dict[str, Material]       # ключ = Material.id
    source: str = "manual"               # "seed" | "manual" | "imported:<filename>"
    notes: str = ""


@dataclass
class CostContribution:
    material_id: str
    mass_kg_per_ton_steel: float         # масса материала на 1 тонну стали
    price_per_kg: float
    contribution_per_ton: float          # mass × price, в валюте snapshot


@dataclass
class CostBreakdown:
    total_per_ton: float                 # полная стоимость 1 т стали в валюте snapshot
    contributions: list[CostContribution]
    mode: CostMode
    currency: Currency

    @property
    def total_per_kg(self) -> float:
        return self.total_per_ton / 1000
```

**Unit-convention (важно, совместимо с существующим кодом):**
- Composition из NSGA-II приходит в **процентах** (`c_pct=0.08`, `mn_pct=1.5`, `nb_pct=0.04` — т.е. 0.08%, 1.5%, 0.04%). `p_pct`, `n_ppm` и т.д. как в `VARIABLE_BOUNDS_HSLA`.
- Конверсия в массу на тонну: `mass_kg_per_ton = pct_value × 10` (потому что 1% = 10 кг на 1000 кг стали).
- `n_ppm` (азот в ppm) в cost не участвует (не закупается как материал, регулируется режимом плавки).

### 4.2 Маппинг элемент → материал

```python
# В cost_model.py
FERROALLOY_PREFERENCE: dict[str, str] = {
    "Mn": "FeMn-80",
    "Si": "FeSi-75",
    "Cr": "FeCr-HC",
    "Ni": "FeNi",
    "Mo": "FeMo",
    "V":  "FeV-50",
    "Nb": "FeNb-65",
    "Ti": "FeTi-70",
    "Cu": "Cu",
    "Al": "Al",
}

# Элементы, НЕ включаемые в cost (приходят с ломом или регулируются процессом):
NON_PRICED_ELEMENTS: set[str] = {"C", "P", "S", "N"}
```

Если в composition встречается элемент вне `FERROALLOY_PREFERENCE` и вне `NON_PRICED_ELEMENTS` — `ValueError` с понятным сообщением.

### 4.3 Формат seed-файла — `data/prices/seed_2026-04-23.yaml`

```yaml
date: 2026-04-23
currency: RUB
source: seed
notes: "Ориентировочный прайс для MVP-демо; проверьте перед продакшеном"

materials:
  scrap:
    kind: base
    price_per_kg: 42.0
    element_content: {Fe: 1.0}
  FeMn-80:
    kind: ferroalloy
    price_per_kg: 180.0
    element_content: {Mn: 0.80, Fe: 0.20}
  FeSi-75:
    kind: ferroalloy
    price_per_kg: 210.0
    element_content: {Si: 0.75, Fe: 0.25}
  FeCr-HC:
    kind: ferroalloy
    price_per_kg: 260.0
    element_content: {Cr: 0.65, Fe: 0.35}
  FeNi:
    kind: ferroalloy
    price_per_kg: 1200.0
    element_content: {Ni: 0.30, Fe: 0.70}
  FeMo:
    kind: ferroalloy
    price_per_kg: 3400.0
    element_content: {Mo: 0.60, Fe: 0.40}
  FeV-50:
    kind: ferroalloy
    price_per_kg: 1900.0
    element_content: {V: 0.50, Fe: 0.50}
  FeNb-65:
    kind: ferroalloy
    price_per_kg: 3600.0
    element_content: {Nb: 0.65, Fe: 0.35}
  FeTi-70:
    kind: ferroalloy
    price_per_kg: 720.0
    element_content: {Ti: 0.70, Fe: 0.30}
  Cu:
    kind: pure
    price_per_kg: 850.0
    element_content: {Cu: 1.0}
  Al:
    kind: pure
    price_per_kg: 240.0
    element_content: {Al: 1.0}
```

### 4.4 Валидация при загрузке snapshot

- `price_per_kg > 0`.
- `abs(sum(element_content.values()) - 1.0) < 0.02`.
- Все keys `element_content` — односимвольные или двухсимвольные обозначения известных элементов.
- Для `kind="ferroalloy"` главный элемент в физически допустимых границах (см. секция 8, C02):
  FeNb 0.55–0.75; FeMn 0.70–0.88; FeSi 0.70–0.80; FeCr 0.55–0.70; FeV 0.40–0.60; FeTi 0.65–0.75; FeMo 0.55–0.70; FeNi 0.20–0.40.
- Snapshot обязан содержать `scrap` (нужен для `mode="full"`).

### 4.5 Валидация на совместимость с design space

```python
def required_elements_for_design(variable_bounds: dict) -> set[str]:
    """Из VARIABLE_BOUNDS_HSLA извлекает элементы, для которых нужна цена.
    Пропускает NON_PRICED_ELEMENTS и не-химические переменные (rolling_finish_temp и т.д.).
    """
    required = set()
    for var in variable_bounds:
        if not var.endswith("_pct"):
            continue
        elem = var[:-4].capitalize()   # "mn_pct" → "Mn"
        if elem in NON_PRICED_ELEMENTS:
            continue
        required.add(elem)
    return required


def validate_snapshot(snapshot: PriceSnapshot, required: set[str]) -> list[str]:
    """Возвращает список непокрытых элементов (для красивого сообщения UI).
    """
    covered = set()
    for m in snapshot.materials.values():
        covered.update(m.element_content.keys())
    return sorted(required - covered)
```

---

## 5. Cost function

### Подпись

```python
def compute_cost(
    composition_pct: dict[str, float],   # {"c_pct": 0.08, "mn_pct": 1.5, "nb_pct": 0.04, ...}
    snapshot: PriceSnapshot,
    mode: CostMode = "full",
) -> CostBreakdown:
    ...
```

### Алгоритм (units: за 1 тонну стали)

1. **Ferroalloy масса на тонну стали:**
   ```
   for var, pct in composition_pct.items():
       if not var.endswith("_pct"): continue
       elem = var[:-4].capitalize()       # "mn_pct" → "Mn"
       if elem in NON_PRICED_ELEMENTS: continue
       if pct <= 0: continue
       material_id = FERROALLOY_PREFERENCE[elem]
       material = snapshot.materials[material_id]
       content = material.element_content[elem]
       elem_mass_kg_per_ton = pct * 10     # 1% = 10 кг на 1 т
       material_mass_kg_per_ton = elem_mass_kg_per_ton / content
       contributions.append(CostContribution(
           material_id=material_id,
           mass_kg_per_ton_steel=material_mass_kg_per_ton,
           price_per_kg=material.price_per_kg,
           contribution_per_ton=material_mass_kg_per_ton * material.price_per_kg,
       ))
   ```

2. **База:**
   ```
   total_alloy_mass = Σ mass_kg_per_ton_steel
   base_mass = max(0.0, 1000.0 - total_alloy_mass)
   scrap = snapshot.materials["scrap"]
   base_contribution = CostContribution(
       material_id="scrap",
       mass_kg_per_ton_steel=base_mass,
       price_per_kg=scrap.price_per_kg,
       contribution_per_ton=base_mass * scrap.price_per_kg,
   )
   ```

3. **Итог:**
   ```
   if mode == "full":
       contributions.insert(0, base_contribution)
       total = sum(c.contribution_per_ton for c in contributions)
   else:  # incremental
       total = sum(c.contribution_per_ton for c in contributions_alloy_only)
   ```

4. Возврат `CostBreakdown(total_per_ton=total, contributions=contributions, mode, currency=snapshot.currency)`.

### Edge cases
- `total_alloy_mass > 1000` (ошибка ввода или сильно легированная сталь на границе bounds) → `base_mass = 0`, warning в логи, но не hard fail (NSGA-II может сгенерировать такие точки на границах).
- `composition_pct` пустая или только base-elements → `total = base_mass × base_price` (в full), `0` (в incremental).
- `"scrap"` отсутствует в snapshot при `mode="full"` → `KeyError` с понятным сообщением (ловится на `validate_snapshot_for_mode`, вызывается в pre-check).

---

## 6. Интеграция с inverse_designer

### 6.1 Что меняется в `app/backend/inverse_designer.py`

**Удаляется:**
- Константа `ELEMENT_PRICES_EUR_PER_KG` (заменяется snapshot-параметром).
- Примитивный cost-счёт в `_evaluate`: `cost += row[key] * 10 * price`.

**Добавляется:**
- Поле `price_snapshot: PriceSnapshot | None` и `cost_mode: CostMode` в `HSLADesignProblem.__init__`.
- Новая сигнатура:
  ```python
  class HSLADesignProblem(ElementwiseProblem):
      def __init__(
          self,
          model_bundle, targets, hard_constraints, variable_bounds,
          price_snapshot: PriceSnapshot | None = None,
          cost_mode: CostMode = "full",
      ):
          ...
          # n_obj остаётся = 3 (target, cost, uncertainty) — формат результата совместим
  ```
- Если `price_snapshot is None` → используется fallback: **тот же старый расчёт** `Σ pct × 10 × ELEMENT_PRICES_EUR_PER_KG` с deprecation warning в логи. Это сохраняет обратную совместимость smoke-теста.
- Если `price_snapshot` задан → в `_evaluate`:
  ```python
  row = dict(zip(self.var_names, x))
  breakdown = compute_cost(
      {k: v for k, v in row.items() if k.endswith("_pct")},
      self.price_snapshot,
      mode=self.cost_mode,
  )
  f2 = breakdown.total_per_ton    # в валюте snapshot
  ```

### 6.2 Сигнатура `run_inverse_design`

```python
def run_inverse_design(
    model_version: str,
    targets: dict,
    hard_constraints: dict | None = None,
    variable_bounds: dict | None = None,
    population_size: int = 80,
    n_generations: int = 60,
    random_seed: int = 42,
    # НОВОЕ:
    price_snapshot: PriceSnapshot | None = None,
    cost_mode: CostMode = "full",
) -> dict:
    ...
```

### 6.3 Pre-check

В начале `run_inverse_design`, перед созданием Problem:

```python
from app.backend.cost_model import (
    validate_snapshot, required_elements_for_design, PriceSnapshotIncomplete,
)

bounds = variable_bounds or VARIABLE_BOUNDS_HSLA

if price_snapshot is not None:
    required = required_elements_for_design(bounds)
    missing = validate_snapshot(price_snapshot, required)
    if missing:
        raise PriceSnapshotIncomplete(missing)
    if cost_mode == "full" and "scrap" not in price_snapshot.materials:
        raise PriceSnapshotIncomplete(["scrap (base material)"])
```

`PriceSnapshotIncomplete` — новый exception в `cost_model.py`:
```python
class PriceSnapshotIncomplete(ValueError):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"Нет цен для: {', '.join(missing)}")
```

### 6.4 Cost breakdown в результатах

```python
for i, x in enumerate(res.X):
    ...
    breakdown = (
        asdict(compute_cost(composition_pct, price_snapshot, cost_mode))
        if price_snapshot else None
    )
    candidates.append({
        ...,
        "objectives": {
            "distance_to_target": float(res.F[i, 0]),
            "alloying_cost": float(res.F[i, 1]),         # уже в валюте snapshot/т
            "prediction_uncertainty": float(res.F[i, 2]),
        },
        "cost": breakdown,                                # NEW; None если legacy режим
    })
```

Также в top-level result:
```python
result["cost_currency"] = price_snapshot.currency if price_snapshot else "EUR (legacy)"
result["cost_mode"] = cost_mode if price_snapshot else "legacy"
result["price_snapshot_path"] = saved_snapshot_path      # путь к YAML, см. 6.5
```

### 6.5 Decision Log snapshot

Перед запуском NSGA-II сохраняем snapshot:

```python
if price_snapshot:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = (
        PROJECT_ROOT / "decision_log" / "price_snapshots" / f"{run_id}.yaml"
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    save_snapshot(price_snapshot, snapshot_path)

    log_decision(
        phase="inverse_design",
        decision=f"Inverse design c cost-optimization "
                 f"(snapshot {price_snapshot.date}, {price_snapshot.currency}, mode={cost_mode})",
        reasoning=f"Source: {price_snapshot.source}, "
                  f"{len(price_snapshot.materials)} материалов. "
                  f"Legacy ELEMENT_PRICES_EUR_PER_KG отключён.",
        context={
            "snapshot_path": str(snapshot_path),
            "currency": price_snapshot.currency,
            "n_materials": len(price_snapshot.materials),
            "cost_mode": cost_mode,
        },
        author="inverse_designer",
        tags=["cost_optimization", str(price_snapshot.date)],
    )
```

---

## 7. UI (Streamlit) — `app/frontend/app.py`

### 7.1 Новая секция на вкладке «🎯 Дизайн сплава»

**Над** кнопкой «Запустить дизайн», **под** целевыми свойствами:

```
┌─ Expander «💰 Прайс материалов» (раскрыт по умолчанию) ─┐
│  Активный snapshot: seed_2026-04-23 · RUB (10 позиций)  │
│  [⬆ Загрузить YAML]   [💾 Скачать текущий]              │
│                                                          │
│  ☑ Учитывать стоимость в оптимизации                    │
│  Валюта: RUB (из snapshot)                              │
│  Режим cost: (●) full (полная себестоимость)            │
│              ( ) incremental (только легирование)       │
│                                                          │
│  ┌─ Редактор цен (st.data_editor, num_rows=dynamic) ─┐  │
│  │ id       │kind     │price/кг │element_content  │    │
│  │ scrap    │base     │  42.00  │Fe=1.00          │    │
│  │ FeMn-80  │ferroal. │ 180.00  │Mn=0.80;Fe=0.20  │    │
│  │ FeSi-75  │ferroal. │ 210.00  │Si=0.75;Fe=0.25  │    │
│  │ …                                               │    │
│  └──────────────────────────────────────────────────┘   │
│  Валидация inline: красная ячейка при ошибке            │
└──────────────────────────────────────────────────────────┘
```

Реализация:
- `st.data_editor(df_prices, num_rows="dynamic", key="price_editor", column_config={...})`.
- `element_content` отображается как строка `"Mn=0.80;Fe=0.20"`; внутри — парсер в `dict[str, float]`.
- Checkbox «Учитывать стоимость» (`st.session_state.use_cost`) — если снят, `price_snapshot=None`, legacy-поведение (для обратной совместимости демо-сценариев и smoke-теста).
- Radio «Режим cost» — session_state, default `full`.
- Кнопка «Скачать» — `st.download_button` с YAML-сериализацией.
- Кнопка «Загрузить» — `st.file_uploader`, парсит YAML, обновляет session_state.

### 7.2 Pre-check handler

```python
try:
    snapshot = _build_snapshot_from_editor(st.session_state.price_editor) \
        if st.session_state.use_cost else None
    result = run_inverse_design(..., price_snapshot=snapshot,
                                 cost_mode=st.session_state.cost_mode)
except PriceSnapshotIncomplete as e:
    st.error(f"❌ В прайсе нет цен для: **{', '.join(e.missing)}**. "
             f"Добавьте строки в таблицу выше.")
    st.stop()
```

### 7.3 Результаты

Между существующими «обзорными метриками» и expander'ами кандидатов — **Pareto plot**:

```python
import altair as alt

df_pareto = pd.DataFrame([{
    "idx": c["idx"],
    "sigma_t": c["predicted"]["mean"],
    "cost": c["cost"]["total_per_ton"] if c.get("cost") else c["objectives"]["alloying_cost"],
    "ood": c["predicted"]["ood_flag"],
    "valid": c.get("validation", {}).get("passed", True),
} for c in result["pareto_candidates"]])

chart = alt.Chart(df_pareto).mark_circle(size=120).encode(
    x=alt.X("sigma_t:Q", title="σт, МПа"),
    y=alt.Y("cost:Q", title=f"Стоимость, {result['cost_currency']}/т"),
    color=alt.Color("valid:N", scale=alt.Scale(domain=[True,False], range=["#2ecc71","#e67e22"])),
    tooltip=["idx","sigma_t","cost","ood"],
).interactive()
st.altair_chart(chart, use_container_width=True)
```

Внутри каждого кандидатного expander — к существующим секциям добавляется:

```
💰 Себестоимость: 47 870 ₽/т (47.87 ₽/кг, mode=full)

Материал  │ Масса, кг/т │ Цена, ₽/кг │ Вклад, ₽/т │ Доля
scrap     │   975.0     │    42.00   │  40 950    │ 86%
FeMn-80   │    18.8     │   180.00   │   3 375    │  7%
FeSi-75   │     5.3     │   210.00   │   1 120    │  2%
FeNb-65   │     0.62    │  3600.00   │   2 215    │  5%
FeTi-70   │     0.29    │   720.00   │     206    │  0%
───────────────────────────────────────────────────
                              Итого: 47 866 ₽/т
[📋 Экспорт в CSV]
```

### 7.4 Обратная совместимость

- Снятая галочка «Учитывать стоимость» → `price_snapshot=None` → старое поведение с `ELEMENT_PRICES_EUR_PER_KG` и объявлением «legacy mode» в UI-warning.
- Smoke-test не меняется (он не задаёт `price_snapshot`).
- После успешного запуска нового пути smoke-test расширяется ещё одним шагом с прайсом (см. секция 9.8).

---

## 8. Pattern Library extensions

Новые проверки в `pattern_library/patterns.py`, все под фазой `Phase.INVERSE_DESIGN`:

| ID | Severity | Условие срабатывания | Сообщение |
|---|---|---|---|
| **C01** | MEDIUM | `(today − snapshot.date).days > 30` | «Прайс старше 30 дней. Актуализируйте перед презентацией/продакшеном.» |
| **C02** | HIGH | Для `material.kind == "ferroalloy"` — главный элемент вне физически допустимого диапазона (таблица в 4.4) | «FeNb-65 содержит Nb=0.90 — физически невозможно. Проверьте ввод.» |
| **C03** | HIGH | В `CostBreakdown.contributions` есть `contribution_per_ton < 0` или `mass_kg_per_ton_steel > 1000` | «Баг в compute_cost: отрицательный вклад или масса > 1000 кг/т.» |
| **C04** | HIGH | Элемент в composition > 0, но нет в preference map ИЛИ не в snapshot (дублирование pre-check на случай нештатного пути) | «Элемент X в кандидате не покрыт прайсом.» |

Critic получает `cost_breakdown` и `snapshot` в context через расширенный `_build_critic_context` в `engine.py` для фазы `inverse_design`:
```python
ctx.update({
    ...,
    "price_snapshot": result.output.get("price_snapshot_meta"),  # date, currency, source
    "cost_breakdown_samples": [c["cost"] for c in state.candidates[:5]],
})
```

---

## 9. Тесты — `app/tests/test_cost_model.py`

(Каталог `app/tests/` сейчас пуст. Этот файл — первый тест.)

1. **`test_compute_cost_pure_iron_full_mode`** — `composition_pct = {}` → только scrap 1000 кг × 42 = 42 000 ₽/т. Contributions содержит один item = "scrap".
2. **`test_compute_cost_pure_iron_incremental_mode`** — `composition_pct = {}`, mode=incremental → total = 0, contributions = [].
3. **`test_compute_cost_nb_mass_math`** — `{"nb_pct": 0.65}` (0.65% Nb).
   - Expected: FeNb-65 mass = 6.5/0.65 = 10 кг/т, alloy cost = 10 × 3600 = 36 000 ₽/т.
   - Full: base_mass = 990 кг × 42 = 41 580 ₽/т. Total = 77 580 ₽/т.
   - Incremental: Total = 36 000 ₽/т.
4. **`test_compute_cost_full_minus_incremental_equals_base`** — для любой composition разность `full - incremental` = `(1000 - total_alloy_mass) × scrap_price`.
5. **`test_snapshot_validation_missing_element`** — snapshot без FeMo, design variables содержат `mo_pct` → `required_elements_for_design(...) ⊇ {"Mo"}`, `validate_snapshot(...) == ["Mo"]`.
6. **`test_snapshot_invalid_element_content_sum`** — YAML с `{Mn: 0.80, Fe: 0.10}` (sum=0.9) → `ValueError` на загрузке.
7. **`test_seed_snapshot_covers_hsla_design_space`** — `seed_snapshot()` покрывает `required_elements_for_design(VARIABLE_BOUNDS_HSLA)`.
8. **`test_cost_sanity_range_typical_hsla`** — composition = `{"c_pct":0.08, "mn_pct":1.5, "si_pct":0.4, "nb_pct":0.04, "ti_pct":0.02, "cu_pct":0.20, "al_pct":0.035}`.
   - Expected full cost ∈ [45 000, 90 000] ₽/т (диапазон для seed RUB).
9. **`test_ferroalloy_content_out_of_physical_range_flagged`** — Material FeNb-65 с Nb=0.90 → паттерн C02 срабатывает.
10. **`test_legacy_fallback_still_works`** — `run_inverse_design(..., price_snapshot=None)` проходит без ошибок (smoke-level test, без NSGA-II — можно мокнуть через direct Problem._evaluate).

---

## 10. Файлы, изменяемые в реализации

### Новые
- `app/backend/cost_model.py` — типы, compute_cost, seed_snapshot, save/load snapshot, валидация.
- `data/prices/seed_2026-04-23.yaml` — seed-прайс из секции 4.3.
- `app/tests/test_cost_model.py` — тесты 1–10.
- `docs/superpowers/specs/2026-04-23-cost-optimization-design.md` — этот документ.
- `decision_log/price_snapshots/` — runtime-папка (создаётся автоматически, `.gitignore`).

### Изменяемые
- `app/backend/inverse_designer.py`:
  - Удалить `ELEMENT_PRICES_EUR_PER_KG` как default-константу (оставить только как legacy-fallback внутри `_evaluate` если `price_snapshot is None`, плюс deprecation-лог).
  - Расширить `HSLADesignProblem.__init__` и `_evaluate`.
  - Расширить `run_inverse_design` сигнатуру и pre-check.
  - Добавить `breakdown` в возвращаемые candidates.
- `app/backend/engine.py` — `_build_critic_context` для `inverse_design` дополнить `price_snapshot` метой и `cost_breakdown_samples`.
- `app/frontend/app.py` — expander прайса с `st.data_editor`, checkbox, radio; Pareto plot; breakdown в per-candidate expander; handler `PriceSnapshotIncomplete`.
- `pattern_library/patterns.py` — C01–C04 с соответствующими `_check_cNN_*` функциями.
- `requirements.txt` — добавить `pyyaml>=6.0` (altair уже идёт со Streamlit).
- `.gitignore` — добавить `decision_log/price_snapshots/`.
- `CLAUDE.md` — под «ML-конвенциями» — строка про `cost_model` и ferroalloy pricing, замена legacy `ELEMENT_PRICES_EUR_PER_KG`.

---

## 11. Допущения и риски

- **Fe из ферросплава «бесплатен»** — покупаем ферросплав целиком (его цена включает Fe), этот Fe замещает часть лома в базовой массе (`base_mass = 1000 − Σ alloy_mass`). Массовый баланс сходится.
- **Угар и потери при плавке не моделируются** — реальная расходная норма +5–8%. Допустимо для MVP; можно добавить коэффициент `process_yield_coeff` в Material позже.
- **Один base material = scrap** — расширение на pig_iron/slab с выбором в UI — следующая итерация.
- **Элементы C/P/S/N не покупаются** — считаем, что приходят с ломом «бесплатно». В реальности шихта имеет ограничения по P/S, и для их снижения нужна дополнительная дорогая обработка. Это не покрывается MVP.
- **Нарушение совместимости Pareto**: существующий `alloying_cost` был в EUR/т по хардкоду, после миграции — в валюте snapshot (RUB/USD/EUR). Старые Decision Log записи с `alloying_cost: 58.3` не сопоставимы со новыми `alloying_cost: 47866`. Mitigation: в Decision Log пишем `cost_currency` и `cost_mode` тегами, старые записи оставляем как есть (не мигрируем).
- **`n_obj` остаётся 3** — `prediction_uncertainty` продолжает быть третьим objective. Визуализировать 3D Pareto сложно, но для MVP показываем проекцию (σт × cost), uncertainty — colour/size точки на графике. Возможное расширение в v2 — toggle «2D / 3D Pareto».

---

## 12. Acceptance criteria

1. ✅ `pytest app/tests/test_cost_model.py -v` — все 10 тестов проходят.
2. ✅ `PYTHONPATH=. python scripts/smoke_test.py` продолжает проходить (legacy-путь без snapshot).
3. ✅ Новый smoke-путь (добавляется в `smoke_test.py`): `run_inverse_design(..., price_snapshot=seed_snapshot(), cost_mode="full")` возвращает Pareto ≥5 кандидатов, у каждого заполнено `cost.total_per_ton` в диапазоне 40 000–150 000 ₽/т.
4. ✅ Streamlit UI: expander прайса редактируется в `st.data_editor`; Pareto plot рисуется; breakdown в per-candidate expander показывает таблицу.
5. ✅ При запуске дизайна с прайсом в `decision_log/price_snapshots/` появляется YAML, в `decisions.db` — запись с tag `cost_optimization`.
6. ✅ Удалить из прайса строку FeNb-65 и запустить дизайн → `PriceSnapshotIncomplete`, в UI `st.error` «Нет цен для: Nb».
7. ✅ Pattern Library: C02 срабатывает при Material FeNb-65 с `Nb=0.90`; C01 — при snapshot старше 30 дней.
8. ✅ Снять галочку «Учитывать стоимость» в UI → дизайн запускается по legacy-пути с deprecation-warning.

---

## 13. Следующие шаги после реализации (out of scope MVP)

- Процессные энергозатраты (₽/МВт·ч × требуемая энергия нагрева + прокатки + термообработки) → cost_model v2.
- Курсы валют + импорт котировок LME для Ni/Mo в автоматическом режиме.
- Tenant capability profile: «этот клиент не использует FeV» → автоматическое исключение из preference map.
- Сравнение «текущая рецептура vs новая» с ROI-дельтой (sales-level feature).
- Вспомогательные материалы: раскислители (Al/Ca), модификаторы, шлакообразующие.
- Multi-base выбор: scrap / pig_iron / slab с разными базовыми ценами.
- 3D Pareto (σт × cost × uncertainty) с интерактивной визуализацией.
- Учёт process yield (угар легирующих), особенно для Ti и Al.
