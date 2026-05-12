# Design: Slag-aware ASIS-модель раскисления с cost-оптимизацией

**Дата:** 2026-05-12
**Регламент:** R-001 (Feature Development), step 2 — design-doc от Architect
**Статус:** DRAFT (ожидает approve от пользователя перед PR-планом)

## Context

Текущий модуль `app/backend/deoxidation.py` — физико-термодинамический калькулятор Al-раскисления стадии LF, оперирующий только растворённым кислородом [O]_a. Реальная плавка BOF→LF, описанная в Excel-калькуляторе пользователя (371-т ASIS-инжекция, шлак переноса 2.2 т FeO=18%), требует учёта **дополнительных источников кислорода** (FeO/MnO/SiO₂ в шлаке) и **выбора метода подачи Al** (чушка / гранула / ASIS-дробь / cored wire / FeAl) с разной эффективностью η_Al (0.50-0.95) и разной стоимостью. Эта фича расширяет deoxidation.py до **slag-aware advisory с cost-оптимизацией метода подачи**, оставаясь physics-only — без ML/feedback-loop/кинетики Stokes (это v0.7+).

## Constraints

**Hard:**
- Сохранить обратную совместимость публичного API `compute_al_demand` / `compute_al_quality` / `compare_all_models` (используется в `app/api/routers/deox.py`, `DeoxidationAdvisor` LLM-pair, smoke_test).
- Соблюдать project-prior №2: деттерминистический core, LLM на краях. Slag-aware код — pure physics в `app/backend/deoxidation.py`.
- Соблюдать project-prior №3: справочные данные в YAML, не зашитые dict-литералы (для editable maintenance метуллургами без правки .py).
- Reproducibility: каждый «оптимальный метод» recommendation должен попадать в Decision Log с tag `deox_method_recommendation`, и snapshot price-of-Al-methods сохраняется в `decision_log/deox_methods_snapshots/<ts>.yaml` (по аналогии с `price_snapshots/`).
- Pattern Library extensions подчиняются конвенции `DX*` ID и `Phase.DEOXIDATION`.
- UI — FastAPI + vanilla JS (Streamlit удалён PR 13). Все добавления в `app/web/static/js/views/deox.js` и `app/api/routers/deox.py`.
- Русский в UI и user-facing messages; английский в parser/validation-уровне ошибок.

**Soft:**
- Минимизировать поверхность изменений. Новая физика — отдельный модуль `slag_aware.py`, **не** в один монолитный `deoxidation.py`.
- Не вводить базовый класс "DeoxidationModel" пока пользователей нового абстракта < 3 (project-prior №5).
- Калибровка η_Al исторически — пока **out-of-scope** (см. §1 Scope). Модель работает на литературных диапазонах из YAML; пользователь может вручную переопределить η_Al в форме.

## Scope

### Входит в эту фичу (v0.6)

1. **Slag-aware O-баланс:** O для удаления = [O]_dissolved + 0.222 × M_slag × %FeO/100. Опционально — учёт MnO (0.226) и SiO₂ (0.533) как extension fields.
2. **Co-deoxidation by Si:** опциональный pre-step — если на плавке вводился FeSi/SiMn до Al, рассчитать сколько O связал Si (η_Si=0.95, стехиометрия O₂/Si=1.14) и **вычесть** из delta_O перед расчётом Al.
3. **Методы подачи Al** как параметр модели: каталог в YAML `data/deox_methods/al_addition_methods.yaml` (метод, η_Al-диапазон, типичный размер, surface m²/kg, premium_per_kg vs commodity Al, carrier-gas требования).
4. **API расширение:**
   - `compute_al_demand_slag_aware(...) -> AlDemandResult` (новая функция).
   - `compare_addition_methods(...) -> MethodComparisonReport` (DataFrame-like).
   - `recommend_optimal_method(...) -> OptimizationRecommendation` (cost-EI с constraints на [Al]_window, [N]_max).
5. **Cost-интеграция:** новая категория material kind `"deox_consumable"` в `cost_model.PriceSnapshot` для метод-specific Al-цен. Базовая `"Al"` остаётся (fallback, commodity ingot).
6. **UI:** новый sub-tab «🎯 Оптимизация метода» (5-й tab во вкладке Раскисление). Existing forward/inverse/compare остаются без изменений (compat).
7. **Pattern Library:** DX04-DX07 (см. §5).
8. **Decision Log:** opt-in запись рекомендации (кнопка «Сохранить рекомендацию», как у существующего forward-tab).

### НЕ входит (v0.7+)

- **Кинетика растворения** Al-дроби (Stokes, число Schmidt, время пенетрации) — пока считается «мгновенно при η из метода».
- **ML-калибровка η_Al** на исторических данных — модель статическая, литература + ручной override.
- **Real-time анализаторы O** (Celox feedback loop) — out-of-scope.
- **Активная интеграция с `active_learner.py`** (DOE на методах подачи) — отдельная задача в v0.8 если данных накопим.
- **Комбинированное Al+FeSi+Ca раскисление с пересчётом активностей по Wagner** — за рамки advisory; cap'аем на простой O-баланс с η_Si=const.
- **Баланс шлака FeO в динамике** (как меняется FeO ПОСЛЕ Al-добавки) — Out-of-scope, FeO считается константой *до* подачи.

## Options considered

### Option A: Новая модель `"asis_slag_aware"` в существующем `THERMO_MODELS` registry

- **Approach:** добавить четвёртый ключ в `THERMO_MODELS` с `log_k=` (берём Fruehan/SE/HY как базу) + flag `slag_aware=True`. `compute_al_demand` ветвится по флагу.
- **Pros:**
  - Минимальная поверхность изменений в API.
  - Один UI dropdown «модель» уже работает — selector переключает.
- **Cons:**
  - **Семантическая путаница:** Fruehan/SE/HY — это **термодинамические модели Al-O равновесия**, а ASIS-slag-aware — это **физика баланса масс + метод подачи**. Это разные слои (термо vs процессная). Смешивать в одном registry приведёт к багам типа «hayashi_2013 vs asis_slag_aware» — пользователь не поймёт, что выбрать.
  - `compute_al_demand` распухает if/else (slag/no-slag, method/no-method, co-deox/no).
  - Pattern Library checks становятся conditional («DX04 only if model_id=asis_slag_aware») — антипаттерн.
- **Touch surface:** `deoxidation.py` (+200 строк в одну функцию), `deox.py` router (расширение AlDemandRequest), `deox.js` (conditional fields в форме).

### Option B: Отдельный модуль `slag_aware.py` + новые функции, термо-models остаются как dependency

- **Approach:** новый файл `app/backend/slag_aware_deox.py` с:
  - `SlagState(M_slag_kg, feo_pct, mno_pct=0.0, sio2_pct=0.0)` dataclass.
  - `AdditionMethod(id, name, eta_al_range, premium_eur_per_kg, ...)` dataclass + loader из YAML.
  - `compute_al_demand_slag_aware(heat: HeatState, slag: SlagState, method: AdditionMethod, target, thermo_model_id, co_deox_si=None) -> AlDemandResult` — внутри:
    1. Считает O в шлаке через стехиометрию (0.222 × M_slag × %FeO/100 + опц. MnO/SiO₂).
    2. Если задан co_deox_si — вычитает O, связанный Si.
    3. Вызывает существующий `compute_al_demand` из `deoxidation.py` с расширенным delta_O и η_Al из метода (через burn_off_pct = (1-η)*100).
  - `compare_addition_methods(heat, slag, target, methods_list) -> list[MethodCompareRow]`.
  - `recommend_optimal_method(heat, slag, target, constraints) -> OptimizationRecommendation`.
- **Pros:**
  - **Чёткое разделение слоёв:** термодинамика (deoxidation.py) vs процессная физика (slag_aware_deox.py).
  - Существующий API не ломается, добавляются новые функции — расширение, не модификация.
  - Pattern Library расширяется новыми DX04+ безусловно (новые проверки активируются на новых ctx-ключах `slag_state`, `method_id`).
  - Termo-models переиспользуются как dependency — DRY.
  - Тестируется изолированно — `test_slag_aware_deox.py` отдельный.
- **Cons:**
  - Два файла вместо одного — выше когнитивная нагрузка при первом чтении.
  - UI должен решать какую функцию звать (existing forward vs new optimization-tab).
- **Touch surface:**
  - `app/backend/slag_aware_deox.py` — NEW (~250 строк).
  - `data/deox_methods/al_addition_methods.yaml` — NEW.
  - `app/backend/cost_model.py` — добавить `kind="deox_consumable"` в Literal type, loader пропускает валидацию суммы element_content для этого kind (поскольку это форма Al, а не сплав).
  - `pattern_library/patterns.py` — 4 новых паттерна + расширение `_build_critic_context` (но `Phase.DEOXIDATION` ctx собирается в router'е, не в engine).
  - `app/api/routers/deox.py` — новый POST `/api/deox/optimize` + `/api/deox/methods` (GET catalog).
  - `app/web/static/js/views/deox.js` — пятый sub-tab «🎯 Оптимизация».
  - `app/tests/test_slag_aware_deox.py` — NEW (≥8 тестов, см. §QA).

### Option C: Глубокая переработка — единый «DeoxidationCalculator» класс с pluggable strategies

- **Approach:** ввести базовый класс `DeoxidationStrategy` с методами `compute_o_demand()`, `compute_al_addition()`, и подклассы `BasicThermoStrategy`, `SlagAwareStrategy`, `AsisOptimizedStrategy`.
- **Pros:** красивая ООП-архитектура, легко добавлять стратегии.
- **Cons:**
  - **Нарушает project-prior №5** («No new abstractions without 3+ users»): сейчас 2 use-case (basic forward + slag-aware), вводить базовый класс преждевременно.
  - Полный rewrite существующего `compute_al_demand` — ломает обратную совместимость API.
  - Высокий риск регресса (smoke_test, LLM-advisor pair, UI).
- **Touch surface:** весь `deoxidation.py` rewrite + router + tests.

## Recommendation

**Option B** (отдельный модуль `slag_aware_deox.py` + YAML-каталог методов).

**Rationale:** даёт чистое разделение слоёв (термодинамика остаётся в `deoxidation.py` без изменений → нет регресса), позволяет расширять каталог методов через YAML без правки кода (project-prior №3), и не вводит преждевременную абстракцию (project-prior №5). Cost-интеграция через мягкое расширение `cost_model.py` (новый `kind`) — без рефакторинга core логики `compute_cost`. Это самый дешёвый по maintenance путь и самый консистентный с дизайном `cost_model` / `steel_classes` (декларативный YAML + executive functions).

## Implementation outline (для Developer)

### 1. YAML-каталог методов подачи Al
- `data/deox_methods/al_addition_methods.yaml`. Схема:
```yaml
version: "2026-05-12"
source: "Литературный обзор + Excel-калькулятор пользователя (371 т ASIS BOF)"
methods:
  ingot:
    name: "Чушка Al"
    size_mm: [1000, 10000]      # 1-10 кг масса; size_mm используется только для информативности
    surface_m2_per_kg: 0.02
    eta_al_range: [0.50, 0.65]
    eta_al_typical: 0.58
    premium_eur_per_kg: 0.0      # commodity baseline
    carrier_gas: null
    notes: "Атмосферное окисление 20-25%, шлак 55-60%, эффективное раскисление ≤20%"
  asis_shot:
    name: "ASIS-дробь (инжекция)"
    size_mm: [1.6, 6.0]
    surface_m2_per_kg: 1.7
    eta_al_range: [0.75, 0.90]
    eta_al_typical: 0.82
    premium_eur_per_kg: 0.30
    carrier_gas: "Ar"            # required for [N]<50 ppm; N2 allowed if [N] target relaxed
    notes: "Высокая эффективность за счёт развитой поверхности и инжекции под зеркало."
  granule_water_quenched: { ... }
  cored_wire_feal30: { ... }
  submerged_ingot: { ... }
```
- Дополнительные поля по необходимости (T_drying_max_c для гранул — для DX05).
- Loader: `slag_aware_deox.load_addition_methods() -> dict[str, AdditionMethod]` с кешем (lru_cache).

### 2. Datamodel в `slag_aware_deox.py`

```python
@dataclass(frozen=True)
class AdditionMethod:
    id: str
    name: str
    eta_al_typical: float
    eta_al_range: tuple[float, float]
    premium_eur_per_kg: float
    carrier_gas: str | None
    surface_m2_per_kg: float
    notes: str
    raw: dict  # full YAML row for forward-compat

@dataclass(frozen=True)
class SlagState:
    mass_kg: float           # kg slag (per heat, not per ton)
    feo_pct: float
    mno_pct: float = 0.0
    sio2_pct: float = 0.0

@dataclass(frozen=True)
class CoDeoxSi:
    """Optional Si pre-deoxidation block."""
    fesi_added_kg: float
    fesi_si_content_pct: float = 75.0
    eta_si: float = 0.95  # 95% Si burns to SiO2

@dataclass
class SlagAwareDemandResult:
    al_total_kg: float
    al_active_kg: float
    al_burn_off_kg: float
    o_in_dissolved_kg: float
    o_in_slag_kg: float
    o_consumed_by_si_kg: float   # 0 if no co-deox
    o_total_to_remove_kg: float
    method_id: str
    eta_al_used: float
    cost_eur: float
    cost_breakdown: dict   # {al_commodity_eur, al_premium_eur, gas_eur, handling_eur}
    thermo_model_id: str
    inputs: dict
    warnings: list[str]

@dataclass
class MethodCompareRow:
    method_id: str
    method_name: str
    eta_al_used: float
    al_total_kg: float
    cost_per_heat_eur: float
    cost_per_ton_eur: float
    scatter_kg: float   # ± range from η_al_range
    notes: str

@dataclass
class OptimizationRecommendation:
    chosen_method_id: str
    chosen_method_name: str
    rationale: str
    runner_up_method_id: str | None
    runner_up_delta_eur: float
    constraints_active: list[str]
    pareto_table: list[MethodCompareRow]
```

### 3. Функции

- `compute_o_from_slag(slag: SlagState) -> float` — стехиометрия: `o_kg = M_slag × (0.222·%FeO + 0.226·%MnO + 0.533·%SiO2) / 100`.
- `compute_o_consumed_by_si(co_deox: CoDeoxSi) -> float` — `(fesi_kg × si_content/100 × η_si) × (32/28) / 28 × 16` (упрощённо: O₂/Si=1.14 по массе).
- `compute_al_demand_slag_aware(...)` — собирает delta_O = O_dissolved + O_slag − O_si_consumed, вызывает существующий `deoxidation.compute_al_demand` с переопределённым `burn_off_pct = (1 − η_al)×100` из выбранного method, добавляет cost breakdown с premium.
- `compare_addition_methods(...)` — пробегает все методы из YAML, возвращает list[MethodCompareRow] отсортированный по cost_per_heat_eur.
- `recommend_optimal_method(...)` — фильтрует методы по constraints (carrier_gas=Ar если target_n<50, premium-cap если задан), возвращает min-cost + runner-up.

### 4. Cost-model расширение

- В `cost_model.py` добавить `kind="deox_consumable"` в `Kind` Literal.
- `_validate_material_dict`: для `kind="deox_consumable"` пропустить проверку суммы element_content (это не сплав), но **требовать** ключ `addition_method_id` (ссылка на YAML-каталог).
- В price snapshot можно опционально регистрировать method-specific Al-цены (e.g. `Al-ASIS-shot` с premium):
```yaml
Al-asis-shot:
  kind: deox_consumable
  price_per_kg: 2.70
  element_content: {Al: 1.0}
  addition_method_id: asis_shot
```
- Это **опциональное** расширение — slag_aware_deox по умолчанию читает premium из YAML методов; если в snapshot есть деteil-specific цена, она побеждает.

### 5. Pattern Library — новые DX*

| ID | Severity | Trigger | Suggestion |
|---|---|---|---|
| DX04 | **HIGH** | slag-aware расчёт запущен без `slag_state` (M_slag=None или feo_pct=None) | «Slag-aware расчёт требует измерение M_slag и %FeO шлака переноса. Запросите данные у LF-оператора или используйте basic forward.» |
| DX05 | MEDIUM | method=`granule_water_quenched` и T_сушки > 200 °C (из form input) | «Гранулы Al при T_сушки > 200 °C дают H-pickup; снизьте до ≤150 °C или выберите другой метод.» |
| DX06 | MEDIUM | пользователь вручную ввёл η_Al вне `eta_al_range` метода более чем на ±5% | «Введённый η_Al отклоняется от литературного диапазона для выбранного метода. Если основано на исторических данных — рекомендуется калибровка на ≥30 плавках (v0.7).» |
| DX07 | **HIGH** | method.carrier_gas=`N2` и target [N] < 50 ppm | «N₂ как несущий газ повысит [N] на 5-15 ppm. Для марок с [N]<50 ppm используйте Ar.» |

- Реализация в `pattern_library/patterns.py`: 4 новых `_check_dx0X` функции, добавление в `PATTERNS` list.
- `Phase.DEOXIDATION` ctx расширяется: `{"slag_state": SlagState|None, "method": AdditionMethod|None, "co_deox_si": CoDeoxSi|None, "user_override_eta_al": float|None, "target_n_ppm": float|None, "t_drying_c": float|None}`.
- Сборка ctx — в `app/api/routers/deox.py` (там, где DX01-DX03 уже собираются для существующих endpoint'ов), новый helper `_build_slag_aware_critic_ctx()`.

### 6. API endpoints (`app/api/routers/deox.py`)

- `GET /api/deox/methods` → `{items: [AdditionMethod.raw, ...], default: "asis_shot"}` — для UI dropdown.
- `POST /api/deox/optimize` body: `OptimizationRequest`:
```python
class OptimizationRequest(BaseModel):
    # Block A — heat
    steel_mass_ton: float = Field(..., ge=1.0, le=500.0)
    o_a_initial_ppm: float
    temperature_C: float
    target_o_a_ppm: float
    # Block B — slag
    slag_mass_kg: float | None = None
    slag_feo_pct: float | None = None
    slag_mno_pct: float = 0.0
    slag_sio2_pct: float = 0.0
    # Block C — co-deox optional
    co_deox_fesi_kg: float | None = None
    co_deox_fesi_si_content_pct: float = 75.0
    # Block D — method (single or "all")
    method_ids: list[str] | None = None  # None = compare all
    user_override_eta_al: float | None = None
    t_drying_c: float | None = None       # for DX05
    # Block E — constraints
    target_n_ppm: float | None = None     # for DX07
    al_window_pct: tuple[float, float] | None = None
    # Block F — economics
    thermo_model_id: str = DEFAULT_MODEL_ID
    use_price_snapshot: bool = True       # uses seed_2026-04-26 by default
```
- Response: `OptimizationResponse = {recommendation: OptimizationRecommendation, pattern_warnings: [...], thermo_model_used: str, price_snapshot_date: str}`.
- `POST /api/deox/optimize/save` — opt-in запись в Decision Log (см. §7).

### 7. Decision Log integration

- Новый tag `deox_method_recommendation` для `log_decision()`.
- При opt-in save:
```python
log_decision(
    phase="deoxidation",
    decision=f"method={rec.chosen_method_id}; al_kg={rec.al_total_kg:.1f}; cost={rec.cost_per_heat_eur:.0f}€",
    reasoning=rec.rationale,
    alternatives_considered=[r.method_id for r in rec.pareto_table if r.method_id != rec.chosen_method_id],
    context={
        "heat_id": <user-provided>,
        "slag_state": asdict(slag),
        "thermo_model": thermo_model_id,
        "methods_snapshot_path": f"decision_log/deox_methods_snapshots/{ts}.yaml",
        "price_snapshot_date": snapshot.date.isoformat(),
    },
    author="user",
)
```
- Каждый save копирует **текущий** `al_addition_methods.yaml` в `decision_log/deox_methods_snapshots/<ISO-ts>.yaml` (gitignored). Это reproducibility якорь: даже если YAML методов поменяется, recommendation воспроизводится из snapshot'а.

### 8. UI — пятый sub-tab «🎯 Оптимизация метода»

- Не модифицировать существующие forward/inverse/compare/ai sub-tabs (zero-regression правило).
- Layout нового tab:
  - **Блок A (Плавка):** M_heat, [O]_init, T_steel, target O_a (default from active class), опц. target [N].
  - **Блок B (Шлак):** M_slag (кг), %FeO, опц. %MnO, %SiO₂. С checkbox «нет данных по шлаку → basic forward» (выключает slag-aware).
  - **Блок C (Co-deoxidation, collapsed):** «Введены FeSi/SiMn до Al?» — если да, поля fesi_kg / si_content.
  - **Блок D (Методы):** multi-select из `/api/deox/methods` (default all), checkbox «Use literature η_Al» vs override input.
  - **Блок E (Constraints):** carrier_gas filter, premium cap.
  - **Кнопка «🎯 Найти оптимальный»** → POST /api/deox/optimize.
  - **Результат:**
    - Recommendation card: chosen method + Al_kg + cost/heat + cost/ton + rationale.
    - Pareto table: все методы отсортированы по cost, с scatter-error bars (от η_al_range).
    - Pattern warnings (DX04-DX07) — severity-coloured banners (re-use existing component).
    - Кнопка «💾 Сохранить рекомендацию» → POST /api/deox/optimize/save.

### 9. Reproducibility

- Каждый `/api/deox/optimize` response включает `methods_snapshot_hash` (SHA256 от al_addition_methods.yaml на момент запроса) — UI отображает в footer.
- При save: копия YAML + price snapshot date + git commit hash (из `subprocess.check_output(['git','rev-parse','HEAD'])` если в repo).

## Files to touch

- `app/backend/slag_aware_deox.py` — **NEW**, ~280 строк (dataclasses + 5 функций + YAML loader с lru_cache).
- `data/deox_methods/al_addition_methods.yaml` — **NEW**, 5 методов × ~10 полей.
- `app/backend/cost_model.py` — добавить `"deox_consumable"` в `Kind` Literal; relaxed валидация для нового kind в `_validate_material_dict`; минимально.
- `pattern_library/patterns.py` — 4 новых `_check_dx0X` функции + 4 записи в `PATTERNS`. Расширение `_build_critic_context` **не нужно** (ctx собирается в router'е).
- `app/api/routers/deox.py` — 3 новых endpoint'а: `GET /api/deox/methods`, `POST /api/deox/optimize`, `POST /api/deox/optimize/save`. Helper `_build_slag_aware_critic_ctx`.
- `app/api/routers/deox.py` — расширить pydantic schemas (`OptimizationRequest`, `OptimizationResponse`).
- `app/web/static/js/views/deox.js` — добавить SUBTAB `optimize`, формы блоков A-F, render результата. ~350 строк.
- `app/web/static/css/...` — переиспользовать существующие стили; новых классов не вводим (project-prior №5).
- `decision_log/logger.py` — НЕ трогать; новый tag passes through существующий `log_decision()`.
- `decision_log/deox_methods_snapshots/` — добавить в `.gitignore` (как `price_snapshots/`).
- `app/tests/test_slag_aware_deox.py` — **NEW**, ≥10 тестов:
  - test_compute_o_from_slag_basic (FeO-only)
  - test_compute_o_from_slag_with_mno_sio2
  - test_co_deox_si_subtracts_o
  - test_demand_matches_excel_base_case (371 т, 657 ppm, M_slag=2200, FeO=18, target=8 ppm → Al ≈ 280-320 kg, проверить в пределах ±10%)
  - test_compare_methods_ordered_by_cost
  - test_recommend_filters_by_n_constraint (target_n=30 ppm → исключает N₂-carrier методы)
  - test_recommend_returns_runner_up
  - test_method_yaml_loader_validates_required_fields
  - test_cost_model_accepts_deox_consumable_kind
  - test_pattern_dx04_blocks_when_slag_missing
  - test_pattern_dx07_blocks_n2_carrier_for_low_n_target
- `app/tests/test_deoxidation.py` — добавить **regression**-тесты что существующий API не сломан (compat smoke).
- `scripts/smoke_test.py` — добавить slag-aware шаг (опционально, если smoke короткий) ИЛИ отдельный `scripts/optimize_deox_method.py` demo.

## Tradeoffs accepted

- **Не калибруем η_Al на исторических данных** — приняли литературные диапазоны из YAML. Точность ±0.10-0.15 для конкретного предприятия. Это compromise ради scope; калибровка — отдельный feature v0.7 после accumulation реальных данных.
- **Не моделируем кинетику Stokes** для дроби — η_Al применяется как const, не зависит от V_inj/d_p/Q_gas. Те input-поля Block C из задания (V_inj, d_p, Q_gas, H_nozzle) приняты в UI **только для документирования** (попадают в Decision Log context), но не влияют на расчёт. Маркируем поля как «informational, for log».
- **Slag-FeO считается постоянным** до и после Al-добавки — динамика FeO-восстановления не моделируется. Приемлемо для advisory; погрешность в пределах ошибки термо-модели (±20-40 ppm O_a).
- **active_learner.py интеграция out-of-scope** — нет cost-EI DOE на методах; recommend_optimal_method работает на static literature ranges.
- **Один UI sub-tab «Оптимизация»**, не два (cost-comparison не подмешиваем в существующий Compare; Compare остаётся для термо-моделей). Это сохраняет менталку «Compare = thermo, Optimize = method».

## Build sequence (PR plan)

Атомарные PR'ы в порядке зависимостей. Каждый PR — отдельный коммит на `feature/asis-slag-aware-deox` (новая ветка от `feature/cost-optimization` или main, см. open questions).

| PR | Title | Scope | Зависит от |
|---|---|---|---|
| **PR 1** | `feat(deox): YAML catalog для методов подачи Al` | `data/deox_methods/al_addition_methods.yaml` + `slag_aware_deox.py` (только dataclasses + `load_addition_methods()`) + test_method_yaml_loader_validates_required_fields | — |
| **PR 2** | `feat(deox): slag-aware O-balance + Si co-deox` | `slag_aware_deox.compute_o_from_slag`, `compute_o_consumed_by_si`, dataclasses SlagState/CoDeoxSi + 3 теста | PR 1 |
| **PR 3** | `feat(deox): compute_al_demand_slag_aware + cost integration` | Главная функция + `kind="deox_consumable"` в cost_model + Excel-base-case regression тест | PR 2 |
| **PR 4** | `feat(deox): compare_addition_methods + recommend_optimal_method` | Cost-EI / constraint filtering + 4 теста | PR 3 |
| **PR 5** | `feat(deox): Pattern Library DX04-DX07` | 4 новых паттерна + тесты + расширение ctx в router-helper | PR 4 |
| **PR 6** | `feat(deox): API endpoints — methods/optimize/save` | 3 router endpoint'а + pydantic schemas + smoke через httpx TestClient | PR 5 |
| **PR 7** | `feat(deox): UI sub-tab «🎯 Оптимизация метода»` | `deox.js` extension — 5-й sub-tab, формы блоков A-F, render результата | PR 6 |
| **PR 8** | `feat(deox): Decision Log integration + methods snapshot` | Opt-in save endpoint, snapshot YAML копирование, `.gitignore` записи | PR 7 |

Каждый PR проходит R-001 цикл: Architect (если есть design-уровень) → Developer → QA (pytest + smoke) → Reviewer → MLOps (merge). PR 1-2 могут идти параллельно с PR 3 после готовности dataclass'ов.

## Open questions for user

1. **Калибровка η_Al** — подтверждаешь что **out-of-scope** для v0.6? Альтернатива — добавить тонкий CSV-loader сейчас (без ML), просто чтобы пользователь мог ввести «у нас на заводе η_Al=0.78» как override per-метод. Это +1 день работы, но повышает «realness» advisory.
2. **Ветка** — стартовать `feature/asis-slag-aware-deox` от `main` (clean) или от текущей `feature/cost-optimization` (build on top)? Я бы шёл от `main`, так как cost-optimization уже в PR-фазе и смешивать рискованно для review.
3. **AI advisor pair** — обновлять ли существующий `DeoxidationAdvisor` LLM-prompt с упоминанием новых методов и slag-aware? Или это отдельный feature v0.7? Если да — это +PR в plan'е на правку gitignored `prompts/deoxidation_advisor.md`.
4. **Excel base-case validation** — Excel-калькулятор пользователя даёт расход Al (или формулу с empirical 0.89 коэффициентом). Можно ли получить numerical таблицу 10-15 плавок (O_init, T, M_slag, FeO, Al_factual) для validation теста? Без неё regression-тест работает только на одной точке = слабый якорь.
5. **DX06 калибровка-suggestion** — текст DX06 сейчас говорит «v0.7 калибровка». Если open question №1 решит включить CSV-override, текст DX06 надо менять. Подтверди order решений.
6. **UI sub-tab или модал?** Сейчас 4 sub-tabs (forward/inverse/compare/ai). Добавление 5-го может сделать tab-bar тесным. Альтернатива — кнопка «🎯 Оптимизировать метод» внутри существующего forward-tab, разворачивает overlay/modal. Я рекомендую отдельный sub-tab (явнее, проще обнаружить), но это UX-call за пользователем.
