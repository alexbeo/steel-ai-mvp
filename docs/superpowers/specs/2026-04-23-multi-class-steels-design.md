# Multi-class steels — pipe HSLA + EN 10083-2 Q&T

**Дата:** 2026-04-23
**Статус:** Design (готов к implementation plan)
**Baseline:** tag `v0.3-llm-critic` (~commit `21495fd`)

---

## 1. Цель и мотивация

Текущий MVP жёстко параметризован под pipe-HSLA: `PIPE_HSLA_FEATURE_SET` в `feature_eng.py`, `VARIABLE_BOUNDS_HSLA` в `inverse_designer.py`, hardcoded bounds/expected features в `patterns.py` и `data_curator.py`. Для демонстрации потенциальным клиентам **европейского** рынка нужно показать, что архитектура не привязана к одному классу.

**Цель:** ввести понятие `SteelClassProfile` и поддержку **двух** классов в первой итерации:
1. `pipe_hsla` — текущий baseline (API 5L X60-X70 / EN 10208-2).
2. `en10083_qt` — европейские heat-treatable углеродистые стали (C22/C35/C45/C60 по EN 10083-2). **Никаких советских марок** — только европейская номенклатура.

Каждый класс имеет свой feature set, physical bounds, target-свойства, synthetic-генератор, expected top-features для Critic M05. Класс — **атрибут модели** (в `meta.json`); пользователь выбирает класс один раз при обучении, и все downstream-вкладки следуют классу активной модели.

---

## 2. Scope

### В MVP входит (scope A из brainstorm)
- 2 класса: `pipe_hsla` + `en10083_qt`.
- YAML-профили в `data/steel_classes/<id>.yaml`: feature_set, physical_bounds, target_properties, expected_top_features, process_params, synthetic_generator_name.
- Python synthetic-генераторы (разные физики): `generate_synthetic_hsla_dataset()` (существующий) + `generate_synthetic_en10083_qt_dataset()` (новый).
- `TrainedModel.steel_class` + сохранение в `meta.json`.
- Per-class feature engineering роутер `compute_features_for_class(df, class_id)`.
- Pattern Library `_check_m05` / `_check_d07` читают expected features и bounds из Critic context (per-class).
- UI:
  - Sidebar: badge класса активной модели.
  - Tab «🤖 Обучение»: dropdown «Класс стали» + target-dropdown фильтруется по классу.
  - Tab «📊 Прогноз»: поля ввода — по `feature_set` активной модели.
  - Tab «🎯 Дизайн сплава»: если активная модель — Q&T, disabled + banner «Inverse design пока HSLA-only».

### Явно вне MVP
- **Inverse design для Q&T** — NSGA-II остаётся pipe-HSLA-only. Это значит: cost-optimization, Pareto plot, breakdown — всё только для HSLA. Для Q&T пользователь получает train + single-state predict.
- Реальные данные — по-прежнему синтетика. Загрузка клиентских CSV — следующая фаза.
- Дополнительные классы (стрёсс-свободные S235/S275/S355, инструментальные, нержавеющие, подшипниковые).
- Совместимость моделей между классами (модель HSLA не применима к Q&T — блокируется через `steel_class` чек, но не конвертируется).
- LLM-Critic per-class expected features (Critic v2 использует общий pipe-HSLA system prompt — обновить в v3).

---

## 3. Архитектурные решения (из brainstorm)

| Решение | Выбор | Почему |
|---|---|---|
| Scope классов | HSLA + EN 10083-2 Q&T | Максимальный физический контраст для демо |
| Номенклатура | Европейские стандарты | Целевой рынок — EU; не использовать ГОСТ/советские марки |
| Config формат | YAML + Python гибрид | Данные — в YAML (редактируемо), логика (синтетик) — в Python |
| UI активации класса | Атрибут модели | Один выбор при обучении, всё downstream следует автоматически; нельзя «перепутать» модели и классы |
| Target для Q&T | `hardness_hrc` основной + `tensile_strength_mpa` | HRC — характерный target для heat-treatable steels |
| Target dropdown | Фильтруется по классу | UX честный, Critic физики per-class |
| Inverse design для Q&T | Отложен | Cost-optimization уже заточена под HSLA; Q&T inverse — значительная работа, не оправдана для демо |

---

## 4. Компоненты

### 4.1 YAML профили — `data/steel_classes/`

#### `pipe_hsla.yaml`

```yaml
id: pipe_hsla
name: "Pipe HSLA (API 5L X60-X70)"
standard: "API 5L / EN 10208-2"
target_properties:
  - id: yield_strength_mpa
    label: "σт, МПа"
    range: [380, 800]
  - id: tensile_strength_mpa
    label: "σв, МПа"
    range: [450, 900]
  - id: elongation_pct
    label: "δ, %"
    range: [15, 45]
  - id: kcv_neg60_j_cm2
    label: "KCV-60, Дж/см²"
    range: [20, 300]
feature_set:
  - c_pct
  - si_pct
  - mn_pct
  - p_pct
  - s_pct
  - cr_pct
  - ni_pct
  - mo_pct
  - cu_pct
  - al_pct
  - v_pct
  - nb_pct
  - ti_pct
  - n_ppm
  - rolling_finish_temp
  - cooling_rate_c_per_s
physical_bounds:
  c_pct: [0.04, 0.12]
  si_pct: [0.15, 0.55]
  mn_pct: [0.90, 1.75]
  p_pct: [0.005, 0.025]
  s_pct: [0.002, 0.012]
  cr_pct: [0.0, 0.30]
  ni_pct: [0.0, 0.40]
  mo_pct: [0.0, 0.10]
  cu_pct: [0.05, 0.35]
  al_pct: [0.020, 0.050]
  v_pct: [0.0, 0.10]
  nb_pct: [0.0, 0.06]
  ti_pct: [0.0, 0.025]
  n_ppm: [30.0, 80.0]
  rolling_finish_temp: [750.0, 850.0]
  cooling_rate_c_per_s: [8.0, 28.0]
expected_top_features:
  - c_pct
  - mn_pct
  - nb_pct
  - ti_pct
  - v_pct
  - rolling_finish_temp
  - cooling_rate_c_per_s
  - cev_iiw
  - pcm
  - microalloying_sum
process_params:
  - rolling_finish_temp
  - cooling_rate_c_per_s
synthetic_generator_name: pipe_hsla
cost_seed_path: data/prices/seed_2026-04-23.yaml
feature_engineering: compute_hsla_features
```

#### `en10083_qt.yaml`

```yaml
id: en10083_qt
name: "Q&T Carbon Steels (EN 10083-2)"
standard: "EN 10083-2"
target_properties:
  - id: hardness_hrc
    label: "Твёрдость, HRC"
    range: [20, 65]
  - id: tensile_strength_mpa
    label: "σв, МПа"
    range: [500, 1500]
feature_set:
  - c_pct
  - si_pct
  - mn_pct
  - p_pct
  - s_pct
  - cr_pct
  - austenitizing_temp
  - tempering_temp
  - tempering_time_min
  - section_thickness_mm
physical_bounds:
  c_pct: [0.18, 0.65]
  si_pct: [0.15, 0.40]
  mn_pct: [0.40, 0.80]
  p_pct: [0.0, 0.035]
  s_pct: [0.0, 0.035]
  cr_pct: [0.0, 0.40]
  austenitizing_temp: [820.0, 900.0]
  tempering_temp: [150.0, 650.0]
  tempering_time_min: [30.0, 180.0]
  section_thickness_mm: [10.0, 100.0]
expected_top_features:
  - c_pct
  - tempering_temp
  - austenitizing_temp
  - mn_pct
  - section_thickness_mm
process_params:
  - austenitizing_temp
  - tempering_temp
  - tempering_time_min
  - section_thickness_mm
synthetic_generator_name: en10083_qt
cost_seed_path: data/prices/seed_2026-04-23.yaml
feature_engineering: passthrough
```

### 4.2 `app/backend/steel_classes.py` — loader + registry

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEEL_CLASSES_DIR = PROJECT_ROOT / "data" / "steel_classes"
AVAILABLE_CLASS_IDS = ["pipe_hsla", "en10083_qt"]


@dataclass
class TargetProperty:
    id: str
    label: str
    range: list[float]


@dataclass
class SteelClassProfile:
    id: str
    name: str
    standard: str
    target_properties: list[TargetProperty]
    feature_set: list[str]
    physical_bounds: dict[str, list[float]]
    expected_top_features: list[str]
    process_params: list[str]
    synthetic_generator_name: str
    cost_seed_path: str
    feature_engineering: str   # "compute_hsla_features" | "passthrough"

    def target_ids(self) -> list[str]:
        return [t.id for t in self.target_properties]


_PROFILE_CACHE: dict[str, SteelClassProfile] = {}


def load_steel_class(class_id: str) -> SteelClassProfile:
    if class_id in _PROFILE_CACHE:
        return _PROFILE_CACHE[class_id]
    path = STEEL_CLASSES_DIR / f"{class_id}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    profile = SteelClassProfile(
        id=data["id"], name=data["name"], standard=data["standard"],
        target_properties=[TargetProperty(**t) for t in data["target_properties"]],
        feature_set=data["feature_set"],
        physical_bounds={k: list(v) for k, v in data["physical_bounds"].items()},
        expected_top_features=data["expected_top_features"],
        process_params=data["process_params"],
        synthetic_generator_name=data["synthetic_generator_name"],
        cost_seed_path=data.get("cost_seed_path", ""),
        feature_engineering=data.get("feature_engineering", "passthrough"),
    )
    _PROFILE_CACHE[class_id] = profile
    return profile


def available_steel_classes() -> list[SteelClassProfile]:
    return [load_steel_class(cid) for cid in AVAILABLE_CLASS_IDS]


def get_synthetic_generator(generator_name: str) -> Callable:
    """Lazy import to avoid circular deps."""
    from app.backend.data_curator import (
        generate_synthetic_hsla_dataset,
        generate_synthetic_en10083_qt_dataset,
    )
    return {
        "pipe_hsla": generate_synthetic_hsla_dataset,
        "en10083_qt": generate_synthetic_en10083_qt_dataset,
    }[generator_name]


def compute_features_for_class(df, class_id: str):
    profile = load_steel_class(class_id)
    if profile.feature_engineering == "compute_hsla_features":
        from app.backend.feature_eng import compute_hsla_features
        return compute_hsla_features(df)
    return df  # passthrough (Q&T — no derived features)
```

### 4.3 Synthetic generator для Q&T — `data_curator.py`

```python
def generate_synthetic_en10083_qt_dataset(
    n_samples: int = 2000, random_seed: int = 42
) -> pd.DataFrame:
    """EN 10083-2 Q&T carbon steels (C22/C35/C45/C60).

    Physical model:
    - HRC_quenched = 20 + 85*C + 3*ln(Mn+0.5) − 0.05*thickness
    - HRC_tempered = HRC_quenched − 0.4*((temper_T − 150)/10)*ln(1 + temper_t/30)
    - tensile_strength ≈ 34.5*HRC (empirical)
    """
    rng = np.random.default_rng(random_seed)
    c  = rng.uniform(0.18, 0.65, n_samples)
    si = rng.uniform(0.15, 0.40, n_samples)
    mn = rng.uniform(0.40, 0.80, n_samples)
    p  = rng.uniform(0.0, 0.035, n_samples)
    s  = rng.uniform(0.0, 0.035, n_samples)
    cr = rng.uniform(0.0, 0.40, n_samples)
    austenit_T = rng.uniform(820.0, 900.0, n_samples)
    temper_T   = rng.uniform(150.0, 650.0, n_samples)
    temper_t   = rng.uniform(30.0, 180.0, n_samples)
    thick_mm   = rng.uniform(10.0, 100.0, n_samples)

    hrc_q = 20 + 85 * c + 3 * np.log(mn + 0.5) - 0.05 * thick_mm
    hrc_q = np.clip(hrc_q, 20, 65)
    temper_loss = ((temper_T - 150) / 10) * np.log1p(temper_t / 30)
    hrc = hrc_q - 0.4 * temper_loss
    hrc = np.clip(hrc + rng.normal(0, 1.5, n_samples), 15, 65)
    tensile = 34.5 * hrc * 10 + rng.normal(0, 50, n_samples)
    tensile = np.clip(tensile, 400, 1600)

    campaign_id = rng.integers(1, 50, n_samples)
    heat_date = pd.date_range("2024-01-01", periods=n_samples, freq="2h")

    return pd.DataFrame({
        "c_pct": c, "si_pct": si, "mn_pct": mn, "p_pct": p, "s_pct": s, "cr_pct": cr,
        "austenitizing_temp": austenit_T, "tempering_temp": temper_T,
        "tempering_time_min": temper_t, "section_thickness_mm": thick_mm,
        "hardness_hrc": hrc,
        "tensile_strength_mpa": tensile,
        "campaign_id": campaign_id,
        "heat_date": heat_date,
    })
```

Сохраняется в `data/hsla_en10083_qt_synthetic.parquet` через `save_sample_dataset_en10083_qt()` (симметрично существующей `save_sample_dataset()`).

### 4.4 `train_model` — новый параметр steel_class

```python
def train_model(
    df_features: pd.DataFrame,
    target: str,
    feature_list: list[str],
    n_optuna_trials: int = 40,
    random_seed: int = 42,
    steel_class: str = "pipe_hsla",       # NEW
) -> TrainedModel:
    ...
    # meta.json gets "steel_class" field
    # Version string prefix: "hsla_..." for HSLA, "en10083qt_..." for Q&T
    version_prefix = {"pipe_hsla": "hsla", "en10083_qt": "en10083qt"}[steel_class]
    version = f"{version_prefix}_{target.replace('_mpa','').replace('_j_cm2','')}_xgb_{ts}"
    ...
```

`TrainedModel.steel_class: str` добавлен в dataclass. `load_model(version)` возвращает bundle с `meta["steel_class"]`; fallback `"pipe_hsla"` для старых моделей без поля.

### 4.5 `engine.py` — передача per-class context в Critic

В `Orchestrator._build_critic_context` для фазы `training`:

```python
if phase == "training":
    from app.backend.steel_classes import load_steel_class
    steel_class_id = result.output.get("steel_class", "pipe_hsla")
    try:
        profile = load_steel_class(steel_class_id)
        ctx["expected_top_features"] = profile.expected_top_features
        ctx["physical_bounds"] = profile.physical_bounds
        ctx["steel_class"] = steel_class_id
    except Exception:
        pass
    ctx.update({
        "split_strategy": result.output.get("split_strategy", "unknown"),
        "cv_strategy": result.output.get("cv_strategy", "unknown"),
        "prediction_has_ci": result.output.get("has_uncertainty", False),
        "ood_detector_configured": result.output.get("has_ood_detector", False),
    })
```

### 4.6 Pattern Library — per-class проверки

**`_check_m05_feature_importance_sanity`:**

```python
def _check_m05_feature_importance_sanity(ctx: dict) -> CheckResult:
    importance = ctx.get("feature_importance", {})
    expected = set(ctx.get("expected_top_features") or [])
    if not importance or not expected:
        return CheckResult(False)
    top_names = {f for f, _ in sorted(importance.items(), key=lambda x: -x[1])[:5]}
    overlap = top_names & expected
    if len(overlap) < 2:
        return CheckResult(
            True,
            message=(
                f"Top-5 feature importance не включает ожидаемых для класса "
                f"{ctx.get('steel_class','?')}. Top: {sorted(top_names)}. "
                f"Ожидалось минимум 2 из: {sorted(expected)}."
            ),
        )
    return CheckResult(False)
```

**`_check_d07_physical_bounds`:** fallback на текущие HSLA-константы, если `ctx["physical_bounds"]` отсутствует, иначе использует bounds из ctx.

### 4.7 UI — `app/frontend/app.py`

**Sidebar badge класса активной модели:**

```python
if selected_model:
    meta_path = PROJECT_ROOT / "models" / selected_model / "meta.json"
    try:
        _meta = json.loads(meta_path.read_text())
        _class_id = _meta.get("steel_class", "pipe_hsla")
        _class_label = {
            "pipe_hsla": "🔩 Pipe HSLA",
            "en10083_qt": "🔨 EN 10083 Q&T",
        }.get(_class_id, _class_id)
        st.sidebar.caption(f"Класс: **{_class_label}**")
    except Exception:
        pass
```

**Вкладка «🤖 Обучение модели» — новый блок сверху:**

```python
from app.backend.steel_classes import available_steel_classes
classes = available_steel_classes()
class_labels = {c.id: f"{c.name} ({c.standard})" for c in classes}
selected_class_id = st.selectbox(
    "Класс стали", options=[c.id for c in classes],
    format_func=lambda cid: class_labels[cid],
)
selected_profile = next(c for c in classes if c.id == selected_class_id)
target_col = st.selectbox(
    "Target property",
    options=[t.id for t in selected_profile.target_properties],
    format_func=lambda tid: next(
        t.label for t in selected_profile.target_properties if t.id == tid
    ),
)
```

При обучении:
- Получаем `df` через `get_synthetic_generator(profile.synthetic_generator_name)()`.
- Применяем `compute_features_for_class(df, selected_class_id)`.
- `train_model(df, target, feature_list=profile.feature_set, steel_class=selected_class_id)`.

**Вкладка «📊 Прогноз»** — поля ввода читаются из `profile.feature_set` активной модели + `profile.physical_bounds` для min/max. Если модель Q&T — показываем HRC, если HSLA — σт (читаем `target` из meta).

**Вкладка «🎯 Дизайн сплава»** — если активная модель `en10083_qt`:

```python
if active_class == "en10083_qt":
    st.info(
        "ℹ️ Inverse design пока работает только для **Pipe HSLA**. "
        "Для класса EN 10083-2 Q&T используйте вкладку «📊 Прогноз». "
        "Поддержка inverse design для Q&T запланирована на v2."
    )
else:
    # существующий дизайн-flow
    ...
```

---

## 5. Файлы, изменяемые в реализации

### Новые
- `app/backend/steel_classes.py`
- `data/steel_classes/pipe_hsla.yaml`
- `data/steel_classes/en10083_qt.yaml`
- `app/tests/test_steel_classes.py`
- `docs/superpowers/specs/2026-04-23-multi-class-steels-design.md`

### Изменяемые
- `app/backend/data_curator.py` — `generate_synthetic_en10083_qt_dataset` + `save_sample_dataset_en10083_qt`.
- `app/backend/model_trainer.py` — `train_model(steel_class="pipe_hsla")`, `TrainedModel.steel_class`, `meta["steel_class"]`, version prefix per class.
- `app/backend/engine.py` — `_build_critic_context` для training подставляет expected_top_features, physical_bounds, steel_class.
- `pattern_library/patterns.py` — `_check_m05_feature_importance_sanity`, `_check_d07_physical_bounds` per-class.
- `app/frontend/app.py` — dropdown в training, badge в sidebar, conditional predict/design UI.
- `CLAUDE.md` — про multi-class конвенцию.

---

## 6. Acceptance criteria

1. `pytest app/tests/test_steel_classes.py -v` — 7 тестов (load profiles, registry, synthetic generator sanity, pattern m05 per-class, trained_model.steel_class, meta persistence, feature_eng routing).
2. `pytest app/tests/ -q` — все предыдущие тесты проходят (без регрессий).
3. Обучение модели Q&T через UI: выбираем «EN 10083-2 Q&T», target «Твёрдость HRC», жмём «Обучить». Проходит за 1-3 мин, R² test > 0.7, в `models/en10083qt_hardness_hrc_*/meta.json` поле `steel_class: "en10083_qt"`.
4. Sidebar показывает «Класс: 🔨 EN 10083 Q&T» при выборе этой модели.
5. Prediction с Q&T: ввод химии 0.45% C + 600°C tempering → HRC в ожидаемом диапазоне 25-35.
6. Попытка открыть «🎯 Дизайн сплава» с активной моделью Q&T → показывается баннер, кнопка disabled.
7. Обучение HSLA-модели продолжает работать как раньше; cost-optimization + Pareto plot в «Дизайн» доступны.
8. Pattern M05 для Q&T: ожидаемые top-features — `c_pct, tempering_temp, austenitizing_temp, mn_pct`; если модель получила в top-5 иные — срабатывает warning.

---

## 7. Out of scope (для v5 и дальше)

- Inverse design + cost-optimization для Q&T.
- Третий класс (S235/S275/S355 по EN 10025-2) — прокат без термообработки.
- Инструментальные и нержавеющие стали (multi-model ensemble, feature sets совсем другие).
- Real data ingestion (замена синтетики на клиентские CSV).
- LLM-Critic per-class system prompts.
- Tenant-specific capability profiles (клиент-специфические процессные ограничения).

---

## 8. Риски и допущения

- **HRC-генератор — эмпирическая формула**, не истинная металлургия. Для MVP-демо достаточно: R² > 0.7 на synthetic data — реалистично-звучащий результат. Реальные данные дадут другое; это work item «real data ingestion» в v6.
- **`campaign_id` и `heat_date`** генерируются одинаково в обоих генераторах — нужны для time-based split + GroupKFold в `train_model`. Без них CV ломается.
- **`compute_features_for_class` returns df as-is для Q&T** — никаких derived features. Если потом понадобятся (например, `idealdia_mm` для hardenability) — добавим новую функцию и зарегистрируем в `feature_engineering` поле YAML.
- **Старые модели без `meta["steel_class"]`** — fallback на `"pipe_hsla"`. Ни одна существующая модель не сломается.
- **Inverse design + Q&T blocker**: UI баннер вежливый, но пользователь не сможет «запустить дизайн» на Q&T. Это ожидаемо — decision brainstormed в scope A.
- **Target dropdown фильтруется по классу**: если у пользователя сохранена HSLA-модель с target `kcv_neg60_j_cm2`, при смене класса на Q&T этот target пропадёт из UI — нужно обучить новую модель. Ожидаемо.
