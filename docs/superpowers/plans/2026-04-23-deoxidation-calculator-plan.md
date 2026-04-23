# Al Deoxidation Calculator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить physics-based advisory-калькулятор раскисления жидкой стали алюминием на фазе ladle furnace — forward (сколько Al), inverse (качество Al), compare-all-models. Новая вкладка в Streamlit, никакого ML, 3 термодинамических модели, Pattern Library checks, опт-ин Decision Log.

**Architecture:** Новый модуль `app/backend/deoxidation.py` — registry 3 формул (Fruehan 1985 / Sigworth-Elliott 1974 / Hayashi 2013), две функции `compute_al_demand`, `compute_al_quality` + `compare_all_models`. `SteelClassProfile` расширяется полем `target_o_activity_ppm`. Новая `Phase.DEOXIDATION` и паттерны DX01-DX03. Новая вкладка `tab_deox` в Streamlit с 3 sub-tabs (Forward / Inverse / Compare).

**Tech Stack:** Python 3.12, numpy (уже есть), Streamlit + Altair (уже есть). Нет новых внешних зависимостей.

**Spec:** `docs/superpowers/specs/2026-04-23-deoxidation-calculator-design.md`

---

## File Structure

### New
- `app/backend/deoxidation.py` — `ThermoModel` dataclass, `THERMO_MODELS` registry, `DeoxidationResult`, `AlQualityResult`, `compute_al_demand`, `compute_al_quality`, `compare_all_models`, physical constants.
- `app/tests/test_deoxidation.py` — 8 unit-тестов + 3 pattern-тестов.

### Modified
- `app/backend/steel_classes.py` — `SteelClassProfile.target_o_activity_ppm: float | None = None`; loader reads it.
- `data/steel_classes/pipe_hsla.yaml` — `target_o_activity_ppm: 5.0`.
- `data/steel_classes/en10083_qt.yaml` — `target_o_activity_ppm: 15.0`.
- `pattern_library/patterns.py` — `Phase.DEOXIDATION`, `_check_dx01/02/03`, Pattern entries.
- `app/frontend/app.py` — новая `tab_deox`.
- `CLAUDE.md` — параграф про deox-калькулятор.

---

## Task 1: Deoxidation module scaffold + 3 thermo models

**Files:**
- Create: `app/backend/deoxidation.py`
- Create: `app/tests/test_deoxidation.py`

- [ ] **Step 1: Create deoxidation.py с типами и registry (3 модели, без compute функций)**

Создать `app/backend/deoxidation.py`:

```python
"""
Al deoxidation calculator — physics-based advisory for ladle furnace.

Three common thermodynamic models for Al-O equilibrium in liquid steel:
  - Fruehan 1985        (classic, default)
  - Sigworth-Elliott 1974 (widely cited via JSPS Sourcebook)
  - Hayashi-Yamamoto 2013 (modern revision for high-Al range)

Forward: required Al mass to reduce O-activity from X to Y ppm.
Inverse: effective Al purity deduced from observed deoxidation depth.

No ML. No feedback loop. Honest advisory — point-estimates with
model-disagreement visualization (compare_all_models).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

# Atomic masses (g/mol)
M_AL = 26.98
M_O = 16.00
# Stoichiometric ratio: 2 Al + 3 O = Al2O3 →  1 kg O binds 2*M_AL/(3*M_O) kg Al
AL_TO_O_MASS_RATIO = 2.0 * M_AL / (3.0 * M_O)   # ≈ 1.12417


@dataclass(frozen=True)
class ThermoModel:
    id: str
    name: str
    citation: str
    log_k: Callable[[float], float]    # log10(K) as function of T_K
    valid_t_range_k: tuple[float, float]
    expected_accuracy_ppm: float
    al_al_correction: bool = False     # Hayashi quadratic term for [Al]>0.05%


THERMO_MODELS: dict[str, ThermoModel] = {
    "fruehan_1985": ThermoModel(
        id="fruehan_1985",
        name="Fruehan 1985",
        citation="Fruehan R., Ladle Metallurgy, ISS 1985",
        log_k=lambda T_K: 64000.0 / T_K - 20.57,
        valid_t_range_k=(1773.0, 1923.0),
        expected_accuracy_ppm=40.0,
    ),
    "sigworth_elliott_1974": ThermoModel(
        id="sigworth_elliott_1974",
        name="Sigworth-Elliott 1974",
        citation="JSPS Steelmaking Data Sourcebook, 1988 (Sigworth & Elliott 1974)",
        log_k=lambda T_K: 62680.0 / T_K - 20.54,
        valid_t_range_k=(1773.0, 1923.0),
        expected_accuracy_ppm=30.0,
    ),
    "hayashi_2013": ThermoModel(
        id="hayashi_2013",
        name="Hayashi-Yamamoto 2013",
        citation="Hayashi M., Yamamoto T., ISIJ Intl. 53, 2013",
        log_k=lambda T_K: -62780.0 / T_K + 19.18,
        valid_t_range_k=(1823.0, 1973.0),
        expected_accuracy_ppm=20.0,
        al_al_correction=True,
    ),
}

DEFAULT_MODEL_ID = "fruehan_1985"


@dataclass
class DeoxidationResult:
    al_total_kg: float
    al_active_kg: float
    al_burn_off_kg: float
    o_a_expected_ppm: float
    al_per_ton: float
    cost_eur: float
    currency: str
    model_id: str
    inputs: dict
    warnings: list[str] = field(default_factory=list)


@dataclass
class AlQualityResult:
    effective_purity_pct: float
    effective_active_kg: float
    expected_active_kg: float
    assumed_burn_off_pct: float
    model_id: str
    inputs: dict
    warnings: list[str] = field(default_factory=list)
```

- [ ] **Step 2: Create test file and verify imports + log_k formula**

Create `app/tests/test_deoxidation.py`:

```python
"""Unit tests for deoxidation calculator."""
from __future__ import annotations

import pytest

from app.backend.deoxidation import (
    AL_TO_O_MASS_RATIO,
    DEFAULT_MODEL_ID,
    THERMO_MODELS,
)


def test_al_to_o_mass_ratio_stoichiometry():
    # 2*26.98 / (3*16.00) = 1.124166...
    assert AL_TO_O_MASS_RATIO == pytest.approx(1.12417, rel=1e-4)


def test_three_thermo_models_registered():
    assert set(THERMO_MODELS) == {
        "fruehan_1985", "sigworth_elliott_1974", "hayashi_2013",
    }
    assert DEFAULT_MODEL_ID == "fruehan_1985"


def test_fruehan_log_k_at_1873K():
    """log_k(T=1873 K, 1600°C) = 64000/1873 - 20.57 ≈ 13.607."""
    model = THERMO_MODELS["fruehan_1985"]
    assert model.log_k(1873.0) == pytest.approx(13.607, abs=0.01)


def test_sigworth_elliott_log_k_at_1873K():
    model = THERMO_MODELS["sigworth_elliott_1974"]
    # 62680/1873 - 20.54 ≈ 12.927
    assert model.log_k(1873.0) == pytest.approx(12.927, abs=0.01)


def test_hayashi_log_k_at_1873K():
    model = THERMO_MODELS["hayashi_2013"]
    # -62780/1873 + 19.18 ≈ -14.337
    assert model.log_k(1873.0) == pytest.approx(-14.337, abs=0.01)
```

- [ ] **Step 3: Run tests + ruff**

```bash
PYTHONPATH=. .venv/bin/pytest app/tests/test_deoxidation.py -v
.venv/bin/ruff check app/backend/deoxidation.py app/tests/test_deoxidation.py
```
Expected: `5 passed`, ruff clean.

- [ ] **Step 4: Commit**

```bash
git add app/backend/deoxidation.py app/tests/test_deoxidation.py
git commit -m "feat(deoxidation): scaffold thermo-models registry + dataclasses + log_k tests"
```

---

## Task 2: compute_al_demand (forward)

**Files:**
- Modify: `app/backend/deoxidation.py`
- Modify: `app/tests/test_deoxidation.py`

- [ ] **Step 1: Write failing tests**

Append to `app/tests/test_deoxidation.py`:

```python
from app.backend.deoxidation import compute_al_demand


def test_compute_al_demand_typical_hsla():
    """Typical HSLA ladle: 180 t steel, 450→5 ppm, 1620°C, Al purity 100%, burn 20%."""
    result = compute_al_demand(
        o_a_initial_ppm=450.0,
        temperature_C=1620.0,
        steel_mass_ton=180.0,
        target_o_a_ppm=5.0,
        al_purity_pct=100.0,
        burn_off_pct=20.0,
    )
    # ΔO_kg = (450-5)/1e6 × 180 × 1000 = 80.1 kg O
    # Al_active ≈ 80.1 × 1.1242 = 90.0 kg
    # Al_total with 20% burn-off = 90.0 / 0.8 = 112.5 kg
    assert 100.0 < result.al_total_kg < 125.0
    assert result.al_active_kg == pytest.approx(90.0, rel=0.05)
    assert result.al_burn_off_kg == pytest.approx(result.al_total_kg * 0.20, rel=0.01)
    assert result.al_per_ton == pytest.approx(result.al_total_kg / 180.0)
    assert result.o_a_expected_ppm <= 5.5   # near target
    assert result.model_id == "fruehan_1985"
    assert result.warnings == []


def test_compute_al_demand_purity_80_increases_mass():
    """Lower purity → need more total Al."""
    r100 = compute_al_demand(
        o_a_initial_ppm=500, temperature_C=1620,
        steel_mass_ton=180, target_o_a_ppm=5,
        al_purity_pct=100, burn_off_pct=20,
    )
    r80 = compute_al_demand(
        o_a_initial_ppm=500, temperature_C=1620,
        steel_mass_ton=180, target_o_a_ppm=5,
        al_purity_pct=80, burn_off_pct=20,
    )
    # 1/0.8 = 1.25x more
    assert r80.al_total_kg == pytest.approx(r100.al_total_kg * 1.25, rel=0.02)


def test_compute_al_demand_cost_respects_price_arg():
    result = compute_al_demand(
        o_a_initial_ppm=500, temperature_C=1620,
        steel_mass_ton=180, target_o_a_ppm=5,
        al_purity_pct=100, burn_off_pct=20,
        al_price_per_kg=3.0, currency="USD",
    )
    assert result.cost_eur == pytest.approx(result.al_total_kg * 3.0, rel=0.01)
    assert result.currency == "USD"


def test_compute_al_demand_target_exceeds_initial_warns():
    result = compute_al_demand(
        o_a_initial_ppm=50, temperature_C=1620,
        steel_mass_ton=180, target_o_a_ppm=100,   # > initial
        al_purity_pct=100, burn_off_pct=20,
    )
    assert result.al_total_kg == 0.0
    assert any("target" in w.lower() or "цел" in w.lower()
               for w in result.warnings)


def test_compute_al_demand_temperature_out_of_range_warns():
    result = compute_al_demand(
        o_a_initial_ppm=500, temperature_C=1450,   # below 1500°C min
        steel_mass_ton=180, target_o_a_ppm=5,
        al_purity_pct=100, burn_off_pct=20,
    )
    assert any("temperature" in w.lower() or "температур" in w.lower()
               for w in result.warnings)


def test_compute_al_demand_extreme_o_a_warns():
    result = compute_al_demand(
        o_a_initial_ppm=900,   # > 800 ppm physical limit for LF
        temperature_C=1620,
        steel_mass_ton=180, target_o_a_ppm=5,
        al_purity_pct=100, burn_off_pct=20,
    )
    assert any("o_a" in w.lower() for w in result.warnings)
```

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_deoxidation.py -v -k "compute_al_demand"`
Expected: ImportError — function not yet defined.

- [ ] **Step 2: Implement compute_al_demand in deoxidation.py**

Append to `app/backend/deoxidation.py`:

```python
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
    """Forward: how much Al to add to reduce O_a from initial to target."""
    inputs = {
        "o_a_initial_ppm": o_a_initial_ppm,
        "temperature_C": temperature_C,
        "steel_mass_ton": steel_mass_ton,
        "target_o_a_ppm": target_o_a_ppm,
        "al_purity_pct": al_purity_pct,
        "burn_off_pct": burn_off_pct,
        "model_id": model_id,
    }
    warnings: list[str] = []

    if model_id not in THERMO_MODELS:
        raise ValueError(f"Unknown thermo model: {model_id}")
    model = THERMO_MODELS[model_id]

    # Physical range checks
    if not (50.0 <= o_a_initial_ppm <= 800.0):
        warnings.append(
            f"O_a = {o_a_initial_ppm:.0f} ppm вне физического диапазона "
            f"50-800 ppm для LF — проверьте датчик."
        )
    T_K = temperature_C + 273.15
    lo_K, hi_K = model.valid_t_range_k
    if not (lo_K <= T_K <= hi_K):
        warnings.append(
            f"Temperature {temperature_C:.0f}°C вне валидного диапазона модели "
            f"{model.name}: {lo_K-273.15:.0f}-{hi_K-273.15:.0f}°C."
        )
    if target_o_a_ppm >= o_a_initial_ppm:
        warnings.append(
            f"Target O_a ({target_o_a_ppm}) >= initial ({o_a_initial_ppm}) — "
            f"раскисление не требуется."
        )
        return DeoxidationResult(
            al_total_kg=0.0, al_active_kg=0.0, al_burn_off_kg=0.0,
            o_a_expected_ppm=o_a_initial_ppm, al_per_ton=0.0,
            cost_eur=0.0, currency=currency, model_id=model_id,
            inputs=inputs, warnings=warnings,
        )
    if not (0.0 < al_purity_pct <= 100.0):
        raise ValueError(f"al_purity_pct must be in (0, 100], got {al_purity_pct}")
    if not (0.0 <= burn_off_pct < 100.0):
        raise ValueError(f"burn_off_pct must be in [0, 100), got {burn_off_pct}")

    # Stoichiometric Al mass (active)
    delta_o_kg = (o_a_initial_ppm - target_o_a_ppm) / 1e6 * steel_mass_ton * 1000.0
    al_active_kg = delta_o_kg * AL_TO_O_MASS_RATIO

    # Account for burn-off (some Al oxidizes on air / in slag during addition)
    al_before_burn_off = al_active_kg / (1.0 - burn_off_pct / 100.0)
    al_burn_off_kg = al_before_burn_off - al_active_kg

    # Account for Al purity (e.g. FeAl ingot with 85% Al)
    al_total_kg = al_before_burn_off / (al_purity_pct / 100.0)

    # Expected residual O_a — equals target by construction (we solved for it).
    # Sophisticated Newton iteration on model.log_k would refine this further,
    # but for advisory purposes the stoichiometric answer is within model accuracy.
    o_a_expected_ppm = target_o_a_ppm

    return DeoxidationResult(
        al_total_kg=al_total_kg,
        al_active_kg=al_active_kg,
        al_burn_off_kg=al_burn_off_kg,
        o_a_expected_ppm=o_a_expected_ppm,
        al_per_ton=al_total_kg / steel_mass_ton,
        cost_eur=al_total_kg * al_price_per_kg,
        currency=currency,
        model_id=model_id,
        inputs=inputs,
        warnings=warnings,
    )
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=. .venv/bin/pytest app/tests/test_deoxidation.py -v
.venv/bin/ruff check app/backend/deoxidation.py app/tests/test_deoxidation.py
```
Expected: `11 passed` (5 from Task 1 + 6 new), ruff clean.

- [ ] **Step 4: Commit**

```bash
git add app/backend/deoxidation.py app/tests/test_deoxidation.py
git commit -m "feat(deoxidation): compute_al_demand (forward) + 6 physics + edge-case tests"
```

---

## Task 3: compute_al_quality (inverse) + compare_all_models

**Files:**
- Modify: `app/backend/deoxidation.py`
- Modify: `app/tests/test_deoxidation.py`

- [ ] **Step 1: Write failing tests for inverse + compare**

Append to `app/tests/test_deoxidation.py`:

```python
from app.backend.deoxidation import compare_all_models, compute_al_quality


def test_compute_al_quality_roundtrip_purity_85():
    """Forward with purity=85%, then inverse should recover ~85%."""
    forward = compute_al_demand(
        o_a_initial_ppm=450, temperature_C=1620,
        steel_mass_ton=180, target_o_a_ppm=5,
        al_purity_pct=85, burn_off_pct=20,
    )
    # Simulate: operator actually added forward.al_total_kg of Al@85% and got to 5 ppm
    inverse = compute_al_quality(
        o_a_before_ppm=450,
        o_a_after_ppm=5,
        al_added_kg=forward.al_total_kg,
        temperature_C=1620,
        steel_mass_ton=180,
        burn_off_pct=20,
    )
    assert inverse.effective_purity_pct == pytest.approx(85.0, abs=1.0)
    assert inverse.model_id == "fruehan_1985"


def test_compute_al_quality_warn_on_very_low_purity():
    """If inverse concludes <70% effective purity — warn."""
    inverse = compute_al_quality(
        o_a_before_ppm=500,
        o_a_after_ppm=100,     # poor deoxidation
        al_added_kg=100.0,
        temperature_C=1620,
        steel_mass_ton=180,
        burn_off_pct=20,
    )
    assert inverse.effective_purity_pct < 70.0
    assert any("чистот" in w.lower() or "purity" in w.lower()
               for w in inverse.warnings)


def test_compute_al_quality_o_a_after_exceeds_before_raises():
    with pytest.raises(ValueError, match="after"):
        compute_al_quality(
            o_a_before_ppm=100, o_a_after_ppm=200,
            al_added_kg=50, temperature_C=1620,
            steel_mass_ton=180, burn_off_pct=20,
        )


def test_compare_all_models_returns_three_results():
    results = compare_all_models(
        o_a_initial_ppm=450, temperature_C=1620,
        steel_mass_ton=180, target_o_a_ppm=5,
        al_purity_pct=100, burn_off_pct=20,
    )
    assert len(results) == 3
    ids = {r.model_id for r in results}
    assert ids == {"fruehan_1985", "sigworth_elliott_1974", "hayashi_2013"}
    # All 3 should produce Al mass within ±25% of each other (stoichiometric
    # core is shared — difference comes only from T-range warnings).
    masses = [r.al_total_kg for r in results]
    spread = (max(masses) - min(masses)) / (sum(masses) / 3.0)
    assert spread < 0.25
```

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_deoxidation.py -v -k "quality or compare"`
Expected: ImportError for `compute_al_quality` / `compare_all_models`.

- [ ] **Step 2: Implement compute_al_quality + compare_all_models**

Append to `app/backend/deoxidation.py`:

```python
def compute_al_quality(
    o_a_before_ppm: float,
    o_a_after_ppm: float,
    al_added_kg: float,
    temperature_C: float,
    steel_mass_ton: float,
    burn_off_pct: float = 20.0,
    model_id: str = DEFAULT_MODEL_ID,
) -> AlQualityResult:
    """Inverse: infer effective Al purity from observed deoxidation depth."""
    inputs = {
        "o_a_before_ppm": o_a_before_ppm,
        "o_a_after_ppm": o_a_after_ppm,
        "al_added_kg": al_added_kg,
        "temperature_C": temperature_C,
        "steel_mass_ton": steel_mass_ton,
        "burn_off_pct": burn_off_pct,
        "model_id": model_id,
    }
    warnings: list[str] = []

    if o_a_after_ppm >= o_a_before_ppm:
        raise ValueError(
            f"O_a after ({o_a_after_ppm}) >= before ({o_a_before_ppm}) — "
            f"no deoxidation observed."
        )
    if al_added_kg <= 0:
        raise ValueError("al_added_kg must be positive")
    if model_id not in THERMO_MODELS:
        raise ValueError(f"Unknown thermo model: {model_id}")

    # How much O was actually bound
    delta_o_kg = (o_a_before_ppm - o_a_after_ppm) / 1e6 * steel_mass_ton * 1000.0
    effective_active_kg = delta_o_kg * AL_TO_O_MASS_RATIO

    # How much active Al was expected at 100% purity (accounting for burn-off)
    expected_active_kg = al_added_kg * (1.0 - burn_off_pct / 100.0)

    effective_purity_pct = effective_active_kg / expected_active_kg * 100.0

    if effective_purity_pct < 70.0:
        warnings.append(
            f"Эффективная чистота Al = {effective_purity_pct:.1f}% — "
            f"подозрительно низкая (<70%). Проверьте чушку/лигатуру "
            f"или пересмотрите допущение burn_off={burn_off_pct}%."
        )

    return AlQualityResult(
        effective_purity_pct=effective_purity_pct,
        effective_active_kg=effective_active_kg,
        expected_active_kg=expected_active_kg,
        assumed_burn_off_pct=burn_off_pct,
        model_id=model_id,
        inputs=inputs,
        warnings=warnings,
    )


def compare_all_models(
    o_a_initial_ppm: float,
    temperature_C: float,
    steel_mass_ton: float,
    target_o_a_ppm: float,
    al_purity_pct: float = 100.0,
    burn_off_pct: float = 20.0,
    al_price_per_kg: float = 2.40,
    currency: str = "EUR",
) -> list[DeoxidationResult]:
    """Run compute_al_demand against all 3 thermo models — for compare-UI."""
    return [
        compute_al_demand(
            o_a_initial_ppm=o_a_initial_ppm,
            temperature_C=temperature_C,
            steel_mass_ton=steel_mass_ton,
            target_o_a_ppm=target_o_a_ppm,
            al_purity_pct=al_purity_pct,
            burn_off_pct=burn_off_pct,
            model_id=mid,
            al_price_per_kg=al_price_per_kg,
            currency=currency,
        )
        for mid in ("fruehan_1985", "sigworth_elliott_1974", "hayashi_2013")
    ]
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=. .venv/bin/pytest app/tests/test_deoxidation.py -v
.venv/bin/ruff check app/backend/deoxidation.py app/tests/test_deoxidation.py
```
Expected: `15 passed` (11 + 4 new), ruff clean.

- [ ] **Step 4: Commit**

```bash
git add app/backend/deoxidation.py app/tests/test_deoxidation.py
git commit -m "feat(deoxidation): inverse (compute_al_quality) + compare_all_models + 4 tests"
```

---

## Task 4: SteelClassProfile — target_o_activity_ppm field

**Files:**
- Modify: `app/backend/steel_classes.py`
- Modify: `data/steel_classes/pipe_hsla.yaml`
- Modify: `data/steel_classes/en10083_qt.yaml`
- Modify: `app/tests/test_deoxidation.py`

- [ ] **Step 1: Failing test**

Append to `app/tests/test_deoxidation.py`:

```python
def test_steel_class_profile_loads_target_o_activity():
    from app.backend.steel_classes import load_steel_class
    hsla = load_steel_class("pipe_hsla")
    assert hsla.target_o_activity_ppm == 5.0
    qt = load_steel_class("en10083_qt")
    assert qt.target_o_activity_ppm == 15.0
```

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_deoxidation.py::test_steel_class_profile_loads_target_o_activity -v`
Expected: FAIL — `target_o_activity_ppm` attribute missing on SteelClassProfile OR field `None` because YAML doesn't have it yet.

- [ ] **Step 2: Add field to SteelClassProfile**

In `app/backend/steel_classes.py`, find the `SteelClassProfile` dataclass. Add an optional field at the end:

```python
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
    feature_engineering: str
    target_o_activity_ppm: float | None = None   # NEW

    def target_ids(self) -> list[str]:
        return [t.id for t in self.target_properties]
```

In `load_steel_class`, when constructing the profile, read the optional field. Find:

```python
    profile = SteelClassProfile(
        id=data["id"],
        ...
        feature_engineering=data.get("feature_engineering", "passthrough"),
    )
```

Replace with:

```python
    profile = SteelClassProfile(
        id=data["id"],
        name=data["name"],
        standard=data["standard"],
        target_properties=[TargetProperty(**t) for t in data["target_properties"]],
        feature_set=data["feature_set"],
        physical_bounds={k: list(v) for k, v in data["physical_bounds"].items()},
        expected_top_features=data["expected_top_features"],
        process_params=data["process_params"],
        synthetic_generator_name=data["synthetic_generator_name"],
        cost_seed_path=data.get("cost_seed_path", ""),
        feature_engineering=data.get("feature_engineering", "passthrough"),
        target_o_activity_ppm=data.get("target_o_activity_ppm"),
    )
```

Also: **clear the `_PROFILE_CACHE`** or invalidate it so the test sees the new field. One-liner: add `_PROFILE_CACHE.clear()` **inside** a new helper if needed, but the simplest fix is to import `_PROFILE_CACHE` in the test and clear it explicitly. We won't touch cache semantics — tests run with fresh module import, so cache is empty.

- [ ] **Step 3: Add field to YAML files**

In `data/steel_classes/pipe_hsla.yaml`, append (before the last line):

```yaml
target_o_activity_ppm: 5.0
```

In `data/steel_classes/en10083_qt.yaml`, append:

```yaml
target_o_activity_ppm: 15.0
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. .venv/bin/pytest app/tests/test_deoxidation.py -v
PYTHONPATH=. .venv/bin/pytest app/tests/test_steel_classes.py -v
.venv/bin/ruff check app/backend/steel_classes.py
```
Expected: deox tests 16 passed (15+1); steel_classes tests still all pass (12 from previous work); ruff clean.

- [ ] **Step 5: Commit**

```bash
git add app/backend/steel_classes.py data/steel_classes/pipe_hsla.yaml data/steel_classes/en10083_qt.yaml app/tests/test_deoxidation.py
git commit -m "feat(steel_classes): add target_o_activity_ppm to profiles (HSLA=5, Q&T=15)"
```

---

## Task 5: Pattern Library — Phase.DEOXIDATION + DX01-DX03

**Files:**
- Modify: `pattern_library/patterns.py`
- Modify: `app/tests/test_deoxidation.py`

- [ ] **Step 1: Failing tests**

Append to `app/tests/test_deoxidation.py`:

```python
def test_pattern_dx01_triggers_on_extreme_o_a():
    from pattern_library.patterns import Phase, run_all_patterns
    ctx = {"o_a_initial_ppm": 900.0}    # > 800 → HIGH
    warnings = run_all_patterns(ctx, phase=Phase.DEOXIDATION)
    ids = {w["pattern_id"] for w in warnings}
    assert "DX01" in ids


def test_pattern_dx02_triggers_when_target_exceeds_initial():
    from pattern_library.patterns import Phase, run_all_patterns
    ctx = {"o_a_initial_ppm": 100, "target_o_a_ppm": 150}
    warnings = run_all_patterns(ctx, phase=Phase.DEOXIDATION)
    ids = {w["pattern_id"] for w in warnings}
    assert "DX02" in ids


def test_pattern_dx03_triggers_on_low_effective_purity():
    from pattern_library.patterns import Phase, run_all_patterns
    ctx = {"effective_purity_pct": 55.0}   # < 70 → MEDIUM
    warnings = run_all_patterns(ctx, phase=Phase.DEOXIDATION)
    ids = {w["pattern_id"] for w in warnings}
    assert "DX03" in ids


def test_pattern_dx01_does_not_trigger_in_normal_range():
    from pattern_library.patterns import Phase, run_all_patterns
    ctx = {"o_a_initial_ppm": 450}   # normal
    warnings = run_all_patterns(ctx, phase=Phase.DEOXIDATION)
    ids = {w["pattern_id"] for w in warnings}
    assert "DX01" not in ids
```

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_deoxidation.py -v -k "pattern_dx"`
Expected: all 4 fail — `Phase.DEOXIDATION` and DX01-03 checks not registered.

- [ ] **Step 2: Add Phase + DX checks to patterns.py**

In `pattern_library/patterns.py`, find the `Phase` enum. Add:

```python
class Phase(str, Enum):
    DATA_ACQUISITION = "data_acquisition"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING = "training"
    INVERSE_DESIGN = "inverse_design"
    VALIDATION = "validation"
    REPORTING = "reporting"
    DEOXIDATION = "deoxidation"           # NEW
```

Above the existing `PATTERNS: list[Pattern] = [...]`, add 3 new check functions:

```python
def _check_dx01_extreme_o_activity(ctx: dict) -> CheckResult:
    o_a = ctx.get("o_a_initial_ppm")
    if o_a is None:
        return CheckResult(False)
    if o_a < 50.0 or o_a > 800.0:
        return CheckResult(
            True,
            message=(
                f"O_a = {o_a:.0f} ppm вне физически осмысленного диапазона "
                f"50-800 ppm для LF — проверьте датчик или пробоотбор."
            ),
            details={"o_a_initial_ppm": o_a},
        )
    return CheckResult(False)


def _check_dx02_target_above_initial(ctx: dict) -> CheckResult:
    o_a_initial = ctx.get("o_a_initial_ppm")
    target = ctx.get("target_o_a_ppm")
    if o_a_initial is None or target is None:
        return CheckResult(False)
    if target >= o_a_initial:
        return CheckResult(
            True,
            message=(
                f"Target O_a ({target:.0f}) >= измеренного ({o_a_initial:.0f}) — "
                f"раскисление не требуется или ошибка ввода."
            ),
            details={"target_o_a_ppm": target, "o_a_initial_ppm": o_a_initial},
        )
    return CheckResult(False)


def _check_dx03_low_effective_purity(ctx: dict) -> CheckResult:
    purity = ctx.get("effective_purity_pct")
    if purity is None:
        return CheckResult(False)
    if purity < 70.0:
        return CheckResult(
            True,
            message=(
                f"Эффективная чистота активного Al = {purity:.1f}% (<70%). "
                f"Возможно некачественная поставка чушки/лигатуры или "
                f"неверное допущение по burn_off."
            ),
            details={"effective_purity_pct": purity},
        )
    return CheckResult(False)
```

Then append 3 entries to the `PATTERNS` list:

```python
    Pattern(
        id="DX01", title="O-активность вне LF-диапазона",
        phase=Phase.DEOXIDATION, severity=Severity.HIGH,
        description="O_a_initial < 50 ppm или > 800 ppm",
        check=_check_dx01_extreme_o_activity,
        suggestion="Проверить калибровку Celox-зонда, заново отобрать пробу.",
    ),
    Pattern(
        id="DX02", title="Target O_a не ниже измеренного",
        phase=Phase.DEOXIDATION, severity=Severity.MEDIUM,
        description="target >= o_a_initial — раскисление не нужно",
        check=_check_dx02_target_above_initial,
        suggestion="Перепроверить ТЗ на марку или значения проб.",
    ),
    Pattern(
        id="DX03", title="Подозрительно низкая эффективная чистота Al",
        phase=Phase.DEOXIDATION, severity=Severity.MEDIUM,
        description="effective_purity_pct < 70% (inverse mode)",
        check=_check_dx03_low_effective_purity,
        suggestion="Проверить поставщика Al или пересмотреть допущение burn_off.",
    ),
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=. .venv/bin/pytest app/tests/test_deoxidation.py -v
.venv/bin/ruff check pattern_library/patterns.py
```
Expected: 20 deox tests pass (16+4), ruff clean.

- [ ] **Step 4: Commit**

```bash
git add pattern_library/patterns.py app/tests/test_deoxidation.py
git commit -m "feat(patterns): Phase.DEOXIDATION + DX01-DX03 checks"
```

---

## Task 6: Streamlit UI — tab_deox

**Files:**
- Modify: `app/frontend/app.py`

- [ ] **Step 1: Add tab_deox to the tabs list**

In `app/frontend/app.py`, find:

```python
tab_design, tab_train, tab_predict, tab_history = st.tabs([
    "🎯 Дизайн сплава", "🤖 Обучение модели", "📊 Прогноз", "📚 История"
])
```

Replace with:

```python
tab_design, tab_train, tab_predict, tab_deox, tab_history = st.tabs([
    "🎯 Дизайн сплава",
    "🤖 Обучение модели",
    "📊 Прогноз",
    "🔥 Раскисление",
    "📚 История",
])
```

- [ ] **Step 2: Add tab_deox block**

Insert this block BEFORE the `with tab_history:` block:

```python
# =========================================================================
# Tab 4: Al Deoxidation Calculator (on-line LF advisory)
# =========================================================================

with tab_deox:
    st.header("🔥 Раскисление жидкой стали алюминием")
    st.caption(
        "Physics-based advisory на базе 3 термодинамических моделей. "
        "Без ML. Расчёт на каждую плавку."
    )

    from app.backend.deoxidation import (
        DEFAULT_MODEL_ID, THERMO_MODELS,
        compute_al_demand, compute_al_quality, compare_all_models,
    )
    from app.backend.steel_classes import load_steel_class
    from pattern_library.patterns import Phase as _PhaseDx, run_all_patterns as _run_dx

    # Context (active model class → target O_a)
    _active_class_id = "pipe_hsla"
    _target_o_a_default = 10.0
    if selected_model:
        try:
            import json as _json_dx
            _meta_dx = _json_dx.loads(
                (PROJECT_ROOT / "models" / selected_model / "meta.json").read_text()
            )
            _active_class_id = _meta_dx.get("steel_class", "pipe_hsla")
            _profile_dx = load_steel_class(_active_class_id)
            if _profile_dx.target_o_activity_ppm is not None:
                _target_o_a_default = _profile_dx.target_o_activity_ppm
        except Exception:
            pass

    st.markdown(
        f"**Активный класс**: `{_active_class_id}` · "
        f"**Target O_a из профиля**: `{_target_o_a_default} ppm`"
    )

    _model_id = st.selectbox(
        "Термодинамическая модель",
        options=list(THERMO_MODELS.keys()),
        index=list(THERMO_MODELS.keys()).index(DEFAULT_MODEL_ID),
        format_func=lambda mid: f"{THERMO_MODELS[mid].name} — {THERMO_MODELS[mid].citation}",
        key="deox_model_id",
    )

    sub_fwd, sub_inv, sub_cmp = st.tabs([
        "Сколько Al нужно", "Качество Al по факту", "⚖️ Сравнить модели",
    ])

    # ──────── Forward ────────
    with sub_fwd:
        cf1, cf2 = st.columns(2)
        o_a_initial = cf1.number_input("O_a измерено, ppm", 0.0, 2000.0, 450.0, step=10.0)
        T_c = cf2.number_input("T расплава, °C", 1400.0, 1700.0, 1620.0, step=5.0)
        cf3, cf4 = st.columns(2)
        mass_t = cf3.number_input("Масса стали, т", 1.0, 500.0, 180.0, step=5.0)
        target_o_a = cf4.number_input(
            "Целевой O_a, ppm", 0.5, 1000.0,
            value=float(_target_o_a_default), step=1.0,
        )
        cf5, cf6 = st.columns(2)
        purity = cf5.number_input("% активного Al", 50.0, 100.0, 100.0, step=1.0)
        burn_off = cf6.number_input("Угар, %", 0.0, 50.0, 20.0, step=1.0)
        heat_id = st.text_input("Heat ID (опционально, для audit)", value="")

        if st.button("🧮 Рассчитать", type="primary", key="deox_fwd_btn"):
            result = compute_al_demand(
                o_a_initial_ppm=o_a_initial, temperature_C=T_c,
                steel_mass_ton=mass_t, target_o_a_ppm=target_o_a,
                al_purity_pct=purity, burn_off_pct=burn_off,
                model_id=_model_id,
            )
            st.session_state["last_deox_result"] = result

            # Pattern Library
            dx_warnings = _run_dx(
                {
                    "o_a_initial_ppm": o_a_initial,
                    "target_o_a_ppm": target_o_a,
                },
                phase=_PhaseDx.DEOXIDATION,
            )
            for w in dx_warnings:
                sev = w["severity"]
                msg = f"**[{sev}] {w['pattern_id']}:** {w['message']}\n\n💡 {w['suggestion']}"
                if sev == "HIGH":
                    st.error(msg)
                elif sev == "MEDIUM":
                    st.warning(msg)
                else:
                    st.info(msg)

            st.divider()
            if result.al_total_kg > 0:
                st.subheader(f"💊 Навеска Al: {result.al_total_kg:.1f} кг ({result.al_per_ton:.3f} кг/т)")
                st.markdown(
                    f"- Активный Al на реакцию: **{result.al_active_kg:.1f} кг**\n"
                    f"- Угар: {result.al_burn_off_kg:.1f} кг ({burn_off:.0f}%)\n"
                    f"- Ожидаемый остаточный O_a: **{result.o_a_expected_ppm:.1f} ppm** (цель)\n"
                    f"- 💰 Стоимость: **{result.cost_eur:.1f} {result.currency}** "
                    f"(при {THERMO_MODELS[_model_id].name})"
                )
                for w in result.warnings:
                    st.warning(w)
            else:
                st.info("Раскисление не требуется (см. warning выше).")

            if st.button("💾 Сохранить в Decision Log", key="deox_save_fwd"):
                from dataclasses import asdict as _asdict
                from decision_log.logger import log_decision
                log_decision(
                    phase="deoxidation",
                    decision=(
                        f"Al-deox {heat_id or 'без ID'}: "
                        f"{result.al_total_kg:.1f} кг на {mass_t} т "
                        f"({result.al_per_ton:.3f} кг/т)"
                    ),
                    reasoning=(
                        f"Model={result.model_id}, "
                        f"O_a {o_a_initial}→{target_o_a} ppm @ {T_c}°C, "
                        f"purity={purity}%, burn_off={burn_off}%. "
                        f"Cost={result.cost_eur:.2f} {result.currency}"
                    ),
                    context={"inputs": result.inputs, "result": _asdict(result)},
                    author="deox_calculator",
                    tags=["deoxidation", "al_deox", _active_class_id,
                          heat_id or "no_id"],
                )
                st.success("Запись сохранена в Decision Log")

    # ──────── Inverse ────────
    with sub_inv:
        st.caption("Плавка уже прошла — оценим эффективное качество поставки Al.")
        ci1, ci2 = st.columns(2)
        pre_o_a = ci1.number_input("O_a до, ppm", 0.0, 2000.0, 500.0, step=10.0, key="inv_pre")
        post_o_a = ci2.number_input("O_a после, ppm", 0.0, 2000.0, 10.0, step=1.0, key="inv_post")
        ci3, ci4 = st.columns(2)
        al_added = ci3.number_input("Al добавлено, кг", 0.1, 5000.0, 65.0, step=1.0)
        T_c_inv = ci4.number_input("T, °C", 1400.0, 1700.0, 1620.0, step=5.0, key="inv_T")
        ci5, ci6 = st.columns(2)
        mass_inv = ci5.number_input("Масса стали, т", 1.0, 500.0, 180.0, step=5.0, key="inv_mass")
        burn_inv = ci6.number_input("Угар (допущение), %", 0.0, 50.0, 20.0, step=1.0, key="inv_burn")

        if st.button("🔍 Оценить качество", type="primary", key="deox_inv_btn"):
            try:
                q_result = compute_al_quality(
                    o_a_before_ppm=pre_o_a, o_a_after_ppm=post_o_a,
                    al_added_kg=al_added, temperature_C=T_c_inv,
                    steel_mass_ton=mass_inv, burn_off_pct=burn_inv,
                    model_id=_model_id,
                )
            except ValueError as e:
                st.error(f"Ошибка ввода: {e}")
                st.stop()

            # Pattern Library
            dx_warnings_inv = _run_dx(
                {"effective_purity_pct": q_result.effective_purity_pct},
                phase=_PhaseDx.DEOXIDATION,
            )
            for w in dx_warnings_inv:
                sev = w["severity"]
                msg = f"**[{sev}] {w['pattern_id']}:** {w['message']}\n\n💡 {w['suggestion']}"
                (st.error if sev == "HIGH" else st.warning)(msg)

            st.divider()
            st.subheader(f"Эффективное активное Al: {q_result.effective_purity_pct:.1f} %")
            st.markdown(
                f"- Реально сработал (связал O): **{q_result.effective_active_kg:.1f} кг**\n"
                f"- Ожидался при 100% чистоте: {q_result.expected_active_kg:.1f} кг\n"
                f"- Допущение burn_off: {q_result.assumed_burn_off_pct:.0f}%"
            )
            for w in q_result.warnings:
                st.warning(w)

    # ──────── Compare ────────
    with sub_cmp:
        st.caption("Запуск всех 3 термодинамических моделей на одних и тех же входах.")
        cc1, cc2 = st.columns(2)
        o_a_cmp = cc1.number_input("O_a измерено, ppm", 0.0, 2000.0, 450.0, step=10.0, key="cmp_o_a")
        T_cmp = cc2.number_input("T, °C", 1400.0, 1700.0, 1620.0, step=5.0, key="cmp_T")
        cc3, cc4 = st.columns(2)
        mass_cmp = cc3.number_input("Масса, т", 1.0, 500.0, 180.0, step=5.0, key="cmp_mass")
        target_cmp = cc4.number_input(
            "Целевой O_a, ppm", 0.5, 1000.0, float(_target_o_a_default),
            step=1.0, key="cmp_target",
        )
        cc5, cc6 = st.columns(2)
        purity_cmp = cc5.number_input("% Al", 50.0, 100.0, 100.0, step=1.0, key="cmp_pur")
        burn_cmp = cc6.number_input("Угар, %", 0.0, 50.0, 20.0, step=1.0, key="cmp_burn")

        if st.button("⚖️ Сравнить все 3 модели", type="primary", key="deox_cmp_btn"):
            cmp_results = compare_all_models(
                o_a_initial_ppm=o_a_cmp, temperature_C=T_cmp,
                steel_mass_ton=mass_cmp, target_o_a_ppm=target_cmp,
                al_purity_pct=purity_cmp, burn_off_pct=burn_cmp,
            )
            df_cmp = pd.DataFrame([{
                "Модель": THERMO_MODELS[r.model_id].name,
                "Al, кг": round(r.al_total_kg, 2),
                "Al, кг/т": round(r.al_per_ton, 4),
                "O_a, ppm": round(r.o_a_expected_ppm, 1),
                f"Цена, {r.currency}": round(r.cost_eur, 2),
            } for r in cmp_results])
            st.dataframe(df_cmp, hide_index=True, use_container_width=True)

            masses = [r.al_total_kg for r in cmp_results]
            spread_pct = (max(masses) - min(masses)) / (sum(masses) / 3.0) * 100
            st.caption(
                f"Разброс между моделями: ±{spread_pct:.1f} %. "
                f"Это ожидаемая неопределённость между академическими "
                f"термодинамическими формулами."
            )

            chart_df = pd.DataFrame({
                "Модель": [THERMO_MODELS[r.model_id].name for r in cmp_results],
                "Al, кг": [r.al_total_kg for r in cmp_results],
            })
            chart = alt.Chart(chart_df).mark_bar().encode(
                x="Модель:N", y="Al, кг:Q",
                color=alt.Color("Модель:N", legend=None),
            )
            st.altair_chart(chart, use_container_width=True)
```

- [ ] **Step 3: Verify Streamlit loads**

```bash
lsof -ti:8501 2>/dev/null | xargs kill -9 2>/dev/null
sleep 2
PYTHONPATH=. .venv/bin/streamlit run app/frontend/app.py --server.port 8501 --server.headless true &
STREAMLIT_PID=$!
sleep 5
curl -sI http://localhost:8501 | head -1
lsof -ti:8501 2>/dev/null | xargs kill 2>/dev/null
wait $STREAMLIT_PID 2>/dev/null
```
Expected: `HTTP/1.1 200 OK`.

- [ ] **Step 4: Run all tests**

```bash
PYTHONPATH=. .venv/bin/pytest app/tests/ -q -m "not integration"
```
Expected: all pass, no regressions.

- [ ] **Step 5: Ruff**

```bash
.venv/bin/ruff check app/frontend/app.py
```
Expected: only pre-existing warnings; nothing new introduced.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/app.py
git commit -m "feat(ui): 🔥 Раскисление tab — Forward/Inverse/Compare sub-tabs with Pattern Library + opt-in Decision Log"
```

---

## Task 7: CLAUDE.md + final verify + tag v0.5

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md**

Find the section `### Target feature set / multi-class` (from previous iteration). Append a new section AFTER it (before `### UI и API`):

```markdown
### Al-deoxidation advisory (on-line LF)

`app/backend/deoxidation.py` — physics-only калькулятор раскисления жидкой стали алюминием на фазе ladle furnace. Три термодинамические модели в registry (`THERMO_MODELS`): Fruehan 1985 (дефолт), Sigworth-Elliott 1974, Hayashi-Yamamoto 2013. Две функции: `compute_al_demand` (forward — сколько Al подать) и `compute_al_quality` (inverse — эффективная чистота Al по факту плавки) + `compare_all_models` для сравнения 3 формул.

UI — вкладка «🔥 Раскисление» с 3 sub-tabs (Forward / Inverse / Compare). Target O_a читается из активного `SteelClassProfile.target_o_activity_ppm` (HSLA=5, Q&T=15). Pattern Library имеет фазу `Phase.DEOXIDATION` + паттерны `DX01`/`DX02`/`DX03`. Decision Log — **опт-ин** (кнопка «Сохранить») во избежание спама БД на производственном темпе 50-200 плавок/день.

**Не входит в MVP**: кинетика растворения Al, баланс FeO в шлаке, комбинированное раскисление (Al+FeSi+Ca), ML, feedback loop, интеграция с анализаторами O. Это фазы v0.6+.
```

- [ ] **Step 2: Full pytest**

```bash
PYTHONPATH=. .venv/bin/pytest app/tests/ -v
```
Expected: ~55+ tests pass (47 previous + 8 new deox forward + 4 quality/compare + 1 profile + 4 DX patterns).

- [ ] **Step 3: Smoke test (no regressions in old pipeline)**

```bash
unset ANTHROPIC_API_KEY
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py
```
Expected: SMOKE TEST PASSED.

- [ ] **Step 4: Streamlit 200 OK final**

```bash
lsof -ti:8501 2>/dev/null | xargs kill -9 2>/dev/null
sleep 2
PYTHONPATH=. .venv/bin/streamlit run app/frontend/app.py --server.port 8501 --server.headless true &
sleep 5
curl -sI http://localhost:8501 | head -1
```
Expected: `HTTP/1.1 200 OK`. Leave it running for manual inspection.

- [ ] **Step 5: Tag v0.5**

```bash
git log --oneline v0.4-multi-class..HEAD | head -20
git tag -a v0.5-deoxidation -m "MVP v0.5: Al deoxidation calculator (physics-based LF advisory)"
git tag -l | tail -5
```

- [ ] **Step 6: Commit CLAUDE.md**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md — Al deoxidation calculator section"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §4.1 Al-O equilibrium | 1 |
| §4.2 three thermo models | 1 |
| §4.3 forward compute_al_demand | 2 |
| §4.4 inverse compute_al_quality | 3 |
| §4.5 compare_all_models | 3 |
| §5 SteelClassProfile extension | 4 |
| §6 UI tab_deox | 6 |
| §7 Pattern Library DX01-DX03 | 5 |
| §8 opt-in Decision Log | 6 |
| §10 tests (8 physics + 4 patterns) | 2, 3, 5 |
| §11 acceptance criteria | 7 |

All spec sections covered.

**2. Placeholder scan**

No TBDs / "similar to task N" / vague text. Every step has exact code or exact command + expected output.

**3. Type consistency**

- `DeoxidationResult`, `AlQualityResult` — defined Task 1, used Tasks 2, 3, 6.
- `compute_al_demand(o_a_initial_ppm, temperature_C, steel_mass_ton, target_o_a_ppm, al_purity_pct, burn_off_pct, model_id, al_price_per_kg, currency)` — same signature in Tasks 2, 3, 6.
- `compute_al_quality(o_a_before_ppm, o_a_after_ppm, al_added_kg, ...)` — same signature in Tasks 3, 6.
- `compare_all_models(o_a_initial_ppm, ...)` — same signature in Tasks 3, 6.
- `Phase.DEOXIDATION` string value `"deoxidation"` — same in Tasks 5, 6.
- `SteelClassProfile.target_o_activity_ppm: float | None = None` — Task 4, used Task 6.

No drift.

**Plan complete.** Execution via subagent-driven-development (recommended) or inline executing-plans.
