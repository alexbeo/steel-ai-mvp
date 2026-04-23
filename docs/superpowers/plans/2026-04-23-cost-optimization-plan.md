# Cost-optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Заменить хардкод `ELEMENT_PRICES_EUR_PER_KG` в `inverse_designer.py` на полноценный ferroalloy-based price snapshot с датой, валютой, редактируемым Streamlit UI, Pareto plot σт×cost, per-candidate breakdown и Decision Log аудитом.

**Architecture:** Новый модуль `app/backend/cost_model.py` инкапсулирует типы (`PriceSnapshot`, `Material`, `CostBreakdown`) и функцию `compute_cost(composition_pct, snapshot, mode)`. `inverse_designer.py` получает опциональный `price_snapshot` — если задан, использует `compute_cost`, иначе fallback на legacy. Streamlit добавляет expander с `st.data_editor` + Pareto plot + breakdown. Все runs сохраняют snapshot в `decision_log/price_snapshots/<ts>.yaml` и пишут в Decision Log. Pattern Library расширяется паттернами C01–C04.

**Tech Stack:** Python 3.11+, pymoo (NSGA-II), pandas, PyYAML, Streamlit (`st.data_editor`, Altair). Существующий стек проекта — ничего нового крупного.

**Spec:** `docs/superpowers/specs/2026-04-23-cost-optimization-design.md`

---

## File Structure

### Новые файлы
- `app/backend/cost_model.py` — типы, `compute_cost`, `load_snapshot`, `save_snapshot`, `seed_snapshot`, `validate_snapshot`, `required_elements_for_design`, `PriceSnapshotIncomplete`.
- `data/prices/seed_2026-04-23.yaml` — seed-прайс (10 позиций, RUB).
- `app/tests/__init__.py` — пустой маркер пакета.
- `app/tests/test_cost_model.py` — 10 unit-тестов.

### Модифицируемые
- `app/backend/inverse_designer.py` — приём `price_snapshot`, pre-check, breakdown, Decision Log snapshot.
- `app/backend/engine.py` — `_build_critic_context` для фазы `inverse_design` дополняется cost-мета.
- `app/frontend/app.py` — price-editor expander, Pareto plot, breakdown в per-candidate expander.
- `pattern_library/patterns.py` — C01–C04.
- `requirements.txt` — добавить `pyyaml>=6.0`.
- `.gitignore` — добавить `decision_log/price_snapshots/`.
- `CLAUDE.md` — строка про cost_model под «ML-конвенциями».
- `scripts/smoke_test.py` — дополнить вторым проходом с `seed_snapshot()`.

---

## Task 1: cost_model — типы и исключения

**Files:**
- Create: `app/backend/cost_model.py`

- [ ] **Step 1: Создать каркас модуля с типами**

Создать файл `app/backend/cost_model.py` со следующим содержимым:

```python
"""
Cost model for HSLA steel inverse design.

Ferroalloy-based pricing: each alloying element maps to a preferred
ferroalloy (FeNb-65, FeMn-80, ...). Compute cost per ton of steel
given a composition (in %) and a PriceSnapshot.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Literal

Kind = Literal["base", "ferroalloy", "pure"]
Currency = Literal["RUB", "USD", "EUR"]
CostMode = Literal["full", "incremental"]


# Element -> material preference. Elements not listed in NON_PRICED_ELEMENTS
# must appear here, else compute_cost raises.
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

# Elements that come "for free" with scrap/process — not priced separately.
NON_PRICED_ELEMENTS: set[str] = {"C", "P", "S", "N"}

# Physically plausible content ranges for ferroalloys (used by C02).
FERROALLOY_CONTENT_RANGES: dict[str, tuple[str, float, float]] = {
    "FeNb-65": ("Nb", 0.55, 0.75),
    "FeMn-80": ("Mn", 0.70, 0.88),
    "FeSi-75": ("Si", 0.70, 0.80),
    "FeCr-HC": ("Cr", 0.55, 0.70),
    "FeV-50":  ("V",  0.40, 0.60),
    "FeTi-70": ("Ti", 0.65, 0.75),
    "FeMo":    ("Mo", 0.55, 0.70),
    "FeNi":    ("Ni", 0.20, 0.40),
}


@dataclass(frozen=True)
class Material:
    id: str
    kind: Kind
    price_per_kg: float
    element_content: dict[str, float]


@dataclass
class PriceSnapshot:
    date: date
    currency: Currency
    materials: dict[str, Material]
    source: str = "manual"
    notes: str = ""


@dataclass
class CostContribution:
    material_id: str
    mass_kg_per_ton_steel: float
    price_per_kg: float
    contribution_per_ton: float


@dataclass
class CostBreakdown:
    total_per_ton: float
    contributions: list[CostContribution]
    mode: CostMode
    currency: Currency

    @property
    def total_per_kg(self) -> float:
        return self.total_per_ton / 1000.0


class PriceSnapshotIncomplete(ValueError):
    """Raised when a price snapshot is missing entries for required elements."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"Нет цен для: {', '.join(missing)}")
```

- [ ] **Step 2: Убедиться, что модуль импортируется**

Run: `PYTHONPATH=. .venv/bin/python -c "from app.backend.cost_model import PriceSnapshot, Material, compute_cost" 2>&1`
Expected: `ImportError: cannot import name 'compute_cost'` (функция ещё не добавлена — это ожидаемо, но сам модуль загружается без SyntaxError).

Если import типов проходит — ок, переходим к Task 2. Если SyntaxError — починить.

- [ ] **Step 3: Commit**

```bash
git add app/backend/cost_model.py
git commit -m "feat(cost_model): scaffold types, exceptions, element-material preference"
```

---

## Task 2: compute_cost — базовая реализация (pure iron + один легирующий)

**Files:**
- Create: `app/tests/__init__.py`
- Create: `app/tests/test_cost_model.py`
- Modify: `app/backend/cost_model.py`

- [ ] **Step 1: Создать пустой `app/tests/__init__.py`**

```bash
touch app/tests/__init__.py
```

- [ ] **Step 2: Создать тестовый файл с двумя первыми тестами**

Создать `app/tests/test_cost_model.py`:

```python
"""Unit tests for cost_model."""
from __future__ import annotations

from datetime import date

import pytest

from app.backend.cost_model import (
    Material, PriceSnapshot, CostMode,
    compute_cost, PriceSnapshotIncomplete,
)


def _rub_seed() -> PriceSnapshot:
    """Tiny fixture: scrap + FeNb-65 + FeMn-80 — enough for early tests."""
    return PriceSnapshot(
        date=date(2026, 4, 23),
        currency="RUB",
        materials={
            "scrap":   Material("scrap",   "base",       42.0,  {"Fe": 1.0}),
            "FeMn-80": Material("FeMn-80", "ferroalloy", 180.0, {"Mn": 0.80, "Fe": 0.20}),
            "FeNb-65": Material("FeNb-65", "ferroalloy", 3600.0, {"Nb": 0.65, "Fe": 0.35}),
        },
    )


def test_compute_cost_pure_iron_full_mode():
    """Empty composition → 1000 kg of scrap per ton × 42 ₽/kg = 42 000 ₽/т."""
    snapshot = _rub_seed()
    breakdown = compute_cost({}, snapshot, mode="full")

    assert breakdown.mode == "full"
    assert breakdown.currency == "RUB"
    assert breakdown.total_per_ton == pytest.approx(42_000.0, rel=1e-6)
    assert len(breakdown.contributions) == 1
    assert breakdown.contributions[0].material_id == "scrap"
    assert breakdown.contributions[0].mass_kg_per_ton_steel == pytest.approx(1000.0)


def test_compute_cost_nb_only_full_and_incremental():
    """0.65% Nb via FeNb-65:
        alloy_mass = 6.5 / 0.65 = 10 kg/t; alloy_cost = 10 × 3600 = 36 000 ₽/т
        full = 990 × 42 + 36 000 = 77 580 ₽/т
        incremental = 36 000 ₽/т
    """
    snapshot = _rub_seed()
    composition = {"nb_pct": 0.65}

    full = compute_cost(composition, snapshot, mode="full")
    assert full.total_per_ton == pytest.approx(77_580.0, rel=1e-6)

    inc = compute_cost(composition, snapshot, mode="incremental")
    assert inc.total_per_ton == pytest.approx(36_000.0, rel=1e-6)

    # Discover FeNb-65 contribution inside either breakdown.
    fenb = next(c for c in full.contributions if c.material_id == "FeNb-65")
    assert fenb.mass_kg_per_ton_steel == pytest.approx(10.0)
    assert fenb.contribution_per_ton == pytest.approx(36_000.0)
```

- [ ] **Step 3: Запустить тесты — они должны упасть**

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_cost_model.py -v`
Expected: `ImportError: cannot import name 'compute_cost'`.

- [ ] **Step 4: Реализовать `compute_cost` в `cost_model.py`**

Добавить в конец `app/backend/cost_model.py`:

```python
def compute_cost(
    composition_pct: dict[str, float],
    snapshot: PriceSnapshot,
    mode: CostMode = "full",
) -> CostBreakdown:
    """Compute cost per ton of steel from composition (in %).

    composition_pct uses pct-suffix keys matching VARIABLE_BOUNDS_HSLA:
    {"mn_pct": 1.5, "nb_pct": 0.04, ...}. Elements not in
    FERROALLOY_PREFERENCE and not in NON_PRICED_ELEMENTS raise ValueError.
    n_ppm and non-_pct keys are ignored.
    """
    alloy_contribs: list[CostContribution] = []
    total_alloy_mass = 0.0

    for var, pct in composition_pct.items():
        if not var.endswith("_pct"):
            continue
        elem = var[:-4].capitalize()
        if elem in NON_PRICED_ELEMENTS:
            continue
        if pct <= 0:
            continue
        if elem not in FERROALLOY_PREFERENCE:
            raise ValueError(f"Нет маппинга для элемента {elem}")
        material_id = FERROALLOY_PREFERENCE[elem]
        if material_id not in snapshot.materials:
            raise PriceSnapshotIncomplete([elem])
        material = snapshot.materials[material_id]
        content = material.element_content.get(elem, 0.0)
        if content <= 0:
            raise ValueError(f"Material {material_id} не содержит {elem}")
        elem_mass = pct * 10.0                    # 1% = 10 kg на 1 т
        alloy_mass = elem_mass / content
        contribution = alloy_mass * material.price_per_kg
        alloy_contribs.append(CostContribution(
            material_id=material_id,
            mass_kg_per_ton_steel=alloy_mass,
            price_per_kg=material.price_per_kg,
            contribution_per_ton=contribution,
        ))
        total_alloy_mass += alloy_mass

    if mode == "full":
        if "scrap" not in snapshot.materials:
            raise PriceSnapshotIncomplete(["scrap (base material)"])
        scrap = snapshot.materials["scrap"]
        base_mass = max(0.0, 1000.0 - total_alloy_mass)
        base_contribution = CostContribution(
            material_id="scrap",
            mass_kg_per_ton_steel=base_mass,
            price_per_kg=scrap.price_per_kg,
            contribution_per_ton=base_mass * scrap.price_per_kg,
        )
        contributions = [base_contribution] + alloy_contribs
    else:
        contributions = alloy_contribs

    total = sum(c.contribution_per_ton for c in contributions)
    return CostBreakdown(
        total_per_ton=total,
        contributions=contributions,
        mode=mode,
        currency=snapshot.currency,
    )
```

- [ ] **Step 5: Запустить тесты — должны проходить**

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_cost_model.py -v`
Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add app/backend/cost_model.py app/tests/__init__.py app/tests/test_cost_model.py
git commit -m "feat(cost_model): compute_cost with full/incremental modes (TDD)"
```

---

## Task 3: compute_cost — расширенные тесты и sanity ranges

**Files:**
- Modify: `app/tests/test_cost_model.py`

- [ ] **Step 1: Добавить тесты типичного HSLA и invariant full − incremental == base**

Дописать в `app/tests/test_cost_model.py`:

```python
def _full_seed_rub() -> PriceSnapshot:
    """Full 10-material seed (matching data/prices/seed_2026-04-23.yaml)."""
    return PriceSnapshot(
        date=date(2026, 4, 23),
        currency="RUB",
        source="seed",
        materials={
            "scrap":   Material("scrap",   "base",       42.0,   {"Fe": 1.0}),
            "FeMn-80": Material("FeMn-80", "ferroalloy", 180.0,  {"Mn": 0.80, "Fe": 0.20}),
            "FeSi-75": Material("FeSi-75", "ferroalloy", 210.0,  {"Si": 0.75, "Fe": 0.25}),
            "FeCr-HC": Material("FeCr-HC", "ferroalloy", 260.0,  {"Cr": 0.65, "Fe": 0.35}),
            "FeNi":    Material("FeNi",    "ferroalloy", 1200.0, {"Ni": 0.30, "Fe": 0.70}),
            "FeMo":    Material("FeMo",    "ferroalloy", 3400.0, {"Mo": 0.60, "Fe": 0.40}),
            "FeV-50":  Material("FeV-50",  "ferroalloy", 1900.0, {"V":  0.50, "Fe": 0.50}),
            "FeNb-65": Material("FeNb-65", "ferroalloy", 3600.0, {"Nb": 0.65, "Fe": 0.35}),
            "FeTi-70": Material("FeTi-70", "ferroalloy", 720.0,  {"Ti": 0.70, "Fe": 0.30}),
            "Cu":      Material("Cu",      "pure",       850.0,  {"Cu": 1.0}),
            "Al":      Material("Al",      "pure",       240.0,  {"Al": 1.0}),
        },
    )


def test_compute_cost_full_minus_incremental_equals_base_cost():
    snapshot = _full_seed_rub()
    composition = {"mn_pct": 1.5, "nb_pct": 0.04, "ti_pct": 0.02}
    full = compute_cost(composition, snapshot, mode="full")
    inc  = compute_cost(composition, snapshot, mode="incremental")
    base_contrib = next(c for c in full.contributions if c.material_id == "scrap")
    assert (full.total_per_ton - inc.total_per_ton) == pytest.approx(
        base_contrib.contribution_per_ton, rel=1e-9
    )


def test_compute_cost_sanity_range_typical_hsla():
    """Realistic pipe-HSLA composition: cost should land in 40-90 k ₽/т full."""
    snapshot = _full_seed_rub()
    composition = {
        "c_pct": 0.08, "si_pct": 0.4, "mn_pct": 1.5,
        "p_pct": 0.015, "s_pct": 0.005,
        "cr_pct": 0.10, "ni_pct": 0.10, "mo_pct": 0.02,
        "cu_pct": 0.20, "al_pct": 0.035,
        "v_pct": 0.03, "nb_pct": 0.04, "ti_pct": 0.02,
    }
    breakdown = compute_cost(composition, snapshot, mode="full")
    assert 40_000 <= breakdown.total_per_ton <= 90_000, (
        f"cost={breakdown.total_per_ton:.0f} out of sanity range"
    )
    # C/P/S/N are ignored:
    priced_ids = {c.material_id for c in breakdown.contributions}
    assert "Fe-C" not in priced_ids and "FeP" not in priced_ids


def test_compute_cost_unknown_element_raises():
    snapshot = _full_seed_rub()
    with pytest.raises(ValueError, match="Нет маппинга"):
        compute_cost({"w_pct": 0.5}, snapshot, mode="full")
```

- [ ] **Step 2: Запустить все тесты**

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_cost_model.py -v`
Expected: `5 passed`.

Если упал `test_compute_cost_sanity_range_typical_hsla` (out-of-range) — вывести полученное значение `print(breakdown.total_per_ton)` и сверить с ручным расчётом spec 11. Правим — либо исправляем bug в compute_cost, либо (если цифра корректная, но границы нужно сдвинуть) пересогласуем с spec.

- [ ] **Step 3: Commit**

```bash
git add app/tests/test_cost_model.py
git commit -m "test(cost_model): full/incremental invariant + HSLA sanity range"
```

---

## Task 4: Snapshot YAML I/O (`save_snapshot`, `load_snapshot`, `seed_snapshot`)

**Files:**
- Modify: `app/backend/cost_model.py`
- Modify: `app/tests/test_cost_model.py`
- Modify: `requirements.txt`
- Create: `data/prices/seed_2026-04-23.yaml`

- [ ] **Step 1: Добавить `pyyaml` в requirements и установить**

Открыть `requirements.txt`, добавить под `pyarrow>=13.0`:

```
pyyaml>=6.0
```

Установить:

```bash
.venv/bin/pip install "pyyaml>=6.0"
```

- [ ] **Step 2: Создать seed YAML**

Создать `data/prices/seed_2026-04-23.yaml`:

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

- [ ] **Step 3: Добавить failing-test для YAML I/O**

Добавить в `app/tests/test_cost_model.py`:

```python
from pathlib import Path
from app.backend.cost_model import (
    load_snapshot, save_snapshot, seed_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_load_seed_snapshot_yaml():
    path = PROJECT_ROOT / "data" / "prices" / "seed_2026-04-23.yaml"
    snapshot = load_snapshot(path)
    assert snapshot.currency == "RUB"
    assert snapshot.date == date(2026, 4, 23)
    assert "scrap" in snapshot.materials
    assert "FeNb-65" in snapshot.materials
    assert snapshot.materials["FeNb-65"].element_content == {"Nb": 0.65, "Fe": 0.35}


def test_save_then_load_roundtrip(tmp_path):
    original = _full_seed_rub()
    path = tmp_path / "snap.yaml"
    save_snapshot(original, path)
    loaded = load_snapshot(path)
    assert loaded.date == original.date
    assert loaded.currency == original.currency
    assert set(loaded.materials) == set(original.materials)
    for mid, mat in original.materials.items():
        assert loaded.materials[mid].price_per_kg == pytest.approx(mat.price_per_kg)
        assert loaded.materials[mid].element_content == mat.element_content


def test_seed_snapshot_is_loadable():
    snapshot = seed_snapshot()
    assert snapshot.source == "seed"
    assert "FeNb-65" in snapshot.materials
```

Запустить: `PYTHONPATH=. .venv/bin/pytest app/tests/test_cost_model.py::test_load_seed_snapshot_yaml -v`
Expected: `ImportError: cannot import name 'load_snapshot'`.

- [ ] **Step 4: Реализовать YAML I/O в `cost_model.py`**

В конец `app/backend/cost_model.py` добавить:

```python
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_SNAPSHOT_PATH = PROJECT_ROOT / "data" / "prices" / "seed_2026-04-23.yaml"


def load_snapshot(path: Path) -> PriceSnapshot:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    materials = {}
    for mid, md in data["materials"].items():
        _validate_material_dict(mid, md)
        materials[mid] = Material(
            id=mid,
            kind=md["kind"],
            price_per_kg=float(md["price_per_kg"]),
            element_content={k: float(v) for k, v in md["element_content"].items()},
        )
    snap_date = data["date"]
    if isinstance(snap_date, str):
        snap_date = date.fromisoformat(snap_date)
    return PriceSnapshot(
        date=snap_date,
        currency=data["currency"],
        materials=materials,
        source=data.get("source", "manual"),
        notes=data.get("notes", ""),
    )


def save_snapshot(snapshot: PriceSnapshot, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": snapshot.date.isoformat(),
        "currency": snapshot.currency,
        "source": snapshot.source,
        "notes": snapshot.notes,
        "materials": {
            mid: {
                "kind": m.kind,
                "price_per_kg": m.price_per_kg,
                "element_content": dict(m.element_content),
            }
            for mid, m in snapshot.materials.items()
        },
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def seed_snapshot() -> PriceSnapshot:
    """Load the canonical seed snapshot from data/prices/."""
    return load_snapshot(SEED_SNAPSHOT_PATH)


def _validate_material_dict(mid: str, md: dict) -> None:
    if md.get("price_per_kg", 0) <= 0:
        raise ValueError(f"{mid}: price_per_kg must be > 0")
    ec = md.get("element_content") or {}
    if not ec:
        raise ValueError(f"{mid}: element_content is empty")
    s = sum(float(v) for v in ec.values())
    if abs(s - 1.0) > 0.02:
        raise ValueError(
            f"{mid}: element_content sum = {s:.3f}, must be ≈ 1.0 (±0.02)"
        )
    kind = md.get("kind")
    if kind not in ("base", "ferroalloy", "pure"):
        raise ValueError(f"{mid}: kind must be base|ferroalloy|pure, got {kind}")
```

- [ ] **Step 5: Запустить тесты**

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_cost_model.py -v`
Expected: `8 passed`.

- [ ] **Step 6: Commit**

```bash
git add app/backend/cost_model.py app/tests/test_cost_model.py requirements.txt data/prices/seed_2026-04-23.yaml
git commit -m "feat(cost_model): YAML snapshot I/O + seed_2026-04-23 (10 materials)"
```

---

## Task 5: validate_snapshot + required_elements_for_design

**Files:**
- Modify: `app/backend/cost_model.py`
- Modify: `app/tests/test_cost_model.py`

- [ ] **Step 1: Добавить failing-тесты**

В `app/tests/test_cost_model.py` дописать:

```python
from app.backend.cost_model import (
    validate_snapshot, required_elements_for_design,
)


def test_required_elements_for_design_skips_non_priced_and_nonchem():
    bounds = {
        "c_pct": [0.04, 0.12],
        "mn_pct": [0.9, 1.75],
        "nb_pct": [0.0, 0.06],
        "n_ppm": [30, 80],
        "rolling_finish_temp": [750.0, 850.0],
    }
    required = required_elements_for_design(bounds)
    # C is non-priced, n_ppm/rolling_finish_temp — skipped entirely.
    assert required == {"Mn", "Nb"}


def test_validate_snapshot_returns_missing_elements():
    snapshot = _rub_seed()                    # only scrap, FeMn-80, FeNb-65
    required = {"Mn", "Nb", "Ti", "Mo"}
    missing = validate_snapshot(snapshot, required)
    assert missing == ["Mo", "Ti"]           # alphabetical


def test_validate_snapshot_all_covered():
    snapshot = _full_seed_rub()
    required = {"Mn", "Nb", "Ti"}
    assert validate_snapshot(snapshot, required) == []


def test_full_seed_covers_all_hsla_design_elements():
    from app.backend.inverse_designer import VARIABLE_BOUNDS_HSLA
    snapshot = seed_snapshot()
    missing = validate_snapshot(
        snapshot, required_elements_for_design(VARIABLE_BOUNDS_HSLA)
    )
    assert missing == []
```

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_cost_model.py -v -k required or validate`
Expected: `ImportError: cannot import name 'validate_snapshot'`.

- [ ] **Step 2: Реализовать функции**

Добавить в `app/backend/cost_model.py`:

```python
def required_elements_for_design(variable_bounds: dict) -> set[str]:
    """From VARIABLE_BOUNDS_HSLA-style dict → set of elements that need pricing.

    Skips NON_PRICED_ELEMENTS (C/P/S/N) and non-chemistry variables
    (rolling_finish_temp, cooling_rate_c_per_s, n_ppm).
    """
    required: set[str] = set()
    for var in variable_bounds:
        if not var.endswith("_pct"):
            continue
        elem = var[:-4].capitalize()
        if elem in NON_PRICED_ELEMENTS:
            continue
        required.add(elem)
    return required


def validate_snapshot(
    snapshot: PriceSnapshot, required_elements: set[str]
) -> list[str]:
    """Return sorted list of required elements NOT covered by snapshot.

    Empty list = snapshot is sufficient for the given design space.
    """
    covered: set[str] = set()
    for m in snapshot.materials.values():
        covered.update(m.element_content.keys())
    return sorted(required_elements - covered)
```

- [ ] **Step 3: Запустить тесты**

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_cost_model.py -v`
Expected: `12 passed`.

- [ ] **Step 4: Commit**

```bash
git add app/backend/cost_model.py app/tests/test_cost_model.py
git commit -m "feat(cost_model): validate_snapshot + required_elements_for_design"
```

---

## Task 6: Pattern Library C01–C04

**Files:**
- Modify: `pattern_library/patterns.py`
- Modify: `app/tests/test_cost_model.py` (добавить тесты паттернов)

- [ ] **Step 1: Написать failing-тесты для паттернов**

В `app/tests/test_cost_model.py` дописать:

```python
from datetime import date, timedelta
from pattern_library.patterns import run_all_patterns, Phase


def _ctx_for_inverse_with_snapshot(snapshot: PriceSnapshot) -> dict:
    from dataclasses import asdict
    return {
        "phase": "inverse_design",
        "pareto_size": 10,
        "objectives_normalized": True,
        "n_objectives": 3,
        "variable_bounds": {"nb_pct": [0.0, 0.06]},
        "training_variable_ranges": {"nb_pct": [0.0, 0.06]},
        # New cost-related context keys:
        "price_snapshot_meta": {
            "date": snapshot.date.isoformat(),
            "currency": snapshot.currency,
            "source": snapshot.source,
            "n_materials": len(snapshot.materials),
        },
        "snapshot_materials": [asdict(m) for m in snapshot.materials.values()],
        "cost_breakdown_samples": [],
    }


def test_pattern_c01_stale_snapshot():
    old_snap = PriceSnapshot(
        date=date.today() - timedelta(days=45),
        currency="RUB",
        materials=_full_seed_rub().materials,
    )
    warnings = run_all_patterns(
        _ctx_for_inverse_with_snapshot(old_snap), phase=Phase.INVERSE_DESIGN
    )
    ids = {w["pattern_id"] for w in warnings}
    assert "C01" in ids


def test_pattern_c02_ferroalloy_physical_range():
    bad = _full_seed_rub()
    bad.materials["FeNb-65"] = Material(
        "FeNb-65", "ferroalloy", 3600.0, {"Nb": 0.90, "Fe": 0.10}
    )
    warnings = run_all_patterns(
        _ctx_for_inverse_with_snapshot(bad), phase=Phase.INVERSE_DESIGN
    )
    ids = {w["pattern_id"] for w in warnings}
    assert "C02" in ids


def test_pattern_c03_corrupt_breakdown():
    ctx = _ctx_for_inverse_with_snapshot(_full_seed_rub())
    ctx["cost_breakdown_samples"] = [{
        "total_per_ton": -100.0,
        "contributions": [{
            "material_id": "FeNb-65",
            "mass_kg_per_ton_steel": 2000.0,      # > 1000 → corrupt
            "price_per_kg": 3600.0,
            "contribution_per_ton": -50.0,         # < 0 → corrupt
        }],
        "mode": "full",
        "currency": "RUB",
    }]
    warnings = run_all_patterns(ctx, phase=Phase.INVERSE_DESIGN)
    ids = {w["pattern_id"] for w in warnings}
    assert "C03" in ids


def test_pattern_c04_element_missing_in_snapshot():
    ctx = _ctx_for_inverse_with_snapshot(_rub_seed())  # no FeMo
    ctx["design_required_elements"] = ["Mn", "Nb", "Mo"]
    warnings = run_all_patterns(ctx, phase=Phase.INVERSE_DESIGN)
    ids = {w["pattern_id"] for w in warnings}
    assert "C04" in ids
```

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_cost_model.py -v -k pattern`
Expected: все 4 падают — паттерны ещё не зарегистрированы.

- [ ] **Step 2: Реализовать проверки в `pattern_library/patterns.py`**

В `pattern_library/patterns.py` над финальной `PATTERNS: list[Pattern] = [...]` добавить 4 функции и расширить список:

```python
from datetime import date, datetime

def _check_c01_snapshot_age(ctx: dict) -> CheckResult:
    meta = ctx.get("price_snapshot_meta")
    if not meta:
        return CheckResult(False)
    try:
        snap_date = date.fromisoformat(meta["date"])
    except (KeyError, TypeError, ValueError):
        return CheckResult(False)
    age_days = (date.today() - snap_date).days
    if age_days > 30:
        return CheckResult(
            True,
            message=f"Прайс от {snap_date} старше 30 дней ({age_days} дн.). "
                    f"Актуализируйте перед презентацией/продакшеном.",
            details={"age_days": age_days},
        )
    return CheckResult(False)


_FERROALLOY_RANGES = {
    "FeNb-65": ("Nb", 0.55, 0.75),
    "FeMn-80": ("Mn", 0.70, 0.88),
    "FeSi-75": ("Si", 0.70, 0.80),
    "FeCr-HC": ("Cr", 0.55, 0.70),
    "FeV-50":  ("V",  0.40, 0.60),
    "FeTi-70": ("Ti", 0.65, 0.75),
    "FeMo":    ("Mo", 0.55, 0.70),
    "FeNi":    ("Ni", 0.20, 0.40),
}


def _check_c02_ferroalloy_content(ctx: dict) -> CheckResult:
    materials = ctx.get("snapshot_materials") or []
    violations = []
    for m in materials:
        if m.get("kind") != "ferroalloy":
            continue
        rule = _FERROALLOY_RANGES.get(m["id"])
        if not rule:
            continue
        elem, lo, hi = rule
        content = (m.get("element_content") or {}).get(elem)
        if content is None:
            continue
        if content < lo or content > hi:
            violations.append(
                f"{m['id']} содержит {elem}={content:.2f} (допустимо {lo}-{hi})"
            )
    if violations:
        return CheckResult(
            True,
            message="Физически невозможное содержание в ферросплаве: "
                    + "; ".join(violations),
        )
    return CheckResult(False)


def _check_c03_corrupt_breakdown(ctx: dict) -> CheckResult:
    samples = ctx.get("cost_breakdown_samples") or []
    anomalies = []
    for i, b in enumerate(samples):
        if not isinstance(b, dict):
            continue
        for c in b.get("contributions", []):
            if c.get("contribution_per_ton", 0) < 0:
                anomalies.append(f"#{i}: отрицательный вклад {c['material_id']}")
            if c.get("mass_kg_per_ton_steel", 0) > 1000.0:
                anomalies.append(f"#{i}: масса {c['material_id']} > 1000 кг/т")
    if anomalies:
        return CheckResult(
            True,
            message="Баг в compute_cost: " + "; ".join(anomalies[:5]),
        )
    return CheckResult(False)


def _check_c04_missing_element_in_snapshot(ctx: dict) -> CheckResult:
    required = set(ctx.get("design_required_elements") or [])
    if not required:
        return CheckResult(False)
    materials = ctx.get("snapshot_materials") or []
    covered: set[str] = set()
    for m in materials:
        covered.update((m.get("element_content") or {}).keys())
    missing = sorted(required - covered)
    if missing:
        return CheckResult(
            True,
            message=f"Элементы {missing} присутствуют в design space, "
                    f"но не покрыты прайс-снимком.",
            details={"missing": missing},
        )
    return CheckResult(False)
```

Затем расширить список `PATTERNS`, добавив в конец перед закрывающей `]`:

```python
    Pattern(
        id="C01", title="Устаревший прайс-снимок",
        phase=Phase.INVERSE_DESIGN, severity=Severity.MEDIUM,
        description="Snapshot старше 30 дней",
        check=_check_c01_snapshot_age,
        suggestion="Обновите snapshot перед продакшеном.",
    ),
    Pattern(
        id="C02", title="Физически невозможное содержание ферросплава",
        phase=Phase.INVERSE_DESIGN, severity=Severity.HIGH,
        description="element_content в ферросплаве вне физического диапазона",
        check=_check_c02_ferroalloy_content,
        suggestion="Проверьте ввод element_content, сверьте со справочником.",
    ),
    Pattern(
        id="C03", title="Некорректный CostBreakdown",
        phase=Phase.INVERSE_DESIGN, severity=Severity.HIGH,
        description="В contributions есть отрицательный вклад или масса > 1000 кг/т",
        check=_check_c03_corrupt_breakdown,
        suggestion="Баг в compute_cost или в парсинге snapshot.",
    ),
    Pattern(
        id="C04", title="Элемент не покрыт прайс-снимком",
        phase=Phase.INVERSE_DESIGN, severity=Severity.HIGH,
        description="design_required_elements содержит элемент без материала",
        check=_check_c04_missing_element_in_snapshot,
        suggestion="Добавьте соответствующий ферросплав/чистый материал в snapshot.",
    ),
```

- [ ] **Step 3: Запустить ВСЕ тесты**

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_cost_model.py -v`
Expected: `16 passed`.

- [ ] **Step 4: Commit**

```bash
git add pattern_library/patterns.py app/tests/test_cost_model.py
git commit -m "feat(patterns): C01-C04 for cost-optimization (stale price, bad ferroalloy, corrupt breakdown, missing element)"
```

---

## Task 7: Рефакторинг inverse_designer — приём snapshot, pre-check, breakdown

**Files:**
- Modify: `app/backend/inverse_designer.py`
- Modify: `app/tests/test_cost_model.py` (integration-тест)

- [ ] **Step 1: Написать integration-тест `run_inverse_design` с snapshot**

В `app/tests/test_cost_model.py` дописать:

```python
import pytest


@pytest.mark.integration
def test_run_inverse_design_with_snapshot_adds_breakdown(monkeypatch):
    """End-to-end: prep prices, call run_inverse_design, check candidates carry cost."""
    from app.backend.inverse_designer import run_inverse_design
    from app.backend.model_trainer import train_model
    from app.backend.data_curator import save_sample_dataset
    from app.backend.feature_eng import compute_hsla_features, PIPE_HSLA_FEATURE_SET
    import pandas as pd

    # Reuse cached data if present, otherwise generate tiny sample.
    data_path = PROJECT_ROOT / "data" / "hsla_synthetic.parquet"
    if not data_path.exists():
        save_sample_dataset()
    df = pd.read_parquet(data_path)
    df_feat = compute_hsla_features(df)
    feat = [f for f in PIPE_HSLA_FEATURE_SET if f in df_feat.columns]
    trained = train_model(df_feat, "yield_strength_mpa", feat, n_optuna_trials=5)

    snapshot = seed_snapshot()
    result = run_inverse_design(
        model_version=trained.version,
        targets={"yield_strength_mpa": {"min": 485, "max": 580}},
        hard_constraints={"cev_iiw": {"max": 0.43}},
        population_size=20, n_generations=10,
        price_snapshot=snapshot, cost_mode="full",
    )
    assert result["n_candidates"] >= 1
    c = result["pareto_candidates"][0]
    assert c.get("cost") is not None
    assert c["cost"]["currency"] == "RUB"
    assert c["cost"]["mode"] == "full"
    assert 20_000 <= c["cost"]["total_per_ton"] <= 200_000
    assert any(x["material_id"] == "scrap" for x in c["cost"]["contributions"])


def test_run_inverse_design_missing_price_raises():
    from app.backend.inverse_designer import run_inverse_design
    partial = _rub_seed()                           # only Mn + Nb, no Ti/Mo/...
    with pytest.raises(PriceSnapshotIncomplete) as exc:
        run_inverse_design(
            model_version="dummy",
            targets={"yield_strength_mpa": {"min": 485}},
            price_snapshot=partial, cost_mode="full",
        )
    assert "Ti" in exc.value.missing or "Mo" in exc.value.missing
```

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_cost_model.py::test_run_inverse_design_missing_price_raises -v`
Expected: падает — `run_inverse_design` ещё не принимает `price_snapshot`.

- [ ] **Step 2: Модифицировать `inverse_designer.py` — `HSLADesignProblem`**

Открыть `app/backend/inverse_designer.py`. В импортах добавить:

```python
from dataclasses import asdict
from datetime import datetime
from app.backend.cost_model import (
    PriceSnapshot, CostMode, compute_cost, save_snapshot,
    validate_snapshot, required_elements_for_design, PriceSnapshotIncomplete,
)
```

Заменить `HSLADesignProblem.__init__` (строки ~57–83), добавив два новых параметра:

```python
class HSLADesignProblem(ElementwiseProblem):

    def __init__(
        self,
        model_bundle: dict,
        targets: dict[str, dict],
        hard_constraints: dict[str, dict],
        variable_bounds: dict[str, list[float]],
        element_prices: dict[str, float],
        price_snapshot: PriceSnapshot | None = None,
        cost_mode: CostMode = "full",
    ):
        self.model_bundle = model_bundle
        self.targets = targets
        self.hard_constraints = hard_constraints
        self.variable_bounds = variable_bounds
        self.element_prices = element_prices
        self.price_snapshot = price_snapshot
        self.cost_mode = cost_mode

        self.var_names = list(variable_bounds.keys())
        xl = np.array([variable_bounds[n][0] for n in self.var_names])
        xu = np.array([variable_bounds[n][1] for n in self.var_names])

        super().__init__(
            n_var=len(self.var_names),
            n_obj=3,
            n_ieq_constr=len(hard_constraints),
            xl=xl, xu=xu,
        )
```

Заменить блок «Objective 2: alloying cost» в `_evaluate` (строки ~105–111):

```python
        # Objective 2: alloying cost
        if self.price_snapshot is not None:
            composition_pct = {k: v for k, v in row.items() if k.endswith("_pct")}
            breakdown = compute_cost(
                composition_pct, self.price_snapshot, mode=self.cost_mode
            )
            f2 = breakdown.total_per_ton
        else:
            # Legacy fallback: simple element-weighted cost (deprecated).
            cost = 0.0
            for elem, price in self.element_prices.items():
                key = f"{elem}_pct"
                if key in row:
                    cost += row[key] * 10 * price
            f2 = cost
```

- [ ] **Step 3: Модифицировать `run_inverse_design`**

Заменить сигнатуру и добавить pre-check + breakdown:

```python
def run_inverse_design(
    model_version: str,
    targets: dict,
    hard_constraints: dict | None = None,
    variable_bounds: dict | None = None,
    population_size: int = 80,
    n_generations: int = 60,
    random_seed: int = 42,
    price_snapshot: PriceSnapshot | None = None,
    cost_mode: CostMode = "full",
) -> dict:
    bounds = variable_bounds or VARIABLE_BOUNDS_HSLA
    constraints = hard_constraints or {}

    # Pre-check prices (before loading model — fail fast)
    if price_snapshot is not None:
        required = required_elements_for_design(bounds)
        missing = validate_snapshot(price_snapshot, required)
        if missing:
            raise PriceSnapshotIncomplete(missing)
        if cost_mode == "full" and "scrap" not in price_snapshot.materials:
            raise PriceSnapshotIncomplete(["scrap (base material)"])

    model_bundle = load_model(model_version)

    problem = HSLADesignProblem(
        model_bundle=model_bundle,
        targets=targets,
        hard_constraints=constraints,
        variable_bounds=bounds,
        element_prices=ELEMENT_PRICES_EUR_PER_KG,
        price_snapshot=price_snapshot,
        cost_mode=cost_mode,
    )

    algorithm = NSGA2(
        pop_size=population_size,
        sampling=LHS(),
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem, algorithm,
        termination=get_termination("n_gen", n_generations),
        seed=random_seed, verbose=False,
    )

    # Save snapshot to decision_log for audit trail
    snapshot_path: str | None = None
    if price_snapshot is not None:
        from decision_log.logger import log_decision
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_dir = Path(__file__).resolve().parents[2] / "decision_log" / "price_snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snap_dir / f"{run_ts}.yaml"
        save_snapshot(price_snapshot, snap_path)
        snapshot_path = str(snap_path)
        log_decision(
            phase="inverse_design",
            decision=f"Inverse design с cost-optimization "
                     f"(snapshot {price_snapshot.date}, {price_snapshot.currency}, mode={cost_mode})",
            reasoning=f"Source: {price_snapshot.source}. "
                      f"{len(price_snapshot.materials)} материалов. "
                      f"Legacy ELEMENT_PRICES_EUR_PER_KG отключён для этого run.",
            context={
                "snapshot_path": snapshot_path,
                "currency": price_snapshot.currency,
                "n_materials": len(price_snapshot.materials),
                "cost_mode": cost_mode,
            },
            author="inverse_designer",
            tags=["cost_optimization", str(price_snapshot.date)],
        )

    if res.X is None:
        return {
            "pareto_candidates": [], "n_candidates": 0,
            "objectives_normalized": False, "n_objectives": 3,
            "variable_bounds": bounds,
            "cost_currency": price_snapshot.currency if price_snapshot else "EUR (legacy)",
            "cost_mode": cost_mode if price_snapshot else "legacy",
            "price_snapshot_path": snapshot_path,
        }

    # Build candidates
    candidates = []
    var_names = list(bounds.keys())
    for i, x in enumerate(res.X):
        row = dict(zip(var_names, x.tolist()))
        df = pd.DataFrame([row])
        df_feat = compute_hsla_features(df)
        pred = predict_with_uncertainty(model_bundle, df_feat)

        breakdown_dict = None
        if price_snapshot is not None:
            composition_pct = {k: v for k, v in row.items() if k.endswith("_pct")}
            breakdown_dict = asdict(
                compute_cost(composition_pct, price_snapshot, cost_mode)
            )

        candidates.append({
            "idx": i,
            "composition": {k: round(v, 4) for k, v in row.items() if k.endswith("_pct")},
            "processing": {k: round(v, 2) for k, v in row.items() if not k.endswith("_pct")},
            "predicted": {
                "mean": float(pred["prediction"].iloc[0]),
                "lower_90": float(pred["lower_90"].iloc[0]),
                "upper_90": float(pred["upper_90"].iloc[0]),
                "ci_half_width": float(pred["ci_half_width"].iloc[0]),
                "ood_flag": bool(pred["ood_flag"].iloc[0]),
            },
            "derived": {
                "cev_iiw": round(float(df_feat["cev_iiw"].iloc[0]), 4),
                "pcm": round(float(df_feat["pcm"].iloc[0]), 4),
                "cen": round(float(df_feat["cen"].iloc[0]), 4),
                "microalloying_sum": round(float(df_feat["microalloying_sum"].iloc[0]), 4),
            },
            "objectives": {
                "distance_to_target": float(res.F[i, 0]),
                "alloying_cost": float(res.F[i, 1]),
                "prediction_uncertainty": float(res.F[i, 2]),
            },
            "cost": breakdown_dict,
        })

    candidates.sort(key=lambda c: (
        c["objectives"]["distance_to_target"],
        c["objectives"]["alloying_cost"],
    ))

    return {
        "pareto_candidates": candidates,
        "n_candidates": len(candidates),
        "objectives_normalized": False,
        "n_objectives": 3,
        "variable_bounds": bounds,
        "training_variable_ranges": model_bundle["meta"].get("training_ranges", {}),
        "cost_currency": price_snapshot.currency if price_snapshot else "EUR (legacy)",
        "cost_mode": cost_mode if price_snapshot else "legacy",
        "price_snapshot_path": snapshot_path,
    }
```

- [ ] **Step 4: Запустить тесты**

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_cost_model.py -v -m "not integration"`
Expected: `16 passed`.

Integration-тест (`test_run_inverse_design_with_snapshot_adds_breakdown`) запускать отдельно — он обучает модель и может быть медленным:

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_cost_model.py::test_run_inverse_design_with_snapshot_adds_breakdown -v`
Expected: PASS (1–3 минуты).

Также:
Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_cost_model.py::test_run_inverse_design_missing_price_raises -v`
Expected: PASS (<5 сек — падает до загрузки модели).

- [ ] **Step 5: Проверить, что legacy smoke_test всё ещё проходит**

Run: `PYTHONPATH=. .venv/bin/python scripts/smoke_test.py`
Expected: ✓ SMOKE TEST PASSED, как и раньше (smoke-тест вызывает `run_inverse_design` без `price_snapshot`).

- [ ] **Step 6: Commit**

```bash
git add app/backend/inverse_designer.py app/tests/test_cost_model.py
git commit -m "feat(inverse_designer): accept PriceSnapshot; pre-check + breakdown + snapshot audit"
```

---

## Task 8: engine._build_critic_context — расширение для cost

**Files:**
- Modify: `app/backend/engine.py`

- [ ] **Step 1: Найти `_build_critic_context` в engine.py**

Read: `app/backend/engine.py:394-415` (примерно). Найти секцию для `phase == "inverse_design"`.

- [ ] **Step 2: Расширить контекст**

В методе `_build_critic_context` в блок `if phase == "inverse_design":` добавить дополнительные ключи:

```python
        if phase == "inverse_design":
            ctx.update({
                "pareto_size": len(state.candidates),
                "objectives_normalized": result.output.get("objectives_normalized", False),
                "n_objectives": result.output.get("n_objectives", 1),
                "variable_bounds": result.output.get("variable_bounds", {}),
                "training_variable_ranges": state.features.get("training_ranges", {}),
                # Cost-optimization context:
                "price_snapshot_meta": _extract_snapshot_meta(state.user_request),
                "snapshot_materials": _extract_snapshot_materials(state.user_request),
                "design_required_elements": sorted(
                    _elements_from_bounds(result.output.get("variable_bounds", {}))
                ),
                "cost_breakdown_samples": [
                    c.get("cost") for c in state.candidates[:5] if c.get("cost")
                ],
            })
```

Добавить хелперы в начало файла (после импортов):

```python
def _extract_snapshot_meta(user_request: dict) -> dict | None:
    snap = user_request.get("price_snapshot")
    if snap is None:
        return None
    return {
        "date": snap.date.isoformat(),
        "currency": snap.currency,
        "source": snap.source,
        "n_materials": len(snap.materials),
    }


def _extract_snapshot_materials(user_request: dict) -> list[dict]:
    snap = user_request.get("price_snapshot")
    if snap is None:
        return []
    from dataclasses import asdict
    return [asdict(m) for m in snap.materials.values()]


def _elements_from_bounds(bounds: dict) -> set[str]:
    from app.backend.cost_model import (
        NON_PRICED_ELEMENTS, required_elements_for_design,
    )
    return required_elements_for_design(bounds)
```

- [ ] **Step 3: Расширить `_build_task_for_phase` для передачи snapshot агенту**

В `Orchestrator._build_task_for_phase`, в ветке `phase == "inverse_design"` добавить ключи `price_snapshot` и `cost_mode`:

```python
        if phase == "inverse_design":
            return {
                **base_task,
                "operation": "run_nsga2",
                "targets": user_request.get("targets", {}),
                "hard_constraints": user_request.get("constraints", {}),
                "model_version": state.model.get("version"),
                "price_snapshot": user_request.get("price_snapshot"),
                "cost_mode": user_request.get("cost_mode", "full"),
            }
```

И в `InverseDesignerAgent.run` (в `inverse_designer.py`) пробросить параметры в `run_inverse_design`:

```python
            result = run_inverse_design(
                model_version=model_version,
                targets=targets,
                hard_constraints=constraints,
                population_size=task.get("population_size", 80),
                n_generations=task.get("n_generations", 60),
                price_snapshot=task.get("price_snapshot"),
                cost_mode=task.get("cost_mode", "full"),
            )
```

- [ ] **Step 4: Убедиться, что smoke-test всё ещё проходит**

Run: `PYTHONPATH=. .venv/bin/python scripts/smoke_test.py`
Expected: SMOKE TEST PASSED (snapshot не передаётся — legacy путь).

- [ ] **Step 5: Commit**

```bash
git add app/backend/engine.py app/backend/inverse_designer.py
git commit -m "feat(engine): route price_snapshot through Orchestrator + Critic context"
```

---

## Task 9: .gitignore, CLAUDE.md, smoke_test расширение

**Files:**
- Modify: `.gitignore`
- Modify: `CLAUDE.md`
- Modify: `scripts/smoke_test.py`

- [ ] **Step 1: Обновить .gitignore**

В секцию «Decision log» добавить:

```
decision_log/price_snapshots/
```

- [ ] **Step 2: Обновить CLAUDE.md**

В секции «ML-конвенции, заложенные в коде» добавить пункт перед **Target feature set**:

```markdown
- **Cost objective использует ferroalloy pricing** — `app/backend/cost_model.py` с `PriceSnapshot(date, currency, materials)`. Legacy `ELEMENT_PRICES_EUR_PER_KG` остаётся только как fallback при `price_snapshot=None`. Seed-прайс — `data/prices/seed_2026-04-23.yaml` (RUB, 10 позиций, покрывает весь `PIPE_HSLA_FEATURE_SET`). Pattern Library проверяет C01–C04.
```

- [ ] **Step 3: Расширить smoke_test.py — второй проход с snapshot**

В `scripts/smoke_test.py` после существующего STEP 5/6 (inverse design) добавить STEP 5b:

```python
    log.info("=" * 60)
    log.info("STEP 5b/6: Inverse design with cost-optimization (seed RUB prices)")
    log.info("=" * 60)
    from app.backend.cost_model import seed_snapshot
    snapshot = seed_snapshot()
    design_with_cost = run_inverse_design(
        model_version=trained.version,
        targets={"yield_strength_mpa": {"min": 485, "max": 580}},
        hard_constraints={"cev_iiw": {"max": 0.43}, "pcm": {"max": 0.22}},
        population_size=30, n_generations=20,
        price_snapshot=snapshot, cost_mode="full",
    )
    log.info("Candidates with cost: %d", design_with_cost["n_candidates"])
    if design_with_cost["pareto_candidates"]:
        c0 = design_with_cost["pareto_candidates"][0]
        log.info("  Top candidate cost: %.0f %s/т (%s mode)",
                 c0["cost"]["total_per_ton"],
                 c0["cost"]["currency"],
                 c0["cost"]["mode"])
        assert 20_000 <= c0["cost"]["total_per_ton"] <= 200_000
    assert design_with_cost["price_snapshot_path"]
```

- [ ] **Step 4: Запустить smoke test**

Run: `PYTHONPATH=. .venv/bin/python scripts/smoke_test.py`
Expected: проходит обе ветки (legacy + with cost), STEP 5b выводит `Top candidate cost: NN 000 RUB/т (full mode)`.

- [ ] **Step 5: Commit**

```bash
git add .gitignore CLAUDE.md scripts/smoke_test.py
git commit -m "chore: gitignore runtime snapshots, document cost_model, extend smoke_test"
```

---

## Task 10: Streamlit UI — price editor expander

**Files:**
- Modify: `app/frontend/app.py`

- [ ] **Step 1: Добавить импорты и helper-функции в начало файла**

В `app/frontend/app.py` после существующих импортов добавить:

```python
import yaml
from datetime import date
from dataclasses import asdict
from app.backend.cost_model import (
    PriceSnapshot, Material, seed_snapshot, load_snapshot, save_snapshot,
    PriceSnapshotIncomplete,
)


def _snapshot_to_editor_df(snapshot: PriceSnapshot) -> pd.DataFrame:
    rows = []
    for m in snapshot.materials.values():
        elems_str = ";".join(f"{k}={v:.2f}" for k, v in m.element_content.items())
        rows.append({
            "id": m.id, "kind": m.kind,
            "price_per_kg": m.price_per_kg,
            "element_content": elems_str,
        })
    return pd.DataFrame(rows)


def _editor_df_to_snapshot(
    df: pd.DataFrame, snap_date: date, currency: str, source: str
) -> PriceSnapshot:
    materials = {}
    for _, row in df.iterrows():
        mid = str(row["id"]).strip()
        if not mid:
            continue
        ec_str = str(row["element_content"])
        ec = {}
        for pair in ec_str.split(";"):
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            ec[k.strip()] = float(v)
        materials[mid] = Material(
            id=mid,
            kind=str(row["kind"]),
            price_per_kg=float(row["price_per_kg"]),
            element_content=ec,
        )
    return PriceSnapshot(
        date=snap_date, currency=currency, materials=materials, source=source
    )
```

- [ ] **Step 2: Внутри `tab_design` добавить expander прайса**

Найти в `app/frontend/app.py` секцию `with tab_design:` — это вкладка «🎯 Дизайн сплава». Над кнопкой запуска дизайна (обычно `st.button("Запустить дизайн")`) вставить:

```python
    st.divider()
    with st.expander("💰 Прайс материалов", expanded=True):
        if "price_snapshot" not in st.session_state:
            st.session_state["price_snapshot"] = seed_snapshot()

        snap: PriceSnapshot = st.session_state["price_snapshot"]

        cols = st.columns([2, 1, 1, 1])
        use_cost = cols[0].checkbox(
            "Учитывать стоимость в оптимизации", value=True, key="use_cost"
        )
        cols[1].metric("Валюта", snap.currency)
        cols[2].metric("Дата", snap.date.isoformat())
        cost_mode = cols[3].radio(
            "Режим cost", ["full", "incremental"],
            horizontal=False, key="cost_mode"
        )

        uploaded = st.file_uploader("⬆ Загрузить YAML-прайс", type=["yaml", "yml"])
        if uploaded is not None:
            tmp_path = Path("/tmp") / uploaded.name
            tmp_path.write_bytes(uploaded.read())
            try:
                st.session_state["price_snapshot"] = load_snapshot(tmp_path)
                st.success(f"Загружено: {uploaded.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Не удалось загрузить: {e}")

        df_editor = _snapshot_to_editor_df(snap)
        edited = st.data_editor(
            df_editor, num_rows="dynamic", key="price_editor",
            use_container_width=True,
            column_config={
                "id": "ID",
                "kind": st.column_config.SelectboxColumn(
                    "kind", options=["base", "ferroalloy", "pure"]
                ),
                "price_per_kg": st.column_config.NumberColumn("₽/кг", min_value=0.0),
                "element_content": "element_content (Mn=0.80;Fe=0.20)",
            },
        )

        # Persist edits back into snapshot immediately so they're used on run.
        try:
            st.session_state["price_snapshot"] = _editor_df_to_snapshot(
                edited, snap.date, snap.currency, source="manual"
            )
        except Exception as e:
            st.error(f"Ошибка парсинга прайса: {e}")

        # Download button
        snap_yaml = yaml.safe_dump({
            "date": st.session_state["price_snapshot"].date.isoformat(),
            "currency": st.session_state["price_snapshot"].currency,
            "source": "manual",
            "materials": {
                mid: {
                    "kind": m.kind,
                    "price_per_kg": m.price_per_kg,
                    "element_content": dict(m.element_content),
                }
                for mid, m in st.session_state["price_snapshot"].materials.items()
            },
        }, sort_keys=False, allow_unicode=True)
        st.download_button(
            "💾 Скачать текущий прайс как YAML",
            data=snap_yaml, file_name=f"prices_{snap.date.isoformat()}.yaml",
        )
```

- [ ] **Step 3: Запустить Streamlit локально и проверить**

Run: `PYTHONPATH=. .venv/bin/streamlit run app/frontend/app.py --server.port 8501 --server.headless true`

Открыть http://localhost:8501, вкладка «🎯 Дизайн сплава», раскрыть expander «💰 Прайс материалов». Убедиться:
- таблица из 11 строк (scrap + 8 ферросплавов + Cu + Al);
- чекбокс «Учитывать стоимость» активен;
- radio «full / incremental» работает;
- download-кнопка выгружает YAML.

Остановить: `lsof -ti:8501 | xargs kill`

- [ ] **Step 4: Commit**

```bash
git add app/frontend/app.py
git commit -m "feat(ui): price editor expander with data_editor, upload/download, mode toggle"
```

---

## Task 11: Streamlit UI — pre-check handler + передача snapshot в design

**Files:**
- Modify: `app/frontend/app.py`

- [ ] **Step 1: Найти кнопку запуска дизайна и обернуть вызов**

В том же `tab_design`, найти `if st.button("Запустить дизайн"):` (или аналогичный). Заменить блок `run_inverse_design(...)` на:

```python
    if st.button("🎯 Запустить дизайн", type="primary"):
        from app.backend.inverse_designer import run_inverse_design

        snapshot = (
            st.session_state.get("price_snapshot")
            if st.session_state.get("use_cost", True) else None
        )
        mode = st.session_state.get("cost_mode", "full")

        with st.spinner("NSGA-II ищет Pareto-оптимальные кандидаты..."):
            try:
                result = run_inverse_design(
                    model_version=selected_model,
                    targets={"yield_strength_mpa": {"min": yt_min, "max": yt_max}},
                    hard_constraints={
                        "cev_iiw": {"max": cev_max},
                        "pcm": {"max": pcm_max},
                    },
                    population_size=pop_size,
                    n_generations=n_gen,
                    price_snapshot=snapshot,
                    cost_mode=mode,
                )
            except PriceSnapshotIncomplete as e:
                st.error(
                    f"❌ В прайсе нет цен для: **{', '.join(e.missing)}**. "
                    f"Добавьте строки в таблицу выше и повторите запуск."
                )
                st.stop()

        st.session_state["last_design_result"] = result
```

(Точные имена переменных `yt_min`, `yt_max`, `cev_max`, `pcm_max`, `pop_size`, `n_gen`, `selected_model` — из существующего кода вкладки; не менять.)

- [ ] **Step 2: Проверить в UI**

Запустить Streamlit (см. Task 10, Step 3). Убедиться:
- при `use_cost=True` и корректном прайсе — дизайн запускается и результаты кешируются в `st.session_state["last_design_result"]`;
- удалить строку `FeNb-65` в редакторе → запустить дизайн → появляется красный баннер «Нет цен для: Nb»;
- снять галочку «Учитывать стоимость» → дизайн проходит через legacy-путь без ошибки.

- [ ] **Step 3: Commit**

```bash
git add app/frontend/app.py
git commit -m "feat(ui): pass price_snapshot to run_inverse_design + PriceSnapshotIncomplete handler"
```

---

## Task 12: Streamlit UI — Pareto plot + breakdown table

**Files:**
- Modify: `app/frontend/app.py`

- [ ] **Step 1: Добавить рендеринг результатов — Pareto plot**

После блока запуска (Task 11), в том же `tab_design`, заменить/добавить секцию результатов:

```python
    result = st.session_state.get("last_design_result")
    if result and result["pareto_candidates"]:
        st.subheader(f"Pareto front ({result['n_candidates']} кандидатов)")

        import altair as alt

        df_pareto = pd.DataFrame([{
            "idx": c["idx"],
            "sigma_t": c["predicted"]["mean"],
            "ci_half": c["predicted"]["ci_half_width"],
            "cost": (c["cost"]["total_per_ton"] if c.get("cost")
                     else c["objectives"]["alloying_cost"]),
            "ood": "OOD" if c["predicted"]["ood_flag"] else "ok",
        } for c in result["pareto_candidates"]])

        chart = (
            alt.Chart(df_pareto)
            .mark_circle(size=140)
            .encode(
                x=alt.X("sigma_t:Q", title="σт, МПа"),
                y=alt.Y("cost:Q",
                        title=f"Стоимость, {result['cost_currency']}/т"),
                color=alt.Color("ood:N",
                                scale=alt.Scale(domain=["ok", "OOD"],
                                                range=["#2ecc71", "#e67e22"])),
                tooltip=["idx", "sigma_t", "ci_half", "cost", "ood"],
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
```

- [ ] **Step 2: Добавить breakdown в per-candidate expander**

В существующий блок `for i, c in enumerate(result["pareto_candidates"][:5], 1):` — внутри `st.expander(...)` после существующих секций добавить:

```python
            if c.get("cost"):
                cb = c["cost"]
                st.markdown(
                    f"**💰 Себестоимость:** "
                    f"{cb['total_per_ton']:,.0f} {cb['currency']}/т "
                    f"({cb['total_per_ton']/1000:,.2f} {cb['currency']}/кг, "
                    f"{cb['mode']})"
                )
                df_bd = pd.DataFrame(cb["contributions"])
                if not df_bd.empty:
                    df_bd["share_%"] = (
                        df_bd["contribution_per_ton"] / cb["total_per_ton"] * 100
                    ).round(1)
                    df_bd = df_bd[[
                        "material_id",
                        "mass_kg_per_ton_steel",
                        "price_per_kg",
                        "contribution_per_ton",
                        "share_%",
                    ]]
                    df_bd.columns = [
                        "Материал", "Масса, кг/т", "Цена, ₽/кг",
                        f"Вклад, {cb['currency']}/т", "Доля, %",
                    ]
                    st.dataframe(df_bd, use_container_width=True, hide_index=True)
                    st.download_button(
                        f"📋 Экспорт breakdown #{c['idx']} в CSV",
                        data=df_bd.to_csv(index=False).encode("utf-8"),
                        file_name=f"breakdown_candidate_{c['idx']}.csv",
                        key=f"dl_bd_{c['idx']}",
                    )
```

- [ ] **Step 3: Проверить в UI**

Запустить Streamlit, обучить модель (вкладка «🤖 Обучение модели»), вернуться на «🎯 Дизайн сплава», запустить дизайн с `use_cost=True`. Убедиться:
- появляется Pareto plot с точками (≥ 5);
- в каждом кандидатном expander видна секция «💰 Себестоимость: NN ₽/т» с таблицей breakdown;
- кнопка «📋 Экспорт breakdown в CSV» скачивает корректный файл;
- при снятой галочке cost — Pareto plot всё ещё рисуется (по legacy objective), но breakdown-таблицы нет.

- [ ] **Step 4: Commit**

```bash
git add app/frontend/app.py
git commit -m "feat(ui): Pareto plot σт×cost + per-candidate cost breakdown with CSV export"
```

---

## Task 13: Финальная верификация

**Files:**
- None (verification only)

- [ ] **Step 1: Все unit-тесты**

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/ -v`
Expected: `16 passed, 1 passed` (включая integration-тест, который обучает модель).

- [ ] **Step 2: Smoke test обе ветки**

Run: `PYTHONPATH=. .venv/bin/python scripts/smoke_test.py`
Expected: SMOKE TEST PASSED, в STEP 5b Top candidate cost появляется.

- [ ] **Step 3: Manual UI verification**

Run: `PYTHONPATH=. .venv/bin/streamlit run app/frontend/app.py --server.headless true`

В браузере (http://localhost:8501) пройти:
1. Вкладка «🤖 Обучение модели» → обучить (Optuna trials = 10 для скорости).
2. Вкладка «🎯 Дизайн сплава»:
   - expander «💰 Прайс материалов» раскрыт;
   - отредактировать цену FeNb-65 с 3600 → 5400 (+50%), нажать дизайн;
   - убедиться: в Pareto plot точки с высоким Nb сдвинулись вправо (вверх по cost);
   - Верификация Critic: вкладка «📚 История» содержит запись с tag `cost_optimization`.
3. Удалить строку FeNb-65 в редакторе → запустить дизайн → красный баннер «Нет цен для: Nb».
4. Снять галочку «Учитывать стоимость» → запустить → работает по legacy-пути.

Остановить Streamlit: `lsof -ti:8501 | xargs kill`

- [ ] **Step 4: Final commit сообщение проекта и tag**

```bash
git log --oneline | head -20
```

Убедиться, что история коммитов чистая (13 commits поверх initial). Если всё ок:

```bash
git tag -a v0.2-cost-optimization -m "MVP v0.2: ferroalloy-based cost optimization"
```

---

## Self-Review (выполнено автором плана)

1. **Spec coverage:**
   - Data model → Tasks 1, 4
   - Cost function → Tasks 2, 3
   - YAML snapshot I/O → Task 4
   - Snapshot validation → Task 5
   - Pattern Library C01–C04 → Task 6
   - inverse_designer integration + Decision Log → Task 7
   - engine.py orchestration → Task 8
   - .gitignore / CLAUDE.md / smoke_test → Task 9
   - UI price editor → Task 10
   - UI pre-check handler → Task 11
   - UI Pareto plot + breakdown → Task 12
   - Acceptance verification → Task 13
   Все секции spec покрыты.

2. **Placeholder scan:** ни одного «TBD», «TODO», «similar to Task N»; каждый шаг содержит явный код, команду или явную проверку.

3. **Type consistency:**
   - `PriceSnapshot.date: date` — используется `date.fromisoformat` и `date.today()` везде.
   - `CostBreakdown.total_per_ton` — одно имя во всех задачах (1, 4, 7, 12).
   - `CostContribution.mass_kg_per_ton_steel` — одно имя в Task 1 и в UI breakdown (Task 12).
   - `price_snapshot: PriceSnapshot | None` — параметр с этим именем в inverse_designer (Task 7), engine (Task 8), UI (Tasks 10, 11).
   - `FERROALLOY_PREFERENCE` — одна таблица в cost_model.py (Task 1), используется в compute_cost (Task 2).
   - `_FERROALLOY_RANGES` в patterns.py — отдельная копия для проверки C02 (Task 6); риск дрейфа двух таблиц минимальный, обе основаны на spec §4.4.

4. **Testability:** каждая задача включает failing-test → implementation → verification run, кроме чистых рефакторинг-задач (8, 9, 10–12), которые проверяются manual UI и существующим smoke-тестом.
