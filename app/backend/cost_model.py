"""
Cost model for HSLA steel inverse design.

Ferroalloy-based pricing: each alloying element maps to a preferred
ferroalloy (FeNb-65, FeMn-80, ...). Compute cost per ton of steel
given a composition (in %) and a PriceSnapshot.

Naming convention inside this module:
- ``Material.element_content`` keys are **capitalized element symbols**
  (``"Mn"``, ``"Nb"``, ``"Fe"``) carrying **mass fractions in [0, 1]**.
- Composition dicts consumed by ``compute_cost`` use the rest of the
  project's ``{element}_pct`` lowercase keys (``"mn_pct"``, ``"nb_pct"``)
  carrying values **in percent** (e.g. ``1.5`` means 1.5%). ``compute_cost``
  translates between the two.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
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
