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
