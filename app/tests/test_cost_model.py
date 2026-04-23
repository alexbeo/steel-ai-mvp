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

    fenb = next(c for c in full.contributions if c.material_id == "FeNb-65")
    assert fenb.mass_kg_per_ton_steel == pytest.approx(10.0)
    assert fenb.contribution_per_ton == pytest.approx(36_000.0)
