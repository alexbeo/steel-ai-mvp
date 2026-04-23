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
    priced_ids = {c.material_id for c in breakdown.contributions}
    assert "Fe-C" not in priced_ids and "FeP" not in priced_ids


def test_compute_cost_unknown_element_raises():
    snapshot = _full_seed_rub()
    with pytest.raises(ValueError, match="Нет маппинга"):
        compute_cost({"w_pct": 0.5}, snapshot, mode="full")
