"""Tests for Pattern Library DX04-DX07 (slag-aware deox защитные паттерны).

PR 5 scope:
- DX04 (HIGH): slag_aware_calculation=True требует slag_state с mass+feo.
- DX05 (MEDIUM): T_сушки гранул выше method.t_drying_max_c → H-pickup.
- DX06 (MEDIUM): user_override_eta_al вне method.eta_al_range ± 0.05.
- DX07 (HIGH): method.carrier_gas='N2' + target_n_ppm < 50.

Тесты вызывают pattern check-функции напрямую с минимальными ctx-словарями.
"""
from __future__ import annotations

import pytest

from app.backend.slag_aware_deox import (
    AdditionMethod,
    SlagState,
    load_addition_methods,
)
from pattern_library.patterns import (
    Phase,
    Severity,
    _check_dx04_slag_aware_requires_slag_state,
    _check_dx05_drying_temp_too_high,
    _check_dx06_eta_override_outside_range,
    _check_dx07_n2_carrier_for_low_n_target,
    PATTERNS,
    run_all_patterns,
)


@pytest.fixture(autouse=True)
def _clear_loader_cache():
    """Каждый тест начинается с чистого lru_cache YAML-каталога."""
    load_addition_methods.cache_clear()
    yield
    load_addition_methods.cache_clear()


# ---------------------------------------------------------------------------
# DX04 — slag-aware требует slag_state
# ---------------------------------------------------------------------------


def test_dx04_blocks_when_slag_missing():
    """slag_aware_calculation=True + slag_state=None → triggered=True (HIGH)."""
    ctx = {"slag_aware_calculation": True, "slag_state": None}
    result = _check_dx04_slag_aware_requires_slag_state(ctx)
    assert result.triggered is True
    assert "slag" in result.message.lower() or "шлак" in result.message.lower()


def test_dx04_blocks_when_mass_kg_is_none():
    """slag_state присутствует, но mass_kg=None → triggered."""
    # SlagState — frozen с float — нельзя создать с mass_kg=None.
    # Имитируем "dict-like" state с None-полем.
    ctx = {
        "slag_aware_calculation": True,
        "slag_state": {"mass_kg": None, "feo_pct": 18.0},
    }
    result = _check_dx04_slag_aware_requires_slag_state(ctx)
    assert result.triggered is True


def test_dx04_blocks_when_feo_pct_is_none():
    """slag_state с feo_pct=None → triggered."""
    ctx = {
        "slag_aware_calculation": True,
        "slag_state": {"mass_kg": 2200.0, "feo_pct": None},
    }
    result = _check_dx04_slag_aware_requires_slag_state(ctx)
    assert result.triggered is True


def test_dx04_passes_when_slag_present():
    """SlagState с mass=2200, feo=20 → PASS."""
    slag = SlagState(mass_kg=2200.0, feo_pct=20.0)
    ctx = {"slag_aware_calculation": True, "slag_state": slag}
    result = _check_dx04_slag_aware_requires_slag_state(ctx)
    assert result.triggered is False


def test_dx04_inactive_when_flag_off():
    """slag_aware_calculation=False (или отсутствует) → check молчит."""
    ctx = {"slag_aware_calculation": False, "slag_state": None}
    assert _check_dx04_slag_aware_requires_slag_state(ctx).triggered is False
    # Также не падает на пустом ctx
    assert _check_dx04_slag_aware_requires_slag_state({}).triggered is False


# ---------------------------------------------------------------------------
# DX05 — drying temp above method limit
# ---------------------------------------------------------------------------


def test_dx05_warns_when_drying_above_200():
    """granule_water_quenched (t_drying_max_c=200), t_drying_c=220 → triggered."""
    methods = load_addition_methods()
    method = methods["granule_water_quenched"]
    ctx = {"method": method, "t_drying_c": 220.0}
    result = _check_dx05_drying_temp_too_high(ctx)
    assert result.triggered is True
    assert "H-pickup" in result.message or "сушки" in result.message.lower()
    assert result.details["t_drying_c"] == 220.0
    assert result.details["t_drying_max_c"] == 200


def test_dx05_passes_when_no_drying_info():
    """t_drying_c=None → не должно false-positive."""
    methods = load_addition_methods()
    method = methods["granule_water_quenched"]
    ctx = {"method": method, "t_drying_c": None}
    result = _check_dx05_drying_temp_too_high(ctx)
    assert result.triggered is False


def test_dx05_passes_when_method_has_no_drying_limit():
    """ingot (без t_drying_max_c в YAML) — check молчит даже если t_drying_c задан."""
    methods = load_addition_methods()
    method = methods["ingot"]
    ctx = {"method": method, "t_drying_c": 500.0}
    result = _check_dx05_drying_temp_too_high(ctx)
    assert result.triggered is False


def test_dx05_passes_within_limit():
    """t_drying_c=150 < 200 для granule_water_quenched → PASS."""
    methods = load_addition_methods()
    method = methods["granule_water_quenched"]
    ctx = {"method": method, "t_drying_c": 150.0}
    result = _check_dx05_drying_temp_too_high(ctx)
    assert result.triggered is False


# ---------------------------------------------------------------------------
# DX06 — eta override outside literature range
# ---------------------------------------------------------------------------


def test_dx06_warns_eta_override_outside_range():
    """asis_shot (range 0.75-0.90), override=0.50 → triggered (MEDIUM)."""
    methods = load_addition_methods()
    method = methods["asis_shot"]
    ctx = {"method": method, "user_override_eta_al": 0.50}
    result = _check_dx06_eta_override_outside_range(ctx)
    assert result.triggered is True
    assert "η_Al" in result.message or "0.50" in result.message


def test_dx06_passes_eta_within_range():
    """asis_shot, override=0.82 (внутри 0.75-0.90) → PASS."""
    methods = load_addition_methods()
    method = methods["asis_shot"]
    ctx = {"method": method, "user_override_eta_al": 0.82}
    result = _check_dx06_eta_override_outside_range(ctx)
    assert result.triggered is False


def test_dx06_passes_eta_within_tolerance():
    """asis_shot, override=0.73 (на 0.02 ниже lo=0.75, в tolerance ±0.05) → PASS."""
    methods = load_addition_methods()
    method = methods["asis_shot"]
    ctx = {"method": method, "user_override_eta_al": 0.73}
    result = _check_dx06_eta_override_outside_range(ctx)
    assert result.triggered is False


def test_dx06_inactive_when_no_override():
    """user_override_eta_al=None → check молчит."""
    methods = load_addition_methods()
    method = methods["asis_shot"]
    ctx = {"method": method, "user_override_eta_al": None}
    result = _check_dx06_eta_override_outside_range(ctx)
    assert result.triggered is False


# ---------------------------------------------------------------------------
# DX07 — N2 carrier for low [N] target
# ---------------------------------------------------------------------------


def _make_n2_method() -> AdditionMethod:
    """Тестовый AdditionMethod с carrier_gas='N2' — в default YAML нет такого."""
    return AdditionMethod(
        id="asis_shot_n2",
        name="ASIS N2 (test)",
        eta_al_typical=0.82,
        eta_al_range=(0.75, 0.90),
        premium_eur_per_kg=0.20,
        carrier_gas="N2",
        surface_m2_per_kg=1.7,
        notes="test",
        raw={"carrier_gas": "N2"},
    )


def test_dx07_blocks_n2_for_low_n_target():
    """N2-carrier + target_n_ppm=30 < 50 → triggered (HIGH)."""
    method = _make_n2_method()
    ctx = {"method": method, "target_n_ppm": 30.0}
    result = _check_dx07_n2_carrier_for_low_n_target(ctx)
    assert result.triggered is True
    assert "N" in result.message  # N₂ упоминается
    assert result.details["carrier_gas"] == "N2"
    assert result.details["target_n_ppm"] == 30.0


def test_dx07_passes_n2_for_high_n_target():
    """N2-carrier + target_n_ppm=100 → PASS (>=50)."""
    method = _make_n2_method()
    ctx = {"method": method, "target_n_ppm": 100.0}
    result = _check_dx07_n2_carrier_for_low_n_target(ctx)
    assert result.triggered is False


def test_dx07_passes_when_target_n_none():
    """target_n_ppm=None (no [N] constraint) → check молчит."""
    method = _make_n2_method()
    ctx = {"method": method, "target_n_ppm": None}
    result = _check_dx07_n2_carrier_for_low_n_target(ctx)
    assert result.triggered is False


def test_dx07_passes_for_ar_carrier():
    """asis_shot (carrier_gas='Ar') + target_n=10 → PASS."""
    methods = load_addition_methods()
    method = methods["asis_shot"]  # Ar
    ctx = {"method": method, "target_n_ppm": 10.0}
    result = _check_dx07_n2_carrier_for_low_n_target(ctx)
    assert result.triggered is False


def test_dx07_passes_for_no_carrier_gas():
    """ingot (carrier_gas=None) + target_n=10 → PASS."""
    methods = load_addition_methods()
    method = methods["ingot"]  # None
    ctx = {"method": method, "target_n_ppm": 10.0}
    result = _check_dx07_n2_carrier_for_low_n_target(ctx)
    assert result.triggered is False


# ---------------------------------------------------------------------------
# Integration: PATTERNS list registration + run_all_patterns
# ---------------------------------------------------------------------------


def test_dx04_07_registered_in_patterns_list():
    """Все четыре новых паттерна присутствуют в PATTERNS и относятся к DEOXIDATION."""
    ids = {p.id: p for p in PATTERNS}
    for code in ("DX04", "DX05", "DX06", "DX07"):
        assert code in ids, f"{code} missing from PATTERNS list"
        assert ids[code].phase == Phase.DEOXIDATION

    # Severity per design-doc
    assert ids["DX04"].severity == Severity.HIGH
    assert ids["DX05"].severity == Severity.MEDIUM
    assert ids["DX06"].severity == Severity.MEDIUM
    assert ids["DX07"].severity == Severity.HIGH


def test_run_all_patterns_dx04_07_phase_filter():
    """run_all_patterns(phase=DEOXIDATION) с ctx, триггерящим DX04 и DX07,
    возвращает оба warnings."""
    method = _make_n2_method()
    ctx = {
        "slag_aware_calculation": True,
        "slag_state": None,
        "method": method,
        "target_n_ppm": 30.0,
    }
    warnings = run_all_patterns(ctx, phase=Phase.DEOXIDATION)
    triggered_ids = {w["pattern_id"] for w in warnings}
    assert "DX04" in triggered_ids
    assert "DX07" in triggered_ids
    # DX05/DX06 не должны сработать на этом ctx (нет t_drying_c, нет override)
    assert "DX05" not in triggered_ids
    assert "DX06" not in triggered_ids
