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


from app.backend.deoxidation import compute_al_demand  # noqa: E402


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
    assert result.o_a_expected_ppm <= 5.5
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
        steel_mass_ton=180, target_o_a_ppm=100,
        al_purity_pct=100, burn_off_pct=20,
    )
    assert result.al_total_kg == 0.0
    assert any("target" in w.lower() or "цел" in w.lower()
               for w in result.warnings)


def test_compute_al_demand_temperature_out_of_range_warns():
    result = compute_al_demand(
        o_a_initial_ppm=500, temperature_C=1450,
        steel_mass_ton=180, target_o_a_ppm=5,
        al_purity_pct=100, burn_off_pct=20,
    )
    assert any("temperature" in w.lower() or "температур" in w.lower()
               for w in result.warnings)


def test_compute_al_demand_extreme_o_a_warns():
    result = compute_al_demand(
        o_a_initial_ppm=900,
        temperature_C=1620,
        steel_mass_ton=180, target_o_a_ppm=5,
        al_purity_pct=100, burn_off_pct=20,
    )
    assert any("o_a" in w.lower() for w in result.warnings)


from app.backend.deoxidation import compare_all_models, compute_al_quality  # noqa: E402


def test_compute_al_quality_roundtrip_purity_85():
    """Forward with purity=85%, then inverse should recover ~85%."""
    forward = compute_al_demand(
        o_a_initial_ppm=450, temperature_C=1620,
        steel_mass_ton=180, target_o_a_ppm=5,
        al_purity_pct=85, burn_off_pct=20,
    )
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
    # 180 t, 500→100 ppm → ΔO = 72 kg → Al_active ≈ 80.9 kg.
    # With 200 kg Al added and 20% burn-off, expected_active = 160 kg,
    # so effective purity ≈ 50.6% — triggers the <70% warning.
    inverse = compute_al_quality(
        o_a_before_ppm=500,
        o_a_after_ppm=100,
        al_added_kg=200.0,
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
    masses = [r.al_total_kg for r in results]
    spread = (max(masses) - min(masses)) / (sum(masses) / 3.0)
    assert spread < 0.25


def test_steel_class_profile_loads_target_o_activity():
    from app.backend.steel_classes import load_steel_class
    hsla = load_steel_class("pipe_hsla")
    assert hsla.target_o_activity_ppm == 5.0
    qt = load_steel_class("en10083_qt")
    assert qt.target_o_activity_ppm == 15.0


def test_pattern_dx01_triggers_on_extreme_o_a():
    from pattern_library.patterns import Phase, run_all_patterns
    ctx = {"o_a_initial_ppm": 900.0}
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
    ctx = {"effective_purity_pct": 55.0}
    warnings = run_all_patterns(ctx, phase=Phase.DEOXIDATION)
    ids = {w["pattern_id"] for w in warnings}
    assert "DX03" in ids


def test_pattern_dx01_does_not_trigger_in_normal_range():
    from pattern_library.patterns import Phase, run_all_patterns
    ctx = {"o_a_initial_ppm": 450}
    warnings = run_all_patterns(ctx, phase=Phase.DEOXIDATION)
    ids = {w["pattern_id"] for w in warnings}
    assert "DX01" not in ids
