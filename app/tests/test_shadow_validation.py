"""Tests для shadow comparison engine (Phase 2 S1)."""
from __future__ import annotations

import math

from app.backend.heat_records import HeatRecord
from app.backend.shadow_validation import (
    RESIDUAL_AL_SPEC_DEFAULT,
    ShadowComparison,
    _residual_spec_for_class,
    run_shadow_comparison,
)


class _StubPredictor:
    """Duck-typed predictor returning fixed η via logit posterior.

    compute_al_demand_slag_aware only calls predict_eta_al(plant_id, method_id,
    features) — no other attrs needed."""

    def __init__(self, eta=0.82):
        self._mu = math.log(eta / (1 - eta))

    def predict_eta_al(self, plant_id, method_id, features=None):
        from app.backend.eta_al_calibration import _invlogit
        from app.backend.eta_al_predictor import EtaPrediction

        return EtaPrediction(
            eta_mean=_invlogit(self._mu),
            eta_logit_mu=self._mu,
            eta_logit_sigma=0.1,
            plant_weight=1.0,
            global_weight=0.0,
            source="plant_only",
            n_plant_heats=50,
            metadata={},
        )


def _heat(**kw) -> HeatRecord:
    defaults = dict(
        source="synthetic",
        plant_id="P1",
        steel_mass_ton=100.0,
        o_a_initial_ppm=500.0,
        o_a_after_ppm=5.0,
        al_added_kg=120.0,
        method_id="asis_shot",
        al_residual_pct=0.025,
    )
    defaults.update(kw)
    return HeatRecord(**defaults)


def test_shadow_comparison_basic():
    heats = [_heat() for _ in range(3)]
    res = run_shadow_comparison(heats, _StubPredictor())
    assert len(res) == 3
    for c in res:
        assert isinstance(c, ShadowComparison)
        assert c.skip_reason is None
        assert c.al_ai_p50_kg is not None
        assert c.delta_pct is not None
        assert c.eta_ai_source == "plant_only"


def test_quality_gate_pass_when_in_spec():
    heats = [_heat(al_residual_pct=0.025)]
    res = run_shadow_comparison(heats, _StubPredictor())
    assert res[0].residual_in_spec is True


def test_quality_gate_fail_residual_out_of_spec():
    heats = [_heat(al_residual_pct=0.060)]  # > 0.040 max
    res = run_shadow_comparison(heats, _StubPredictor())
    assert res[0].residual_in_spec is False
    assert res[0].quality_pass is False


def test_skip_heat_without_al_actual():
    # NB: HeatRecord allows al_added_kg None; has_outcome filter обычно отсекает,
    # но engine защищается явной проверкой.
    h = _heat(al_added_kg=None, o_a_after_ppm=5.0)
    res = run_shadow_comparison([h], _StubPredictor())
    assert res[0].skip_reason == "no_outcome"


def test_skip_zero_al_actual():
    res = run_shadow_comparison([_heat(al_added_kg=0.0)], _StubPredictor())
    assert res[0].skip_reason == "zero_al_actual"


def test_skip_unknown_method():
    res = run_shadow_comparison([_heat(method_id="bogus")], _StubPredictor())
    assert res[0].skip_reason == "unknown_method"


def test_skip_degenerate_o_balance():
    # target o_a >= initial → нечего раскислять, comparison не имеет смысла
    res = run_shadow_comparison(
        [_heat(o_a_initial_ppm=5.0, o_a_after_ppm=5.0)], _StubPredictor()
    )
    assert res[0].skip_reason == "degenerate_o_balance"


def test_delta_pct_calculation():
    res = run_shadow_comparison([_heat(al_added_kg=200.0)], _StubPredictor(eta=0.90))
    c = res[0]
    expected = (c.al_ai_p50_kg - c.al_actual_kg) / c.al_actual_kg * 100
    assert abs(c.delta_pct - expected) < 1e-6


def test_ai_recommends_less_for_overdosed_actual():
    """Завышенный actual (250 kg ≈ 247.5 pure) → AI с η=0.90 даст меньше → delta<0."""
    res = run_shadow_comparison([_heat(al_added_kg=250.0)], _StubPredictor(eta=0.90))
    assert res[0].delta_pct < 0


def test_residual_spec_default():
    assert _residual_spec_for_class(None) == RESIDUAL_AL_SPEC_DEFAULT
    assert _residual_spec_for_class("deox_calibration") == RESIDUAL_AL_SPEC_DEFAULT


def test_residual_none_notes_but_passes_residual():
    res = run_shadow_comparison([_heat(al_residual_pct=None)], _StubPredictor())
    assert res[0].residual_in_spec is True
    assert any("отсутствует" in n for n in res[0].notes)


def test_dose_sufficient_and_quality_pass_nominal():
    res = run_shadow_comparison([_heat()], _StubPredictor())
    c = res[0]
    assert c.al_ai_p90_kg is not None and c.al_ai_p90_kg > 0
    assert c.dose_sufficient is True
    # ai_target_o_a == actual_o_a (A1) → o_a_ok True; residual in spec → pass
    assert c.quality_pass is True
