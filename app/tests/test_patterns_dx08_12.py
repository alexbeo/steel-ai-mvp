"""Tests for Pattern Library DX08-DX12 (η_Al calibration quality gates, PR 13).

R-005 scope:
- DX08 (MEDIUM): predictor η_Al вне method.eta_al_range±0.05 под literature_fallback.
- DX09 (MEDIUM): plant_n_heats_for_method < min_heats_threshold (default 30).
- DX10 (MEDIUM): current_slag_basicity вне historical_basicity_range (ships DORMANT).
- DX11 (HIGH, TRAINING): steel_class=='deox_calibration' AND coverage_90_ci < 0.85.
- DX12 (MEDIUM): |posterior_mu - global_mu| / posterior_sigma > 2 (logit-space).

Тесты вызывают run_all_patterns с минимальными ctx-словарями (как в
test_patterns_dx04_07.py — формат warnings: pattern_id/severity/message/details).
"""
from __future__ import annotations

from pattern_library.patterns import (
    PATTERNS,
    Phase,
    Severity,
    run_all_patterns,
)


def _run_deox(ctx: dict) -> list[dict]:
    return run_all_patterns(ctx, phase=Phase.DEOXIDATION)


def _run_training(ctx: dict) -> list[dict]:
    return run_all_patterns(ctx, phase=Phase.TRAINING)


def _ids(warnings: list[dict]) -> set[str]:
    return {w["pattern_id"] for w in warnings}


# ---------------------------------------------------------------------------
# DX08 — η out of literature calibration under literature_fallback
# ---------------------------------------------------------------------------


def test_dx08_triggers_eta_out_of_range_literature():
    ctx = {
        "eta_al_used": 0.98,
        "method_eta_range": [0.75, 0.90],
        "eta_calibration_source": "literature_fallback",
    }
    assert "DX08" in _ids(_run_deox(ctx))


def test_dx08_no_trigger_within_range():
    ctx = {
        "eta_al_used": 0.82,
        "method_eta_range": [0.75, 0.90],
        "eta_calibration_source": "literature_fallback",
    }
    assert "DX08" not in _ids(_run_deox(ctx))


def test_dx08_no_trigger_within_tolerance():
    """0.93 = hi(0.90) + 0.03 — в пределах ±0.05 толеранса → PASS."""
    ctx = {
        "eta_al_used": 0.93,
        "method_eta_range": [0.75, 0.90],
        "eta_calibration_source": "literature_fallback",
    }
    assert "DX08" not in _ids(_run_deox(ctx))


def test_dx08_no_trigger_when_plant_calibrated():
    """plant калибровка оправдывает deviation — DX08 молчит на не-literature источнике."""
    ctx = {
        "eta_al_used": 0.95,
        "method_eta_range": [0.75, 0.90],
        "eta_calibration_source": "plant_only",
    }
    assert "DX08" not in _ids(_run_deox(ctx))


def test_dx08_no_trigger_when_mixed_source():
    ctx = {
        "eta_al_used": 0.95,
        "method_eta_range": [0.75, 0.90],
        "eta_calibration_source": "mixed",
    }
    assert "DX08" not in _ids(_run_deox(ctx))


# ---------------------------------------------------------------------------
# DX09 — insufficient plant data for Bayesian calibration
# ---------------------------------------------------------------------------


def test_dx09_triggers_below_threshold():
    ctx = {"plant_n_heats_for_method": 10, "min_heats_threshold": 30}
    assert "DX09" in _ids(_run_deox(ctx))


def test_dx09_no_trigger_above_threshold():
    ctx = {"plant_n_heats_for_method": 50, "min_heats_threshold": 30}
    assert "DX09" not in _ids(_run_deox(ctx))


def test_dx09_uses_default_threshold_30():
    """min_heats_threshold отсутствует → default 30; n=20 < 30 → triggered."""
    ctx = {"plant_n_heats_for_method": 20}
    assert "DX09" in _ids(_run_deox(ctx))


# ---------------------------------------------------------------------------
# DX10 — slag basicity extrapolation (DORMANT)
# ---------------------------------------------------------------------------


def test_dx10_dormant_without_basicity():
    ctx: dict = {}
    assert "DX10" not in _ids(_run_deox(ctx))


def test_dx10_triggers_outside_range():
    ctx = {"current_slag_basicity": 3.5, "historical_basicity_range": [1.5, 2.5]}
    assert "DX10" in _ids(_run_deox(ctx))


def test_dx10_no_trigger_within_range():
    ctx = {"current_slag_basicity": 2.0, "historical_basicity_range": [1.5, 2.5]}
    assert "DX10" not in _ids(_run_deox(ctx))


# ---------------------------------------------------------------------------
# DX11 — η_Al model conformal coverage low (TRAINING, HIGH)
# ---------------------------------------------------------------------------


def test_dx11_triggers_low_coverage():
    ctx = {"steel_class": "deox_calibration", "coverage_90_ci": 0.78}
    assert "DX11" in _ids(_run_training(ctx))


def test_dx11_no_trigger_good_coverage():
    ctx = {"steel_class": "deox_calibration", "coverage_90_ci": 0.89}
    assert "DX11" not in _ids(_run_training(ctx))


def test_dx11_no_trigger_wrong_class():
    """DX11 fires только для deox_calibration (M02 owns HSLA/fatigue)."""
    ctx = {"steel_class": "pipe_hsla", "coverage_90_ci": 0.78}
    assert "DX11" not in _ids(_run_training(ctx))


def test_dx11_is_high_severity():
    ctx = {"steel_class": "deox_calibration", "coverage_90_ci": 0.70}
    warnings = _run_training(ctx)
    dx11 = next(w for w in warnings if w["pattern_id"] == "DX11")
    assert dx11["severity"] == "HIGH"


def test_dx11_not_in_deox_phase():
    """DX11 принадлежит TRAINING — не должен сработать в DEOXIDATION фазе."""
    ctx = {"steel_class": "deox_calibration", "coverage_90_ci": 0.50}
    assert "DX11" not in _ids(_run_deox(ctx))


# ---------------------------------------------------------------------------
# DX12 — plant-posterior vs global ML conflict
# ---------------------------------------------------------------------------


def test_dx12_triggers_large_conflict():
    # z = |2.0 - 1.0| / 0.3 = 3.33 > 2
    ctx = {
        "posterior_eta_logit_mu": 2.0,
        "global_eta_logit_mu": 1.0,
        "posterior_logit_sigma": 0.3,
    }
    assert "DX12" in _ids(_run_deox(ctx))


def test_dx12_no_trigger_small_conflict():
    # z = |1.6 - 1.5| / 0.3 = 0.33 < 2
    ctx = {
        "posterior_eta_logit_mu": 1.6,
        "global_eta_logit_mu": 1.5,
        "posterior_logit_sigma": 0.3,
    }
    assert "DX12" not in _ids(_run_deox(ctx))


def test_dx12_no_trigger_missing_global():
    """global_eta_logit_mu отсутствует (partial-dormant) → молчит."""
    ctx = {"posterior_eta_logit_mu": 2.0, "posterior_logit_sigma": 0.3}
    assert "DX12" not in _ids(_run_deox(ctx))


def test_dx12_no_trigger_zero_sigma():
    """posterior_logit_sigma <= 0 → защита от деления на ноль → молчит."""
    ctx = {
        "posterior_eta_logit_mu": 2.0,
        "global_eta_logit_mu": 1.0,
        "posterior_logit_sigma": 0.0,
    }
    assert "DX12" not in _ids(_run_deox(ctx))


# ---------------------------------------------------------------------------
# Graceful degradation + details shape
# ---------------------------------------------------------------------------


def test_empty_ctx_no_dx08_12_triggers():
    ids = _ids(_run_deox({}))
    assert not ({"DX08", "DX09", "DX10", "DX12"} & ids)


def test_empty_training_ctx_no_dx11():
    assert "DX11" not in _ids(_run_training({}))


def test_details_format_small():
    """details dict < 5 fields, no DataFrames / heavy objects."""
    ctx = {
        "eta_al_used": 0.98,
        "method_eta_range": [0.75, 0.90],
        "eta_calibration_source": "literature_fallback",
    }
    warnings = _run_deox(ctx)
    dx08 = next(w for w in warnings if w["pattern_id"] == "DX08")
    assert len(dx08.get("details", {})) < 5


# ---------------------------------------------------------------------------
# Integration: PATTERNS list registration
# ---------------------------------------------------------------------------


def test_dx08_12_registered_in_patterns_list():
    ids = {p.id: p for p in PATTERNS}
    for code in ("DX08", "DX09", "DX10", "DX12"):
        assert code in ids, f"{code} missing from PATTERNS list"
        assert ids[code].phase == Phase.DEOXIDATION
    assert "DX11" in ids
    assert ids["DX11"].phase == Phase.TRAINING

    assert ids["DX08"].severity == Severity.MEDIUM
    assert ids["DX09"].severity == Severity.MEDIUM
    assert ids["DX10"].severity == Severity.MEDIUM
    assert ids["DX11"].severity == Severity.HIGH
    assert ids["DX12"].severity == Severity.MEDIUM
