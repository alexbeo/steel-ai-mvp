"""Tests для EtaAlPredictor (PR 9)."""
from __future__ import annotations

import pytest

from app.backend.eta_al_calibration import (
    EtaAlCalibrator,
    _logit,
)
from app.backend.eta_al_predictor import (
    EtaAlPredictor,
    _load_plant_offsets,
    _sigmoid,
)


def test_sigmoid_at_zero_is_half():
    assert abs(_sigmoid(0) - 0.5) < 1e-9


def test_no_calibrator_no_model_uses_literature():
    """No plant calibration, no ML model → literature fallback."""
    predictor = EtaAlPredictor(
        calibrator=None,
        model_version="NONEXISTENT_VERSION",
    )
    pred = predictor.predict_eta_al(
        plant_id="PLANT_A",
        method_id="asis_shot",
        features={"c_pct": 0.06, "mn_pct": 1.5},
    )
    assert pred.source == "literature_fallback"
    # asis_shot eta_typical = 0.82
    assert abs(pred.eta_mean - 0.82) < 0.01


def test_calibrator_only_returns_plant_only(tmp_path, monkeypatch):
    """Posterior exists, no model → plant_only."""
    from app.backend import heat_records as hr
    from app.backend.heat_records import HeatRecord, bulk_insert_heats

    db = tmp_path / "h.db"
    monkeypatch.setattr(hr, "DEFAULT_DB_PATH", db)
    heats = [
        HeatRecord(
            source="synthetic",
            plant_id="P_X",
            steel_mass_ton=100,
            o_a_initial_ppm=500,
            o_a_after_ppm=5.0,
            al_added_kg=200.0,
            method_id="asis_shot",
            eta_al_effective=0.88,
        )
        for _ in range(50)
    ]
    bulk_insert_heats(heats, db_path=db)
    calibrator = EtaAlCalibrator(
        db_path=db,
        calibrations_dir=tmp_path / "calibs",
    )
    calibrator.calibrate_plant("P_X")
    predictor = EtaAlPredictor(
        calibrator=calibrator,
        model_version="NONEXISTENT",  # forces no ML
    )
    pred = predictor.predict_eta_al(
        plant_id="P_X",
        method_id="asis_shot",
        features=None,
    )
    assert pred.source == "plant_only"
    assert pred.plant_weight == 1.0
    assert pred.n_plant_heats == 50


def test_mixed_balanced_at_n_30(tmp_path, monkeypatch):
    """N=30 → w=0.5 (sigmoid midpoint)."""
    from app.backend import heat_records as hr
    from app.backend.heat_records import HeatRecord, bulk_insert_heats

    db = tmp_path / "h.db"
    monkeypatch.setattr(hr, "DEFAULT_DB_PATH", db)
    bulk_insert_heats(
        [
            HeatRecord(
                source="synthetic",
                plant_id="P_X",
                steel_mass_ton=100,
                o_a_initial_ppm=500,
                o_a_after_ppm=5.0,
                al_added_kg=200.0,
                method_id="asis_shot",
                eta_al_effective=0.85,
            )
            for _ in range(30)
        ],
        db_path=db,
    )
    calibrator = EtaAlCalibrator(
        db_path=db,
        calibrations_dir=tmp_path / "calibs",
    )
    calibrator.calibrate_plant("P_X")

    fake_predictor = EtaAlPredictor(
        calibrator=calibrator,
        model_version="NONEXISTENT",
    )
    # Monkeypatch _predict_global to return synthetic global η=0.75, σ_logit=0.2
    fake_predictor._predict_global = lambda f: (_logit(0.75), 0.2, "fake_v1")
    pred = fake_predictor.predict_eta_al(
        plant_id="P_X",
        method_id="asis_shot",
        features={"x": 1},
    )
    assert pred.source == "mixed"
    assert abs(pred.plant_weight - 0.5) < 0.1  # N=30 близко к sigmoid midpoint


def test_lewis_variance_includes_disagreement():
    """σ_mix² должен включать disagreement term."""
    mu_p, sigma_p = -1.0, 0.5
    mu_g, sigma_g = +1.0, 0.5
    w = 0.5
    var_individual_max = max(sigma_p**2, sigma_g**2)  # 0.25
    var_mix = (
        w * sigma_p**2
        + (1 - w) * sigma_g**2
        + w * (1 - w) * (mu_p - mu_g) ** 2
    )
    # var_mix = 0.25 + 0.25 * 4 = 1.25
    assert var_mix > var_individual_max
    assert abs(var_mix - 1.25) < 1e-9


def test_unknown_plant_zero_offset_warning():
    predictor = EtaAlPredictor(
        calibrator=None,
        model_version="NONEXISTENT",
    )
    pred = predictor.predict_eta_al(
        plant_id="UNKNOWN_PLANT",
        method_id="asis_shot",
        features=None,
    )
    warnings_str = " ".join(pred.metadata.get("warnings", []))
    assert "unknown plant" in warnings_str.lower()


def test_unknown_method_raises():
    predictor = EtaAlPredictor(
        calibrator=None,
        model_version="NONEXISTENT",
    )
    with pytest.raises(ValueError, match="Unknown method_id"):
        predictor.predict_eta_al(
            plant_id="PLANT_A",
            method_id="invalid",
            features=None,
        )


def test_plant_offsets_yaml_loads():
    """plant_offsets.yaml correctly loads."""
    offsets = _load_plant_offsets()
    assert "PLANT_A" in offsets
    assert offsets["PLANT_A"] == 0.02


def test_predictor_integration_with_compute_al_demand(tmp_path, monkeypatch):
    """compute_al_demand_slag_aware с predictor overrides literature η."""
    from app.backend import heat_records as hr
    from app.backend.heat_records import HeatRecord, bulk_insert_heats
    from app.backend.slag_aware_deox import compute_al_demand_slag_aware

    db = tmp_path / "h.db"
    monkeypatch.setattr(hr, "DEFAULT_DB_PATH", db)
    # 50 плавок с η=0.92 (выше literature 0.82) → posterior shifted upward
    bulk_insert_heats(
        [
            HeatRecord(
                source="synthetic",
                plant_id="P_HIGH",
                steel_mass_ton=100,
                o_a_initial_ppm=500,
                o_a_after_ppm=5.0,
                al_added_kg=200.0,
                method_id="asis_shot",
                eta_al_effective=0.92,
            )
            for _ in range(50)
        ],
        db_path=db,
    )
    calibrator = EtaAlCalibrator(
        db_path=db,
        calibrations_dir=tmp_path / "calibs",
    )
    calibrator.calibrate_plant("P_HIGH")
    predictor = EtaAlPredictor(
        calibrator=calibrator,
        model_version="NONEXISTENT",
    )

    # Без predictor
    res_lit = compute_al_demand_slag_aware(
        steel_mass_ton=100.0,
        o_a_initial_ppm=500.0,
        target_o_a_ppm=5.0,
        target_al_pct=0.02,
        method="asis_shot",
    )
    # С predictor
    res_pred = compute_al_demand_slag_aware(
        steel_mass_ton=100.0,
        o_a_initial_ppm=500.0,
        target_o_a_ppm=5.0,
        target_al_pct=0.02,
        method="asis_shot",
        eta_al_predictor=predictor,
        plant_id="P_HIGH",
    )
    # η выше → меньше Al нужно → al_pure_kg меньше
    assert res_pred.al_pure_kg < res_lit.al_pure_kg
    assert "predicted_eta_metadata" in res_pred.inputs
    assert res_pred.inputs["predicted_eta_metadata"]["source"] == "plant_only"


def test_predictor_requires_plant_id():
    from app.backend.slag_aware_deox import compute_al_demand_slag_aware

    predictor = EtaAlPredictor(model_version="NONEXISTENT")
    with pytest.raises(ValueError, match="plant_id"):
        compute_al_demand_slag_aware(
            steel_mass_ton=100.0,
            o_a_initial_ppm=500.0,
            target_o_a_ppm=5.0,
            target_al_pct=0.02,
            method="asis_shot",
            eta_al_predictor=predictor,
            # missing plant_id
        )


def test_predictor_collision_with_override_warns():
    from app.backend.slag_aware_deox import compute_al_demand_slag_aware

    predictor = EtaAlPredictor(model_version="NONEXISTENT")
    res = compute_al_demand_slag_aware(
        steel_mass_ton=100.0,
        o_a_initial_ppm=500.0,
        target_o_a_ppm=5.0,
        target_al_pct=0.02,
        method="asis_shot",
        eta_al_override=0.95,
        eta_al_predictor=predictor,
        plant_id="PLANT_A",
    )
    assert any("predictor" in w.lower() for w in res.warnings)


def test_predict_global_error_surfaced_in_metadata(monkeypatch):
    """Model loaded but bad features → metadata['global_error'] surfaced (PR 10).

    No calibrator + global predict returns None → literature_fallback, but
    because the model bundle *is* present we attach the failure reason to
    metadata so the UI can distinguish "no model" from "model present but
    couldn't predict for these features".
    """
    predictor = EtaAlPredictor(calibrator=None, model_version="NONEXISTENT")
    # Pretend a model bundle is loaded so the global_error branch fires.
    predictor._model_loaded = True
    predictor._model_bundle = object()  # truthy sentinel

    def bad_predict(features):
        predictor._last_global_error = "missing column X"
        return None

    monkeypatch.setattr(predictor, "_predict_global", bad_predict)

    pred = predictor.predict_eta_al(
        plant_id="PLANT_A",
        method_id="asis_shot",
        features={"c_pct": 0.06},
    )
    # No calibration + global failed → literature fallback.
    assert pred.source == "literature_fallback"
    assert pred.metadata.get("global_error") == "missing column X"
