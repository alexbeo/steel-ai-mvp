"""Tests для EtaAlCalibrator (PR 5)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.backend import heat_records as _heat_records
from app.backend.eta_al_calibration import (
    EtaAlCalibrator,
    _clip_for_logit,
    _compute_posterior_logit,
    _invlogit,
    _logit,
)
from app.backend.heat_records import HeatRecord, bulk_insert_heats


def _make_heat(plant: str, method: str, eta: float | None) -> HeatRecord:
    # Outcome fields (o_a_after_ppm, al_added_kg) must be set so list_heats(has_outcome=True)
    # returns the record. eta_al_effective is what calibrator actually consumes.
    return HeatRecord(
        source="synthetic", plant_id=plant,
        steel_mass_ton=100.0, o_a_initial_ppm=500.0,
        o_a_after_ppm=5.0, al_added_kg=100.0,
        method_id=method, eta_al_effective=eta,
    )


@pytest.fixture
def db_setup(tmp_path, monkeypatch):
    db = tmp_path / "heats.db"
    monkeypatch.setattr(_heat_records, "DEFAULT_DB_PATH", db)
    return db


@pytest.fixture
def calibrator(tmp_path, db_setup):
    return EtaAlCalibrator(
        db_path=db_setup,
        calibrations_dir=tmp_path / "calibs",
    )


def test_logit_invlogit_roundtrip():
    for p in [0.1, 0.3, 0.5, 0.7, 0.9, 0.05, 0.95]:
        assert abs(_invlogit(_logit(p)) - p) < 1e-9


def test_clip_for_logit_at_boundaries():
    v, w = _clip_for_logit(0.0)
    assert v > 0 and w is True
    v, w = _clip_for_logit(1.0)
    assert v < 1 and w is True
    v, w = _clip_for_logit(0.5)
    assert v == 0.5 and w is False


def test_prior_from_catalog_asis_shot(calibrator):
    mu0, sigma0 = calibrator.get_prior_from_catalog("asis_shot")
    # asis_shot: typical=0.82, range=[0.75, 0.90]
    assert abs(mu0 - _logit(0.82)) < 1e-6
    assert sigma0 > 0


def test_prior_from_catalog_unknown_method(calibrator):
    with pytest.raises(ValueError):
        calibrator.get_prior_from_catalog("UNKNOWN_METHOD")


def test_posterior_with_no_data_returns_prior():
    mu_post, sigma_post = _compute_posterior_logit(
        mu0=1.5, sigma0=0.3, sample_mean_logit=None, n_obs=0, sigma_likelihood=0.5
    )
    assert mu_post == 1.5
    assert sigma_post == 0.3


def test_posterior_with_many_data_converges_to_mean():
    mu_post, sigma_post = _compute_posterior_logit(
        mu0=1.5, sigma0=0.3, sample_mean_logit=2.0, n_obs=10000, sigma_likelihood=0.5
    )
    # n→∞, posterior → sample mean
    assert abs(mu_post - 2.0) < 0.001
    assert sigma_post < 0.01


def test_posterior_shifts_toward_observations():
    mu_post, _ = _compute_posterior_logit(
        mu0=1.5, sigma0=0.3, sample_mean_logit=2.5, n_obs=50, sigma_likelihood=0.5
    )
    # должен сдвинуться от 1.5 в сторону 2.5
    assert 1.5 < mu_post < 2.5


def test_calibrate_plant_below_threshold_no_yaml(calibrator, tmp_path):
    heats = [_make_heat("PLANT_A", "asis_shot", 0.85) for _ in range(10)]
    bulk_insert_heats(heats, db_path=calibrator.db_path)
    result = calibrator.calibrate_plant("PLANT_A")
    assert result.yaml_written is False
    # All methods skipped (only 10 heats for asis_shot, 0 for others)
    asis = next(p for p in result.calibrations if p.method_id == "asis_shot")
    assert asis.skipped_reason is not None
    assert "n_heats=10" in asis.skipped_reason


def test_calibrate_plant_above_threshold_writes_yaml(calibrator, tmp_path):
    heats = [_make_heat("PLANT_A", "asis_shot", 0.85) for _ in range(30)]
    bulk_insert_heats(heats, db_path=calibrator.db_path)
    result = calibrator.calibrate_plant("PLANT_A")
    assert result.yaml_written is True
    yaml_path = Path(result.yaml_path)
    assert yaml_path.exists()
    import yaml as _yaml
    data = _yaml.safe_load(yaml_path.read_text())
    assert data["plant_id"] == "PLANT_A"
    assert "asis_shot" in data["calibrations"]
    asis = data["calibrations"]["asis_shot"]
    assert asis["n_heats_used"] == 30
    assert asis["posterior_eta_mean"] > 0.80


def test_calibrate_plant_mixed_methods(calibrator):
    heats = [_make_heat("PLANT_A", "asis_shot", 0.85) for _ in range(40)]
    heats += [_make_heat("PLANT_A", "cored_wire_feal30", 0.88) for _ in range(5)]
    bulk_insert_heats(heats, db_path=calibrator.db_path)
    result = calibrator.calibrate_plant("PLANT_A")
    asis = next(p for p in result.calibrations if p.method_id == "asis_shot")
    cored = next(p for p in result.calibrations if p.method_id == "cored_wire_feal30")
    assert asis.skipped_reason is None
    assert asis.n_heats_used == 40
    assert cored.skipped_reason is not None
    assert cored.n_heats_used == 5


def test_calibrate_plant_filters_none_eta(calibrator):
    # 20 with eta + 20 with eta=None but valid outcome — only 20 should be counted
    heats = [_make_heat("PLANT_A", "asis_shot", 0.85) for _ in range(20)]
    heats += [_make_heat("PLANT_A", "asis_shot", None) for _ in range(20)]
    bulk_insert_heats(heats, db_path=calibrator.db_path)
    result = calibrator.calibrate_plant("PLANT_A")
    asis = next(p for p in result.calibrations if p.method_id == "asis_shot")
    # Only 20 valid (eta non-null) → below threshold (30)
    assert asis.n_heats_used == 20
    assert asis.skipped_reason is not None


def test_calibrate_all_plants_iterates(calibrator):
    for plant in ["P1", "P2", "P3"]:
        bulk_insert_heats(
            [_make_heat(plant, "asis_shot", 0.82) for _ in range(35)],
            db_path=calibrator.db_path,
        )
    results = calibrator.calibrate_all_plants()
    assert len(results) == 3
    plant_ids = {r.plant_id for r in results}
    assert plant_ids == {"P1", "P2", "P3"}


def test_get_posterior_reads_existing_yaml(calibrator):
    bulk_insert_heats(
        [_make_heat("PLANT_A", "asis_shot", 0.85) for _ in range(30)],
        db_path=calibrator.db_path,
    )
    calibrator.calibrate_plant("PLANT_A")
    # New instance — без recompute
    cal2 = EtaAlCalibrator(
        db_path=calibrator.db_path,
        calibrations_dir=calibrator.calibrations_dir,
    )
    post = cal2.get_posterior("PLANT_A", "asis_shot")
    assert post is not None
    assert post.n_heats_used == 30


def test_get_posterior_none_for_uncalibrated_plant(calibrator):
    assert calibrator.get_posterior("MISSING_PLANT", "asis_shot") is None


def test_get_posterior_none_for_skipped_method(calibrator):
    bulk_insert_heats(
        [_make_heat("PLANT_A", "asis_shot", 0.85) for _ in range(30)],
        db_path=calibrator.db_path,
    )
    bulk_insert_heats(
        [_make_heat("PLANT_A", "ingot", 0.60) for _ in range(5)],
        db_path=calibrator.db_path,
    )
    calibrator.calibrate_plant("PLANT_A")
    assert calibrator.get_posterior("PLANT_A", "asis_shot") is not None
    assert calibrator.get_posterior("PLANT_A", "ingot") is None  # skipped


def test_atomic_write_cleanup_on_error(calibrator, tmp_path, monkeypatch):
    bulk_insert_heats(
        [_make_heat("PLANT_A", "asis_shot", 0.85) for _ in range(30)],
        db_path=calibrator.db_path,
    )
    # Force replace to raise — patch in the eta_al_calibration module namespace
    import app.backend.eta_al_calibration as mod

    def bad_replace(*a, **kw):
        raise OSError("simulated")

    monkeypatch.setattr(mod.os, "replace", bad_replace)
    with pytest.raises(OSError):
        calibrator.calibrate_plant("PLANT_A")
    # tmp files должны быть очищены
    if calibrator.calibrations_dir.exists():
        tmp_files = list(calibrator.calibrations_dir.glob("*.tmp"))
        assert tmp_files == []


def test_decision_log_entry_created(calibrator, monkeypatch):
    calls = []
    import decision_log.logger as dl
    monkeypatch.setattr(dl, "log_decision", lambda *a, **kw: calls.append(kw) or 1)
    bulk_insert_heats(
        [_make_heat("PLANT_A", "asis_shot", 0.85) for _ in range(30)],
        db_path=calibrator.db_path,
    )
    calibrator.calibrate_plant("PLANT_A")
    assert len(calls) == 1
    assert "eta_al_calibration" in calls[0]["tags"]


def test_eta_posterior_within_bounds(calibrator):
    bulk_insert_heats(
        [_make_heat("PLANT_A", "asis_shot", 0.95) for _ in range(50)],
        db_path=calibrator.db_path,
    )
    result = calibrator.calibrate_plant("PLANT_A")
    asis = next(p for p in result.calibrations if p.method_id == "asis_shot")
    assert 0.0 < asis.posterior_eta_mean < 1.0
    assert asis.posterior_eta_q05 < asis.posterior_eta_mean < asis.posterior_eta_q95
