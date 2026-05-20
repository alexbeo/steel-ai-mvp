"""Tests для /api/deox/shadow/* (Phase 2 S4)."""
import functools

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.backend import heat_records as hr


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with an isolated heats DB.

    ``DEFAULT_DB_PATH`` is captured as a bound default kwarg in
    ``app.backend.heat_records``, so patching the constant alone has no effect
    on ``count_heats``/``list_heats``. The shadow endpoint imports those helpers
    lazily from that module, so we rebind them to ``functools.partial`` with the
    tmp ``db_path`` injected (mirrors the test_api_heats.py fixture).
    """
    db_path = tmp_path / "heats.db"
    monkeypatch.setattr(hr, "DEFAULT_DB_PATH", db_path)
    for name in ("count_heats", "list_heats", "bulk_insert_heats"):
        original = getattr(hr, name)
        monkeypatch.setattr(hr, name, functools.partial(original, db_path=db_path))
    return TestClient(app)


def test_shadow_run_synthetic(client):
    r = client.post(
        "/api/deox/shadow/run",
        json={"source": "synthetic", "n_synthetic_heats": 100, "save_report": False},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["is_synthetic"] is True
    assert "hypothesis_met" in data["stats"]
    assert "median_delta_pct" in data["stats"]


def test_shadow_run_heats_db_empty(client):
    r = client.post(
        "/api/deox/shadow/run",
        json={"source": "heats_db", "save_report": False},
    )
    assert r.status_code == 200
    assert r.json()["n_compared"] == 0


def test_shadow_run_limit_exceeded(client):
    from app.backend.heat_records import HeatRecord

    heats = [
        HeatRecord(
            source="synthetic", plant_id="P1", steel_mass_ton=100,
            o_a_initial_ppm=500, o_a_after_ppm=5, al_added_kg=120,
            method_id="asis_shot", eta_al_effective=0.85,
        )
        for _ in range(15)
    ]
    # hr.bulk_insert_heats is partial-bound to the tmp db_path by the fixture.
    hr.bulk_insert_heats(heats)
    r = client.post(
        "/api/deox/shadow/run",
        json={"source": "heats_db", "limit": 5, "save_report": False},
    )
    assert r.status_code == 400


def test_shadow_report_no_report(client, monkeypatch, tmp_path):
    import app.backend.shadow_reporter as sr

    monkeypatch.setattr(sr, "REPORTS_DIR", tmp_path / "noreports")
    r = client.get("/api/deox/shadow/report")
    assert r.status_code == 404


def test_shadow_report_returns_latest(client, monkeypatch, tmp_path):
    import json
    import time

    import app.backend.shadow_reporter as sr

    rdir = tmp_path / "reports"
    rdir.mkdir()
    monkeypatch.setattr(sr, "REPORTS_DIR", rdir)
    (rdir / "shadow_20260101_000000.json").write_text(json.dumps({"stats": {"n": 1}}))
    time.sleep(0.01)
    (rdir / "shadow_20260102_000000.json").write_text(json.dumps({"stats": {"n": 2}}))
    r = client.get("/api/deox/shadow/report")
    assert r.status_code == 200
    # latest by mtime
    assert r.json()["stats"]["n"] == 2
