"""Tests for /api/deox/heats* endpoints (PR 2 of ASIS-deox calibration).

The router wraps ``app.backend.heat_records`` (already covered by the
backend's own 23 unit tests) and adds:

  - request validation (Pydantic field bounds mirror HeatRecord)
  - keyset pagination via ``before_id``
  - distinct-plant aggregate query
  - Decision Log integration ONLY on PATCH (outcome confirmation)

We isolate state via two fixtures:

  - ``client`` monkey-patches ``DEFAULT_DB_PATH`` to ``tmp_path/heats.db``
    so every test sees an empty DB. The router resolves the path at
    call-time, so the patch takes effect without re-importing.
  - ``decision_log_spy`` replaces the router's ``log_decision`` symbol
    with a list-collector so we can assert the audit-trail contract
    (POST: no write; PATCH: 1 write per call) without touching the
    decision_log SQLite.
"""
from __future__ import annotations

import functools
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routers import heats as heats_router
from app.backend import heat_records as _heat_records


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture()
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """TestClient with isolated heats DB.

    ``DEFAULT_DB_PATH`` is captured as a default kwarg at function-definition
    time in ``app.backend.heat_records``, so a plain ``monkeypatch.setattr``
    on the constant has no effect on already-bound defaults. We instead
    rebind each helper symbol the router calls to a ``functools.partial``
    that injects ``db_path`` explicitly. The router resolves these symbols
    via ``app.api.routers.heats``, so patching there is sufficient — the
    backend module itself stays unchanged.

    Also patches ``_heat_records.DEFAULT_DB_PATH`` itself because the
    ``GET /plants`` endpoint reads it directly (raw-SQL aggregate).
    """
    db_path = tmp_path / "heats.db"
    # Constant used by GET /plants (raw-SQL helper inside the router).
    monkeypatch.setattr(_heat_records, "DEFAULT_DB_PATH", db_path)
    # Rebind the helpers imported into the router. functools.partial keeps
    # the original signatures intact — kwargs only.
    for name in (
        "insert_heat",
        "get_heat_by_id",
        "list_heats",
        "update_heat_outcome",
        "delete_heat",
        "count_heats",
    ):
        original = getattr(heats_router, name)
        monkeypatch.setattr(
            heats_router, name, functools.partial(original, db_path=db_path)
        )
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def decision_log_spy(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace router's ``log_decision`` with a list-collector.

    Returns the list — tests assert ``len(calls)`` and inspect
    ``calls[0]["tags"]`` / ``calls[0]["context"]``. Returning ``1`` as
    the synthetic decision_id mirrors the real function's contract
    (positive int on success).
    """
    calls: list[dict] = []

    def _spy(**kwargs):
        calls.append(kwargs)
        return 1

    monkeypatch.setattr(heats_router, "log_decision", _spy)
    return calls


# ──────────────────────────────────────────────────────────────────────
# Payload helpers
# ──────────────────────────────────────────────────────────────────────


def _minimal_payload(**overrides) -> dict:
    """Smallest body that passes Pydantic + DB CHECK constraints."""
    base = {
        "source": "manual",
        "plant_id": "ASIS_BOF",
        "steel_mass_ton": 100.0,
        "o_a_initial_ppm": 500.0,
    }
    base.update(overrides)
    return base


# ──────────────────────────────────────────────────────────────────────
# POST /api/deox/heats — create
# ──────────────────────────────────────────────────────────────────────


def test_create_heat_minimal(client: TestClient) -> None:
    """Minimal valid body → 201 + echoed record + ISO timestamp."""
    r = client.post("/api/deox/heats", json=_minimal_payload())
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"] >= 1
    assert "created_at" in body
    # Created timestamp should be ISO 8601 with timezone.
    assert "T" in body["created_at"]
    assert body["heat"]["plant_id"] == "ASIS_BOF"
    assert body["heat"]["source"] == "manual"
    assert body["heat"]["steel_mass_ton"] == 100.0
    assert body["heat"]["o_a_initial_ppm"] == 500.0


def test_create_heat_full_payload_roundtrip(client: TestClient) -> None:
    """Full body with all optional fields → roundtrip preserves them."""
    payload = _minimal_payload(
        heat_id="H-12345",
        steel_class_id="pipe_hsla",
        t_tap_c=1650.0,
        t_lf_arrival_c=1620.0,
        t_al_addition_c=1605.0,
        al_added_kg=454.0,
        al_residual_pct=0.022,
        slag_mass_kg=2200,
        slag_feo_pct=18,
        slag_mno_pct=2.5,
        slag_sio2_pct=15.0,
        slag_cao_pct=42,
        slag_mgo_pct=8.0,
        slag_al2o3_pct=12.0,
        c_pct=0.06,
        mn_pct=1.5,
        si_pct=0.25,
        s_pct=0.005,
        p_pct=0.012,
        method_id="asis_shot",
        addition_timing="in_stream",
        carrier_gas="Ar",
        co_deox_fesi_kg=120.0,
        dt_to_al_min=8.5,
        t_drying_c=180.0,
        ar_stir_nm3=12.5,
        vacuum_treatment="VD",
        refractory_heat_count=42,
        eta_al_effective=0.82,
        quality_flag="accept",
        notes="тестовая плавка PR 2",
        extras={"h_ppm": 1.5, "комментарий": "роудтрип юникода"},
    )
    r = client.post("/api/deox/heats", json=payload)
    assert r.status_code == 201, r.text
    got = client.get(f"/api/deox/heats/{r.json()['id']}").json()["heat"]
    assert got["slag_cao_pct"] == 42
    assert got["addition_timing"] == "in_stream"
    assert got["carrier_gas"] == "Ar"
    assert got["vacuum_treatment"] == "VD"
    assert got["eta_al_effective"] == 0.82
    assert got["notes"] == "тестовая плавка PR 2"
    assert got["extras"]["комментарий"] == "роудтрип юникода"
    assert got["extras"]["h_ppm"] == 1.5


def test_create_heat_rejects_out_of_bounds_mass(client: TestClient) -> None:
    """``steel_mass_ton > 500`` → 422 (Pydantic field bound)."""
    r = client.post(
        "/api/deox/heats", json=_minimal_payload(steel_mass_ton=600)
    )
    assert r.status_code == 422


def test_create_heat_rejects_missing_required_plant(client: TestClient) -> None:
    """``plant_id`` is required → 422 when omitted."""
    r = client.post(
        "/api/deox/heats",
        json={
            "source": "manual",
            "steel_mass_ton": 100,
            "o_a_initial_ppm": 500,
        },
    )
    assert r.status_code == 422


def test_create_heat_rejects_invalid_source_enum(client: TestClient) -> None:
    """``source`` is Literal — typoed value → 422."""
    r = client.post(
        "/api/deox/heats", json=_minimal_payload(source="manual_typo")
    )
    assert r.status_code == 422


def test_create_heat_does_not_write_decision_log(
    client: TestClient, decision_log_spy: list[dict]
) -> None:
    """POST is data ingestion — must NOT touch Decision Log."""
    r = client.post("/api/deox/heats", json=_minimal_payload())
    assert r.status_code == 201
    assert decision_log_spy == []


# ──────────────────────────────────────────────────────────────────────
# GET /api/deox/heats/{id} — fetch single
# ──────────────────────────────────────────────────────────────────────


def test_get_heat_by_id_404_when_missing(client: TestClient) -> None:
    assert client.get("/api/deox/heats/9999").status_code == 404


def test_get_heat_by_id_returns_record(client: TestClient) -> None:
    pk = client.post(
        "/api/deox/heats", json=_minimal_payload()
    ).json()["id"]
    r = client.get(f"/api/deox/heats/{pk}")
    assert r.status_code == 200
    assert r.json()["heat"]["id"] == pk


# ──────────────────────────────────────────────────────────────────────
# GET /api/deox/heats — list with filters + pagination
# ──────────────────────────────────────────────────────────────────────


def test_list_heats_default_returns_all(client: TestClient) -> None:
    for i in range(5):
        client.post(
            "/api/deox/heats",
            json=_minimal_payload(plant_id=f"P{i % 2}"),
        )
    body = client.get("/api/deox/heats").json()
    assert body["count"] == 5
    assert body["total"] == 5
    # Sorted id DESC — most recent first.
    ids = [item["id"] for item in body["items"]]
    assert ids == sorted(ids, reverse=True)


def test_list_heats_filter_by_plant(client: TestClient) -> None:
    for plant in ("A", "A", "B"):
        client.post(
            "/api/deox/heats", json=_minimal_payload(plant_id=plant)
        )
    body = client.get("/api/deox/heats?plant_id=A").json()
    assert body["count"] == 2
    assert all(it["plant_id"] == "A" for it in body["items"])
    # total respects the same filter
    assert body["total"] == 2


def test_list_heats_filter_has_outcome(client: TestClient) -> None:
    """has_outcome requires both o_a_after_ppm AND al_added_kg (PR 1 contract)."""
    # Heat 1 — full outcome (after PATCH-equivalent: POST with both fields).
    client.post(
        "/api/deox/heats",
        json=_minimal_payload(
            plant_id="P1", o_a_after_ppm=5.0, al_added_kg=400.0
        ),
    )
    # Heat 2 — in-progress (no outcome).
    client.post("/api/deox/heats", json=_minimal_payload(plant_id="P2"))
    finished = client.get("/api/deox/heats?has_outcome=true").json()
    assert finished["count"] == 1
    assert finished["items"][0]["plant_id"] == "P1"
    in_progress = client.get("/api/deox/heats?has_outcome=false").json()
    assert in_progress["count"] == 1
    assert in_progress["items"][0]["plant_id"] == "P2"


def test_list_heats_pagination_keyset(client: TestClient) -> None:
    """Keyset paging via ``before_id`` walks the table without skips."""
    for _ in range(15):
        client.post("/api/deox/heats", json=_minimal_payload())
    page1 = client.get("/api/deox/heats?limit=10").json()
    assert page1["count"] == 10
    assert page1["next_before_id"] == page1["items"][-1]["id"]
    page2 = client.get(
        f"/api/deox/heats?limit=10&before_id={page1['next_before_id']}"
    ).json()
    assert page2["count"] == 5
    assert page2["next_before_id"] is None
    # Pages must not overlap.
    p1_ids = {it["id"] for it in page1["items"]}
    p2_ids = {it["id"] for it in page2["items"]}
    assert p1_ids.isdisjoint(p2_ids)


def test_list_heats_limit_validation(client: TestClient) -> None:
    """``limit`` must be 1-1000 — Query() bounds."""
    assert client.get("/api/deox/heats?limit=0").status_code == 422
    assert client.get("/api/deox/heats?limit=1001").status_code == 422


# ──────────────────────────────────────────────────────────────────────
# GET /api/deox/heats/plants — distinct aggregate
# ──────────────────────────────────────────────────────────────────────


def test_list_plants_distinct_counts(client: TestClient) -> None:
    for plant in ("A", "A", "B", "C", "C", "C"):
        client.post(
            "/api/deox/heats", json=_minimal_payload(plant_id=plant)
        )
    body = client.get("/api/deox/heats/plants").json()
    assert body["total"] == 3
    plants = {it["plant_id"]: it["count"] for it in body["items"]}
    assert plants == {"A": 2, "B": 1, "C": 3}
    # Sorted by count DESC so the heaviest-traffic plant lands first.
    assert body["items"][0]["plant_id"] == "C"


def test_list_plants_empty(client: TestClient) -> None:
    """No heats → empty list (not error)."""
    body = client.get("/api/deox/heats/plants").json()
    assert body == {"items": [], "total": 0}


# ──────────────────────────────────────────────────────────────────────
# PATCH /api/deox/heats/{id} — outcome update + Decision Log
# ──────────────────────────────────────────────────────────────────────


def test_patch_heat_outcome_updates_and_writes_decision_log(
    client: TestClient, decision_log_spy: list[dict]
) -> None:
    pk = client.post(
        "/api/deox/heats", json=_minimal_payload()
    ).json()["id"]
    r = client.patch(
        f"/api/deox/heats/{pk}",
        json={"o_a_after_ppm": 5.0, "eta_al_effective": 0.82},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["heat"]["o_a_after_ppm"] == 5.0
    assert body["heat"]["eta_al_effective"] == 0.82
    assert body["decision_id"] == 1
    # Exactly one Decision Log call.
    assert len(decision_log_spy) == 1
    call = decision_log_spy[0]
    assert "heat_outcome_update" in call["tags"]
    assert any(t.startswith("plant:") for t in call["tags"])
    # Context carries before/after for forensic reconstruction.
    assert call["context"]["heat_id"] == pk
    assert "before" in call["context"]
    assert "after" in call["context"]
    assert call["context"]["after"] == {
        "o_a_after_ppm": 5.0,
        "eta_al_effective": 0.82,
    }


def test_patch_heat_404_when_missing(client: TestClient) -> None:
    r = client.patch("/api/deox/heats/9999", json={"o_a_after_ppm": 5.0})
    assert r.status_code == 404


def test_patch_heat_empty_body_400(client: TestClient) -> None:
    pk = client.post(
        "/api/deox/heats", json=_minimal_payload()
    ).json()["id"]
    r = client.patch(f"/api/deox/heats/{pk}", json={})
    assert r.status_code == 400


def test_patch_heat_rejects_out_of_bounds(client: TestClient) -> None:
    """eta_al_effective > 1.5 → 422."""
    pk = client.post(
        "/api/deox/heats", json=_minimal_payload()
    ).json()["id"]
    r = client.patch(
        f"/api/deox/heats/{pk}", json={"eta_al_effective": 2.5}
    )
    assert r.status_code == 422


# ──────────────────────────────────────────────────────────────────────
# DELETE /api/deox/heats/{id}
# ──────────────────────────────────────────────────────────────────────


def test_delete_heat_existing(client: TestClient) -> None:
    pk = client.post(
        "/api/deox/heats", json=_minimal_payload()
    ).json()["id"]
    r = client.delete(f"/api/deox/heats/{pk}")
    assert r.status_code == 200
    assert r.json() == {"deleted": True, "id": pk}
    # Row really gone.
    assert client.get(f"/api/deox/heats/{pk}").status_code == 404


def test_delete_heat_missing_404(client: TestClient) -> None:
    assert client.delete("/api/deox/heats/9999").status_code == 404


def test_delete_heat_does_not_write_decision_log(
    client: TestClient, decision_log_spy: list[dict]
) -> None:
    """DELETE is non-audited (operators use it to undo manual-entry typos)."""
    pk = client.post(
        "/api/deox/heats", json=_minimal_payload()
    ).json()["id"]
    r = client.delete(f"/api/deox/heats/{pk}")
    assert r.status_code == 200
    assert decision_log_spy == []
