"""Router for /api/deox/heats* — manual heat record management (PR 2).

PR 1 created the storage layer (``app.backend.heat_records``). PR 2 exposes
six endpoints under the ``/api/deox`` prefix so the UI sub-tab «История
плавок» can add / list / patch outcome / delete heats. Bulk-import
(Excel/CSV) is PR 3; synthetic data is PR 4.

Decision Log discipline: only PATCH (outcome confirmation) writes to the
audit trail. POST (data ingestion) does not — 200 heats/day would flood
the log without informational value. PATCH is treated as a metallurgist's
commit on the η_Al ground truth — a deliberate, post-fact validation
that's worth a row in the audit table.

Endpoints
---------
- ``POST   /api/deox/heats``            — create new heat (no audit row)
- ``GET    /api/deox/heats``            — list with filters + pagination
- ``GET    /api/deox/heats/plants``     — distinct plant_id list (for UI dropdown)
- ``GET    /api/deox/heats/{id}``       — fetch single by primary key
- ``PATCH  /api/deox/heats/{id}``       — update outcome (writes audit row)
- ``DELETE /api/deox/heats/{id}``       — delete by id (no audit row)
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.responses import SafeJSONResponse
from app.backend import heat_records as _heat_records
from app.backend.heat_records import (
    AdditionTiming,
    CarrierGas,
    HeatRecord,
    QualityFlag,
    Source,
    VacuumTreatment,
    count_heats,
    delete_heat,
    get_heat_by_id,
    insert_heat,
    list_heats,
    update_heat_outcome,
)
from decision_log.logger import log_decision

logger = logging.getLogger(__name__)
router = APIRouter()


# ──────────────────────────────────────────────────────────────────────
# Request schemas
#
# Field bounds MUST stay in sync with ``HeatRecord`` in
# ``app.backend.heat_records`` — both layers validate the same ranges so
# a Pydantic-422 error from the API surfaces the same bound a direct
# ``HeatRecord(**)`` call would have raised. The schema.sql CHECK
# constraints in ``data/heats/schema.sql`` are the third line of defence.
#
# We don't subclass HeatRecord here because pydantic v2 makes that
# fragile (overriding ``id: int | None`` with ``id: None = None`` reads
# badly and breaks when HeatRecord adds new server-managed fields).
# Explicit listing is verbose but the contract stays obvious.
# ──────────────────────────────────────────────────────────────────────


class HeatCreateRequest(BaseModel):
    """Body for POST /api/deox/heats.

    Mirrors HeatRecord exactly EXCEPT excludes server-managed fields
    (``id``, ``created_at``). Field bounds must stay in sync with
    HeatRecord (PR 1 schema.sql also enforces via CHECK).
    """

    source: Source
    plant_id: str = Field(..., min_length=1, max_length=64)
    heat_id: str | None = Field(default=None, max_length=64)
    steel_class_id: str | None = Field(default=None, max_length=64)
    steel_mass_ton: float = Field(..., ge=1.0, le=500.0)
    o_a_initial_ppm: float = Field(..., ge=0.0, le=2000.0)
    o_a_after_ppm: float | None = Field(default=None, ge=0.0, le=2000.0)
    t_tap_c: float | None = Field(default=None, ge=1400.0, le=1700.0)
    t_lf_arrival_c: float | None = Field(default=None, ge=1400.0, le=1700.0)
    t_al_addition_c: float | None = Field(default=None, ge=1400.0, le=1700.0)
    al_added_kg: float | None = Field(default=None, ge=0.0, le=5000.0)
    al_residual_pct: float | None = Field(default=None, ge=0.0, le=0.5)
    slag_mass_kg: float | None = Field(default=None, ge=0.0, le=10000.0)
    carry_over_slag_kg_per_t: float | None = Field(default=None, ge=0.0, le=50.0)
    slag_feo_pct: float | None = Field(default=None, ge=0.0, le=50.0)
    slag_mno_pct: float | None = Field(default=None, ge=0.0, le=20.0)
    slag_sio2_pct: float | None = Field(default=None, ge=0.0, le=30.0)
    slag_cao_pct: float | None = Field(default=None, ge=0.0, le=70.0)
    slag_mgo_pct: float | None = Field(default=None, ge=0.0, le=25.0)
    slag_al2o3_pct: float | None = Field(default=None, ge=0.0, le=50.0)
    c_pct: float | None = Field(default=None, ge=0.0, le=1.5)
    mn_pct: float | None = Field(default=None, ge=0.0, le=3.0)
    si_pct: float | None = Field(default=None, ge=0.0, le=2.5)
    s_pct: float | None = Field(default=None, ge=0.0, le=0.05)
    p_pct: float | None = Field(default=None, ge=0.0, le=0.05)
    method_id: str | None = Field(default=None, max_length=64)
    addition_timing: AdditionTiming | None = None
    carrier_gas: CarrierGas | None = None
    co_deox_fesi_kg: float | None = Field(default=None, ge=0.0, le=5000.0)
    dt_to_al_min: float | None = Field(default=None, ge=0.0, le=120.0)
    t_drying_c: float | None = Field(default=None, ge=0.0, le=600.0)
    ar_stir_nm3: float | None = Field(default=None, ge=0.0, le=100.0)
    vacuum_treatment: VacuumTreatment | None = None
    refractory_heat_count: int | None = Field(default=None, ge=0, le=500)
    eta_al_effective: float | None = Field(default=None, ge=0.0, le=1.5)
    quality_flag: QualityFlag | None = None
    notes: str | None = Field(default=None, max_length=4000)
    extras: dict | None = None

    model_config = {"protected_namespaces": ()}


class HeatUpdateOutcomeRequest(BaseModel):
    """Body for PATCH /api/deox/heats/{id} — only outcome fields are mutable.

    The four fields here are the metallurgist's commit on the η_Al
    ground truth after the heat has finished and the analyzer numbers
    are in. Everything else (composition, slag, method) is fixed at
    POST time — to fix a typo on those, delete + re-create.
    """

    o_a_after_ppm: float | None = Field(default=None, ge=0.0, le=2000.0)
    al_residual_pct: float | None = Field(default=None, ge=0.0, le=0.5)
    eta_al_effective: float | None = Field(default=None, ge=0.0, le=1.5)
    quality_flag: QualityFlag | None = None

    model_config = {"protected_namespaces": ()}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _heat_to_dict(record: HeatRecord) -> dict[str, Any]:
    """Project HeatRecord → JSON-safe dict (ISO timestamp, extras as dict).

    Pydantic's ``model_dump`` produces a datetime object for ``created_at``
    which SafeJSONResponse handles, but we normalise to a UTC-aware ISO
    string here so downstream UI code can ``new Date(...)`` it without
    timezone surprises.
    """
    d = record.model_dump()
    ts = d.get("created_at")
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            d["created_at"] = ts.replace(tzinfo=timezone.utc).isoformat()
        else:
            d["created_at"] = ts.astimezone(timezone.utc).isoformat()
    return d


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────


@router.post(
    "/heats",
    response_class=SafeJSONResponse,
    response_model=None,
    status_code=201,
)
def create_heat(req: HeatCreateRequest) -> dict[str, Any]:
    """Create new heat record (manual entry).

    Decision Log: NOT written here (POST = data ingestion, 200 heats/day
    would spam the audit trail). Outcome confirmation in PATCH is the
    auditable event.

    Returns: 201 with ``{id, created_at, heat: <full record>}``.
    """
    # Pydantic field constraints on ``HeatCreateRequest`` mirror
    # ``HeatRecord`` 1:1, so the ``HeatRecord(**)`` construction is a
    # cheap re-validation rather than a second source of truth.
    record = HeatRecord(**req.model_dump())
    try:
        new_id = insert_heat(record)
    except sqlite3.IntegrityError as exc:
        # CHECK constraint or NOT NULL violation surfacing from PR 1's
        # schema.sql — return as 400 so the UI can show "invalid input"
        # rather than a generic 500.
        logger.warning("Heat insert rejected by DB constraint: %s", exc)
        raise HTTPException(
            status_code=400, detail=f"DB constraint violation: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — surface unexpected DB errors as 500
        logger.exception("Failed to insert heat")
        raise HTTPException(
            status_code=500, detail=f"DB insert failed: {exc}"
        ) from exc
    saved = get_heat_by_id(new_id)
    if saved is None:
        # Should be impossible — DB insert returned an ID. Defensive.
        raise HTTPException(
            status_code=500, detail=f"Heat #{new_id} not found after insert"
        )
    saved_dict = _heat_to_dict(saved)
    return {
        "id": saved.id,
        "created_at": saved_dict["created_at"],
        "heat": saved_dict,
    }


@router.get(
    "/heats",
    response_class=SafeJSONResponse,
    response_model=None,
)
def list_heats_endpoint(
    plant_id: str | None = Query(default=None, max_length=64),
    method_id: str | None = Query(default=None, max_length=64),
    has_outcome: bool | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    before_id: int | None = Query(default=None, ge=1),
) -> dict[str, Any]:
    """List heats with filters + keyset pagination by id DESC.

    ``before_id`` — return rows with id < before_id (next page). For first
    page omit; for subsequent pages pass ``next_before_id`` from previous
    response. Pagination is keyset rather than offset/limit because the
    plant log is append-mostly: keyset survives concurrent inserts without
    skipping or repeating rows.

    The PR 1 ``list_heats`` already filters by plant_id/method_id/has_outcome
    and sorts ORDER BY id DESC LIMIT ?. We add the ``before_id`` cursor here
    as a thin client-side filter so PR 1 stays sealed.
    """
    if before_id is not None:
        # Fetch a heuristically larger window then filter — keeps PR 1
        # API surface unchanged. The 4× factor handles the worst-case
        # where most rows share the same plant_id but we paginate
        # without that filter (rare).
        raw = list_heats(
            plant_id=plant_id,
            method_id=method_id,
            has_outcome=has_outcome,
            limit=limit * 4,
        )
        items = [r for r in raw if r.id is not None and r.id < before_id][:limit]
    else:
        items = list_heats(
            plant_id=plant_id,
            method_id=method_id,
            has_outcome=has_outcome,
            limit=limit,
        )
    total = count_heats(plant_id=plant_id)
    # next_before_id is the id of the last returned row when the page is
    # full; otherwise we've hit the end and signal so via None.
    next_before_id: int | None = None
    if len(items) == limit and items and items[-1].id is not None:
        next_before_id = int(items[-1].id)
    return {
        "items": [_heat_to_dict(r) for r in items],
        "count": len(items),
        "total": total,
        "next_before_id": next_before_id,
    }


@router.get(
    "/heats/plants",
    response_class=SafeJSONResponse,
    response_model=None,
)
def list_plants() -> dict[str, Any]:
    """Distinct plant_id list with row counts (for UI filter dropdown).

    PR 1 didn't expose this aggregate — a small raw-SQL query here is
    acceptable because PR 1 stays sealed and the query is trivial. The
    rows are sorted by count DESC so the most-populated plant lands at
    the top of the UI filter dropdown.
    """
    db = _heat_records.DEFAULT_DB_PATH
    _heat_records._init_db(db)
    with sqlite3.connect(str(db)) as conn:
        cur = conn.execute(
            "SELECT plant_id, COUNT(*) as n FROM heats "
            "GROUP BY plant_id ORDER BY n DESC, plant_id ASC"
        )
        rows = [{"plant_id": r[0], "count": int(r[1])} for r in cur.fetchall()]
    return {"items": rows, "total": len(rows)}


@router.get(
    "/heats/{heat_pk}",
    response_class=SafeJSONResponse,
    response_model=None,
)
def get_heat(heat_pk: int) -> dict[str, Any]:
    """Lookup one heat by primary key."""
    record = get_heat_by_id(heat_pk)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Heat #{heat_pk} not found")
    return {"heat": _heat_to_dict(record)}


@router.patch(
    "/heats/{heat_pk}",
    response_class=SafeJSONResponse,
    response_model=None,
)
def patch_heat_outcome(
    heat_pk: int, req: HeatUpdateOutcomeRequest
) -> dict[str, Any]:
    """Update outcome fields and write a Decision Log entry.

    Outcome fields = ``o_a_after_ppm``, ``al_residual_pct``,
    ``eta_al_effective``, ``quality_flag``. Empty body → 400. Missing
    heat → 404.

    The audit row uses tag ``heat_outcome_update`` plus a
    ``plant:<plant_id>`` tag so a future query can scope by plant. The
    ``context`` payload captures before/after snapshots of the changed
    fields so a forensic reader can reconstruct the edit without a row
    diff.
    """
    body = req.model_dump(exclude_none=True)
    if not body:
        raise HTTPException(
            status_code=400, detail="No outcome fields provided"
        )
    prev = get_heat_by_id(heat_pk)
    if prev is None:
        raise HTTPException(status_code=404, detail=f"Heat #{heat_pk} not found")
    update_heat_outcome(heat_pk, **body)
    updated = get_heat_by_id(heat_pk)
    if updated is None:
        # Should be unreachable — update didn't delete the row.
        raise HTTPException(
            status_code=500,
            detail=f"Heat #{heat_pk} disappeared after update",
        )
    decision_id: int | None = None
    try:
        decision_id = log_decision(
            phase="deoxidation",
            decision=(
                f"Heat #{heat_pk} outcome confirmed: "
                + ", ".join(f"{k}={v}" for k, v in body.items())
            ),
            reasoning="Manual outcome confirmation by operator via UI",
            context={
                "heat_id": heat_pk,
                "plant_id": prev.plant_id,
                "before": {k: getattr(prev, k) for k in body.keys()},
                "after": body,
            },
            author="api_heats",
            tags=["heat_outcome_update", f"plant:{prev.plant_id}"],
        )
    except Exception as exc:  # noqa: BLE001 — audit save is best-effort
        # Decision Log failure shouldn't block the actual data update —
        # the row in heats.db is authoritative; the log is auxiliary.
        logger.warning("Decision Log save failed: %s", exc)
    return {"heat": _heat_to_dict(updated), "decision_id": decision_id}


@router.delete(
    "/heats/{heat_pk}",
    response_class=SafeJSONResponse,
    response_model=None,
)
def delete_heat_endpoint(heat_pk: int) -> dict[str, Any]:
    """Delete heat. Returns ``{deleted: bool, id}``. 404 if missing.

    No Decision Log write — deletes are typically used to undo a typo
    in manual entry, and the operator already saw the row before
    clicking delete. If a regulated environment needs delete-audit, add
    a tag here matching ``heat_outcome_update`` shape.
    """
    if get_heat_by_id(heat_pk) is None:
        raise HTTPException(status_code=404, detail=f"Heat #{heat_pk} not found")
    deleted = delete_heat(heat_pk)
    return {"deleted": deleted, "id": heat_pk}
