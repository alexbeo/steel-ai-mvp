"""Router for /api/prices/* — ferroalloy price snapshot CRUD.

PR 7 of the Streamlit→FastAPI migration. See
``docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md``
(Endpoint map → Tab «Дизайн сплава» → price snapshot).

The «💰 Прайс материалов» editor shows the seed snapshot, lets the
user upload a YAML, edits rows in place, and downloads back.

Endpoints:
- ``GET  /api/prices/active`` — return the current active snapshot. We
  do NOT persist a separate "active" snapshot per session yet (no auth,
  no session); active = seed unless the caller has POSTed a custom one
  in the same uvicorn process. Future PR can add a session-keyed store.
- ``POST /api/prices/upload`` — accepts an inline JSON dict in the same
  shape as ``yaml.safe_load(seed_2026-04-26.yaml)``, validates via the
  ``cost_model.PriceSnapshot`` dataclass round-trip, and persists to
  ``decision_log/price_snapshots/<ts>.yaml`` with a Decision Log entry
  (matches what ``inverse_designer.run_inverse_design`` does on its own
  but explicit for "save current edit" workflows).

We deliberately accept the snapshot as JSON (NOT multipart) — the price
editor in the UI maintains a JS object, and JSON-in-JSON is simpler than
encoding a YAML blob into multipart for a frontend that already speaks
JSON. The on-disk format remains YAML for parity with the seed file and
``cost_model.load_snapshot``.

Risks #3 (SafeJSONResponse): ``date`` fields and dataclass payloads are
handled by the encoder.
"""
from __future__ import annotations

import logging
import datetime as _dt
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from app.api.responses import SafeJSONResponse
from app.backend.cost_model import (
    Material,
    PriceSnapshot,
    save_snapshot,
    seed_snapshot,
)
from decision_log.logger import log_decision

logger = logging.getLogger(__name__)

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRICE_SNAPSHOTS_DIR = PROJECT_ROOT / "decision_log" / "price_snapshots"


# ──────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────


class MaterialPayload(BaseModel):
    """One material row in the upload body.

    Fields mirror the YAML structure used by ``cost_model.load_snapshot``:
        kind: base | ferroalloy | pure
        price_per_kg: > 0
        element_content: dict[Element, fraction] summing to ~1.0
    ``id`` is provided by the parent dict key, not by the row itself
    (to match the on-disk YAML shape).
    """

    kind: str
    price_per_kg: float = Field(..., gt=0.0)
    element_content: dict[str, float]


class SnapshotUploadRequest(BaseModel):
    """Request body for ``POST /api/prices/upload``.

    Pydantic validation handles top-level shape; we still re-run the
    domain-level invariants (sum of element_content ≈ 1.0, no negatives,
    valid kind) by round-tripping through ``Material`` dataclass — that
    is the source of truth for what counts as a valid snapshot.
    """

    # Pydantic v2 + `from __future__ import annotations` cannot resolve a
    # field named ``date`` if the type is also ``date``; we use Optional
    # and a different attribute name to side-step the shadowing, then alias
    # it to the on-disk YAML key. Field(alias="date") preserves wire format.
    snapshot_date: Optional[_dt.date] = Field(default=None, alias="date")
    currency: str = Field(..., min_length=3, max_length=3)
    materials: dict[str, MaterialPayload]
    source: str = "manual"
    notes: str = ""

    model_config = {
        "populate_by_name": True,  # accept both 'date' and 'snapshot_date'
    }


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _snapshot_to_dict(snapshot: PriceSnapshot) -> dict[str, Any]:
    """Project a PriceSnapshot dataclass to the JSON shape the UI consumes.

    Same key layout as ``yaml.safe_dump`` of the seed snapshot, so the
    frontend can round-trip through POST /upload without reshaping the
    body.
    """
    return {
        "date": snapshot.date.isoformat() if snapshot.date else None,
        "currency": snapshot.currency,
        "source": snapshot.source,
        "notes": snapshot.notes,
        "materials": {
            mid: {
                "kind": m.kind,
                "price_per_kg": m.price_per_kg,
                "element_content": dict(m.element_content),
            }
            for mid, m in snapshot.materials.items()
        },
    }


def _build_snapshot_from_request(req: SnapshotUploadRequest) -> PriceSnapshot:
    """Round-trip the request through Material/PriceSnapshot dataclasses.

    Reuses the dataclass-level invariants (positive prices, kind enum,
    element_content sum) without duplicating that logic here. Failures
    raise ``ValueError`` (parser-level English message — see CLAUDE.md
    "Языковая конвенция") which the caller maps to a 400.
    """
    materials: dict[str, Material] = {}
    for mid, payload in req.materials.items():
        # Mimic _validate_material_dict from cost_model: sum check + kind check.
        if payload.kind not in ("base", "ferroalloy", "pure"):
            raise ValueError(
                f"{mid}: kind must be base|ferroalloy|pure, got {payload.kind}"
            )
        if not payload.element_content:
            raise ValueError(f"{mid}: element_content is empty")
        if any(float(v) < 0 for v in payload.element_content.values()):
            raise ValueError(f"{mid}: element_content has negative values")
        s = sum(float(v) for v in payload.element_content.values())
        if abs(s - 1.0) > 0.02:
            raise ValueError(
                f"{mid}: element_content sum = {s:.3f}, must be ≈ 1.0 (±0.02)"
            )
        materials[mid] = Material(
            id=mid,
            kind=payload.kind,
            price_per_kg=float(payload.price_per_kg),
            element_content={k: float(v) for k, v in payload.element_content.items()},
        )
    snap_date = req.snapshot_date or _dt.date.today()
    if req.currency not in ("RUB", "USD", "EUR"):
        raise ValueError(
            f"currency must be RUB|USD|EUR, got {req.currency!r}"
        )
    return PriceSnapshot(
        date=snap_date,
        currency=req.currency,  # type: ignore[arg-type]
        materials=materials,
        source=req.source,
        notes=req.notes,
    )


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────


@router.get(
    "/active",
    response_class=SafeJSONResponse,
    response_model=None,
)
def get_active_snapshot() -> dict[str, Any]:
    """Return the active price snapshot (= seed in MVP).

    No session/user state yet, so "active" = whatever ``seed_snapshot()``
    returns. UI uses this to populate the price editor on first render;
    edits stay in JS state until the user POSTs ``/upload``.
    """
    try:
        snapshot = seed_snapshot()
    except Exception as exc:  # pragma: no cover — config error
        logger.exception("Failed to load seed price snapshot")
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось загрузить seed snapshot: {exc}",
        ) from exc
    return _snapshot_to_dict(snapshot)


@router.post(
    "/upload",
    response_class=SafeJSONResponse,
    response_model=None,
)
def upload_snapshot(req: SnapshotUploadRequest) -> dict[str, Any]:
    """Persist an edited snapshot to ``decision_log/price_snapshots/`` + log.

    The UI calls this when the user clicks «Сохранить snapshot». We:

      1. Round-trip the body through ``PriceSnapshot`` dataclass so the
         existing domain validation runs (positive prices, kind enum,
         element_content sum, currency whitelist).
      2. Write to ``decision_log/price_snapshots/<ts>.yaml`` via
         ``cost_model.save_snapshot`` so the file is byte-compatible
         with ``load_snapshot``.
      3. Add a Decision Log entry with tag ``cost_optimization`` so the
         History tab shows the upload (parity with what
         ``inverse_designer.run_inverse_design`` does on its own).

    Returns the persisted snapshot dict + the on-disk path so the UI
    can show "saved as <path>".
    """
    try:
        snapshot = _build_snapshot_from_request(req)
    except (ValueError, ValidationError) as exc:
        # Parser-level errors stay English (CLAUDE.md "Языковая
        # конвенция") — this 400 surfaces straight to the UI which
        # wraps it in a Russian banner.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        PRICE_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_path = PRICE_SNAPSHOTS_DIR / f"{ts}.yaml"
        save_snapshot(snapshot, snap_path)
    except Exception as exc:
        logger.exception("Failed to persist price snapshot")
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось сохранить snapshot: {exc}",
        ) from exc

    # Best-effort: a logging failure must not turn a successful upload
    # into a 500 — same pattern as active_learning.propose.
    try:
        log_decision(
            phase="inverse_design",
            decision=(
                f"Price snapshot uploaded via API "
                f"(date={snapshot.date}, currency={snapshot.currency}, "
                f"{len(snapshot.materials)} materials)"
            ),
            reasoning=(
                f"Source: {snapshot.source}. "
                f"Saved to {snap_path.relative_to(PROJECT_ROOT)}."
            ),
            context={
                "snapshot_path": str(snap_path),
                "currency": snapshot.currency,
                "n_materials": len(snapshot.materials),
            },
            author="api:prices.upload",
            tags=["cost_optimization", str(snapshot.date)],
        )
    except Exception:  # pragma: no cover — best-effort
        logger.exception("log_decision failed for prices.upload")

    return {
        "snapshot": _snapshot_to_dict(snapshot),
        "saved_path": str(snap_path.relative_to(PROJECT_ROOT)),
    }
