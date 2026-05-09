"""Router for /api/deox/* — Al deoxidation calculator (physics-only).

PR 4 of the Streamlit→FastAPI migration. See
``docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md``
(Endpoint map → Tab «Раскисление»).

Streamlit parity reference: ``app/frontend/app.py`` lines 920-1219
(``with tab_deox:`` block, sub_fwd / sub_inv / sub_cmp). The AI advisor
sub-tab (``sub_ai``) lands in PR 9 — this router intentionally stops
at three sync endpoints + a tiny model-listing helper.

Endpoints:
- ``GET  /api/deox/models``   — registry of thermo models for the UI dropdown
- ``POST /api/deox/forward``  — Al demand for a target O_a
- ``POST /api/deox/inverse``  — effective Al purity from observed deox depth
- ``POST /api/deox/compare``  — same forward inputs against all 3 models

Pattern Library hookup: each response carries ``pattern_warnings`` —
the subset of DX01/DX02/DX03 (Phase.DEOXIDATION) that triggered for the
relevant context. Streamlit shows them inline above the result; the JS
view will render the same set with severity-coloured banners.

Risks #3 (SafeJSONResponse): every endpoint declares
``response_class=SafeJSONResponse, response_model=None`` so dataclass +
numpy results from ``deoxidation`` serialise via the custom encoder
instead of Pydantic v2's default revalidation.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.responses import SafeJSONResponse
from app.backend.deoxidation import (
    DEFAULT_MODEL_ID,
    THERMO_MODELS,
    compare_all_models,
    compute_al_demand,
    compute_al_quality,
)
from pattern_library.patterns import Phase, run_all_patterns

logger = logging.getLogger(__name__)

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────
# Request schemas
#
# Pydantic v2 BaseModel is a thin validation layer — fields mirror the
# kwargs of the corresponding ``deoxidation.compute_*`` functions so the
# router body is just ``func(**req.model_dump())``. Bounds match the
# Streamlit number-input ranges (``app/frontend/app.py`` lines 1050-1185).
# ──────────────────────────────────────────────────────────────────────


class AlDemandRequest(BaseModel):
    """Forward calculation input — how much Al to add.

    Field bounds mirror the UI inputs (Streamlit ``st.number_input`` in
    ``app/frontend/app.py:1050-1185`` and ``deox.js`` ``buildField``
    ranges) so the API rejects values that the UI would not allow. This
    is a defensive consistency requirement raised in PR 4 review:
    Pydantic must not silently accept inputs the UI blocks (e.g.
    ``al_purity_pct=30`` would imply 30 % active Al — physically a
    booze-grade alloy, not metallurgical-grade).
    """

    o_a_initial_ppm: float = Field(..., ge=0.0, le=2000.0)
    temperature_C: float = Field(..., ge=1400.0, le=1700.0)
    steel_mass_ton: float = Field(..., ge=1.0, le=500.0)
    target_o_a_ppm: float = Field(..., gt=0.0, le=1000.0)
    al_purity_pct: float = Field(default=100.0, ge=50.0, le=100.0)
    burn_off_pct: float = Field(default=20.0, ge=0.0, le=50.0)
    model_id: str = Field(default=DEFAULT_MODEL_ID)
    al_price_per_kg: float = Field(default=2.40, ge=0.0)
    currency: str = Field(default="EUR", min_length=1, max_length=8)

    model_config = {
        # ``model_id``/``model_*`` collides with Pydantic's protected namespace;
        # disable the warning so we keep the kwarg name aligned with backend.
        "protected_namespaces": (),
    }


class AlQualityRequest(BaseModel):
    """Inverse calculation input — observed deox depth → effective purity.

    Field bounds align with the UI ranges in ``app/frontend/app.py``
    (lines 1131-1138) and ``deox.js`` (``renderInverseForm``). Same
    defensive principle as :class:`AlDemandRequest`.
    """

    o_a_before_ppm: float = Field(..., ge=0.0, le=2000.0)
    o_a_after_ppm: float = Field(..., ge=0.0, le=2000.0)
    al_added_kg: float = Field(..., ge=0.1, le=5000.0)
    temperature_C: float = Field(..., ge=1400.0, le=1700.0)
    steel_mass_ton: float = Field(..., ge=1.0, le=500.0)
    burn_off_pct: float = Field(default=20.0, ge=0.0, le=50.0)
    model_id: str = Field(default=DEFAULT_MODEL_ID)

    model_config = {"protected_namespaces": ()}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _serialise_warnings(warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project Pattern Library warnings to the public response shape.

    ``run_all_patterns`` already returns plain dicts with keys ``pattern_id``,
    ``title``, ``severity`` (string), ``message``, ``suggestion``, ``details``.
    We keep all of these — the UI uses ``severity`` for colour, ``message`` +
    ``suggestion`` for the banner text, and ``details`` is reserved for future
    debug/audit panels.
    """
    return [
        {
            "id": w.get("pattern_id"),
            "title": w.get("title"),
            "severity": w.get("severity"),
            "message": w.get("message"),
            "suggestion": w.get("suggestion"),
            "details": w.get("details") or {},
        }
        for w in warnings
    ]


def _deox_warnings_for_forward(
    o_a_initial_ppm: float, target_o_a_ppm: float
) -> list[dict[str, Any]]:
    """Run DX patterns relevant to the forward path.

    DX01 reads ``o_a_initial_ppm``; DX02 reads both ``o_a_initial_ppm`` and
    ``target_o_a_ppm``. DX03 (low effective purity) is inverse-only and
    returns False here — but we keep ``phase=DEOXIDATION`` rather than
    cherry-picking ids, so any future DX-pattern that triggers from
    forward inputs picks up automatically.
    """
    ctx = {
        "o_a_initial_ppm": o_a_initial_ppm,
        "target_o_a_ppm": target_o_a_ppm,
    }
    warnings = run_all_patterns(ctx, phase=Phase.DEOXIDATION)
    return _serialise_warnings(warnings)


def _deox_warnings_for_inverse(
    o_a_before_ppm: float,
    o_a_after_ppm: float,
    effective_purity_pct: float,
) -> list[dict[str, Any]]:
    """Run DX patterns relevant to the inverse path.

    Streamlit (line 1152-1155) only feeds ``effective_purity_pct`` into the
    pattern run, so DX01/DX02 don't fire there. We extend this slightly:
    feed the before/after pair too, so a sensor-out-of-range condition
    (DX01 on ``o_a_before_ppm``) still surfaces as a warning even in the
    inverse mode. This is strictly additive — Streamlit cannot mis-fire
    here because DX02 would only trigger when after >= before, which
    ``compute_al_quality`` already rejects with ValueError before we get
    here.
    """
    ctx = {
        "o_a_initial_ppm": o_a_before_ppm,
        "target_o_a_ppm": o_a_after_ppm,
        "effective_purity_pct": effective_purity_pct,
    }
    warnings = run_all_patterns(ctx, phase=Phase.DEOXIDATION)
    return _serialise_warnings(warnings)


def _validate_model_id(model_id: str) -> None:
    """Reject unknown thermo-model IDs with a 400 (not Pydantic 422).

    The set is small (3 entries) and lives in ``THERMO_MODELS`` registry,
    so a custom check produces a clearer error than a JSON-schema literal
    union and keeps the request model resilient to future entries
    (Hayashi-Yamamoto sibling models, fictitious calibration variants
    for tests).
    """
    if model_id not in THERMO_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown thermo model_id '{model_id}'. "
                f"Available: {sorted(THERMO_MODELS.keys())}"
            ),
        )


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────


@router.get(
    "/models",
    response_class=SafeJSONResponse,
    response_model=None,
)
def list_thermo_models() -> dict[str, Any]:
    """Return all registered thermo models for the UI dropdown.

    Shape mirrors ``GET /api/system/steel-classes`` (``items`` + ``count``)
    so the frontend can use the same render helper.
    """
    items = [
        {
            "id": m.id,
            "name": m.name,
            "citation": m.citation,
            "valid_t_range_c": [
                m.valid_t_range_k[0] - 273.15,
                m.valid_t_range_k[1] - 273.15,
            ],
            "expected_accuracy_ppm": m.expected_accuracy_ppm,
            "is_default": m.id == DEFAULT_MODEL_ID,
        }
        for m in THERMO_MODELS.values()
    ]
    return {"items": items, "count": len(items), "default": DEFAULT_MODEL_ID}


@router.post(
    "/forward",
    response_class=SafeJSONResponse,
    response_model=None,
)
def forward(req: AlDemandRequest) -> dict[str, Any]:
    """Compute Al demand to reduce O_a from initial to target.

    Validation order:
        1) Pydantic field bounds → 422 on failure (numbers out of range).
        2) Unknown ``model_id`` → 400 (cleaner than 422 generic).
        3) Backend ``compute_al_demand`` may raise ValueError on
           edge-case inputs (purity / burn_off out of (0, 100]) — we
           surface those as 400 too. ``target >= initial`` does NOT raise;
           the backend returns a result with al_total_kg=0 and a warning,
           which is the Streamlit behaviour.
    """
    _validate_model_id(req.model_id)
    try:
        result = compute_al_demand(**req.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pattern_warnings = _deox_warnings_for_forward(
        o_a_initial_ppm=req.o_a_initial_ppm,
        target_o_a_ppm=req.target_o_a_ppm,
    )
    return {
        "result": asdict(result),
        "pattern_warnings": pattern_warnings,
    }


@router.post(
    "/inverse",
    response_class=SafeJSONResponse,
    response_model=None,
)
def inverse(req: AlQualityRequest) -> dict[str, Any]:
    """Infer effective Al purity from observed deoxidation depth.

    ``compute_al_quality`` raises ValueError on physically impossible
    inputs (after >= before, al_added_kg <= 0). We surface those as 400
    so the UI can display an actionable message; Pydantic field bounds
    don't catch the after >= before case because both are independent
    fields.
    """
    _validate_model_id(req.model_id)
    try:
        result = compute_al_quality(**req.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pattern_warnings = _deox_warnings_for_inverse(
        o_a_before_ppm=req.o_a_before_ppm,
        o_a_after_ppm=req.o_a_after_ppm,
        effective_purity_pct=result.effective_purity_pct,
    )
    return {
        "result": asdict(result),
        "pattern_warnings": pattern_warnings,
    }


@router.post(
    "/compare",
    response_class=SafeJSONResponse,
    response_model=None,
)
def compare(req: AlDemandRequest) -> dict[str, Any]:
    """Run all 3 thermo models on identical inputs.

    The ``model_id`` field of the request is ignored on purpose — compare
    iterates over the full registry. We still validate it (so a typoed ID
    doesn't slip through) but the result is keyed by registry IDs, not
    by ``req.model_id``.

    Response shape::

        {
          "models": {<id>: <AlDemandResult>, ...},   # 3 entries
          "spread_pct": float,                       # max-min over mean × 100
          "pattern_warnings": [...],                 # same context as forward
        }

    ``spread_pct`` mirrors Streamlit's caption (line 1203): the relative
    disagreement between models. UI uses it for the "academic uncertainty"
    note.
    """
    _validate_model_id(req.model_id)
    payload = req.model_dump()
    # ``compare_all_models`` doesn't take model_id — drop it so the kwarg
    # set lines up with the function signature.
    payload.pop("model_id", None)
    try:
        results = compare_all_models(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    models_by_id: dict[str, dict[str, Any]] = {r.model_id: asdict(r) for r in results}
    masses = [r.al_total_kg for r in results]
    if masses and sum(masses) > 0:
        spread_pct = (max(masses) - min(masses)) / (sum(masses) / len(masses)) * 100
    else:
        spread_pct = 0.0

    pattern_warnings = _deox_warnings_for_forward(
        o_a_initial_ppm=req.o_a_initial_ppm,
        target_o_a_ppm=req.target_o_a_ppm,
    )
    return {
        "models": models_by_id,
        "spread_pct": float(spread_pct),
        "pattern_warnings": pattern_warnings,
    }
