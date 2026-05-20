"""Router for /api/deox/* — Al deoxidation calculator (physics + AI).

PR 4 of the Streamlit→FastAPI migration introduced three sync endpoints
(forward / inverse / compare). PR 9 added a fourth, asynchronous
endpoint — the AI advisor cycle (``sub_ai`` sub-tab). See
``docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md``.

Endpoints:
- ``GET  /api/deox/models``   — registry of thermo models for the UI dropdown
- ``POST /api/deox/forward``  — Al demand for a target O_a
- ``POST /api/deox/inverse``  — effective Al purity from observed deox depth
- ``POST /api/deox/compare``  — same forward inputs against all 3 models
- ``POST /api/deox/ai-cycle`` — PhD advisor + adversarial PhD critic cycle
                                (long-running ~30-180 s, returns ``{job_id}``)

Pattern Library hookup: every sync response carries ``pattern_warnings``
— the subset of DX01/DX02/DX03 (Phase.DEOXIDATION) that triggered for
the relevant context. The AI cycle attaches the same forward-context
warnings to the result so the UI can render risks alongside the LLM
verdict — the JS view renders them with severity-coloured banners.

Risks #3 (SafeJSONResponse): every endpoint declares
``response_class=SafeJSONResponse, response_model=None`` so dataclass +
numpy results from ``deoxidation`` (and dataclass results from the
advisor / critic) serialise via the custom encoder instead of Pydantic
v2's default revalidation.

PR 9 — AI cycle architecture:
The cycle reuses PR 6's ``run_as_job`` job infrastructure. Worker calls
the existing paired pattern: ``DeoxidationAdvisor`` (Sonnet PhD ladle
metallurgist) produces an operator protocol, then ``DeoxidationCritic``
(Sonnet PhD reviewer) does adversarial peer-review with evidence
fact-check. This mirrors recipe_designer + recipe_critic and the A2
hypothesis_generator + hypothesis_critic pair already in production.

Cooperative cancellation lands at one point — *before* the expensive
LLM call. The Anthropic SDK in this repo doesn't expose a streaming
cancel API for ``messages.create``, so once the network call is in
flight we cannot interrupt it. ``DELETE /api/jobs/{id}`` while the job
is in the "calling LLM" state therefore flips ``cancellation_requested``
but the worker burns through to completion (UI stops polling either
way). Documented in the design doc PR 9 row.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.jobs import run_as_job
from app.api.llm_gating import (
    check_llm_ready,
    make_progress_cancel_check,
)
from app.api.responses import SafeJSONResponse
from app.backend.deoxidation import (
    DEFAULT_MODEL_ID,
    THERMO_MODELS,
    compare_all_models,
    compute_al_demand,
    compute_al_quality,
)
from app.backend.slag_aware_deox import (
    DEFAULT_AL_COMMODITY_PRICE_EUR_PER_KG,
    DEOX_METHODS_PATH,
    CoDeoxSi,
    SlagState,
    load_addition_methods,
    recommend_optimal_method,
)
from decision_log.logger import log_decision
from pattern_library.patterns import Phase, run_all_patterns

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Decision Log snapshot paths (PR 8 — slag-aware deox audit trail) ──
# PROJECT_ROOT mirrors the constant in ``app.api.routers.prices`` so the
# two routers compute identical paths even if the package is moved.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEOX_METHODS_SNAPSHOTS_DIR = PROJECT_ROOT / "decision_log" / "deox_methods_snapshots"


# ──────────────────────────────────────────────────────────────────────
# Request schemas
#
# Pydantic v2 BaseModel is a thin validation layer — fields mirror the
# kwargs of the corresponding ``deoxidation.compute_*`` functions so the
# router body is just ``func(**req.model_dump())``. Bounds match the
# Field bounds align with ``deox.js`` ``buildField`` ranges in the UI.
# ──────────────────────────────────────────────────────────────────────


class AlDemandRequest(BaseModel):
    """Forward calculation input — how much Al to add.

    Field bounds mirror the ``deox.js`` ``buildField`` ranges so the
    API rejects values that the UI would not allow. This is a
    defensive consistency requirement raised in PR 4 review:
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

    Field bounds align with ``deox.js`` ``renderInverseForm`` ranges.
    Same defensive principle as :class:`AlDemandRequest`.
    """

    o_a_before_ppm: float = Field(..., ge=0.0, le=2000.0)
    o_a_after_ppm: float = Field(..., ge=0.0, le=2000.0)
    al_added_kg: float = Field(..., ge=0.1, le=5000.0)
    temperature_C: float = Field(..., ge=1400.0, le=1700.0)
    steel_mass_ton: float = Field(..., ge=1.0, le=500.0)
    burn_off_pct: float = Field(default=20.0, ge=0.0, le=50.0)
    model_id: str = Field(default=DEFAULT_MODEL_ID)

    model_config = {"protected_namespaces": ()}


class AlAdvisoryComposition(BaseModel):
    """Optional composition snapshot — feeds the critic risk-detection.

    All fields default to None so the UI can submit a partial picture
    (e.g. only C / Mn). The advisor + critic prompts handle missing
    fields gracefully — they're tagged "optional" in the schema.
    """

    c_pct: float | None = Field(default=None, ge=0.0, le=1.5)
    mn_pct: float | None = Field(default=None, ge=0.0, le=3.0)
    si_pct: float | None = Field(default=None, ge=0.0, le=2.5)
    s_pct: float | None = Field(default=None, ge=0.0, le=0.05)
    p_pct: float | None = Field(default=None, ge=0.0, le=0.05)


class AlAdvisoryRequest(BaseModel):
    """AI-cycle input — full heat context for advisor + critic.

    The cycle is a single LLM round-trip per agent (~80 s advisor +
    ~80 s critic = ~3 min total). Includes an opt-in
    ``save_to_decision_log`` flag so automated/test callers can submit
    without polluting the audit trail.
    """

    # Heat parameters — tighter bounds than forward (advisor expects a
    # plausible LF heat; pathological values would noise the LLM call).
    o_a_initial_ppm: float = Field(..., ge=0.0, le=2000.0)
    target_o_a_ppm: float = Field(..., gt=0.0, le=50.0)
    temperature_C: float = Field(..., ge=1400.0, le=1700.0)
    steel_mass_ton: float = Field(..., ge=1.0, le=500.0)
    al_purity_pct: float = Field(default=99.7, ge=50.0, le=100.0)
    burn_off_pct: float = Field(default=20.0, ge=0.0, le=50.0)
    thermo_model: str = Field(
        default=DEFAULT_MODEL_ID,
        description="Thermo model id used to seed the advisor's estimates",
    )
    # Optional context — empty object means "no composition known".
    composition: AlAdvisoryComposition = Field(
        default_factory=AlAdvisoryComposition,
    )
    slag_feo_pct: float | None = Field(default=None, ge=0.0, le=15.0)
    grade_target: str = Field(
        default="общая конструкционная",
        max_length=200,
        description="Free-text target grade / production task — context for critic",
    )
    heat_id: str | None = Field(
        default=None,
        max_length=50,
        description="Optional heat identifier for audit trail",
    )
    operator_notes: str | None = Field(
        default=None,
        max_length=2000,
        description="Operator's notes — surfaced to advisor + critic",
    )
    save_to_decision_log: bool = Field(
        default=False,
        description=(
            "Opt-in audit-trail save with tag 'deoxidation_cycle'. "
            "Default false to keep test/automation flows quiet — the UI "
            "exposes a «Сохранить» toggle to flip it explicitly."
        ),
    )

    model_config = {
        # ``thermo_model`` doesn't trip the protected-namespace warning
        # but consistency with the other request schemas — keep it off.
        "protected_namespaces": (),
    }


# ──────────────────────────────────────────────────────────────────────
# Slag-aware optimization schemas (PR 6 — Block 6 of asis-slag-aware spec)
#
# Three endpoints sit on top of ``app.backend.slag_aware_deox``:
#   GET  /api/deox/methods           — UI dropdown catalog (raw YAML rows)
#   POST /api/deox/optimize          — recommend_optimal_method + DX04-DX07
#   POST /api/deox/optimize/save     — Decision Log integration (PR 8 stub)
#
# The save endpoint currently returns 501 by design — full Decision Log
# integration (with deox_methods snapshot copy + price_snapshot_date echo)
# lands in PR 8 of the build sequence. The placeholder is shipped now so
# the frontend (PR 7) can wire the «Сохранить» button without a 404 race.
# ──────────────────────────────────────────────────────────────────────


class OptimizationRequest(BaseModel):
    """Slag-aware Al-deox optimization input.

    Mirrors the design-doc §6 schema. Field bounds are intentionally wider
    than the basic forward request because the slag-aware path covers
    BOF-tap inputs (657 ppm O_a in the Excel base case, 371 t heat) that
    the LF-only ``AlDemandRequest`` would reject. Bound order:

      * ``steel_mass_ton``  1-500 t      (covers small EAF to 400 t BOF)
      * ``o_a_initial_ppm`` 10-1500 ppm  (LF samples + BOF tap)
      * ``target_o_a_ppm``  1-50 ppm     (always low post-deox)
      * ``target_al_pct``   0.005-0.1 %  (residual [Al] window for HSLA / Q&T)

    Optional blocks (slag B / co-deox C / constraints E) default to None
    so the endpoint also serves as a slim "compare all methods, no slag"
    sanity caller — useful for tests.
    """

    # ── Block A — heat ────────────────────────────────────────────────
    steel_mass_ton: float = Field(..., ge=1.0, le=500.0)
    o_a_initial_ppm: float = Field(..., ge=10.0, le=1500.0)
    temperature_C: float = Field(default=1600.0, ge=1500.0, le=1700.0)
    target_o_a_ppm: float = Field(..., ge=1.0, le=50.0)
    target_al_pct: float = Field(..., ge=0.005, le=0.1)

    # ── Block B — slag carry-over (optional) ──────────────────────────
    # ``slag_mass_kg`` + ``slag_feo_pct`` together activate the
    # slag-aware path. If either is None, DX04 fires when the calculation
    # is marked slag-aware in the critic ctx (see _build_slag_aware_critic_ctx).
    slag_mass_kg: float | None = Field(default=None, ge=0.0, le=10000.0)
    slag_feo_pct: float | None = Field(default=None, ge=0.0, le=50.0)
    slag_mno_pct: float = Field(default=0.0, ge=0.0, le=20.0)
    slag_sio2_pct: float = Field(default=0.0, ge=0.0, le=30.0)

    # ── Block C — Si pre-deoxidation (optional) ───────────────────────
    co_deox_fesi_kg: float | None = Field(default=None, ge=0.0, le=5000.0)
    co_deox_fesi_si_content_pct: float = Field(default=75.0, gt=0.0, le=100.0)

    # ── Block D — methods ─────────────────────────────────────────────
    # ``method_ids=None`` → compare against the full YAML catalog. List of
    # strings restricts the candidate pool (handy for UI single-method drill-down).
    method_ids: list[str] | None = None
    user_override_eta_al: float | None = Field(default=None, ge=0.1, le=1.0)
    t_drying_c: float | None = Field(default=None, ge=0.0, le=600.0)

    # ── Block E — constraints ─────────────────────────────────────────
    target_n_ppm: float | None = Field(default=None, ge=0.0, le=500.0)
    premium_cap_eur_per_kg: float | None = Field(default=None, ge=0.0, le=20.0)

    # ── Block F — economics + thermo ──────────────────────────────────
    thermo_model_id: str = Field(default=DEFAULT_MODEL_ID)
    al_commodity_price_eur_per_kg: float = Field(
        default=DEFAULT_AL_COMMODITY_PRICE_EUR_PER_KG, ge=0.0, le=20.0
    )
    use_price_snapshot: bool = Field(
        default=True,
        description=(
            "Reserved for PR 8 — when true, override "
            "``al_commodity_price_eur_per_kg`` from active PriceSnapshot. "
            "Currently informational (the optimizer reads the field "
            "directly); kept in the schema so the frontend doesn't need "
            "a contract bump when PR 8 wires it up."
        ),
    )

    # ── Block G — multi-objective optimization (PR 7) ─────────────────
    # ``cost`` keeps the legacy behavior (min cost_per_heat_eur, default).
    # ``al_mass`` minimises pure Al mass (carbon footprint / inventory proxy).
    # ``pareto`` returns the non-dominated frontier; chosen point is the knee
    # (min L2 to utopia in min-max normalized space). The regex pattern keeps
    # Pydantic-side validation aligned with the backend ``OBJECTIVES`` tuple
    # so an invalid request short-circuits at 422 before reaching the optimizer.
    objective: str = Field(
        default="cost",
        pattern="^(cost|al_mass|pareto)$",
        description="Optimization objective: cost / al_mass / pareto",
    )

    # ── Block H — η_Al prediction (PR 10) ─────────────────────────────
    # When ``enable_eta_prediction`` is True the optimizer threads an
    # EtaAlPredictor (plant Bayesian posterior + global ML) through
    # ``recommend_optimal_method`` instead of literature η. ``plant_id``
    # is required in that case (the predictor needs it to look up the
    # posterior). Default False reproduces PR 7 behavior byte-identical.
    plant_id: str | None = Field(default=None, max_length=64)
    enable_eta_prediction: bool = Field(
        default=False,
        description="Use EtaAlPredictor (plant posterior + ML) instead of literature η",
    )

    model_config = {"protected_namespaces": ()}


class OptimizationSaveRequest(OptimizationRequest):
    """Body for ``POST /api/deox/optimize/save`` (PR 8 — Variant A).

    Vorint design decision (see spec §7 + PR 8 task brief):
    we accept the **full ``OptimizationRequest`` payload** rather than an
    echo of ``OptimizationResponse``. The endpoint re-executes
    ``recommend_optimal_method`` server-side using the same inputs, so the
    Decision Log entry reflects backend truth (single source) — no drift
    between UI display state and what's persisted in audit trail. Heat
    identifier + author live alongside the inputs.

    Inherits every field from :class:`OptimizationRequest` (Block A heat,
    Block B slag, Block C co-deox, Block D methods, Block E constraints,
    Block F economics) and adds the audit-trail-only fields below.
    """

    heat_id: str | None = Field(
        default=None,
        max_length=50,
        description="Optional heat identifier for audit trail.",
    )
    author: str = Field(default="user", max_length=50)


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

    The pattern run feeds ``effective_purity_pct`` plus the before/after
    pair so a sensor-out-of-range condition (DX01 on ``o_a_before_ppm``)
    still surfaces as a warning even in the inverse mode. DX02 cannot
    mis-fire here because it would only trigger when after >= before,
    which ``compute_al_quality`` already rejects with ValueError before
    we get here.
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
           the backend returns a result with al_total_kg=0 and a warning.
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

    ``spread_pct`` is the relative disagreement between models. UI uses
    it for the "academic uncertainty" note.
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


# ──────────────────────────────────────────────────────────────────────
# AI cycle (PR 9) — advisor + critic pair as a long-running job
# ──────────────────────────────────────────────────────────────────────


# PR 10 deduplication: previously a per-router copy of the closure-walk
# cancellation peek. Now delegates to the shared helper in
# ``app.api.llm_gating`` — the train.py copy moved with it, so the three
# duplicates collapse to a single source of truth. The wrapper kept here
# preserves the call site shape (``_check_cancelled(progress) -> bool``)
# so the worker body and any external callers stay untouched.
def _check_cancelled(progress: Any) -> bool:
    """Back-compat wrapper around :func:`make_progress_cancel_check`.

    The shared helper returns a *callable*; for the per-call style this
    file uses, we resolve once and invoke immediately. Keeping the wrapper
    means the worker code below (which calls ``_check_cancelled(progress)``
    twice — once before each LLM call) doesn't have to thread a fresh
    closure through the call sites.
    """
    return make_progress_cancel_check(progress)()


def _build_heat_context(req: AlAdvisoryRequest) -> dict[str, Any]:
    """Project the request into the dict shape the advisor/critic expect.

    Composition fields with None values are dropped from the dict so the
    LLM payload doesn't carry "null" keys that the prompt isn't tuned
    for. ``mn_s_ratio`` is a derived feature the prompt explicitly
    references — compute it when both Mn and S are present, skip
    otherwise.
    """
    comp_in = req.composition
    composition: dict[str, float] = {}
    if comp_in.c_pct is not None:
        composition["c_pct"] = float(comp_in.c_pct)
    if comp_in.mn_pct is not None:
        composition["mn_pct"] = float(comp_in.mn_pct)
    if comp_in.si_pct is not None:
        composition["si_pct"] = float(comp_in.si_pct)
    if comp_in.s_pct is not None:
        composition["s_pct"] = float(comp_in.s_pct)
    if comp_in.p_pct is not None:
        composition["p_pct"] = float(comp_in.p_pct)
    if comp_in.mn_pct is not None and comp_in.s_pct is not None:
        # Derived: mn_s_ratio = mn / max(s, 1e-6) — matches the prompt's
        # vocabulary and avoids div-by-zero on trace-S heats.
        composition["mn_s_ratio"] = float(
            comp_in.mn_pct / max(float(comp_in.s_pct), 1e-6)
        )

    ctx: dict[str, Any] = {
        "o_a_init_ppm": float(req.o_a_initial_ppm),
        "target_o_a_ppm": float(req.target_o_a_ppm),
        "temp_c": float(req.temperature_C),
        "mass_t": float(req.steel_mass_ton),
        "grade_target": req.grade_target,
    }
    if composition:
        ctx["composition"] = composition
    if req.slag_feo_pct is not None:
        ctx["slag_feo_pct"] = float(req.slag_feo_pct)
    if req.heat_id:
        ctx["heat_id"] = req.heat_id
    if req.operator_notes:
        ctx["operator_notes"] = req.operator_notes
    return ctx


def _run_ai_cycle_job(
    *,
    req_dict: dict[str, Any],
    save_to_decision_log: bool,
    progress: Any = None,
) -> dict[str, Any]:
    """Worker — invoked through ``run_as_job``.

    Stages:
        1. ``compare_all_models`` (sync, <100 ms) → seed thermo estimates
        2. cancellation gate
        3. ``DeoxidationAdvisor.advise`` (Sonnet ~80 s)
        4. cancellation gate
        5. ``DeoxidationCritic.review`` (Sonnet ~80 s)
        6. (opt-in) ``log_decision`` with tag ``deoxidation_cycle``

    Each LLM call already auto-logs token usage to the Decision Log
    inside the advisor/critic modules (their ``_log_usage`` helpers).
    The cycle-level ``log_decision`` call is the bundle entry that
    captures the *full operator protocol + verdict* for audit.

    ``req_dict`` is the validated body re-serialised as a plain dict
    rather than the Pydantic model instance — this keeps the JobStore
    worker arguments pure JSON-safe data and stays compatible with any
    future executor backend that demands plain Python objects.
    """
    started = time.monotonic()

    # Re-validate inside the worker — cheap (small model) and means we
    # don't trust whatever the caller stored into kwargs.
    req = AlAdvisoryRequest.model_validate(req_dict)

    if callable(progress):
        progress(0.05, "Считаю 3 thermo-модели…")

    # Stage 1 — sync thermo math. compare_all_models doesn't accept a
    # model_id, so we just run all three.
    cmp_res = compare_all_models(
        o_a_initial_ppm=req.o_a_initial_ppm,
        target_o_a_ppm=req.target_o_a_ppm,
        temperature_C=req.temperature_C,
        steel_mass_ton=req.steel_mass_ton,
        al_purity_pct=req.al_purity_pct,
        burn_off_pct=req.burn_off_pct,
    )
    thermo_estimates = {
        r.model_id: round(float(r.al_total_kg), 2) for r in cmp_res
    }

    # Pattern Library DX01/DX02 — same context as forward calc. We
    # capture these now (before LLM) so even a cancelled cycle still
    # surfaces the deterministic risks.
    pattern_warnings = _deox_warnings_for_forward(
        o_a_initial_ppm=req.o_a_initial_ppm,
        target_o_a_ppm=req.target_o_a_ppm,
    )

    heat_context = _build_heat_context(req)
    ctx = {"heat_context": heat_context, "thermo_estimates": thermo_estimates}

    # Stage 2 — cancellation gate before advisor LLM call. Once the
    # network call is in flight we cannot interrupt; this is the last
    # cheap exit point.
    if _check_cancelled(progress):
        raise RuntimeError("Cancelled by user (before advisor LLM call)")

    if callable(progress):
        progress(0.20, "PhD-советник формирует operator protocol (~80 с)…")

    # Lazy imports keep the FastAPI cold-start path light and let tests
    # stub the modules before they're loaded. The advisor/critic call
    # ``load_prompt`` at import time — if prompts/ is missing, the
    # PromptNotFoundError surfaces here as a job error (the endpoint
    # below pre-validates the prompt presence to avoid this case).
    from app.backend.deoxidation_advisor import make_deoxidation_advisor
    from app.backend.deoxidation_critic import make_deoxidation_critic

    advisor = make_deoxidation_advisor()
    critic = make_deoxidation_critic()
    if advisor is None or critic is None:
        # Late guard — in normal flow the endpoint pre-checks the API
        # key so we shouldn't hit this. If we do, surface as job error
        # (not 503) because the request was already accepted.
        raise RuntimeError(
            "AI advisor unavailable: ANTHROPIC_API_KEY missing or "
            "anthropic SDK not installed."
        )

    advisory = advisor.advise(ctx)
    if advisory is None:
        raise RuntimeError(
            "Advisor вернул None — Sonnet API call failed or response was malformed. "
            "Проверьте логи uvicorn."
        )

    # Stage 4 — cancellation gate before critic LLM call.
    if _check_cancelled(progress):
        raise RuntimeError("Cancelled by user (after advisor, before critic)")

    if callable(progress):
        progress(0.55, "PhD-критик делает peer-review (~80 с)…")

    advisory_dict = asdict(advisory)
    verdict = critic.review(ctx, advisory_dict)
    verdict_dict = asdict(verdict) if verdict is not None else None

    # Stage 6 — opt-in Decision Log save. We import here (not at module
    # top) so tests can monkeypatch the logger module and inject a tmp
    # DB path via ``log_decision`` kwargs.
    decision_log_id: int | None = None
    if save_to_decision_log:
        if callable(progress):
            progress(0.95, "Сохраняю в Decision Log…")
        try:
            from decision_log.logger import log_decision

            decision_log_id = log_decision(
                phase="deoxidation",
                decision=(
                    f"Deox cycle: Al={advisory.al_addition_kg:.1f} kg "
                    f"({advisory.al_form}), "
                    f"критик={verdict.verdict if verdict else 'N/A'}"
                ),
                reasoning=(
                    f"O_a {req.o_a_initial_ppm:g}→{req.target_o_a_ppm:g} ppm, "
                    f"T={req.temperature_C:g}°C, mass={req.steel_mass_ton:g}т"
                ),
                context={
                    "heat_context": heat_context,
                    "thermo_estimates": thermo_estimates,
                    "advisory": advisory_dict,
                    "review": verdict_dict,
                },
                author="api",
                tags=["deoxidation_cycle", "sonnet-4-6"],
            )
        except Exception as exc:  # noqa: BLE001 — audit save is best-effort
            # Don't fail the whole cycle if SQLite is unhappy — the
            # advisor / critic outputs are already in the response.
            logger.warning("Decision Log save failed: %s", exc)

    if callable(progress):
        progress(1.0, "Готово")

    duration_s = round(time.monotonic() - started, 2)

    return {
        "advisor": advisory_dict,
        "critic": verdict_dict,
        "pattern_warnings": pattern_warnings,
        "thermo_estimates": thermo_estimates,
        "duration_s": duration_s,
        "decision_log_id": decision_log_id,
        "config": {
            "thermo_model": req.thermo_model,
            "save_to_decision_log": save_to_decision_log,
            "heat_id": req.heat_id,
        },
    }


def _llm_ready_or_503() -> None:
    """Back-compat shim — the implementation moved to ``app.api.llm_gating``.

    PR 10 extracted the body into :func:`app.api.llm_gating.check_llm_ready`
    so PR 10/11 routers can share the exact same fail-fast contract
    (single Russian-language 503 detail, single prompt-files iteration
    style). This wrapper keeps the deox-specific prompt list local —
    callers stay readable and tests that monkeypatch
    ``app.backend.prompt_loader.load_prompt`` continue to work because
    the shared helper imports the symbol at call time.
    """
    check_llm_ready(["deoxidation_advisor", "deoxidation_critic"])


@router.post(
    "/ai-cycle",
    response_class=SafeJSONResponse,
    response_model=None,
)
def ai_cycle(req: AlAdvisoryRequest) -> dict[str, Any]:
    """Submit a PhD advisor + critic cycle as a long-running job.

    Validation order:
        1) Pydantic field bounds → 422 on failure.
        2) Unknown ``thermo_model`` → 400 (cleaner than 422 union).
        3) ``ANTHROPIC_API_KEY`` missing OR prompt files missing → 503
           with a clear remediation message.
        4) Submit job to the singleton store. Return ``{job_id}``.

    The actual cycle takes ~3 minutes (advisor + critic each ~80 s).
    The frontend polls ``GET /api/jobs/{job_id}`` (PR 6) until status
    flips to ``done``, then renders the result. ``DELETE /api/jobs/{id}``
    sets the cooperative cancellation flag — see module docstring for
    the cancellation semantics (effective only *before* the LLM call,
    not during).
    """
    _validate_model_id(req.thermo_model)
    _llm_ready_or_503()

    # ``model_dump`` produces a JSON-safe dict (None for missing
    # composition fields, etc.). We pass it through the worker as a
    # plain dict — keeps JobStore worker arguments executor-agnostic.
    job_id = run_as_job(
        _run_ai_cycle_job,
        req_dict=req.model_dump(),
        save_to_decision_log=bool(req.save_to_decision_log),
    )
    return {
        "job_id": job_id,
        "config": {
            "heat_id": req.heat_id,
            "o_a_initial_ppm": float(req.o_a_initial_ppm),
            "target_o_a_ppm": float(req.target_o_a_ppm),
            "temperature_C": float(req.temperature_C),
            "steel_mass_ton": float(req.steel_mass_ton),
            "thermo_model": req.thermo_model,
            "save_to_decision_log": bool(req.save_to_decision_log),
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Slag-aware optimization endpoints (PR 6)
# ──────────────────────────────────────────────────────────────────────


def _build_slag_state(req: OptimizationRequest) -> SlagState | None:
    """Project Block B fields onto a :class:`SlagState` if either anchor is set.

    "Slag-aware" is activated when **either** ``slag_mass_kg`` **or**
    ``slag_feo_pct`` is provided — the deliberately loose trigger lets
    DX04 fire when only one of the two is present (a half-filled form).
    Both None → returns None (purely dissolved-O calculation; no slag
    contribution).
    """
    if req.slag_mass_kg is None and req.slag_feo_pct is None:
        return None
    # Coerce missing component to 0 so the dataclass validates; DX04
    # picks up the original None from the ctx dict (see helper below).
    return SlagState(
        mass_kg=float(req.slag_mass_kg or 0.0),
        feo_pct=float(req.slag_feo_pct or 0.0),
        mno_pct=float(req.slag_mno_pct),
        sio2_pct=float(req.slag_sio2_pct),
    )


def _build_co_deox(req: OptimizationRequest) -> CoDeoxSi | None:
    """Project Block C onto :class:`CoDeoxSi` or return None.

    The optional FeSi pre-deox block needs the source mass to be > 0 to
    contribute O-consumption. ``co_deox_fesi_kg=0`` reads as "no co-deox"
    and is treated as None — keeps the response payload clean.
    """
    if req.co_deox_fesi_kg is None or req.co_deox_fesi_kg <= 0.0:
        return None
    return CoDeoxSi(
        si_source_kg=float(req.co_deox_fesi_kg),
        si_content_pct=float(req.co_deox_fesi_si_content_pct),
    )


def _build_eta_features(req: OptimizationRequest) -> dict[str, float]:
    """Build features_for_eta dict from OptimizationRequest fields that map
    to the deox_calibration model feature_set. method_eta_baseline and
    plant_offset_baseline are filled by EtaAlPredictor._resolve_features.
    Missing features → predictor degrades gracefully to plant/literature η.
    """
    features: dict[str, float] = {}
    # Map request fields → model features (subset present in request). The
    # model feature_list also expects composition / temperature-history /
    # stir features that the OptimizationRequest does not carry — those are
    # simply absent, and the predictor's _predict_global will fail-soft
    # (returns None, surfaces metadata["global_error"]); the optimizer then
    # falls back to plant posterior or literature η.
    if req.slag_feo_pct is not None:
        features["slag_feo_pct"] = float(req.slag_feo_pct)
    if req.slag_mno_pct is not None:
        features["slag_mno_pct"] = float(req.slag_mno_pct)
    if req.slag_sio2_pct is not None:
        features["slag_sio2_pct"] = float(req.slag_sio2_pct)
    if req.slag_mass_kg is not None:
        features["slag_mass_kg"] = float(req.slag_mass_kg)
    # ``temperature_C`` maps to the model's t_al_addition_c feature (the
    # temperature at the Al addition step — the request only carries one
    # bulk melt temperature).
    features["t_al_addition_c"] = float(req.temperature_C)
    features["o_a_initial_ppm"] = float(req.o_a_initial_ppm)
    features["steel_mass_ton"] = float(req.steel_mass_ton)
    if req.co_deox_fesi_kg is not None:
        features["co_deox_fesi_kg"] = float(req.co_deox_fesi_kg)
    return features


def _build_eta_calibration_ctx(
    request: OptimizationRequest,
    chosen_method_id: str,
    method: Any,
    eta_predictor: Any,
) -> dict[str, Any]:
    """Build the DX08/DX09/DX12 ctx keys from a chosen-method η_Al prediction.

    Only populated when an ``eta_predictor`` was actually used (i.e.
    ``enable_eta_prediction`` + ``plant_id``). The chosen method's prediction is
    re-derived here (cheap — no LLM, lazy model load already warm from the
    optimize call). All keys are best-effort: any failure leaves them absent so
    the DX08/DX09/DX12 checks degrade to no-trigger (``ctx.get`` → None).

    DX12 note (partial-dormant): ``EtaPrediction`` does not surface the global
    ML logit-μ as a standalone field. For ``source == 'mixed'`` the predictor
    records ``metadata['disagreement_logit'] = |mu_plant - mu_global|`` — we
    reconstruct ``global_eta_logit_mu`` from it. For ``plant_only`` /
    ``global_only`` / ``literature_fallback`` no global-vs-plant comparison
    exists, so ``global_eta_logit_mu`` stays absent and DX12 stays dormant.
    """
    ctx: dict[str, Any] = {"min_heats_threshold": 30}
    if eta_predictor is None or request.plant_id is None:
        return ctx
    try:
        pred = eta_predictor.predict_eta_al(
            plant_id=request.plant_id,
            method_id=chosen_method_id,
            features=_build_eta_features(request),
        )
    except Exception:
        return ctx  # graceful — keys remain absent, DX08/09/12 no-trigger

    ctx["eta_al_used"] = float(pred.eta_mean)
    ctx["eta_calibration_source"] = pred.source
    if method is not None:
        eta_range = getattr(method, "eta_al_range", None)
        if eta_range is not None and len(eta_range) == 2:
            ctx["method_eta_range"] = [float(eta_range[0]), float(eta_range[1])]

    # DX09 / DX12: plant posterior (n_heats + logit μ/σ for conflict check).
    calibrator = getattr(eta_predictor, "calibrator", None)
    posterior = None
    if calibrator is not None:
        try:
            posterior = calibrator.get_posterior(request.plant_id, chosen_method_id)
        except Exception:
            posterior = None
    if posterior is not None:
        ctx["plant_n_heats_for_method"] = int(posterior.n_heats_used)
        if posterior.posterior_logit_mu is not None:
            ctx["posterior_eta_logit_mu"] = float(posterior.posterior_logit_mu)
        if posterior.posterior_logit_sigma is not None:
            ctx["posterior_logit_sigma"] = float(posterior.posterior_logit_sigma)

    # DX12: reconstruct global ML logit-μ from the mixed-source disagreement.
    disagreement = pred.metadata.get("disagreement_logit") if pred.metadata else None
    if (
        pred.source == "mixed"
        and disagreement is not None
        and "posterior_eta_logit_mu" in ctx
    ):
        # Sign is irrelevant — DX12 uses |posterior_mu - global_mu|.
        ctx["global_eta_logit_mu"] = ctx["posterior_eta_logit_mu"] - float(disagreement)

    return ctx


def _build_slag_aware_critic_ctx(
    request: OptimizationRequest,
    chosen_method_id: str,
    eta_predictor: Any = None,
) -> dict[str, Any]:
    """Assemble the ctx dict consumed by DX04-DX12 pattern checks.

    Key design points:

    * ``slag_aware_calculation`` is True when *any* slag/co-deox block is
      populated — this matches the "the user intended slag-aware semantics"
      reading that DX04 wants. A heat with only ``slag_mass_kg`` set (no
      FeO yet) still triggers DX04, which is correct.
    * ``slag_state`` is a dict (not SlagState) so DX04 can inspect None
      sub-fields. The dataclass version coerces missing values to 0.
    * ``method`` is the *chosen* AdditionMethod from the YAML catalog —
      DX05/DX06/DX07 read its ``raw`` / ``eta_al_range`` / ``carrier_gas``.
    * ``co_deox_si`` mirrors Block C as a dict so future pattern checks
      can inspect Si-content without touching the dataclass internals.
    * DX08/DX09/DX12 keys are filled only when ``eta_predictor`` was used —
      see :func:`_build_eta_calibration_ctx`. DX10 ships dormant (basicity
      keys are never set here yet).
    """
    methods = load_addition_methods()
    method = methods.get(chosen_method_id)

    slag_aware = (
        request.slag_mass_kg is not None
        or request.slag_feo_pct is not None
        or request.slag_mno_pct > 0.0
        or request.slag_sio2_pct > 0.0
    )

    slag_state_dict: dict[str, Any] | None
    if slag_aware:
        slag_state_dict = {
            "mass_kg": request.slag_mass_kg,
            "feo_pct": request.slag_feo_pct,
            "mno_pct": request.slag_mno_pct,
            "sio2_pct": request.slag_sio2_pct,
        }
    else:
        slag_state_dict = None

    co_deox_dict: dict[str, Any] | None = None
    if request.co_deox_fesi_kg is not None and request.co_deox_fesi_kg > 0.0:
        co_deox_dict = {
            "si_source_kg": request.co_deox_fesi_kg,
            "si_content_pct": request.co_deox_fesi_si_content_pct,
        }

    ctx: dict[str, Any] = {
        # DX04 trigger anchors
        "slag_aware_calculation": slag_aware,
        "slag_state": slag_state_dict,
        # DX05 / DX06 / DX07 anchors
        "method": method,
        "user_override_eta_al": request.user_override_eta_al,
        "t_drying_c": request.t_drying_c,
        "target_n_ppm": request.target_n_ppm,
        # Informational — surfaced into pattern.details on future checks
        "co_deox_si": co_deox_dict,
        # DX01/DX02 anchors (kept so the same ctx can also be passed to
        # the forward-path patterns if someone reuses this helper).
        "o_a_initial_ppm": request.o_a_initial_ppm,
        "target_o_a_ppm": request.target_o_a_ppm,
    }
    # PR 13: DX08/DX09/DX12 η_Al calibration keys (only if predictor used).
    ctx.update(
        _build_eta_calibration_ctx(request, chosen_method_id, method, eta_predictor)
    )
    return ctx


@router.get(
    "/methods",
    response_class=SafeJSONResponse,
    response_model=None,
)
def list_addition_methods() -> dict[str, Any]:
    """Expose the YAML catalog of Al addition methods for the UI dropdown.

    Shape mirrors :func:`list_thermo_models` (``items`` + ``count`` +
    ``default``) so the frontend can reuse the same render helper. Each
    item is the *raw* YAML row with the method ``id`` added on top — the
    frontend renders ``name`` / ``eta_al_typical`` / ``premium_eur_per_kg``
    directly and the ``raw`` extras (``size_mm``, ``t_drying_max_c``,
    ``notes``) feed the tooltip / inspector panel.

    Default method (``asis_shot``) is hard-coded here for now — it's the
    most representative for the BOF→LF advisory we ship; future revisions
    may read a ``default`` key from the YAML if maintenance ergonomics
    require flipping it without a code change.
    """
    methods = load_addition_methods()
    items: list[dict[str, Any]] = []
    for method_id, method in methods.items():
        # Carry the full YAML row in ``raw`` for forward-compat (UI tooltip /
        # inspector) plus flatten the canonical fields at the top level so
        # the frontend doesn't have to dig.
        row: dict[str, Any] = {
            "id": method_id,
            "name": method.name,
            "eta_al_typical": float(method.eta_al_typical),
            "eta_al_range": list(method.eta_al_range),
            "premium_eur_per_kg": float(method.premium_eur_per_kg),
            "surface_m2_per_kg": float(method.surface_m2_per_kg),
            "carrier_gas": method.carrier_gas,
            "notes": method.notes,
            "extras": dict(method.raw),
        }
        items.append(row)

    default_id = "asis_shot" if "asis_shot" in methods else next(iter(methods))

    return {
        "items": items,
        "count": len(items),
        "default": default_id,
    }


def _pareto_row_to_dict(row: Any) -> dict[str, Any]:
    """Project ``MethodCompareRow`` (frozen dataclass) to the API shape.

    We intentionally don't reuse ``dataclasses.asdict`` — the frozen
    dataclass nests no custom types but listing the keys here documents
    the API contract for the frontend (PR 7) and the test (test_api_deox.py).
    """
    return {
        "method_id": row.method_id,
        "method_name": row.method_name,
        "eta_al_used": float(row.eta_al_used),
        "al_pure_kg": float(row.al_pure_kg),
        "al_charge_kg": float(row.al_charge_kg),
        "cost_per_heat_eur": float(row.cost_per_heat_eur),
        "cost_per_ton_eur": float(row.cost_per_ton_eur),
        "al_specific_kg_per_t": float(row.al_specific_kg_per_t),
        "carrier_gas": row.carrier_gas,
        "scatter_kg": float(row.scatter_kg),
        "warnings": list(row.warnings),
    }


@router.post(
    "/optimize",
    response_class=SafeJSONResponse,
    response_model=None,
)
def optimize_deox_method(req: OptimizationRequest) -> dict[str, Any]:
    """Recommend the optimal Al addition method for one heat.

    Validation order:
        1) Pydantic field bounds → 422 on failure.
        2) Unknown ``thermo_model_id`` → 400 (router-level).
        3) Unknown ``method_ids`` entries → 400 (surface from backend).
        4) Backend ``recommend_optimal_method`` may raise ValueError if
           constraints leave no surviving methods — surface as 400.
        5) Pattern Library DX04-DX07 attached to ``pattern_warnings``.

    The endpoint always returns a ``pareto_table`` sorted ascending by
    ``cost_per_heat_eur``; the first row is the chosen method. When more
    than one method survives the constraint filters, ``runner_up_*`` is
    populated; otherwise it's None.

    Decision Log save is **not** triggered here — that's the role of
    ``POST /api/deox/optimize/save`` (PR 8 placeholder).
    """
    _validate_model_id(req.thermo_model_id)

    slag = _build_slag_state(req)
    co_deox = _build_co_deox(req)

    # PR 10 — optional η_Al prediction. Mutual exclusion: a manual
    # ``user_override_eta_al`` and the predictor can't both drive η.
    if req.enable_eta_prediction and req.user_override_eta_al is not None:
        raise HTTPException(
            status_code=400,
            detail="enable_eta_prediction несовместим с user_override_eta_al",
        )

    eta_predictor = None
    features_for_eta = None
    if req.enable_eta_prediction:
        if req.plant_id is None:
            raise HTTPException(
                status_code=400,
                detail="enable_eta_prediction requires plant_id",
            )
        # Per-request instantiation (simple — the predictor lazy-loads the
        # model bundle on first predict and the calibrator reads YAML on
        # demand, so the per-request cost is small).
        from app.backend.eta_al_calibration import EtaAlCalibrator
        from app.backend.eta_al_predictor import EtaAlPredictor

        eta_predictor = EtaAlPredictor(calibrator=EtaAlCalibrator())
        features_for_eta = _build_eta_features(req)

    try:
        recommendation = recommend_optimal_method(
            steel_mass_ton=req.steel_mass_ton,
            o_a_initial_ppm=req.o_a_initial_ppm,
            target_o_a_ppm=req.target_o_a_ppm,
            target_al_pct=req.target_al_pct,
            slag=slag,
            co_deox_si=co_deox,
            temperature_C=req.temperature_C,
            thermo_model_id=req.thermo_model_id,
            al_commodity_price_eur_per_kg=req.al_commodity_price_eur_per_kg,
            method_ids=req.method_ids,
            target_n_ppm=req.target_n_ppm,
            premium_cap_eur_per_kg=req.premium_cap_eur_per_kg,
            objective=req.objective,
            eta_al_predictor=eta_predictor,
            plant_id=req.plant_id,
            features_for_eta=features_for_eta,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Pattern Library DX04-DX07. The ctx is keyed off the *chosen* method —
    # DX05/DX06/DX07 only need to surface for the recommendation, not for
    # every rejected candidate (those have their own ``reason`` echoed in
    # rejected_methods). DX04 doesn't depend on the chosen method at all,
    # so it fires regardless.
    critic_ctx = _build_slag_aware_critic_ctx(
        req, recommendation.chosen_method_id, eta_predictor=eta_predictor
    )
    pattern_dicts = run_all_patterns(critic_ctx, phase=Phase.DEOXIDATION)
    # DX01/DX02 always run alongside DX04-DX07 since they share the phase;
    # they're informational here (forward-path target/initial both inside
    # the slag-aware request schema), so we keep them in the response.
    # Use ``_serialise_warnings`` for contract parity with sibling
    # endpoints (forward / inverse / compare) — single ``id`` key in the
    # public payload, so the PR 7 UI can read ``w.id`` uniformly.
    pattern_warnings = _serialise_warnings(pattern_dicts)

    pareto_table = [_pareto_row_to_dict(r) for r in recommendation.pareto_table]
    # PR 7 — multi-objective: echo the requested objective + (for "pareto")
    # the non-dominated frontier so the UI can render a scatter chart.
    # For "cost"/"al_mass" the frontier is empty by design.
    pareto_frontier = [
        _pareto_row_to_dict(r) for r in recommendation.pareto_frontier
    ]

    return {
        "chosen_method_id": recommendation.chosen_method_id,
        "chosen_method_name": recommendation.chosen_method_name,
        "chosen_cost_eur": float(recommendation.chosen_cost_eur),
        "rationale": recommendation.rationale,
        "runner_up_method_id": recommendation.runner_up_method_id,
        "runner_up_cost_eur": (
            float(recommendation.runner_up_cost_eur)
            if recommendation.runner_up_cost_eur is not None
            else None
        ),
        "runner_up_delta_eur": (
            float(recommendation.runner_up_delta_eur)
            if recommendation.runner_up_delta_eur is not None
            else None
        ),
        "constraints_active": list(recommendation.constraints_active),
        "rejected_methods": list(recommendation.rejected_methods),
        "pareto_table": pareto_table,
        "pattern_warnings": pattern_warnings,
        "thermo_model_used": req.thermo_model_id,
        "inputs": dict(recommendation.inputs),
        "objective": recommendation.objective,
        "pareto_frontier": pareto_frontier,
        "eta_prediction_used": bool(req.enable_eta_prediction),
    }


def _git_head_sha() -> str | None:
    """Best-effort ``git rev-parse HEAD`` for reproducibility metadata.

    Returns the full 40-char SHA when the project sits inside a working
    git tree; ``None`` otherwise (e.g. a Docker container without the
    ``.git`` directory). Failures are swallowed silently — this is audit
    metadata, not a control-flow signal.
    """
    try:
        out = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
        return out.strip() or None
    except Exception:  # noqa: BLE001 — best-effort metadata
        return None


def _git_branch() -> str | None:
    """Best-effort ``git rev-parse --abbrev-ref HEAD`` — same contract as SHA."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
        branch = out.strip()
        return branch if branch and branch != "HEAD" else None
    except Exception:  # noqa: BLE001
        return None


def _sha256_file(path: Path) -> str:
    """SHA-256 hex digest of ``path`` (read fully, no streaming).

    The al_addition_methods.yaml catalog is tiny (~3 KB, 5 methods), so
    we read it all at once rather than chunked-hash for code simplicity.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_deox_methods_yaml(
    *, snapshots_dir: Path, source_yaml: Path
) -> tuple[Path, str]:
    """Copy current methods YAML to ``snapshots_dir/<ISO-ts>.yaml`` + hash.

    ISO-timestamp form uses ``-`` separators inside the time component
    (``2026-05-12T20-30-00``) because ``:`` is illegal in filenames on
    Windows / macOS-on-FAT. Returns ``(snapshot_path, sha256_hex)``.

    If the source YAML is missing we raise — this is a 500-grade
    failure (the optimizer that just produced ``rec`` had to read the
    same file), not a recoverable 400.
    """
    if not source_yaml.exists():
        raise FileNotFoundError(
            f"Source deox methods YAML not found at {source_yaml}; "
            "snapshot cannot be taken."
        )
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    # ``datetime.now()`` (local) — same convention as
    # ``inverse_designer.run_inverse_design`` for price_snapshots.
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    snapshot_path = snapshots_dir / f"{ts}.yaml"
    # Collision guard — extremely unlikely on second-level ISO ts, but
    # cheap to defend against operators clicking «Сохранить» rapidly.
    if snapshot_path.exists():
        suffix = 1
        while (alt := snapshots_dir / f"{ts}_{suffix}.yaml").exists():
            suffix += 1
        snapshot_path = alt
    shutil.copyfile(source_yaml, snapshot_path)
    sha256 = _sha256_file(snapshot_path)
    return snapshot_path, sha256


@router.post(
    "/optimize/save",
    response_class=SafeJSONResponse,
    response_model=None,
)
def save_optimization_recommendation(
    req: OptimizationSaveRequest,
) -> dict[str, Any]:
    """Persist an optimize recommendation to Decision Log + YAML snapshot.

    Flow (Variant A — backend recompute, single source of truth):
        1. Re-validate thermo model + run ``recommend_optimal_method``
           with the exact same inputs the UI submitted.
        2. Copy the current ``data/deox_methods/al_addition_methods.yaml``
           to ``decision_log/deox_methods_snapshots/<ISO-ts>.yaml`` and
           hash it (SHA-256) for reproducibility.
        3. Emit a Decision Log row with phase=deoxidation,
           tags=[deoxidation, asis, deox_method_recommendation,
           method:<chosen_id>].

    The Decision Log save is the *opt-in* part — the user already saw the
    recommendation on the «Оптимизация метода» tab and clicked «Сохранить»
    explicitly. Without that click the optimize endpoint is read-only.

    Response shape::

        {
          "decision_id": <int>,
          "methods_snapshot_path": "decision_log/deox_methods_snapshots/<ts>.yaml",
          "methods_snapshot_sha256": "<64-char hex>",
          "chosen_method_id": "asis_shot",
          "chosen_cost_eur": 1230.45,
        }
    """
    _validate_model_id(req.thermo_model_id)

    slag = _build_slag_state(req)
    co_deox = _build_co_deox(req)

    # Re-execute optimize so the persisted record reflects backend truth
    # (Variant A). Any ValueError from the optimizer surfaces as 400 —
    # mirrors the contract of /optimize.
    try:
        recommendation = recommend_optimal_method(
            steel_mass_ton=req.steel_mass_ton,
            o_a_initial_ppm=req.o_a_initial_ppm,
            target_o_a_ppm=req.target_o_a_ppm,
            target_al_pct=req.target_al_pct,
            slag=slag,
            co_deox_si=co_deox,
            temperature_C=req.temperature_C,
            thermo_model_id=req.thermo_model_id,
            al_commodity_price_eur_per_kg=req.al_commodity_price_eur_per_kg,
            method_ids=req.method_ids,
            target_n_ppm=req.target_n_ppm,
            premium_cap_eur_per_kg=req.premium_cap_eur_per_kg,
            objective=req.objective,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Snapshot methods YAML — required for reproducibility (the audit
    # entry references a specific file path/hash).
    try:
        snapshot_path, snapshot_sha256 = _snapshot_deox_methods_yaml(
            snapshots_dir=DEOX_METHODS_SNAPSHOTS_DIR,
            source_yaml=DEOX_METHODS_PATH,
        )
    except FileNotFoundError as exc:
        logger.exception("Methods YAML missing — cannot snapshot")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except OSError as exc:
        logger.exception("Failed to write methods snapshot")
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось сохранить snapshot методов: {exc}",
        ) from exc

    # Path stored in Decision Log is repo-relative for portability —
    # avoids absolute paths that would leak the operator's home dir.
    try:
        snapshot_rel = snapshot_path.relative_to(PROJECT_ROOT)
    except ValueError:  # tmp_path test fixture sits outside PROJECT_ROOT
        snapshot_rel = snapshot_path

    # Slag + co-deox sub-context — only included when populated so the
    # Decision Log row doesn't carry "null" placeholders for the
    # basic-forward sanity-check case.
    slag_state_ctx: dict[str, Any] | None = None
    if (
        req.slag_mass_kg is not None
        or req.slag_feo_pct is not None
        or req.slag_mno_pct > 0.0
        or req.slag_sio2_pct > 0.0
    ):
        slag_state_ctx = {
            "mass_kg": req.slag_mass_kg,
            "feo_pct": req.slag_feo_pct,
            "mno_pct": req.slag_mno_pct,
            "sio2_pct": req.slag_sio2_pct,
        }
    co_deox_ctx: dict[str, Any] | None = None
    if req.co_deox_fesi_kg is not None and req.co_deox_fesi_kg > 0.0:
        co_deox_ctx = {
            "fesi_kg": req.co_deox_fesi_kg,
            "si_content_pct": req.co_deox_fesi_si_content_pct,
        }

    # Top-5 runner-ups by ascending cost — keeps the alternatives list
    # informative without bloating the Decision Log row.
    runner_up_ids = [
        row.method_id for row in recommendation.pareto_table[1:5]
    ]

    context_payload: dict[str, Any] = {
        "heat_id": req.heat_id,
        "branch": _git_branch(),
        "commit_sha": _git_head_sha(),
        "slag_state": slag_state_ctx,
        "co_deox": co_deox_ctx,
        "thermo_model": req.thermo_model_id,
        "methods_snapshot_path": str(snapshot_rel),
        "methods_snapshot_sha256": snapshot_sha256,
        "al_commodity_price_eur_per_kg": req.al_commodity_price_eur_per_kg,
        "user_override_eta_al": req.user_override_eta_al,
        "target_n_ppm": req.target_n_ppm,
        "premium_cap_eur_per_kg": req.premium_cap_eur_per_kg,
        "t_drying_c": req.t_drying_c,
        "chosen_cost_eur": float(recommendation.chosen_cost_eur),
        "chosen_method_name": recommendation.chosen_method_name,
        "constraints_active": list(recommendation.constraints_active),
        "inputs": dict(recommendation.inputs),
        # PR 7 — multi-objective: persist the requested criterion so the
        # History tab can render the choice basis without re-deriving it
        # from the rationale string.
        "objective": req.objective,
    }

    # ``al_pure_kg`` of the chosen row goes into the decision summary —
    # it's the single most useful number for a metallurgist scanning the
    # History tab ("how much Al did we recommend?").
    chosen_row = recommendation.pareto_table[0] if recommendation.pareto_table else None
    al_kg_str = f"{chosen_row.al_pure_kg:.1f}" if chosen_row is not None else "?"

    decision_id = log_decision(
        phase="deoxidation",
        decision=(
            f"method={recommendation.chosen_method_id}; "
            f"al_kg={al_kg_str}; "
            f"cost={recommendation.chosen_cost_eur:.0f}€"
        ),
        reasoning=recommendation.rationale,
        alternatives_considered=runner_up_ids,
        context=context_payload,
        author=req.author or "user",
        tags=[
            "deoxidation",
            "asis",
            "deox_method_recommendation",
            f"method:{recommendation.chosen_method_id}",
        ],
    )

    return {
        "decision_id": decision_id,
        "methods_snapshot_path": str(snapshot_rel),
        "methods_snapshot_sha256": snapshot_sha256,
        "chosen_method_id": recommendation.chosen_method_id,
        "chosen_cost_eur": float(recommendation.chosen_cost_eur),
    }


# ──────────────────────────────────────────────────────────────────────
# η_Al calibration + model status endpoints (PR 10)
#
# Three read/run endpoints feed the «🎯 Калибровка η_Al» sub-tab:
#   GET  /api/deox/eta-al-model/status — trained ML model metrics + plants
#   GET  /api/deox/calibrations        — plant×method Bayesian posteriors
#   POST /api/deox/calibrations/run    — (re)run calibration sync
#
# Calibration runs synchronously (no job infra) — the Bayesian update is a
# closed-form logit-space conjugate step over a handful of heats per plant,
# fast enough to block the request thread.
# ──────────────────────────────────────────────────────────────────────


def _list_calibrated_plants() -> list[str]:
    """Return plant_ids that have a calibration YAML on disk (stem = plant_id)."""
    from app.backend.eta_al_calibration import DEFAULT_CALIB_DIR

    if not DEFAULT_CALIB_DIR.exists():
        return []
    return sorted(p.stem for p in DEFAULT_CALIB_DIR.glob("*.yaml"))


@router.get(
    "/eta-al-model/status",
    response_class=SafeJSONResponse,
    response_model=None,
)
def eta_al_model_status() -> dict[str, Any]:
    """Status of trained η_Al ML model + list of calibrated plants.

    ``coverage_in_target`` flags whether the 90% CI empirical coverage sits
    in the M02 acceptance band (85–95%). The UI renders an amber pill when
    it's out of band.
    """
    import json

    from app.backend.eta_al_predictor import _DEFAULT_MODELS_DIR, _scan_latest_model

    version = _scan_latest_model(_DEFAULT_MODELS_DIR)
    if version is None:
        return {
            "model_present": False,
            "model_version": None,
            "r2_test": None,
            "coverage_90_ci": None,
            "coverage_in_target": None,
            "trained_at": None,
            "n_train": None,
            "calibrated_plants": _list_calibrated_plants(),
        }
    meta_path = _DEFAULT_MODELS_DIR / version / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    metrics = meta.get("metrics", {})
    r2 = metrics.get("r2_test")
    coverage = metrics.get("coverage_90_ci")
    coverage_in_target = coverage is not None and 0.85 <= coverage <= 0.95
    return {
        "model_present": True,
        "model_version": version,
        "r2_test": r2,
        "coverage_90_ci": coverage,
        "coverage_in_target": coverage_in_target,
        "trained_at": meta.get("trained_at"),
        "n_train": metrics.get("n_train"),
        "calibrated_plants": _list_calibrated_plants(),
    }


@router.get(
    "/calibrations",
    response_class=SafeJSONResponse,
    response_model=None,
)
def list_calibrations() -> dict[str, Any]:
    """All plant×method posteriors flattened from the calibration YAML files.

    One row per (plant_id, method_id). ``skipped_reason`` is non-null for
    methods that had too few heats to update the prior — the UI tags those.
    """
    import yaml

    from app.backend.eta_al_calibration import DEFAULT_CALIB_DIR

    items: list[dict[str, Any]] = []
    if DEFAULT_CALIB_DIR.exists():
        for yaml_path in sorted(DEFAULT_CALIB_DIR.glob("*.yaml")):
            data = yaml.safe_load(yaml_path.read_text()) or {}
            plant_id = data.get("plant_id", yaml_path.stem)
            for method_id, c in (data.get("calibrations") or {}).items():
                items.append(
                    {
                        "plant_id": plant_id,
                        "method_id": method_id,
                        "n_heats_used": c.get("n_heats_used"),
                        "posterior_eta_mean": c.get("posterior_eta_mean"),
                        "posterior_eta_q05": c.get("posterior_eta_q05"),
                        "posterior_eta_q95": c.get("posterior_eta_q95"),
                        "prior_eta_mean": c.get("prior_eta_mean"),
                        "skipped_reason": c.get("skipped_reason"),
                    }
                )
    return {"items": items, "count": len(items)}


class CalibrationRunRequest(BaseModel):
    """Body for ``POST /api/deox/calibrations/run``.

    ``plant_id=None`` → calibrate every plant found in the heats DB.
    A non-null ``plant_id`` restricts the run to that single plant.
    """

    plant_id: str | None = Field(default=None, max_length=64)

    model_config = {"protected_namespaces": ()}


@router.post(
    "/calibrations/run",
    response_class=SafeJSONResponse,
    response_model=None,
)
def run_calibration(req: CalibrationRunRequest) -> dict[str, Any]:
    """Run Bayesian η_Al calibration synchronously.

    Optional ``plant_id`` for a single-plant run; otherwise all plants in the
    heats DB are calibrated. Each plant with ≥1 calibrated method writes a
    YAML posterior file + a Decision Log entry (handled inside the calibrator).
    """
    from app.backend.eta_al_calibration import EtaAlCalibrator

    cal = EtaAlCalibrator()
    if req.plant_id:
        results = [cal.calibrate_plant(req.plant_id)]
    else:
        results = cal.calibrate_all_plants()
    return {
        "plants_calibrated": len(results),
        "yaml_written": sum(1 for r in results if r.yaml_written),
        "results": [
            {
                "plant_id": r.plant_id,
                "n_total_heats": r.n_total_heats,
                "methods_calibrated": sum(
                    1 for p in r.calibrations if p.skipped_reason is None
                ),
                "methods_skipped": sum(
                    1 for p in r.calibrations if p.skipped_reason is not None
                ),
            }
            for r in results
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# Shadow validation (Phase 2 S4) — retrospective AI-vs-actual Al dosing.
#
#   POST /api/deox/shadow/run    — run comparison sync (heats_db | synthetic)
#   GET  /api/deox/shadow/report — latest persisted shadow report JSON
#
# Runs synchronously with a hard ``limit`` guard (default 2000 heats). If the
# heats DB holds more outcomes than ``limit`` we return 400 with a CLI hint
# rather than turning this into a background job — keeps the surface small for
# the MVP. The full unbounded run lives in scripts/generate_shadow_report.py.
#
# Predictor reuses the existing trained posteriors via
# ``EtaAlPredictor(calibrator=EtaAlCalibrator())`` — we never re-calibrate here.
# ──────────────────────────────────────────────────────────────────────


class ShadowRunRequest(BaseModel):
    """Body for ``POST /api/deox/shadow/run``.

    ``source=heats_db`` compares against persisted outcomes (subject to the
    ``limit`` guard); ``source=synthetic`` generates a self-contained demo
    batch (η re-prediction is tautological — flagged ``is_synthetic`` in the
    response and the report disclaimers).
    """

    source: Literal["heats_db", "synthetic"] = "heats_db"
    n_synthetic_heats: int = Field(500, ge=10, le=2000)
    plant_id: str | None = Field(None, max_length=64)
    limit: int = Field(2000, ge=1, le=20000)
    al_price_eur_per_kg: float = Field(2.40, gt=0)
    save_report: bool = True
    target_reduction_pct: float = Field(10.0, gt=0)

    model_config = {"protected_namespaces": ()}


def _nan_to_none(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN/±inf) with ``None``.

    ``ShadowStats`` legitimately carries NaN for the empty / no-quality-pass
    edge cases (median_delta_pct, ci_low/high). ``SafeJSONResponse`` renders
    with ``allow_nan=False``, so we sanitise the payload to JSON-null before it
    reaches the encoder. The UI already treats null + NaN as "—".
    """
    import math

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def _synthetic_heats_for_shadow(n: int) -> list:
    """Generate synthetic HeatRecords with derived ``al_added_kg`` for the demo.

    The calibration generator emits η + slag/process features but no Al charge;
    we back-derive a plausible actual Al addition from the mass-balance O removal
    and the synthetic η so the shadow comparison has an ``al_actual_kg`` to beat.
    """
    from app.backend.data_curator import (
        generate_synthetic_deox_calibration_dataset,
    )
    from app.backend.heat_records import HeatRecord
    from app.backend.slag_aware_deox import AL_TO_O_MASS_RATIO

    df = generate_synthetic_deox_calibration_dataset(n_heats=n)
    records = []
    for _, row in df.iterrows():
        o_init = float(row["o_a_initial_ppm"])
        o_after = 5.0
        mass = float(row["steel_mass_ton"])
        eta = float(row["eta_al_effective"])
        o_removed = (o_init - o_after) / 1e6 * mass * 1000
        al_added = o_removed * AL_TO_O_MASS_RATIO / eta if eta > 0 else 0
        records.append(
            HeatRecord(
                source="synthetic",
                plant_id=str(row["plant_id"]),
                steel_mass_ton=mass,
                o_a_initial_ppm=o_init,
                o_a_after_ppm=o_after,
                al_added_kg=al_added,
                method_id=str(row["method_id"]),
                eta_al_effective=eta,
                al_residual_pct=0.025,
                slag_feo_pct=float(row["slag_feo_pct"]),
                slag_mass_kg=float(row["slag_mass_kg"]),
                t_al_addition_c=float(row["t_al_addition_c"]),
            )
        )
    return records


@router.post(
    "/shadow/run",
    response_class=SafeJSONResponse,
    response_model=None,
)
def shadow_run(req: ShadowRunRequest) -> dict[str, Any]:
    """Run shadow validation synchronously. ``source=heats_db|synthetic``."""
    from dataclasses import asdict as _asdict

    from app.backend.eta_al_calibration import EtaAlCalibrator
    from app.backend.eta_al_predictor import EtaAlPredictor
    from app.backend.heat_records import count_heats, list_heats
    from app.backend.shadow_reporter import (
        _compute_savings_eur,
        generate_shadow_report,
    )
    from app.backend.shadow_stats import compute_shadow_stats
    from app.backend.shadow_validation import run_shadow_comparison

    predictor = EtaAlPredictor(calibrator=EtaAlCalibrator())
    is_synthetic = req.source == "synthetic"

    if is_synthetic:
        heats = _synthetic_heats_for_shadow(req.n_synthetic_heats)
    else:
        n = count_heats(plant_id=req.plant_id)
        if n > req.limit:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Плавок с outcome ({n}) > limit ({req.limit}). "
                    f"Используйте scripts/generate_shadow_report.py "
                    f"--source heats_db для полного прогона без лимита."
                ),
            )
        heats = list_heats(plant_id=req.plant_id, has_outcome=True, limit=req.limit)

    comparisons = run_shadow_comparison(heats, predictor)
    stats = compute_shadow_stats(
        comparisons, target_reduction_pct=req.target_reduction_pct
    )

    report_html = report_json = None
    if req.save_report:
        h, j = generate_shadow_report(
            comparisons,
            stats,
            al_price_eur_per_kg=req.al_price_eur_per_kg,
            is_synthetic=is_synthetic,
        )
        # Prefer repo-relative paths; fall back to absolute if outside the tree.
        try:
            report_html = str(h.relative_to(PROJECT_ROOT))
            report_json = str(j.relative_to(PROJECT_ROOT))
        except ValueError:
            report_html, report_json = str(h), str(j)

    compared = [
        _asdict(c) for c in comparisons if c.skip_reason is None
    ][:100]
    savings = _compute_savings_eur(stats, req.al_price_eur_per_kg)
    return _nan_to_none(
        {
            "stats": _asdict(stats),
            "savings_eur": savings,
            "al_price_eur_per_kg": req.al_price_eur_per_kg,
            "is_synthetic": is_synthetic,
            "n_total": stats.n_total,
            "n_compared": stats.n_compared,
            "report_html": report_html,
            "report_json": report_json,
            "compared_sample": compared,
        }
    )


@router.get(
    "/shadow/report",
    response_class=SafeJSONResponse,
    response_model=None,
)
def shadow_report(report_id: str | None = None) -> dict[str, Any]:
    """Return the latest persisted shadow report JSON (or one matching id)."""
    import json

    from app.backend import shadow_reporter

    reports_dir = shadow_reporter.REPORTS_DIR
    if not reports_dir.exists():
        raise HTTPException(status_code=404, detail="Нет shadow-отчётов")
    jsons = sorted(
        reports_dir.glob("shadow_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if report_id:
        jsons = [p for p in jsons if report_id in p.name]
    if not jsons:
        raise HTTPException(status_code=404, detail="Нет shadow-отчётов")
    latest = jsons[0]
    data = json.loads(latest.read_text(encoding="utf-8"))
    data["report_json_path"] = (
        str(latest.relative_to(PROJECT_ROOT))
        if PROJECT_ROOT in latest.parents
        else str(latest)
    )
    html_candidate = latest.with_suffix(".html")
    if html_candidate.exists():
        data["report_html_path"] = (
            str(html_candidate.relative_to(PROJECT_ROOT))
            if PROJECT_ROOT in html_candidate.parents
            else str(html_candidate)
        )
    return data
