"""Router for /api/deox/* — Al deoxidation calculator (physics + AI).

PR 4 of the Streamlit→FastAPI migration introduced three sync endpoints
(forward / inverse / compare). PR 9 adds a fourth, asynchronous
endpoint — the AI advisor cycle, which mirrors Streamlit's ``sub_ai``
tab (``app/frontend/app.py`` lines 1221-1463). See
``docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md``.

Streamlit parity reference: ``app/frontend/app.py`` lines 920-1463
(``with tab_deox:`` block, sub_fwd / sub_inv / sub_cmp / sub_ai).

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
verdict. Streamlit shows them inline above the result; the JS view
renders them with severity-coloured banners.

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

import logging
import time
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.jobs import get_job_store, run_as_job
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


class AlAdvisoryComposition(BaseModel):
    """Optional composition snapshot — feeds the critic risk-detection.

    All fields default to None so the UI can submit a partial picture
    (e.g. only C / Mn). The advisor + critic prompts handle missing
    fields gracefully — they're tagged "optional" in the schema. Bounds
    mirror the Streamlit number_inputs (lines 1255-1264).
    """

    c_pct: float | None = Field(default=None, ge=0.0, le=1.5)
    mn_pct: float | None = Field(default=None, ge=0.0, le=3.0)
    si_pct: float | None = Field(default=None, ge=0.0, le=2.5)
    s_pct: float | None = Field(default=None, ge=0.0, le=0.05)
    p_pct: float | None = Field(default=None, ge=0.0, le=0.05)


class AlAdvisoryRequest(BaseModel):
    """AI-cycle input — full heat context for advisor + critic.

    The cycle is a single LLM round-trip per agent (~80 s advisor +
    ~80 s critic = ~3 min total). Field bounds mirror the Streamlit
    AI sub-tab (``app/frontend/app.py`` lines 1234-1268) plus an
    opt-in ``save_to_decision_log`` flag — Streamlit always saves;
    the API exposes the choice so automated/test callers can submit
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
            "Streamlit defaults to true; the API defaults to false to "
            "keep test/automation flows quiet."
        ),
    )

    model_config = {
        # ``thermo_model`` doesn't trip the protected-namespace warning
        # but consistency with the other request schemas — keep it off.
        "protected_namespaces": (),
    }


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


# ──────────────────────────────────────────────────────────────────────
# AI cycle (PR 9) — advisor + critic pair as a long-running job
# ──────────────────────────────────────────────────────────────────────


# TODO(jobs.py): consolidate _check_cancelled with train.py:_check_cancelled и deox.py:_check_cancelled
# (когда _make_progress_cb в jobs.py получит третий freevar — обе копии silently сломаются).
# Move в app/api/jobs.py как public helper get_job_cancellation_flag(progress) -> bool.
def _check_cancelled(progress: Any) -> bool:
    """Peek at the job's cancellation flag through the progress closure.

    Same trick as ``app/api/routers/train.py:_check_cancelled`` — the
    progress callback is a closure over ``job_id``, so we walk back to
    the JobStore to read ``cancellation_requested`` without threading
    the store through fn signatures. Duplicating the helper here (vs
    importing from train) keeps the deox router self-contained and
    avoids cross-router coupling for a 20-line utility.

    Returns True iff cancellation was requested. Returns False when
    progress is None (no JobStore wired) or the closure can't be
    introspected (silent fallback: at worst cancellation becomes a
    no-op until someone notices the missing flag).
    """
    if progress is None:
        return False
    try:
        code = getattr(progress, "__code__", None)
        if code is None or "job_id" not in code.co_freevars:
            return False
        idx = code.co_freevars.index("job_id")
        job_id = progress.__closure__[idx].cell_contents  # type: ignore[index]
    except (AttributeError, IndexError, TypeError):
        return False
    job = get_job_store().get(str(job_id))
    return bool(job and job.cancellation_requested)


def _build_heat_context(req: AlAdvisoryRequest) -> dict[str, Any]:
    """Project the request into the dict shape the advisor/critic expect.

    Mirrors Streamlit ``app/frontend/app.py`` line 1304-1319
    (``heat_context = {...}``). Composition fields with None values are
    dropped from the dict so the LLM payload doesn't carry "null" keys
    that the prompt isn't tuned for. ``mn_s_ratio`` is a derived feature
    the prompt explicitly references — compute it when both Mn and S
    are present, skip otherwise.
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
        # Mirror Streamlit's mn_s_ratio = mn / max(s, 1e-6) (line 1315).
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
    # model_id, so we just run all three (matches Streamlit line 1292).
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
    """Validate ANTHROPIC_API_KEY + advisor/critic prompts before submit.

    Failing fast at the endpoint rather than inside the worker means
    the UI sees a 503 banner ("AI не настроен") instead of a generic
    "job ended with error" — clearer remediation. The two checks:

    1. ``ANTHROPIC_API_KEY`` env var present (the SDK + factories
       defer-fail without it; we check up-front for clarity).
    2. Both prompt files (``deoxidation_advisor.md``,
       ``deoxidation_critic.md``) exist on disk. Public clones may
       be missing them — gitignored intellectual property.
    """
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM не настроен — ANTHROPIC_API_KEY отсутствует в окружении. "
                "Добавьте ключ в .env и перезапустите uvicorn."
            ),
        )

    # Prompt presence check — load_prompt is cached, so this is cheap
    # on the warm path and clear on the cold one.
    from app.backend.prompt_loader import PromptNotFoundError, load_prompt

    for name in ("deoxidation_advisor", "deoxidation_critic"):
        try:
            load_prompt(name)
        except PromptNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Prompt {name}.md не найден. Промпты — gitignored "
                    f"intellectual property; обратитесь к владельцу проекта. "
                    f"({exc})"
                ),
            ) from exc


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
