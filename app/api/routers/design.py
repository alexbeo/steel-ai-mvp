"""Router for /api/design/* — NSGA-II inverse alloy design.

PR 7 of the Streamlit→FastAPI migration. See
``docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md``
(Endpoint map → Tab «Дизайн сплава»).

Flow:

    composition bounds → NSGA-II (3 obj: distance/cost/uncertainty)
    → validator.validate_batch → Pareto front response

Endpoints:
- ``POST /api/design/run``     — submit a design job; returns ``{job_id}``.
                                 Frontend polls ``GET /api/jobs/{id}``
                                 (PR 6) and reads ``result`` once status
                                 flips to ``done``.

NOTE: there is no separate ``/api/design/result/{job_id}`` — we route
through the generic ``/api/jobs/{id}`` endpoint as the spec recommends
(DRY). The result payload exposed there is the dict returned by
``_run_design_job`` below, which is already shaped for the UI.

PR 7 — class gate: only ``pipe_hsla`` models accepted. Other classes
return 400 with a clear message; the UI surfaces the same restriction
as an info banner upstream.

Risks #3 (SafeJSONResponse): the result dict carries numpy floats from
NSGA-II / predict_with_uncertainty. ``response_class=SafeJSONResponse,
response_model=None`` is set on every endpoint.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.api.jobs import run_as_job
from app.api.responses import SafeJSONResponse
from app.api.routers import system as system_router
from app.backend.cost_model import (
    PriceSnapshot,
    PriceSnapshotIncomplete,
    seed_snapshot,
)
from app.backend.inverse_designer import (
    VARIABLE_BOUNDS_HSLA,
    run_inverse_design,
)
from app.backend.validator import validate_batch

logger = logging.getLogger(__name__)

router = APIRouter()

# Inverse design only runs for HSLA classes in this iteration. Other
# classes (Q&T, fatigue) lack process-parameter bounds and would crash
# pymoo on missing columns.
SUPPORTED_STEEL_CLASS = "pipe_hsla"

# Default knob values for the request schema. The slider/number input
# ranges are clamped here as Pydantic validators so the UI can show
# identical ranges and the API enforces them at the boundary too.
DEFAULT_TARGET_MIN_MPA = 485.0
DEFAULT_TARGET_MAX_MPA = 580.0
DEFAULT_CEV_MAX = 0.43
DEFAULT_PCM_MAX = 0.22
DEFAULT_POPULATION_SIZE = 80
DEFAULT_N_GENERATIONS = 60


# ──────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────


class TargetRange(BaseModel):
    """Min/max envelope for a target property (MPa for HSLA yield strength)."""

    min: float = Field(..., gt=0, description="Lower bound, inclusive")
    max: float = Field(..., gt=0, description="Upper bound, inclusive")

    @model_validator(mode="after")
    def _check_order(self) -> TargetRange:
        if self.min >= self.max:
            raise ValueError("target.min must be < target.max")
        return self


class HardConstraints(BaseModel):
    """Optional CEV/Pcm caps. None → use defaults (DEFAULT_CEV_MAX /
    DEFAULT_PCM_MAX). Empty dict {} also means defaults; omit the
    constraint to drop it entirely (rare — the UI always sends both).
    """

    cev_iiw_max: float | None = Field(
        default=None, ge=0.30, le=0.60, description="CEV(IIW) hard cap"
    )
    pcm_max: float | None = Field(
        default=None, ge=0.15, le=0.35, description="Pcm hard cap"
    )


class DesignRunRequest(BaseModel):
    """Request body for ``POST /api/design/run``.

    We accept ``price_snapshot`` as an inline dict (same shape as
    ``GET /api/prices/active`` response) — pass ``None`` (or omit) to
    fall back to ``seed_snapshot()``.

    ``include_cost`` toggles the cost objective: True → use snapshot
    (or seed), False → run with ``price_snapshot=None`` (mirrors the
    UI's "Учитывать стоимость" checkbox).

    Note: NSGA-II is a true multi-objective optimiser — the spec's
    ``cost_weight`` knob is intentionally NOT exposed because the
    backend uses 3 separate objectives (distance/cost/uncertainty) and
    Pareto-ranks them. Trade-off is consumed by the user via candidate
    selection on the chart, not by a scalar weight at submit time.

    Baseline semantics: ``response.baseline`` and ``candidate.delta_*_vs_baseline``
    are computed against the **median of the produced Pareto front**
    (a self-reference, NOT a separate reference recipe). Deltas thus
    answer "is this candidate above or below the cluster?" — see the
    longer explanation in ``_run_design_job``. Field names are kept as
    ``baseline``/``delta_*_vs_baseline`` to preserve a stable contract
    with the UI/CSV export consumers.
    """

    model_version: str = Field(..., description="Version directory under models/")
    target: TargetRange = Field(
        default_factory=lambda: TargetRange(
            min=DEFAULT_TARGET_MIN_MPA, max=DEFAULT_TARGET_MAX_MPA
        ),
        description="Yield strength target range, MPa",
    )
    hard_constraints: HardConstraints = Field(
        default_factory=HardConstraints,
        description="Optional CEV/Pcm caps; defaults applied when fields are null",
    )
    population_size: int = Field(
        default=DEFAULT_POPULATION_SIZE, ge=10, le=200,
        description="NSGA-II population size",
    )
    n_generations: int = Field(
        default=DEFAULT_N_GENERATIONS, ge=1, le=200,
        description="NSGA-II generations",
    )
    include_cost: bool = Field(
        default=True,
        description="Use the price snapshot as cost objective; False → uncertainty+distance only",
    )
    cost_mode: str = Field(
        default="full",
        description="full = scrap base + alloys; incremental = alloys only",
    )
    price_snapshot: dict[str, Any] | None = Field(
        default=None,
        description="Inline snapshot dict; if null, seed_snapshot() is used",
    )

    model_config = {
        # ``model_*`` is reserved by Pydantic v2 for internal config; turn
        # off the warning so the field name stays a stable kwarg.
        "protected_namespaces": (),
    }

    @model_validator(mode="after")
    def _check_cost_mode(self) -> DesignRunRequest:
        if self.cost_mode not in ("full", "incremental"):
            raise ValueError("cost_mode must be 'full' or 'incremental'")
        return self


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _resolve_price_snapshot(req: DesignRunRequest) -> PriceSnapshot | None:
    """Build a PriceSnapshot from the request, or fall back to seed.

    Returns None when the user disabled cost optimisation; the
    ``run_inverse_design`` path then uses the legacy element-weighted
    cost stub which is acceptable for "no real cost data" runs.
    """
    if not req.include_cost:
        return None
    if req.price_snapshot is None:
        return seed_snapshot()
    # Reuse the prices.upload validator so an inline snapshot goes through
    # the same domain checks.
    from app.api.routers.prices import (
        SnapshotUploadRequest,
        _build_snapshot_from_request,
    )

    try:
        upload = SnapshotUploadRequest.model_validate(req.price_snapshot)
        return _build_snapshot_from_request(upload)
    except (ValueError, ValidationError) as exc:
        # Surface as 400 — the snapshot is part of the request body.
        raise HTTPException(
            status_code=400,
            detail=f"Invalid price_snapshot: {exc}",
        ) from exc


def _hard_constraints_dict(hc: HardConstraints) -> dict[str, dict[str, float]]:
    """Map HardConstraints schema to the dict shape run_inverse_design expects.

    The UI always sends both bounds; we mirror that by applying defaults
    when the user omitted a field. Use of None as "default please" rather
    than 0.0 keeps the Pydantic ge/le validators meaningful.
    """
    out: dict[str, dict[str, float]] = {}
    out["cev_iiw"] = {"max": hc.cev_iiw_max if hc.cev_iiw_max is not None else DEFAULT_CEV_MAX}
    out["pcm"] = {"max": hc.pcm_max if hc.pcm_max is not None else DEFAULT_PCM_MAX}
    return out


def _validate_one_to_dict(validation: dict[str, Any] | None) -> dict[str, Any] | None:
    """Project the dict embedded in validate_batch output to a stable shape."""
    if not validation:
        return None
    return {
        "overall": validation.get("overall"),
        "n_passed": int(validation.get("n_passed", 0)),
        "n_failed": int(validation.get("n_failed", 0)),
        "n_warnings": int(validation.get("n_warnings", 0)),
        "failed_checks": list(validation.get("failed_checks") or []),
        "warnings": list(validation.get("warnings") or []),
    }


def _project_candidate(
    cand: dict[str, Any],
    rank: int,
    cev_max: float,
    pcm_max: float,
    baseline_property: float | None,
    baseline_cost: float | None,
) -> dict[str, Any]:
    """Project a backend candidate dict to the spec-friendly UI shape.

    Backend shape (run_inverse_design):
        {idx, composition, processing,
         predicted: {mean, lower_90, upper_90, ci_half_width, ood_flag},
         derived: {cev_iiw, pcm, cen, microalloying_sum},
         objectives: {distance_to_target, alloying_cost, prediction_uncertainty},
         cost: {total_per_ton, contributions, mode, currency} | None,
         validation: {overall, n_passed, ...} (added by validate_batch)}

    UI / spec shape (this function):
        rank, idx, composition, processing, derived, predicted_mean/q05/q95,
        cost_eur_per_t, currency, in_spec, is_ood,
        delta_property_vs_baseline, delta_cost_vs_baseline,
        breakdown[], validation, objectives.
    ``in_spec`` is computed here from the validator overall + derived
    constraints (CEV/Pcm caps already applied by NSGA-II, but we double-
    check for the chart marker color).
    """
    pred = cand.get("predicted") or {}
    derived = cand.get("derived") or {}
    cost = cand.get("cost")
    validation = cand.get("validation") or {}

    mean = float(pred.get("mean", 0.0))
    q05 = float(pred.get("lower_90", mean))
    q95 = float(pred.get("upper_90", mean))
    ci_half = float(pred.get("ci_half_width", (q95 - q05) / 2.0))
    ood = bool(pred.get("ood_flag", False))

    # Cost projection: prefer the breakdown total_per_ton (real ferroalloy
    # math) over objectives.alloying_cost (legacy element-weighted stub).
    if cost:
        cost_eur_per_t = float(cost.get("total_per_ton", 0.0))
        currency = cost.get("currency", "EUR")
        breakdown = list(cost.get("contributions") or [])
    else:
        cost_eur_per_t = float(
            (cand.get("objectives") or {}).get("alloying_cost", 0.0)
        )
        currency = "EUR (legacy)"
        breakdown = []

    in_spec = (
        validation.get("overall") in ("PASS", "PASS_WITH_WARNINGS")
        and derived.get("cev_iiw", 0.0) <= cev_max + 1e-6
        and derived.get("pcm", 0.0) <= pcm_max + 1e-6
        and not ood
    )

    delta_property = (
        mean - baseline_property if baseline_property is not None else None
    )
    delta_cost = (
        cost_eur_per_t - baseline_cost if baseline_cost is not None else None
    )

    return {
        "rank": rank,
        "idx": int(cand.get("idx", rank - 1)),
        "composition": dict(cand.get("composition") or {}),
        "processing": dict(cand.get("processing") or {}),
        "derived": dict(derived),
        "predicted_mean": mean,
        "predicted_q05": q05,
        "predicted_q95": q95,
        "ci_half_width": ci_half,
        "property": mean,  # alias the spec uses for chart Y-axis
        "cost_eur_per_t": cost_eur_per_t,
        "currency": currency,
        "in_spec": bool(in_spec),
        "is_ood": ood,
        "delta_property_vs_baseline": delta_property,
        "delta_cost_vs_baseline": delta_cost,
        "breakdown": breakdown,
        "objectives": dict(cand.get("objectives") or {}),
        "validation": _validate_one_to_dict(validation),
    }


def _run_design_job(
    *,
    model_version: str,
    targets: dict[str, dict[str, float]],
    hard_constraints: dict[str, dict[str, float]],
    population_size: int,
    n_generations: int,
    price_snapshot: PriceSnapshot | None,
    cost_mode: str,
    cev_max: float,
    pcm_max: float,
    progress: Any = None,
) -> dict[str, Any]:
    """Background worker — invoked through ``run_as_job``.

    ``progress`` is injected by the JobStore when the function declares
    the parameter (see jobs.py docstring). We emit four canonical
    milestones because pymoo's ``minimize`` is a single blocking call —
    no per-generation hook is plumbed yet, so we can't stream true
    progress. The four-stage UX is honest about the abstraction.
    """
    if callable(progress):
        progress(0.05, "Загрузка модели и проверка прайса")

    try:
        result = run_inverse_design(
            model_version=model_version,
            targets=targets,
            hard_constraints=hard_constraints,
            population_size=population_size,
            n_generations=n_generations,
            price_snapshot=price_snapshot,
            cost_mode=cost_mode,  # type: ignore[arg-type]
        )
    except PriceSnapshotIncomplete as exc:
        # Re-raise with a friendlier message — the worker's exception
        # path serialises this into Job.error which the UI surfaces in
        # a Russian banner. No HTTPException here: we're inside a
        # background thread, not a request scope.
        raise RuntimeError(
            f"В прайсе нет цен для: {', '.join(exc.missing)}. "
            f"Добавьте материалы и повторите запуск."
        ) from exc

    if callable(progress):
        progress(0.85, "Валидация Pareto-кандидатов")

    pareto = result.get("pareto_candidates") or []
    val_result = validate_batch(pareto)
    # validate_batch returns approved+rejected with validation injected;
    # rebuild a candidates list preserving NSGA-II ordering. We use the
    # union (approved + rejected) keyed by idx so chart markers can show
    # rejected candidates in red (we expose all for the chart even
    # though only approved Top-5 surface in the main result table).
    by_idx: dict[int, dict[str, Any]] = {}
    for c in val_result.get("approved", []):
        by_idx[int(c.get("idx", -1))] = c
    for c in val_result.get("rejected", []):
        by_idx[int(c.get("idx", -1))] = c
    enriched = []
    for orig in pareto:
        idx = int(orig.get("idx", -1))
        enriched.append(by_idx.get(idx, orig))

    # Compute "baseline" = MEDIAN OF THE PRODUCED PARETO FRONT for
    # delta computations. NB: this is intentionally a SELF-REFERENCE,
    # not an independent reference recipe — every candidate's delta is
    # measured against the cluster median of this very run, so the
    # values answer "is this candidate above or below the front's
    # midpoint?" rather than "is this better than some external
    # baseline alloy?". Field names are kept as ``baseline`` /
    # ``delta_*_vs_baseline`` because the API contract was already
    # exposed to UI/CSV-export consumers; rename was deliberately
    # deferred (see DesignRunRequest docstring + reviewer NIT #4 of
    # PR 7). If a future iteration adds a real external baseline (e.g.
    # the user's current production recipe), introduce a new field
    # alongside this one rather than overloading ``baseline``.
    baseline_property: float | None = None
    baseline_cost: float | None = None
    if enriched:
        prop_vals = [
            float((c.get("predicted") or {}).get("mean", 0.0))
            for c in enriched
        ]
        cost_vals = [
            float(
                (c.get("cost") or {}).get(
                    "total_per_ton",
                    (c.get("objectives") or {}).get("alloying_cost", 0.0),
                )
            )
            for c in enriched
        ]
        prop_vals.sort()
        cost_vals.sort()
        mid = len(prop_vals) // 2
        baseline_property = prop_vals[mid]
        baseline_cost = cost_vals[mid]

    candidates_out = [
        _project_candidate(
            c, rank=i + 1,
            cev_max=cev_max, pcm_max=pcm_max,
            baseline_property=baseline_property,
            baseline_cost=baseline_cost,
        )
        for i, c in enumerate(enriched)
    ]

    if callable(progress):
        progress(1.0, "Готово")

    snapshot_payload: dict[str, Any] | None = None
    if price_snapshot is not None:
        snapshot_payload = {
            "date": price_snapshot.date.isoformat() if price_snapshot.date else None,
            "currency": price_snapshot.currency,
            "source": price_snapshot.source,
            "n_materials": len(price_snapshot.materials),
        }

    return {
        "candidates": candidates_out,
        "n_candidates": result.get("n_candidates", len(candidates_out)),
        "n_approved": len(val_result.get("approved", [])),
        "n_rejected": len(val_result.get("rejected", [])),
        "rejection_summary": dict(val_result.get("rejection_summary") or {}),
        "baseline": {
            "predicted_property": baseline_property,
            "cost_per_ton": baseline_cost,
            "method": "median_of_pareto",
        },
        "model": {
            "version": model_version,
            "steel_class": SUPPORTED_STEEL_CLASS,
        },
        "config": {
            "targets": targets,
            "hard_constraints": hard_constraints,
            "population_size": population_size,
            "n_generations": n_generations,
            "cost_mode": cost_mode if price_snapshot is not None else "legacy",
            "include_cost": price_snapshot is not None,
        },
        "price_snapshot": snapshot_payload,
        "price_snapshot_path": result.get("price_snapshot_path"),
        "cost_currency": result.get("cost_currency"),
    }


# ──────────────────────────────────────────────────────────────────────
# Endpoint
# ──────────────────────────────────────────────────────────────────────


@router.post(
    "/run",
    response_class=SafeJSONResponse,
    response_model=None,
)
def run_design(req: DesignRunRequest) -> dict[str, Any]:
    """Submit an NSGA-II inverse design job.

    Validation order:
        1) ``_safe_version_dir`` — path traversal guard (CWE-22). 400.
        2) Directory exists + meta.json readable. 404.
        3) Steel class is pipe_hsla (this iteration's only supported
           class for inverse design). 400 with a friendly explanation.
        4) Price snapshot resolves cleanly (seed or inline). 400 on
           invalid inline shape.
        5) Submit job to the singleton store. Return ``{job_id}``.

    The actual NSGA-II run takes ~40 seconds, so we deliberately do
    NOT block the request. The frontend polls
    ``GET /api/jobs/{job_id}`` (PR 6) until status flips to ``done``,
    then renders the result.
    """
    # -------- 1. Locate model + meta -----------------------------------
    version_dir = system_router._safe_version_dir(req.model_version)
    safe_version = version_dir.name
    if not version_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Model version '{safe_version}' not found",
        )
    meta = system_router._read_model_meta(version_dir)
    if meta is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{safe_version}' is missing meta.json",
        )

    # -------- 2. Class gate --------------------------------------------
    steel_class = meta.get("steel_class") or system_router.LEGACY_STEEL_CLASS_FALLBACK
    if steel_class != SUPPORTED_STEEL_CLASS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Inverse design supported only for HSLA models in this "
                f"iteration. Got steel_class='{steel_class}'. Use «Прогноз» "
                f"tab for non-HSLA models."
            ),
        )

    # -------- 3. Resolve price snapshot --------------------------------
    snapshot = _resolve_price_snapshot(req)

    # -------- 4. Build NSGA-II inputs ----------------------------------
    targets = {
        # The model's target id drives the targets dict key — the backend
        # zip the prediction to ``meta.target`` (see HSLADesignProblem._evaluate).
        # For HSLA this is always yield_strength_mpa today.
        meta.get("target") or "yield_strength_mpa": {
            "min": float(req.target.min),
            "max": float(req.target.max),
        }
    }
    hard_constraints = _hard_constraints_dict(req.hard_constraints)
    cev_max = hard_constraints["cev_iiw"]["max"]
    pcm_max = hard_constraints["pcm"]["max"]

    # -------- 5. Submit job --------------------------------------------
    job_id = run_as_job(
        _run_design_job,
        model_version=safe_version,
        targets=targets,
        hard_constraints=hard_constraints,
        population_size=int(req.population_size),
        n_generations=int(req.n_generations),
        price_snapshot=snapshot,
        cost_mode=req.cost_mode,
        cev_max=cev_max,
        pcm_max=pcm_max,
    )

    # We deliberately do NOT use VARIABLE_BOUNDS_HSLA from the request —
    # the design space is defined by the backend module, the API just
    # picks targets/constraints/popsize. If a future PR introduces
    # bounds-relaxation, this is the place to surface it.
    return {
        "job_id": job_id,
        "model_version": safe_version,
        "steel_class": steel_class,
        # Echo the resolved config so the UI can pin "what was actually
        # submitted" alongside the polling state — useful when the user
        # navigates away mid-run and comes back.
        "config": {
            "targets": targets,
            "hard_constraints": hard_constraints,
            "population_size": int(req.population_size),
            "n_generations": int(req.n_generations),
            "cost_mode": req.cost_mode if snapshot is not None else "legacy",
            "include_cost": snapshot is not None,
            "variable_bounds": dict(VARIABLE_BOUNDS_HSLA),
        },
    }
