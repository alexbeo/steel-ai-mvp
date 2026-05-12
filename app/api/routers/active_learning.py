"""Router for /api/active-learning/* — cost-weighted EI next experiments.

PR 5 of the Streamlit→FastAPI migration. See
``docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md``
(Endpoint map → Tab «Следующие эксперименты»).

Only ``fatigue_carbon_steel`` is supported on this endpoint — the
active_learner runner expects the Agrawal dataset and the Q&T/HSLA
classes don't have a matching real dataset wired in yet (see
``scripts/propose_next_experiments.py``).

Endpoints:
- ``POST /api/active-learning/propose`` — synchronous LHS scan + EI ranking.
  Operation is cheap (~150 ms - 3 s for n_samples ≤ 5000), so polling
  infrastructure (PR 6) is intentionally not used.

Pattern: thin wrapper around ``app.backend.active_learner.propose_next_experiments``
+ ``app.backend.cost_model.seed_snapshot`` + ``compute_cost``. We do not
duplicate the LHS / EI math — only validation, model loading, baseline
computation, and serialisation.

Risks #3 (SafeJSONResponse): every endpoint declares
``response_class=SafeJSONResponse, response_model=None`` so the
``ExperimentProposal`` dataclass + numpy floats serialise via the custom
encoder instead of Pydantic v2's default revalidation.

Decision Log: successful runs are persisted with tag ``active_learning``
so the History tab shows them.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.responses import SafeJSONResponse
from app.api.routers import system as system_router
from app.backend.active_learner import propose_next_experiments
from app.backend.cost_model import compute_cost, seed_snapshot
from app.backend.data_curator import load_real_agrawal_fatigue_dataset
from app.backend.model_trainer import load_model, predict_with_uncertainty
from decision_log.logger import log_decision

logger = logging.getLogger(__name__)

router = APIRouter()


# Decision-vars catalogue mirrors ``scripts/propose_next_experiments.py``
# (DESIGN_COMPOSITION + DESIGN_PROCESS). Kept module-level so tests can
# monkeypatch if they want to vary the design space.
DESIGN_COMPOSITION = ["si_pct", "mn_pct", "ni_pct", "cr_pct", "cu_pct", "mo_pct"]
DESIGN_PROCESS = [
    "normalizing_temp_c",
    "carburizing_temp_c",
    "carburizing_time_min",
    "tempering_temp_c",
    "tempering_time_min",
    "through_hardening_cooling_rate_c_per_s",
]

# Hard-coded class constraint: the active_learner pipeline assumes the
# Agrawal NIMS dataset is available and decision_vars match Agrawal's
# feature space; other classes (HSLA, Q&T) need their own baseline-row +
# dataset wiring before the same endpoint can serve them.
SUPPORTED_STEEL_CLASS = "fatigue_carbon_steel"


# ──────────────────────────────────────────────────────────────────────
# Request schema
# ──────────────────────────────────────────────────────────────────────


class ProposeRequest(BaseModel):
    """Request body for ``POST /api/active-learning/propose``.

    ``n_samples`` is the LHS coverage of the feasible space — increase
    for higher precision, decrease for quick-look. The upper bound
    (10000) protects against accidental DoS via huge sample counts.

    ``top_k`` is the number of ranked candidates returned. We allow up
    to 50 to leave headroom for batch experiment planning use-cases
    without sacrificing the cheap-run promise.

    ``seed`` is forwarded as-is to LHS so reproducibility is exact;
    ``np.random.default_rng(seed)`` accepts any non-negative int.
    """

    model_version: str = Field(
        ..., description="Version directory name under models/ (must be fatigue_carbon_steel)"
    )
    n_samples: int = Field(
        default=2000, ge=100, le=10000,
        description="LHS scan size.",
    )
    top_k: int = Field(
        default=5, ge=1, le=50,
        description="Top-K ranked candidates to return.",
    )
    seed: int = Field(
        default=42, ge=0, le=2**31 - 1,
        description="LHS seed — reproducibility guarantee.",
    )

    model_config = {
        # Pydantic v2 reserves ``model_*`` for internal config; turn off the
        # warning so the field name aligns with the rest of the API.
        "protected_namespaces": (),
    }


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _baseline_row(df_raw: pd.DataFrame) -> pd.Series:
    """Prefer the carbon_low_alloy subset for the baseline.

    If the sub_class column exists and the carbon_low_alloy partition
    has at least 50 rows, take the median there. Otherwise fall back to
    the median over the full dataset. The 50-row threshold matches the
    runner script (``scripts/propose_next_experiments.py:74``); we keep
    it identical to avoid divergent "baseline" semantics between CLI and
    API.
    """
    if "sub_class" in df_raw.columns:
        sub = df_raw[df_raw["sub_class"] == "carbon_low_alloy"]
        if len(sub) < 50:
            sub = df_raw
    else:
        sub = df_raw
    return sub.select_dtypes(include=[np.number]).median()


def _decision_vars(feature_list: list[str], training_ranges: dict[str, Any]) -> list[str]:
    """Subset of design vars present in both feature_list and training_ranges.

    Mirrors ``scripts/propose_next_experiments.py:94-97``. We restrict
    to vars that have a training_ranges entry so the LHS bounds are
    well-defined; vars missing from training_ranges would break
    ``bounds[v]`` lookup later.
    """
    return [
        v for v in DESIGN_COMPOSITION + DESIGN_PROCESS
        if v in feature_list and v in training_ranges
    ]


def _proposal_to_dict(p: Any) -> dict[str, Any]:
    """Project an ExperimentProposal dataclass to a stable response shape.

    We keep the existing dataclass field names (acquisition_score,
    expected_improvement, …) since the decision_log consumes that shape
    and changing names here would silently break history-replay
    rendering. ``rank`` is added at the router level — it's a
    presentation concern, not part of the dataclass contract.
    """
    base = asdict(p)
    return {
        "id": base["id"],
        "composition": base["composition"],
        "process_params": base["process_params"],
        "predicted_property": base["predicted_property"],
        "lower_90": base["lower_90"],
        "upper_90": base["upper_90"],
        "uncertainty_width": base["uncertainty_width"],
        "is_ood": base["ood_flag"],
        # Keep the dataclass key too so existing code that reads
        # ``proposal.ood_flag`` from the JSON history (decision_log
        # replay) still works. Both flags reflect the same boolean.
        "ood_flag": base["ood_flag"],
        "estimated_cost_eur_per_t": base["cost_per_ton"],
        "expected_improvement": base["expected_improvement"],
        "acquisition_score": base["acquisition_score"],
        "delta_vs_baseline_property": base["delta_vs_baseline_property"],
        "delta_vs_baseline_cost": base["delta_vs_baseline_cost"],
        "tags": list(base.get("tags") or []),
    }


# ──────────────────────────────────────────────────────────────────────
# Endpoint
# ──────────────────────────────────────────────────────────────────────


@router.post(
    "/propose",
    response_class=SafeJSONResponse,
    response_model=None,
)
def propose(req: ProposeRequest) -> dict[str, Any]:
    """LHS scan + EI ranking → top-K next-experiment candidates.

    Validation order mirrors PR 3 ``predict``:
        1) ``_safe_version_dir`` — path traversal guard (CWE-22). 400.
        2) Directory exists + meta.json readable. 404.
        3) Steel class is fatigue_carbon_steel. 400.
        4) Dataset / model load + LHS + scoring. 200.

    Returns a structured response with ranked candidates, the baseline
    used to compute deltas, model echo (normalised version + steel class),
    and ``computation_time_s`` so the UI can surface "took N seconds" UX.
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
    # Refuse to run on non-fatigue classes — surface as 400 since the API
    # is supposed to be explicit about what it doesn't support.
    steel_class = meta.get("steel_class") or system_router.LEGACY_STEEL_CLASS_FALLBACK
    if steel_class != SUPPORTED_STEEL_CLASS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Active learner currently supports only '{SUPPORTED_STEEL_CLASS}' "
                f"models (Agrawal NIMS dataset). Got steel_class='{steel_class}'."
            ),
        )

    # -------- 3. Load model + dataset + baseline -----------------------
    try:
        bundle = load_model(safe_version)
    except Exception as exc:
        logger.exception("load_model failed for %s", safe_version)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model artifacts: {exc}",
        ) from exc

    bundle_meta = bundle["meta"]
    feature_list = list(bundle_meta["feature_list"])
    target = bundle_meta["target"]
    training_ranges = bundle_meta.get("training_ranges") or {}

    try:
        df_raw = load_real_agrawal_fatigue_dataset()
    except Exception as exc:
        logger.exception("load_real_agrawal_fatigue_dataset failed")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load Agrawal dataset: {exc}",
        ) from exc

    baseline = _baseline_row(df_raw)
    f_star = float(df_raw[target].max())
    snapshot = seed_snapshot()

    # Baseline prediction + cost so candidates can report Δ vs baseline.
    x_base = pd.DataFrame(
        [[float(baseline[f]) for f in feature_list]],
        columns=feature_list,
    )
    base_pred = float(predict_with_uncertainty(bundle, x_base).iloc[0]["prediction"])
    base_comp = {k: float(v) for k, v in baseline.items() if k.endswith("_pct")}
    base_cost = float(
        compute_cost(base_comp, snapshot, mode="full").total_per_ton
    )

    decision_vars = _decision_vars(feature_list, training_ranges)
    if not decision_vars:
        # Sanity: a model without any LHS-eligible decision vars cannot
        # produce candidates. This shouldn't happen for fatigue_carbon_steel
        # in practice, but a synthetic legacy meta could trigger it.
        raise HTTPException(
            status_code=500,
            detail=(
                "No decision variables available for this model — "
                "training_ranges is missing entries for the design space."
            ),
        )
    bounds = {v: tuple(training_ranges[v]) for v in decision_vars}

    def _cost_fn(comp: dict[str, float]) -> float:
        return float(compute_cost(comp, snapshot, mode="full").total_per_ton)

    # -------- 4. LHS scan + ranking ------------------------------------
    t0 = time.perf_counter()
    proposals = propose_next_experiments(
        model_bundle=bundle,
        baseline_row=baseline,
        feature_list=feature_list,
        decision_vars=decision_vars,
        bounds=bounds,
        f_star=f_star,
        cost_fn=_cost_fn,
        baseline_cost=base_cost,
        baseline_property=base_pred,
        n_samples=int(req.n_samples),
        top_k=int(req.top_k),
        seed=int(req.seed),
    )
    elapsed = time.perf_counter() - t0

    candidate_rows: list[dict[str, Any]] = []
    for rank, p in enumerate(proposals, 1):
        row = _proposal_to_dict(p)
        row["rank"] = rank
        candidate_rows.append(row)

    # -------- 5. Decision Log ------------------------------------------
    # Best-effort: a logging failure must not turn a successful active-
    # learning run into a 500 to the user. Fire-and-forget by design.
    try:
        top1_score = (
            f"{proposals[0].acquisition_score:.4f}" if proposals else "none"
        )
        log_decision(
            phase="inverse_design",
            decision=f"ActiveLearner: top-{req.top_k} via API",
            reasoning=(
                f"model={safe_version}, n_samples={req.n_samples}, "
                f"f*={f_star:.0f}, top1 score={top1_score}"
            ),
            context={
                "model_version": safe_version,
                "baseline": {
                    "predicted_property": base_pred,
                    "cost_per_ton": base_cost,
                },
                "f_star": f_star,
                "n_samples": int(req.n_samples),
                "decision_vars": decision_vars,
                "proposals": [asdict(p) for p in proposals],
            },
            author="api",
            tags=["active_learning"],
        )
    except Exception:  # pragma: no cover — best-effort
        logger.exception("log_decision failed for active-learning run")

    return {
        "candidates": candidate_rows,
        "baseline": {
            "predicted_property": base_pred,
            "cost_per_ton": base_cost,
            "f_star": f_star,
        },
        "model": {
            "version": safe_version,
            "steel_class": steel_class,
            "target": target,
        },
        "config": {
            "n_samples": int(req.n_samples),
            "top_k": int(req.top_k),
            "seed": int(req.seed),
            "decision_vars": decision_vars,
        },
        "computation_time_s": float(elapsed),
    }
