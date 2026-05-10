"""Router for /api/train/* — XGBoost + Optuna training pipeline.

PR 8 of the Streamlit→FastAPI migration. See
``docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md``
(Endpoint map → Tab «Обучение»).

Streamlit parity reference: ``app/frontend/app.py`` lines 509-680
(``with tab_train:``). The flow:

    class_id → generate dataset → compute_features_for_class
    → train_model (XGBoost + Optuna + quantile regression + OOD GMM)
    → run_all_patterns(Phase.TRAINING) → make_llm_critic.review_training
    → return {version, metrics, feature_importance, critic, duration_s}

Endpoints:
- ``POST /api/train/run``  — submit training as a job; returns ``{job_id}``.
                              Frontend polls ``GET /api/jobs/{id}``
                              (PR 6) and reads ``result`` once status
                              flips to ``done``.

Cooperative cancellation: ``train_model`` accepts a
``cancellation_callback`` kwarg (added in PR 8 review fix) — a
zero-arg predicate we wire to ``_check_cancelled(progress)``. Optuna
calls our ``_maybe_stop`` callback after every trial; if the flag is
set, ``study.stop()`` exits the loop before the next trial. End-to-end
latency from DELETE → status=error is bounded by the duration of one
Optuna trial (~1-3 s on the smoke dataset, ~10-20 s on real Agrawal).

Coarse-grained progress milestones inside ``_run_train_job`` complement
the per-trial check:

    0.05 → "Генерация датасета"
    0.15 → "Feature engineering"
    0.25 → "XGBoost + Optuna trials"
    0.85 → "Pattern Library checks"
    0.95 → "LLM-Critic review"
    1.00 → done

The cancellation flag is read between coarse stages AND per Optuna
trial — DELETE during dataset generation aborts within ms, DELETE
mid-Optuna aborts within one trial duration.

Risks #3 (SafeJSONResponse): the result dict carries numpy floats from
``feature_importance`` and ``training_ranges``. ``response_class=
SafeJSONResponse, response_model=None`` is set on every endpoint.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.jobs import run_as_job
from app.api.llm_gating import make_progress_cancel_check
from app.api.responses import SafeJSONResponse
from app.backend.steel_classes import (
    AVAILABLE_CLASS_IDS,
    compute_features_for_class,
    get_synthetic_generator,
    load_steel_class,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Default sample sizes mirror Streamlit (lines 580-595): the synthetic
# generator default is 2500 but UI offers a slider; n_trials default is
# 40 in Streamlit but we use 30 here — the migration trades a slightly
# faster default run for the same result quality at MVP scope.
DEFAULT_N_SAMPLES = 800
DEFAULT_N_TRIALS = 30
DEFAULT_SEED = 42

# Pydantic Literal forces the class enum at the schema layer, so 422
# fires before the worker is scheduled. AVAILABLE_CLASS_IDS is the
# single source of truth — adding a new class to that list automatically
# makes it accepted here.
ClassIdLiteral = Literal["pipe_hsla", "en10083_qt", "fatigue_carbon_steel"]


# ──────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────


class TrainRunRequest(BaseModel):
    """Request body for ``POST /api/train/run``.

    ``n_samples`` is silently ignored for steel classes whose synthetic
    generator is data-bound (e.g. fatigue_carbon_steel uses the fixed
    437-record Agrawal NIMS parquet). We still validate the bounds so a
    malformed payload fails fast with 422.

    ``n_trials`` mirrors Streamlit slider 10-150 with 5-200 here to
    accommodate quick smoke tests (n_trials=5 in test_api_train.py)
    without bumping CI duration.
    """

    class_id: ClassIdLiteral = Field(
        default="pipe_hsla",
        description="Steel class profile id; one of AVAILABLE_CLASS_IDS",
    )
    n_samples: int = Field(
        default=DEFAULT_N_SAMPLES, ge=100, le=5000,
        description=(
            "Synthetic dataset size; ignored for classes whose generator "
            "loads a fixed real-data parquet (e.g. fatigue_carbon_steel)."
        ),
    )
    n_trials: int = Field(
        default=DEFAULT_N_TRIALS, ge=5, le=200,
        description="Optuna trials count; controls hyperparameter search budget",
    )
    seed: int = Field(
        default=DEFAULT_SEED, ge=0, le=2**31 - 1,
        description="Random seed for reproducibility",
    )

    model_config = {
        # Avoid pydantic v2's reserved-namespace warning if a field name
        # ever starts with ``model_``. Cheap, harmless.
        "protected_namespaces": (),
    }


# ──────────────────────────────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────────────────────────────


# PR 10 deduplication: previously a per-router copy of the closure-walk
# cancellation peek. Now delegates to the shared helper in
# ``app.api.llm_gating`` (extracted alongside the LLM 503-gating helper
# during PR 10 hypotheses migration). The wrapper preserves the call
# site shape (``_check_cancelled(progress) -> bool``) so the per-trial
# Optuna hook and the coarse stage gates stay untouched.
def _check_cancelled(progress: Any) -> bool:
    """Back-compat wrapper around :func:`make_progress_cancel_check`.

    The shared helper returns a *callable*; for the per-call style this
    file uses (each Optuna trial reads the flag once), we resolve once
    and invoke immediately. Keeping the wrapper means the worker body
    and the partial captured in ``cancellation_callback=`` continue to
    work unchanged.
    """
    return make_progress_cancel_check(progress)()


def _run_train_job(
    *,
    class_id: str,
    n_samples: int,
    n_trials: int,
    seed: int,
    progress: Any = None,
) -> dict[str, Any]:
    """Background worker — invoked through ``run_as_job``.

    See module docstring for the progress milestone schedule. ``progress``
    is auto-injected by JobStore when the function declares the keyword.
    Callers (the POST endpoint below) MUST NOT pass ``progress=...``
    explicitly; the store's introspection handles it.
    """
    started = time.monotonic()

    # Local imports keep heavy ML deps off the FastAPI cold-start path.
    # train_model pulls in xgboost + optuna which add ~600 ms per import.
    from app.backend.critic_llm import make_llm_critic
    from app.backend.model_trainer import train_model
    from pattern_library.patterns import Phase, run_all_patterns

    if callable(progress):
        progress(0.02, "Загрузка профиля стали")

    profile = load_steel_class(class_id)
    target = profile.target_properties[0].id

    # ── 1. Dataset generation ─────────────────────────────────────────
    if _check_cancelled(progress):
        raise RuntimeError("Cancelled by user (before dataset generation)")
    if callable(progress):
        progress(0.05, "Генерация датасета")

    gen = get_synthetic_generator(profile.synthetic_generator_name)
    # Real Agrawal loader honours n_samples too (it slices), so we pass
    # both kwargs unconditionally. Synthetic generators ignore unknown
    # kwargs because the project's signatures are positional-default —
    # we use kwargs to be explicit. ``random_seed`` keyword matches the
    # generator signatures (data_curator.py).
    try:
        df_raw = gen(n_samples=n_samples, random_seed=seed)
    except TypeError:
        # Defensive: a generator with a different signature falls back
        # to default args. This shouldn't happen with the current
        # registry but keeps the worker resilient to refactors.
        df_raw = gen()

    # ── 2. Feature engineering ────────────────────────────────────────
    if _check_cancelled(progress):
        raise RuntimeError("Cancelled by user (before feature engineering)")
    if callable(progress):
        progress(0.15, "Feature engineering")

    df_feat = compute_features_for_class(df_raw, class_id)
    feature_list = [f for f in profile.feature_set if f in df_feat.columns]

    # ── 3. Training (XGBoost + Optuna + quantile + GMM) ───────────────
    if _check_cancelled(progress):
        raise RuntimeError("Cancelled by user (before training)")
    if callable(progress):
        progress(
            0.25,
            f"XGBoost + Optuna ({n_trials} trials) — это занимает 1-5 минут",
        )

    trained = train_model(
        df_feat,
        target=target,
        feature_list=feature_list,
        n_optuna_trials=n_trials,
        random_seed=seed,
        steel_class=class_id,
        # Per-trial cooperative cancellation. ``_check_cancelled`` reads
        # ``progress.__closure__`` → JobStore.cancellation_requested. If
        # progress is None (no JobStore) the predicate is None → no
        # callback registered → identical behaviour to the legacy path.
        cancellation_callback=(
            (lambda: _check_cancelled(progress)) if progress is not None else None
        ),
    )

    # ── 4. Pattern Library critic ─────────────────────────────────────
    if callable(progress):
        progress(0.85, "Pattern Library: training-phase checks")

    critic_ctx = {
        "r2_train": trained.metrics.r2_train,
        "r2_val": trained.metrics.r2_val,
        "r2_test": trained.metrics.r2_test,
        "mae_test": trained.metrics.mae_test,
        "rmse_test": trained.metrics.rmse_test,
        "coverage_90_ci": trained.metrics.coverage_90_ci,
        "n_train": trained.metrics.n_train,
        "n_val": trained.metrics.n_val,
        "n_test": trained.metrics.n_test,
        "prediction_has_ci": True,
        "has_time_column": True,
        "has_groups": True,
        "split_strategy": "time_based",
        "cv_strategy": "group_kfold",
        "feature_importance": trained.feature_importance,
        "training_ranges": trained.training_ranges,
        "steel_class": class_id,
        "expected_top_features": profile.expected_top_features,
        "physical_bounds": profile.physical_bounds,
        "ood_detector_configured": True,
        "target": target,
    }
    pattern_warnings = run_all_patterns(critic_ctx, phase=Phase.TRAINING)

    # ── 5. LLM-Critic (optional) ──────────────────────────────────────
    llm_observations: list[dict[str, Any]] | None
    llm_critic = make_llm_critic()
    if llm_critic is None:
        # No ANTHROPIC_API_KEY → silent skip (matches Streamlit). UI
        # renders "LLM-Critic disabled" only when this is None vs []
        # to disambiguate "not configured" from "configured, no findings".
        llm_observations = None
    else:
        if callable(progress):
            progress(0.95, "LLM-Critic review")
        try:
            obs = llm_critic.review_training(critic_ctx)
            llm_observations = [asdict(o) for o in obs]
        except Exception as exc:  # noqa: BLE001 — defensive
            # LLM failures must not fail the whole training — surface
            # to logs and continue with empty observations. This mirrors
            # the LLMCritic.review_training internal try/except.
            logger.warning("LLM-Critic raised in worker: %s", exc)
            llm_observations = []

    # ── 6. Final result ───────────────────────────────────────────────
    if callable(progress):
        progress(1.0, "Готово")

    duration_s = round(time.monotonic() - started, 2)

    return {
        "version": trained.version,
        "class_id": class_id,
        "target": target,
        "target_label": profile.target_properties[0].label,
        "feature_list": list(feature_list),
        "metrics": asdict(trained.metrics),
        "feature_importance": [
            {"feature": k, "importance": float(v)}
            for k, v in sorted(
                trained.feature_importance.items(),
                key=lambda kv: -kv[1],
            )
        ],
        "training_ranges": trained.training_ranges,
        "critic": {
            "pattern_warnings": pattern_warnings,
            "llm_observations": llm_observations,
        },
        "duration_s": duration_s,
        "config": {
            "n_samples": n_samples,
            "n_trials": n_trials,
            "seed": seed,
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Endpoint
# ──────────────────────────────────────────────────────────────────────


@router.post(
    "/run",
    response_class=SafeJSONResponse,
    response_model=None,
)
def run_train(req: TrainRunRequest) -> dict[str, Any]:
    """Submit a training job and return the job id.

    Validation order:
        1) Pydantic Literal/Field bounds — class_id ∈ AVAILABLE_CLASS_IDS,
           n_samples ∈ [100, 5000], n_trials ∈ [5, 200]. 422 on miss.
        2) Submit job to the singleton store. Return ``{job_id}``.

    The actual training takes ~1-5 minutes (Optuna + XGBoost). The
    frontend polls ``GET /api/jobs/{job_id}`` (PR 6) until status flips
    to ``done``, then renders the result. ``DELETE /api/jobs/{job_id}``
    sets the cooperative cancellation flag, and ``train_model`` checks it
    after every Optuna trial (see module docstring).
    """
    # Defensive: re-check class is registered. The Literal would have
    # rejected anything else with 422, so this is guard-against-future-
    # changes belt-and-suspenders. ``not in`` would fail-fast at runtime
    # if someone adds a class to the Literal but forgets to register it.
    if req.class_id not in AVAILABLE_CLASS_IDS:
        # Mirror the design router's 400 shape — the user can correct
        # the body and retry without the API confusion of a 500.
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail=(
                f"class_id='{req.class_id}' not in AVAILABLE_CLASS_IDS "
                f"({AVAILABLE_CLASS_IDS}). Update steel_classes registry "
                f"before retraining."
            ),
        )

    job_id = run_as_job(
        _run_train_job,
        class_id=req.class_id,
        n_samples=int(req.n_samples),
        n_trials=int(req.n_trials),
        seed=int(req.seed),
    )

    return {
        "job_id": job_id,
        "config": {
            "class_id": req.class_id,
            "n_samples": int(req.n_samples),
            "n_trials": int(req.n_trials),
            "seed": int(req.seed),
        },
    }
