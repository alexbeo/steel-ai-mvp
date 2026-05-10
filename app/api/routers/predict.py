"""Router for /api/predict — single-composition property prediction +
anomaly explainer hookup.

PR 3 of the Streamlit→FastAPI migration introduced the prediction
endpoint with a placeholder OOD warning button. PR 12 (this revision)
wires the placeholder up to a real Sonnet-backed anomaly explainer
running as a long-running job. See
``docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md``
(Endpoint map → Tab «Прогноз»).

Flow:

    composition (dict) → DataFrame → compute_features_for_class()
    → load_model() → predict_with_uncertainty() → response
    [optional follow-up] → POST /predict/anomaly-explain → job_id
    → poll /api/jobs/{id} → diagnosis dict.

PR 12 — endpoint convention note:
We expose the explainer at ``/predict/anomaly-explain`` rather than
its own router because the call only makes sense as a *follow-up*
to a prediction (the request body echoes the predicted mean / q05 /
q95 / OOD flag the prediction endpoint just returned). It is a
*single-LLM* endpoint (one Sonnet call → diagnosis dict) but lives
under the predict scope, so we use the action-specific suffix
``/anomaly-explain`` rather than the bare ``/run`` reserved for
top-level paired-LLM cycles. This matches the convention spelled
out in the design doc:

    «Single-LLM endpoints (PR 10 hypotheses) — просто <scope>/run
    или <scope>/generate без ai-cycle суффикса. Action-specific
    suffix when the action is a follow-up to a prior endpoint and
    the suffix communicates intent better than ``/run``.»

PR 7 wide-CI heuristic (added with PR 12):
The UI auto-shows the explainer button when the prediction is OOD
*or* the 90% CI is wide relative to the target's normal range. The
API surfaces the same boolean as ``response.prediction.wide_ci`` so
the JS view can drive the button visibility without re-implementing
the heuristic. Threshold: ``ci_width > 0.5 * target_normal_range``
when the active class profile's target has a known range; falls
back to ``(q95-q05)/mean > 0.20`` (>20% relative half-width)
otherwise.

Cooperative cancellation (PR 12 anomaly-explain only):
``check_llm_ready`` fails fast at submit time on missing API key
or prompt; once the worker is in flight the singleton call to
Sonnet (~30-60 s) cannot be interrupted, but a DELETE before the
LLM call short-circuits via ``make_progress_cancel_check``. The UI
treats that as a regular cancel.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.jobs import run_as_job
from app.api.llm_gating import check_llm_ready, make_progress_cancel_check
from app.api.responses import SafeJSONResponse
from app.api.routers import system as system_router
from app.backend.model_trainer import load_model, predict_with_uncertainty
from app.backend.steel_classes import compute_features_for_class, load_steel_class

logger = logging.getLogger(__name__)

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────
# Wide-CI heuristic — driven by class profile + relative-width fallback
# ──────────────────────────────────────────────────────────────────────


# Fallback relative half-width threshold. Any profile target with a
# concrete ``range`` overrides this with ``ci_width / target_span > 0.5``.
# Pure relative half-width is only used when the profile's target
# metadata lacks a range — defensive default tuned so the explainer
# button doesn't flash on every prediction.
_WIDE_CI_RELATIVE_THRESHOLD = 0.20


def _compute_wide_ci_flag(
    *,
    mean: float,
    q05: float,
    q95: float,
    target: str | None,
    profile,
) -> tuple[bool, float]:
    """Return ``(is_wide, threshold_used)`` for the prediction CI.

    Primary path:
        target_range = profile.target_properties[target].range
        target_span = target_range[1] - target_range[0]
        wide_ci = (q95 - q05) > 0.5 * target_span

    When the profile lacks a matching target or the target span is
    non-positive (degenerate), falls back to ``(q95 - q05) / max(|mean|, 1e-9)
    > _WIDE_CI_RELATIVE_THRESHOLD``. The threshold returned is in the
    same units as the underlying comparison (absolute span on the
    primary path, the dimensionless ratio on the fallback path) so the
    UI can render an actionable tooltip if needed.
    """
    ci_width = float(q95) - float(q05)
    target_prop = next(
        (t for t in profile.target_properties if t.id == target),
        None,
    )
    if target_prop is not None and len(target_prop.range) == 2:
        lo, hi = float(target_prop.range[0]), float(target_prop.range[1])
        target_span = hi - lo
        if target_span > 0:
            threshold = 0.5 * target_span
            return ci_width > threshold, threshold
    # Fallback — relative half-width over absolute mean. Avoids divide
    # by zero on the rare zero-mean prediction (e.g. centred residual
    # targets in a future class).
    denom = abs(float(mean)) if abs(float(mean)) > 1e-9 else 1e-9
    rel = ci_width / denom
    return rel > _WIDE_CI_RELATIVE_THRESHOLD, _WIDE_CI_RELATIVE_THRESHOLD


class PredictRequest(BaseModel):
    """Request body for ``POST /api/predict``.

    ``composition`` keys must cover the model's class ``feature_set``
    exactly (order doesn't matter; extra keys are ignored). Missing
    keys → 400. The strict-coverage rule mirrors what the UI form
    submits (built from ``feature_set``) and makes that contract
    explicit at the API boundary.
    """

    model_version: str = Field(..., description="Version directory name under models/")
    composition: dict[str, float] = Field(
        ..., description="feature_name → numeric value (wt% or process units)"
    )

    model_config = {
        # ``model_*`` is reserved by Pydantic v2 for internal config; turn off the
        # warning so we can keep the field name aligned with the rest of the API.
        "protected_namespaces": (),
    }


@router.post(
    "/predict",
    response_class=SafeJSONResponse,
    response_model=None,
)
def predict(req: PredictRequest) -> dict[str, Any]:
    """Run point prediction with conformal-corrected 90% CI + OOD flag.

    Validation order:
        1) model exists → 404 if not
        2) class profile loadable → 500 if YAML missing (config error)
        3) composition covers feature_set → 400 if missing keys
        4) compute features + predict → 200

    Extra keys in ``composition`` (e.g. derived features the user
    pre-computed) are silently dropped — backend recomputes them via
    ``compute_features_for_class`` to ensure they match training-time
    formulas exactly.

    Response shape (PR 12 additions called out):
        prediction.mean / q05 / q95 / ci_half_width — primary numerics
        prediction.lower_p / upper_p — **DEPRECATED** alias of q05/q95.
            Kept for backward compat with PR 3 frontend builds; planned
            removal once views/predict.js is cut over to q05/q95
            exclusively (tracked in design-doc backlog).
        prediction.wide_ci / wide_ci_threshold — PR 12: surfaces the
            "show anomaly explainer" trigger flag described in PR 7
            backlog. ``wide_ci=True`` when the prediction's CI is wide
            relative to the target's normal range; UI shows the
            explainer button when ``ood.is_ood`` OR ``prediction.wide_ci``.
        ood.is_ood / log_density — GMM detector verdict.
        derived.* — HSLA-only carbon-equivalent style features.
    """
    # -------- 1. Locate model + meta -----------------------------------
    # _safe_version_dir validates the slug regex AND confirms the resolved
    # path stays inside MODELS_DIR — closes CWE-22 (path traversal via
    # ``../app`` or absolute ``/etc/passwd``). Returns 400 on bad input;
    # genuine "no such version" still surfaces as 404 below.
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

    steel_class = meta.get("steel_class") or system_router.LEGACY_STEEL_CLASS_FALLBACK
    target = meta.get("target")

    # -------- 2. Load class profile (drives feature_set + label) -------
    try:
        profile = load_steel_class(steel_class)
    except Exception as exc:  # YAML missing or malformed
        logger.exception("Failed to load steel class %s", steel_class)
        raise HTTPException(
            status_code=500,
            detail=f"Steel class profile '{steel_class}' unavailable: {exc}",
        ) from exc

    # -------- 3. Validate composition keys cover feature_set -----------
    # The UI form builds inputs from profile.feature_set, so a missing
    # key here means the caller is hitting the API directly without
    # sending all required fields. 400 is correct (client error), and
    # we name the missing keys for actionable debugging.
    required = set(profile.feature_set)
    provided = set(req.composition.keys())
    missing = sorted(required - provided)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"composition missing required keys for class "
                f"'{steel_class}': {missing}"
            ),
        )

    # -------- 4. Compute features + predict ----------------------------
    # Subset to feature_set to drop extras (e.g. UI sent derived
    # features that compute_features_for_class would recompute).
    raw_row = {k: float(req.composition[k]) for k in profile.feature_set}
    df_input = pd.DataFrame([raw_row])
    try:
        df_feat = compute_features_for_class(df_input, steel_class)
    except Exception as exc:
        logger.exception("compute_features_for_class failed for %s", steel_class)
        raise HTTPException(
            status_code=500,
            detail=f"Feature engineering failed: {exc}",
        ) from exc

    try:
        bundle = load_model(safe_version)
    except Exception as exc:
        logger.exception("load_model failed for %s", safe_version)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model artifacts: {exc}",
        ) from exc

    pred_df = predict_with_uncertainty(bundle, df_feat)
    row = pred_df.iloc[0]

    mean = float(row["prediction"])
    lo_p = float(row["lower_90"])
    hi_p = float(row["upper_90"])
    ci_half_width = (hi_p - lo_p) / 2.0
    ood_flag = bool(row["ood_flag"])
    log_density = float(row["log_density"])

    # Resolve target label from profile.
    # Fallback chain: profile match → bare ``target`` id → "Прогноз" so the
    # UI never renders an empty string ("null"-looking metric heading) when
    # a legacy meta.json has no ``target`` and the class profile has no
    # matching property.
    target_label = next(
        (t.label for t in profile.target_properties if t.id == target),
        target or "",
    )
    target_label = target_label or "Прогноз"

    # Derived HSLA features surfaced for the UI. We only include them
    # when present (HSLA class with compute_hsla_features feature
    # engineering).
    derived: dict[str, float] = {}
    for col in ("cev_iiw", "pcm", "cen", "microalloying_sum"):
        if col in df_feat.columns:
            derived[col] = float(df_feat[col].iloc[0])

    # PR 12: wide-CI heuristic — drives the anomaly-explainer button
    # when the prediction is *not* OOD but the CI is wide relative to
    # the target's normal range. Returning the threshold lets the UI
    # render an actionable tooltip ("CI 0.5× нормального диапазона")
    # rather than just a binary flag.
    wide_ci_flag, wide_ci_threshold = _compute_wide_ci_flag(
        mean=mean,
        q05=lo_p,
        q95=hi_p,
        target=target,
        profile=profile,
    )

    return {
        "prediction": {
            "mean": mean,
            "q05": lo_p,
            "q95": hi_p,
            # DEPRECATED aliases — kept for backward compat with PR 3
            # frontend builds. Planned removal once views/predict.js
            # uses q05/q95 exclusively.
            "lower_p": lo_p,
            "upper_p": hi_p,
            "ci_half_width": ci_half_width,
            "wide_ci": wide_ci_flag,
            "wide_ci_threshold": wide_ci_threshold,
            "target_property": target,
            "target_label": target_label,
        },
        "ood": {
            "is_ood": ood_flag,
            "log_density": log_density,
        },
        "derived": derived,
        "model": {
            # Echo the normalised slug rather than ``req.model_version`` so
            # we never reflect attacker-controlled traversal segments back
            # to the client (closes reviewer nit #9 alongside the 400 guard
            # above).
            "version": safe_version,
            "steel_class": steel_class,
            "target": target,
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Anomaly explain — Sonnet PhD diagnosis on demand (PR 12)
# ──────────────────────────────────────────────────────────────────────


class AnomalyExplainRequest(BaseModel):
    """Body for ``POST /api/predict/anomaly-explain``.

    The frontend calls this after the predict endpoint returns and the
    user clicks «Объяснить почему рискованно». We re-take the same
    composition the predict call used (rather than walking back to the
    job-cached prediction) so the request is idempotent and stateless
    — the body fully describes the diagnosis context.

    All numeric inputs are floats; ``is_ood`` is an explicit bool that
    the caller already inspected. We trust the caller because the
    detector verdict is also recomputed from the model artifacts on
    the backend; the explicit flag drives the UI's auto-expand
    behaviour upstream and is echoed in the LLM context for the
    diagnosis prompt.
    """

    model_version: str = Field(
        ...,
        description=(
            "Trained model version slug (matches ``models/<slug>/`` directory)."
            " Validated against the path-traversal regex in ``_safe_version_dir``."
        ),
    )
    composition: dict[str, float] = Field(
        ..., description="feature_name → numeric value. Same shape as /api/predict."
    )
    predicted_mean: float = Field(
        ..., description="Mean prediction echoed from the prior /api/predict call."
    )
    predicted_q05: float = Field(
        ..., description="90% CI lower bound echoed from /api/predict.prediction.q05."
    )
    predicted_q95: float = Field(
        ..., description="90% CI upper bound echoed from /api/predict.prediction.q95."
    )
    is_ood: bool = Field(
        ...,
        description=(
            "OOD verdict echoed from /api/predict.ood.is_ood. Used as a "
            "context tag in the LLM prompt so it can prioritise OOD-specific "
            "narrative even when only some features are out-of-range."
        ),
    )
    save_to_decision_log: bool = Field(
        default=False,
        description=(
            "Opt-in audit-trail save with tag 'anomaly_cycle'. "
            "Default false to keep test/automation flows quiet — the UI "
            "exposes a «Сохранить» button to flip it explicitly."
        ),
    )

    model_config = {
        # Same rationale as PredictRequest — silence Pydantic's
        # ``model_*`` reserved-namespace warning so ``model_version``
        # stays a stable kwarg across the API surface.
        "protected_namespaces": (),
    }


def _build_anomaly_context(
    *,
    safe_version: str,
    profile,
    target: str | None,
    composition: dict[str, float],
    df_feat: pd.DataFrame,
    predicted_mean: float,
    predicted_q05: float,
    predicted_q95: float,
    is_ood: bool,
    log_density: float | None,
    training_ranges: dict[str, list[float]],
) -> dict[str, Any]:
    """Assemble the dict consumed by ``AnomalyExplainer.explain``.

    The explainer accepts:
        recipe          — feature → numeric (same keys as df_feat columns
                          that intersect training_ranges)
        training_ranges — feature → [lo, hi] from the model meta
        training_medians — feature → midpoint as a coarse default
        ml_prediction   — predicted/lower_90/upper_90/ci_width
        ood_flag, ood_score, out_of_range_features

    Recipe is computed from ``df_feat`` (engineered features), not the
    raw composition, so derived columns (cev_iiw, pcm, …) flow through
    when present. We restrict to keys present in ``training_ranges``
    so the LLM doesn't receive stray derived features the GMM was
    never fit on.
    """
    recipe: dict[str, float] = {}
    out_of_range: list[dict[str, Any]] = []
    for feat in df_feat.columns:
        if feat not in training_ranges:
            continue
        try:
            v = float(df_feat[feat].iloc[0])
        except (TypeError, ValueError):
            continue
        recipe[feat] = v
        rng = training_ranges[feat]
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        lo_f, hi_f = float(rng[0]), float(rng[1])
        if v < lo_f or v > hi_f:
            out_of_range.append(
                {"feature": feat, "value": v, "training_range": [lo_f, hi_f]}
            )

    medians = {
        feat: (float(rng[0]) + float(rng[1])) / 2.0
        for feat, rng in training_ranges.items()
        if isinstance(rng, (list, tuple)) and len(rng) == 2
    }

    return {
        "model_version": safe_version,
        "steel_class": profile.id,
        "target": target,
        "recipe": recipe,
        "training_ranges": training_ranges,
        "training_medians": medians,
        "ml_prediction": {
            "predicted": float(predicted_mean),
            "lower_90": float(predicted_q05),
            "upper_90": float(predicted_q95),
            "ci_width": float(predicted_q95) - float(predicted_q05),
        },
        "ood_flag": bool(is_ood),
        "ood_score": float(log_density) if log_density is not None else None,
        "out_of_range_features": out_of_range,
    }


def _run_anomaly_explain_job(
    *,
    model_version: str,
    composition: dict[str, float],
    predicted_mean: float,
    predicted_q05: float,
    predicted_q95: float,
    is_ood: bool,
    save_to_decision_log: bool,
    progress: Any = None,
) -> dict[str, Any]:
    """Background worker — single Sonnet call → AnomalyExplanation dict.

    Stages:
        0.05 → "Загружаю модель и контекст"
        0.30 → "Sonnet PhD анализирует (~30-60 с)"
        0.95 → "Сохраняю в Decision Log" (only if opt-in)
        1.00 → done

    Cancellation gates fire at 0.05 (before context load) and 0.30
    (immediately before the LLM call). Per PR 9/10 conventions, the
    gate cannot interrupt an in-flight Sonnet call.

    Returns dict consumed by the predict view's anomaly panel:
        {
          "model_version": str,
          "explanation": AnomalyExplanation-asdict | None,
          "evidence": [{feature, value, training_range}, ...],
          "duration_s": float,
          "decision_log_id": int | None,
          "config": {...echo of inputs...},
        }
    The asdict explanation is None when the LLM call fails or returns
    no tool_use block — the UI surfaces that as «Объяснение не
    получено» without retrying.
    """
    started = time.monotonic()

    is_cancelled = make_progress_cancel_check(progress)

    if callable(progress):
        progress(0.05, "Загружаю модель и контекст")

    # Path-traversal guard. The endpoint already checks but we re-verify
    # inside the worker so any future path that schedules this worker
    # bypassing the endpoint still gets the regex check. ``_safe_version_dir``
    # raises HTTPException(400); inside a worker we re-raise as a plain
    # ``RuntimeError`` so the JobStore captures it as a job error.
    try:
        version_dir = system_router._safe_version_dir(model_version)
    except HTTPException as exc:
        raise RuntimeError(
            f"Invalid model_version inside worker: {exc.detail}"
        ) from exc
    safe_version = version_dir.name

    if not version_dir.is_dir():
        raise RuntimeError(f"Model version '{safe_version}' not found")

    meta = system_router._read_model_meta(version_dir)
    if meta is None:
        raise RuntimeError(f"Model '{safe_version}' is missing meta.json")

    steel_class = meta.get("steel_class") or system_router.LEGACY_STEEL_CLASS_FALLBACK
    target = meta.get("target")

    profile = load_steel_class(steel_class)

    # Subset to feature_set so extras don't leak into compute_features.
    required = set(profile.feature_set)
    provided = set(composition.keys())
    missing = sorted(required - provided)
    if missing:
        raise RuntimeError(
            "composition missing required keys for class "
            f"'{steel_class}': {missing}"
        )
    raw_row = {k: float(composition[k]) for k in profile.feature_set}
    df_input = pd.DataFrame([raw_row])
    df_feat = compute_features_for_class(df_input, steel_class)

    # Recompute log_density via the model's GMM so the explainer sees
    # a fresh value matching ``is_ood`` rather than relying on the
    # caller's echo. Cheap (one GMM score) and removes a synchronisation
    # bug class where stale predicted_* + fresh meta could mislead the
    # LLM.
    bundle = load_model(safe_version)
    pred_df = predict_with_uncertainty(bundle, df_feat)
    log_density = float(pred_df.iloc[0]["log_density"])

    training_ranges = meta.get("training_ranges") or {}

    if is_cancelled():
        raise RuntimeError("Cancelled by user (before anomaly LLM call)")

    if callable(progress):
        progress(0.30, "Sonnet PhD анализирует (~30-60 с)…")

    # Lazy import — keeps the FastAPI cold-start light and lets tests
    # stub the factory before it's loaded. The factory raises
    # ``PromptNotFoundError`` at module import if prompts/ is missing —
    # ``check_llm_ready`` at the endpoint pre-validated, so we shouldn't
    # normally hit that here.
    from app.backend.anomaly_explainer import make_anomaly_explainer

    explainer = make_anomaly_explainer()
    if explainer is None:
        # Late guard — the endpoint pre-checks the API key so this is
        # rare. Surface as a job error (not 503) because the request
        # was already accepted.
        raise RuntimeError(
            "AnomalyExplainer unavailable: ANTHROPIC_API_KEY missing or "
            "anthropic SDK not installed."
        )

    ctx = _build_anomaly_context(
        safe_version=safe_version,
        profile=profile,
        target=target,
        composition=composition,
        df_feat=df_feat,
        predicted_mean=predicted_mean,
        predicted_q05=predicted_q05,
        predicted_q95=predicted_q95,
        is_ood=is_ood,
        log_density=log_density,
        training_ranges=training_ranges,
    )

    explanation = explainer.explain(ctx)
    explanation_dict = asdict(explanation) if explanation is not None else None

    # Evidence list = features the worker itself flagged as out-of-range
    # vs the model's training_ranges. Independent of anything the LLM
    # said so the UI can render an objective table even if Sonnet
    # returned None.
    evidence = list(ctx["out_of_range_features"])

    decision_log_id: int | None = None
    if save_to_decision_log and explanation is not None:
        if callable(progress):
            progress(0.95, "Сохраняю в Decision Log…")
        try:
            from decision_log.logger import log_decision

            decision_log_id = log_decision(
                phase="validation",
                decision=(
                    f"Anomaly explanation: severity={explanation.severity}, "
                    f"n_features={len(explanation.anomalous_features)}"
                ),
                reasoning=(
                    f"Model={safe_version}, "
                    f"target={target}, "
                    f"is_ood={is_ood}, "
                    f"out_of_range={len(evidence)}"
                ),
                context={
                    "model_version": safe_version,
                    "steel_class": steel_class,
                    "explanation": explanation_dict,
                    "evidence": evidence,
                    "ml_prediction": ctx["ml_prediction"],
                },
                author="api",
                tags=["anomaly_cycle", "sonnet-4-6"],
            )
        except Exception as exc:  # noqa: BLE001 — audit save is best-effort
            logger.warning("Decision Log save failed: %s", exc)

    if callable(progress):
        progress(1.0, "Готово")

    duration_s = round(time.monotonic() - started, 2)

    return {
        "model_version": safe_version,
        "explanation": explanation_dict,
        "evidence": evidence,
        "duration_s": duration_s,
        "decision_log_id": decision_log_id,
        "config": {
            "model_version": safe_version,
            "is_ood": bool(is_ood),
            "save_to_decision_log": bool(save_to_decision_log),
        },
    }


@router.post(
    "/predict/anomaly-explain",
    response_class=SafeJSONResponse,
    response_model=None,
)
def anomaly_explain(req: AnomalyExplainRequest) -> dict[str, Any]:
    """Submit an anomaly-explanation cycle and return the job id.

    Validation order:
        1) ``_safe_version_dir`` — path traversal guard (CWE-22). 400.
        2) Model directory + meta.json present. 404 on miss.
        3) Composition covers feature_set. 400 with missing keys.
        4) ``check_llm_ready`` — ANTHROPIC_API_KEY + prompt file. 503.
        5) Submit job. Return ``{job_id, config}``.

    Total wall time on a typical fatigue prediction is ~30-60 s
    (single Sonnet call with cache_control=ephemeral on the system
    prompt). The frontend polls ``/api/jobs/{id}`` (PR 6) until status
    flips to ``done``, then renders the explanation card. ``DELETE``
    sets the cooperative cancellation flag — see module docstring for
    the cancellation semantics (effective only before the LLM call).
    """
    # 1-2. Path-traversal validation + model existence.
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

    # 3. Composition coverage — same contract as /api/predict so the
    # caller can reuse the same composition dict.
    steel_class = meta.get("steel_class") or system_router.LEGACY_STEEL_CLASS_FALLBACK
    try:
        profile = load_steel_class(steel_class)
    except Exception as exc:
        logger.exception("Failed to load steel class %s", steel_class)
        raise HTTPException(
            status_code=500,
            detail=f"Steel class profile '{steel_class}' unavailable: {exc}",
        ) from exc
    required = set(profile.feature_set)
    provided = set(req.composition.keys())
    missing = sorted(required - provided)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"composition missing required keys for class "
                f"'{steel_class}': {missing}"
            ),
        )

    # 4. Fail-fast LLM gating — return 503 with a clear remediation
    # message instead of a generic "job ended with error" once the
    # worker fails on a missing key or prompt.
    check_llm_ready(["anomaly_explainer"])

    # 5. Submit. We pass the validated ``safe_version`` (not the raw
    # request value) so future paths cannot bypass the regex via direct
    # ``run_as_job`` calls.
    job_id = run_as_job(
        _run_anomaly_explain_job,
        model_version=safe_version,
        composition={k: float(v) for k, v in req.composition.items()},
        predicted_mean=float(req.predicted_mean),
        predicted_q05=float(req.predicted_q05),
        predicted_q95=float(req.predicted_q95),
        is_ood=bool(req.is_ood),
        save_to_decision_log=bool(req.save_to_decision_log),
    )
    return {
        "job_id": job_id,
        "config": {
            "model_version": safe_version,
            "is_ood": bool(req.is_ood),
            "save_to_decision_log": bool(req.save_to_decision_log),
        },
    }
