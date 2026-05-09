"""Router for /api/predict — single-composition property prediction.

PR 3 of the Streamlit→FastAPI migration. See
``docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md``
(Endpoint map → Tab «Прогноз»).

Streamlit parity reference: ``app/frontend/app.py`` lines 685-791
(``with tab_predict:`` block). The flow:

    composition (dict) → DataFrame → compute_features_for_class()
    → load_model() → predict_with_uncertainty() → response

We do **not** dispatch the anomaly explainer in PR 3 (that lands in
PR 12). The OOD flag is surfaced in the response so the UI can show a
warning + a placeholder button.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.responses import SafeJSONResponse
from app.api.routers import system as system_router
from app.backend.model_trainer import load_model, predict_with_uncertainty
from app.backend.steel_classes import compute_features_for_class, load_steel_class

logger = logging.getLogger(__name__)

router = APIRouter()


class PredictRequest(BaseModel):
    """Request body for ``POST /api/predict``.

    ``composition`` keys must cover the model's class ``feature_set``
    exactly (order doesn't matter; extra keys are ignored). Missing
    keys → 400. Streamlit always submits a complete form because it
    builds inputs from ``feature_set``, so the strict-coverage rule
    just makes that contract explicit at the API boundary.
    """

    model_version: str = Field(..., description="Version directory name under models/")
    composition: dict[str, float] = Field(
        ..., description="feature_name → numeric value (wt% or process units)"
    )

    model_config = {
        # ``model_*`` is reserved by Pydantic v2 for internal config; turn off the
        # warning so we can keep the field name aligned with Streamlit semantics.
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
    # Streamlit form builds inputs from profile.feature_set, so a
    # missing key here means the caller is hitting the API directly
    # without sending all required fields. 400 is correct (client
    # error), and we name the missing keys for actionable debugging.
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

    # Resolve target label from profile (Streamlit parity, line 772-776).
    # Fallback chain: profile match → bare ``target`` id → "Прогноз" so the
    # UI never renders an empty string ("null"-looking metric heading) when
    # a legacy meta.json has no ``target`` and the class profile has no
    # matching property.
    target_label = next(
        (t.label for t in profile.target_properties if t.id == target),
        target or "",
    )
    target_label = target_label or "Прогноз"

    # Derived HSLA features surfaced for the UI, mirroring Streamlit
    # lines 783-791. We only include them when present (HSLA class with
    # compute_hsla_features feature engineering).
    derived: dict[str, float] = {}
    for col in ("cev_iiw", "pcm", "cen", "microalloying_sum"):
        if col in df_feat.columns:
            derived[col] = float(df_feat[col].iloc[0])

    return {
        "prediction": {
            "mean": mean,
            "q05": lo_p,
            "q95": hi_p,
            "lower_p": lo_p,  # backwards-compat alias matching design doc shape
            "upper_p": hi_p,
            "ci_half_width": ci_half_width,
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
