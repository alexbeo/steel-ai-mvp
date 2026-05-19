"""Unified η_Al predictor mixing plant-specific Bayesian posterior + global ML model.

PR 9 — Block III: combines:
  - EtaAlCalibrator.get_posterior(plant_id, method_id) → plant-specific posterior
  - Trained ML model (XGBoost + conformal) → feature-aware global prediction

Mix via mixture-of-experts in logit-space (Lewis variance includes disagreement penalty):
  μ_mix = w·μ_plant + (1-w)·μ_global
  σ_mix² = w·σ_plant² + (1-w)·σ_global² + w·(1-w)·(μ_plant - μ_global)²

Weight w = sigmoid((N_plant - 30) / 10) — smooth transition, N=30 → w=0.5.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from app.backend.eta_al_calibration import (
    Z_90,
    _compute_prior_logit,
    _invlogit,
    _logit,
)
from app.backend.slag_aware_deox import load_addition_methods

logger = logging.getLogger(__name__)

# Defaults aligned with EtaAlCalibrator min_heats_for_posterior = 30
_DEFAULT_SIGMOID_N50 = 30.0
_DEFAULT_SIGMOID_SCALE = 10.0
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODELS_DIR = _PROJECT_ROOT / "models"
_DEFAULT_PLANT_OFFSETS_PATH = (
    _PROJECT_ROOT / "data" / "deox_methods" / "plant_offsets.yaml"
)
# Clip eta to (eps, 1-eps) before logit transform — mirrors LOGIT_CLIP_EPS
# in eta_al_calibration but kept local to avoid importing a private constant.
_LOGIT_CLIP_EPS = 1e-3


PredictionSource = Literal[
    "plant_only", "global_only", "mixed", "literature_fallback"
]


@dataclass
class EtaPrediction:
    eta_mean: float
    eta_logit_mu: float
    eta_logit_sigma: float
    plant_weight: float
    global_weight: float
    source: PredictionSource
    n_plant_heats: int
    metadata: dict = field(default_factory=dict)


def _load_plant_offsets(
    path: Path = _DEFAULT_PLANT_OFFSETS_PATH,
) -> dict[str, float]:
    """Load plant_offset_baseline mapping from YAML."""
    if not path.exists():
        logger.warning("plant_offsets.yaml not found at %s — empty mapping", path)
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {str(k): float(v) for k, v in data.get("plants", {}).items()}


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        ex = math.exp(-x)
        return 1.0 / (1.0 + ex)
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _scan_latest_model(
    models_dir: Path, prefix: str = "deox_eta_al_effective_xgb_"
) -> str | None:
    """Find latest model artifact by lexicographic sort of directory names."""
    if not models_dir.exists():
        return None
    candidates = sorted(
        [
            d.name
            for d in models_dir.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        ],
        reverse=True,
    )
    return candidates[0] if candidates else None


def _clip_eta(p: float) -> float:
    """Clip eta into (eps, 1-eps) for a finite logit."""
    return max(_LOGIT_CLIP_EPS, min(1.0 - _LOGIT_CLIP_EPS, float(p)))


class EtaAlPredictor:
    """Unified η_Al predictor mixing plant posterior + global ML model.

    Lazy model loading: model bundle loaded on first predict() call.
    Use single shared instance for production (FastAPI lifespan).
    """

    def __init__(
        self,
        calibrator: Any | None = None,
        model_version: str | None = None,
        models_dir: Path | None = None,
        plant_offsets_path: Path | None = None,
        sigmoid_n50: float = _DEFAULT_SIGMOID_N50,
        sigmoid_scale: float = _DEFAULT_SIGMOID_SCALE,
    ):
        self.calibrator = calibrator
        self.model_version = model_version
        self.models_dir = models_dir or _DEFAULT_MODELS_DIR
        self.plant_offsets_path = plant_offsets_path or _DEFAULT_PLANT_OFFSETS_PATH
        self.sigmoid_n50 = float(sigmoid_n50)
        self.sigmoid_scale = float(sigmoid_scale)
        self._model_bundle: Any | None = None
        self._model_loaded: bool = False
        self._resolved_version: str | None = None
        self._plant_offsets: dict[str, float] | None = None
        self._methods_catalog: dict | None = None
        # PR 10 nit (was self._last_global_error in PR 9 review): set by
        # _predict_global on failure, surfaced into metadata["global_error"].
        self._last_global_error: str | None = None

    def _ensure_model_loaded(self) -> None:
        if self._model_loaded:
            return
        version = self.model_version or _scan_latest_model(self.models_dir)
        if version is None:
            logger.warning("No deox η_Al model found in %s", self.models_dir)
            self._model_bundle = None
            self._model_loaded = True
            return
        try:
            from app.backend.model_trainer import load_model

            self._model_bundle = load_model(version)
            self._resolved_version = version
            logger.info("Loaded η_Al model version=%s", version)
        except Exception as exc:
            logger.warning("Failed to load model %s: %s", version, exc)
            self._model_bundle = None
        self._model_loaded = True

    def _ensure_plant_offsets(self) -> dict[str, float]:
        if self._plant_offsets is None:
            self._plant_offsets = _load_plant_offsets(self.plant_offsets_path)
        return self._plant_offsets

    def _ensure_methods(self) -> dict:
        if self._methods_catalog is None:
            self._methods_catalog = load_addition_methods()
        return self._methods_catalog

    def _resolve_features(
        self, plant_id: str, method_id: str, partial: dict | None
    ) -> tuple[dict[str, float], list[str]]:
        """Fill in method_eta_baseline + plant_offset_baseline; warn on unknowns."""
        partial = dict(partial or {})
        warnings: list[str] = []
        methods = self._ensure_methods()
        if method_id not in methods:
            raise ValueError(f"Unknown method_id: {method_id}")
        if "method_eta_baseline" not in partial:
            partial["method_eta_baseline"] = float(methods[method_id].eta_al_typical)
        offsets = self._ensure_plant_offsets()
        if "plant_offset_baseline" not in partial:
            if plant_id in offsets:
                partial["plant_offset_baseline"] = float(offsets[plant_id])
            else:
                warnings.append(
                    f"unknown plant '{plant_id}' — plant_offset_baseline set to 0.0"
                )
                partial["plant_offset_baseline"] = 0.0
        return partial, warnings

    def _predict_global(
        self, features: dict | None
    ) -> tuple[float, float, str | None] | None:
        """Run ML model → (μ_logit, σ_logit, model_version) or None if no model."""
        self._ensure_model_loaded()
        if self._model_bundle is None or features is None:
            return None
        try:
            import pandas as pd

            from app.backend.model_trainer import predict_with_uncertainty

            df = pd.DataFrame([features])
            out = predict_with_uncertainty(self._model_bundle, df)
            point_v = float(out["prediction"].iloc[0])
            lo_v = float(out["lower_90"].iloc[0])
            hi_v = float(out["upper_90"].iloc[0])
            # Clip + logit-transform
            mu = _logit(_clip_eta(point_v))
            sigma = (
                _logit(_clip_eta(hi_v)) - _logit(_clip_eta(lo_v))
            ) / (2.0 * Z_90)
            if sigma <= 0:
                logger.warning("Global model σ_logit ≤ 0; falling back")
                return None
            return (mu, sigma, self._resolved_version)
        except Exception as exc:
            logger.warning("ML prediction failed: %s", exc)
            self._last_global_error = str(exc)
            return None

    def _literature_fallback(self, method_id: str) -> tuple[float, float]:
        """Return (μ_logit, σ_logit) from literature catalog (same as calibrator prior)."""
        methods = self._ensure_methods()
        m = methods[method_id]
        eta_typical = float(m.eta_al_typical)
        eta_lo, eta_hi = float(m.eta_al_range[0]), float(m.eta_al_range[1])
        return _compute_prior_logit(eta_typical, eta_lo, eta_hi)

    def predict_eta_al(
        self,
        plant_id: str,
        method_id: str,
        features: dict | None = None,
    ) -> EtaPrediction:
        """Main entry point. Returns mixture of plant posterior + ML prediction."""
        # Reset per-call so a stale error from a previous call doesn't leak.
        self._last_global_error = None
        resolved_features, warnings = self._resolve_features(
            plant_id, method_id, features
        )
        plant_post = (
            self.calibrator.get_posterior(plant_id, method_id)
            if self.calibrator is not None
            else None
        )
        global_pred = self._predict_global(resolved_features)

        metadata: dict = {
            "model_version": global_pred[2] if global_pred else None,
            "method_id": method_id,
            "plant_id": plant_id,
            "warnings": warnings,
        }
        # PR 10: surface global prediction error (model present but features bad)
        if global_pred is None and self._model_loaded and self._model_bundle is not None:
            if self._last_global_error:
                metadata["global_error"] = self._last_global_error

        # Branch by available sources
        if plant_post is None and global_pred is None:
            mu_lit, sigma_lit = self._literature_fallback(method_id)
            return EtaPrediction(
                eta_mean=_invlogit(mu_lit),
                eta_logit_mu=mu_lit,
                eta_logit_sigma=sigma_lit,
                plant_weight=0.0,
                global_weight=0.0,
                source="literature_fallback",
                n_plant_heats=0,
                metadata=metadata,
            )

        if plant_post is None:
            mu_g, sigma_g, _ = global_pred
            return EtaPrediction(
                eta_mean=_invlogit(mu_g),
                eta_logit_mu=mu_g,
                eta_logit_sigma=sigma_g,
                plant_weight=0.0,
                global_weight=1.0,
                source="global_only",
                n_plant_heats=0,
                metadata=metadata,
            )

        mu_p, sigma_p, N = (
            plant_post.posterior_logit_mu,
            plant_post.posterior_logit_sigma,
            plant_post.n_heats_used,
        )

        if global_pred is None:
            return EtaPrediction(
                eta_mean=_invlogit(mu_p),
                eta_logit_mu=mu_p,
                eta_logit_sigma=sigma_p,
                plant_weight=1.0,
                global_weight=0.0,
                source="plant_only",
                n_plant_heats=N,
                metadata=metadata,
            )

        mu_g, sigma_g, _ = global_pred
        w = _sigmoid((N - self.sigmoid_n50) / self.sigmoid_scale)
        mu_mix = w * mu_p + (1 - w) * mu_g
        var_mix = (
            w * sigma_p**2
            + (1 - w) * sigma_g**2
            + w * (1 - w) * (mu_p - mu_g) ** 2
        )
        sigma_mix = math.sqrt(var_mix)
        metadata["disagreement_logit"] = abs(mu_p - mu_g)
        return EtaPrediction(
            eta_mean=_invlogit(mu_mix),
            eta_logit_mu=mu_mix,
            eta_logit_sigma=sigma_mix,
            plant_weight=w,
            global_weight=1 - w,
            source="mixed",
            n_plant_heats=N,
            metadata=metadata,
        )


__all__ = [
    "EtaPrediction",
    "EtaAlPredictor",
    "_sigmoid",
    "_load_plant_offsets",
    "_scan_latest_model",
]
