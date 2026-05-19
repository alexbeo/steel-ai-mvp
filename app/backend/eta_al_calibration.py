"""Bayesian per-plant×per-method η_Al калибратор (PR 5 Block II).

Normal-Normal conjugate update в logit-space над literature prior из
data/deox_methods/al_addition_methods.yaml. Читает HeatRecord'ы с
eta_al_effective IS NOT NULL из heats.db, группирует по (plant_id, method_id),
вычисляет posterior, записывает в data/deox_methods/calibrations/<plant_id>.yaml
(atomic). N≥30 threshold для commit posterior; иначе остаёмся на literature.

Downstream consumer — PR 9 (predict_eta_al mix global+plant-specific).

Math (закрытая формула — см. design doc):
  Prior: μ₀ = logit(η_typical), σ₀ = (logit(η_hi) - logit(η_lo)) / (2*1.96)
  Posterior:
    1/σ_post² = 1/σ₀² + n/σ_likelihood²
    μ_post    = σ_post² × (μ₀/σ₀² + n × ȳ / σ_likelihood²)
  η_mean = invlogit(μ_post)  # median, не expected value (small σ approximation)
  η_q05 = invlogit(μ_post - 1.645 σ_post)
  η_q95 = invlogit(μ_post + 1.645 σ_post)
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from app.backend.heat_records import (
    DEFAULT_DB_PATH as HEATS_DB_DEFAULT,
    HeatRecord,
    count_heats,
    list_heats,
)
from app.backend.slag_aware_deox import DEOX_METHODS_PATH

logger = logging.getLogger(__name__)

# Project layout
DEFAULT_CALIB_DIR = Path(__file__).resolve().parents[2] / "data" / "deox_methods" / "calibrations"
DEFAULT_METHODS_CATALOG_PATH = DEOX_METHODS_PATH

# Hyperparameters
MIN_HEATS_DEFAULT = 30
LIKELIHOOD_SIGMA_LOGIT_DEFAULT = 0.5
LOGIT_CLIP_EPS = 1e-3
Z_95 = 1.96
Z_90 = 1.645


# ── Math helpers ──────────────────────────────────────────────────────


def _clip_for_logit(p: float, eps: float = LOGIT_CLIP_EPS) -> tuple[float, bool]:
    """Clip p to [eps, 1-eps]. Returns (clipped_value, was_clipped)."""
    clipped = max(eps, min(1.0 - eps, float(p)))
    return clipped, clipped != p


def _logit(p: float) -> float:
    """logit(p) = log(p / (1-p))."""
    return math.log(p / (1.0 - p))


def _invlogit(z: float) -> float:
    """invlogit(z) = exp(z) / (1 + exp(z)) = 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + math.exp(-z))


def _compute_prior_logit(eta_typical: float, eta_lo: float, eta_hi: float) -> tuple[float, float]:
    """Convert literature (typical, [lo, hi]) → (μ₀, σ₀) в logit-space."""
    mu0 = _logit(eta_typical)
    sigma0 = (_logit(eta_hi) - _logit(eta_lo)) / (2.0 * Z_95)
    if sigma0 <= 0:
        raise ValueError(
            f"Invalid literature range: hi={eta_hi}, lo={eta_lo} → σ₀≤0"
        )
    return mu0, sigma0


def _compute_posterior_logit(
    mu0: float,
    sigma0: float,
    sample_mean_logit: float | None,
    n_obs: int,
    sigma_likelihood: float,
) -> tuple[float, float]:
    """Closed-form Normal-Normal posterior."""
    if n_obs <= 0:
        return mu0, sigma0
    if sample_mean_logit is None:
        raise ValueError("sample_mean_logit None at n_obs > 0")
    inv_var_post = 1.0 / (sigma0 ** 2) + n_obs / (sigma_likelihood ** 2)
    sigma_post = math.sqrt(1.0 / inv_var_post)
    mu_post = (sigma_post ** 2) * (
        mu0 / (sigma0 ** 2) + n_obs * sample_mean_logit / (sigma_likelihood ** 2)
    )
    return mu_post, sigma_post


# ── Dataclasses ───────────────────────────────────────────────────────


@dataclass
class EtaPosterior:
    plant_id: str
    method_id: str
    n_heats_used: int
    posterior_eta_mean: float | None
    posterior_eta_q05: float | None
    posterior_eta_q95: float | None
    posterior_logit_mu: float | None
    posterior_logit_sigma: float | None
    prior_eta_mean: float
    prior_eta_lo: float
    prior_eta_hi: float
    skipped_reason: str | None = None
    n_clipped: int = 0


@dataclass
class PlantCalibration:
    plant_id: str
    last_updated: str  # ISO-8601 UTC
    data_window_from: str | None
    data_window_to: str | None
    n_total_heats: int
    calibrations: list[EtaPosterior] = field(default_factory=list)
    yaml_written: bool = False
    yaml_path: str | None = None


# ── Catalog loader ────────────────────────────────────────────────────


def _load_methods_catalog(path: Path = DEFAULT_METHODS_CATALOG_PATH) -> dict[str, dict]:
    """Load al_addition_methods.yaml as dict[method_id, raw dict]."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    methods = data.get("methods", {})
    if not methods:
        raise ValueError(f"No 'methods' section in {path}")
    return methods


# ── Atomic YAML write ─────────────────────────────────────────────────


def _write_yaml_atomic(path: Path, data: dict) -> None:
    """Write YAML atomically: tempfile in same dir + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".yaml.tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── Main class ────────────────────────────────────────────────────────


class EtaAlCalibrator:
    """Bayesian Normal-Normal conjugate η_Al калибратор per (plant, method)."""

    def __init__(
        self,
        db_path: Path | None = None,
        calibrations_dir: Path | None = None,
        catalog_path: Path | None = None,
        *,
        min_heats_for_posterior: int = MIN_HEATS_DEFAULT,
        likelihood_sigma_logit: float = LIKELIHOOD_SIGMA_LOGIT_DEFAULT,
    ):
        self.db_path = db_path if db_path is not None else HEATS_DB_DEFAULT
        self.calibrations_dir = calibrations_dir or DEFAULT_CALIB_DIR
        self.catalog_path = catalog_path or DEFAULT_METHODS_CATALOG_PATH
        self.min_heats = int(min_heats_for_posterior)
        self.sigma_likelihood = float(likelihood_sigma_logit)
        self._methods: dict[str, dict] | None = None

    def _methods_dict(self) -> dict[str, dict]:
        if self._methods is None:
            self._methods = _load_methods_catalog(self.catalog_path)
        return self._methods

    def get_prior_from_catalog(self, method_id: str) -> tuple[float, float]:
        """Return (μ₀, σ₀) в logit-space из literature catalog."""
        methods = self._methods_dict()
        if method_id not in methods:
            raise ValueError(f"Unknown method_id: {method_id}")
        m = methods[method_id]
        eta_typical = float(m["eta_al_typical"])
        eta_range = m["eta_al_range"]
        eta_lo = float(eta_range[0])
        eta_hi = float(eta_range[1])
        return _compute_prior_logit(eta_typical, eta_lo, eta_hi)

    def _calibrate_one_method(
        self,
        plant_id: str,
        method_id: str,
        heats: list[HeatRecord],
    ) -> EtaPosterior:
        """Compute posterior for one (plant, method). Doesn't write YAML."""
        methods = self._methods_dict()
        if method_id not in methods:
            # Unknown method — skip
            return EtaPosterior(
                plant_id=plant_id, method_id=method_id, n_heats_used=0,
                posterior_eta_mean=None, posterior_eta_q05=None, posterior_eta_q95=None,
                posterior_logit_mu=None, posterior_logit_sigma=None,
                prior_eta_mean=0.0, prior_eta_lo=0.0, prior_eta_hi=0.0,
                skipped_reason=f"unknown method_id: {method_id}",
            )
        m = methods[method_id]
        eta_typical = float(m["eta_al_typical"])
        eta_range = m["eta_al_range"]
        eta_lo, eta_hi = float(eta_range[0]), float(eta_range[1])
        mu0, sigma0 = _compute_prior_logit(eta_typical, eta_lo, eta_hi)

        # Filter heats with non-null eta
        valid_etas: list[float] = []
        n_clipped = 0
        for h in heats:
            if h.eta_al_effective is None:
                continue
            clipped, was_clipped = _clip_for_logit(h.eta_al_effective)
            if was_clipped:
                n_clipped += 1
            valid_etas.append(clipped)

        n = len(valid_etas)
        if n < self.min_heats:
            # Below threshold — return prior as posterior, mark skipped
            return EtaPosterior(
                plant_id=plant_id, method_id=method_id, n_heats_used=n,
                posterior_eta_mean=eta_typical,
                posterior_eta_q05=eta_lo, posterior_eta_q95=eta_hi,
                posterior_logit_mu=mu0, posterior_logit_sigma=sigma0,
                prior_eta_mean=eta_typical, prior_eta_lo=eta_lo, prior_eta_hi=eta_hi,
                skipped_reason=f"n_heats={n} < min_heats={self.min_heats}",
                n_clipped=n_clipped,
            )

        # Compute posterior
        logits = [_logit(e) for e in valid_etas]
        sample_mean_logit = float(np.mean(logits))
        mu_post, sigma_post = _compute_posterior_logit(
            mu0, sigma0, sample_mean_logit, n, self.sigma_likelihood
        )
        eta_mean = _invlogit(mu_post)
        eta_q05 = _invlogit(mu_post - Z_90 * sigma_post)
        eta_q95 = _invlogit(mu_post + Z_90 * sigma_post)
        return EtaPosterior(
            plant_id=plant_id, method_id=method_id, n_heats_used=n,
            posterior_eta_mean=eta_mean,
            posterior_eta_q05=eta_q05, posterior_eta_q95=eta_q95,
            posterior_logit_mu=mu_post, posterior_logit_sigma=sigma_post,
            prior_eta_mean=eta_typical, prior_eta_lo=eta_lo, prior_eta_hi=eta_hi,
            n_clipped=n_clipped,
        )

    def calibrate_plant(self, plant_id: str) -> PlantCalibration:
        """Read all heats for plant_id, group by method, compute posterior, write YAML."""
        heats = list_heats(
            plant_id=plant_id, has_outcome=True, limit=100000, db_path=self.db_path
        )
        by_method: dict[str, list[HeatRecord]] = {}
        for h in heats:
            if h.method_id is None or h.eta_al_effective is None:
                continue
            by_method.setdefault(h.method_id, []).append(h)

        posteriors: list[EtaPosterior] = []
        methods = self._methods_dict()
        for method_id in methods:
            h_list = by_method.get(method_id, [])
            posteriors.append(self._calibrate_one_method(plant_id, method_id, h_list))

        # Data window
        if heats:
            dates = [h.created_at for h in heats if h.created_at is not None]
            window_from = min(dates).isoformat() if dates else None
            window_to = max(dates).isoformat() if dates else None
        else:
            window_from = window_to = None

        now_iso = datetime.now(timezone.utc).isoformat()
        n_total = count_heats(plant_id=plant_id, db_path=self.db_path)
        calib = PlantCalibration(
            plant_id=plant_id, last_updated=now_iso,
            data_window_from=window_from, data_window_to=window_to,
            n_total_heats=n_total, calibrations=posteriors,
        )

        # Write YAML if at least one method passed threshold
        any_calibrated = any(p.skipped_reason is None for p in posteriors)
        if any_calibrated:
            yaml_path = self.calibrations_dir / f"{plant_id}.yaml"
            yaml_data = {
                "plant_id": plant_id,
                "last_updated": now_iso,
                "data_window": {
                    "from": window_from, "to": window_to,
                    "n_total_heats": n_total,
                },
                "calibrations": {
                    p.method_id: {
                        "n_heats_used": p.n_heats_used,
                        "prior_eta_mean": p.prior_eta_mean,
                        "prior_eta_range": [p.prior_eta_lo, p.prior_eta_hi],
                        "posterior_eta_mean": p.posterior_eta_mean,
                        "posterior_eta_q05": p.posterior_eta_q05,
                        "posterior_eta_q95": p.posterior_eta_q95,
                        "posterior_logit_mu": p.posterior_logit_mu,
                        "posterior_logit_sigma": p.posterior_logit_sigma,
                        "n_clipped": p.n_clipped,
                        **({"skipped_reason": p.skipped_reason}
                           if p.skipped_reason else {}),
                    }
                    for p in posteriors
                },
            }
            _write_yaml_atomic(yaml_path, yaml_data)
            calib.yaml_written = True
            calib.yaml_path = str(yaml_path)

            # Decision Log (one entry per calibrate_plant)
            try:
                from decision_log.logger import log_decision
                summary = {p.method_id: {
                    "n_heats": p.n_heats_used,
                    "posterior_eta_mean": p.posterior_eta_mean,
                    "skipped": p.skipped_reason,
                } for p in posteriors}
                log_decision(
                    phase="deoxidation",
                    decision=f"η_Al calibration for plant {plant_id}: "
                             f"{sum(1 for p in posteriors if p.skipped_reason is None)}/"
                             f"{len(posteriors)} methods calibrated",
                    reasoning=f"min_heats={self.min_heats}, sigma_likelihood={self.sigma_likelihood}",
                    context={
                        "plant_id": plant_id,
                        "yaml_path": str(yaml_path),
                        "calibrations": summary,
                    },
                    author="eta_al_calibration",
                    tags=["eta_al_calibration", f"plant:{plant_id}"],
                )
            except Exception as exc:
                logger.warning("Decision Log save failed: %s", exc)
        return calib

    def calibrate_all_plants(self) -> list[PlantCalibration]:
        """Find distinct plant_ids in DB, calibrate each."""
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT DISTINCT plant_id FROM heats WHERE eta_al_effective IS NOT NULL"
            ).fetchall()
        plant_ids = [r[0] for r in rows]
        return [self.calibrate_plant(pid) for pid in plant_ids]

    def get_posterior(self, plant_id: str, method_id: str) -> EtaPosterior | None:
        """Read existing YAML (no recompute). None если YAML отсутствует или метод skipped."""
        yaml_path = self.calibrations_dir / f"{plant_id}.yaml"
        if not yaml_path.exists():
            return None
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        calibs = data.get("calibrations", {})
        if method_id not in calibs:
            return None
        c = calibs[method_id]
        if "skipped_reason" in c:
            return None
        return EtaPosterior(
            plant_id=plant_id, method_id=method_id,
            n_heats_used=int(c["n_heats_used"]),
            posterior_eta_mean=float(c["posterior_eta_mean"]),
            posterior_eta_q05=float(c["posterior_eta_q05"]),
            posterior_eta_q95=float(c["posterior_eta_q95"]),
            posterior_logit_mu=float(c["posterior_logit_mu"]),
            posterior_logit_sigma=float(c["posterior_logit_sigma"]),
            prior_eta_mean=float(c["prior_eta_mean"]),
            prior_eta_lo=float(c["prior_eta_range"][0]),
            prior_eta_hi=float(c["prior_eta_range"][1]),
            n_clipped=int(c.get("n_clipped", 0)),
        )


__all__ = [
    "EtaPosterior", "PlantCalibration", "EtaAlCalibrator",
    "DEFAULT_CALIB_DIR", "DEFAULT_METHODS_CATALOG_PATH",
    "MIN_HEATS_DEFAULT", "LIKELIHOOD_SIGMA_LOGIT_DEFAULT",
    "_clip_for_logit", "_logit", "_invlogit",
    "_compute_prior_logit", "_compute_posterior_logit",
]
