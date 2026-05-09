"""Router for /api/system/* — steel classes + trained models metadata.

PR 3 of the Streamlit→FastAPI migration. See
``docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md``
(Endpoint map → Common / system).

Endpoints:
- ``GET /api/system/steel-classes`` — list of YAML profiles for the UI
  to drive class-aware composition forms.
- ``GET /api/system/models`` — list of trained models (one entry per
  ``models/<version>/meta.json``).
- ``GET /api/system/models/active`` — last (sorted by name) trained
  model. Mirrors Streamlit sidebar default selection
  (``index=len(available_models)-1``).

Responses use ``SafeJSONResponse`` because metrics dicts inside
``meta.json`` may contain numpy floats / NaN-like values once we expand
to other backends; the encoder handles them transparently. The
``response_model=None`` avoids Pydantic v2 trying to revalidate the
dict and bypass our custom encoder (Risks #3 in the design doc).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.responses import SafeJSONResponse
from app.backend.steel_classes import (
    AVAILABLE_CLASS_IDS,
    SteelClassProfile,
    load_steel_class,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Models directory mirrors ``app.backend.model_trainer.MODELS_DIR``. We
# don't import that constant directly so tests can monkeypatch this
# module-level value without touching the trainer module.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"

# Streamlit fallback: legacy meta.json without ``steel_class`` defaults
# to the original HSLA pipeline class (CLAUDE.md «Working as a team» +
# design doc User decisions #5).
LEGACY_STEEL_CLASS_FALLBACK = "pipe_hsla"

# CWE-22 mitigation: ``model_version`` arrives from untrusted clients and is
# concatenated to ``MODELS_DIR``. ``Path("models") / "../app"`` resolves
# outside ``models/`` and ``Path("models") / "/etc/passwd"`` discards the
# prefix entirely — both would let a caller load XGBoost artifacts from any
# directory the API process can read. The regex below matches the actual
# names produced by ``model_trainer`` (timestamped slugs: letters, digits,
# underscores, hyphens, dots) so we reject everything else cheaply, and
# ``relative_to`` enforces the no-escape invariant after symlink resolution.
_VERSION_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _safe_version_dir(version: str) -> Path:
    """Resolve ``MODELS_DIR / version`` safely, rejecting path traversal.

    Validates the name against the version-slug regex (matches the format
    emitted by ``app.backend.model_trainer`` — see commits creating dirs
    like ``hsla_yield_strength_xgb_20260426_204941``) and confirms the
    resolved path stays inside ``MODELS_DIR``. Raises ``HTTPException(400)``
    on any deviation so endpoints get a uniform error response.
    """
    if not _VERSION_RE.match(version):
        raise HTTPException(
            status_code=400,
            detail="Invalid model_version format",
        )
    version_dir = (MODELS_DIR / version).resolve()
    try:
        version_dir.relative_to(MODELS_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="model_version escapes models directory",
        ) from exc
    return version_dir


def _profile_to_dict(profile: SteelClassProfile) -> dict[str, Any]:
    """Serialise a SteelClassProfile to a JSON-friendly dict.

    target_properties is a list of dataclasses; SafeJSONResponse would
    handle them, but we expand explicitly so the shape contract is
    visible in this file (frontend reads ``id/label/range``).
    """
    return {
        "id": profile.id,
        "name": profile.name,
        "standard": profile.standard,
        "target_properties": [
            {"id": t.id, "label": t.label, "range": list(t.range)}
            for t in profile.target_properties
        ],
        "feature_set": list(profile.feature_set),
        "physical_bounds": {k: list(v) for k, v in profile.physical_bounds.items()},
        "expected_top_features": list(profile.expected_top_features),
        "process_params": list(profile.process_params),
        "target_o_activity_ppm": profile.target_o_activity_ppm,
        "data_source": profile.data_source,
        "data_source_doi": profile.data_source_doi,
    }


def _read_model_meta(version_dir: Path) -> dict[str, Any] | None:
    """Read ``meta.json`` for a model version directory.

    Returns ``None`` if the directory is missing meta.json or it can't
    be parsed — silently skipped so a single corrupt model can't break
    the listing endpoint. We log at WARNING so MLOps can spot it.
    """
    meta_path = version_dir / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        with meta_path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read %s: %s", meta_path, exc)
        return None


def _version_dir_for_meta(meta: dict[str, Any]) -> Path | None:
    """Locate the model directory for a given meta.json content.

    Used to check filesystem artifacts (ood_detector.pkl) without
    re-walking MODELS_DIR. Returns None if version field is missing.
    """
    version = meta.get("version")
    if not version:
        return None
    return MODELS_DIR / version


def _model_summary(meta: dict[str, Any]) -> dict[str, Any]:
    """Compact per-model entry for ``GET /api/system/models``.

    We keep the full ``meta`` reachable via ``training_ranges`` /
    ``feature_list`` because the predict UI needs feature_list to send
    composition keys, and design UI (PR 7) reads training_ranges. If
    payload size becomes an issue we can split into list (compact) +
    detail endpoint, but for the current ~10 models that is premature.
    """
    metrics = meta.get("metrics") or {}
    vdir = _version_dir_for_meta(meta)
    has_ood = (vdir / "ood_detector.pkl").is_file() if vdir is not None else False
    # Quantile booster artifacts live next to ``main.json`` as ``q05.json`` /
    # ``q95.json`` (see ``model_trainer.train_xgboost_with_quantiles`` and any
    # real model dir, e.g. ``models/hsla_yield_strength_xgb_*/``). Older
    # checkpoints that pre-date the q-models won't have these files, so the
    # flag must reflect the actual filesystem state — frontend hides the CI
    # band when ``has_uncertainty`` is false.
    if vdir is not None:
        has_uncertainty = (
            (vdir / "q05.json").is_file() and (vdir / "q95.json").is_file()
        )
    else:
        has_uncertainty = False
    return {
        "version": meta.get("version") or "",
        "steel_class": meta.get("steel_class") or LEGACY_STEEL_CLASS_FALLBACK,
        "target": meta.get("target"),
        "feature_list": list(meta.get("feature_list") or []),
        "training_ranges": meta.get("training_ranges") or {},
        "metrics": {
            "r2_test": metrics.get("r2_test"),
            "mae_test": metrics.get("mae_test"),
            "rmse_test": metrics.get("rmse_test"),
            "coverage_90_ci": metrics.get("coverage_90_ci"),
            "n_train": metrics.get("n_train"),
            "n_val": metrics.get("n_val"),
            "n_test": metrics.get("n_test"),
        },
        "has_uncertainty": has_uncertainty,
        "has_ood_detector": has_ood,
        "trained_at": meta.get("trained_at"),
        "data_source": meta.get("data_source"),
        "data_source_doi": meta.get("data_source_doi"),
        "conformal_correction_mpa": meta.get("conformal_correction_mpa"),
    }


def _list_model_metas() -> list[dict[str, Any]]:
    """Walk ``MODELS_DIR`` and return parsed metas sorted by name asc.

    Sort by directory name matches Streamlit's ``sorted([...])`` —
    timestamps in version names mean asc-sort == oldest-first, so the
    "active" model (last) is the most recently trained one.
    """
    if not MODELS_DIR.is_dir():
        return []
    metas: list[dict[str, Any]] = []
    for entry in sorted(MODELS_DIR.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        meta = _read_model_meta(entry)
        if meta is None:
            continue
        metas.append(meta)
    return metas


@router.get(
    "/steel-classes",
    response_class=SafeJSONResponse,
    response_model=None,
)
def list_steel_classes() -> dict[str, Any]:
    """Return all registered steel-class profiles.

    Wraps ``app.backend.steel_classes.AVAILABLE_CLASS_IDS`` +
    ``load_steel_class``. The frontend uses this to drive composition
    form rendering in the predict view (and design view in PR 7).
    """
    items = [_profile_to_dict(load_steel_class(cid)) for cid in AVAILABLE_CLASS_IDS]
    return {"items": items, "count": len(items)}


@router.get(
    "/models",
    response_class=SafeJSONResponse,
    response_model=None,
)
def list_models() -> dict[str, Any]:
    """List all trained-model directories with summary metadata.

    Returns ``{items: [...], count}`` consistent with /steel-classes
    and /decisions for the frontend to consume. Empty list when
    ``models/`` is missing or has no valid entries.
    """
    metas = _list_model_metas()
    items = [_model_summary(meta) for meta in metas]
    return {"items": items, "count": len(items)}


@router.get(
    "/models/active",
    response_class=SafeJSONResponse,
    response_model=None,
)
def get_active_model() -> dict[str, Any]:
    """Return the most recently trained model.

    Mirrors Streamlit sidebar default: ``available_models[-1]`` after
    ``sorted`` by directory name. 404 when ``models/`` has zero valid
    entries — frontend renders a "train a model first" banner in that
    case (parity with Streamlit ``if not selected_model``).
    """
    metas = _list_model_metas()
    if not metas:
        raise HTTPException(
            status_code=404,
            detail="Нет обученных моделей. Обучите модель во вкладке «Обучение».",
        )
    return _model_summary(metas[-1])
