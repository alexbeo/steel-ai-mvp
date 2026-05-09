"""Tests for /api/system/* + /api/predict (PR 3 of FastAPI migration).

Test isolation strategy:
    - We avoid spinning up Optuna training in tests (5 min budget). Instead
      we copy ONE pre-existing model directory from the repo's ``models/``
      into a temp dir and monkeypatch both ``app.api.routers.system.MODELS_DIR``
      and ``app.backend.model_trainer.MODELS_DIR`` to point there.
    - For the empty-dir test we just don't copy anything.
    - For the legacy-fallback test we strip ``steel_class`` from a copied
      meta.json before letting the endpoint read it.
    - For predict tests we send the full feature_set with mid-range values
      drawn from the class profile's physical_bounds, mirroring how the
      Streamlit UI builds default values.

    This pattern keeps tests fast (~1 sec total) and exercises the real
    XGBoost prediction path. Down side: tests skip cleanly when the repo
    has zero pre-trained HSLA models — that case is covered by smoke_test.py
    in CI but should rarely happen locally.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routers import system as system_router
from app.backend import model_trainer
from app.backend.steel_classes import load_steel_class

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_MODELS_DIR = PROJECT_ROOT / "models"

# Required artifact filenames for a complete HSLA model directory.
# The OOD detector file is a serialised GMM produced by model_trainer.
_REQUIRED_MODEL_ARTIFACTS = frozenset(
    {"main.json", "q05.json", "q95.json", "ood_detector.pkl"}
)


def _find_real_hsla_model() -> Path | None:
    """Locate any complete HSLA model dir in the repo.

    Returns the first directory whose meta.json declares
    ``steel_class == 'pipe_hsla'`` AND has the four artifact files.
    """
    if not REAL_MODELS_DIR.is_dir():
        return None
    for entry in sorted(REAL_MODELS_DIR.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("steel_class") != "pipe_hsla":
            continue
        if not _REQUIRED_MODEL_ARTIFACTS.issubset(
            {p.name for p in entry.iterdir()}
        ):
            continue
        return entry
    return None


@pytest.fixture()
def empty_models_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Empty MODELS_DIR — for endpoint behaviour tests that don't need a model."""
    fake_dir = tmp_path / "models"
    fake_dir.mkdir()
    monkeypatch.setattr(system_router, "MODELS_DIR", fake_dir)
    monkeypatch.setattr(model_trainer, "MODELS_DIR", fake_dir)
    yield fake_dir


@pytest.fixture()
def populated_models_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """MODELS_DIR with one real HSLA model copied in for predict tests.

    Skips the test if the repo has no HSLA models — caller decides what
    that means (predict-flow tests can't run, but they're not the only
    coverage for the endpoint).
    """
    src = _find_real_hsla_model()
    if src is None:
        pytest.skip("No real HSLA model available in repo to drive predict tests")
    fake_dir = tmp_path / "models"
    fake_dir.mkdir()
    dst = fake_dir / src.name
    shutil.copytree(src, dst)
    monkeypatch.setattr(system_router, "MODELS_DIR", fake_dir)
    monkeypatch.setattr(model_trainer, "MODELS_DIR", fake_dir)
    yield fake_dir


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _midrange_composition(class_id: str) -> dict[str, float]:
    """Build a composition dict of midpoint values from physical_bounds.

    Mirrors Streamlit form defaults (lines 746-748: ``default = (lo + hi) / 2``).
    """
    profile = load_steel_class(class_id)
    return {
        feat: (lo + hi) / 2.0
        for feat, (lo, hi) in profile.physical_bounds.items()
        if feat in profile.feature_set
    }


# ---------- /api/system/steel-classes -------------------------------------


def test_list_steel_classes(client: TestClient) -> None:
    """Endpoint returns the three registered profiles with feature_set."""
    resp = client.get("/api/system/steel-classes")
    assert resp.status_code == 200
    payload = resp.json()
    assert "items" in payload
    assert payload["count"] == len(payload["items"])

    ids = {item["id"] for item in payload["items"]}
    assert ids == {"pipe_hsla", "en10083_qt", "fatigue_carbon_steel"}

    # Shape sanity for one profile.
    hsla = next(item for item in payload["items"] if item["id"] == "pipe_hsla")
    assert isinstance(hsla["feature_set"], list)
    assert "c_pct" in hsla["feature_set"]
    assert isinstance(hsla["physical_bounds"], dict)
    assert isinstance(hsla["target_properties"], list)
    yt = next(t for t in hsla["target_properties"] if t["id"] == "yield_strength_mpa")
    assert isinstance(yt["range"], list) and len(yt["range"]) == 2


# ---------- /api/system/models --------------------------------------------


def test_list_models_empty(empty_models_dir: Path, client: TestClient) -> None:
    resp = client.get("/api/system/models")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["items"] == []
    assert payload["count"] == 0


def test_list_models_returns_meta(empty_models_dir: Path, client: TestClient) -> None:
    """Hand-craft three fake meta.json — exercise the metadata-shape mapping
    without touching XGBoost. We don't need to read real models here, only
    confirm meta keys make it into the response."""
    for i, version in enumerate(["aaa_old", "bbb_mid", "ccc_new"]):
        d = empty_models_dir / version
        d.mkdir()
        (d / "meta.json").write_text(
            json.dumps(
                {
                    "version": version,
                    "target": "yield_strength_mpa",
                    "feature_list": ["c_pct", "mn_pct"],
                    "training_ranges": {"c_pct": [0.04, 0.12]},
                    "metrics": {
                        "r2_test": 0.85 + i * 0.01,
                        "mae_test": 12.0,
                        "rmse_test": 18.0,
                        "coverage_90_ci": 0.91,
                        "n_train": 80,
                        "n_val": 10,
                        "n_test": 10,
                    },
                    "steel_class": "pipe_hsla",
                    "trained_at": f"2026-05-09T1{i}:00:00",
                    "conformal_correction_mpa": 5.0,
                }
            ),
            encoding="utf-8",
        )

    resp = client.get("/api/system/models")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 3
    versions = [item["version"] for item in payload["items"]]
    # sorted ascending by directory name
    assert versions == ["aaa_old", "bbb_mid", "ccc_new"]
    first = payload["items"][0]
    assert first["steel_class"] == "pipe_hsla"
    assert first["target"] == "yield_strength_mpa"
    assert first["metrics"]["r2_test"] == pytest.approx(0.85)
    assert first["feature_list"] == ["c_pct", "mn_pct"]
    # has_ood_detector is False because we didn't write the artifact file.
    assert first["has_ood_detector"] is False


def test_active_model_is_latest(empty_models_dir: Path, client: TestClient) -> None:
    """Active model = last entry of sorted dirs (Streamlit parity)."""
    for version in ["aaa_old", "ccc_latest", "bbb_mid"]:
        d = empty_models_dir / version
        d.mkdir()
        (d / "meta.json").write_text(
            json.dumps(
                {
                    "version": version,
                    "steel_class": "pipe_hsla",
                    "target": "yield_strength_mpa",
                    "feature_list": [],
                    "training_ranges": {},
                    "metrics": {},
                }
            ),
            encoding="utf-8",
        )

    resp = client.get("/api/system/models/active")
    assert resp.status_code == 200
    assert resp.json()["version"] == "ccc_latest"


def test_active_model_fallback_pipe_hsla(
    empty_models_dir: Path, client: TestClient
) -> None:
    """Legacy meta.json without ``steel_class`` falls back to pipe_hsla."""
    d = empty_models_dir / "legacy_v1"
    d.mkdir()
    (d / "meta.json").write_text(
        json.dumps(
            {
                "version": "legacy_v1",
                "target": "yield_strength_mpa",
                "feature_list": [],
                "training_ranges": {},
                "metrics": {},
                # NOTE: no ``steel_class`` key
            }
        ),
        encoding="utf-8",
    )

    resp = client.get("/api/system/models/active")
    assert resp.status_code == 200
    assert resp.json()["steel_class"] == "pipe_hsla"


def test_active_model_404_when_empty(
    empty_models_dir: Path, client: TestClient
) -> None:
    resp = client.get("/api/system/models/active")
    assert resp.status_code == 404
    assert "обуч" in resp.json()["detail"].lower()


# ---------- /api/predict --------------------------------------------------


def test_predict_happy_path(populated_models_dir: Path, client: TestClient) -> None:
    """Full prediction roundtrip on a real HSLA model."""
    version = next(populated_models_dir.iterdir()).name
    composition = _midrange_composition("pipe_hsla")

    resp = client.post(
        "/api/predict",
        json={"model_version": version, "composition": composition},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    # Top-level shape contract.
    assert set(payload.keys()) == {"prediction", "ood", "derived", "model"}

    pred = payload["prediction"]
    assert isinstance(pred["mean"], (int, float))
    assert isinstance(pred["q05"], (int, float))
    assert isinstance(pred["q95"], (int, float))
    assert pred["q05"] <= pred["mean"] <= pred["q95"]
    assert pred["ci_half_width"] == pytest.approx(
        (pred["q95"] - pred["q05"]) / 2.0, abs=1e-6
    )
    assert pred["target_property"] == "yield_strength_mpa"
    assert pred["target_label"]

    # OOD section.
    assert isinstance(payload["ood"]["is_ood"], bool)
    assert isinstance(payload["ood"]["log_density"], (int, float))

    # Derived HSLA features (cev_iiw / pcm / cen / microalloying_sum).
    assert "cev_iiw" in payload["derived"]
    assert "pcm" in payload["derived"]

    # Model echo.
    assert payload["model"]["version"] == version
    assert payload["model"]["steel_class"] == "pipe_hsla"


def test_predict_missing_feature_returns_400(
    populated_models_dir: Path, client: TestClient
) -> None:
    """Composition that doesn't cover feature_set must 400 with a clear
    list of missing keys (callable from the UI's error banner)."""
    version = next(populated_models_dir.iterdir()).name
    full = _midrange_composition("pipe_hsla")
    # Drop two keys.
    full.pop("c_pct")
    full.pop("mn_pct")

    resp = client.post(
        "/api/predict",
        json={"model_version": version, "composition": full},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "c_pct" in detail and "mn_pct" in detail


def test_predict_extra_keys_ignored(
    populated_models_dir: Path, client: TestClient
) -> None:
    """Extra keys must not break the prediction (UI may pre-compute
    derived features that the backend recomputes anyway)."""
    version = next(populated_models_dir.iterdir()).name
    composition = _midrange_composition("pipe_hsla")
    composition["bogus_key"] = 999.0
    composition["cev_iiw"] = 0.42  # would be recomputed inside

    resp = client.post(
        "/api/predict",
        json={"model_version": version, "composition": composition},
    )
    assert resp.status_code == 200, resp.text


def test_predict_unknown_model_404(
    populated_models_dir: Path, client: TestClient
) -> None:
    resp = client.post(
        "/api/predict",
        json={
            "model_version": "no_such_model_v0",
            "composition": _midrange_composition("pipe_hsla"),
        },
    )
    assert resp.status_code == 404
    assert "no_such_model_v0" in resp.json()["detail"]


def test_predict_rejects_path_traversal(
    populated_models_dir: Path, client: TestClient
) -> None:
    """CWE-22 regression: ``../app`` must not let an attacker load XGBoost
    artifacts from a sibling directory. Even an empty composition is enough
    to trigger the guard, which fires before any feature validation."""
    resp = client.post(
        "/api/predict",
        json={"model_version": "../app", "composition": {}},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"].lower()
    assert "model_version" in detail or "invalid" in detail


def test_predict_rejects_absolute_path(
    populated_models_dir: Path, client: TestClient
) -> None:
    """CWE-22 regression: an absolute path silently overrides ``MODELS_DIR``
    in ``Path``-concat semantics, so the regex guard must reject it before
    we ever touch the filesystem."""
    resp = client.post(
        "/api/predict",
        json={"model_version": "/etc/passwd", "composition": {}},
    )
    assert resp.status_code == 400


def test_predict_safe_json_numeric(
    populated_models_dir: Path, client: TestClient
) -> None:
    """Regression: SafeJSONResponse must emit numbers, not strings, for
    numpy scalars produced by predict_with_uncertainty."""
    version = next(populated_models_dir.iterdir()).name
    composition = _midrange_composition("pipe_hsla")

    resp = client.post(
        "/api/predict",
        json={"model_version": version, "composition": composition},
    )
    assert resp.status_code == 200
    payload = resp.json()
    # Sanity: every numeric leaf is a real number, not a string.
    for key in ("mean", "q05", "q95", "ci_half_width"):
        assert isinstance(payload["prediction"][key], (int, float)), (
            f"{key} must be numeric, got {type(payload['prediction'][key])}"
        )
    assert isinstance(payload["ood"]["log_density"], (int, float))
    for k, v in payload["derived"].items():
        assert isinstance(v, (int, float)), f"derived[{k}] must be numeric"
