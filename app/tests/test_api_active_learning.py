"""Tests for /api/active-learning/* (PR 5 of FastAPI migration).

Test isolation strategy mirrors ``test_api_predict.py``:
    - We avoid retraining a model in tests (long-running). Instead we
      copy ONE pre-existing fatigue_carbon_steel model directory from
      the repo's ``models/`` into a temp dir and monkeypatch
      ``app.api.routers.system.MODELS_DIR`` and
      ``app.backend.model_trainer.MODELS_DIR`` to point there.
    - We also redirect the Decision Log SQLite DB into the tmp dir so
      successful runs don't pollute the real ``decision_log/decisions.db``.
    - We use a small ``n_samples`` (200) to keep total runtime under
      ~3 seconds while still exercising the real LHS+EI path on a real
      XGBoost model. The endpoint is documented as supporting up to
      10000 — large values are stress-tested manually, not in pytest.
    - For path-traversal / validation-error tests we don't even need
      a real model: those checks fire before the heavy model load.

Skips cleanly when the repo has no fatigue_carbon_steel model — that
case is covered by ``scripts/smoke_test.py`` in CI.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routers import active_learning as active_learning_router
from app.api.routers import system as system_router
from app.backend import model_trainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_MODELS_DIR = PROJECT_ROOT / "models"

# Required artifact filenames for a complete model directory (mirrors
# ``test_api_predict.py``).
_REQUIRED_MODEL_ARTIFACTS = frozenset(
    {"main.json", "q05.json", "q95.json", "ood_detector.pkl"}
)


def _find_real_fatigue_model() -> Path | None:
    """Locate any complete fatigue_carbon_steel model dir in the repo."""
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
        if meta.get("steel_class") != "fatigue_carbon_steel":
            continue
        if not _REQUIRED_MODEL_ARTIFACTS.issubset(
            {p.name for p in entry.iterdir()}
        ):
            continue
        return entry
    return None


def _find_real_hsla_model() -> Path | None:
    """Locate any complete pipe_hsla model dir — used to check the class gate."""
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
def isolated_decision_log(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict]:
    """Stub Decision Log writes so tests don't touch the real DB.

    Active learning logs ``log_decision(...)`` on every successful run.
    We can't redirect via monkeypatching ``DEFAULT_DB_PATH`` because that
    module-level constant becomes a frozen kwarg default at function
    definition time; instead we replace the ``log_decision`` symbol
    imported into the router module with a list-collector. Tests that
    care can read the captured calls; tests that don't simply benefit
    from a no-side-effects baseline.
    """
    captured: list[dict] = []

    def _fake_log_decision(**kwargs):  # type: ignore[no-untyped-def]
        captured.append(kwargs)
        return -1  # return ID-shape compatible with real signature

    monkeypatch.setattr(
        active_learning_router, "log_decision", _fake_log_decision
    )
    return captured


@pytest.fixture()
def fatigue_models_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_decision_log: list[dict],
) -> Iterator[Path]:
    """MODELS_DIR with one real fatigue model copied in for happy-path tests."""
    src = _find_real_fatigue_model()
    if src is None:
        pytest.skip(
            "No real fatigue_carbon_steel model available in repo to drive tests"
        )
    fake_dir = tmp_path / "models"
    fake_dir.mkdir()
    dst = fake_dir / src.name
    shutil.copytree(src, dst)
    monkeypatch.setattr(system_router, "MODELS_DIR", fake_dir)
    monkeypatch.setattr(model_trainer, "MODELS_DIR", fake_dir)
    yield fake_dir


@pytest.fixture()
def hsla_models_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_decision_log: list[dict],
) -> Iterator[Path]:
    """MODELS_DIR with one real HSLA model — for the class-gate test."""
    src = _find_real_hsla_model()
    if src is None:
        pytest.skip("No real pipe_hsla model available in repo to drive tests")
    fake_dir = tmp_path / "models"
    fake_dir.mkdir()
    dst = fake_dir / src.name
    shutil.copytree(src, dst)
    monkeypatch.setattr(system_router, "MODELS_DIR", fake_dir)
    monkeypatch.setattr(model_trainer, "MODELS_DIR", fake_dir)
    yield fake_dir


@pytest.fixture()
def empty_models_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_decision_log: list[dict],
) -> Iterator[Path]:
    """Empty MODELS_DIR — for tests where the model never loads."""
    fake_dir = tmp_path / "models"
    fake_dir.mkdir()
    monkeypatch.setattr(system_router, "MODELS_DIR", fake_dir)
    monkeypatch.setattr(model_trainer, "MODELS_DIR", fake_dir)
    yield fake_dir


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# ──────────────────── happy path ────────────────────


def test_propose_happy_path(
    fatigue_models_dir: Path, client: TestClient
) -> None:
    """Valid body → 200 with a non-empty ranked candidate list."""
    version = next(fatigue_models_dir.iterdir()).name

    resp = client.post(
        "/api/active-learning/propose",
        json={
            "model_version": version,
            "n_samples": 200,
            "top_k": 5,
            "seed": 42,
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    # Top-level shape contract.
    assert set(payload.keys()) == {
        "candidates", "baseline", "model", "config", "computation_time_s",
    }

    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    assert len(candidates) == 5  # top_k respected on a real run

    # Per-candidate shape sanity.
    first = candidates[0]
    for key in (
        "rank", "id", "composition", "process_params", "predicted_property",
        "lower_90", "upper_90", "uncertainty_width", "is_ood",
        "estimated_cost_eur_per_t", "expected_improvement",
        "acquisition_score", "delta_vs_baseline_property",
        "delta_vs_baseline_cost",
    ):
        assert key in first, f"missing key in candidate: {key}"

    assert first["rank"] == 1
    # Sorted descending by acquisition_score.
    scores = [c["acquisition_score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_propose_n_candidates_respected(
    fatigue_models_dir: Path, client: TestClient
) -> None:
    """``top_k=3`` must produce exactly 3 candidates."""
    version = next(fatigue_models_dir.iterdir()).name

    resp = client.post(
        "/api/active-learning/propose",
        json={
            "model_version": version,
            "n_samples": 200,
            "top_k": 3,
            "seed": 7,
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert len(payload["candidates"]) == 3
    assert [c["rank"] for c in payload["candidates"]] == [1, 2, 3]


def test_propose_response_includes_baseline_and_model(
    fatigue_models_dir: Path, client: TestClient
) -> None:
    """``baseline`` + ``model`` blocks must echo the data the UI needs."""
    version = next(fatigue_models_dir.iterdir()).name

    resp = client.post(
        "/api/active-learning/propose",
        json={"model_version": version, "n_samples": 200, "top_k": 3},
    )
    assert resp.status_code == 200
    payload = resp.json()

    baseline = payload["baseline"]
    assert "predicted_property" in baseline
    assert "cost_per_ton" in baseline
    assert "f_star" in baseline
    assert isinstance(baseline["predicted_property"], (int, float))
    assert isinstance(baseline["cost_per_ton"], (int, float))
    # f* (max in dataset) must dominate baseline median prediction.
    assert baseline["f_star"] >= baseline["predicted_property"]

    model = payload["model"]
    assert model["version"] == version
    assert model["steel_class"] == "fatigue_carbon_steel"
    assert model["target"] == "fatigue_strength_mpa"


def test_propose_ood_flag_per_candidate(
    fatigue_models_dir: Path, client: TestClient
) -> None:
    """Every candidate carries a boolean ``is_ood`` (UI badge driver)."""
    version = next(fatigue_models_dir.iterdir()).name

    resp = client.post(
        "/api/active-learning/propose",
        json={"model_version": version, "n_samples": 200, "top_k": 5},
    )
    assert resp.status_code == 200
    payload = resp.json()
    for c in payload["candidates"]:
        assert isinstance(c["is_ood"], bool), (
            f"is_ood must be bool, got {type(c['is_ood'])}"
        )


def test_propose_safe_json_numeric(
    fatigue_models_dir: Path, client: TestClient
) -> None:
    """Regression: SafeJSONResponse emits real numbers for numpy scalars."""
    version = next(fatigue_models_dir.iterdir()).name

    resp = client.post(
        "/api/active-learning/propose",
        json={"model_version": version, "n_samples": 200, "top_k": 3},
    )
    assert resp.status_code == 200
    payload = resp.json()

    # Top-level numeric leaves.
    assert isinstance(payload["computation_time_s"], (int, float))
    for key in ("predicted_property", "cost_per_ton", "f_star"):
        assert isinstance(payload["baseline"][key], (int, float)), (
            f"baseline[{key}] must be numeric"
        )

    # Per-candidate numeric leaves.
    for c in payload["candidates"]:
        for key in (
            "predicted_property", "lower_90", "upper_90",
            "uncertainty_width", "estimated_cost_eur_per_t",
            "expected_improvement", "acquisition_score",
            "delta_vs_baseline_property", "delta_vs_baseline_cost",
        ):
            assert isinstance(c[key], (int, float)), (
                f"candidate[{key}] must be numeric, got {type(c[key])}"
            )


# ──────────────────── error paths ────────────────────


def test_propose_unknown_model_404(
    empty_models_dir: Path, client: TestClient
) -> None:
    """Non-existent model_version → 404 with informative detail."""
    resp = client.post(
        "/api/active-learning/propose",
        json={"model_version": "no_such_model_v0", "top_k": 5, "n_samples": 200},
    )
    assert resp.status_code == 404
    assert "no_such_model_v0" in resp.json()["detail"]


def test_propose_path_traversal_400(
    empty_models_dir: Path, client: TestClient
) -> None:
    """CWE-22 regression: ``../app`` must not load XGBoost from outside MODELS_DIR.

    The 400 fires from ``_safe_version_dir`` before any Pydantic body
    validation succeeds for the rest of the request, so we don't need
    a real model in the fixture.
    """
    resp = client.post(
        "/api/active-learning/propose",
        json={"model_version": "../app", "top_k": 5, "n_samples": 200},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"].lower()
    assert "model_version" in detail or "invalid" in detail


def test_propose_absolute_path_400(
    empty_models_dir: Path, client: TestClient
) -> None:
    """CWE-22: absolute path in model_version is silently swallowed by
    ``Path("models") / "/etc/passwd"`` → ``/etc/passwd``. Regex guard
    must reject before any filesystem touch."""
    resp = client.post(
        "/api/active-learning/propose",
        json={"model_version": "/etc/passwd", "top_k": 5, "n_samples": 200},
    )
    assert resp.status_code == 400


def test_propose_invalid_top_k_low_422(
    empty_models_dir: Path, client: TestClient
) -> None:
    """``top_k=0`` violates Pydantic ``ge=1`` → 422."""
    resp = client.post(
        "/api/active-learning/propose",
        json={"model_version": "anything", "top_k": 0, "n_samples": 200},
    )
    assert resp.status_code == 422


def test_propose_invalid_top_k_high_422(
    empty_models_dir: Path, client: TestClient
) -> None:
    """``top_k=999`` violates ``le=50`` → 422."""
    resp = client.post(
        "/api/active-learning/propose",
        json={"model_version": "anything", "top_k": 999, "n_samples": 200},
    )
    assert resp.status_code == 422


def test_propose_invalid_n_samples_low_422(
    empty_models_dir: Path, client: TestClient
) -> None:
    """``n_samples=0`` violates ``ge=100`` → 422."""
    resp = client.post(
        "/api/active-learning/propose",
        json={"model_version": "anything", "n_samples": 0, "top_k": 5},
    )
    assert resp.status_code == 422


def test_propose_invalid_n_samples_high_422(
    empty_models_dir: Path, client: TestClient
) -> None:
    """``n_samples=999999`` violates ``le=10000`` → 422."""
    resp = client.post(
        "/api/active-learning/propose",
        json={"model_version": "anything", "n_samples": 999999, "top_k": 5},
    )
    assert resp.status_code == 422


def test_propose_class_gate_blocks_hsla(
    hsla_models_dir: Path, client: TestClient
) -> None:
    """HSLA models must be rejected with 400 + clear message.

    Non-fatigue classes are not supported by the active_learner pipeline
    (no matching real-world dataset wired in). The API surfaces this
    restriction as a 400 — explicit failure beats silent confusion.
    """
    version = next(hsla_models_dir.iterdir()).name

    resp = client.post(
        "/api/active-learning/propose",
        json={"model_version": version, "n_samples": 200, "top_k": 5},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"].lower()
    assert "fatigue_carbon_steel" in detail
