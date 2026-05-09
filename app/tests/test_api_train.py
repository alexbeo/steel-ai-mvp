"""Tests for /api/train/* — XGBoost + Optuna training pipeline (PR 8).

Test isolation strategy:
    - We monkeypatch ``app.backend.model_trainer.MODELS_DIR`` to a tmp
      path so every training run lands in an isolated directory; the
      real ``models/`` is never mutated.
    - We use **small** Optuna budgets (n_trials=5) and **small** synthetic
      datasets (n_samples=200) for the e2e test so it stays under ~60 s
      per run on a developer laptop.
    - The real-data path is exercised via ``test_train_real_fatigue_data``
      which loads the 437-record Agrawal NIMS parquet (must already exist
      in ``data/agrawal_nims_fatigue.parquet`` — skipped otherwise). The
      generator profile decides real vs synthetic; n_samples is silently
      ignored for fatigue.
    - The cancel test exercises the per-trial Optuna cancellation hook
      end-to-end (PR 8 review fix): submit real training → DELETE shortly
      after → poll until status=error within ~10 s (one Optuna trial).
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.jobs import get_job_store
from app.api.main import app
from app.api.routers import system as system_router
from app.backend import model_trainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGRAWAL_PARQUET = PROJECT_ROOT / "data" / "agrawal_nims_fatigue.parquet"


@pytest.fixture(autouse=True)
def reset_job_store_singleton():
    """Guarantee fresh JobStore per test — avoids cross-test pollution."""
    get_job_store.cache_clear()
    yield
    get_job_store.cache_clear()


@pytest.fixture()
def isolated_models_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Redirect ``model_trainer.MODELS_DIR`` to tmp so artifacts don't leak.

    Both the trainer (writes there) and the system router (reads listings
    from there) need the same MODELS_DIR view, so we patch both modules.
    """
    fake_dir = tmp_path / "models"
    fake_dir.mkdir()
    monkeypatch.setattr(model_trainer, "MODELS_DIR", fake_dir)
    monkeypatch.setattr(system_router, "MODELS_DIR", fake_dir)
    yield fake_dir


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _wait_for_job(
    client: TestClient,
    job_id: str,
    timeout: float = 120.0,
    interval: float = 1.0,
) -> dict:
    """Poll GET /api/jobs/{id} until terminal status or timeout.

    Larger default timeout (120 s) than test_api_design because XGBoost
    + Optuna with 5 trials on 200 rows still takes ~20-40 s on slower CI.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(interval)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


# ─────────────────────────────────────────────────────────────────────
# /api/train/run — submission + validation
# ─────────────────────────────────────────────────────────────────────


def test_train_run_returns_job_id(
    isolated_models_dir: Path, client: TestClient
) -> None:
    """POST /run with valid body → {job_id, config} shape."""
    resp = client.post(
        "/api/train/run",
        json={
            "class_id": "pipe_hsla",
            "n_samples": 200,
            "n_trials": 5,
            "seed": 42,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "job_id" in body
    assert isinstance(body["job_id"], str) and len(body["job_id"]) >= 16
    assert body["config"]["class_id"] == "pipe_hsla"
    assert body["config"]["n_trials"] == 5

    # Tear down the running job — store ref so we don't need to wait.
    store = get_job_store()
    store.request_cancel(body["job_id"])


def test_train_run_invalid_class_422(
    isolated_models_dir: Path, client: TestClient
) -> None:
    """class_id outside the Literal enum is rejected at validation layer."""
    resp = client.post(
        "/api/train/run",
        json={
            "class_id": "invalid_class",
            "n_samples": 200,
            "n_trials": 5,
        },
    )
    assert resp.status_code == 422, resp.text
    # FastAPI renders {"detail": [...errors...]} for validation failures.
    assert "class_id" in resp.text or "literal_error" in resp.text


def test_train_run_invalid_n_trials_422(
    isolated_models_dir: Path, client: TestClient
) -> None:
    """n_trials below ge=5 or above le=200 must reject before scheduling."""
    for n_trials in (0, 4, 999):
        resp = client.post(
            "/api/train/run",
            json={
                "class_id": "pipe_hsla",
                "n_trials": n_trials,
            },
        )
        assert resp.status_code == 422, (
            f"n_trials={n_trials} unexpectedly accepted; status={resp.status_code}"
        )


def test_train_run_invalid_n_samples_422(
    isolated_models_dir: Path, client: TestClient
) -> None:
    """n_samples below ge=100 or above le=5000 rejected."""
    for n_samples in (50, 99, 5001, 9999):
        resp = client.post(
            "/api/train/run",
            json={
                "class_id": "pipe_hsla",
                "n_samples": n_samples,
                "n_trials": 5,
            },
        )
        assert resp.status_code == 422, f"n_samples={n_samples} unexpectedly accepted"


def test_train_run_to_completion(
    isolated_models_dir: Path, client: TestClient
) -> None:
    """Submit a tiny training run → poll → verify result dict shape."""
    resp = client.post(
        "/api/train/run",
        json={
            "class_id": "pipe_hsla",
            "n_samples": 200,
            "n_trials": 5,
            "seed": 42,
        },
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    body = _wait_for_job(client, job_id, timeout=120.0)
    if body["status"] != "done":
        pytest.fail(
            f"job ended with status={body['status']} error={body.get('error')!r}"
        )

    result = body["result"]
    assert "version" in result
    assert isinstance(result["version"], str) and len(result["version"]) > 0
    assert result["class_id"] == "pipe_hsla"
    assert result["target"] == "yield_strength_mpa"

    # Metrics: must be numeric (Risks #3 — SafeJSONResponse path).
    metrics = result["metrics"]
    assert isinstance(metrics["r2_test"], (int, float))
    assert isinstance(metrics["mae_test"], (int, float))
    assert isinstance(metrics["coverage_90_ci"], (int, float))
    assert metrics["n_train"] > 0
    assert metrics["n_test"] > 0

    # Feature importance: non-empty, sorted desc, items shaped right.
    fi = result["feature_importance"]
    assert isinstance(fi, list) and len(fi) > 0
    for item in fi:
        assert "feature" in item and "importance" in item
        assert isinstance(item["importance"], (int, float))
    # Backend sorts desc — verify the chart-friendly order survived.
    importances = [item["importance"] for item in fi]
    assert importances == sorted(importances, reverse=True)

    # Critic block. pattern_warnings always a list; llm_observations
    # is None when no ANTHROPIC_API_KEY (most CI runs).
    critic = result["critic"]
    assert isinstance(critic["pattern_warnings"], list)
    assert critic["llm_observations"] is None or isinstance(
        critic["llm_observations"], list
    )

    # Duration is recorded.
    assert isinstance(result["duration_s"], (int, float))
    assert result["duration_s"] > 0


def test_train_creates_model_directory(
    isolated_models_dir: Path, client: TestClient
) -> None:
    """After completion, models/<version>/ contains the four artefacts.

    Confirms the worker actually persisted via model_trainer (not just
    returned a stubbed dict) and that the patched MODELS_DIR was used.
    """
    resp = client.post(
        "/api/train/run",
        json={
            "class_id": "pipe_hsla",
            "n_samples": 200,
            "n_trials": 5,
            "seed": 42,
        },
    )
    job_id = resp.json()["job_id"]
    body = _wait_for_job(client, job_id, timeout=120.0)
    if body["status"] != "done":
        pytest.fail(
            f"job ended with status={body['status']} error={body.get('error')!r}"
        )
    version = body["result"]["version"]
    version_dir = isolated_models_dir / version
    assert version_dir.is_dir(), f"missing version dir: {version_dir}"
    for fname in ("meta.json", "main.json", "q05.json", "q95.json"):
        assert (version_dir / fname).is_file(), f"missing {fname}"
    # ood_detector.pkl is also written; assert it for completeness.
    assert (version_dir / "ood_detector.pkl").is_file()

    # meta.json must declare the steel class so downstream UI picks
    # the right profile when listing models.
    meta = json.loads((version_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["steel_class"] == "pipe_hsla"
    assert meta["target"] == "yield_strength_mpa"


def test_train_safe_json_numeric(
    isolated_models_dir: Path, client: TestClient
) -> None:
    """SafeJSONResponse must serialise numpy floats as JSON numbers (Risks #3)."""
    resp = client.post(
        "/api/train/run",
        json={
            "class_id": "pipe_hsla",
            "n_samples": 200,
            "n_trials": 5,
            "seed": 42,
        },
    )
    job_id = resp.json()["job_id"]
    body = _wait_for_job(client, job_id, timeout=120.0)
    if body["status"] != "done":
        pytest.fail(
            f"job ended with status={body['status']} error={body.get('error')!r}"
        )
    result = body["result"]

    # No string-encoded numpy scalars allowed in any of these slots.
    for key in ("r2_test", "mae_test", "rmse_test", "coverage_90_ci",
                "conformal_correction_mpa"):
        v = result["metrics"][key]
        assert isinstance(v, (int, float)) and not isinstance(v, bool), (
            f"metrics.{key} should be numeric, got {type(v).__name__}"
        )

    # feature_importance: each importance should be numeric, not string.
    for item in result["feature_importance"]:
        assert isinstance(item["importance"], (int, float))


@pytest.mark.skipif(
    not AGRAWAL_PARQUET.is_file(),
    reason="data/agrawal_nims_fatigue.parquet missing — run scripts/fetch_agrawal_nims_fatigue.py",
)
def test_train_real_fatigue_data(
    isolated_models_dir: Path, client: TestClient
) -> None:
    """End-to-end with the real 437-record Agrawal NIMS dataset.

    This is the primary "production data" path (CLAUDE.md: "первая
    production-модель на real-world data"). We use n_trials=5 to keep
    the run reasonable; the dataset itself is fixed at 437 rows.
    """
    resp = client.post(
        "/api/train/run",
        json={
            "class_id": "fatigue_carbon_steel",
            "n_trials": 5,
            "seed": 42,
        },
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    body = _wait_for_job(client, job_id, timeout=180.0)
    if body["status"] != "done":
        pytest.fail(
            f"job ended with status={body['status']} error={body.get('error')!r}"
        )
    result = body["result"]
    assert result["class_id"] == "fatigue_carbon_steel"
    assert result["target"] == "fatigue_strength_mpa"
    # Real data has 437 records → after time-based split 80/15/5 ish.
    # n_samples in the request is ignored for fatigue; the generator
    # always loads the full 437-record Agrawal NIMS parquet.
    metrics = result["metrics"]
    assert metrics["n_train"] + metrics["n_val"] + metrics["n_test"] <= 437
    assert metrics["n_train"] > 0


def test_train_cancel_running_job(
    isolated_models_dir: Path, client: TestClient
) -> None:
    """Submit real training → DELETE → confirm status=error within ~10 s.

    PR 8 review fix: ``train_model`` now accepts a ``cancellation_callback``
    that is wired through ``_check_cancelled(progress)``. Optuna calls our
    ``_maybe_stop`` callback after every trial; once the flag is set,
    ``study.stop()`` exits before the next trial. End-to-end latency from
    DELETE → status=error is bounded by one Optuna trial duration (~1-3 s
    on the smoke dataset).

    We use n_trials=20 to ensure several trials occur after DELETE. This
    test exercises the real per-trial hook — a stub fn would not catch a
    regression where the kwarg stops being forwarded from the router to
    the trainer.
    """
    resp = client.post(
        "/api/train/run",
        json={
            "class_id": "pipe_hsla",
            "n_samples": 200,
            "n_trials": 20,
            "seed": 42,
        },
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    # Give the worker enough time to enter the Optuna phase (progress
    # 0.25 milestone) — dataset gen + feature eng + first trial start.
    # 4 s is comfortably above the dataset generation time on CI.
    time.sleep(4.0)
    del_resp = client.delete(f"/api/jobs/{job_id}")
    assert del_resp.status_code == 200, del_resp.text
    assert del_resp.json()["cancellation_requested"] is True

    # Per-trial check fires after each Optuna trial; status flips to
    # error within ~1-3 s of DELETE on this dataset. We give it 30 s to
    # accommodate slow CI; in practice it lands far sooner.
    body = _wait_for_job(client, job_id, timeout=30.0, interval=0.5)
    assert body["status"] == "error", body
    err = (body.get("error") or "").lower()
    assert "cancel" in err or "stopped" in err, (
        f"expected cancellation marker in error, got: {body.get('error')!r}"
    )
