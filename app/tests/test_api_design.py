"""Tests for /api/design/* and /api/prices/* (PR 7 of FastAPI migration).

PR 7 introduces the most-complex endpoint set in the migration: NSGA-II
inverse design (~40 s through the JobStore + polling) plus the price
snapshot CRUD that drives the cost objective. We use the same isolation
pattern as test_api_predict — copy ONE pre-existing HSLA model into a
tmp dir and monkeypatch ``MODELS_DIR`` so tests stay fast and don't
mutate the repo state.

Long-running NSGA-II is shrunk to ``population_size=10, n_generations=2``
in the end-to-end test — sufficient to exercise the result shape without
burning ~40 s per CI run.
"""
from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from app.api.jobs import JobStatus, get_job_store
from app.api.main import app
from app.api.routers import prices as prices_router
from app.api.routers import system as system_router
from app.backend import model_trainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_MODELS_DIR = PROJECT_ROOT / "models"

_REQUIRED_MODEL_ARTIFACTS = frozenset(
    {"main.json", "q05.json", "q95.json", "ood_detector.pkl"}
)


def _find_real_hsla_model() -> Path | None:
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


def _find_real_non_hsla_model() -> Path | None:
    """Locate a non-HSLA model dir for the class-gate test.

    Picks any model whose steel_class is neither HSLA nor fatigue (i.e.
    Q&T or future classes) — these are the ones the gate must reject.
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
        if meta.get("steel_class") in (None, "pipe_hsla", "fatigue_carbon_steel"):
            continue
        if not _REQUIRED_MODEL_ARTIFACTS.issubset(
            {p.name for p in entry.iterdir()}
        ):
            continue
        return entry
    return None


def _find_real_fatigue_model() -> Path | None:
    """Locate a fatigue_carbon_steel model dir for design tests."""
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


@pytest.fixture(autouse=True)
def reset_job_store_singleton():
    """Guarantee fresh JobStore per test — avoids cross-test pollution."""
    get_job_store.cache_clear()
    yield
    get_job_store.cache_clear()


@pytest.fixture()
def populated_models_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Tmp MODELS_DIR with one real HSLA model copied in."""
    src = _find_real_hsla_model()
    if src is None:
        pytest.skip("No real HSLA model in repo to drive design tests")
    fake_dir = tmp_path / "models"
    fake_dir.mkdir()
    dst = fake_dir / src.name
    shutil.copytree(src, dst)
    monkeypatch.setattr(system_router, "MODELS_DIR", fake_dir)
    monkeypatch.setattr(model_trainer, "MODELS_DIR", fake_dir)
    yield fake_dir


@pytest.fixture()
def populated_non_hsla_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Tmp MODELS_DIR with one Q&T-or-similar (non-HSLA, non-fatigue) model.

    Used to exercise the class gate — fatigue is now also supported, so
    we deliberately synthesise an ``en10083_qt`` stub if the real repo
    only has fatigue + HSLA classes.
    """
    src = _find_real_non_hsla_model()
    if src is None:
        # Synthesise a minimal en10083_qt meta.json — class gate fires on
        # meta read, so we don't need real artifacts.
        fake_dir = tmp_path / "models"
        fake_dir.mkdir()
        d = fake_dir / "fake_qt_v1"
        d.mkdir()
        (d / "meta.json").write_text(
            json.dumps(
                {
                    "version": "fake_qt_v1",
                    "steel_class": "en10083_qt",
                    "target": "yield_strength_mpa",
                    "feature_list": [],
                    "training_ranges": {},
                    "metrics": {},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(system_router, "MODELS_DIR", fake_dir)
        monkeypatch.setattr(model_trainer, "MODELS_DIR", fake_dir)
        yield fake_dir
        return
    fake_dir = tmp_path / "models"
    fake_dir.mkdir()
    dst = fake_dir / src.name
    shutil.copytree(src, dst)
    monkeypatch.setattr(system_router, "MODELS_DIR", fake_dir)
    monkeypatch.setattr(model_trainer, "MODELS_DIR", fake_dir)
    yield fake_dir


@pytest.fixture()
def populated_fatigue_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Tmp MODELS_DIR with one real fatigue_carbon_steel model copied in."""
    src = _find_real_fatigue_model()
    if src is None:
        pytest.skip("No real fatigue_carbon_steel model in repo")
    fake_dir = tmp_path / "models"
    fake_dir.mkdir()
    dst = fake_dir / src.name
    shutil.copytree(src, dst)
    monkeypatch.setattr(system_router, "MODELS_DIR", fake_dir)
    monkeypatch.setattr(model_trainer, "MODELS_DIR", fake_dir)
    yield fake_dir


@pytest.fixture()
def isolated_price_snapshots_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Redirect the prices.upload save target to a tmp dir."""
    fake_dir = tmp_path / "price_snapshots"
    fake_dir.mkdir()
    monkeypatch.setattr(prices_router, "PRICE_SNAPSHOTS_DIR", fake_dir)
    monkeypatch.setattr(prices_router, "PROJECT_ROOT", tmp_path)
    yield fake_dir


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _wait_for_job(
    client: TestClient, job_id: str, timeout: float = 60.0, interval: float = 0.5
) -> dict:
    """Poll GET /api/jobs/{id} until terminal status or timeout."""
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
# /api/design/run — submission
# ─────────────────────────────────────────────────────────────────────


def test_design_run_returns_job_id(
    populated_models_dir: Path, client: TestClient
) -> None:
    """POST /run → {job_id} with a valid hex slug."""
    version = next(populated_models_dir.iterdir()).name
    resp = client.post(
        "/api/design/run",
        json={
            "model_version": version,
            "target": {"min": 485, "max": 580},
            "population_size": 10,
            "n_generations": 1,
            "include_cost": False,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "job_id" in body
    job_id = body["job_id"]
    assert isinstance(job_id, str) and len(job_id) >= 16
    assert body["model_version"] == version
    assert body["steel_class"] == "pipe_hsla"
    assert "config" in body and "targets" in body["config"]


def test_design_run_path_traversal_400(
    populated_models_dir: Path, client: TestClient
) -> None:
    """CWE-22 regression — ../app rejected by _safe_version_dir."""
    resp = client.post(
        "/api/design/run",
        json={
            "model_version": "../app",
            "target": {"min": 485, "max": 580},
            "include_cost": False,
        },
    )
    assert resp.status_code == 400
    assert "model_version" in resp.json()["detail"].lower() or "invalid" in resp.json()["detail"].lower()


def test_design_run_invalid_target_range_422(
    populated_models_dir: Path, client: TestClient
) -> None:
    """target.min ≥ target.max must reject before scheduling NSGA-II."""
    version = next(populated_models_dir.iterdir()).name
    resp = client.post(
        "/api/design/run",
        json={
            "model_version": version,
            "target": {"min": 600, "max": 500},
            "include_cost": False,
        },
    )
    # Pydantic model_validator runs at body parsing → 422.
    assert resp.status_code == 422
    assert "target" in resp.text.lower()


def test_design_run_invalid_n_gen_422(
    populated_models_dir: Path, client: TestClient
) -> None:
    """n_generations outside [1, 200] rejected by Field(ge/le)."""
    version = next(populated_models_dir.iterdir()).name
    for n_gen in (0, 999):
        resp = client.post(
            "/api/design/run",
            json={
                "model_version": version,
                "target": {"min": 485, "max": 580},
                "n_generations": n_gen,
                "include_cost": False,
            },
        )
        assert resp.status_code == 422, f"n_generations={n_gen} unexpectedly accepted"


def test_design_run_invalid_pop_size_422(
    populated_models_dir: Path, client: TestClient
) -> None:
    """population_size outside [10, 200] rejected."""
    version = next(populated_models_dir.iterdir()).name
    resp = client.post(
        "/api/design/run",
        json={
            "model_version": version,
            "target": {"min": 485, "max": 580},
            "population_size": 5,  # below ge=10
            "include_cost": False,
        },
    )
    assert resp.status_code == 422


def test_design_run_class_gate_400(
    populated_non_hsla_dir: Path, client: TestClient
) -> None:
    """Q&T (non-HSLA, non-fatigue) is rejected at submit time, not silently in worker."""
    version = next(populated_non_hsla_dir.iterdir()).name
    resp = client.post(
        "/api/design/run",
        json={
            "model_version": version,
            "target": {"min": 485, "max": 580},
            "include_cost": False,
        },
    )
    assert resp.status_code == 400
    # Detail must mention HSLA (historical assertion) AND surface the v2
    # backlog hint for the user.
    detail = resp.json()["detail"]
    assert "HSLA" in detail
    assert "fatigue_carbon_steel" in detail or "v2" in detail


def test_design_run_fatigue_returns_job_id(
    populated_fatigue_dir: Path, client: TestClient
) -> None:
    """Fatigue model: POST /run → {job_id} + steel_class echoed."""
    version = next(populated_fatigue_dir.iterdir()).name
    resp = client.post(
        "/api/design/run",
        json={
            "model_version": version,
            "target": {"min": 450, "max": 700},
            "population_size": 10,
            "n_generations": 1,
            "include_cost": False,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "job_id" in body
    assert body["model_version"] == version
    assert body["steel_class"] == "fatigue_carbon_steel"
    # Hard constraints must be empty for fatigue (CEV/Pcm dropped at API
    # layer — they don't apply to non-HSLA classes).
    assert body["config"]["hard_constraints"] == {}
    assert body["config"]["cev_max"] is None
    assert body["config"]["pcm_max"] is None
    # Bounds must come from the YAML profile, not VARIABLE_BOUNDS_HSLA.
    assert "carburizing_temp_c" in body["config"]["variable_bounds"]
    assert "rolling_finish_temp" not in body["config"]["variable_bounds"]


def test_design_run_fatigue_to_completion(
    populated_fatigue_dir: Path, client: TestClient
) -> None:
    """Tiny NSGA-II run on the fatigue class — verifies bounds + FE wiring.

    pop=10, gen=2 keeps wall-clock under ~10s. We assert candidates exist
    and the result payload exposes the fatigue steel_class so the UI can
    pick the right labels.
    """
    version = next(populated_fatigue_dir.iterdir()).name
    resp = client.post(
        "/api/design/run",
        json={
            "model_version": version,
            "target": {"min": 450, "max": 700},
            "population_size": 10,
            "n_generations": 2,
            "include_cost": False,
        },
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    body = _wait_for_job(client, job_id, timeout=120.0)
    if body["status"] != "done":
        pytest.fail(
            f"fatigue job ended with status={body['status']} "
            f"error={body.get('error')!r}"
        )

    result = body["result"]
    assert "candidates" in result
    assert isinstance(result["candidates"], list)
    assert len(result["candidates"]) > 0, "Pareto front is empty for fatigue"
    assert result["model"]["steel_class"] == "fatigue_carbon_steel"
    # Each candidate must carry the canonical numeric fields the chart
    # depends on, even when the derived dict is empty (HSLA-only keys).
    for c in result["candidates"]:
        assert "predicted_mean" in c and isinstance(c["predicted_mean"], (int, float))
        assert "is_ood" in c
        assert "validation" in c
        # Composition uses _pct keys regardless of class.
        assert "c_pct" in c["composition"]
        # Processing carries the fatigue-specific knobs.
        assert "carburizing_temp_c" in c["processing"]


def test_design_run_qt_class_gate_v2_message(
    populated_non_hsla_dir: Path, client: TestClient
) -> None:
    """en10083_qt explicitly marked as v2 backlog in the gate message."""
    version = next(populated_non_hsla_dir.iterdir()).name
    resp = client.post(
        "/api/design/run",
        json={
            "model_version": version,
            "target": {"min": 485, "max": 580},
            "include_cost": False,
        },
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    # Either fatigue is mentioned (the new-supported sibling) or v2 backlog.
    assert "v2" in detail or "fatigue" in detail.lower()


def test_design_run_unknown_model_404(
    populated_models_dir: Path, client: TestClient
) -> None:
    resp = client.post(
        "/api/design/run",
        json={
            "model_version": "nope_v0",
            "target": {"min": 485, "max": 580},
            "include_cost": False,
        },
    )
    assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────────────
# end-to-end: submit + poll job + result shape
# ─────────────────────────────────────────────────────────────────────


def test_design_run_to_completion(
    populated_models_dir: Path, client: TestClient
) -> None:
    """Submit a tiny NSGA-II run → poll → verify result dict shape."""
    version = next(populated_models_dir.iterdir()).name
    resp = client.post(
        "/api/design/run",
        json={
            "model_version": version,
            "target": {"min": 485, "max": 580},
            "population_size": 10,
            "n_generations": 2,
            "include_cost": False,
        },
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    body = _wait_for_job(client, job_id, timeout=90.0)
    if body["status"] != "done":
        pytest.fail(f"job ended with status={body['status']} error={body.get('error')!r}")

    result = body["result"]
    assert "candidates" in result
    assert isinstance(result["candidates"], list)
    assert "baseline" in result
    assert "model" in result
    assert result["model"]["version"] == version
    assert result["model"]["steel_class"] == "pipe_hsla"
    # candidate shape sanity check
    for c in result["candidates"]:
        assert "rank" in c and "predicted_mean" in c
        assert "cost_eur_per_t" in c
        assert "in_spec" in c and "is_ood" in c
        assert "composition" in c
        assert "validation" in c


# ─────────────────────────────────────────────────────────────────────
# DELETE /api/jobs/{id} — cooperative cancellation
# ─────────────────────────────────────────────────────────────────────


def test_design_cancel_running_job(client: TestClient) -> None:
    """Submit a slow job → DELETE → confirm flag/state transitions.

    We submit through the JobStore directly (not a real NSGA-II run) so
    the test stays fast and deterministic.
    """
    import threading

    proceed = threading.Event()
    started = threading.Event()

    def slow_work(progress=None) -> str:
        started.set()
        if progress:
            progress(0.1, "long-running")
        proceed.wait(timeout=5.0)
        return "done"

    store = get_job_store()
    job_id = store.create(slow_work)

    # Wait for the worker to actually start before cancelling.
    started.wait(timeout=2.0)

    resp = client.delete(f"/api/jobs/{job_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["job_id"] == job_id
    # Running fn cannot be cancelled by Future.cancel(), but the flag must be set.
    assert body["cancellation_requested"] is True

    # Let the worker finish so its result lands.
    proceed.set()


def test_design_cancel_pending_job(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pending job (worker pool exhausted) flips to ERROR via Future.cancel().

    The router imports ``get_job_store`` by name (``from app.api.jobs import
    get_job_store``), so we have to patch BOTH modules — the source
    binding plus the router's local re-binding — for the DELETE endpoint
    to see our private single-worker store.
    """
    import threading

    from app.api.jobs import JobStore
    from app.api.routers import jobs as jobs_router

    blocker_event = threading.Event()
    proceed_event = threading.Event()

    def block_worker(progress=None) -> str:
        blocker_event.set()
        proceed_event.wait(timeout=5.0)
        return "blocker-done"

    store = JobStore(max_workers=1)

    def _store() -> JobStore:
        return store

    # Patch both the source module and the re-imported binding in the
    # router that the DELETE handler actually invokes.
    monkeypatch.setattr("app.api.jobs.get_job_store", _store)
    monkeypatch.setattr(jobs_router, "get_job_store", _store)

    try:
        # Job 1 occupies the only worker.
        store.create(block_worker)
        blocker_event.wait(timeout=2.0)

        # Job 2 stays PENDING.
        pending_id = store.create(lambda: "never")
        for _ in range(20):
            j = store.get(pending_id)
            if j and j.status is JobStatus.PENDING:
                break
            time.sleep(0.01)

        resp = client.delete(f"/api/jobs/{pending_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "error"
        assert "Cancelled" in (body["error"] or "")
        assert body["cancellation_requested"] is True
    finally:
        proceed_event.set()


def test_design_cancel_unknown_job_404(client: TestClient) -> None:
    resp = client.delete("/api/jobs/does-not-exist")
    assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────────────
# /api/prices
# ─────────────────────────────────────────────────────────────────────


def test_prices_active_returns_seed(client: TestClient) -> None:
    """GET /active returns the seed snapshot in the canonical shape."""
    resp = client.get("/api/prices/active")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["currency"] == "EUR"
    assert isinstance(body.get("date"), str)
    assert "materials" in body
    # Seed must include scrap (base) and the standard ferroalloys.
    assert "scrap" in body["materials"]
    assert "FeMn-80" in body["materials"]
    scrap = body["materials"]["scrap"]
    assert scrap["kind"] == "base"
    assert scrap["price_per_kg"] > 0
    assert "Fe" in scrap["element_content"]


def test_prices_upload_persists_to_yaml(
    isolated_price_snapshots_dir: Path, client: TestClient
) -> None:
    """POST /upload writes a YAML and round-trips through cost_model.load_snapshot."""
    body = {
        "date": "2026-05-09",
        "currency": "EUR",
        "source": "test",
        "materials": {
            "scrap": {
                "kind": "base",
                "price_per_kg": 0.40,
                "element_content": {"Fe": 1.0},
            },
            "FeMn-80": {
                "kind": "ferroalloy",
                "price_per_kg": 1.85,
                "element_content": {"Mn": 0.80, "Fe": 0.20},
            },
        },
    }
    resp = client.post("/api/prices/upload", json=body)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert "snapshot" in payload and "saved_path" in payload
    # Confirm the file landed in the (monkeypatched) tmp dir.
    files = list(isolated_price_snapshots_dir.glob("*.yaml"))
    assert len(files) == 1
    on_disk = yaml.safe_load(files[0].read_text(encoding="utf-8"))
    assert on_disk["currency"] == "EUR"
    assert "scrap" in on_disk["materials"]
    assert on_disk["materials"]["FeMn-80"]["price_per_kg"] == pytest.approx(1.85)


def test_prices_upload_invalid_400(
    isolated_price_snapshots_dir: Path, client: TestClient
) -> None:
    """element_content sum != 1 → 400 with the parser's English message."""
    body = {
        "date": "2026-05-09",
        "currency": "EUR",
        "materials": {
            "Bogus": {
                "kind": "ferroalloy",
                "price_per_kg": 1.0,
                "element_content": {"Mn": 0.5, "Fe": 0.3},  # sum=0.8
            },
        },
    }
    resp = client.post("/api/prices/upload", json=body)
    assert resp.status_code == 400
    assert "sum" in resp.json()["detail"].lower()


def test_prices_upload_currency_400(
    isolated_price_snapshots_dir: Path, client: TestClient
) -> None:
    body = {
        "date": "2026-05-09",
        "currency": "GBP",  # not in {RUB, USD, EUR}
        "materials": {
            "scrap": {
                "kind": "base",
                "price_per_kg": 0.40,
                "element_content": {"Fe": 1.0},
            },
        },
    }
    resp = client.post("/api/prices/upload", json=body)
    assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────────────
# numeric serialisation regression
# ─────────────────────────────────────────────────────────────────────


def test_safe_json_numeric_in_design_result(
    populated_models_dir: Path, client: TestClient
) -> None:
    """Result payload must carry numeric (not stringified) values for the chart."""
    version = next(populated_models_dir.iterdir()).name
    resp = client.post(
        "/api/design/run",
        json={
            "model_version": version,
            "target": {"min": 485, "max": 580},
            "population_size": 10,
            "n_generations": 1,
            "include_cost": False,
        },
    )
    job_id = resp.json()["job_id"]
    body = _wait_for_job(client, job_id, timeout=60.0)
    if body["status"] != "done":
        pytest.fail(f"job ended with status={body['status']} error={body.get('error')!r}")
    result = body["result"]
    if not result["candidates"]:
        pytest.skip("NSGA-II produced no candidates with this seed; rerun")
    c = result["candidates"][0]
    for k in ("predicted_mean", "predicted_q05", "predicted_q95", "cost_eur_per_t"):
        assert isinstance(c[k], (int, float)), f"{k} must be numeric, got {type(c[k])}"
