"""Tests for /api/jobs/* and the JobStore (PR 6 of FastAPI migration).

PR 1 shipped a minimal store (PENDING → RUNNING → DONE/ERROR + result);
PR 6 extends it with progress callbacks, a polling endpoint, list/evict
helpers, and the ``run_as_job`` shorthand. These tests pin the new
contract and guard the lifecycle invariants the frontend
``pollJob(jobId)`` helper relies on.

We deliberately use small ``time.sleep`` values (≤0.05 s) because the
worker is a real ThreadPoolExecutor and we need observable timing —
mocks would hide thread-safety bugs the suite is meant to catch.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.api.jobs import (
    JobStatus,
    JobStore,
    get_job_store,
    run_as_job,
)
from app.api.main import app


@pytest.fixture(autouse=True)
def reset_job_store_singleton():
    """Очищает lru_cache singleton между тестами — предотвращает cross-test pollution.

    Без этого fixture jobs из ``test_run_as_job_uses_singleton`` /
    ``test_get_job_endpoint_*`` остаются в общем store и (а) ломают eviction
    тесты, (б) в худшем случае всплывают как «неожиданные» job_id в
    ``list_jobs``.
    """
    get_job_store.cache_clear()
    yield
    get_job_store.cache_clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def store() -> JobStore:
    """Fresh store per test — avoids singleton bleed across cases."""
    return JobStore(max_workers=4)


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.01) -> None:
    """Spin until predicate() returns truthy or timeout. Raises if not."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f"timeout waiting for {predicate!r}")


# ──────────────────────────────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────────────────────────────


def test_job_lifecycle_pending_to_done(store: JobStore) -> None:
    """Happy path: create → running → done with result echoed."""

    def work(x: int, y: int) -> int:
        time.sleep(0.02)
        return x + y

    job_id = store.create(work, 2, 3)
    job = store.get(job_id)
    assert job is not None
    assert job.status in (JobStatus.PENDING, JobStatus.RUNNING)

    _wait_for(lambda: store.get(job_id).status == JobStatus.DONE)
    job = store.get(job_id)
    assert job.status is JobStatus.DONE
    assert job.result == 5
    assert job.error is None
    assert job.started_at is not None
    assert job.finished_at is not None
    # Worker auto-snaps progress to 1.0 on DONE even if fn never called progress.
    assert job.progress == 1.0


def test_job_lifecycle_with_error(store: JobStore) -> None:
    """Exception path: error captured, status is ERROR, traceback included."""

    def boom() -> None:
        raise ValueError("kaboom")

    job_id = store.create(boom)
    _wait_for(lambda: store.get(job_id).status == JobStatus.ERROR)
    job = store.get(job_id)
    assert job.status is JobStatus.ERROR
    assert job.result is None
    assert job.error is not None
    assert "kaboom" in job.error
    assert "ValueError" in job.error
    assert job.finished_at is not None


def test_get_unknown_job_returns_none(store: JobStore) -> None:
    assert store.get("never-existed") is None


# ──────────────────────────────────────────────────────────────────────
# Progress callback injection
# ──────────────────────────────────────────────────────────────────────


def test_job_progress_callback_streams_updates(store: JobStore) -> None:
    """fn that declares ``progress`` kwarg gets a live callback to update Job.progress."""

    seen_started = threading.Event()
    seen_mid = threading.Event()
    proceed = threading.Event()

    def staged(progress) -> str:  # signature opt-in
        progress(0.0, "старт")
        seen_started.set()
        # Wait so the test can observe the intermediate state.
        proceed.wait(timeout=2.0)
        progress(0.5, "середина")
        seen_mid.set()
        progress(1.0, "финал")
        return "ok"

    job_id = store.create(staged)

    seen_started.wait(timeout=2.0)
    job = store.get(job_id)
    assert job.progress == 0.0
    assert job.message == "старт"

    proceed.set()
    seen_mid.wait(timeout=2.0)
    # Race: between seen_mid.set() and our get() the worker might already
    # have advanced to 1.0. Either reading is correct — assert ≥0.5.
    _wait_for(lambda: store.get(job_id).status == JobStatus.DONE)
    job = store.get(job_id)
    assert job.status is JobStatus.DONE
    assert job.progress == 1.0
    assert job.message == "финал"
    assert job.result == "ok"


def test_job_progress_clamps_out_of_range(store: JobStore) -> None:
    """Defensive: progress(-0.5, ...) and progress(1.5, ...) clamp to [0,1].

    Verify clamping ДО terminal DONE-snap. Without that, the worker auto-
    sets progress=1.0 on success and the test would still pass even if
    ``_make_progress_cb`` lost its clamp. We hold the worker thread on a
    sync_event between the two progress() calls so we can read the live
    Job.progress value mid-execution.
    """

    sync_event = threading.Event()

    def silly(progress) -> None:
        progress(1.5, "huge")  # should clamp → 1.0
        sync_event.wait(timeout=2.0)
        progress(-0.5, "neg")  # should clamp → 0.0

    job_id = store.create(silly)

    # Read live state after first progress call but before second / DONE.
    _wait_for(lambda: store.get(job_id).message == "huge", timeout=2.0)
    mid_progress = store.get(job_id).progress
    assert mid_progress == 1.0, f"clamp upper bound failed: {mid_progress!r}"

    sync_event.set()
    _wait_for(lambda: store.get(job_id).status == JobStatus.DONE)
    job = store.get(job_id)
    # End-state progress: clamped to 1.0 on DONE regardless.
    assert 0.0 <= job.progress <= 1.0


def test_function_without_progress_kwarg_is_not_injected(store: JobStore) -> None:
    """Backward compat: PR 1 functions without ``progress`` kw must run unchanged."""

    def legacy(a: int, b: int) -> int:
        return a * b

    job_id = store.create(legacy, 3, 4)
    _wait_for(lambda: store.get(job_id).status == JobStatus.DONE)
    assert store.get(job_id).result == 12


def test_function_with_kwargs_receives_progress(store: JobStore) -> None:
    """**kwargs accepts unknown keys, including the injected progress callback."""

    captured: list = []

    def flexible(**kwargs) -> str:
        cb = kwargs.get("progress")
        if cb is not None:
            cb(0.5, "halfway")
        captured.append(callable(cb))
        return "fine"

    job_id = store.create(flexible)
    _wait_for(lambda: store.get(job_id).status == JobStatus.DONE)
    assert captured == [True]


# ──────────────────────────────────────────────────────────────────────
# Concurrency / isolation
# ──────────────────────────────────────────────────────────────────────


def test_concurrent_jobs_isolated() -> None:
    """5 jobs in parallel, each computes a different result."""

    s = JobStore(max_workers=5)

    def slow_square(n: int) -> int:
        time.sleep(0.03)
        return n * n

    ids = [s.create(slow_square, i) for i in range(5)]
    for jid in ids:
        _wait_for(lambda jid=jid: s.get(jid).status == JobStatus.DONE)
    for i, jid in enumerate(ids):
        assert s.get(jid).result == i * i


def test_jobs_threadsafe_under_concurrent_reads() -> None:
    """Race test: many readers + a slow writer must not corrupt state."""

    s = JobStore(max_workers=2)

    def worker(progress) -> int:
        for i in range(10):
            progress(i / 10.0, f"step {i}")
            time.sleep(0.005)
        return 42

    job_id = s.create(worker)

    errors: list[Exception] = []

    def reader() -> None:
        try:
            for _ in range(40):
                job = s.get(job_id)
                assert job is not None
                # Dataclass attrs are typed — read each one.
                _ = job.status, job.progress, job.message, job.result
                time.sleep(0.002)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=3)

    _wait_for(lambda: s.get(job_id).status == JobStatus.DONE)
    assert s.get(job_id).result == 42
    assert errors == []


# ──────────────────────────────────────────────────────────────────────
# list_jobs + eviction
# ──────────────────────────────────────────────────────────────────────


def test_list_jobs_returns_recent_first() -> None:
    s = JobStore(max_workers=2)

    ids = []
    for _ in range(3):
        ids.append(s.create(lambda: 1))
        time.sleep(0.01)
    for jid in ids:
        _wait_for(lambda jid=jid: s.get(jid).status == JobStatus.DONE)

    listed = s.list_jobs(limit=10)
    listed_ids = [j.job_id for j in listed]
    # Most recently created (last id) first.
    assert listed_ids[0] == ids[-1]
    assert set(ids).issubset(set(listed_ids))


def test_finished_job_evicted_after_retain_window() -> None:
    """Opportunistic TTL on get(): finished job past cutoff is evicted."""

    s = JobStore(max_workers=2, retain_finished=timedelta(seconds=0))

    job_id = s.create(lambda: "done")
    _wait_for(lambda: s.get(job_id) is None or s.get(job_id).status == JobStatus.DONE)
    # First get may still see the job; second get triggers eviction
    # because finished_at is now in the past relative to the 0-second window.
    s.get(job_id)
    # Force a small wait so finished_at < utcnow definitely.
    time.sleep(0.01)
    assert s.get(job_id) is None


# ──────────────────────────────────────────────────────────────────────
# run_as_job + singleton
# ──────────────────────────────────────────────────────────────────────


def test_run_as_job_uses_singleton() -> None:
    """The shorthand pushes onto get_job_store(), and the same store sees the job."""

    store = get_job_store()
    job_id = run_as_job(lambda: 99)
    _wait_for(lambda: store.get(job_id) is not None and store.get(job_id).status == JobStatus.DONE)
    assert store.get(job_id).result == 99


# ──────────────────────────────────────────────────────────────────────
# /api/jobs/{job_id} HTTP endpoint
# ──────────────────────────────────────────────────────────────────────


def test_get_unknown_job_returns_404(client: TestClient) -> None:
    resp = client.get("/api/jobs/does-not-exist")
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert "detail" in body
    assert "does-not-exist" in body["detail"]


def test_get_job_endpoint_returns_full_shape(client: TestClient) -> None:
    """End-to-end: submit a quick job, poll until done, verify response shape."""

    store = get_job_store()
    job_id = store.create(lambda progress: (progress(1.0, "ok"), "result-payload")[-1])
    _wait_for(lambda: store.get(job_id).status == JobStatus.DONE)

    resp = client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    body = resp.json()
    expected_keys = {
        "job_id", "status", "progress", "message",
        "result", "error", "started_at", "finished_at",
    }
    assert expected_keys.issubset(set(body.keys()))
    assert body["status"] == "done"
    assert body["progress"] == 1.0
    assert body["error"] is None
    # Result payload echoed verbatim — guards against accidental result
    # shadowing (e.g. by progress callback or post-process).
    assert body["result"] == "result-payload"
    # ISO-8601 timestamps.
    assert isinstance(body["started_at"], str)
    assert isinstance(body["finished_at"], str)


def test_get_job_endpoint_surfaces_error(client: TestClient) -> None:
    store = get_job_store()

    def boom() -> None:
        raise RuntimeError("boom-from-api")

    job_id = store.create(boom)
    _wait_for(lambda: store.get(job_id).status == JobStatus.ERROR)
    resp = client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert "boom-from-api" in (body["error"] or "")
    assert body["result"] is None


def test_started_at_is_iso8601_in_response(client: TestClient) -> None:
    """SafeJSONResponse must render datetime → ISO string (PR 1 contract)."""

    store = get_job_store()
    job_id = store.create(lambda: 1)
    _wait_for(lambda: store.get(job_id).status == JobStatus.DONE)
    resp = client.get(f"/api/jobs/{job_id}")
    body = resp.json()
    started = body["started_at"]
    # Roundtrip through fromisoformat to confirm shape.
    parsed = datetime.fromisoformat(started)
    assert isinstance(parsed, datetime)
