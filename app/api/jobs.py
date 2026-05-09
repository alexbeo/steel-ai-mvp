"""In-memory async job store for long-running operations.

PR 1 introduced the minimal store (PENDING → RUNNING → DONE/ERROR + result).
PR 6 adds:
  • progress / message updates that callers can stream from inside ``fn``
    via an injected ``progress`` keyword (typed as :data:`ProgressCallback`).
  • thread-safe ``_update`` helper so the worker thread + caller don't race
    on the dataclass fields.
  • ``run_as_job`` shorthand so router code reads as
    ``job_id = run_as_job(train_model, ...)`` rather than reaching for the
    singleton.
  • opportunistic TTL cleanup of finished jobs at ``get()`` time. We don't
    add a background thread or scheduler — solo-process FastAPI + ad-hoc
    eviction is good enough for the MVP, and survives ``--reload`` no
    worse than the rest of the in-memory state (design doc Risks #1).
  • ``list_jobs`` for a future debugging UI; harmless to expose now.

Backward compatibility: PR 1 callers used ``store.create(fn, *args)`` and
``store.get(id)``. Both signatures are preserved exactly. The new
``progress`` injection is opt-in — if ``fn`` does not declare a ``progress``
parameter, no callback is passed and the function runs as before.
"""
from __future__ import annotations

import inspect
import logging
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Keep finished jobs around for an hour so a UI that reloads / backgrounds
# during a long poll still resolves the result. After that, ``get()``
# evicts the entry — see ``_maybe_evict``. Tunable via the JobStore ctor.
DEFAULT_RETAIN_FINISHED = timedelta(hours=1)


# Type alias for the progress callback we inject. ``progress(p, m)`` — p in
# [0.0, 1.0] (clamped), m is a short human message. Callers that don't need
# either argument may ignore them; we just record the most recent values.
ProgressCallback = Callable[[float, str], None]


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    # Progress stream (PR 6). ``progress`` is a fraction in [0.0, 1.0]; we
    # store as float (not int %) so callers can use 0.05 / 0.5 / 0.95 stages
    # without quantisation. ``message`` is a short label describing the
    # current step ("Optuna trial 3/10", "NSGA-II generation 12/40", etc.).
    progress: float = 0.0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class JobStore:
    """Threadsafe in-memory job registry. Restart-volatile by design.

    Public API:
        create(fn, *args, **kwargs) -> job_id
        get(job_id) -> Job | None
        list_jobs(limit=50) -> list[Job]

    PR 6 additions:
        _update(job_id, **fields) — thread-safe field mutation; private
        because callers should use the injected ``progress`` callback rather
        than poke the store directly.
    """

    def __init__(
        self,
        max_workers: int = 2,
        retain_finished: timedelta = DEFAULT_RETAIN_FINISHED,
    ) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="steel-ai-job"
        )
        self._retain_finished = retain_finished

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def create(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        """Register a job and dispatch it to the worker pool.

        If ``fn`` declares a ``progress`` keyword (or has ``**kwargs``), we
        inject a closure that streams updates back into the store. We use
        :func:`inspect.signature` rather than raw introspection so wrapped
        functions (functools.wraps, partials) report parameters correctly.
        Falls back to no-injection on signature failure (built-ins, C
        functions) — better to skip progress than crash the submission.
        """
        job_id = uuid.uuid4().hex
        job = Job(job_id=job_id)
        with self._lock:
            self._jobs[job_id] = job

        if self._fn_accepts_progress(fn):
            kwargs = {**kwargs, "progress": self._make_progress_cb(job_id)}

        self._executor.submit(self._run, job_id, fn, args, kwargs)
        return job_id

    def get(self, job_id: str) -> Job | None:
        """Return the live :class:`Job` or ``None`` if unknown / evicted.

        Triggers an opportunistic eviction sweep so memory does not grow
        unbounded across long uvicorn sessions; bounded by ``_retain_finished``.
        """
        self._maybe_evict()
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> list[Job]:
        """Most-recent-first snapshot of jobs (for a future debug endpoint).

        Sorted by ``started_at`` descending so the freshest run leads.
        Pending jobs (no started_at yet) sort last with epoch-zero. We
        return a list of dataclass instances by reference — callers must
        not mutate; the lock only protects the dict, not the rows.
        """
        with self._lock:
            jobs = list(self._jobs.values())
        # tz-aware epoch — must match started_at tzinfo to avoid
        # ``TypeError: can't compare offset-naive and offset-aware datetimes``.
        epoch = datetime.min.replace(tzinfo=timezone.utc)
        jobs.sort(key=lambda j: (j.started_at or epoch), reverse=True)
        return jobs[:limit]

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _update(self, job_id: str, **fields: Any) -> None:
        """Thread-safe partial update. No-op if the job is gone (evicted)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                # Defensive: refuse to set fields that aren't on Job — keeps
                # typos from silently growing dataclass attributes.
                if not hasattr(job, key):
                    logger.warning("JobStore._update: ignoring unknown field %r", key)
                    continue
                setattr(job, key, value)

    @staticmethod
    def _fn_accepts_progress(fn: Callable[..., Any]) -> bool:
        """True if ``fn`` declares ``progress`` keyword (or accepts **kwargs).

        We don't want to crash random callers by injecting an unexpected
        kwarg; equally, we don't want to require explicit opt-in for every
        long-running function. Inspecting the signature is the right
        middle ground.
        """
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            # Built-ins / C extensions don't expose signatures — be safe.
            return False
        for name, param in sig.parameters.items():
            if name == "progress":
                return True
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                # **kwargs accepts anything, including progress
                return True
        return False

    def _make_progress_cb(self, job_id: str) -> ProgressCallback:
        """Closure captures ``job_id`` so workers don't need a reference to
        the store. Clamps ``p`` to [0.0, 1.0] — UI assumes that range.
        """

        def _cb(p: float, m: str = "") -> None:
            try:
                pf = float(p)
            except (TypeError, ValueError):
                pf = 0.0
            if pf < 0.0:
                pf = 0.0
            elif pf > 1.0:
                pf = 1.0
            self._update(job_id, progress=pf, message=str(m or ""))

        return _cb

    def _maybe_evict(self) -> None:
        """Drop finished jobs older than ``_retain_finished``.

        Called from ``get()``; cost is O(N) on each poll but N is small
        (single-digit jobs concurrent in normal use). Avoids a background
        thread which would be the wrong tool for an ephemeral, single-
        process MVP.
        """
        cutoff = datetime.now(timezone.utc) - self._retain_finished
        with self._lock:
            stale = [
                jid
                for jid, job in self._jobs.items()
                if job.finished_at is not None and job.finished_at < cutoff
            ]
            for jid in stale:
                del self._jobs[jid]

    def _run(
        self,
        job_id: str,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Worker entrypoint — flips status, executes, captures result/error.

        We re-fetch the job from the dict on every status transition because
        a paranoid eviction (e.g. tests with retain_finished=0) might have
        removed it. ``_update`` already handles the missing case.
        """
        self._update(
            job_id, status=JobStatus.RUNNING, started_at=datetime.now(timezone.utc)
        )
        try:
            result = fn(*args, **kwargs)
            self._update(
                job_id,
                status=JobStatus.DONE,
                result=result,
                finished_at=datetime.now(timezone.utc),
                # Snap to 1.0 — even if the fn forgot to call progress(1.0, ...).
                progress=1.0,
            )
        except Exception as exc:  # noqa: BLE001 — job boundary, log + persist
            logger.exception("Job %s failed", job_id)
            self._update(
                job_id,
                status=JobStatus.ERROR,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                finished_at=datetime.now(timezone.utc),
            )


@lru_cache(maxsize=1)
def get_job_store() -> JobStore:
    """Process-singleton accessor. ``lru_cache(1)`` avoids the boilerplate
    of a module-global flag without sacrificing testability — tests can
    call ``get_job_store.cache_clear()`` if isolation is needed.
    """
    return JobStore()


def run_as_job(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """Convenience wrapper used by routers — submit ``fn`` to the singleton
    store and return the job id. Reads as natural English in router code:
    ``job_id = run_as_job(train_model, version="v3", n_trials=20)``.
    """
    return get_job_store().create(fn, *args, **kwargs)
