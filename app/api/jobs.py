"""In-memory async job store for long-running operations.

PR 1 minimal scope — single threadsafe dict, ThreadPoolExecutor for offload.
PR 6 will extend with progress callbacks, cancellation, persistent results.
"""
from __future__ import annotations

import logging
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Any, Callable

logger = logging.getLogger(__name__)


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
    # progress / message extension points for PR 6
    progress: int = 0
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
    """Threadsafe in-memory job registry. Restart-volatile by design (PR 1).

    PR 6 will swap implementation for SQLite-backed store if --reload survival
    becomes a problem; public API is intentionally narrow.
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="steel-ai-job"
        )

    def create(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        job_id = uuid.uuid4().hex
        job = Job(job_id=job_id)
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run, job_id, fn, args, kwargs)
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(
        self,
        job_id: str,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
        try:
            result = fn(*args, **kwargs)
            with self._lock:
                job.result = result
                job.status = JobStatus.DONE
                job.finished_at = datetime.utcnow()
        except Exception as exc:  # noqa: BLE001 — job boundary, log + persist
            logger.exception("Job %s failed", job_id)
            with self._lock:
                job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                job.status = JobStatus.ERROR
                job.finished_at = datetime.utcnow()


@lru_cache(maxsize=1)
def get_job_store() -> JobStore:
    return JobStore()
