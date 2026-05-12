"""Router for /api/jobs/* — generic polling endpoint for long-running ops.

PR 6 of the Streamlit→FastAPI migration. See
``docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md``
(Endpoint map → Tab «Дизайн сплава» / «Обучение» / etc — all use the same
``GET /api/jobs/{job_id}`` shape).

Endpoints:
- ``GET /api/jobs/{job_id}``     — poll status / progress / result.
- ``DELETE /api/jobs/{job_id}``  — cooperative cancellation (PR 7).

The endpoint is intentionally minimal: PR 7+ routers create jobs through
``run_as_job(fn, ...)`` (or ``get_job_store().create(...)``) and return
``{"job_id": id}``; the frontend ``pollJob(id)`` helper drives this
endpoint at ~2 s intervals until the status flips to ``done`` / ``error``.

PR 7 — DELETE adds cooperative cancellation. Effective only for pending
jobs (Future.cancel() succeeds before the worker pulls them); running
jobs flip ``cancellation_requested`` and rely on the long-running fn to
poll the singleton store. NSGA-II / training don't poll yet, so DELETE
on a running job is best-effort: the worker continues to completion but
the UI stops polling immediately. Documented as MVP-acceptable in the
design doc PR 7 row.

Risks #3 (SafeJSONResponse): the response carries arbitrary ``result``
payloads from backend functions (numpy floats, dataclasses, datetimes),
so we declare ``response_class=SafeJSONResponse, response_model=None``
to bypass Pydantic v2's default revalidation and route through the
project encoder.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.jobs import get_job_store
from app.api.responses import SafeJSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/{job_id}",
    response_class=SafeJSONResponse,
    response_model=None,
)
def get_job(job_id: str) -> dict[str, Any]:
    """Return current job state.

    Shape mirrors ``Job.to_dict()`` — frontend expects:
        {
          job_id, status: pending|running|done|error,
          progress: 0.0..1.0, message: str,
          result: any | null, error: str | null,
          started_at: iso | null, finished_at: iso | null,
        }

    404 if the id is unknown OR was evicted by the retain-finished TTL —
    we don't disambiguate "never existed" vs "expired" because the
    polling client treats both the same way (give up, surface error).
    """
    store = get_job_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found",
        )
    return job.to_dict()


@router.delete(
    "/{job_id}",
    response_class=SafeJSONResponse,
    response_model=None,
)
def cancel_job(job_id: str) -> dict[str, Any]:
    """PR 7: cooperative cancellation of a long-running job.

    Returns the job's post-cancellation state (same shape as GET). The
    semantics are documented on :meth:`JobStore.request_cancel`:

    * Pending and not yet started → status flips to ``error`` with
      "Cancelled by user" message.
    * Running → ``cancellation_requested=True``; status stays
      ``running`` until the fn completes or reads the flag and bails.
      For NSGA-II / training the bail-path doesn't exist yet — the
      Future.cancel() call is a no-op and the worker burns through to
      completion. UI stops polling either way (status flips eventually).
    * Already done/error → no-op, return current state.

    404 if the id is unknown / evicted, matching GET behaviour. We do
    not return 409 for "already terminal" because a) idempotent DELETE
    is friendlier to the UI's «Отменить» button and b) the response
    body already exposes ``status`` so the UI can detect the case.
    """
    store = get_job_store()
    job = store.request_cancel(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found",
        )
    return job.to_dict()
