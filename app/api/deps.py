"""Shared FastAPI dependencies. Skeleton for PR 1; routers in PR 2+ extend."""
from __future__ import annotations

from app.api.jobs import JobStore, get_job_store


def get_job_store_dep() -> JobStore:
    """FastAPI dependency wrapper around the singleton job store.

    Routers should depend on this rather than importing get_job_store directly,
    so tests can override via app.dependency_overrides.
    """
    return get_job_store()
