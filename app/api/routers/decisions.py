"""Router for /api/decisions — Decision Log read API.

Thin wrapper around :mod:`decision_log.logger`. No business logic here; the
underlying SQLite layer already normalises JSON columns (alternatives_considered,
context, tags) into native Python lists/dicts.

PR 2 of the Streamlit→FastAPI migration. See
``docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md``
(Endpoint map → Tab «История»).
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.api.responses import SafeJSONResponse
from decision_log.logger import (
    get_decision_by_id,
    query_decisions,
    summarize_project_history,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Streamlit «История» offers limit slider 5..100 (default 20). We keep the same
# upper bound so frontend pagination cannot DoS the SQLite read.
_MAX_LIMIT = 100
_DEFAULT_LIMIT = 20


@router.get(
    "/decisions/summary",
    response_class=SafeJSONResponse,
    response_model=None,
)
def decisions_summary() -> dict[str, Any]:
    """Project-history summary used by the History view header.

    Wraps :func:`decision_log.logger.summarize_project_history`. The function
    returns a Markdown-ish string; we surface it as ``summary_md`` so future
    UI can render with a markdown library if desired without an API change.
    """
    summary_md = summarize_project_history()
    return {"summary_md": summary_md}


@router.get(
    "/decisions",
    response_class=SafeJSONResponse,
    response_model=None,
)
def list_decisions(
    phase: str | None = Query(default=None, description="Filter by Phase enum value"),
    tag: str | None = Query(default=None, description="Substring match on tags JSON"),
    keyword: str | None = Query(
        default=None,
        description="Substring match on decision OR reasoning",
    ),
    limit: int = Query(default=_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """List decisions with optional filters.

    The underlying ``query_decisions`` does not support OFFSET natively; we
    read ``limit + offset`` rows then slice. For the Decision Log volumes we
    expect (≤1k rows over a project lifetime) this is well below the cost of
    introducing an OFFSET clause and keeps the logger module untouched.

    Response shape::

        {"items": [Decision, ...], "limit": int, "offset": int, "count": int}

    where ``count`` is the number of items in *this* page (not the global
    total — global COUNT would require a second query and is not used by the
    UI's "load more" pagination).
    """
    rows = query_decisions(
        phase=phase,
        tag=tag,
        keyword=keyword,
        limit=limit + offset,
    )
    page = rows[offset : offset + limit]
    return {
        "items": page,
        "limit": limit,
        "offset": offset,
        "count": len(page),
    }


@router.get(
    "/decisions/{decision_id}",
    response_class=SafeJSONResponse,
    response_model=None,
)
def get_decision(decision_id: int) -> dict[str, Any]:
    """Fetch one decision by primary key. 404 if missing."""
    row = get_decision_by_id(decision_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Decision id={decision_id} not found",
        )
    return row
