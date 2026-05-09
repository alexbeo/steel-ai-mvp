"""FastAPI entry-point — Steel AI MVP API.

Run:
    PYTHONPATH=. uvicorn app.api.main:app --reload --port 8000

PR 1 scope: health endpoint, static files mount, .env loading. Routers for
predict / design / train / deox / hypotheses / recipes / active_learning /
decisions land in subsequent PRs.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles

from app.api.responses import SafeJSONResponse, _json_default  # noqa: F401 — re-exported for back-compat

logger = logging.getLogger(__name__)

API_VERSION = "0.1.0"  # bumped manually until pyproject.toml exists

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = PROJECT_ROOT / "app" / "web" / "static"

# Load .env mirroring app/frontend/app.py — best-effort (no-op if dotenv missing)
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover — dotenv is in requirements but tolerate absence
    logger.warning("python-dotenv not installed; skipping .env load")


app = FastAPI(
    title="Steel AI MVP API",
    version=API_VERSION,
    default_response_class=SafeJSONResponse,
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness probe. UI uses this to confirm backend is reachable."""
    return {
        "status": "ok",
        "version": API_VERSION,
        "llm_ready": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


# ──────────────────────────────────────────────────────────────────────
# Global exception handler — RequestValidationError (PR 6)
#
# Why we override the default:
#   The default FastAPI handler renders errors via the stock JSONResponse,
#   which sets ``allow_nan=True`` implicitly via Python's json module… but
#   when pydantic v2 reports a validation error it embeds the offending
#   value (NaN / +Inf / -Inf) inside ``ctx``. Stock json.dumps then dies
#   with "Out of range float values are not JSON compliant" → 500 + an
#   HTML error page on the wire. Browser ``JSON.stringify`` never emits
#   NaN, so day-to-day UI use doesn't trigger this; but curl / non-browser
#   clients can post raw ``{"x": NaN}`` (parsed by orjson/json with
#   allow_nan=True at request decode time), which then can't be echoed
#   back in the validation error.
#
# Fix:
#   Pre-pass ``exc.errors()`` through ``jsonable_encoder`` with a custom
#   float encoder that replaces non-finite floats with strings ("NaN",
#   "Infinity", "-Infinity"). The result is plain JSON-safe data, which
#   our SafeJSONResponse renders without complaint. Status stays 422 to
#   match the FastAPI default semantics.
# ──────────────────────────────────────────────────────────────────────


def _safe_float(v: float) -> float | str:
    """Return ``v`` unchanged if finite, else its string label.

    Catches the three values that ``json.dumps(allow_nan=False)`` rejects:
      - NaN  (v != v)
      - +Inf (v == float('inf'))
      - -Inf (v == float('-inf'))
    """
    if v != v:  # NaN check (NaN is the only float that isn't equal to itself)
        return "NaN"
    if v == float("inf"):
        return "Infinity"
    if v == float("-inf"):
        return "-Infinity"
    return v


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> SafeJSONResponse:
    """Replace stock 500-on-NaN with a clean 422 + JSON-safe payload.

    Body shape matches FastAPI's default — ``{"detail": [...errors...]}``
    — so existing clients that already inspect ``response.json().detail``
    continue to work. We additionally include a top-level ``body`` key
    when present (also matching default FastAPI behaviour) for parity.
    """
    safe_errors = jsonable_encoder(
        exc.errors(),
        custom_encoder={float: _safe_float},
    )
    payload: dict[str, Any] = {"detail": safe_errors}
    # FastAPI's default also surfaces request body when serializable; we
    # mirror that on a best-effort basis. If the body isn't serialisable
    # (binary, very large, etc.) we drop it silently — the detail array
    # already pinpoints which field failed.
    body = getattr(exc, "body", None)
    if body is not None:
        try:
            payload["body"] = jsonable_encoder(body, custom_encoder={float: _safe_float})
        except Exception:  # pragma: no cover — defensive
            pass
    return SafeJSONResponse(payload, status_code=422)


# Routers — imported after the FastAPI app instance is created so include_router
# has a target. SafeJSONResponse now lives in app.api.responses, so routers no
# longer create a circular import via this module.
from app.api.routers import active_learning as _active_learning_router  # noqa: E402
from app.api.routers import decisions as _decisions_router  # noqa: E402
from app.api.routers import deox as _deox_router  # noqa: E402
from app.api.routers import design as _design_router  # noqa: E402
from app.api.routers import jobs as _jobs_router  # noqa: E402
from app.api.routers import predict as _predict_router  # noqa: E402
from app.api.routers import prices as _prices_router  # noqa: E402
from app.api.routers import system as _system_router  # noqa: E402

app.include_router(_decisions_router.router, prefix="/api", tags=["decisions"])
app.include_router(_system_router.router, prefix="/api/system", tags=["system"])
app.include_router(_predict_router.router, prefix="/api", tags=["predict"])
app.include_router(_deox_router.router, prefix="/api/deox", tags=["deox"])
app.include_router(
    _active_learning_router.router,
    prefix="/api/active-learning",
    tags=["active-learning"],
)
app.include_router(_jobs_router.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(_design_router.router, prefix="/api/design", tags=["design"])
app.include_router(_prices_router.router, prefix="/api/prices", tags=["prices"])


# StaticFiles mount comes LAST so /api/* routes take precedence.
# html=True makes "/" serve index.html automatically.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
