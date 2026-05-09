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

from fastapi import FastAPI
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


# Routers — imported after the FastAPI app instance is created so include_router
# has a target. SafeJSONResponse now lives in app.api.responses, so routers no
# longer create a circular import via this module.
from app.api.routers import decisions as _decisions_router  # noqa: E402
from app.api.routers import deox as _deox_router  # noqa: E402
from app.api.routers import predict as _predict_router  # noqa: E402
from app.api.routers import system as _system_router  # noqa: E402

app.include_router(_decisions_router.router, prefix="/api", tags=["decisions"])
app.include_router(_system_router.router, prefix="/api/system", tags=["system"])
app.include_router(_predict_router.router, prefix="/api", tags=["predict"])
app.include_router(_deox_router.router, prefix="/api/deox", tags=["deox"])


# StaticFiles mount comes LAST so /api/* routes take precedence.
# html=True makes "/" serve index.html automatically.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
