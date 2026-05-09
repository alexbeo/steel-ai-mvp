"""FastAPI entry-point — Steel AI MVP API.

Run:
    PYTHONPATH=. uvicorn app.api.main:app --reload --port 8000

PR 1 scope: health endpoint, static files mount, .env loading. Routers for
predict / design / train / deox / hypotheses / recipes / active_learning /
decisions land in subsequent PRs.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

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


def _json_default(obj: Any) -> Any:
    """Catch-all encoder for non-JSON-native types at the API boundary.

    Order matters: numpy scalars must produce real numbers (not strings) so
    Plotly / charting code in PR 3 can consume them without parseFloat. We
    avoid a module-level numpy import (heavy on cold start) and instead rely
    on duck-typing — numpy scalars expose .item(), arrays expose .tolist(),
    both are detectable without importing numpy.
    """
    # numpy scalar (np.float32, np.int64, np.bool_, ...) — has .item() returning
    # native Python scalar. Guarded by hasattr to avoid importing numpy.
    if hasattr(obj, "item") and callable(obj.item) and hasattr(obj, "dtype"):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass
    # numpy.ndarray (and array-like with .tolist()) → list of native Python types
    if hasattr(obj, "tolist") and callable(obj.tolist) and hasattr(obj, "dtype"):
        return obj.tolist()
    # dataclass instance (not class) → dict; recurse via json.dumps default
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    # datetime / date → ISO 8601 string
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    # last resort — string repr (matches previous default=str behaviour for
    # Path, Enum without .value access, etc.)
    return str(obj)


class SafeJSONResponse(JSONResponse):
    """JSONResponse with numpy/dataclass/datetime-aware encoder.

    Backend functions return numpy floats, datetimes, dataclasses; the project
    discipline is to convert at the API boundary. The custom encoder produces
    JSON-native numbers (not strings) for numpy scalars/arrays so charting code
    in downstream PRs does not need parseFloat. See _json_default.
    """

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        ).encode("utf-8")


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


# StaticFiles mount comes LAST so /api/* routes take precedence.
# html=True makes "/" serve index.html automatically.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
