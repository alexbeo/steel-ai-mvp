"""Shared response utilities for the FastAPI layer.

Lives in its own module so routers can declare ``response_class=SafeJSONResponse``
without importing from ``app.api.main`` (which would create a circular dependency
once routers are wired into the app there). PR 2 carved this out from main.py;
PR 3-12 routers will all import from here.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime
from typing import Any

from starlette.responses import JSONResponse


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
