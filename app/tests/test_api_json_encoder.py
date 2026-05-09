"""Forward-compat tests for SafeJSONResponse encoder.

Backend functions return numpy scalars/arrays, dataclasses, datetimes; PR 3
chart code consumes them via fetch().then(r => r.json()) and feeds them to
Plotly directly. This means numpy.float32(3.14) must serialize as the JSON
number 3.14, NOT the string "3.14" — otherwise every chart module would need
parseFloat() on every field.

These tests pin the encoder behaviour for each non-JSON-native type the
Steel AI backend is known to produce.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.main import SafeJSONResponse, _json_default


# ---------- _json_default unit tests (no HTTP layer) ----------


def test_encoder_numpy_float_becomes_json_number() -> None:
    raw = json.dumps({"x": np.float32(3.14)}, default=_json_default, allow_nan=False)
    parsed = json.loads(raw)
    assert isinstance(parsed["x"], float)
    # np.float32 has limited precision, so accept ~6 significant figures
    assert abs(parsed["x"] - 3.14) < 1e-5


def test_encoder_numpy_int_becomes_json_number() -> None:
    raw = json.dumps({"n": np.int64(42)}, default=_json_default)
    parsed = json.loads(raw)
    assert parsed["n"] == 42
    assert isinstance(parsed["n"], int)


def test_encoder_numpy_bool_becomes_json_bool() -> None:
    raw = json.dumps({"flag": np.bool_(True)}, default=_json_default)
    parsed = json.loads(raw)
    assert parsed["flag"] is True
    assert isinstance(parsed["flag"], bool)


def test_encoder_numpy_array_becomes_json_list() -> None:
    raw = json.dumps({"arr": np.array([1.0, 2.0, 3.0])}, default=_json_default)
    parsed = json.loads(raw)
    assert parsed["arr"] == [1.0, 2.0, 3.0]
    assert all(isinstance(v, float) for v in parsed["arr"])


def test_encoder_numpy_2d_array_becomes_nested_list() -> None:
    raw = json.dumps(
        {"mat": np.array([[1, 2], [3, 4]], dtype=np.int32)},
        default=_json_default,
    )
    parsed = json.loads(raw)
    assert parsed["mat"] == [[1, 2], [3, 4]]


def test_encoder_dataclass_becomes_dict() -> None:
    @dataclass
    class Point:
        x: float
        y: float
        label: str = "p"

    raw = json.dumps(Point(x=1.0, y=2.0), default=_json_default)
    parsed = json.loads(raw)
    assert parsed == {"x": 1.0, "y": 2.0, "label": "p"}


def test_encoder_dataclass_with_numpy_fields() -> None:
    """Real-world shape: dataclass containing numpy scalars (e.g. metrics
    objects). asdict() yields numpy types in the dict, default= recurses to
    convert them.
    """

    @dataclass
    class Metrics:
        rmse: Any
        coverage: Any
        scores: Any = field(default_factory=lambda: np.array([0.1, 0.2, 0.3]))

    obj = Metrics(rmse=np.float32(12.5), coverage=np.float64(0.92))
    raw = json.dumps(obj, default=_json_default, allow_nan=False)
    parsed = json.loads(raw)
    assert isinstance(parsed["rmse"], float)
    assert abs(parsed["rmse"] - 12.5) < 1e-5
    assert isinstance(parsed["coverage"], float)
    assert abs(parsed["coverage"] - 0.92) < 1e-9
    assert parsed["scores"] == [0.1, 0.2, 0.3]


def test_encoder_datetime_becomes_iso_string() -> None:
    dt = datetime(2026, 5, 9, 12, 30, 45)
    raw = json.dumps({"ts": dt}, default=_json_default)
    parsed = json.loads(raw)
    assert parsed["ts"] == "2026-05-09T12:30:45"
    assert isinstance(parsed["ts"], str)


def test_encoder_date_becomes_iso_string() -> None:
    raw = json.dumps({"d": date(2026, 5, 9)}, default=_json_default)
    parsed = json.loads(raw)
    assert parsed["d"] == "2026-05-09"


def test_encoder_unknown_object_falls_back_to_str() -> None:
    class Weird:
        def __str__(self) -> str:
            return "weird-thing"

    raw = json.dumps({"w": Weird()}, default=_json_default)
    parsed = json.loads(raw)
    assert parsed["w"] == "weird-thing"


# ---------- TestClient integration: full HTTP round-trip ----------


def _make_test_app() -> FastAPI:
    """Tiny app exercising the SafeJSONResponse render path end-to-end.

    Note: in FastAPI 0.136 + pydantic v2, dict-returning endpoints are first
    serialized by pydantic (which doesn't know numpy), so our render() never
    sees numpy types via the default code path. To exercise the encoder we
    must either (a) return a Response object directly, or (b) declare
    response_class=SafeJSONResponse with response_model=None and return raw
    Python objects — both work because they bypass pydantic serialize_response.
    PR 2-12 endpoints that return numpy must follow this convention.
    """
    test_app = FastAPI()

    @dataclass
    class Sample:
        rmse: Any
        timestamps: Any

    @test_app.get(
        "/__test/mixed",
        response_class=SafeJSONResponse,
        response_model=None,
    )
    def _mixed() -> Any:
        # Returning a Response directly bypasses pydantic — the encoder runs.
        return SafeJSONResponse(
            content={
                "np_float": np.float32(3.14),
                "np_int": np.int64(42),
                "np_array": np.array([1.0, 2.0]),
                "dc": Sample(
                    rmse=np.float64(7.7),
                    timestamps=np.array([10, 20, 30], dtype=np.int32),
                ),
                "dt": datetime(2026, 5, 9, 0, 0, 0),
                "regular": "string-value",
            }
        )

    return test_app


def test_http_round_trip_yields_native_types() -> None:
    client = TestClient(_make_test_app())
    resp = client.get("/__test/mixed")
    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]

    payload = resp.json()
    # numpy scalar → JSON number (NOT string)
    assert isinstance(payload["np_float"], float)
    assert abs(payload["np_float"] - 3.14) < 1e-5
    assert isinstance(payload["np_int"], int)
    assert payload["np_int"] == 42
    # numpy array → list of native floats
    assert payload["np_array"] == [1.0, 2.0]
    assert all(isinstance(v, float) for v in payload["np_array"])
    # dataclass with numpy fields → dict with native numbers / list of ints
    assert isinstance(payload["dc"], dict)
    assert isinstance(payload["dc"]["rmse"], float)
    assert abs(payload["dc"]["rmse"] - 7.7) < 1e-9
    assert payload["dc"]["timestamps"] == [10, 20, 30]
    # datetime → ISO string
    assert payload["dt"] == "2026-05-09T00:00:00"
    # plain string passes through untouched
    assert payload["regular"] == "string-value"
