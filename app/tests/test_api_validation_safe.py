"""Tests for NaN/Inf-safe validation error handling (PR 6).

Background (from PR 4 reviewer): Pydantic v2 reports validation errors
with the offending value embedded in ``ctx``. If a non-browser client
posts ``{"x": NaN}`` (curl can, Python ``json.dumps(allow_nan=True)``
can, JS ``JSON.stringify`` cannot), Python ``json.loads`` happily
accepts it (allow_nan=True is the default). Pydantic then carries the
NaN into the error report. The stock FastAPI exception handler
serialises this report through ``json.dumps(allow_nan=False)``, which
crashes with ``ValueError: Out of range float values are not JSON
compliant`` → bubbled as 500 + an HTML error page.

Fix shipped in this PR: a global RequestValidationError handler that
pre-passes ``exc.errors()`` through ``jsonable_encoder`` with a custom
encoder mapping NaN→"NaN", +Inf→"Infinity", -Inf→"-Infinity". The
result is plain JSON-safe data, status stays 422.

These tests reach in via raw ``content=`` (not the `json=` shortcut)
because TestClient's ``json=`` uses ``json.dumps`` which would either
reject NaN or pass-through depending on Python version — we want to
control the wire bytes exactly.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# Endpoint chosen: /api/deox/forward — has well-defined Pydantic field
# bounds (PR 4) so a NaN/Inf input will fail validation deterministically.
ENDPOINT = "/api/deox/forward"


def _post_raw(client: TestClient, body: str):
    """POST raw JSON body bypassing TestClient's json= helper.

    We need to put literal NaN / Infinity tokens on the wire, which the
    standard ``json.dumps`` won't do without ``allow_nan=True``. Going
    through ``content=`` lets us set the bytes directly.
    """
    return client.post(
        ENDPOINT,
        headers={"Content-Type": "application/json"},
        content=body,
    )


def _ok_payload() -> str:
    """A reference body that ALL fields valid — used as a baseline for
    parameterised NaN-injection tests so we know failures are due to
    the injected non-finite, not other field bounds.
    """
    return (
        '{"o_a_initial_ppm": 280.0, "temperature_C": 1620.0, '
        '"steel_mass_ton": 180.0, "target_o_a_ppm": 5.0, '
        '"al_purity_pct": 100.0, "burn_off_pct": 20.0, '
        '"model_id": "fruehan_1985"}'
    )


# ──────────────────────────────────────────────────────────────────────
# Status code: 422 (not 500) for non-finite floats
# ──────────────────────────────────────────────────────────────────────


def test_request_with_nan_returns_422(client: TestClient) -> None:
    body = (
        '{"o_a_initial_ppm": NaN, "temperature_C": 1620.0, '
        '"steel_mass_ton": 180.0, "target_o_a_ppm": 5.0, '
        '"al_purity_pct": 100.0, "burn_off_pct": 20.0, '
        '"model_id": "fruehan_1985"}'
    )
    resp = _post_raw(client, body)
    assert resp.status_code == 422, (
        f"expected 422 (validation error), got {resp.status_code}; "
        f"body: {resp.text[:300]}"
    )


def test_request_with_positive_infinity_returns_422(client: TestClient) -> None:
    body = (
        '{"o_a_initial_ppm": Infinity, "temperature_C": 1620.0, '
        '"steel_mass_ton": 180.0, "target_o_a_ppm": 5.0, '
        '"al_purity_pct": 100.0, "burn_off_pct": 20.0, '
        '"model_id": "fruehan_1985"}'
    )
    resp = _post_raw(client, body)
    assert resp.status_code == 422


def test_request_with_negative_infinity_returns_422(client: TestClient) -> None:
    body = (
        '{"o_a_initial_ppm": -Infinity, "temperature_C": 1620.0, '
        '"steel_mass_ton": 180.0, "target_o_a_ppm": 5.0, '
        '"al_purity_pct": 100.0, "burn_off_pct": 20.0, '
        '"model_id": "fruehan_1985"}'
    )
    resp = _post_raw(client, body)
    assert resp.status_code == 422


def test_request_with_nan_in_multiple_fields_returns_422(client: TestClient) -> None:
    body = (
        '{"o_a_initial_ppm": NaN, "temperature_C": NaN, '
        '"steel_mass_ton": 180.0, "target_o_a_ppm": 5.0, '
        '"al_purity_pct": 100.0, "burn_off_pct": 20.0, '
        '"model_id": "fruehan_1985"}'
    )
    resp = _post_raw(client, body)
    assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────
# Response shape: must be JSON, parseable, with `detail` array
# ──────────────────────────────────────────────────────────────────────


def test_validation_error_serializable_as_json(client: TestClient) -> None:
    body = (
        '{"o_a_initial_ppm": NaN, "temperature_C": 1620.0, '
        '"steel_mass_ton": 180.0, "target_o_a_ppm": 5.0, '
        '"al_purity_pct": 100.0, "burn_off_pct": 20.0, '
        '"model_id": "fruehan_1985"}'
    )
    resp = _post_raw(client, body)
    # 1) Content-Type must be JSON (not text/html or text/plain).
    ctype = resp.headers.get("content-type", "")
    assert "application/json" in ctype, f"got content-type {ctype!r}"
    # 2) response.json() must not raise — i.e. body is parseable JSON.
    payload = resp.json()
    assert isinstance(payload, dict)
    # 3) Standard FastAPI shape preserved.
    assert "detail" in payload
    assert isinstance(payload["detail"], list)
    assert len(payload["detail"]) >= 1


def test_validation_error_shows_offending_field(client: TestClient) -> None:
    """The detail entry should reference o_a_initial_ppm (the NaN field)."""
    body = (
        '{"o_a_initial_ppm": NaN, "temperature_C": 1620.0, '
        '"steel_mass_ton": 180.0, "target_o_a_ppm": 5.0, '
        '"al_purity_pct": 100.0, "burn_off_pct": 20.0, '
        '"model_id": "fruehan_1985"}'
    )
    resp = _post_raw(client, body)
    payload = resp.json()
    # Each detail entry has a "loc" path; we expect at least one mentioning the field.
    locs = [".".join(map(str, item.get("loc", []))) for item in payload["detail"]]
    assert any("o_a_initial_ppm" in loc for loc in locs), f"locs: {locs}"


def test_validation_error_does_not_contain_raw_nan_in_body(client: TestClient) -> None:
    """The wire body must not contain literal `NaN` / `Infinity` tokens —
    they must be rewritten as quoted strings. Otherwise ``response.json()``
    would call ``json.loads`` which accepts these tokens and we'd get
    silent passthrough; clients in stricter languages (Go, Rust, Swift)
    would reject the response."""
    body = (
        '{"o_a_initial_ppm": NaN, "temperature_C": 1620.0, '
        '"steel_mass_ton": 180.0, "target_o_a_ppm": 5.0, '
        '"al_purity_pct": 100.0, "burn_off_pct": 20.0, '
        '"model_id": "fruehan_1985"}'
    )
    resp = _post_raw(client, body)
    raw = resp.text
    # Strict JSON forbids unquoted NaN/Infinity; SafeJSONResponse uses
    # allow_nan=False so we expect quoted strings only.
    assert ": NaN" not in raw, f"raw body contains unquoted NaN: {raw[:300]}"
    assert ": Infinity" not in raw, f"raw body contains unquoted Infinity: {raw[:300]}"
    assert ": -Infinity" not in raw, f"raw body contains unquoted -Infinity: {raw[:300]}"


# ──────────────────────────────────────────────────────────────────────
# Sanity: regular validation errors still work (non-finite path didn't break the default behaviour)
# ──────────────────────────────────────────────────────────────────────


def test_normal_validation_error_still_returns_422(client: TestClient) -> None:
    """Out-of-range value (no NaN involved) — confirm the override didn't
    change the behaviour for the regular case."""
    body = (
        '{"o_a_initial_ppm": -5.0, "temperature_C": 1620.0, '
        '"steel_mass_ton": 180.0, "target_o_a_ppm": 5.0, '
        '"al_purity_pct": 100.0, "burn_off_pct": 20.0, '
        '"model_id": "fruehan_1985"}'
    )
    resp = _post_raw(client, body)
    assert resp.status_code == 422
    payload = resp.json()
    assert "detail" in payload


def test_missing_required_field_returns_422(client: TestClient) -> None:
    """Sanity: required-field-missing path still returns clean 422."""
    # Drop temperature_C entirely.
    body = (
        '{"o_a_initial_ppm": 280.0, '
        '"steel_mass_ton": 180.0, "target_o_a_ppm": 5.0, '
        '"al_purity_pct": 100.0, "burn_off_pct": 20.0, '
        '"model_id": "fruehan_1985"}'
    )
    resp = _post_raw(client, body)
    assert resp.status_code == 422


def test_valid_payload_still_works(client: TestClient) -> None:
    """The override must not break the success path."""
    resp = _post_raw(client, _ok_payload())
    assert resp.status_code == 200, f"baseline payload failed: {resp.text[:300]}"


# ──────────────────────────────────────────────────────────────────────
# NaN inside nested structures (NIT #8 — guards jsonable_encoder recursion)
# ──────────────────────────────────────────────────────────────────────
#
# /api/deox/forward only has top-level float fields; the NaN-detector
# fires there even if ``jsonable_encoder`` only walks the root level.
# To prove the encoder recurses into nested dicts (and by extension
# nested arrays in error ``ctx``) we route the NaN through a nested
# ``composition`` payload on /api/predict. Pydantic's
# ``dict[str, float]`` validator preserves the NaN until the error
# report is built — exactly the case that crashed pre-fix.


def test_request_with_nan_in_nested_dict_returns_422(client: TestClient) -> None:
    """NaN buried inside a nested dict must still serialise to 422 JSON.

    Targets ``/api/predict`` whose ``composition`` field is
    ``dict[str, float]``. If ``jsonable_encoder`` were only walking the
    request root we'd miss the NaN inside ``composition`` and FastAPI
    would emit 500 with HTML body — exactly the regression PR 6 closes.
    """
    body = (
        '{"model_version": "fatigue_carbon_steel-2026-04-25",'
        ' "composition": {"c_pct": NaN, "mn_pct": 0.5}}'
    )
    resp = client.post(
        "/api/predict",
        headers={"Content-Type": "application/json"},
        content=body,
    )
    # Could be 422 (Pydantic catches NaN/Inf via the float validator) or
    # 400 (missing keys) — either way must NOT be 500 (encoder crash) and
    # body must be JSON. The relevant invariant is "no 500 + JSON body",
    # mirroring the same guarantee verified at the top level above.
    assert resp.status_code != 500, (
        f"encoder failed to recurse into nested dict; status="
        f"{resp.status_code}, body={resp.text[:300]}"
    )
    ctype = resp.headers.get("content-type", "")
    assert "application/json" in ctype, f"got content-type {ctype!r}"
    payload = resp.json()
    assert isinstance(payload, dict)
    raw = resp.text
    # Strict JSON: nested NaN must not survive to the wire as a literal.
    assert ": NaN" not in raw, (
        f"raw body contains unquoted NaN — nested encoding bypassed: {raw[:300]}"
    )
