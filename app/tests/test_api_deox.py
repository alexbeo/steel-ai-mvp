"""Tests for /api/deox/* — Al deoxidation calculator (PR 4 of FastAPI migration).

Streamlit parity reference: ``app/frontend/app.py`` lines 920-1219.

The router wraps ``app.backend.deoxidation`` (already covered by the
backend's own unit tests) and adds:
  - request validation (Pydantic field bounds + custom 400 on bad model_id)
  - Pattern Library DX01/DX02/DX03 attachment to the response
  - SafeJSONResponse encoding (numbers, not strings, for any numpy bleed-through)

We don't re-test the physics; we test the API surface and integration.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _baseline_forward_payload() -> dict[str, float | str]:
    """Sane mid-range LF heat — no warnings expected.

    50 ≤ o_a_initial_ppm ≤ 800 (DX01 bounds), target < initial (DX02 ok),
    180 t / 1620 °C / 5 ppm target → matches the Streamlit demo defaults.
    """
    return {
        "o_a_initial_ppm": 280.0,
        "temperature_C": 1620.0,
        "steel_mass_ton": 180.0,
        "target_o_a_ppm": 5.0,
        "al_purity_pct": 100.0,
        "burn_off_pct": 20.0,
        "model_id": "fruehan_1985",
    }


def _baseline_inverse_payload() -> dict[str, float | str]:
    return {
        "o_a_before_ppm": 500.0,
        "o_a_after_ppm": 10.0,
        "al_added_kg": 65.0,
        "temperature_C": 1620.0,
        "steel_mass_ton": 180.0,
        "burn_off_pct": 20.0,
        "model_id": "fruehan_1985",
    }


# ───────── /api/deox/models ───────────────────────────────────────────


def test_thermo_models_list(client: TestClient) -> None:
    """Registry exposes 3 thermo models with expected IDs."""
    resp = client.get("/api/deox/models")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 3

    ids = {item["id"] for item in payload["items"]}
    # Note: the backend registry uses ``hayashi_2013`` (not the
    # ``hayashi_yamamoto_2013`` slug from the design doc spec). We follow
    # the source of truth — the actual THERMO_MODELS keys.
    assert ids == {"fruehan_1985", "sigworth_elliott_1974", "hayashi_2013"}

    # Default model present and flagged.
    defaults = [item for item in payload["items"] if item["is_default"]]
    assert len(defaults) == 1
    assert defaults[0]["id"] == payload["default"]

    # Per-model shape sanity.
    fruehan = next(item for item in payload["items"] if item["id"] == "fruehan_1985")
    assert isinstance(fruehan["valid_t_range_c"], list) and len(fruehan["valid_t_range_c"]) == 2
    assert isinstance(fruehan["expected_accuracy_ppm"], (int, float))


# ───────── /api/deox/forward ──────────────────────────────────────────


def test_forward_happy_path(client: TestClient) -> None:
    """Valid forward request returns AlDemandResult shape + warnings list."""
    resp = client.post("/api/deox/forward", json=_baseline_forward_payload())
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    assert set(payload.keys()) == {"result", "pattern_warnings"}
    r = payload["result"]
    # Backend dataclass fields surface verbatim via asdict.
    for key in (
        "al_total_kg", "al_active_kg", "al_burn_off_kg",
        "o_a_expected_ppm", "al_per_ton", "cost_eur",
        "currency", "model_id", "inputs", "warnings",
    ):
        assert key in r, f"missing key {key}"

    assert r["al_total_kg"] > 0
    assert r["model_id"] == "fruehan_1985"
    # Mid-range inputs satisfy DX01 (50-800) and DX02 (target<initial) →
    # no pattern warnings expected. Backend physics warnings (T-range) may
    # exist but are surfaced via result.warnings, not pattern_warnings.
    assert isinstance(payload["pattern_warnings"], list)
    assert len(payload["pattern_warnings"]) == 0


def test_forward_triggers_dx01_when_o_a_below_range(client: TestClient) -> None:
    """O_a < 50 ppm is below DX01 LF range → HIGH warning attached."""
    payload = _baseline_forward_payload()
    payload["o_a_initial_ppm"] = 25.0  # well below the 50 ppm DX01 floor
    payload["target_o_a_ppm"] = 5.0

    resp = client.post("/api/deox/forward", json=payload)
    assert resp.status_code == 200
    warnings = resp.json()["pattern_warnings"]
    ids = {w["id"] for w in warnings}
    assert "DX01" in ids
    dx01 = next(w for w in warnings if w["id"] == "DX01")
    assert dx01["severity"] == "HIGH"
    assert "ppm" in dx01["message"]


def test_forward_triggers_dx02_when_target_above_initial(client: TestClient) -> None:
    """target_o_a >= o_a_initial → DX02 MEDIUM warning attached."""
    payload = _baseline_forward_payload()
    payload["o_a_initial_ppm"] = 60.0  # keep DX01 happy
    payload["target_o_a_ppm"] = 80.0   # but target above initial → DX02

    resp = client.post("/api/deox/forward", json=payload)
    assert resp.status_code == 200
    warnings = resp.json()["pattern_warnings"]
    ids = {w["id"] for w in warnings}
    assert "DX02" in ids


def test_forward_validation_422_on_missing_field(client: TestClient) -> None:
    """Pydantic field validation → 422 with structured detail."""
    payload = _baseline_forward_payload()
    del payload["temperature_C"]
    resp = client.post("/api/deox/forward", json=payload)
    assert resp.status_code == 422


def test_forward_invalid_model_id_400(client: TestClient) -> None:
    """Unknown model_id → custom 400 (router-level) with available IDs listed."""
    payload = _baseline_forward_payload()
    payload["model_id"] = "not_a_real_model"
    resp = client.post("/api/deox/forward", json=payload)
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "not_a_real_model" in detail
    assert "fruehan_1985" in detail


def test_forward_validation_422_on_out_of_bounds(client: TestClient) -> None:
    """Field bounds (al_purity_pct must be > 0 and <= 100) → 422."""
    payload = _baseline_forward_payload()
    payload["al_purity_pct"] = 150.0  # > 100 → reject
    resp = client.post("/api/deox/forward", json=payload)
    assert resp.status_code == 422


def test_forward_rejects_purity_below_50(client: TestClient) -> None:
    """``al_purity_pct < 50`` is below UI lower bound → API must 422.

    The Streamlit slider (app.py:1059) and the JS form (deox.js
    ``renderForwardForm`` field) both clamp to 50–100 %. A bare-API call
    with ``al_purity_pct=30`` previously slipped through Pydantic
    (``gt=0, le=100``) — defensive parity with the UI requires rejection.
    """
    payload = _baseline_forward_payload()
    payload["al_purity_pct"] = 30.0  # below 50 % UI floor
    resp = client.post("/api/deox/forward", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    # Surface field name in detail so a misconfigured client can pinpoint.
    assert any(
        "al_purity_pct" in (err.get("loc") or [])
        or "al_purity_pct" in str(err.get("msg", ""))
        for err in body.get("detail", [])
    )


# ───────── /api/deox/inverse ──────────────────────────────────────────


def test_inverse_happy_path(client: TestClient) -> None:
    """Valid inverse request returns effective purity + result shape."""
    resp = client.post("/api/deox/inverse", json=_baseline_inverse_payload())
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    r = payload["result"]
    for key in (
        "effective_purity_pct", "effective_active_kg",
        "expected_active_kg", "assumed_burn_off_pct",
        "model_id", "inputs", "warnings",
    ):
        assert key in r
    assert isinstance(r["effective_purity_pct"], (int, float))
    assert r["effective_purity_pct"] > 0
    assert isinstance(payload["pattern_warnings"], list)


def test_inverse_after_above_before_400(client: TestClient) -> None:
    """o_a_after >= o_a_before → backend ValueError → router 400."""
    payload = _baseline_inverse_payload()
    payload["o_a_before_ppm"] = 100.0
    payload["o_a_after_ppm"] = 200.0  # observed deox depth negative
    resp = client.post("/api/deox/inverse", json=payload)
    assert resp.status_code == 400
    assert "deox" in resp.json()["detail"].lower() or "before" in resp.json()["detail"].lower()


def test_inverse_dx03_when_low_effective_purity(client: TestClient) -> None:
    """Tiny observed deox depth on a large Al charge → effective purity
    drops well below 70% → DX03 MEDIUM warning attached."""
    payload = _baseline_inverse_payload()
    # 500 → 490 ppm (small deox depth) on 200 kg Al / 100 t / 20% burn-off.
    # Effective active Al ≈ (10 ppm × 100 t × 1000 kg/t × 1.124) / 1e6 ≈ 1.12 kg
    # Expected at 100% purity = 200 × 0.8 = 160 kg → effective purity ≈ 0.7%
    payload["o_a_before_ppm"] = 500.0
    payload["o_a_after_ppm"] = 490.0
    payload["al_added_kg"] = 200.0
    payload["steel_mass_ton"] = 100.0
    resp = client.post("/api/deox/inverse", json=payload)
    assert resp.status_code == 200
    warnings = resp.json()["pattern_warnings"]
    ids = {w["id"] for w in warnings}
    assert "DX03" in ids


# ───────── /api/deox/compare ──────────────────────────────────────────


def test_compare_returns_three_models(client: TestClient) -> None:
    """Compare runs the full registry on identical inputs."""
    resp = client.post("/api/deox/compare", json=_baseline_forward_payload())
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    assert set(payload.keys()) == {"models", "spread_pct", "pattern_warnings"}
    assert set(payload["models"].keys()) == {
        "fruehan_1985", "sigworth_elliott_1974", "hayashi_2013",
    }
    for model_result in payload["models"].values():
        assert "al_total_kg" in model_result
        assert "model_id" in model_result
        assert model_result["al_total_kg"] > 0


def test_compare_models_have_consistent_inputs(client: TestClient) -> None:
    """All three models receive identical heat inputs (mass + target O_a +
    purity) — they only disagree on the thermo formula. al_total_kg
    differs (purpose of the comparison) but the input echo matches."""
    payload = _baseline_forward_payload()
    resp = client.post("/api/deox/compare", json=payload)
    assert resp.status_code == 200
    models = resp.json()["models"]
    masses = [m["steel_mass_ton"] for m in (m["inputs"] for m in models.values())]
    targets = [m["target_o_a_ppm"] for m in (m["inputs"] for m in models.values())]
    assert len(set(masses)) == 1
    assert len(set(targets)) == 1
    assert masses[0] == pytest.approx(payload["steel_mass_ton"])
    assert targets[0] == pytest.approx(payload["target_o_a_ppm"])


def test_compare_spread_pct_is_numeric(client: TestClient) -> None:
    """spread_pct is the relative max-min over mean — should be small for
    realistic inputs (academic disagreement <50%) and a real number."""
    resp = client.post("/api/deox/compare", json=_baseline_forward_payload())
    assert resp.status_code == 200
    payload = resp.json()
    assert isinstance(payload["spread_pct"], (int, float))
    assert payload["spread_pct"] >= 0.0


# ───────── SafeJSONResponse regression ────────────────────────────────


def test_safe_json_numeric(client: TestClient) -> None:
    """Numbers in the response must be native JSON numbers, not strings.

    Even though deoxidation.py uses pure Python floats today, this
    regression locks in the SafeJSONResponse contract so future numpy
    creep (e.g. switching to vectorised computation) doesn't ship strings
    to the chart layer.
    """
    resp = client.post("/api/deox/forward", json=_baseline_forward_payload())
    assert resp.status_code == 200
    payload = resp.json()
    r = payload["result"]
    for key in ("al_total_kg", "al_active_kg", "o_a_expected_ppm", "cost_eur"):
        assert isinstance(r[key], (int, float)), (
            f"{key} must be numeric, got {type(r[key])}"
        )

    cmp_resp = client.post("/api/deox/compare", json=_baseline_forward_payload())
    assert cmp_resp.status_code == 200
    assert isinstance(cmp_resp.json()["spread_pct"], (int, float))
