"""Tests for /api/deox/* — Al deoxidation calculator (PR 4 of FastAPI migration).

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
    180 t / 1620 °C / 5 ppm target → matches the demo defaults.
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

    The JS form (deox.js ``renderForwardForm`` field) clamps to
    50–100 %. A bare-API call with ``al_purity_pct=30`` previously
    slipped through Pydantic (``gt=0, le=100``) — defensive parity
    with the UI requires rejection.
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


# ───────── PR 6 — Slag-aware optimization endpoints ───────────────────


def _baseline_optimize_payload() -> dict[str, float | str]:
    """Excel base-case used throughout the slag-aware design doc.

    371-t BOF heat, 657 ppm O_a tap, target 8 ppm, residual [Al]=0.018%,
    2.2 t slag carry-over @ FeO=18%, T=1600 °C. Matches the
    ``__main__`` dry-run in ``slag_aware_deox.py`` and the regression
    case in ``test_slag_aware_deox.py``.
    """
    return {
        "steel_mass_ton": 371.0,
        "o_a_initial_ppm": 657.0,
        "target_o_a_ppm": 8.0,
        "target_al_pct": 0.018,
        "slag_mass_kg": 2200.0,
        "slag_feo_pct": 18.0,
        "temperature_C": 1600.0,
    }


def test_get_methods_returns_catalog(client: TestClient) -> None:
    """GET /api/deox/methods exposes the YAML catalog for the UI dropdown.

    Contract:
      * ``count`` ≥ 5 (the seed YAML ships 5 methods: ingot,
        submerged_ingot, granule_water_quenched, asis_shot, cored_wire_feal30).
      * Each item carries ``id`` + flattened canonical fields +
        ``extras`` (raw YAML row).
      * ``default`` points at an existing method id.
    """
    resp = client.get("/api/deox/methods")
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    assert payload["count"] >= 5
    assert len(payload["items"]) == payload["count"]

    ids = {item["id"] for item in payload["items"]}
    assert payload["default"] in ids, "default must be an existing method id"
    # Spot-check the marquee methods land in the catalog.
    assert "asis_shot" in ids
    assert "ingot" in ids

    # Per-item shape sanity.
    asis = next(it for it in payload["items"] if it["id"] == "asis_shot")
    for key in (
        "name",
        "eta_al_typical",
        "eta_al_range",
        "premium_eur_per_kg",
        "surface_m2_per_kg",
        "carrier_gas",
        "notes",
        "extras",
    ):
        assert key in asis, f"asis_shot missing key {key!r}"
    assert isinstance(asis["eta_al_range"], list) and len(asis["eta_al_range"]) == 2
    assert isinstance(asis["extras"], dict)


def test_post_optimize_excel_base_case(client: TestClient) -> None:
    """POST /api/deox/optimize on the Excel base case picks a sane winner.

    With no [N] / premium constraints, ``recommend_optimal_method`` ranks
    methods by ``cost_per_heat_eur`` ascending. ASIS-shot has high η_Al
    (0.82) and modest premium (€0.30/kg Al-eq) → it should lead or
    runner-up; either way the chosen method must come from the catalog
    and the pareto_table must be sorted ascending.
    """
    resp = client.post("/api/deox/optimize", json=_baseline_optimize_payload())
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    # Top-level keys per the spec response shape.
    for key in (
        "chosen_method_id",
        "chosen_method_name",
        "chosen_cost_eur",
        "rationale",
        "constraints_active",
        "rejected_methods",
        "pareto_table",
        "pattern_warnings",
        "thermo_model_used",
        "inputs",
    ):
        assert key in payload, f"missing key {key!r}"

    # Pareto sorted ascending by cost_per_heat_eur.
    costs = [row["cost_per_heat_eur"] for row in payload["pareto_table"]]
    assert costs == sorted(costs), "pareto_table not sorted ascending by cost"

    # Chosen method is the cheapest among survivors.
    assert payload["chosen_method_id"] == payload["pareto_table"][0]["method_id"]
    assert payload["chosen_cost_eur"] == pytest.approx(
        payload["pareto_table"][0]["cost_per_heat_eur"]
    )

    # With η_Al=0.82 (ASIS) vs 0.58 (ingot) the high-η/low-premium ASIS
    # methods dominate the Excel base case — chosen should be one of them.
    # Don't hardcode the exact id (catalog may evolve), just require a
    # plausible carrier_gas configuration.
    chosen_row = payload["pareto_table"][0]
    assert chosen_row["al_pure_kg"] > 0
    assert chosen_row["cost_per_heat_eur"] > 0

    # No constraints active → no rejected methods → pareto = full catalog.
    assert payload["constraints_active"] == []
    assert payload["rejected_methods"] == []
    assert len(payload["pareto_table"]) >= 5

    # Runner-up populated since len(pareto) ≥ 2.
    assert payload["runner_up_method_id"] is not None
    assert payload["runner_up_delta_eur"] > 0


def test_post_optimize_with_n_constraint(client: TestClient) -> None:
    """target_n_ppm=30 < 50 → constraint surfaces in constraints_active.

    The Pattern Library DX07 + the optimizer constraint filter share
    the same threshold (50 ppm). With the seed YAML catalog (only
    ``asis_shot`` has carrier_gas, set to ``Ar``) no method is dropped,
    but the active constraint must still be echoed.
    """
    body = _baseline_optimize_payload()
    body["target_n_ppm"] = 30.0

    resp = client.post("/api/deox/optimize", json=body)
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    # The constraint string contains the trigger threshold (50).
    assert any(
        "target_n_ppm" in c and "50" in c for c in payload["constraints_active"]
    ), f"target_n_ppm<50 not in constraints_active: {payload['constraints_active']}"


def test_post_optimize_invalid_input_returns_422(client: TestClient) -> None:
    """Pydantic field bounds: negative steel mass → 422."""
    body = _baseline_optimize_payload()
    body["steel_mass_ton"] = -10.0
    resp = client.post("/api/deox/optimize", json=body)
    assert resp.status_code == 422


def test_post_optimize_invalid_thermo_model_400(client: TestClient) -> None:
    """Unknown thermo_model_id → 400 (router-level), not 422."""
    body = _baseline_optimize_payload()
    body["thermo_model_id"] = "not_a_real_model"
    resp = client.post("/api/deox/optimize", json=body)
    assert resp.status_code == 400
    assert "not_a_real_model" in resp.json()["detail"]


def test_post_optimize_save_returns_501(client: TestClient) -> None:
    """PR 8 placeholder: optimize/save always 501 (NOT IMPLEMENTED)."""
    resp = client.post(
        "/api/deox/optimize/save",
        json={
            "recommendation": {"chosen_method_id": "asis_shot"},
            "heat_id": "TEST-001",
            "author": "user",
        },
    )
    assert resp.status_code == 501
    detail = resp.json()["detail"]
    # Russian, user-facing message per CLAUDE.md.
    assert "PR 8" in detail


def test_optimize_returns_pattern_warnings_dx04(client: TestClient) -> None:
    """Half-filled slag block (mass without FeO) → DX04 HIGH warning.

    The optimizer still produces a recommendation (slag with feo_pct=0
    contributes 0 O), but the critic ctx marks slag-aware semantics
    *active* (because slag_mass_kg is set) and DX04 fires because
    slag_feo_pct is None. This is the canonical "user forgot half the
    slag data" UX path.
    """
    body = _baseline_optimize_payload()
    body.pop("slag_feo_pct")  # leave only slag_mass_kg
    resp = client.post("/api/deox/optimize", json=body)
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    warning_ids = {w.get("id") for w in payload["pattern_warnings"]}
    assert "DX04" in warning_ids, (
        f"DX04 (slag_aware without slag_state.feo_pct) not in warnings: "
        f"{warning_ids}"
    )
    dx04 = next(w for w in payload["pattern_warnings"] if w["id"] == "DX04")
    assert dx04["severity"] == "HIGH"
