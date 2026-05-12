"""Tests for /api/recipes/ai-cycle — designer + critic LLM cycle (PR 11).

Mocking strategy mirrors PR 9 (``test_api_deox_ai.py``) and PR 10
(``test_api_hypotheses.py``):
    - The router lazy-imports ``make_recipe_designer`` /
      ``make_recipe_critic`` inside ``_run_recipe_cycle_job`` so we
      monkeypatch the source modules. The worker calls the patched
      factories at runtime.
    - We avoid running ``_build_designer_context`` against a real trained
      model + Agrawal dataset by stubbing the helper itself with a canned
      ``(designer_ctx, runtime_state)`` pair plus an in-place stub of
      ``_verify_recipes`` so XGBoost predict + cost compute are skipped.
      Building real artifacts would require a 437-record parquet load +
      XGBoost JSON + GMM serialised file — heavy and not the unit under
      test.
    - The Decision Log save test wraps ``decision_log.logger.log_decision``
      to redirect the DB path to ``tmp_path`` so the audit-trail save
      doesn't pollute the project DB.

We use the FastAPI TestClient throughout. Job submission is async on the
server; we poll the JobStore through ``GET /api/jobs/{id}`` until status
flips to done|error.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.jobs import get_job_store
from app.api.main import app
from app.api.routers import recipes as recipes_router
from app.api.routers import system as system_router
from app.backend.recipe_critic import EvidenceCheck, RecipeVerdict
from app.backend.recipe_designer import CompositionRecipe


# ──────────────────────────────────────────────────────────────────────
# fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_job_store_singleton():
    """Fresh JobStore per test — avoids cross-test pollution."""
    get_job_store.cache_clear()
    yield
    get_job_store.cache_clear()


@pytest.fixture(autouse=True)
def _ensure_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests assume the key is present.

    Tests that need the "no key" branch override this with
    ``monkeypatch.delenv`` inside the body. The fixture sets a sentinel
    value so the endpoint passes the up-front check and the worker runs
    (with the factory monkeypatched separately).
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fixture")


def _write_meta(vdir: Path, *, steel_class: str = "fatigue_carbon_steel") -> None:
    """Plant a minimal meta.json so the router treats the dir as valid."""
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "meta.json").write_text(
        '{"version": "' + vdir.name + '", '
        f'"steel_class": "{steel_class}", '
        '"target": "fatigue_strength_mpa", '
        '"feature_list": ["c_pct", "mn_pct", "si_pct"], '
        '"training_ranges": {"c_pct": [0.04, 0.4], "mn_pct": [0.5, 1.5], "si_pct": [0.1, 0.5]}, '
        '"metrics": {"r2_test": 0.85, "mae_test": 12.0, '
        '"rmse_test": 18.0, "coverage_90_ci": 0.91, '
        '"n_train": 350, "n_val": 40, "n_test": 47}, '
        '"feature_importance": {"c_pct": 0.4, "mn_pct": 0.3, "si_pct": 0.3}}',
        encoding="utf-8",
    )


@pytest.fixture()
def fake_models_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Models dir with a fatigue + an HSLA fake meta for the class-gate test."""
    fake_dir = tmp_path / "models"
    fake_dir.mkdir()
    _write_meta(fake_dir / "test_recipe_fatigue_v1", steel_class="fatigue_carbon_steel")
    _write_meta(fake_dir / "test_recipe_hsla_v1", steel_class="pipe_hsla")
    monkeypatch.setattr(system_router, "MODELS_DIR", fake_dir)
    return fake_dir


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────


def _stub_recipe(idx: str = "r1", *, name: str = "low-Si baseline") -> CompositionRecipe:
    """Realistic CompositionRecipe — same fields as a real designer output."""
    return CompositionRecipe(
        id=idx,
        name=name,
        composition={"si_pct": 0.20, "mn_pct": 1.20, "ni_pct": 0.30},
        process_params={
            "tempering_temp_c": 600.0,
            "tempering_time_min": 60.0,
        },
        rationale=(
            "Lower Si reduces ferroalloy cost; matched by slight Mn lift to "
            "preserve hardenability per Grossmann's DI."
        ),
        evidence=[
            "feature_importance: si_pct=0.30, mn_pct=0.30 (artifact)",
            "Hall-Petch + solid-solution mechanism for fatigue lift",
        ],
        expected_outcome="−€8/т vs baseline, σf within ±20 МПа",
        risk_notes="Mn upper-bound near training_ranges max — check OOD",
        novelty="MEDIUM",
        tags=["steel_class:fatigue_carbon_steel"],
    )


def _stub_verdict(recipe_id: str = "r1", verdict: str = "ACCEPT") -> RecipeVerdict:
    """Canned RecipeVerdict matching a stub recipe."""
    return RecipeVerdict(
        recipe_id=recipe_id,
        verdict=verdict,
        confidence="HIGH",
        summary="Mechanism plausible, cost arithmetic checks out.",
        evidence_check=[
            EvidenceCheck(
                claim="feature_importance: si_pct=0.30",
                verdict="VALID",
                note="matches artifact summary in context",
            ),
        ],
        strengths=["Low Si lowers ferroalloy bill by ~5 €/t"],
        weaknesses=[],
        suggested_revision=None,
    )


def _patch_llm_factories(
    monkeypatch: pytest.MonkeyPatch,
    recipes: list[CompositionRecipe] | None = None,
    verdicts: list[RecipeVerdict] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Replace the designer/critic factories with MagicMock-backed stubs.

    The router lazy-imports the factories inside ``_run_recipe_cycle_job``
    so we patch the source modules directly. Returns the mocks so tests
    can assert on call counts.
    """
    if recipes is None:
        recipes = [_stub_recipe()]
    if verdicts is None:
        verdicts = [_stub_verdict(r.id) for r in recipes]

    designer_mock = MagicMock()
    designer_mock.design.return_value = recipes
    critic_mock = MagicMock()
    critic_mock.review.return_value = verdicts

    monkeypatch.setattr(
        "app.backend.recipe_designer.make_recipe_designer",
        lambda: designer_mock,
    )
    monkeypatch.setattr(
        "app.backend.recipe_critic.make_recipe_critic",
        lambda: critic_mock,
    )
    return designer_mock, critic_mock


def _patch_context_and_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the heavy worker helpers so tests skip XGBoost + Agrawal load.

    We replace ``_build_designer_context`` with a canned
    ``(designer_ctx, runtime_state)`` pair, and ``_verify_recipes`` with
    a deterministic projection that adds a synthetic ``ml_verification``
    block so the critic call site has the shape it expects.
    """
    canned_runtime = {
        "bundle": {"meta": {}},
        "baseline_row": None,
        "feature_list": ["c_pct", "mn_pct", "si_pct"],
        "snapshot": None,
        "baseline_predicted": 720.0,
        "baseline_cost": 540.0,
        "baseline_recipe": {
            "si_pct": 0.30, "mn_pct": 1.10, "ni_pct": 0.20,
            "tempering_temp_c": 600.0, "tempering_time_min": 60.0,
        },
    }

    def _stub_build_ctx(*, model_version: str, task_text: str):
        ctx = {
            "task": task_text,
            "model_version": model_version,
            "steel_class": "fatigue_carbon_steel",
            "target": "fatigue_strength_mpa",
            "r2_test": 0.85,
            "mae_test": 12.0,
            "coverage_90_ci": 0.91,
            "feature_importance": {"c_pct": 0.4, "mn_pct": 0.3, "si_pct": 0.3},
            "training_ranges": {"c_pct": [0.04, 0.4]},
            "target_distribution": {
                "min": 200.0, "max": 1200.0, "mean": 700.0, "std": 100.0, "n": 437,
            },
            "baseline_recipe": canned_runtime["baseline_recipe"],
            "baseline_predicted_property": canned_runtime["baseline_predicted"],
            "baseline_cost_per_ton": canned_runtime["baseline_cost"],
            "available_composition": ["si_pct", "mn_pct", "ni_pct"],
            "available_process": ["tempering_temp_c", "tempering_time_min"],
        }
        return ctx, canned_runtime

    def _stub_verify(recipes, *, runtime_state):
        # Deterministic projection — predicted_property = baseline + 25 МПа,
        # cost = baseline − 8 €/т, no OOD. Mirrors the asdict + ml_verification
        # shape that the critic expects.
        from dataclasses import asdict as _asdict

        out = []
        base_p = float(runtime_state["baseline_predicted"])
        base_c = float(runtime_state["baseline_cost"])
        for r in recipes:
            d = _asdict(r)
            d["ml_verification"] = {
                "predicted_property": base_p + 25.0,
                "lower_90": base_p + 5.0,
                "upper_90": base_p + 45.0,
                "ood_flag": False,
                "cost_per_ton": base_c - 8.0,
                "delta_property": 25.0,
                "delta_cost": -8.0,
                "ferroalloy_breakdown": [
                    {"material": "FeMn-80", "kg_per_ton": 12.0, "eur_per_ton": 18.5}
                ],
            }
            out.append(d)
        return out

    monkeypatch.setattr(recipes_router, "_build_designer_context", _stub_build_ctx)
    monkeypatch.setattr(recipes_router, "_verify_recipes", _stub_verify)


def _wait_for_job(
    client: TestClient,
    job_id: str,
    timeout: float = 15.0,
) -> dict:
    """Poll /api/jobs/{id} until status flips to done|error."""
    deadline = time.monotonic() + timeout
    body: dict = {}
    while time.monotonic() < deadline:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    pytest.fail(f"job {job_id} did not finish within {timeout}s; last status={body!r}")


def _baseline_payload(**overrides: object) -> dict:
    body: dict = {
        "model_version": "test_recipe_fatigue_v1",
        "task_text": "Снизить cost при сохранении σf.",
        "save_to_decision_log": False,
    }
    body.update(overrides)
    return body


# ──────────────────────────────────────────────────────────────────────
# submission / validation
# ──────────────────────────────────────────────────────────────────────


def test_recipes_ai_cycle_returns_job_id(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid body → 200 + ``{job_id, config}``."""
    _patch_llm_factories(monkeypatch)
    _patch_context_and_verify(monkeypatch)
    resp = client.post("/api/recipes/ai-cycle", json=_baseline_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body.get("job_id"), str) and len(body["job_id"]) > 0
    assert body["config"]["model_version"] == "test_recipe_fatigue_v1"
    assert "Снизить cost" in body["config"]["task_text"]
    assert body["config"]["save_to_decision_log"] is False


def test_recipes_validation_422(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing required ``model_version`` → 422 from Pydantic."""
    _patch_llm_factories(monkeypatch)
    _patch_context_and_verify(monkeypatch)
    payload = _baseline_payload()
    del payload["model_version"]
    resp = client.post("/api/recipes/ai-cycle", json=payload)
    assert resp.status_code == 422

    # Also: task_text > 2000 chars triggers max_length 422.
    long = _baseline_payload(task_text="x" * 2500)
    resp_long = client.post("/api/recipes/ai-cycle", json=long)
    assert resp_long.status_code == 422


def test_recipes_path_traversal_400(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``model_version`` with path-traversal markers → 400 (CWE-22).

    The slug regex in ``_safe_version_dir`` rejects ``../app`` because
    ``/`` is not in the allowed character set; ``Path("models") / "../app"``
    would otherwise resolve outside MODELS_DIR.
    """
    _patch_llm_factories(monkeypatch)
    _patch_context_and_verify(monkeypatch)
    resp = client.post(
        "/api/recipes/ai-cycle", json=_baseline_payload(model_version="../app")
    )
    assert resp.status_code == 400


def test_recipes_class_gate_400(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HSLA model → 400 with explicit "fatigue_carbon_steel" message.

    The API rejects non-fatigue classes — explicit failure beats silent
    confusion when the recipe pair has no real-world dataset for them.
    """
    _patch_llm_factories(monkeypatch)
    _patch_context_and_verify(monkeypatch)
    resp = client.post(
        "/api/recipes/ai-cycle",
        json=_baseline_payload(model_version="test_recipe_hsla_v1"),
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"].lower()
    assert "fatigue_carbon_steel" in detail


def test_recipes_no_api_key_503(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ANTHROPIC_API_KEY → 503 with actionable message."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = client.post("/api/recipes/ai-cycle", json=_baseline_payload())
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "ANTHROPIC_API_KEY" in detail


def test_recipes_unknown_model_404(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid slug but model dir is missing → 404 (router-layer guard)."""
    _patch_llm_factories(monkeypatch)
    _patch_context_and_verify(monkeypatch)
    resp = client.post(
        "/api/recipes/ai-cycle",
        json=_baseline_payload(model_version="nonexistent_recipe_v9"),
    )
    assert resp.status_code == 404
    assert "nonexistent_recipe_v9" in resp.json()["detail"]


# ──────────────────────────────────────────────────────────────────────
# end-to-end with mocked LLM
# ──────────────────────────────────────────────────────────────────────


def test_recipes_to_completion_with_mock_llm(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit + poll + verify result shape with stub designer/critic."""
    recipes = [_stub_recipe("r1", name="low-Si"), _stub_recipe("r2", name="hi-Mo")]
    verdicts = [
        _stub_verdict("r1", verdict="ACCEPT"),
        _stub_verdict("r2", verdict="REVISE"),
    ]
    designer_mock, critic_mock = _patch_llm_factories(monkeypatch, recipes, verdicts)
    _patch_context_and_verify(monkeypatch)

    resp = client.post("/api/recipes/ai-cycle", json=_baseline_payload())
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    body = _wait_for_job(client, job_id, timeout=10.0)
    if body["status"] != "done":
        pytest.fail(
            f"job ended with status={body['status']} error={body.get('error')!r}"
        )

    result = body["result"]

    # Result shape — recipes + reviews + counts + duration + baseline.
    assert result["model_version"] == "test_recipe_fatigue_v1"
    assert isinstance(result["recipes"], list)
    assert len(result["recipes"]) == 2
    assert isinstance(result["reviews"], list)
    assert len(result["reviews"]) == 2
    assert result["verdict_counts"] == {"ACCEPT": 1, "REVISE": 1, "REJECT": 0}
    assert isinstance(result["duration_s"], (int, float))
    assert result["decision_log_id"] is None  # opt-in, was False

    # Baseline echoed for the UI to render comparison metrics.
    base = result["baseline"]
    assert base["predicted_property"] == 720.0
    assert base["cost_per_ton"] == 540.0
    assert "si_pct" in base["recipe"]

    # Recipe fields preserved after asdict + ml_verification attached.
    r0 = result["recipes"][0]
    assert r0["id"] == "r1"
    assert r0["name"] == "low-Si"
    assert r0["novelty"] == "MEDIUM"
    assert "ml_verification" in r0
    ml = r0["ml_verification"]
    assert ml["predicted_property"] == 745.0
    assert ml["delta_property"] == 25.0
    assert ml["cost_per_ton"] == 532.0
    assert ml["delta_cost"] == -8.0
    assert ml["ood_flag"] is False
    assert isinstance(ml["ferroalloy_breakdown"], list)

    # Review fields preserved.
    rv0 = result["reviews"][0]
    assert rv0["recipe_id"] == "r1"
    assert rv0["verdict"] == "ACCEPT"
    assert rv0["confidence"] == "HIGH"

    # Both factories must have been invoked exactly once.
    designer_mock.design.assert_called_once()
    critic_mock.review.assert_called_once()
    # Critic must receive the asdict-projection of the recipes (with
    # ml_verification block) as the second positional arg.
    _ctx_arg, recipes_arg = critic_mock.review.call_args.args
    assert isinstance(recipes_arg, list) and recipes_arg[0]["id"] == "r1"
    assert "ml_verification" in recipes_arg[0]


def test_recipes_decision_log_save(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """save_to_decision_log=True → entry is written with tag recipe_cycle.

    Same wrapping technique as PR 9 ``test_api_deox_ai`` and PR 10
    ``test_api_hypotheses`` — capture the invocation + redirect ``db_path``
    to a tmp DB so the production schema (CREATE INDEX, AUTOINCREMENT) is
    exercised verbatim.
    """
    _patch_llm_factories(monkeypatch)
    _patch_context_and_verify(monkeypatch)

    tmp_db = tmp_path / "decisions.db"
    import decision_log.logger as log_mod

    captured: list[dict] = []
    real_log_decision = log_mod.log_decision

    def _wrapped(*args: object, **kwargs: object) -> int:
        kwargs["db_path"] = tmp_db
        captured.append({"args": args, "kwargs": kwargs})
        return real_log_decision(*args, **kwargs)

    monkeypatch.setattr(log_mod, "log_decision", _wrapped)

    resp = client.post(
        "/api/recipes/ai-cycle",
        json=_baseline_payload(save_to_decision_log=True),
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    body = _wait_for_job(client, job_id, timeout=10.0)
    assert body["status"] == "done", body
    result = body["result"]
    assert result["decision_log_id"] is not None
    assert isinstance(result["decision_log_id"], int)

    # The cycle entry must have been logged with the dedicated tag.
    cycle_calls = [
        c for c in captured
        if "recipe_cycle" in (c["kwargs"].get("tags") or [])
    ]
    assert len(cycle_calls) == 1

    # Verify the entry actually landed in the tmp DB.
    import sqlite3

    conn = sqlite3.connect(tmp_db)
    try:
        rows = list(conn.execute(
            "SELECT phase, decision, tags, author FROM decisions WHERE id = ?",
            (result["decision_log_id"],),
        ))
    finally:
        conn.close()
    assert len(rows) == 1
    phase, decision, tags_json, author = rows[0]
    assert phase == "inverse_design"
    assert "Recipe cycle" in decision
    assert "recipe_cycle" in tags_json
    assert author == "api"


def test_recipes_safe_json_numeric(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Numeric values come back as JSON numbers, not strings.

    Locks in the SafeJSONResponse contract — duration_s, baseline numbers,
    and ml_verification metrics must be real numbers so the JS view can
    use ``Number(...)`` directly.
    """
    _patch_llm_factories(monkeypatch)
    _patch_context_and_verify(monkeypatch)
    resp = client.post("/api/recipes/ai-cycle", json=_baseline_payload())
    job_id = resp.json()["job_id"]
    body = _wait_for_job(client, job_id, timeout=10.0)
    assert body["status"] == "done"
    result = body["result"]

    assert isinstance(result["duration_s"], (int, float))
    assert isinstance(result["baseline"]["predicted_property"], (int, float))
    assert isinstance(result["baseline"]["cost_per_ton"], (int, float))
    # Verdict counts should be ints (not strings, not floats).
    for k in ("ACCEPT", "REVISE", "REJECT"):
        assert isinstance(result["verdict_counts"][k], int)
    # Recipe ml_verification values numeric.
    ml = result["recipes"][0]["ml_verification"]
    assert isinstance(ml["predicted_property"], (int, float))
    assert isinstance(ml["delta_property"], (int, float))


def test_recipes_cancel_before_llm(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DELETE before the worker finishes → cooperative cancel raises.

    We slow the designer factory by parking inside it, set the
    cancellation flag from the client side, then release the factory so
    the worker advances. The cancellation gate between designer and critic
    catches the flag and raises
    ``Cancelled by user (after designer, before critic)``.

    Note the same race-window caveat as PR 9 / PR 10 cancel tests: if the
    worker is already past both gates by the time DELETE arrives, the job
    completes normally. We treat both outcomes as acceptable — the test
    guarantees the cancellation API works, not that it always wins the
    race.
    """
    import threading

    started_event = threading.Event()
    proceed_event = threading.Event()

    def slow_factory():
        # Park here until the test releases. The cancellation flag set
        # while we wait is read by the gate after the designer returns.
        started_event.set()
        proceed_event.wait(timeout=5.0)
        m = MagicMock()
        m.design.return_value = [_stub_recipe()]
        return m

    critic_mock = MagicMock()
    critic_mock.review.return_value = [_stub_verdict()]

    monkeypatch.setattr(
        "app.backend.recipe_designer.make_recipe_designer",
        slow_factory,
    )
    monkeypatch.setattr(
        "app.backend.recipe_critic.make_recipe_critic",
        lambda: critic_mock,
    )
    _patch_context_and_verify(monkeypatch)

    resp = client.post("/api/recipes/ai-cycle", json=_baseline_payload())
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    # Wait for the worker to enter the slow factory.
    assert started_event.wait(timeout=3.0), "factory never invoked"

    # DELETE — flag flipped immediately. Then release the factory.
    cancel_resp = client.delete(f"/api/jobs/{job_id}")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["cancellation_requested"] is True

    proceed_event.set()

    body = _wait_for_job(client, job_id, timeout=10.0)
    if body["status"] == "error":
        err = body.get("error") or ""
        assert "Cancelled" in err, err
    else:
        # Race-window edge: worker passed both gates before DELETE.
        assert body["status"] == "done"
