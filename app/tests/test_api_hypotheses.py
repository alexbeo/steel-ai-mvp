"""Tests for /api/hypotheses/run — generator + critic LLM cycle (PR 10).

Mocking strategy mirrors ``test_api_deox_ai.py`` (PR 9):
    - The router lazy-imports ``make_hypothesis_generator`` /
      ``make_hypothesis_critic`` inside ``_run_hypothesis_job`` so we
      monkeypatch the source modules. The worker calls the patched
      factories at runtime.
    - We avoid running ``scripts.generate_hypotheses_for_model.build_context``
      against a real trained model by stubbing it with a canned dict.
      Building it would require a real models/<v>/ directory + dataset
      load + XGBoost predict — heavy and not the unit under test here.
    - The Decision Log save test wraps ``decision_log.logger.log_decision``
      to redirect the DB path to ``tmp_path`` so the audit-trail save
      doesn't pollute the project DB.

We use the FastAPI TestClient throughout. Job submission is async on
the server; we poll the JobStore through ``GET /api/jobs/{id}`` until
status flips to done|error.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.jobs import get_job_store
from app.api.main import app
from app.api.routers import system as system_router
from app.backend.hypothesis_critic import CriticVerdict
from app.backend.hypothesis_generator import EconomicImpact, Hypothesis


@pytest.fixture(autouse=True)
def reset_job_store_singleton():
    """Fresh JobStore per test — avoids cross-test pollution.

    PR 6 introduced opportunistic eviction at ``get()`` time; tests that
    submit multiple jobs in sequence still work, but a stale singleton
    from a prior test could surface unexpected state.
    """
    get_job_store.cache_clear()
    yield
    get_job_store.cache_clear()


@pytest.fixture(autouse=True)
def _ensure_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests assume the key is present.

    Tests that need the "no key" branch override this with
    ``monkeypatch.delenv`` inside the body. The fixture sets a sentinel
    value so the endpoint passes the up-front check and the worker
    runs (with the factory monkeypatched separately).
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fixture")


@pytest.fixture()
def fake_models_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Models dir with one fake meta.json that the router treats as valid.

    We only need ``meta.json`` present so ``_safe_version_dir`` +
    ``_read_model_meta`` succeed; the worker's ``build_context`` is
    monkeypatched out, so the real model artifacts never get touched.
    """
    fake_dir = tmp_path / "models"
    fake_dir.mkdir()
    version = "test_hyp_model_v1"
    vdir = fake_dir / version
    vdir.mkdir()
    (vdir / "meta.json").write_text(
        '{"version": "test_hyp_model_v1", "steel_class": "pipe_hsla", '
        '"target": "yield_strength_mpa", "feature_list": ["c_pct"], '
        '"training_ranges": {"c_pct": [0.04, 0.12]}, '
        '"metrics": {"r2_test": 0.85, "mae_test": 12.0, '
        '"rmse_test": 18.0, "coverage_90_ci": 0.91, '
        '"n_train": 80, "n_val": 10, "n_test": 10}, '
        '"feature_importance": {"c_pct": 1.0}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(system_router, "MODELS_DIR", fake_dir)
    return fake_dir


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────


def _stub_hypothesis(idx: str = "h1") -> Hypothesis:
    """Realistic Hypothesis dataclass — same fields as a real generator output."""
    return Hypothesis(
        id=idx,
        statement=f"hypothesis_{idx}: Mn-Si interaction at low C",
        rationale="feature_importance shows mn_pct + si_pct stacking at low c_pct",
        proposed_experiment={
            "fix": {"c_pct": 0.10},
            "sweep": {"variable": "mn_pct", "range": [0.5, 1.5], "step": 0.1},
        },
        expected_outcome="non-monotonic R² lift at mn_pct ~ 0.9",
        novelty="HIGH",
        experiment_cost_estimate="LOW",
        economic_impact=EconomicImpact(
            vs_classical_baseline="vs Pickering 1978 baseline +12%",
            estimated_saving="≈12 €/t",
            measurement_method="industrial trial campaign 5 heats",
        ),
        tags=["steel_class:pipe_hsla", "target:yield_strength_mpa"],
    )


def _stub_verdict(hypothesis_id: str = "h1", verdict: str = "ACCEPT") -> CriticVerdict:
    """Canned CriticVerdict matching a stub hypothesis."""
    return CriticVerdict(
        hypothesis_id=hypothesis_id,
        verdict=verdict,
        confidence="HIGH",
        summary="Mechanism plausible, experiment well-scoped.",
        strengths=["Hall-Petch + solid-solution mechanism is well-known"],
        weaknesses=[],
        suggested_revision=None,
    )


def _patch_llm_factories(
    monkeypatch: pytest.MonkeyPatch,
    hypotheses: list[Hypothesis] | None = None,
    verdicts: list[CriticVerdict] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Replace the generator/critic factories with MagicMock-backed stubs.

    The router lazy-imports the factories inside ``_run_hypothesis_job``
    so we patch the source modules directly. Returns the mocks so tests
    can assert on call counts.
    """
    if hypotheses is None:
        hypotheses = [_stub_hypothesis()]
    if verdicts is None:
        verdicts = [_stub_verdict(h.id) for h in hypotheses]

    gen_mock = MagicMock()
    gen_mock.generate.return_value = hypotheses
    crit_mock = MagicMock()
    crit_mock.review.return_value = verdicts

    monkeypatch.setattr(
        "app.backend.hypothesis_generator.make_hypothesis_generator",
        lambda: gen_mock,
    )
    monkeypatch.setattr(
        "app.backend.hypothesis_critic.make_hypothesis_critic",
        lambda: crit_mock,
    )
    return gen_mock, crit_mock


def _patch_build_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``scripts.generate_hypotheses_for_model.build_context``.

    The real function calls ``model_trainer.load_model`` which loads
    XGBoost JSON + GMM artifacts from ``models/<v>/``. For unit tests we
    don't need any of that; the generator/critic factories are also
    mocked so the context dict is never inspected.
    """
    canned_ctx = {
        "model_version": "test_hyp_model_v1",
        "steel_class": "pipe_hsla",
        "target": "yield_strength_mpa",
        "feature_importance": {"c_pct": 0.4, "mn_pct": 0.3},
        "training_ranges": {"c_pct": [0.04, 0.12]},
        "r2_test": 0.85,
        "mae_test": 12.0,
        "rmse_test": 18.0,
        "coverage_90_ci": 0.91,
        "n_train": 80, "n_val": 10, "n_test": 10,
    }
    monkeypatch.setattr(
        "scripts.generate_hypotheses_for_model.build_context",
        lambda model_version: dict(canned_ctx, model_version=model_version),
    )


def _wait_for_job(
    client: TestClient,
    job_id: str,
    timeout: float = 15.0,
) -> dict:
    """Poll /api/jobs/{id} until status flips to done|error.

    Larger timeout (15 s) than test_api_deox_ai because the worker
    serialises through the JobStore thread pool — under heavy CI the
    initial RUNNING transition can take a moment.
    """
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
        "model_version": "test_hyp_model_v1",
        "n_hypotheses": 5,
        "save_to_decision_log": False,
    }
    body.update(overrides)
    return body


# ──────────────────────────────────────────────────────────────────────
# submission / validation
# ──────────────────────────────────────────────────────────────────────


def test_hypotheses_run_returns_job_id(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid body → 200 + ``{job_id, config}``."""
    _patch_llm_factories(monkeypatch)
    _patch_build_context(monkeypatch)
    resp = client.post("/api/hypotheses/run", json=_baseline_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body.get("job_id"), str) and len(body["job_id"]) > 0
    assert body["config"]["model_version"] == "test_hyp_model_v1"
    assert body["config"]["n_hypotheses"] == 5
    assert body["config"]["save_to_decision_log"] is False


def test_hypotheses_validation_422(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out-of-bounds n_hypotheses → 422 from Pydantic Field validator."""
    _patch_llm_factories(monkeypatch)
    _patch_build_context(monkeypatch)
    # Below ge=1
    resp_low = client.post(
        "/api/hypotheses/run", json=_baseline_payload(n_hypotheses=0)
    )
    assert resp_low.status_code == 422
    # Above le=20
    resp_high = client.post(
        "/api/hypotheses/run", json=_baseline_payload(n_hypotheses=999)
    )
    assert resp_high.status_code == 422


def test_hypotheses_path_traversal_400(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """model_version with path-traversal markers → 400 (CWE-22).

    The slug regex in ``_safe_version_dir`` rejects ``../app`` because
    ``/`` is not in the allowed character set; ``Path("models") / "../app"``
    would otherwise resolve outside MODELS_DIR.
    """
    _patch_llm_factories(monkeypatch)
    _patch_build_context(monkeypatch)
    resp = client.post(
        "/api/hypotheses/run", json=_baseline_payload(model_version="../app")
    )
    assert resp.status_code == 400


def test_hypotheses_no_api_key_503(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ANTHROPIC_API_KEY → 503 with actionable message."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = client.post("/api/hypotheses/run", json=_baseline_payload())
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "ANTHROPIC_API_KEY" in detail


def test_hypotheses_unknown_model_404(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid slug but model dir is missing → 404 (router-layer guard)."""
    _patch_llm_factories(monkeypatch)
    _patch_build_context(monkeypatch)
    resp = client.post(
        "/api/hypotheses/run",
        json=_baseline_payload(model_version="nonexistent_model_v9"),
    )
    assert resp.status_code == 404
    assert "nonexistent_model_v9" in resp.json()["detail"]


def test_hypotheses_missing_prompt_503(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a prompt file is missing on disk → 503 with name in the message."""
    _patch_llm_factories(monkeypatch)
    _patch_build_context(monkeypatch)
    from app.backend.prompt_loader import PromptNotFoundError

    def _raise(name: str) -> str:
        raise PromptNotFoundError(f"missing {name}")

    # The shared ``check_llm_ready`` in ``app.api.llm_gating`` imports
    # ``load_prompt`` lazily inside the function, so patching the source
    # symbol catches the call.
    monkeypatch.setattr("app.backend.prompt_loader.load_prompt", _raise)
    resp = client.post("/api/hypotheses/run", json=_baseline_payload())
    assert resp.status_code == 503
    assert "hypothesis_generator" in resp.json()["detail"]


# ──────────────────────────────────────────────────────────────────────
# end-to-end with mocked LLM
# ──────────────────────────────────────────────────────────────────────


def test_hypotheses_to_completion_with_mock_llm(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit + poll + verify result shape with stub generator/critic."""
    hyps = [_stub_hypothesis("h1"), _stub_hypothesis("h2")]
    verdicts = [
        _stub_verdict("h1", verdict="ACCEPT"),
        _stub_verdict("h2", verdict="REVISE"),
    ]
    gen_mock, crit_mock = _patch_llm_factories(monkeypatch, hyps, verdicts)
    _patch_build_context(monkeypatch)

    resp = client.post("/api/hypotheses/run", json=_baseline_payload())
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    body = _wait_for_job(client, job_id, timeout=10.0)
    if body["status"] != "done":
        pytest.fail(
            f"job ended with status={body['status']} error={body.get('error')!r}"
        )

    result = body["result"]

    # Result shape — hypotheses + reviews + counts + duration.
    assert result["model_version"] == "test_hyp_model_v1"
    assert isinstance(result["hypotheses"], list)
    assert len(result["hypotheses"]) == 2
    assert isinstance(result["reviews"], list)
    assert len(result["reviews"]) == 2
    assert result["verdict_counts"] == {"ACCEPT": 1, "REVISE": 1, "REJECT": 0}
    assert isinstance(result["duration_s"], (int, float))
    assert result["decision_log_id"] is None  # opt-in, was False

    # Hypothesis fields preserved after asdict.
    h0 = result["hypotheses"][0]
    assert h0["id"] == "h1"
    assert h0["novelty"] == "HIGH"
    assert h0["experiment_cost_estimate"] == "LOW"
    assert h0["economic_impact"]["estimated_saving"] == "≈12 €/t"

    # Review fields preserved.
    r0 = result["reviews"][0]
    assert r0["hypothesis_id"] == "h1"
    assert r0["verdict"] == "ACCEPT"
    assert r0["confidence"] == "HIGH"

    # Both factories must have been invoked exactly once.
    gen_mock.generate.assert_called_once()
    crit_mock.review.assert_called_once()
    # Critic must receive the asdict-projection of the hypotheses.
    critic_call_args = crit_mock.review.call_args
    _ctx_arg, hyp_arg = critic_call_args.args
    assert isinstance(hyp_arg, list) and hyp_arg[0]["id"] == "h1"


def test_hypotheses_decision_log_save(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """save_to_decision_log=True → entry is written with tag hypothesis_cycle.

    Same wrapping technique as ``test_api_deox_ai.py`` — capture the
    invocation + redirect ``db_path`` to a tmp DB so the production
    schema (CREATE INDEX, AUTOINCREMENT) is exercised verbatim.
    """
    _patch_llm_factories(monkeypatch)
    _patch_build_context(monkeypatch)

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
        "/api/hypotheses/run",
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
        if "hypothesis_cycle" in (c["kwargs"].get("tags") or [])
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
    assert phase == "training"
    assert "Hypothesis cycle" in decision
    assert "hypothesis_cycle" in tags_json
    assert author == "api"


def test_hypotheses_safe_json_numeric(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Numeric values come back as JSON numbers, not strings.

    Locks in the SafeJSONResponse contract — duration_s must be a real
    number so the JS view can use ``Number(...)`` directly.
    """
    _patch_llm_factories(monkeypatch)
    _patch_build_context(monkeypatch)
    resp = client.post("/api/hypotheses/run", json=_baseline_payload())
    job_id = resp.json()["job_id"]
    body = _wait_for_job(client, job_id, timeout=10.0)
    assert body["status"] == "done"
    result = body["result"]

    assert isinstance(result["duration_s"], (int, float))
    # Verdict counts should be ints (not strings, not floats).
    for k in ("ACCEPT", "REVISE", "REJECT"):
        assert isinstance(result["verdict_counts"][k], int)


def test_hypotheses_cancel_before_llm(
    fake_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DELETE before the worker finishes → cooperative cancel raises.

    We slow the generator factory by parking inside it, set the
    cancellation flag from the client side, then release the factory
    so the worker advances. The cancellation gate between generator
    and critic catches the flag and raises
    ``Cancelled by user (after generator, before critic)``.

    Note the same race-window caveat as PR 9 ``test_ai_cycle_cancel_before_llm``:
    if the worker is already past both gates by the time DELETE arrives,
    the job completes normally. We treat both outcomes as acceptable —
    the test guarantees the cancellation API works, not that it always
    wins the race.
    """
    import threading

    started_event = threading.Event()
    proceed_event = threading.Event()

    def slow_factory():
        # Park here until the test releases. The cancellation flag set
        # while we wait is read by the gate after the generator returns.
        started_event.set()
        proceed_event.wait(timeout=5.0)
        m = MagicMock()
        m.generate.return_value = [_stub_hypothesis()]
        return m

    crit_mock = MagicMock()
    crit_mock.review.return_value = [_stub_verdict()]

    monkeypatch.setattr(
        "app.backend.hypothesis_generator.make_hypothesis_generator",
        slow_factory,
    )
    monkeypatch.setattr(
        "app.backend.hypothesis_critic.make_hypothesis_critic",
        lambda: crit_mock,
    )
    _patch_build_context(monkeypatch)

    resp = client.post("/api/hypotheses/run", json=_baseline_payload())
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
