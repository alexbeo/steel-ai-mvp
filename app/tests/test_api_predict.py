"""Tests for /api/system/* + /api/predict + /api/predict/anomaly-explain.

PR 3 introduced the prediction endpoint with a placeholder OOD warning
button; PR 12 wires the placeholder up to a real Sonnet-backed anomaly
explainer running as a long-running job + adds the wide-CI heuristic.

Test isolation strategy:
    - We avoid spinning up Optuna training in tests (5 min budget). Instead
      we copy ONE pre-existing model directory from the repo's ``models/``
      into a temp dir and monkeypatch both ``app.api.routers.system.MODELS_DIR``
      and ``app.backend.model_trainer.MODELS_DIR`` to point there.
    - For the empty-dir test we just don't copy anything.
    - For the legacy-fallback test we strip ``steel_class`` from a copied
      meta.json before letting the endpoint read it.
    - For predict tests we send the full feature_set with mid-range values
      drawn from the class profile's physical_bounds, mirroring how the
      JS form builds default values.
    - For PR 12 anomaly-explain tests we monkeypatch the
      ``app.backend.anomaly_explainer.make_anomaly_explainer`` factory
      with a MagicMock-backed stub so the LLM never gets called. The
      router lazy-imports the factory inside the worker — patching the
      source module catches the call.

    This pattern keeps tests fast (~1 sec total) and exercises the real
    XGBoost prediction path. Down side: tests skip cleanly when the repo
    has zero pre-trained HSLA models — that case is covered by smoke_test.py
    in CI but should rarely happen locally.
"""
from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.jobs import get_job_store
from app.api.main import app
from app.api.routers import system as system_router
from app.backend import model_trainer
from app.backend.anomaly_explainer import AnomalousFeature, AnomalyExplanation
from app.backend.steel_classes import load_steel_class

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_MODELS_DIR = PROJECT_ROOT / "models"

# Required artifact filenames for a complete HSLA model directory.
# The OOD detector file is a serialised GMM produced by model_trainer.
_REQUIRED_MODEL_ARTIFACTS = frozenset(
    {"main.json", "q05.json", "q95.json", "ood_detector.pkl"}
)


def _find_real_hsla_model() -> Path | None:
    """Locate any complete HSLA model dir in the repo.

    Returns the first directory whose meta.json declares
    ``steel_class == 'pipe_hsla'`` AND has the four artifact files.
    """
    if not REAL_MODELS_DIR.is_dir():
        return None
    for entry in sorted(REAL_MODELS_DIR.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("steel_class") != "pipe_hsla":
            continue
        if not _REQUIRED_MODEL_ARTIFACTS.issubset(
            {p.name for p in entry.iterdir()}
        ):
            continue
        return entry
    return None


@pytest.fixture()
def empty_models_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Empty MODELS_DIR — for endpoint behaviour tests that don't need a model."""
    fake_dir = tmp_path / "models"
    fake_dir.mkdir()
    monkeypatch.setattr(system_router, "MODELS_DIR", fake_dir)
    monkeypatch.setattr(model_trainer, "MODELS_DIR", fake_dir)
    yield fake_dir


@pytest.fixture()
def populated_models_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """MODELS_DIR with one real HSLA model copied in for predict tests.

    Skips the test if the repo has no HSLA models — caller decides what
    that means (predict-flow tests can't run, but they're not the only
    coverage for the endpoint).
    """
    src = _find_real_hsla_model()
    if src is None:
        pytest.skip("No real HSLA model available in repo to drive predict tests")
    fake_dir = tmp_path / "models"
    fake_dir.mkdir()
    dst = fake_dir / src.name
    shutil.copytree(src, dst)
    monkeypatch.setattr(system_router, "MODELS_DIR", fake_dir)
    monkeypatch.setattr(model_trainer, "MODELS_DIR", fake_dir)
    yield fake_dir


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _midrange_composition(class_id: str) -> dict[str, float]:
    """Build a composition dict of midpoint values from physical_bounds.

    Each feature defaults to ``(lo + hi) / 2`` — same convention the JS
    form uses to seed inputs.
    """
    profile = load_steel_class(class_id)
    return {
        feat: (lo + hi) / 2.0
        for feat, (lo, hi) in profile.physical_bounds.items()
        if feat in profile.feature_set
    }


# ---------- /api/system/steel-classes -------------------------------------


def test_list_steel_classes(client: TestClient) -> None:
    """Endpoint returns the three registered profiles with feature_set."""
    resp = client.get("/api/system/steel-classes")
    assert resp.status_code == 200
    payload = resp.json()
    assert "items" in payload
    assert payload["count"] == len(payload["items"])

    ids = {item["id"] for item in payload["items"]}
    assert ids == {"pipe_hsla", "en10083_qt", "fatigue_carbon_steel"}

    # Shape sanity for one profile.
    hsla = next(item for item in payload["items"] if item["id"] == "pipe_hsla")
    assert isinstance(hsla["feature_set"], list)
    assert "c_pct" in hsla["feature_set"]
    assert isinstance(hsla["physical_bounds"], dict)
    assert isinstance(hsla["target_properties"], list)
    yt = next(t for t in hsla["target_properties"] if t["id"] == "yield_strength_mpa")
    assert isinstance(yt["range"], list) and len(yt["range"]) == 2


# ---------- /api/system/models --------------------------------------------


def test_list_models_empty(empty_models_dir: Path, client: TestClient) -> None:
    resp = client.get("/api/system/models")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["items"] == []
    assert payload["count"] == 0


def test_list_models_returns_meta(empty_models_dir: Path, client: TestClient) -> None:
    """Hand-craft three fake meta.json — exercise the metadata-shape mapping
    without touching XGBoost. We don't need to read real models here, only
    confirm meta keys make it into the response."""
    for i, version in enumerate(["aaa_old", "bbb_mid", "ccc_new"]):
        d = empty_models_dir / version
        d.mkdir()
        (d / "meta.json").write_text(
            json.dumps(
                {
                    "version": version,
                    "target": "yield_strength_mpa",
                    "feature_list": ["c_pct", "mn_pct"],
                    "training_ranges": {"c_pct": [0.04, 0.12]},
                    "metrics": {
                        "r2_test": 0.85 + i * 0.01,
                        "mae_test": 12.0,
                        "rmse_test": 18.0,
                        "coverage_90_ci": 0.91,
                        "n_train": 80,
                        "n_val": 10,
                        "n_test": 10,
                    },
                    "steel_class": "pipe_hsla",
                    "trained_at": f"2026-05-09T1{i}:00:00",
                    "conformal_correction_mpa": 5.0,
                }
            ),
            encoding="utf-8",
        )

    resp = client.get("/api/system/models")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 3
    versions = [item["version"] for item in payload["items"]]
    # sorted ascending by directory name
    assert versions == ["aaa_old", "bbb_mid", "ccc_new"]
    first = payload["items"][0]
    assert first["steel_class"] == "pipe_hsla"
    assert first["target"] == "yield_strength_mpa"
    assert first["metrics"]["r2_test"] == pytest.approx(0.85)
    assert first["feature_list"] == ["c_pct", "mn_pct"]
    # has_ood_detector is False because we didn't write the artifact file.
    assert first["has_ood_detector"] is False


def test_active_model_is_latest(empty_models_dir: Path, client: TestClient) -> None:
    """Active model = last entry of sorted dirs."""
    for version in ["aaa_old", "ccc_latest", "bbb_mid"]:
        d = empty_models_dir / version
        d.mkdir()
        (d / "meta.json").write_text(
            json.dumps(
                {
                    "version": version,
                    "steel_class": "pipe_hsla",
                    "target": "yield_strength_mpa",
                    "feature_list": [],
                    "training_ranges": {},
                    "metrics": {},
                }
            ),
            encoding="utf-8",
        )

    resp = client.get("/api/system/models/active")
    assert resp.status_code == 200
    assert resp.json()["version"] == "ccc_latest"


def test_active_model_fallback_pipe_hsla(
    empty_models_dir: Path, client: TestClient
) -> None:
    """Legacy meta.json without ``steel_class`` falls back to pipe_hsla."""
    d = empty_models_dir / "legacy_v1"
    d.mkdir()
    (d / "meta.json").write_text(
        json.dumps(
            {
                "version": "legacy_v1",
                "target": "yield_strength_mpa",
                "feature_list": [],
                "training_ranges": {},
                "metrics": {},
                # NOTE: no ``steel_class`` key
            }
        ),
        encoding="utf-8",
    )

    resp = client.get("/api/system/models/active")
    assert resp.status_code == 200
    assert resp.json()["steel_class"] == "pipe_hsla"


def test_active_model_404_when_empty(
    empty_models_dir: Path, client: TestClient
) -> None:
    resp = client.get("/api/system/models/active")
    assert resp.status_code == 404
    assert "обуч" in resp.json()["detail"].lower()


# ---------- /api/predict --------------------------------------------------


def test_predict_happy_path(populated_models_dir: Path, client: TestClient) -> None:
    """Full prediction roundtrip on a real HSLA model."""
    version = next(populated_models_dir.iterdir()).name
    composition = _midrange_composition("pipe_hsla")

    resp = client.post(
        "/api/predict",
        json={"model_version": version, "composition": composition},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    # Top-level shape contract.
    assert set(payload.keys()) == {"prediction", "ood", "derived", "model"}

    pred = payload["prediction"]
    assert isinstance(pred["mean"], (int, float))
    assert isinstance(pred["q05"], (int, float))
    assert isinstance(pred["q95"], (int, float))
    assert pred["q05"] <= pred["mean"] <= pred["q95"]
    assert pred["ci_half_width"] == pytest.approx(
        (pred["q95"] - pred["q05"]) / 2.0, abs=1e-6
    )
    assert pred["target_property"] == "yield_strength_mpa"
    assert pred["target_label"]

    # OOD section.
    assert isinstance(payload["ood"]["is_ood"], bool)
    assert isinstance(payload["ood"]["log_density"], (int, float))

    # Derived HSLA features (cev_iiw / pcm / cen / microalloying_sum).
    assert "cev_iiw" in payload["derived"]
    assert "pcm" in payload["derived"]

    # Model echo.
    assert payload["model"]["version"] == version
    assert payload["model"]["steel_class"] == "pipe_hsla"


def test_predict_missing_feature_returns_400(
    populated_models_dir: Path, client: TestClient
) -> None:
    """Composition that doesn't cover feature_set must 400 with a clear
    list of missing keys (callable from the UI's error banner)."""
    version = next(populated_models_dir.iterdir()).name
    full = _midrange_composition("pipe_hsla")
    # Drop two keys.
    full.pop("c_pct")
    full.pop("mn_pct")

    resp = client.post(
        "/api/predict",
        json={"model_version": version, "composition": full},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "c_pct" in detail and "mn_pct" in detail


def test_predict_extra_keys_ignored(
    populated_models_dir: Path, client: TestClient
) -> None:
    """Extra keys must not break the prediction (UI may pre-compute
    derived features that the backend recomputes anyway)."""
    version = next(populated_models_dir.iterdir()).name
    composition = _midrange_composition("pipe_hsla")
    composition["bogus_key"] = 999.0
    composition["cev_iiw"] = 0.42  # would be recomputed inside

    resp = client.post(
        "/api/predict",
        json={"model_version": version, "composition": composition},
    )
    assert resp.status_code == 200, resp.text


def test_predict_unknown_model_404(
    populated_models_dir: Path, client: TestClient
) -> None:
    resp = client.post(
        "/api/predict",
        json={
            "model_version": "no_such_model_v0",
            "composition": _midrange_composition("pipe_hsla"),
        },
    )
    assert resp.status_code == 404
    assert "no_such_model_v0" in resp.json()["detail"]


def test_predict_rejects_path_traversal(
    populated_models_dir: Path, client: TestClient
) -> None:
    """CWE-22 regression: ``../app`` must not let an attacker load XGBoost
    artifacts from a sibling directory. Even an empty composition is enough
    to trigger the guard, which fires before any feature validation."""
    resp = client.post(
        "/api/predict",
        json={"model_version": "../app", "composition": {}},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"].lower()
    assert "model_version" in detail or "invalid" in detail


def test_predict_rejects_absolute_path(
    populated_models_dir: Path, client: TestClient
) -> None:
    """CWE-22 regression: an absolute path silently overrides ``MODELS_DIR``
    in ``Path``-concat semantics, so the regex guard must reject it before
    we ever touch the filesystem."""
    resp = client.post(
        "/api/predict",
        json={"model_version": "/etc/passwd", "composition": {}},
    )
    assert resp.status_code == 400


def test_predict_safe_json_numeric(
    populated_models_dir: Path, client: TestClient
) -> None:
    """Regression: SafeJSONResponse must emit numbers, not strings, for
    numpy scalars produced by predict_with_uncertainty."""
    version = next(populated_models_dir.iterdir()).name
    composition = _midrange_composition("pipe_hsla")

    resp = client.post(
        "/api/predict",
        json={"model_version": version, "composition": composition},
    )
    assert resp.status_code == 200
    payload = resp.json()
    # Sanity: every numeric leaf is a real number, not a string.
    for key in ("mean", "q05", "q95", "ci_half_width"):
        assert isinstance(payload["prediction"][key], (int, float)), (
            f"{key} must be numeric, got {type(payload['prediction'][key])}"
        )
    assert isinstance(payload["ood"]["log_density"], (int, float))
    for k, v in payload["derived"].items():
        assert isinstance(v, (int, float)), f"derived[{k}] must be numeric"


# ---------- PR 12: wide_ci flag + deprecated alias regression ------------


def test_predict_response_includes_wide_ci_flag(
    populated_models_dir: Path, client: TestClient
) -> None:
    """PR 12: prediction.wide_ci surfaces the heuristic the UI uses to
    decide whether to show the anomaly-explain button on non-OOD points.

    Threshold details are an implementation detail — we only assert the
    *shape*: a boolean flag + a numeric threshold echoed back. Concrete
    threshold values are unit-tested at the helper level if needed.
    """
    version = next(populated_models_dir.iterdir()).name
    composition = _midrange_composition("pipe_hsla")

    resp = client.post(
        "/api/predict",
        json={"model_version": version, "composition": composition},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert "wide_ci" in payload["prediction"]
    assert isinstance(payload["prediction"]["wide_ci"], bool)
    assert "wide_ci_threshold" in payload["prediction"]
    assert isinstance(payload["prediction"]["wide_ci_threshold"], (int, float))


def test_predict_response_lower_p_deprecated_still_present(
    populated_models_dir: Path, client: TestClient
) -> None:
    """Regression: PR 12 deprecation must not break the lower_p/upper_p
    aliases — the PR 3 frontend still depends on them. They're scheduled
    for removal once views/predict.js cuts over to q05/q95 exclusively
    (tracked in design-doc backlog) — until then they MUST equal q05/q95.
    """
    version = next(populated_models_dir.iterdir()).name
    composition = _midrange_composition("pipe_hsla")

    resp = client.post(
        "/api/predict",
        json={"model_version": version, "composition": composition},
    )
    assert resp.status_code == 200
    pred = resp.json()["prediction"]
    assert "lower_p" in pred and "upper_p" in pred
    assert pred["lower_p"] == pytest.approx(pred["q05"])
    assert pred["upper_p"] == pytest.approx(pred["q95"])


# ---------- PR 12: /api/predict/anomaly-explain --------------------------


@pytest.fixture(autouse=False)
def _ensure_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that target the LLM-backed anomaly endpoint assume the key
    is present at the gating-check step. The factory is monkeypatched
    separately so no Anthropic SDK call ever fires.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anomaly")


@pytest.fixture()
def reset_job_store_singleton() -> Iterator[None]:
    """Fresh JobStore per test — avoids cross-test pollution under the
    opportunistic eviction policy.
    """
    get_job_store.cache_clear()
    yield
    get_job_store.cache_clear()


def _stub_explanation(severity: str = "MEDIUM") -> AnomalyExplanation:
    """Realistic AnomalyExplanation dataclass — same fields a real
    AnomalyExplainer.explain call would emit.
    """
    return AnomalyExplanation(
        id="ax01",
        summary="Mn выше верхней границы training; risk Hall-Petch shift.",
        anomalous_features=[
            AnomalousFeature(
                feature="mn_pct",
                value=2.10,
                training_range=[0.4, 1.8],
                deviation_kind="out_of_range_high",
                note="2.10 wt% Mn — вне training [0.4, 1.8].",
            ),
        ],
        mechanism_concerns=[
            "Hall-Petch grain refinement не проверена для Mn>1.8%",
            "Возможны MnS inclusions при S≥0.005%",
        ],
        production_risks="Hot shortness риск при rolling >1100 °C.",
        suggested_correction="Снизить Mn к 1.6-1.7 wt%, скомпенсировать Si +0.1%.",
        severity=severity,
        tags=["steel_class:pipe_hsla"],
    )


def _patch_anomaly_factory(
    monkeypatch: pytest.MonkeyPatch,
    explanation: AnomalyExplanation | None = None,
) -> MagicMock:
    """Replace the factory with a MagicMock returning a canned explanation."""
    if explanation is None:
        explanation = _stub_explanation()
    explainer_mock = MagicMock()
    explainer_mock.explain.return_value = explanation
    monkeypatch.setattr(
        "app.backend.anomaly_explainer.make_anomaly_explainer",
        lambda: explainer_mock,
    )
    return explainer_mock


def _wait_for_job(
    client: TestClient, job_id: str, timeout: float = 15.0
) -> dict:
    """Poll /api/jobs/{id} until terminal — same shape as test_api_hypotheses."""
    deadline = time.monotonic() + timeout
    body: dict = {}
    while time.monotonic() < deadline:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    pytest.fail(f"job {job_id} did not finish within {timeout}s; last={body!r}")


def _baseline_anomaly_payload(
    *, version: str, composition: dict[str, float], **overrides: object
) -> dict:
    body: dict = {
        "model_version": version,
        "composition": composition,
        "predicted_mean": 540.0,
        "predicted_q05": 480.0,
        "predicted_q95": 620.0,
        "is_ood": True,
        "save_to_decision_log": False,
    }
    body.update(overrides)
    return body


def test_anomaly_explain_returns_job_id(
    populated_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    _ensure_api_key: None,
    reset_job_store_singleton: None,
) -> None:
    """Valid body → 200 + ``{job_id, config}``."""
    _patch_anomaly_factory(monkeypatch)
    version = next(populated_models_dir.iterdir()).name
    composition = _midrange_composition("pipe_hsla")

    resp = client.post(
        "/api/predict/anomaly-explain",
        json=_baseline_anomaly_payload(version=version, composition=composition),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body.get("job_id"), str) and len(body["job_id"]) > 0
    assert body["config"]["model_version"] == version
    assert body["config"]["is_ood"] is True
    assert body["config"]["save_to_decision_log"] is False

    # Drain the job — the worker runs on the JobStore thread pool which
    # persists across tests. If we don't wait here, the next test's
    # ``explainer_mock.assert_called_once`` may catch *this* job's call
    # because the monkeypatched factory is shared across tests until the
    # subsequent test re-patches it (and the original call still resolves
    # against the original mock on the worker thread).
    _wait_for_job(client, body["job_id"], timeout=10.0)


def test_anomaly_explain_validation_422(
    populated_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    _ensure_api_key: None,
) -> None:
    """Missing required fields → 422 from Pydantic.

    We test the most common malformed-body case: composition omitted
    entirely. The endpoint should reject before path validation /
    LLM gating fire.
    """
    _patch_anomaly_factory(monkeypatch)
    version = next(populated_models_dir.iterdir()).name
    resp = client.post(
        "/api/predict/anomaly-explain",
        # Missing composition + predicted_* fields.
        json={"model_version": version, "is_ood": True},
    )
    assert resp.status_code == 422


def test_anomaly_explain_path_traversal_400(
    populated_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    _ensure_api_key: None,
) -> None:
    """CWE-22 regression: ``../app`` rejected by ``_safe_version_dir``."""
    _patch_anomaly_factory(monkeypatch)
    composition = _midrange_composition("pipe_hsla")
    resp = client.post(
        "/api/predict/anomaly-explain",
        json=_baseline_anomaly_payload(version="../app", composition=composition),
    )
    assert resp.status_code == 400


def test_anomaly_explain_no_api_key_503(
    populated_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ANTHROPIC_API_KEY → 503 with actionable Russian message."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _patch_anomaly_factory(monkeypatch)
    version = next(populated_models_dir.iterdir()).name
    composition = _midrange_composition("pipe_hsla")
    resp = client.post(
        "/api/predict/anomaly-explain",
        json=_baseline_anomaly_payload(version=version, composition=composition),
    )
    assert resp.status_code == 503
    assert "ANTHROPIC_API_KEY" in resp.json()["detail"]


def test_anomaly_explain_unknown_model_404(
    populated_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    _ensure_api_key: None,
) -> None:
    """Valid slug but model dir is missing → 404."""
    _patch_anomaly_factory(monkeypatch)
    composition = _midrange_composition("pipe_hsla")
    resp = client.post(
        "/api/predict/anomaly-explain",
        json=_baseline_anomaly_payload(
            version="no_such_model_v0", composition=composition
        ),
    )
    assert resp.status_code == 404


def test_anomaly_explain_to_completion_with_mock_llm(
    populated_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    _ensure_api_key: None,
    reset_job_store_singleton: None,
) -> None:
    """Submit + poll + verify result shape with a stub explainer."""
    explainer_mock = _patch_anomaly_factory(monkeypatch)
    version = next(populated_models_dir.iterdir()).name
    composition = _midrange_composition("pipe_hsla")

    resp = client.post(
        "/api/predict/anomaly-explain",
        json=_baseline_anomaly_payload(version=version, composition=composition),
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    body = _wait_for_job(client, job_id, timeout=10.0)
    if body["status"] != "done":
        pytest.fail(
            f"job ended with status={body['status']} error={body.get('error')!r}"
        )

    result = body["result"]

    # Result shape: explanation dict + evidence list + duration.
    assert result["model_version"] == version
    assert isinstance(result["explanation"], dict)
    assert result["explanation"]["severity"] == "MEDIUM"
    assert result["explanation"]["summary"]
    # Anomalous features come back as plain dicts after asdict.
    assert isinstance(result["explanation"]["anomalous_features"], list)
    af0 = result["explanation"]["anomalous_features"][0]
    assert af0["feature"] == "mn_pct"
    # Evidence list is the worker's *own* objective out-of-range list,
    # independent of what the LLM said. Type-check only — composition
    # is mid-range so the list may legitimately be empty.
    assert isinstance(result["evidence"], list)
    assert isinstance(result["duration_s"], (int, float))
    assert result["decision_log_id"] is None  # opt-in, was False

    # Factory was invoked exactly once.
    explainer_mock.explain.assert_called_once()
    # Context dict passed to the explainer must include the canonical
    # fields the AnomalyExplainer prompt depends on.
    ctx_arg = explainer_mock.explain.call_args.args[0]
    assert ctx_arg["steel_class"] == "pipe_hsla"
    assert "training_ranges" in ctx_arg
    assert "ml_prediction" in ctx_arg
    assert ctx_arg["ood_flag"] is True


def test_anomaly_explain_decision_log_save(
    populated_models_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _ensure_api_key: None,
    reset_job_store_singleton: None,
) -> None:
    """save_to_decision_log=True → entry written with tag 'anomaly_cycle'.

    Same wrapping technique as test_api_hypotheses — capture the
    invocation + redirect ``db_path`` to a tmp DB so the production
    schema is exercised verbatim.
    """
    _patch_anomaly_factory(monkeypatch)

    tmp_db = tmp_path / "decisions.db"
    import decision_log.logger as log_mod

    captured: list[dict] = []
    real_log_decision = log_mod.log_decision

    def _wrapped(*args: object, **kwargs: object) -> int:
        kwargs["db_path"] = tmp_db
        captured.append({"args": args, "kwargs": kwargs})
        return real_log_decision(*args, **kwargs)

    monkeypatch.setattr(log_mod, "log_decision", _wrapped)

    version = next(populated_models_dir.iterdir()).name
    composition = _midrange_composition("pipe_hsla")

    resp = client.post(
        "/api/predict/anomaly-explain",
        json=_baseline_anomaly_payload(
            version=version,
            composition=composition,
            save_to_decision_log=True,
        ),
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
        if "anomaly_cycle" in (c["kwargs"].get("tags") or [])
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
    assert phase == "validation"
    assert "Anomaly explanation" in decision
    assert "anomaly_cycle" in tags_json
    assert author == "api"
