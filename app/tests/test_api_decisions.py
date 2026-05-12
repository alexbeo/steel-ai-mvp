"""Tests for /api/decisions — Decision Log read API (PR 2 of FastAPI migration).

Test isolation strategy:
    `decision_log.logger.{query_decisions,get_decision_by_id,
    summarize_project_history}` use ``DEFAULT_DB_PATH`` as a default-argument
    value, which is bound at function-definition time. Monkeypatching the
    module-level constant therefore does NOT redirect default callers.

    Instead we monkeypatch the *names already imported into the router
    module* (``app.api.routers.decisions``) with thin wrappers that pin a
    temp DB path. This keeps ``decision_log/logger.py`` untouched (R-001
    constraint) and avoids a server restart.
"""
from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routers import decisions as decisions_router
from decision_log.logger import (
    get_decision_by_id as _real_get_by_id,
    log_decision,
    query_decisions as _real_query,
    summarize_project_history as _real_summary,
)


@pytest.fixture()
def temp_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Spin up an empty Decision Log DB and route all router calls to it."""
    tmp = Path(tempfile.mkdtemp()) / "test_decisions.db"

    def query_with_temp(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("db_path", tmp)
        return _real_query(*args, **kwargs)

    def get_with_temp(decision_id: int, **kwargs: object) -> object:
        kwargs.setdefault("db_path", tmp)
        return _real_get_by_id(decision_id, **kwargs)

    def summary_with_temp(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("db_path", tmp)
        return _real_summary(*args, **kwargs)

    monkeypatch.setattr(decisions_router, "query_decisions", query_with_temp)
    monkeypatch.setattr(decisions_router, "get_decision_by_id", get_with_temp)
    monkeypatch.setattr(
        decisions_router, "summarize_project_history", summary_with_temp,
    )
    yield tmp
    # Cleanup — best effort; tempdir auto-removed by OS, but unlink for hygiene.
    try:
        if tmp.exists():
            tmp.unlink()
        tmp.parent.rmdir()
    except OSError:
        pass


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# ---------- helpers ----------


def _seed_three(db: Path) -> dict[str, int]:
    """Insert three records covering different phases / tags / outcomes."""
    id_train = log_decision(
        phase="training",
        decision="Use XGBoost with max_depth=5",
        reasoning="Optuna search showed stable R²=0.87 at depth=5.",
        alternatives_considered=["depth=3", "depth=10"],
        context={"trials": 100, "best_r2": 0.87},
        author="model_trainer",
        tags=["xgboost", "hyperparameters"],
        db_path=db,
    )
    id_inverse = log_decision(
        phase="inverse_design",
        decision="NSGA-II population_size=80",
        reasoning="Pareto front converges by gen 30 with pop=80.",
        alternatives_considered=["pop=40", "pop=160"],
        context={"pareto_size": 12},
        author="orchestrator",
        tags=["nsga2"],
        db_path=db,
    )
    id_meta = log_decision(
        phase="meta",
        decision="Switch primary class default to pipe_hsla",
        reasoning="Most demos run on HSLA — make it the implicit default.",
        alternatives_considered=[],
        context={},
        author="user",
        tags=["ux"],
        db_path=db,
    )
    return {"training": id_train, "inverse_design": id_inverse, "meta": id_meta}


# ---------- list endpoint ----------


def test_list_decisions_empty(temp_db: Path, client: TestClient) -> None:
    resp = client.get("/api/decisions")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["items"] == []
    assert payload["limit"] == 20
    assert payload["offset"] == 0
    assert payload["count"] == 0


def test_list_decisions_returns_all(temp_db: Path, client: TestClient) -> None:
    _seed_three(temp_db)
    resp = client.get("/api/decisions")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 3
    assert len(payload["items"]) == 3
    # Items returned newest-first (logger ORDER BY timestamp DESC).
    decisions_seen = {row["decision"] for row in payload["items"]}
    assert decisions_seen == {
        "Use XGBoost with max_depth=5",
        "NSGA-II population_size=80",
        "Switch primary class default to pipe_hsla",
    }
    # Shape sanity: alternatives_considered, context, tags are deserialised.
    one = payload["items"][0]
    assert isinstance(one["alternatives_considered"], list)
    assert isinstance(one["context"], dict)
    assert isinstance(one["tags"], list)


def test_list_decisions_filters_by_phase(temp_db: Path, client: TestClient) -> None:
    _seed_three(temp_db)
    resp = client.get("/api/decisions", params={"phase": "training"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 1
    assert payload["items"][0]["phase"] == "training"
    assert payload["items"][0]["decision"] == "Use XGBoost with max_depth=5"


def test_list_decisions_filters_by_tag(temp_db: Path, client: TestClient) -> None:
    _seed_three(temp_db)
    resp = client.get("/api/decisions", params={"tag": "xgboost"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 1
    assert "xgboost" in payload["items"][0]["tags"]


def test_list_decisions_filters_by_keyword(temp_db: Path, client: TestClient) -> None:
    """`?keyword=` matches the substring against decision OR reasoning, per
    ``decision_log.logger.query_decisions`` semantics. Smoke-level: PR 3-12
    UI search box will rely on this — guard against silent breakage."""
    _seed_three(temp_db)

    # "XGBoost" appears only in the training decision text.
    resp = client.get("/api/decisions", params={"keyword": "XGBoost"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 1
    assert payload["items"][0]["decision"] == "Use XGBoost with max_depth=5"

    # Non-matching keyword → empty page (still 200, valid request).
    resp_empty = client.get("/api/decisions", params={"keyword": "NoSuchString"})
    assert resp_empty.status_code == 200
    assert resp_empty.json()["count"] == 0

    # Keyword also matches against the reasoning column ("Pareto" only appears
    # in inverse_design.reasoning, not in any decision text).
    resp_reasoning = client.get("/api/decisions", params={"keyword": "Pareto"})
    assert resp_reasoning.status_code == 200
    payload_r = resp_reasoning.json()
    assert payload_r["count"] == 1
    assert payload_r["items"][0]["phase"] == "inverse_design"


def test_list_decisions_pagination(temp_db: Path, client: TestClient) -> None:
    _seed_three(temp_db)
    page1 = client.get("/api/decisions", params={"limit": 2, "offset": 0}).json()
    page2 = client.get("/api/decisions", params={"limit": 2, "offset": 2}).json()
    assert page1["count"] == 2
    assert page2["count"] == 1
    assert page1["limit"] == 2 and page1["offset"] == 0
    assert page2["limit"] == 2 and page2["offset"] == 2
    # No overlap between pages.
    ids_page1 = {item["id"] for item in page1["items"]}
    ids_page2 = {item["id"] for item in page2["items"]}
    assert ids_page1.isdisjoint(ids_page2)


def test_list_decisions_rejects_bad_limit(temp_db: Path, client: TestClient) -> None:
    # Validation: limit must be 1..100. 0 → 422; 1000 → 422.
    assert client.get("/api/decisions", params={"limit": 0}).status_code == 422
    assert client.get("/api/decisions", params={"limit": 1000}).status_code == 422


# ---------- single record endpoint ----------


def test_get_decision_by_id(temp_db: Path, client: TestClient) -> None:
    ids = _seed_three(temp_db)
    target = ids["training"]
    resp = client.get(f"/api/decisions/{target}")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["id"] == target
    assert payload["phase"] == "training"
    assert payload["decision"] == "Use XGBoost with max_depth=5"
    assert payload["alternatives_considered"] == ["depth=3", "depth=10"]
    assert payload["context"] == {"trials": 100, "best_r2": 0.87}
    assert payload["tags"] == ["xgboost", "hyperparameters"]
    assert payload["author"] == "model_trainer"


def test_get_decision_404(temp_db: Path, client: TestClient) -> None:
    _seed_three(temp_db)
    resp = client.get("/api/decisions/999999")
    assert resp.status_code == 404
    # FastAPI's HTTPException renders application/json {detail: ...}.
    assert "application/json" in resp.headers.get("content-type", "")
    assert "999999" in resp.json().get("detail", "")


def test_get_decision_summary_and_id_paths_disambiguate(
    temp_db: Path, client: TestClient
) -> None:
    """Route-order regression: /api/decisions/summary must NOT be parsed as
    /api/decisions/{decision_id} with id="summary" (which would 422 on the
    int conversion)."""
    _seed_three(temp_db)
    resp = client.get("/api/decisions/summary")
    assert resp.status_code == 200, resp.text
    assert "summary_md" in resp.json()


# ---------- summary endpoint ----------


def test_summary_endpoint_empty(temp_db: Path, client: TestClient) -> None:
    resp = client.get("/api/decisions/summary")
    assert resp.status_code == 200
    payload = resp.json()
    assert "summary_md" in payload
    assert isinstance(payload["summary_md"], str)
    # logger returns a fixed string when DB is empty.
    assert "пуст" in payload["summary_md"].lower()


def test_summary_endpoint_with_data(temp_db: Path, client: TestClient) -> None:
    _seed_three(temp_db)
    resp = client.get("/api/decisions/summary")
    assert resp.status_code == 200
    payload = resp.json()
    summary_md = payload["summary_md"]
    assert isinstance(summary_md, str) and summary_md.strip()
    # Markdown-ish output enumerates phases.
    assert "Фаза:" in summary_md
    assert "training" in summary_md
    assert "inverse_design" in summary_md


# ---------- response class regression ----------


def test_list_response_uses_safe_json_response(
    temp_db: Path, client: TestClient
) -> None:
    """SafeJSONResponse is the response_class for list_decisions — content-type
    must be application/json (not text/plain) and not fall through to the
    SPA shell. Catches accidental router→StaticFiles regressions."""
    _seed_three(temp_db)
    resp = client.get("/api/decisions")
    assert "application/json" in resp.headers.get("content-type", "")
    # Sanity: not the index.html shell.
    assert "<!doctype html>" not in resp.text.lower()


# ---------- DB sanity ----------


def test_seed_helper_writes_to_temp_db(temp_db: Path) -> None:
    """Ensures the fixture redirects writes correctly — guards against future
    refactors of the monkeypatch strategy."""
    _seed_three(temp_db)
    # Direct sqlite read — bypass logger entirely.
    conn = sqlite3.connect(temp_db)
    rows = conn.execute("SELECT phase, decision FROM decisions").fetchall()
    conn.close()
    assert len(rows) == 3
    phases = {r[0] for r in rows}
    assert phases == {"training", "inverse_design", "meta"}
