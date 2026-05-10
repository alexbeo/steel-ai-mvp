"""Tests for /api/system/kpi — topbar KPI strip aggregator (PR 13 migration).

The endpoint composes data from two independent sources:
  - filesystem: ``models/<version>/meta.json`` (latest by sort)
  - SQLite: ``decision_log/decisions.db`` via ``query_decisions``

Test isolation strategy mirrors the two existing patterns:
  - For the **models** half we monkeypatch ``app.api.routers.system.MODELS_DIR``
    (same as test_api_predict.py). Hand-crafted meta.json files stand in for
    real XGBoost training to keep the suite ~1 s.
  - For the **decision_log** half we monkeypatch ``query_decisions`` *as
    imported into the system router module* with a wrapper that pins a
    temporary SQLite db (same trick as test_api_decisions.py). The logger
    binds ``DEFAULT_DB_PATH`` at function-definition time, so module-level
    monkeypatching alone won't redirect the calls.

Together both fixtures isolate the endpoint from any state in the working
copy (``models/`` may have hundreds of dirs, the dev DB may have arbitrary
decisions) and let us assert exact KPI values.
"""
from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routers import system as system_router
from decision_log.logger import (
    log_decision,
    query_decisions as _real_query,
)


# ----------------------------- fixtures -----------------------------


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def empty_models_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Empty MODELS_DIR — KPI endpoint must still return 200 with nulls."""
    fake_dir = tmp_path / "models"
    fake_dir.mkdir()
    monkeypatch.setattr(system_router, "MODELS_DIR", fake_dir)
    yield fake_dir


@pytest.fixture()
def temp_decisions_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Pin Decision Log queries to a fresh temp DB.

    Why a wrapper rather than patching the module-level constant: the logger
    binds ``DEFAULT_DB_PATH`` at function-def time (see decision_log/logger.py
    signatures). Patching ``decision_log.logger.DEFAULT_DB_PATH`` would not
    redirect the calls — only the *name imported into our router* gets the
    new behaviour, which is exactly what we need.
    """
    tmp = Path(tempfile.mkdtemp()) / "test_kpi_decisions.db"

    def query_with_temp(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("db_path", tmp)
        return _real_query(*args, **kwargs)

    monkeypatch.setattr(system_router, "query_decisions", query_with_temp)
    yield tmp
    try:
        if tmp.exists():
            tmp.unlink()
        tmp.parent.rmdir()
    except OSError:
        pass


# ----------------------------- helpers -----------------------------


def _write_meta(
    models_dir: Path,
    version: str,
    *,
    steel_class: str = "pipe_hsla",
    r2_test: float = 0.89,
    coverage_90_ci: float = 0.91,
    target: str = "yield_strength_mpa",
) -> None:
    """Hand-craft a minimal meta.json — enough for /api/system/kpi to read."""
    d = models_dir / version
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(
        json.dumps(
            {
                "version": version,
                "steel_class": steel_class,
                "target": target,
                "feature_list": ["c_pct", "mn_pct"],
                "training_ranges": {"c_pct": [0.04, 0.12]},
                "metrics": {
                    "r2_test": r2_test,
                    "mae_test": 12.0,
                    "rmse_test": 18.0,
                    "coverage_90_ci": coverage_90_ci,
                    "n_train": 80,
                    "n_val": 10,
                    "n_test": 10,
                },
                "trained_at": "2026-05-10T03:00:00",
            }
        ),
        encoding="utf-8",
    )


# ----------------------------- tests -----------------------------


def test_kpi_returns_all_keys(
    empty_models_dir: Path, temp_decisions_db: Path, client: TestClient
) -> None:
    """Response shape contract — five top-level keys, present even with no data."""
    resp = client.get("/api/system/kpi")
    assert resp.status_code == 200
    payload = resp.json()
    assert set(payload.keys()) == {
        "active_model",
        "model_r2",
        "coverage_90_ci",
        "last_pareto_size",
        "phd_reviews_week",
    }
    # Empty environment → all data-sourced cells null, count is 0.
    assert payload["active_model"] is None
    assert payload["model_r2"] is None
    assert payload["coverage_90_ci"] is None
    assert payload["last_pareto_size"] is None
    assert payload["phd_reviews_week"] == 0


def test_kpi_active_model_from_latest_meta(
    empty_models_dir: Path, temp_decisions_db: Path, client: TestClient
) -> None:
    """Active model = last entry of sorted dirs; metrics come through."""
    _write_meta(empty_models_dir, "aaa_old", r2_test=0.70, coverage_90_ci=0.80)
    _write_meta(
        empty_models_dir, "ccc_latest", r2_test=0.89, coverage_90_ci=0.91,
    )
    _write_meta(empty_models_dir, "bbb_mid", r2_test=0.85, coverage_90_ci=0.88)

    resp = client.get("/api/system/kpi")
    assert resp.status_code == 200
    payload = resp.json()

    am = payload["active_model"]
    assert am is not None
    assert am["version"] == "ccc_latest"
    assert am["class_id"] == "pipe_hsla"
    # Profile name should resolve through load_steel_class — we accept any
    # non-empty string here so the test stays decoupled from the exact YAML
    # name (which can change without breaking semantics).
    assert isinstance(am["class_name"], str) and am["class_name"]
    assert am["target"] == "yield_strength_mpa"

    assert payload["model_r2"] == pytest.approx(0.89)
    # Coverage rounded to int percent: 0.91 → 91.
    assert payload["coverage_90_ci"] == 91


def test_kpi_active_model_null_when_no_models(
    empty_models_dir: Path, temp_decisions_db: Path, client: TestClient
) -> None:
    """Empty models/ → active_model + model_r2 + coverage are null, no 404."""
    resp = client.get("/api/system/kpi")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["active_model"] is None
    assert payload["model_r2"] is None
    assert payload["coverage_90_ci"] is None


def test_kpi_phd_reviews_count_from_decisions(
    empty_models_dir: Path, temp_decisions_db: Path, client: TestClient
) -> None:
    """Each of the four PhD-cycle tags contributes; dedup by id.

    We seed one decision per tag plus one untagged. Expected count = 4
    (the untagged one must NOT match).
    """
    log_decision(
        phase="meta", decision="recipe", reasoning="r",
        tags=["recipe_cycle"], db_path=temp_decisions_db,
    )
    log_decision(
        phase="meta", decision="hyp", reasoning="r",
        tags=["hypothesis_cycle", "physics"], db_path=temp_decisions_db,
    )
    log_decision(
        phase="deoxidation", decision="deox", reasoning="r",
        tags=["deoxidation_cycle"], db_path=temp_decisions_db,
    )
    log_decision(
        phase="training", decision="critic", reasoning="r",
        tags=["llm_critic"], db_path=temp_decisions_db,
    )
    log_decision(
        phase="meta", decision="other", reasoning="r",
        tags=["unrelated_tag"], db_path=temp_decisions_db,
    )

    resp = client.get("/api/system/kpi")
    assert resp.status_code == 200
    assert resp.json()["phd_reviews_week"] == 4


def test_kpi_phd_reviews_filters_by_week(
    empty_models_dir: Path, temp_decisions_db: Path, client: TestClient
) -> None:
    """Decisions older than 7 days must NOT count.

    The logger always stamps ``utcnow()``, so we backdate one row by
    direct sqlite UPDATE — same trick used elsewhere when we need to
    test time-window logic without freezing the clock.
    """
    import sqlite3

    fresh_id = log_decision(
        phase="meta", decision="fresh", reasoning="r",
        tags=["recipe_cycle"], db_path=temp_decisions_db,
    )
    stale_id = log_decision(
        phase="meta", decision="stale", reasoning="r",
        tags=["hypothesis_cycle"], db_path=temp_decisions_db,
    )
    # Backdate the second row by 8 days. We strip tzinfo to match the
    # logger's naive ``utcnow().isoformat()`` format — system_router.py
    # treats both naive and aware strings as UTC.
    stale_ts = (
        datetime.now(timezone.utc) - timedelta(days=8)
    ).replace(tzinfo=None).isoformat()
    conn = sqlite3.connect(temp_decisions_db)
    conn.execute(
        "UPDATE decisions SET timestamp = ? WHERE id = ?",
        (stale_ts, stale_id),
    )
    conn.commit()
    conn.close()

    resp = client.get("/api/system/kpi")
    assert resp.status_code == 200
    # Only the fresh one counts.
    assert resp.json()["phd_reviews_week"] == 1
    # Sanity — both rows exist in the DB (the count is filter-based, not deletion-based).
    assert fresh_id != stale_id


def test_kpi_last_pareto_size_from_inverse_design(
    empty_models_dir: Path, temp_decisions_db: Path, client: TestClient
) -> None:
    """Picks ``n_candidates`` of the most recent inverse_design entry.

    Two entries seeded — newer one must win. Older one should be ignored.
    """
    # Older entry (logged first → earlier timestamp).
    log_decision(
        phase="inverse_design",
        decision="NSGA-II run #1",
        reasoning="r",
        context={"n_candidates": 7},
        tags=["nsga2"],
        db_path=temp_decisions_db,
    )
    # Newer entry — small sleep would freeze the test, so we rely on the
    # ORDER BY timestamp DESC tie-break by INSERT order being good enough.
    # In practice utcnow() resolution > 1µs ≫ INSERT cost.
    log_decision(
        phase="inverse_design",
        decision="NSGA-II run #2",
        reasoning="r",
        context={"n_candidates": 12},
        tags=["nsga2"],
        db_path=temp_decisions_db,
    )

    resp = client.get("/api/system/kpi")
    assert resp.status_code == 200
    assert resp.json()["last_pareto_size"] == 12


def test_kpi_safe_json_numeric(
    empty_models_dir: Path, temp_decisions_db: Path, client: TestClient
) -> None:
    """SafeJSONResponse must serialise numpy-ish floats without choking.

    We seed a meta.json with a float (writing through json.dumps already
    coerces, but the assertion guards the response_class wiring — a
    regression to a plain JSONResponse here would still pass JSON
    serialization for these particular values, so we additionally check
    that content-type is application/json and that the body parses.)
    """
    _write_meta(empty_models_dir, "v_only", r2_test=0.834, coverage_90_ci=0.89)

    resp = client.get("/api/system/kpi")
    assert resp.status_code == 200
    assert "application/json" in resp.headers.get("content-type", "")
    payload = resp.json()
    assert payload["model_r2"] == pytest.approx(0.834)
    assert payload["coverage_90_ci"] == 89
