"""PR 1 sanity checks for FastAPI surface — health endpoint and static mount."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert "version" in payload
    assert isinstance(payload["version"], str) and payload["version"]
    # llm_ready key present regardless of environment (boolean type)
    assert "llm_ready" in payload
    assert isinstance(payload["llm_ready"], bool)


def test_static_index_served() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    ctype = resp.headers.get("content-type", "")
    assert "text/html" in ctype, f"expected text/html, got {ctype!r}"
    body = resp.text
    # Sanity-check the shell renders the brand and the CSS link.
    assert "STEEL.AI" in body
    assert "/css/app.css" in body


def test_static_css_served() -> None:
    resp = client.get("/css/app.css")
    assert resp.status_code == 200
    ctype = resp.headers.get("content-type", "")
    assert "css" in ctype, f"expected css content-type, got {ctype!r}"
    # Confirm the oklch palette token landed (extracted from index.html).
    assert "--bg:" in resp.text
    assert "oklch" in resp.text


def test_app_js_module_served() -> None:
    resp = client.get("/js/app.js")
    assert resp.status_code == 200
    # Browser needs JS MIME for type="module" to evaluate; FastAPI/Starlette
    # serves .js as application/javascript by default.
    assert "javascript" in resp.headers.get("content-type", "")
    assert "mountRouter" in resp.text


def test_404_for_unknown_api_path() -> None:
    """Mount-order invariant: /api/* must 404 cleanly, NOT fall through to
    StaticFiles(html=True) which would return index.html with 200.

    Protects against regressions in PR 2-12 when new routers are added — if
    someone accidentally mounts StaticFiles before the API routers, this test
    catches it: a missing /api/* would silently return the SPA shell instead
    of a JSON 404.
    """
    resp = client.get("/api/non-existent")
    assert resp.status_code == 404
    ctype = resp.headers.get("content-type", "")
    assert "application/json" in ctype, f"expected JSON 404, got {ctype!r}"
    payload = resp.json()
    assert "detail" in payload, f"expected FastAPI 'detail' key, got {payload!r}"
