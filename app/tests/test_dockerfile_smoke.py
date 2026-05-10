"""PR 14 — статический анализ Dockerfile + docker-compose.yml.

Не запускает `docker build`/`docker run` — только парсит файлы. Полный
build идёт в CI или вручную перед deploy на HF Spaces.

Контракт PR 14:
- Dockerfile — multi-stage (builder → runtime), non-root user, EXPOSE 7860,
  PYTHONPATH=/app, без `COPY . .`.
- docker-compose.yml — port 8000:8000 + PORT=8000 env (override default 7860).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
COMPOSE = REPO_ROOT / "docker-compose.yml"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    assert DOCKERFILE.exists(), f"Dockerfile not found at {DOCKERFILE}"
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def compose_yaml() -> dict:
    assert COMPOSE.exists(), f"docker-compose.yml not found at {COMPOSE}"
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


# ---------- Dockerfile ----------


def test_dockerfile_is_multi_stage(dockerfile_text: str) -> None:
    """Builder + runtime stages explicitly named."""
    assert "AS builder" in dockerfile_text, "builder stage missing"
    assert "AS runtime" in dockerfile_text, "runtime stage missing"
    # at least 2 FROM statements
    from_count = sum(
        1 for line in dockerfile_text.splitlines()
        if line.strip().upper().startswith("FROM ")
    )
    assert from_count >= 2, f"expected >=2 FROM lines, got {from_count}"


def test_dockerfile_uses_python_311_slim(dockerfile_text: str) -> None:
    assert "python:3.11-slim" in dockerfile_text


def test_dockerfile_creates_non_root_user(dockerfile_text: str) -> None:
    assert "useradd" in dockerfile_text, "non-root user must be created"
    assert "USER steel" in dockerfile_text, "must switch to non-root `steel` user"


def test_dockerfile_copies_venv_from_builder(dockerfile_text: str) -> None:
    assert "COPY --from=builder /opt/venv /opt/venv" in dockerfile_text


def test_dockerfile_exposes_7860(dockerfile_text: str) -> None:
    """HF Spaces Docker SDK default port."""
    assert "EXPOSE 7860" in dockerfile_text


def test_dockerfile_sets_pythonpath(dockerfile_text: str) -> None:
    assert "PYTHONPATH=/app" in dockerfile_text


def test_dockerfile_runs_uvicorn_app_api_main(dockerfile_text: str) -> None:
    assert "uvicorn app.api.main:app" in dockerfile_text


def test_dockerfile_does_not_copy_everything(dockerfile_text: str) -> None:
    """`COPY . .` would pull in tests/, models/, .git/ etc. — banned in PR 14."""
    forbidden_lines = [
        line.strip()
        for line in dockerfile_text.splitlines()
        if line.strip().upper().startswith("COPY . .")
        or line.strip().upper() == "COPY . /APP"
    ]
    assert not forbidden_lines, f"forbidden COPY found: {forbidden_lines}"


def test_dockerfile_has_port_env_override(dockerfile_text: str) -> None:
    """ENV PORT=7860 + ${PORT} in CMD — for docker-compose override."""
    assert "ENV PORT=7860" in dockerfile_text
    assert "${PORT}" in dockerfile_text


def test_dockerfile_does_not_reference_pyproject(dockerfile_text: str) -> None:
    """pyproject.toml не существует в репо (проект не упакован) — copy её
    сломал бы build. Защита от регрессии дизайн-доки PR 14.
    """
    assert "pyproject.toml" not in dockerfile_text


def test_cmd_uses_exec_for_signal_forwarding(dockerfile_text: str) -> None:
    """CMD должен использовать `exec` чтобы uvicorn получал SIGTERM напрямую.

    Без `exec` PID 1 = sh, который не форвардит сигналы — HF Spaces redeploy
    превращается в SIGKILL после grace period. См. PR 14 reviewer nit.
    """
    assert "exec uvicorn" in dockerfile_text, (
        "CMD must use 'exec uvicorn' for PID 1 signal forwarding"
    )


# ---------- .dockerignore ----------


def test_dockerignore_exists() -> None:
    """`.dockerignore` обязателен — без него COPY тащит __pycache__/tests/secrets."""
    p = REPO_ROOT / ".dockerignore"
    assert p.exists(), ".dockerignore must exist (HF Spaces image hygiene)"


def test_dockerignore_excludes_runtime_artifacts() -> None:
    """`.dockerignore` обязан исключать SQLite DB, parquet, tests, __pycache__."""
    text = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    required = [
        "**/__pycache__",
        "decision_log/*.db",
        "decision_log/price_snapshots/",
        "data/*.parquet",
        "app/tests/",
        ".env",
        ".git/",
    ]
    for rule in required:
        assert rule in text, f".dockerignore must contain rule '{rule}'"


# ---------- docker-compose.yml ----------


def test_compose_has_steel_ai_service(compose_yaml: dict) -> None:
    assert "services" in compose_yaml
    assert "steel-ai" in compose_yaml["services"]


def test_compose_maps_port_8000(compose_yaml: dict) -> None:
    svc = compose_yaml["services"]["steel-ai"]
    ports = svc.get("ports", [])
    assert ports, "ports list must be non-empty"
    # Docker compose accepts both "8000:8000" and {"target": ..., "published": ...}
    has_8000 = any(
        (isinstance(p, str) and p.startswith("8000:"))
        or (isinstance(p, dict) and p.get("published") == 8000)
        for p in ports
    )
    assert has_8000, f"expected 8000 published, got {ports}"


def test_compose_sets_port_env_to_8000(compose_yaml: dict) -> None:
    """Required for the host:container port match (8000:8000)."""
    svc = compose_yaml["services"]["steel-ai"]
    env = svc.get("environment", [])
    # env can be list of "K=V" or dict
    if isinstance(env, list):
        flat = {item.split("=", 1)[0]: item.split("=", 1)[1]
                for item in env if "=" in item and not item.lstrip().startswith("#")}
    else:
        flat = {k: str(v) for k, v in env.items()}
    assert flat.get("PORT") == "8000", (
        f"docker-compose must override PORT=8000 to match host port mapping; got {flat!r}"
    )


def test_compose_mounts_persistent_volumes(compose_yaml: dict) -> None:
    svc = compose_yaml["services"]["steel-ai"]
    vols = svc.get("volumes", [])
    expected_targets = {"/app/data", "/app/models", "/app/reports", "/app/decision_log"}
    found_targets = set()
    for v in vols:
        if isinstance(v, str) and ":" in v:
            found_targets.add(v.split(":", 1)[1].split(":")[0])
        elif isinstance(v, dict) and "target" in v:
            found_targets.add(v["target"])
    missing = expected_targets - found_targets
    assert not missing, f"missing volume mounts: {missing}"
