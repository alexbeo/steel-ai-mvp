"""PR 14 — HF Spaces metadata в README.md front-matter.

После миграции с Streamlit на FastAPI нужен `sdk: docker` (Docker SDK template),
а не `sdk: streamlit`. `app_file:` для Docker SDK не используется. `app_port`
должен совпадать с EXPOSE 7860 в Dockerfile.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"


@pytest.fixture(scope="module")
def front_matter() -> dict:
    assert README.exists(), f"README.md not found at {README}"
    text = README.read_text(encoding="utf-8")
    # Front-matter: блок между двумя строками `---` в самом начале.
    if not text.startswith("---"):
        pytest.fail("README.md must start with HF Spaces YAML front-matter (---)")
    # split into [empty, fm, body...]
    parts = text.split("---", 2)
    assert len(parts) >= 3, "README.md front-matter must be terminated by `---`"
    fm = yaml.safe_load(parts[1])
    assert isinstance(fm, dict), f"front-matter must parse to dict, got {type(fm)}"
    return fm


def test_sdk_is_docker(front_matter: dict) -> None:
    assert front_matter.get("sdk") == "docker", (
        f"HF Spaces SDK must be 'docker' after PR 14, got {front_matter.get('sdk')!r}"
    )


def test_no_streamlit_app_file(front_matter: dict) -> None:
    """Docker SDK does NOT use app_file. The PR 13 reviewer flagged this:
    leftover `app_file: app/frontend/app.py` would point at a deleted module.
    """
    app_file = front_matter.get("app_file")
    assert app_file != "app/frontend/app.py", (
        "stale Streamlit app_file pointer — must be removed for Docker SDK"
    )
    # Frontend dir was deleted in PR 13; if app_file is set at all, it must
    # point to something that exists (defensive).
    if app_file is not None:
        assert (REPO_ROOT / app_file).exists(), (
            f"app_file={app_file!r} declared but path does not exist"
        )


def test_app_port_matches_dockerfile(front_matter: dict) -> None:
    """If app_port is declared, it must match Dockerfile EXPOSE."""
    if "app_port" in front_matter:
        assert front_matter["app_port"] == 7860, (
            f"app_port must be 7860 to match Dockerfile EXPOSE, "
            f"got {front_matter['app_port']!r}"
        )


def test_metadata_has_basic_fields(front_matter: dict) -> None:
    for key in ("title", "license"):
        assert key in front_matter, f"missing required HF metadata key: {key}"


def test_no_streamlit_sdk_version(front_matter: dict) -> None:
    """`sdk_version` is meaningful only for Streamlit/Gradio SDKs."""
    assert "sdk_version" not in front_matter or front_matter.get("sdk") != "docker", (
        "sdk_version is irrelevant for Docker SDK; remove it"
    )
