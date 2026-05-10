"""R-001 Step 4 — visual polish smoke checks.

Verifies the ten or so static assets that R-001 Step 4 added or amended:
  - overrides.css gained shimmer + pulse keyframes, hover transitions
    on .nav-item / .btn / KPI cells, and the footer thermal-strip rules.
  - utils/dom.js exports skeletonRow + skeletonStack helpers (used by
    history / predict / experiments views).
  - design.js inlines the SVG target-band element (data-target-band).
  - index.html mounts a permanent <footer class="thermal-footer"> with
    a data-region="thermal-footer" hook and the gradient bar.

These tests intentionally inspect the static assets rather than the live
DOM — JSDOM is not in the toolchain and the FastAPI surface only serves
the files. If a future PR introduces a JS test runner, these can move to
real DOM assertions.
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.api.main import app

STATIC = Path(__file__).resolve().parents[1] / "web" / "static"
client = TestClient(app)


# ─────────────────────── overrides.css surface ─────────────────────────


def test_overrides_css_has_hover_rules() -> None:
    """At least 5 :hover declarations land in overrides.css after Step 4.

    Rationale: the spec asks for hover transitions across nav, buttons,
    KPI cards, candidate rows and Pareto markers. A regression that
    drops these declarations from the bundle would make the UI feel
    dead again.
    """
    css = (STATIC / "css" / "overrides.css").read_text(encoding="utf-8")
    hits = css.count(":hover")
    assert hits >= 5, f"expected >= 5 :hover rules in overrides.css, got {hits}"


def test_overrides_css_has_keyframes() -> None:
    """Both `shimmer` (skeleton loader) and `pulse` (status indicator)
    keyframes must be declared in overrides.css."""
    css = (STATIC / "css" / "overrides.css").read_text(encoding="utf-8")
    assert "@keyframes shimmer" in css, "missing @keyframes shimmer"
    assert "@keyframes pulse" in css, "missing @keyframes pulse"
    # Bonus check: at least 2 @keyframes blocks total.
    assert css.count("@keyframes ") >= 2


def test_overrides_css_has_transition_on_nav_item() -> None:
    """`.nav-item` should declare a smooth transition (not instant flash)."""
    css = (STATIC / "css" / "overrides.css").read_text(encoding="utf-8")
    # Anchor on the selector + transition keyword in close proximity.
    assert ".nav-item {" in css, "missing .nav-item rule block"
    # Pull the .nav-item block content: from the rule to first '}' that
    # closes it. Cheap parse — just substring-match transition appears
    # in the same window.
    idx = css.index(".nav-item {")
    end = css.index("}", idx)
    block = css[idx:end]
    assert "transition" in block, ".nav-item should declare transition for smooth hover"


def test_overrides_css_has_thermal_footer_block() -> None:
    """Footer thermal strip CSS must exist (linked from index.html)."""
    css = (STATIC / "css" / "overrides.css").read_text(encoding="utf-8")
    assert ".thermal-footer" in css
    assert ".thermal-bar" in css
    # Gradient-fill on .thermal-bar gives the heat-scale colour ramp.
    assert "linear-gradient" in css


# ─────────────────────── DOM helper exports ────────────────────────────


def test_skeleton_helper_exists_in_dom_utils() -> None:
    """utils/dom.js must export skeletonRow + skeletonStack so views can
    import them. The presence of `export function skeletonRow` /
    `skeletonStack` is sufficient for ESM module resolution."""
    src = (STATIC / "js" / "utils" / "dom.js").read_text(encoding="utf-8")
    assert "export function skeletonRow" in src
    assert "export function skeletonStack" in src


# ─────────────────────── design.js target band ─────────────────────────


def test_targets_band_svg_in_design_view() -> None:
    """design.js should attach a `data-target-band` element somewhere
    so the SVG renderer has a stable mount point. Also verifies the
    helper function name `updateTargetBand` is wired."""
    src = (STATIC / "js" / "views" / "design.js").read_text(encoding="utf-8")
    assert "data-target-band" in src, (
        "Дизайн tab is expected to mount inline target-band SVG with a "
        "`data-target-band` marker — used both by tests and CSS."
    )
    assert "function updateTargetBand" in src
    # We construct SVG via createElementNS, not innerHTML (security hook
    # in this repo refuses innerHTML). Pin that.
    assert "createElementNS" in src


# ─────────────────────── index.html footer mount ────────────────────────


def test_footer_thermal_strip_in_index_html() -> None:
    """index.html must contain the footer thermal strip element so the
    chrome is always visible across all 8 tabs."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'class="thermal-footer"' in html or "class='thermal-footer'" in html
    assert 'data-region="thermal-footer"' in html
    assert 'class="thermal-bar"' in html


def test_index_served_includes_thermal_footer() -> None:
    """Live FastAPI surface returns the thermal-footer element — guards
    against the static mount silently regressing."""
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    assert "thermal-footer" in body, (
        "GET / response must include the visible footer chrome element"
    )
    assert "thermal-bar" in body
