"""Tests для shadow report generator (Phase 2 S3)."""
from __future__ import annotations

import json

from app.backend.shadow_reporter import (
    _compute_savings_eur,
    generate_shadow_report,
    render_shadow_html,
)
from app.backend.shadow_stats import ShadowStats, StratumStats
from app.backend.shadow_validation import ShadowComparison


def _stats(**kw) -> ShadowStats:
    d = dict(
        n_total=50,
        n_compared=45,
        n_quality_pass=40,
        median_delta_pct=-15.0,
        median_delta_pct_all=-14.0,
        ci_low=-17.0,
        ci_high=-13.0,
        wilcoxon_p=1e-8,
        hypothesis_met=True,
        total_al_saved_kg=500.0,
        n_literature_fallback=0,
        n_resamples=2000,
        random_state=42,
        per_stratum=[],
        notes=[],
    )
    d.update(kw)
    return ShadowStats(**d)


def _comp(**kw) -> ShadowComparison:
    d = dict(
        heat_pk=1,
        heat_id="H1",
        plant_id="P1",
        method_id="asis_shot",
        al_actual_kg=200.0,
        al_ai_p50_kg=170.0,
        al_ai_p90_kg=180.0,
        delta_pct=-15.0,
        delta_p90_pct=-10.0,
        ai_target_o_a_ppm=5.0,
        actual_o_a_ppm=5.0,
        residual_in_spec=True,
        dose_sufficient=True,
        quality_pass=True,
        eta_ai_source="plant_only",
        skip_reason=None,
        notes=[],
    )
    d.update(kw)
    return ShadowComparison(**d)


def test_savings_eur():
    s = _stats(total_al_saved_kg=1000.0)
    assert _compute_savings_eur(s, 2.40) == 2400.0


def test_savings_eur_nan_safe():
    s = _stats(total_al_saved_kg=float("nan"))
    assert _compute_savings_eur(s, 2.40) == 0.0


def test_render_html_contains_hypothesis():
    html = render_shadow_html([_comp()], _stats(hypothesis_met=True))
    assert "ПОДТВЕРЖДЕНА" in html
    assert "shadow" in html.lower()


def test_render_html_not_met():
    html = render_shadow_html([_comp()], _stats(hypothesis_met=False))
    assert "НЕ подтверждена" in html


def test_render_html_disclaimers_present():
    """Честный framing обязателен."""
    html = render_shadow_html([_comp()], _stats())
    assert "Retrospective" in html
    assert "η_Al" in html  # экономия от η
    assert "counterfactual" in html.lower()


def test_render_html_synthetic_warning():
    html = render_shadow_html([_comp()], _stats(), is_synthetic=True)
    assert "SYNTHETIC" in html or "synthetic" in html
    assert "тавтолог" in html.lower()


def test_render_html_literature_note():
    html = render_shadow_html([_comp()], _stats(n_literature_fallback=5))
    assert "literature" in html.lower()


def test_generate_report_writes_files(tmp_path):
    html, jsn = generate_shadow_report([_comp()], _stats(), out_dir=tmp_path)
    assert html.exists() and jsn.exists()
    data = json.loads(jsn.read_text())
    assert data["stats"]["hypothesis_met"] is True
    assert "savings_eur" in data
    assert data["savings_eur"] == 500.0 * 2.40


def test_render_html_per_stratum():
    strat = StratumStats(
        plant_id="P1",
        method_id="asis_shot",
        n=20,
        n_quality_pass=20,
        median_delta_pct=-15.0,
        ci_low=-17.0,
        ci_high=-13.0,
        ci_method="percentile",
        wilcoxon_p=1e-5,
        mean_al_saved_kg=25.0,
    )
    html = render_shadow_html([_comp()], _stats(per_stratum=[strat]))
    assert "asis_shot" in html


def test_render_html_empty_compared():
    """Edge: всё skip → нет строк, но HTML валиден."""
    skipped = _comp(skip_reason="no_outcome")
    html = render_shadow_html([skipped], _stats(n_compared=0, n_quality_pass=0))
    assert "нет сравнённых плавок" in html


def test_render_html_escapes_user_strings():
    """S4: heat_id/plant_id из user-данных экранируются (XSS hardening)."""
    from app.backend.shadow_reporter import render_shadow_html

    c = _comp(heat_id="<script>alert(1)</script>", plant_id="P<b>1</b>")
    html = render_shadow_html([c], _stats())
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
