"""Shadow validation report generator (Phase 2 S3).

HTML + JSON отчёт по результатам shadow validation (S1 comparison + S2 stats).
Per-heat таблица, plant×method агрегаты, savings (Al kg + €), гипотеза −10%.

Честный framing: retrospective (не counterfactual), экономия от η (не недодоз),
synthetic тавтология, between-heat vs per-heat uncertainty.
"""
from __future__ import annotations

import html
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.backend.shadow_stats import ShadowStats
from app.backend.shadow_validation import ShadowComparison
from app.backend.slag_aware_deox import DEFAULT_AL_COMMODITY_PRICE_EUR_PER_KG

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Seed Al commodity price fallback (EUR/kg) — re-export из slag_aware_deox.
DEFAULT_AL_PRICE_EUR_PER_KG = DEFAULT_AL_COMMODITY_PRICE_EUR_PER_KG

_DISCLAIMERS = [
    "Retrospective shadow-валидация — НЕ counterfactual. Оператор при фактической "
    "плавке мог видеть информацию, недоступную модели.",
    "Экономия Al обусловлена более высоким прогнозным η_Al (меньше потерь в шлак/угар), "
    "НЕ недодозировкой: AI-доза бюджетирует тот же target O_a и residual [Al].",
    "CI отражает between-heat вариативность; per-heat conformal p10/p90 — отдельный "
    "канал неопределённости (см. S1).",
]


def _esc(x) -> str:
    """HTML-escape a user-derived string (heat_id / plant_id / method_id / η source).

    Numeric and ``None`` callers pass through to ``"—"`` so this is safe to wrap
    around any cell value. Normal strings escape to a no-op for the S3 tests.
    """
    return html.escape(str(x)) if x is not None else "—"


def _compute_savings_eur(
    stats: ShadowStats,
    al_price_eur_per_kg: float = DEFAULT_AL_PRICE_EUR_PER_KG,
) -> float:
    """€-savings = Al_saved_kg × Al price. NaN-safe (нет compared → 0)."""
    if stats.total_al_saved_kg != stats.total_al_saved_kg:  # NaN guard
        return 0.0
    return stats.total_al_saved_kg * al_price_eur_per_kg


def _fmt(x: float | None, spec: str = "+.1f") -> str:
    """Format float NaN/None-safe."""
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return format(x, spec)


def render_shadow_html(
    comparisons: list[ShadowComparison],
    stats: ShadowStats,
    *,
    al_price_eur_per_kg: float = DEFAULT_AL_PRICE_EUR_PER_KG,
    is_synthetic: bool = False,
) -> str:
    """Возвращает HTML-строку отчёта."""
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    savings_eur = _compute_savings_eur(stats, al_price_eur_per_kg)

    disclaimers = list(_DISCLAIMERS)
    if is_synthetic:
        disclaimers.insert(
            0,
            "⚠️ SYNTHETIC данные: AI re-predicts свою η-формулу (тавтология). "
            "Реальная валидация — на real heats_db.",
        )
    if stats.n_literature_fallback > 0:
        disclaimers.append(
            f"{stats.n_literature_fallback} плавок на literature_fallback "
            f"(нет plant-калибровки) — down-weight при интерпретации."
        )

    hyp_color = "#2a9d4a" if stats.hypothesis_met else "#c44"
    hyp_text = "ПОДТВЕРЖДЕНА" if stats.hypothesis_met else "НЕ подтверждена"

    # Compared rows (cap 100 для UI)
    compared = [c for c in comparisons if c.skip_reason is None]
    rows_html = "".join(
        f"<tr><td>{_esc(c.heat_id) if c.heat_id else (c.heat_pk if c.heat_pk is not None else '—')}</td>"
        f"<td>{_esc(c.plant_id)}</td>"
        f"<td>{_esc(c.method_id)}</td><td>{_fmt(c.al_actual_kg, '.1f')}</td>"
        f"<td>{_fmt(c.al_ai_p50_kg, '.1f')}</td>"
        f"<td style='color:{'#2a9d4a' if (c.delta_pct or 0) < 0 else '#c44'}'>"
        f"{_fmt(c.delta_pct)}%</td>"
        f"<td>{'✓' if c.quality_pass else '✗'}</td>"
        f"<td>{_esc(c.eta_ai_source)}</td></tr>"
        for c in compared[:100]
    )

    # Per-stratum
    stratum_html = "".join(
        f"<tr><td>{_esc(s.plant_id)}</td><td>{_esc(s.method_id)}</td><td>{s.n}</td>"
        f"<td>{_fmt(s.median_delta_pct)}%</td>"
        f"<td>[{_fmt(s.ci_low)}, {_fmt(s.ci_high)}] {_esc(s.ci_method)}</td>"
        f"<td>{f'{s.wilcoxon_p:.2e}' if s.wilcoxon_p is not None else '—'}</td></tr>"
        for s in stats.per_stratum
    )

    notes_html = "".join(f"<li>{_esc(n)}</li>" for n in stats.notes)
    disc_html = "".join(f"<li>{_esc(d)}</li>" for d in disclaimers)
    qfail = stats.n_compared - stats.n_quality_pass

    return f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>Shadow Validation Report — {now}</title>
<style>
body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222}}
h1{{border-bottom:2px solid #333;padding-bottom:8px}}
h2{{color:#2a4e6e;margin-top:1.5em}}
.hyp{{font-size:1.3em;font-weight:bold;color:{hyp_color};padding:0.5em;border:2px solid {hyp_color};border-radius:6px;display:inline-block}}
.kpi{{display:flex;gap:1.5em;margin:1em 0;flex-wrap:wrap}}
.kpi div{{background:#f5f5f5;padding:0.8em 1.2em;border-radius:6px}}
.kpi b{{font-size:1.4em;display:block}}
table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:0.9em}}
th,td{{border:1px solid #ddd;padding:5px 8px;text-align:right}}
th{{background:#333;color:#fff}}
td:first-child,td:nth-child(2),td:nth-child(3){{text-align:left}}
.disclaimers{{background:#fff8e1;border-left:4px solid #f0a000;padding:0.5em 1em;margin:1em 0}}
.notes{{background:#f0f7ff;border-left:4px solid #4080d0;padding:0.5em 1em}}
</style></head><body>
<h1>Shadow Validation — η_Al calibration</h1>
<p>Сгенерирован {now}</p>
<p class="hyp">Гипотеза −10% Al: {hyp_text}</p>
<div class="kpi">
  <div><b>{_fmt(stats.median_delta_pct)}%</b>median Δ Al (quality-pass)</div>
  <div><b>[{_fmt(stats.ci_low)}, {_fmt(stats.ci_high)}]</b>95% bootstrap CI</div>
  <div><b>{f'{stats.wilcoxon_p:.2e}' if stats.wilcoxon_p is not None else 'n/a'}</b>Wilcoxon p</div>
  <div><b>{stats.total_al_saved_kg:.0f} kg</b>Al сэкономлено</div>
  <div><b>€{savings_eur:,.0f}</b>при €{al_price_eur_per_kg}/kg Al</div>
  <div><b>{stats.n_quality_pass}/{stats.n_compared}</b>quality-pass / compared ({qfail} fail)</div>
</div>
<div class="disclaimers"><b>Ограничения интерпретации:</b><ul>{disc_html}</ul></div>
<h2>Стратификация plant × method</h2>
<table><tr><th>Plant</th><th>Method</th><th>N</th><th>median Δ%</th><th>95% CI</th><th>Wilcoxon p</th></tr>
{stratum_html or '<tr><td colspan="6" style="text-align:center">нет стратумов</td></tr>'}</table>
<h2>Плавки (первые 100 из {len(compared)})</h2>
<table><tr><th>Heat</th><th>Plant</th><th>Method</th><th>Al факт, kg</th><th>Al AI p50, kg</th><th>Δ%</th><th>Quality</th><th>η source</th></tr>
{rows_html or '<tr><td colspan="8" style="text-align:center">нет сравнённых плавок</td></tr>'}</table>
{f'<div class="notes"><b>Заметки:</b><ul>{notes_html}</ul></div>' if notes_html else ''}
</body></html>"""


def generate_shadow_report(
    comparisons: list[ShadowComparison],
    stats: ShadowStats,
    *,
    al_price_eur_per_kg: float = DEFAULT_AL_PRICE_EUR_PER_KG,
    is_synthetic: bool = False,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Записывает HTML + JSON в reports/. Возвращает (html_path, json_path)."""
    out = out_dir or REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = out / f"shadow_{ts}.html"
    json_path = out / f"shadow_{ts}.json"
    html_path.write_text(
        render_shadow_html(
            comparisons,
            stats,
            al_price_eur_per_kg=al_price_eur_per_kg,
            is_synthetic=is_synthetic,
        ),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "is_synthetic": is_synthetic,
                "stats": asdict(stats),
                "savings_eur": _compute_savings_eur(stats, al_price_eur_per_kg),
                "al_price_eur_per_kg": al_price_eur_per_kg,
                "comparisons": [
                    asdict(c) for c in comparisons if c.skip_reason is None
                ][:500],
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info("Shadow report written: %s / %s", html_path, json_path)
    return html_path, json_path


__all__ = [
    "render_shadow_html",
    "generate_shadow_report",
    "REPORTS_DIR",
    "DEFAULT_AL_PRICE_EUR_PER_KG",
]
