"""
Reporter Agent — HTML отчёт с топ-кандидатами.
"""
from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def render_html_report(
    candidates: list[dict],
    model_info: dict,
    user_request: dict,
    critic_reports: list[dict] | None = None,
) -> str:
    """Возвращает HTML-строку отчёта."""
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    top_n = candidates[:5]
    
    targets = user_request.get("targets", {})
    target_str = ", ".join(f"{k}: {v}" for k, v in targets.items())
    constraints = user_request.get("constraints", {})
    constraints_str = ", ".join(f"{k}: {v}" for k, v in constraints.items())
    
    html_parts = [f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Steel AI — Отчёт по дизайну HSLA</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; max-width: 1000px;
       margin: 20px auto; padding: 20px; color: #1a1a1a; line-height: 1.5; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
h2 {{ margin-top: 30px; color: #2a4e6e; }}
h3 {{ color: #444; }}
.meta {{ color: #666; font-size: 0.9em; }}
.summary {{ background: #f0f6fc; padding: 15px; border-left: 4px solid #2a4e6e;
            border-radius: 4px; margin: 20px 0; }}
.candidate {{ background: #fafafa; padding: 15px; margin: 15px 0;
              border-radius: 6px; border: 1px solid #ddd; }}
.candidate.top-pick {{ border-left: 4px solid #2d9a5a; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.9em; }}
th, td {{ border: 1px solid #ccc; padding: 5px 8px; text-align: center; }}
th {{ background: #eee; font-weight: 600; }}
.metric {{ display: inline-block; margin-right: 20px; }}
.metric strong {{ color: #2a4e6e; }}
.risk-low {{ color: #2d9a5a; font-weight: bold; }}
.risk-med {{ color: #e08a00; font-weight: bold; }}
.risk-high {{ color: #c93737; font-weight: bold; }}
.warning-box {{ background: #fff5e0; padding: 10px; border-left: 3px solid #e08a00;
                margin: 10px 0; border-radius: 3px; }}
.critic-box {{ background: #fef0f0; padding: 10px; border-left: 3px solid #c93737;
               margin: 10px 0; border-radius: 3px; }}
.success {{ color: #2d9a5a; }}
.mono {{ font-family: 'Menlo', 'Consolas', monospace; font-size: 0.88em; }}
</style>
</head>
<body>

<h1>Steel AI — Рекомендации по составу HSLA</h1>
<p class="meta">Сгенерировано: {now}<br>
Модель: <span class="mono">{model_info.get('version', 'n/a')}</span>
(R² test = {model_info.get('r2_test', 0):.3f}, MAE = {model_info.get('mae_test', 0):.1f} МПа)</p>

<h2>Задача</h2>
<div class="summary">
<div class="metric"><strong>Targets:</strong> {target_str}</div><br>
<div class="metric"><strong>Constraints:</strong> {constraints_str}</div>
</div>

<h2>Топ-{len(top_n)} кандидатов</h2>
"""]
    
    for i, c in enumerate(top_n, 1):
        comp = c.get("composition", {})
        proc = c.get("processing", {})
        derived = c.get("derived", {})
        pred = c.get("predicted", {})
        val = c.get("validation", {})
        
        overall = val.get("overall", "PASS")
        risk_class = {"PASS": "risk-low", "PASS_WITH_WARNINGS": "risk-med", "FAIL": "risk-high"}.get(overall, "risk-med")
        is_top = " top-pick" if i == 1 else ""
        
        comp_table_rows = " ".join(
            f"<td>{k.replace('_pct','').upper()}: {v:.4f}</td>"
            for k, v in comp.items() if v > 0.001
        )
        proc_items = " ".join(f"<strong>{k}:</strong> {v}" for k, v in proc.items())
        
        warning_html = ""
        if val.get("warnings"):
            warn_items = "".join(f"<li>{w['message']}</li>" for w in val["warnings"])
            warning_html = f'<div class="warning-box"><strong>Предупреждения:</strong><ul>{warn_items}</ul></div>'
        
        html_parts.append(f"""
<div class="candidate{is_top}">
<h3>Кандидат #{i} — <span class="{risk_class}">риск {overall}</span></h3>

<p><strong>Химический состав (%):</strong></p>
<table><tr>{comp_table_rows}</tr></table>

<p><strong>Маршрут:</strong> {proc_items}</p>

<p>
<div class="metric"><strong>σт прогноз:</strong> 
  {pred.get('mean', 0):.0f} ± {pred.get('ci_half_width', 0):.0f} МПа (90% ДИ)</div>
<div class="metric"><strong>90% ДИ:</strong> 
  [{pred.get('lower_90', 0):.0f}, {pred.get('upper_90', 0):.0f}]</div>
</p>

<p>
<div class="metric">CEV(IIW) = <strong>{derived.get('cev_iiw', 0):.3f}</strong></div>
<div class="metric">Pcm = <strong>{derived.get('pcm', 0):.3f}</strong></div>
<div class="metric">CEN = <strong>{derived.get('cen', 0):.3f}</strong></div>
<div class="metric">Микролегирование ∑ = <strong>{derived.get('microalloying_sum', 0):.4f}</strong></div>
</p>

<p>
<div class="metric">Стоимость легирования ≈ <strong>{c.get('objectives', {}).get('alloying_cost', 0):.1f} €/т</strong></div>
<div class="metric">Валидация: {val.get('n_passed', 0)}/{val.get('n_passed', 0) + val.get('n_failed', 0) + val.get('n_warnings', 0)} проверок пройдено</div>
</p>

{warning_html}
</div>
""")
    
    # Critic reports
    if critic_reports:
        html_parts.append('<h2>Отчёт критика</h2>')
        for cr in critic_reports:
            verdict = cr.get("verdict", "?")
            phase = cr.get("phase", "?")
            warnings = cr.get("warnings", [])
            if warnings:
                items = "".join(
                    f"<li><strong>[{w.get('severity')}] {w.get('pattern_id')}:</strong> "
                    f"{w.get('message', '')[:200]}<br><em>→ {w.get('suggestion', '')}</em></li>"
                    for w in warnings
                )
                html_parts.append(
                    f'<div class="critic-box"><strong>{phase} ({verdict}):</strong>'
                    f'<ul>{items}</ul></div>'
                )
    
    # Model provenance
    html_parts.append(f"""
<h2>О модели</h2>
<ul>
<li>Версия: <span class="mono">{model_info.get('version', 'n/a')}</span></li>
<li>Target: {model_info.get('target', 'n/a')}</li>
<li>Метрики: R² train = {model_info.get('r2_train', 0):.3f}, 
    R² val = {model_info.get('r2_val', 0):.3f},
    R² test = <strong>{model_info.get('r2_test', 0):.3f}</strong></li>
<li>MAE test = {model_info.get('mae_test', 0):.2f} МПа</li>
<li>Coverage 90% CI = {model_info.get('coverage_90_ci', 0):.1%}</li>
<li>Split strategy: {model_info.get('split_strategy', 'n/a')}</li>
<li>CV strategy: {model_info.get('cv_strategy', 'n/a')}</li>
</ul>

<h2>Рекомендованные действия</h2>
<ol>
<li>Опытная плавка кандидата #1 (20-50 кг, стандартная лабораторная АКОС)</li>
<li>Механические испытания по ISO 6892 или ГОСТ 1497</li>
<li>KCV при -40 и -60°C (минимум по 3 образца, ГОСТ 9454 или ISO 148-1)</li>
<li>Металлография: размер зерна, фазовый состав</li>
<li>Сопоставить результат с прогнозом и CI</li>
<li>Обновить модель (active learning) с этой плавкой в датасете</li>
</ol>

<p class="meta" style="margin-top: 30px;">
<em>Это отчёт MVP-демо на синтетических HSLA-данных. 
Для реальной производственной практики необходимо обучение на данных конкретного завода 
и адаптация ограничений под АКОС клиента.</em>
</p>

</body></html>
""")
    return "".join(html_parts)


def save_report(html: str, filename: str | None = None) -> Path:
    filename = filename or f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    path = REPORTS_DIR / filename
    path.write_text(html, encoding="utf-8")
    return path


class ReporterAgent:
    name = "reporter"
    
    def run(self, state, task):
        from app.backend.engine import AgentResult
        from decision_log.logger import log_decision
        try:
            candidates = task.get("candidates") or state.validated_candidates
            model_info = task.get("model_info") or state.model
            user_request = state.user_request
            critic_reports = [
                {
                    "phase": r.phase,
                    "verdict": r.verdict.value if hasattr(r.verdict, "value") else str(r.verdict),
                    "warnings": r.warnings,
                }
                for r in state.critic_reports
            ]
            
            html = render_html_report(candidates, model_info, user_request, critic_reports)
            path = save_report(html)
            
            log_decision(
                phase="reporting",
                decision=f"Отчёт сгенерирован: {path.name}",
                reasoning=f"HTML-отчёт на {len(candidates)} кандидатов, 5 в топ.",
                context={"path": str(path), "n_candidates": len(candidates)},
                author="reporter", tags=["report", "html"],
            )
            
            return AgentResult(
                agent_name=self.name, success=True,
                output={"report_html_path": str(path), "report_html": html[:1000]},
            )
        except Exception as e:
            logger.exception("Reporter failed")
            return AgentResult(agent_name=self.name, success=False, output={}, error=str(e))
