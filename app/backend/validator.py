"""
Validator Agent.

Проверяет кандидатов от inverse design на:
- Физ. осмысленность составов
- Свариваемость (CEV, Pcm, зона Грэвилла)
- Горячую деформируемость (S/Mn, hot shortness Cu)
- OOD flag (не экстраполируем ли модель)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    verdict: Literal["PASS", "FAIL", "WARNING"]
    severity: Literal["HARD", "SOFT"] = "HARD"
    message: str = ""


@dataclass
class CandidateValidation:
    overall: Literal["PASS", "FAIL", "PASS_WITH_WARNINGS"]
    checks: list[CheckResult] = field(default_factory=list)
    
    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.verdict == "PASS")
    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if c.verdict == "FAIL")
    @property
    def n_warnings(self) -> int:
        return sum(1 for c in self.checks if c.verdict == "WARNING")


def validate_one(candidate: dict, max_cev: float = 0.43, max_pcm: float = 0.22) -> CandidateValidation:
    composition = candidate.get("composition", {})
    processing = candidate.get("processing", {})
    derived = candidate.get("derived", {})
    predicted = candidate.get("predicted", {})
    
    checks = []
    
    # 1. Chemical sense
    ti = composition.get("ti_pct", 0)
    al = composition.get("al_pct", 0)
    if ti > 0.05 and al < 0.015:
        checks.append(CheckResult("chem_ti_al", "FAIL",
            message=f"Ti={ti:.3f} требует Al≥0.015 для связывания N"))
    else:
        checks.append(CheckResult("chem_ti_al", "PASS"))
    
    # 2. Weldability CEV
    cev = derived.get("cev_iiw", 0)
    checks.append(CheckResult(
        "weldability_cev",
        "PASS" if cev <= max_cev else "FAIL",
        message=f"CEV={cev:.3f}, лимит {max_cev}",
    ))
    
    # 3. Pcm
    pcm = derived.get("pcm", 0)
    checks.append(CheckResult(
        "weldability_pcm",
        "PASS" if pcm <= max_pcm else "FAIL",
        message=f"Pcm={pcm:.3f}, лимит {max_pcm}",
    ))
    
    # 4. Graville zone
    c = composition.get("c_pct", 0)
    if cev >= 0.30 + 0.5 * c and c >= 0.20:
        zone = "C"
        checks.append(CheckResult("graville", "FAIL",
            message=f"Зона C по Грэвиллу (C={c}, CEV={cev:.3f})"))
    elif cev <= 0.30 + 0.5 * c and c <= 0.10:
        checks.append(CheckResult("graville", "PASS", message="Зона A — хорошая свариваемость"))
    else:
        checks.append(CheckResult("graville", "WARNING", severity="SOFT",
            message="Зона B — требуется контроль H2"))
    
    # 5. S/Mn ratio (hot redshort)
    s = composition.get("s_pct", 0)
    mn = composition.get("mn_pct", 0)
    if s > 0.005:
        ratio = s / mn if mn > 0 else float("inf")
        if ratio > 0.04:
            checks.append(CheckResult("hot_redshort", "FAIL",
                message=f"S/Mn={ratio:.4f} > 0.04 при %S={s} — риск красноломкости"))
        else:
            checks.append(CheckResult("hot_redshort", "PASS"))
    else:
        checks.append(CheckResult("hot_redshort", "PASS",
            message="S низкий, риск отсутствует"))
    
    # 6. Cu hot shortness
    cu = composition.get("cu_pct", 0)
    ni = composition.get("ni_pct", 0)
    if cu > 0.2 and (ni + cu == 0 or ni / (ni + cu) < 0.5):
        checks.append(CheckResult("hot_shortness_cu", "FAIL",
            message=f"Cu={cu:.2f} без достаточного Ni — риск hot shortness"))
    else:
        checks.append(CheckResult("hot_shortness_cu", "PASS"))
    
    # 7. OOD flag
    if predicted.get("ood_flag"):
        checks.append(CheckResult("ood", "FAIL", severity="HARD",
            message="Состав вне training distribution — прогноз ненадёжен"))
    else:
        checks.append(CheckResult("ood", "PASS"))
    
    # Aggregate
    hard_fails = [c for c in checks if c.verdict == "FAIL" and c.severity == "HARD"]
    warnings = [c for c in checks if c.verdict == "WARNING" or c.severity == "SOFT"]
    
    if hard_fails:
        overall = "FAIL"
    elif warnings:
        overall = "PASS_WITH_WARNINGS"
    else:
        overall = "PASS"
    
    return CandidateValidation(overall=overall, checks=checks)


def validate_batch(candidates: list[dict]) -> dict:
    approved = []
    rejected = []
    rejection_summary: dict[str, int] = {}
    
    for cand in candidates:
        val = validate_one(cand)
        enriched = {
            **cand,
            "validation": {
                "overall": val.overall,
                "n_passed": val.n_passed,
                "n_failed": val.n_failed,
                "n_warnings": val.n_warnings,
                "failed_checks": [
                    {"name": c.name, "message": c.message}
                    for c in val.checks if c.verdict == "FAIL"
                ],
                "warnings": [
                    {"name": c.name, "message": c.message}
                    for c in val.checks if c.verdict == "WARNING"
                ],
            },
        }
        if val.overall == "FAIL":
            rejected.append(enriched)
            for c in val.checks:
                if c.verdict == "FAIL":
                    rejection_summary[c.name] = rejection_summary.get(c.name, 0) + 1
        else:
            approved.append(enriched)
    
    return {
        "input_count": len(candidates),
        "approved": approved,
        "rejected": rejected,
        "rejection_summary": rejection_summary,
    }


class ValidatorAgent:
    name = "validator"
    
    def run(self, state, task):
        from app.backend.engine import AgentResult
        from decision_log.logger import log_decision
        
        try:
            candidates = task.get("candidates") or state.candidates
            result = validate_batch(candidates)
            
            log_decision(
                phase="validation",
                decision=f"Валидация: {len(result['approved'])}/{result['input_count']} прошли",
                reasoning=(
                    f"Проверены {result['input_count']} кандидатов по чеклисту: "
                    f"chemical sense, weldability (CEV/Pcm/Graville), hot workability, OOD. "
                    f"Причины отсева: {result['rejection_summary']}."
                ),
                context=result["rejection_summary"],
                author="validator", tags=["validation"],
            )
            
            return AgentResult(
                agent_name=self.name, success=True,
                output={
                    "approved": result["approved"],
                    "rejected_count": len(result["rejected"]),
                    "rejection_summary": result["rejection_summary"],
                },
            )
        except Exception as e:
            logger.exception("Validator failed")
            return AgentResult(
                agent_name=self.name, success=False, output={}, error=str(e),
            )


if __name__ == "__main__":
    import json
    sample = {
        "composition": {"c_pct": 0.08, "si_pct": 0.3, "mn_pct": 1.5, "s_pct": 0.003,
                       "p_pct": 0.012, "cu_pct": 0.15, "ni_pct": 0.1, "al_pct": 0.035,
                       "ti_pct": 0.015, "nb_pct": 0.04},
        "derived": {"cev_iiw": 0.38, "pcm": 0.18, "cen": 0.24},
        "predicted": {"mean": 520, "ood_flag": False},
    }
    val = validate_one(sample)
    print(f"Overall: {val.overall}")
    print(f"Passed {val.n_passed}, failed {val.n_failed}, warnings {val.n_warnings}")
    for c in val.checks:
        print(f"  [{c.verdict}] {c.name}: {c.message}")
