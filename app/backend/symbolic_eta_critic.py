"""
Symbolic η_Al Correction Critic — adversarial PhD peer review of a
closed-form correction factor found by symbolic regression (gplearn, B2 / PR 11).

Mirrors deoxidation_critic architecture: gitignored system prompt loaded via
prompt_loader, structured output via tool_use, prompt caching (ephemeral),
graceful fallback to None when prompt or ANTHROPIC_API_KEY are absent.

The critic evaluates *physical plausibility* of a correction formula
    correction = eta_al_effective / method_eta_baseline  (≈ 1.0 ± 0.3)
checking that dominant term signs match process-metallurgy expectations
(FeO re-oxidation, high-T burn-off, Si pre-deox, refractory wear, etc.).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from app.backend.prompt_loader import PromptNotFoundError, load_prompt

logger = logging.getLogger(__name__)

Verdict = Literal["ACCEPT", "REVISE", "REJECT"]
Confidence = Literal["HIGH", "MEDIUM", "LOW"]


@dataclass
class PhysicalCheck:
    term: str
    expected_sign: str
    found_sign: str
    verdict: str


@dataclass
class EtaFormulaVerdict:
    formula: str
    verdict: Verdict
    confidence: Confidence
    summary: str
    physical_checks: list[PhysicalCheck] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)


try:
    _SYSTEM_PROMPT_TEXT: str | None = load_prompt("symbolic_eta_critic")
except PromptNotFoundError:
    _SYSTEM_PROMPT_TEXT = None


_TOOL_SCHEMA = {
    "name": "report_formula_review",
    "description": "Submit peer-review verdict for a symbolic η_Al correction formula",
    "input_schema": {
        "type": "object",
        "properties": {
            "formula": {"type": "string"},
            "verdict": {
                "type": "string",
                "enum": ["ACCEPT", "REVISE", "REJECT"],
            },
            "confidence": {
                "type": "string",
                "enum": ["HIGH", "MEDIUM", "LOW"],
            },
            "summary": {"type": "string"},
            "physical_checks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string"},
                        "expected_sign": {"type": "string"},
                        "found_sign": {"type": "string"},
                        "verdict": {"type": "string"},
                    },
                    "required": ["term", "expected_sign", "found_sign", "verdict"],
                },
            },
            "concerns": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "formula", "verdict", "confidence", "summary",
            "physical_checks", "concerns",
        ],
    },
}


def _build_user_payload(
    formula_infix: str, r2: float, complexity: int, feature_names: list[str]
) -> str:
    payload = {
        "formula_infix": formula_infix,
        "r2": r2,
        "complexity": complexity,
        "feature_names": feature_names,
    }
    return (
        "Closed-form correction factor на peer-review:\n```json\n"
        + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        + "\n```"
    )


class SymbolicEtaCritic:
    MODEL_ID = "claude-sonnet-4-6"
    MAX_TOKENS = 2048
    TIMEOUT_S = 120.0

    def __init__(self, client: Any, model: str | None = None):
        self.client = client
        self.model = model or self.MODEL_ID

    def review_formula(
        self,
        formula_infix: str,
        r2: float,
        complexity: int,
        feature_names: list[str],
    ) -> EtaFormulaVerdict | None:
        user_payload = _build_user_payload(
            formula_infix, r2, complexity, feature_names
        )
        start = time.monotonic()
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.MAX_TOKENS,
                system=[{
                    "type": "text",
                    "text": _SYSTEM_PROMPT_TEXT,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=[_TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": "report_formula_review"},
                messages=[{"role": "user", "content": user_payload}],
                timeout=self.TIMEOUT_S,
            )
        except Exception as e:
            logger.warning("SymbolicEtaCritic API call failed: %s", e)
            return None

        elapsed = time.monotonic() - start
        tool_block = next(
            (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            logger.warning("SymbolicEtaCritic: no tool_use in response")
            return None

        try:
            r = tool_block.input
            checks = [
                PhysicalCheck(
                    term=c["term"],
                    expected_sign=c["expected_sign"],
                    found_sign=c["found_sign"],
                    verdict=c["verdict"],
                )
                for c in r.get("physical_checks", [])
            ]
            verdict = EtaFormulaVerdict(
                formula=r["formula"],
                verdict=r["verdict"],
                confidence=r["confidence"],
                summary=r["summary"],
                physical_checks=checks,
                concerns=list(r.get("concerns", [])),
            )
        except (KeyError, TypeError) as e:
            logger.warning("SymbolicEtaCritic: bad payload: %s", e)
            return None

        _log_usage(resp, elapsed, verdict)
        return verdict


def _log_usage(resp, elapsed_s, verdict):
    from decision_log.logger import log_decision
    usage = resp.usage
    log_decision(
        phase="deoxidation",
        decision=(
            f"SymbolicEtaCritic: {verdict.verdict} "
            f"(conf {verdict.confidence})"
        ),
        reasoning=(
            f"Model={getattr(resp, 'model', 'unknown')}, "
            f"input={getattr(usage, 'input_tokens', 0)} "
            f"(cache_read={getattr(usage, 'cache_read_input_tokens', 0)}, "
            f"cache_create={getattr(usage, 'cache_creation_input_tokens', 0)}), "
            f"output={getattr(usage, 'output_tokens', 0)}, "
            f"latency={elapsed_s:.2f}s"
        ),
        context={"review": asdict(verdict)},
        author="symbolic_eta_critic",
        tags=["symbolic_eta_critic", "sonnet-4-6"],
    )


def make_symbolic_eta_critic() -> SymbolicEtaCritic | None:
    if _SYSTEM_PROMPT_TEXT is None:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("anthropic missing — SymbolicEtaCritic disabled")
        return None
    return SymbolicEtaCritic(client=Anthropic())
