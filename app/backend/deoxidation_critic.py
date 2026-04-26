"""
Deoxidation Critic — adversarial PhD peer review of ladle deoxidation
advisory. Mirrors recipe_critic architecture.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Literal

from app.backend.prompt_loader import load_prompt_optional

logger = logging.getLogger(__name__)

Verdict = Literal["ACCEPT", "REVISE", "REJECT"]
Confidence = Literal["HIGH", "MEDIUM", "LOW"]
EvidenceVerdict = Literal["VALID", "INVALID", "UNVERIFIABLE"]


@dataclass
class EvidenceCheck:
    claim: str
    verdict: EvidenceVerdict
    note: str


@dataclass
class AdvisoryVerdict:
    advisory_id: str
    verdict: Verdict
    confidence: Confidence
    summary: str
    evidence_check: list[EvidenceCheck] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    suggested_revision: str | None = None


_SYSTEM_PROMPT_TEXT = load_prompt_optional("deoxidation_critic")


_TOOL_SCHEMA = {
    "name": "report_advisory_review",
    "description": "Submit peer-review verdict for deoxidation advisory",
    "input_schema": {
        "type": "object",
        "properties": {
            "advisory_id": {"type": "string"},
            "verdict": {
                "type": "string",
                "enum": ["ACCEPT", "REVISE", "REJECT"],
            },
            "confidence": {
                "type": "string",
                "enum": ["HIGH", "MEDIUM", "LOW"],
            },
            "summary": {"type": "string"},
            "evidence_check": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "verdict": {
                            "type": "string",
                            "enum": ["VALID", "INVALID", "UNVERIFIABLE"],
                        },
                        "note": {"type": "string"},
                    },
                    "required": ["claim", "verdict", "note"],
                },
            },
            "strengths": {
                "type": "array",
                "items": {"type": "string"},
            },
            "weaknesses": {
                "type": "array",
                "items": {"type": "string"},
            },
            "suggested_revision": {"type": ["string", "null"]},
        },
        "required": [
            "advisory_id", "verdict", "confidence",
            "summary", "evidence_check", "strengths", "weaknesses",
        ],
    },
}


def _build_user_payload(ctx: dict, advisory: dict) -> str:
    payload = {
        "heat_context": ctx.get("heat_context", {}),
        "thermo_estimates": ctx.get("thermo_estimates", {}),
        "advisory_to_review": advisory,
    }
    return (
        "Контекст плавки + advisory на peer-review:\n```json\n"
        + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        + "\n```"
    )


class DeoxidationCritic:
    MODEL_ID = "claude-sonnet-4-6"
    MAX_TOKENS = 4096
    TIMEOUT_S = 120.0

    def __init__(self, client: Any, model: str | None = None):
        self.client = client
        self.model = model or self.MODEL_ID

    def review(self, ctx: dict, advisory: dict) -> AdvisoryVerdict | None:
        user_payload = _build_user_payload(ctx, advisory)
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
                tool_choice={"type": "tool", "name": "report_advisory_review"},
                messages=[{"role": "user", "content": user_payload}],
                timeout=self.TIMEOUT_S,
            )
        except Exception as e:
            logger.warning("DeoxidationCritic API call failed: %s", e)
            return None

        elapsed = time.monotonic() - start
        tool_block = next(
            (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            logger.warning("DeoxidationCritic: no tool_use in response")
            return None

        try:
            r = tool_block.input
            ec = [
                EvidenceCheck(claim=e["claim"], verdict=e["verdict"], note=e["note"])
                for e in r.get("evidence_check", [])
            ]
            verdict = AdvisoryVerdict(
                advisory_id=r["advisory_id"],
                verdict=r["verdict"],
                confidence=r["confidence"],
                summary=r["summary"],
                evidence_check=ec,
                strengths=list(r.get("strengths", [])),
                weaknesses=list(r.get("weaknesses", [])),
                suggested_revision=r.get("suggested_revision"),
            )
        except (KeyError, TypeError) as e:
            logger.warning("DeoxidationCritic: bad payload: %s", e)
            return None

        _log_usage(resp, elapsed, verdict, ctx)
        return verdict


def _log_usage(resp, elapsed_s, verdict, ctx):
    from decision_log.logger import log_decision
    usage = resp.usage
    log_decision(
        phase="deoxidation",
        decision=(
            f"DeoxidationCritic: {verdict.verdict} "
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
        context={
            "review": asdict(verdict),
            "heat_context": ctx.get("heat_context"),
        },
        author="deoxidation_critic",
        tags=["deoxidation_review", "sonnet-4-6"],
    )


def make_deoxidation_critic() -> DeoxidationCritic | None:
    if _SYSTEM_PROMPT_TEXT is None:
        return None  # prompt missing on public clone
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("anthropic missing — DeoxidationCritic disabled")
        return None
    return DeoxidationCritic(client=Anthropic())
