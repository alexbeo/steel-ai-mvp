"""
Deoxidation Advisor — Sonnet PhD ladle metallurgist поверх 3 термо-моделей.

Превращает 3 thermo числа (Fruehan / Sigworth-Elliott / Hayashi-Yamamoto)
в **полный operator protocol** на ладл-операцию: Al mass + форма
(wire/cube/powder) + addition strategy (rate, timing) + expected
recovery + kinetic timing + risk flags + inclusion forecast +
pre/post actions + evidence + confidence.

Архитектурно мирроринг recipe_designer/critic. ANTHROPIC_API_KEY gate,
prompt из gitignored prompts/, structured tool_use, persistence
в Decision Log под тэгом `deoxidation_advisory`.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import Any, Literal

from app.backend.prompt_loader import load_prompt_optional

logger = logging.getLogger(__name__)

AlForm = Literal["wire", "cube", "powder"]
Confidence = Literal["HIGH", "MEDIUM", "LOW"]


@dataclass
class DeoxidationAdvisory:
    id: str
    summary: str
    al_addition_kg: float
    al_form: AlForm
    addition_strategy: str
    expected_recovery_pct: float
    kinetic_timing_min: list[float]  # [min, max]
    risk_flags: list[str]
    inclusion_forecast: str
    pre_actions: list[str]
    post_actions: list[str]
    model_convergence_note: str
    evidence: list[str]
    confidence: Confidence
    tags: list[str] = field(default_factory=list)


_SYSTEM_PROMPT_TEXT = load_prompt_optional("deoxidation_advisor")


_TOOL_SCHEMA = {
    "name": "report_advisory",
    "description": "Submit full ladle deoxidation protocol",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "al_addition_kg": {"type": "number"},
            "al_form": {
                "type": "string",
                "enum": ["wire", "cube", "powder"],
            },
            "addition_strategy": {"type": "string"},
            "expected_recovery_pct": {"type": "number"},
            "kinetic_timing_min": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2, "maxItems": 2,
            },
            "risk_flags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "inclusion_forecast": {"type": "string"},
            "pre_actions": {
                "type": "array",
                "items": {"type": "string"},
            },
            "post_actions": {
                "type": "array",
                "items": {"type": "string"},
            },
            "model_convergence_note": {"type": "string"},
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
            },
            "confidence": {
                "type": "string",
                "enum": ["HIGH", "MEDIUM", "LOW"],
            },
        },
        "required": [
            "summary", "al_addition_kg", "al_form",
            "addition_strategy", "expected_recovery_pct",
            "kinetic_timing_min", "risk_flags",
            "inclusion_forecast", "pre_actions", "post_actions",
            "model_convergence_note", "evidence", "confidence",
        ],
    },
}


def _build_user_payload(ctx: dict) -> str:
    payload = {
        "heat_context": ctx.get("heat_context", {}),
        "thermo_estimates": ctx.get("thermo_estimates", {}),
    }
    return (
        "Запрос на ladle deoxidation advisory:\n```json\n"
        + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        + "\n```"
    )


class DeoxidationAdvisor:
    MODEL_ID = "claude-sonnet-4-6"
    MAX_TOKENS = 6144
    TIMEOUT_S = 180.0

    def __init__(self, client: Any, model: str | None = None):
        self.client = client
        self.model = model or self.MODEL_ID

    def advise(self, context: dict) -> DeoxidationAdvisory | None:
        user_payload = _build_user_payload(context)
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
                tool_choice={"type": "tool", "name": "report_advisory"},
                messages=[{"role": "user", "content": user_payload}],
                timeout=self.TIMEOUT_S,
            )
        except Exception as e:
            logger.warning("DeoxidationAdvisor API call failed: %s", e)
            return None

        elapsed = time.monotonic() - start
        tool_block = next(
            (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            logger.warning("DeoxidationAdvisor: no tool_use in response")
            return None

        try:
            r = tool_block.input
            advisory = DeoxidationAdvisory(
                id=str(uuid.uuid4())[:8],
                summary=r["summary"],
                al_addition_kg=float(r["al_addition_kg"]),
                al_form=r["al_form"],
                addition_strategy=r["addition_strategy"],
                expected_recovery_pct=float(r["expected_recovery_pct"]),
                kinetic_timing_min=[float(x) for x in r["kinetic_timing_min"]],
                risk_flags=list(r.get("risk_flags", [])),
                inclusion_forecast=r["inclusion_forecast"],
                pre_actions=list(r.get("pre_actions", [])),
                post_actions=list(r.get("post_actions", [])),
                model_convergence_note=r["model_convergence_note"],
                evidence=list(r["evidence"]),
                confidence=r["confidence"],
                tags=["deoxidation"],
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("DeoxidationAdvisor: bad payload: %s", e)
            return None

        _log_usage(resp, elapsed, advisory, context)
        return advisory


def _log_usage(resp, elapsed_s, advisory, ctx):
    from decision_log.logger import log_decision
    usage = resp.usage
    log_decision(
        phase="deoxidation",
        decision=(
            f"DeoxidationAdvisor: Al={advisory.al_addition_kg:.1f} кг "
            f"({advisory.al_form}), conf={advisory.confidence}"
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
            "advisory": asdict(advisory),
            "heat_context": ctx.get("heat_context"),
            "thermo_estimates": ctx.get("thermo_estimates"),
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "cache_read": getattr(usage, "cache_read_input_tokens", 0),
                "cache_create": getattr(usage, "cache_creation_input_tokens", 0),
                "latency_s": round(elapsed_s, 2),
            },
        },
        author="deoxidation_advisor",
        tags=["deoxidation_advisory", "sonnet-4-6"],
    )


def make_deoxidation_advisor() -> DeoxidationAdvisor | None:
    if _SYSTEM_PROMPT_TEXT is None:
        return None  # prompt missing on public clone
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("anthropic missing — DeoxidationAdvisor disabled")
        return None
    return DeoxidationAdvisor(client=Anthropic())
