"""Heat ETL critic — Sonnet PhD маппит ambiguous Excel-колонки в HeatRecord поля.

Опциональный второй слой поверх детерминированного детектора в heat_history_etl.py.
Активируется когда оператор передаёт `--use-llm-critic` и есть ANTHROPIC_API_KEY +
prompts/heat_etl_critic.md.

Pattern compliance: R-003 Add AI Capability (mirrors recipe_critic / deoxidation_critic).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from app.backend.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

Confidence = Literal["HIGH", "MEDIUM", "LOW"]


@dataclass
class ProposedMapping:
    excel_column: str
    heat_record_field: str
    confidence: Confidence
    reasoning: str


@dataclass
class UnmappedColumn:
    excel_column: str
    reason: str


@dataclass
class EtlMappingProposal:
    mappings: list[ProposedMapping] = field(default_factory=list)
    unmapped: list[UnmappedColumn] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


# Lazy load — if prompt file missing on fresh clone, _SYSTEM_PROMPT_TEXT stays None
# and make_heat_etl_critic() returns None (matching make_recipe_critic semantics).
try:
    _SYSTEM_PROMPT_TEXT: str | None = load_prompt("heat_etl_critic")
except Exception:  # noqa: BLE001 — explicit graceful fallback for fresh clones
    _SYSTEM_PROMPT_TEXT = None
    logger.warning(
        "heat_etl_critic prompt not found — make_heat_etl_critic will return None"
    )


_TOOL_SCHEMA = {
    "name": "propose_etl_mappings",
    "description": (
        "Map unidentified Excel column names to HeatRecord fields using "
        "column name semantics and sample values."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "mappings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "excel_column": {"type": "string"},
                        "heat_record_field": {"type": "string"},
                        "confidence": {
                            "type": "string",
                            "enum": ["HIGH", "MEDIUM", "LOW"],
                        },
                        "reasoning": {"type": "string"},
                    },
                    "required": [
                        "excel_column",
                        "heat_record_field",
                        "confidence",
                        "reasoning",
                    ],
                },
            },
            "unmapped": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "excel_column": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["excel_column", "reason"],
                },
            },
        },
        "required": ["mappings", "unmapped"],
    },
}


def _build_user_payload(unmapped: list[dict]) -> str:
    return (
        "Ambiguous Excel columns for mapping to HeatRecord fields:\n```json\n"
        + json.dumps(unmapped, indent=2, ensure_ascii=False)
        + "\n```"
    )


class HeatEtlCritic:
    MODEL_ID = "claude-sonnet-4-6"
    MAX_TOKENS = 4096
    TIMEOUT_S = 90.0

    def __init__(self, client: Any, model: str | None = None):
        self.client = client
        self.model = model or self.MODEL_ID

    def propose_mappings(
        self, unmapped: list[dict]
    ) -> EtlMappingProposal | None:
        if not unmapped:
            return EtlMappingProposal(mappings=[], unmapped=[], usage={})

        user_payload = _build_user_payload(unmapped)
        start = time.monotonic()
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT_TEXT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[_TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": "propose_etl_mappings"},
                messages=[{"role": "user", "content": user_payload}],
                timeout=self.TIMEOUT_S,
            )
        except Exception as e:  # noqa: BLE001 — graceful fallback for API outages
            logger.warning("HeatEtlCritic API call failed: %s", e)
            return None

        elapsed = time.monotonic() - start
        tool_block = next(
            (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            logger.warning("HeatEtlCritic: no tool_use in response")
            return None

        try:
            raw = tool_block.input
            proposal = EtlMappingProposal(
                mappings=[
                    ProposedMapping(
                        excel_column=m["excel_column"],
                        heat_record_field=m["heat_record_field"],
                        confidence=m["confidence"],
                        reasoning=m["reasoning"],
                    )
                    for m in raw.get("mappings", [])
                ],
                unmapped=[
                    UnmappedColumn(
                        excel_column=u["excel_column"],
                        reason=u["reason"],
                    )
                    for u in raw.get("unmapped", [])
                ],
                usage=_extract_usage(resp, elapsed),
            )
        except (KeyError, TypeError, ValueError) as e:
            stop_reason = getattr(resp, "stop_reason", "unknown")
            logger.warning(
                "HeatEtlCritic: bad payload (%s, stop_reason=%s)", e, stop_reason
            )
            return None

        _log_usage(resp, elapsed, proposal, unmapped)
        return proposal


def _extract_usage(resp, elapsed_s: float) -> dict:
    usage = getattr(resp, "usage", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "cache_read": getattr(usage, "cache_read_input_tokens", 0),
        "cache_create": getattr(usage, "cache_creation_input_tokens", 0),
        "latency_s": round(elapsed_s, 2),
    }


def _log_usage(
    resp, elapsed_s: float, proposal: EtlMappingProposal, ctx: list[dict]
) -> None:
    try:
        from decision_log.logger import log_decision

        usage = getattr(resp, "usage", None)
        log_decision(
            phase="data_acquisition",
            decision=(
                f"HeatEtlCritic mapped {len(proposal.mappings)} columns, "
                f"unmapped {len(proposal.unmapped)}"
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
                "proposal": asdict(proposal),
                "input_columns": ctx,
            },
            author="heat_etl_critic",
            tags=["heat_etl_critic", "sonnet-4-6"],
        )
    except Exception as exc:  # noqa: BLE001 — never block on logging
        logger.warning("Decision Log save failed: %s", exc)


def make_heat_etl_critic() -> HeatEtlCritic | None:
    """Factory — returns None if prompt missing or API key absent (fresh clone safe)."""
    if _SYSTEM_PROMPT_TEXT is None:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("anthropic package missing — HeatEtlCritic disabled")
        return None
    return HeatEtlCritic(client=Anthropic())


__all__ = [
    "ProposedMapping",
    "UnmappedColumn",
    "EtlMappingProposal",
    "HeatEtlCritic",
    "make_heat_etl_critic",
]
