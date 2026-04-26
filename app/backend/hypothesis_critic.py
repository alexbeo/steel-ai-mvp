"""
Hypothesis Critic — adversarial peer review of LLM-generated hypotheses.

Sits between hypothesis_generator and the user. For each generated
hypothesis, asks Claude (as anonymous PhD reviewer in physical metallurgy
+ applied statistics) to produce a structured verdict:
  ACCEPT / REVISE / REJECT + confidence + summary + strengths +
  weaknesses + suggested_revision.

Architecturally mirrors hypothesis_generator: ANTHROPIC_API_KEY gates
activation, prompt loaded from gitignored prompts/ directory, structured
output via tool_use, usage and verdicts persisted to Decision Log under
tag "hypothesis_review".

Independence note: the critic gets the same artifact summary the
generator saw, plus the list of generated hypotheses. It does NOT see
the generator's system prompt or its private chain of thought, only the
final hypothesis output. This produces real peer review rather than
self-critique.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Literal

from app.backend.prompt_loader import load_prompt_optional

logger = logging.getLogger(__name__)

Verdict = Literal["ACCEPT", "REVISE", "REJECT"]
Confidence = Literal["HIGH", "MEDIUM", "LOW"]


@dataclass
class CriticVerdict:
    hypothesis_id: str
    verdict: Verdict
    confidence: Confidence
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    suggested_revision: str | None = None


_SYSTEM_PROMPT_TEXT = load_prompt_optional("hypothesis_critic")


_TOOL_SCHEMA = {
    "name": "report_reviews",
    "description": (
        "Submit structured peer-review verdicts for each hypothesis"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "hypothesis_id": {"type": "string"},
                        "verdict": {
                            "type": "string",
                            "enum": ["ACCEPT", "REVISE", "REJECT"],
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["HIGH", "MEDIUM", "LOW"],
                        },
                        "summary": {"type": "string"},
                        "strengths": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "weaknesses": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "suggested_revision": {
                            "type": ["string", "null"],
                        },
                    },
                    "required": [
                        "hypothesis_id", "verdict", "confidence",
                        "summary", "strengths", "weaknesses",
                    ],
                },
            },
        },
        "required": ["reviews"],
    },
}


def _build_user_payload(artifact_ctx: dict, hypotheses: list[dict]) -> str:
    """Compose the reviewer's input: artifact summary + hypothesis list.

    Strips internal model_version / tags from each hypothesis before
    showing — reviewer doesn't need provenance, only content.
    """
    importance = artifact_ctx.get("feature_importance") or {}
    top10 = dict(sorted(importance.items(), key=lambda kv: -kv[1])[:10])

    artifact_summary = {
        "steel_class": artifact_ctx.get("steel_class"),
        "target": artifact_ctx.get("target"),
        "metrics": {
            "r2_test": artifact_ctx.get("r2_test"),
            "mae_test": artifact_ctx.get("mae_test"),
            "rmse_test": artifact_ctx.get("rmse_test"),
            "coverage_90_ci": artifact_ctx.get("coverage_90_ci"),
            "conformal_correction_mpa": artifact_ctx.get(
                "conformal_correction_mpa"
            ),
        },
        "n_train": artifact_ctx.get("n_train"),
        "n_val": artifact_ctx.get("n_val"),
        "n_test": artifact_ctx.get("n_test"),
        "feature_importance_top10": top10,
        "training_ranges": artifact_ctx.get("training_ranges") or {},
        "target_distribution": artifact_ctx.get("target_distribution") or {},
        "sample_predictions": artifact_ctx.get("sample_predictions") or [],
    }

    hypotheses_clean = []
    for h in hypotheses:
        hypotheses_clean.append({
            "id": h.get("id"),
            "statement": h.get("statement"),
            "rationale": h.get("rationale"),
            "proposed_experiment": h.get("proposed_experiment"),
            "expected_outcome": h.get("expected_outcome"),
            "novelty": h.get("novelty"),
            "experiment_cost_estimate": h.get("experiment_cost_estimate"),
            "economic_impact": h.get("economic_impact"),
        })

    return (
        "Сводка артефакта модели:\n```json\n"
        + json.dumps(artifact_summary, indent=2, ensure_ascii=False, default=str)
        + "\n```\n\n"
        "Гипотезы на review:\n```json\n"
        + json.dumps(hypotheses_clean, indent=2, ensure_ascii=False, default=str)
        + "\n```"
    )


class HypothesisCritic:
    MODEL_ID = "claude-sonnet-4-6"
    MAX_TOKENS = 6144
    TIMEOUT_S = 180.0

    def __init__(self, client: Any, model: str | None = None):
        self.client = client
        self.model = model or self.MODEL_ID

    def review(
        self,
        artifact_ctx: dict,
        hypotheses: list[dict],
    ) -> list[CriticVerdict]:
        if not hypotheses:
            return []

        user_payload = _build_user_payload(artifact_ctx, hypotheses)
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
                tool_choice={"type": "tool", "name": "report_reviews"},
                messages=[{"role": "user", "content": user_payload}],
                timeout=self.TIMEOUT_S,
            )
        except Exception as e:
            logger.warning("HypothesisCritic API call failed: %s", e)
            return []

        elapsed = time.monotonic() - start

        tool_block = next(
            (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            logger.warning("HypothesisCritic: no tool_use in response")
            return []

        try:
            raw = tool_block.input["reviews"]
        except (KeyError, TypeError) as e:
            stop_reason = getattr(resp, "stop_reason", "unknown")
            shape = (
                list(tool_block.input.keys())
                if isinstance(tool_block.input, dict)
                else type(tool_block.input).__name__
            )
            logger.warning(
                "HypothesisCritic: bad payload shape: %s "
                "(stop_reason=%s, tool_input_keys=%s)",
                e, stop_reason, shape,
            )
            return []

        verdicts: list[CriticVerdict] = []
        for r in raw:
            try:
                verdicts.append(CriticVerdict(
                    hypothesis_id=r["hypothesis_id"],
                    verdict=r["verdict"],
                    confidence=r["confidence"],
                    summary=r["summary"],
                    strengths=list(r.get("strengths", [])),
                    weaknesses=list(r.get("weaknesses", [])),
                    suggested_revision=r.get("suggested_revision"),
                ))
            except (KeyError, TypeError) as e:
                logger.warning("HypothesisCritic: skipping malformed review: %s", e)
                continue

        _log_usage(resp, elapsed, verdicts, artifact_ctx)
        return verdicts


def _log_usage(
    resp: Any,
    elapsed_s: float,
    verdicts: list[CriticVerdict],
    artifact_ctx: dict,
) -> None:
    from decision_log.logger import log_decision

    usage = resp.usage
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    cache_read = getattr(usage, "cache_read_input_tokens", 0)
    cache_create = getattr(usage, "cache_creation_input_tokens", 0)
    model = getattr(resp, "model", "unknown")

    verdict_counts = {"ACCEPT": 0, "REVISE": 0, "REJECT": 0}
    for v in verdicts:
        verdict_counts[v.verdict] = verdict_counts.get(v.verdict, 0) + 1

    log_decision(
        phase="training",
        decision=(
            f"HypothesisCritic: {len(verdicts)} reviews "
            f"(A={verdict_counts['ACCEPT']}, "
            f"R={verdict_counts['REVISE']}, "
            f"X={verdict_counts['REJECT']})"
        ),
        reasoning=(
            f"Model={model}, "
            f"input={input_tokens} (cache_read={cache_read}, "
            f"cache_create={cache_create}), "
            f"output={output_tokens}, latency={elapsed_s:.2f}s"
        ),
        context={
            "reviews": [asdict(v) for v in verdicts],
            "model_version": artifact_ctx.get("model_version"),
            "steel_class": artifact_ctx.get("steel_class"),
            "verdict_counts": verdict_counts,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read": cache_read,
                "cache_create": cache_create,
                "latency_s": round(elapsed_s, 2),
            },
        },
        author="hypothesis_critic",
        tags=["hypothesis_review", "sonnet-4-6"],
    )


def make_hypothesis_critic() -> HypothesisCritic | None:
    if _SYSTEM_PROMPT_TEXT is None:
        return None  # prompt missing on public clone
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("anthropic package missing — HypothesisCritic disabled")
        return None
    return HypothesisCritic(client=Anthropic())
