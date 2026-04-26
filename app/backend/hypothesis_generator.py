"""
Hypothesis Generator — A2 from AI integration roadmap.

Given a TrainedModel + dataset statistics + feature importance, asks Claude
Sonnet to formulate testable hypotheses about model behavior:
  «Модель сильно опирается на normalizing_temp при низком C — возможно,
   в этом диапазоне работает другой механизм закалки. Предлагаю
   эксперимент: фиксировать C=0.2, варьировать normalizing_temp от
   850 до 950 °C с шагом 10.»

Each hypothesis carries:
  statement            — what the hypothesis claims about the model / data
  rationale            — what in the artifact suggests it
  proposed_experiment  — concrete composition + processing values to test
  expected_outcome     — what we'd see if the hypothesis holds
  novelty              — LLM self-rated LOW / MEDIUM / HIGH
                         (LOW = generally known to a metallurgist;
                          HIGH = non-obvious, worth a paper)

Activated via ANTHROPIC_API_KEY (same convention as LLM-Critic). Returns
[] on missing key / API failure / malformed payload — never blocks the
pipeline.

Hypotheses are persisted to Decision Log under tag "hypothesis" so that
acceptance / rejection can be tracked over time and inform RLHF-style
improvements later (см. Уровень C из roadmap).
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import Any, Literal

from app.backend.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

Novelty = Literal["LOW", "MEDIUM", "HIGH"]
CostEstimate = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass
class EconomicImpact:
    vs_classical_baseline: str
    estimated_saving: str
    measurement_method: str


@dataclass
class Hypothesis:
    id: str
    statement: str
    rationale: str
    proposed_experiment: dict[str, Any]
    expected_outcome: str
    novelty: Novelty
    experiment_cost_estimate: CostEstimate
    economic_impact: EconomicImpact
    tags: list[str] = field(default_factory=list)


_SYSTEM_PROMPT_TEXT = load_prompt("hypothesis_generator")


_TOOL_SCHEMA = {
    "name": "report_hypotheses",
    "description": (
        "Submit testable hypotheses about model behavior derived from "
        "the training artifact"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "hypotheses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "statement": {"type": "string"},
                        "rationale": {"type": "string"},
                        "proposed_experiment": {
                            "type": "object",
                            "properties": {
                                "fix": {"type": "object"},
                                "sweep": {
                                    "type": "object",
                                    "properties": {
                                        "variable": {"type": "string"},
                                        "range": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "minItems": 2,
                                            "maxItems": 2,
                                        },
                                        "step": {"type": "number"},
                                    },
                                    "required": ["variable", "range"],
                                },
                            },
                            "required": ["fix", "sweep"],
                        },
                        "expected_outcome": {"type": "string"},
                        "novelty": {
                            "type": "string",
                            "enum": ["LOW", "MEDIUM", "HIGH"],
                        },
                        "experiment_cost_estimate": {
                            "type": "string",
                            "enum": ["LOW", "MEDIUM", "HIGH"],
                        },
                        "economic_impact": {
                            "type": "object",
                            "properties": {
                                "vs_classical_baseline": {"type": "string"},
                                "estimated_saving": {"type": "string"},
                                "measurement_method": {"type": "string"},
                            },
                            "required": [
                                "vs_classical_baseline",
                                "estimated_saving",
                                "measurement_method",
                            ],
                        },
                    },
                    "required": [
                        "statement", "rationale",
                        "proposed_experiment",
                        "expected_outcome", "novelty",
                        "experiment_cost_estimate", "economic_impact",
                    ],
                },
                "minItems": 0,
                "maxItems": 5,
            },
            "summary": {"type": "string"},
        },
        "required": ["hypotheses"],
    },
}


def _build_user_payload(ctx: dict) -> str:
    importance = ctx.get("feature_importance") or {}
    top10 = dict(sorted(importance.items(), key=lambda kv: -kv[1])[:10])

    payload = {
        "steel_class": ctx.get("steel_class"),
        "target": ctx.get("target"),
        "data_source": ctx.get("data_source"),
        "metrics": {
            "r2_train": ctx.get("r2_train"),
            "r2_val": ctx.get("r2_val"),
            "r2_test": ctx.get("r2_test"),
            "mae_test": ctx.get("mae_test"),
            "rmse_test": ctx.get("rmse_test"),
            "coverage_90_ci": ctx.get("coverage_90_ci"),
            "conformal_correction_mpa": ctx.get("conformal_correction_mpa"),
        },
        "dataset_size": {
            "n_train": ctx.get("n_train"),
            "n_val": ctx.get("n_val"),
            "n_test": ctx.get("n_test"),
        },
        "feature_importance_top10": top10,
        "training_ranges": ctx.get("training_ranges") or {},
        "target_distribution": ctx.get("target_distribution") or {},
        "sample_predictions": ctx.get("sample_predictions") or [],
    }
    return (
        "Trained-model artifact for hypothesis generation:\n```json\n"
        + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        + "\n```"
    )


class HypothesisGenerator:
    MODEL_ID = "claude-sonnet-4-6"
    MAX_TOKENS = 8192
    TIMEOUT_S = 180.0

    def __init__(self, client: Any, model: str | None = None):
        self.client = client
        self.model = model or self.MODEL_ID

    def generate(self, context: dict) -> list[Hypothesis]:
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
                tool_choice={"type": "tool", "name": "report_hypotheses"},
                messages=[{"role": "user", "content": user_payload}],
                timeout=self.TIMEOUT_S,
            )
        except Exception as e:
            logger.warning("HypothesisGenerator API call failed: %s", e)
            return []

        elapsed = time.monotonic() - start

        tool_block = next(
            (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            logger.warning("HypothesisGenerator: no tool_use in response")
            return []

        try:
            raw = tool_block.input["hypotheses"]
        except (KeyError, TypeError) as e:
            stop_reason = getattr(resp, "stop_reason", "unknown")
            shape = (
                list(tool_block.input.keys())
                if isinstance(tool_block.input, dict)
                else type(tool_block.input).__name__
            )
            logger.warning(
                "HypothesisGenerator: bad payload shape: %s "
                "(stop_reason=%s, tool_input_keys=%s)",
                e, stop_reason, shape,
            )
            return []

        hypotheses: list[Hypothesis] = []
        for h in raw:
            try:
                hypotheses.append(Hypothesis(
                    id=str(uuid.uuid4())[:8],
                    statement=h["statement"],
                    rationale=h["rationale"],
                    proposed_experiment=h["proposed_experiment"],
                    expected_outcome=h["expected_outcome"],
                    novelty=h["novelty"],
                    experiment_cost_estimate=h["experiment_cost_estimate"],
                    economic_impact=EconomicImpact(
                        vs_classical_baseline=h["economic_impact"]["vs_classical_baseline"],
                        estimated_saving=h["economic_impact"]["estimated_saving"],
                        measurement_method=h["economic_impact"]["measurement_method"],
                    ),
                    tags=[
                        f"steel_class:{context.get('steel_class', 'unknown')}",
                        f"target:{context.get('target', 'unknown')}",
                    ],
                ))
            except (KeyError, TypeError) as e:
                logger.warning("HypothesisGenerator: skipping malformed entry: %s", e)
                continue

        _log_usage(resp, elapsed, hypotheses, context)
        return hypotheses


def _log_usage(
    resp: Any,
    elapsed_s: float,
    hypotheses: list[Hypothesis],
    context: dict,
) -> None:
    from decision_log.logger import log_decision

    usage = resp.usage
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    cache_read = getattr(usage, "cache_read_input_tokens", 0)
    cache_create = getattr(usage, "cache_creation_input_tokens", 0)
    model = getattr(resp, "model", "unknown")

    log_decision(
        phase="training",
        decision=f"HypothesisGenerator: {len(hypotheses)} hypotheses",
        reasoning=(
            f"Model={model}, "
            f"input={input_tokens} (cache_read={cache_read}, "
            f"cache_create={cache_create}), "
            f"output={output_tokens}, latency={elapsed_s:.2f}s"
        ),
        context={
            "hypotheses": [asdict(h) for h in hypotheses],
            "model_version": context.get("model_version"),
            "steel_class": context.get("steel_class"),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read": cache_read,
                "cache_create": cache_create,
                "latency_s": round(elapsed_s, 2),
            },
        },
        author="hypothesis_generator",
        tags=["hypothesis", "sonnet-4-6"],
    )


def make_hypothesis_generator() -> HypothesisGenerator | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("anthropic package missing — HypothesisGenerator disabled")
        return None
    return HypothesisGenerator(client=Anthropic())
