"""
Anomaly Explainer — A3 final piece of AI integration roadmap.

Когда OOD-detector (GaussianMixture) флагает recipe или какая-то фича
выходит за training_ranges — Sonnet даёт diagnosis в стиле PhD-
металлурга: какие фичи аномальны, какие mechanism risks, что
произойдёт в производстве, как скорректировать.

Архитектурно мирроринг hypothesis_generator/recipe_designer:
ANTHROPIC_API_KEY gate, prompt из gitignored, structured output,
persistence в decision_log под тагом `anomaly_explanation`.
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

DeviationKind = Literal[
    "out_of_range_high", "out_of_range_low",
    "unusual_combination", "extreme_within_range",
]
Severity = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass
class AnomalousFeature:
    feature: str
    value: float
    training_range: list[float]
    deviation_kind: DeviationKind
    note: str


@dataclass
class AnomalyExplanation:
    id: str
    summary: str
    anomalous_features: list[AnomalousFeature]
    mechanism_concerns: list[str]
    production_risks: str
    suggested_correction: str
    severity: Severity
    tags: list[str] = field(default_factory=list)


_SYSTEM_PROMPT_TEXT = load_prompt_optional("anomaly_explainer")


_TOOL_SCHEMA = {
    "name": "report_anomaly_explanation",
    "description": "Submit diagnosis of an OOD-flagged recipe",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "anomalous_features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "feature": {"type": "string"},
                        "value": {"type": "number"},
                        "training_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2, "maxItems": 2,
                        },
                        "deviation_kind": {
                            "type": "string",
                            "enum": [
                                "out_of_range_high", "out_of_range_low",
                                "unusual_combination", "extreme_within_range",
                            ],
                        },
                        "note": {"type": "string"},
                    },
                    "required": [
                        "feature", "value", "training_range",
                        "deviation_kind", "note",
                    ],
                },
            },
            "mechanism_concerns": {
                "type": "array",
                "items": {"type": "string"},
            },
            "production_risks": {"type": "string"},
            "suggested_correction": {"type": "string"},
            "severity": {
                "type": "string",
                "enum": ["LOW", "MEDIUM", "HIGH"],
            },
        },
        "required": [
            "summary", "anomalous_features", "mechanism_concerns",
            "production_risks", "suggested_correction", "severity",
        ],
    },
}


def _build_user_payload(ctx: dict) -> str:
    payload = {
        "steel_class": ctx.get("steel_class"),
        "target": ctx.get("target"),
        "recipe": ctx.get("recipe", {}),
        "training_ranges": ctx.get("training_ranges", {}),
        "training_medians": ctx.get("training_medians", {}),
        "ml_prediction": ctx.get("ml_prediction", {}),
        "ood_flag": ctx.get("ood_flag", False),
        "ood_score": ctx.get("ood_score"),
        "ood_threshold": ctx.get("ood_threshold"),
        "out_of_range_features": ctx.get("out_of_range_features", []),
    }
    return (
        "Запрос диагностики OOD-рецепта:\n```json\n"
        + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        + "\n```"
    )


class AnomalyExplainer:
    MODEL_ID = "claude-sonnet-4-6"
    MAX_TOKENS = 4096
    TIMEOUT_S = 120.0

    def __init__(self, client: Any, model: str | None = None):
        self.client = client
        self.model = model or self.MODEL_ID

    def explain(self, context: dict) -> AnomalyExplanation | None:
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
                tool_choice={"type": "tool", "name": "report_anomaly_explanation"},
                messages=[{"role": "user", "content": user_payload}],
                timeout=self.TIMEOUT_S,
            )
        except Exception as e:
            logger.warning("AnomalyExplainer API call failed: %s", e)
            return None

        elapsed = time.monotonic() - start
        tool_block = next(
            (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            logger.warning("AnomalyExplainer: no tool_use in response")
            return None

        try:
            r = tool_block.input
            af = [
                AnomalousFeature(
                    feature=a["feature"],
                    value=float(a["value"]),
                    training_range=[float(x) for x in a["training_range"]],
                    deviation_kind=a["deviation_kind"],
                    note=a["note"],
                )
                for a in r.get("anomalous_features", [])
            ]
            explanation = AnomalyExplanation(
                id=str(uuid.uuid4())[:8],
                summary=r["summary"],
                anomalous_features=af,
                mechanism_concerns=list(r.get("mechanism_concerns", [])),
                production_risks=r["production_risks"],
                suggested_correction=r["suggested_correction"],
                severity=r["severity"],
                tags=[
                    f"steel_class:{context.get('steel_class', 'unknown')}",
                ],
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("AnomalyExplainer: bad payload: %s", e)
            return None

        _log_usage(resp, elapsed, explanation, context)
        return explanation


def _log_usage(resp, elapsed_s, exp, ctx):
    from decision_log.logger import log_decision
    usage = resp.usage
    log_decision(
        phase="validation",
        decision=f"AnomalyExplainer: severity={exp.severity}, n_features={len(exp.anomalous_features)}",
        reasoning=(
            f"Model={getattr(resp, 'model', 'unknown')}, "
            f"input={getattr(usage, 'input_tokens', 0)} "
            f"(cache_read={getattr(usage, 'cache_read_input_tokens', 0)}, "
            f"cache_create={getattr(usage, 'cache_creation_input_tokens', 0)}), "
            f"output={getattr(usage, 'output_tokens', 0)}, "
            f"latency={elapsed_s:.2f}s"
        ),
        context={
            "explanation": asdict(exp),
            "model_version": ctx.get("model_version"),
            "steel_class": ctx.get("steel_class"),
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "cache_read": getattr(usage, "cache_read_input_tokens", 0),
                "cache_create": getattr(usage, "cache_creation_input_tokens", 0),
                "latency_s": round(elapsed_s, 2),
            },
        },
        author="anomaly_explainer",
        tags=["anomaly_explanation", "sonnet-4-6"],
    )


def make_anomaly_explainer() -> AnomalyExplainer | None:
    if _SYSTEM_PROMPT_TEXT is None:
        return None  # prompt missing on public clone
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("anthropic package missing — AnomalyExplainer disabled")
        return None
    return AnomalyExplainer(client=Anthropic())
