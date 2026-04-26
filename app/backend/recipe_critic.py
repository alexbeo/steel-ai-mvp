"""
Recipe Critic — adversarial PhD peer review of LLM-designed composition
recipes, including evidence-base fact-check.

Sits between recipe_designer + ML-numerical truth gate, and the user.
Получает: артефакт сводку, рецепты с заполненной evidence, и ML-cost
verification (predicted property + 90% CI + cost + Δ vs baseline).
Возвращает structured verdict с явным fact-check каждой строки
evidence.

Архитектурно мирроринг hypothesis_critic (вердикт ACCEPT/REVISE/REJECT
+ confidence + summary + strengths + weaknesses + suggested_revision),
плюс уникальное поле evidence_check — построчная проверка цитат на
artifact-data (числа должны совпадать) и mechanism-claims (закон должен
быть применим в этой композиционной зоне).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Literal

from app.backend.prompt_loader import load_prompt

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
class RecipeVerdict:
    recipe_id: str
    verdict: Verdict
    confidence: Confidence
    summary: str
    evidence_check: list[EvidenceCheck] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    suggested_revision: str | None = None


_SYSTEM_PROMPT_TEXT = load_prompt("recipe_critic")


_TOOL_SCHEMA = {
    "name": "report_recipe_reviews",
    "description": "Submit structured peer-review verdicts for each recipe",
    "input_schema": {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "recipe_id": {"type": "string"},
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
                        "recipe_id", "verdict", "confidence",
                        "summary", "evidence_check",
                        "strengths", "weaknesses",
                    ],
                },
            },
        },
        "required": ["reviews"],
    },
}


def _build_user_payload(
    artifact_ctx: dict,
    recipes_with_verification: list[dict],
) -> str:
    importance = artifact_ctx.get("feature_importance") or {}
    top10 = dict(sorted(importance.items(), key=lambda kv: -kv[1])[:10])
    artifact_summary = {
        "steel_class": artifact_ctx.get("steel_class"),
        "target": artifact_ctx.get("target"),
        "metrics": {
            "r2_test": artifact_ctx.get("r2_test"),
            "mae_test": artifact_ctx.get("mae_test"),
            "coverage_90_ci": artifact_ctx.get("coverage_90_ci"),
            "conformal_correction_mpa": artifact_ctx.get("conformal_correction_mpa"),
        },
        "n_train": artifact_ctx.get("n_train"),
        "feature_importance_top10": top10,
        "training_ranges": artifact_ctx.get("training_ranges") or {},
        "target_distribution": artifact_ctx.get("target_distribution") or {},
        "baseline_recipe": artifact_ctx.get("baseline_recipe", {}),
        "baseline_predicted_property": artifact_ctx.get("baseline_predicted_property"),
        "baseline_cost_per_ton": artifact_ctx.get("baseline_cost_per_ton"),
    }
    return (
        "Сводка артефакта:\n```json\n"
        + json.dumps(artifact_summary, indent=2, ensure_ascii=False, default=str)
        + "\n```\n\nРецепты на review (с ML+cost верификацией):\n```json\n"
        + json.dumps(recipes_with_verification, indent=2, ensure_ascii=False, default=str)
        + "\n```"
    )


class RecipeCritic:
    MODEL_ID = "claude-sonnet-4-6"
    MAX_TOKENS = 8192
    TIMEOUT_S = 180.0

    def __init__(self, client: Any, model: str | None = None):
        self.client = client
        self.model = model or self.MODEL_ID

    def review(
        self,
        artifact_ctx: dict,
        recipes_with_verification: list[dict],
    ) -> list[RecipeVerdict]:
        if not recipes_with_verification:
            return []
        user_payload = _build_user_payload(artifact_ctx, recipes_with_verification)
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
                tool_choice={"type": "tool", "name": "report_recipe_reviews"},
                messages=[{"role": "user", "content": user_payload}],
                timeout=self.TIMEOUT_S,
            )
        except Exception as e:
            logger.warning("RecipeCritic API call failed: %s", e)
            return []

        elapsed = time.monotonic() - start
        tool_block = next(
            (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            logger.warning("RecipeCritic: no tool_use in response")
            return []

        try:
            raw = tool_block.input["reviews"]
        except (KeyError, TypeError) as e:
            stop_reason = getattr(resp, "stop_reason", "unknown")
            logger.warning(
                "RecipeCritic: bad payload shape: %s (stop_reason=%s)",
                e, stop_reason,
            )
            return []

        verdicts: list[RecipeVerdict] = []
        for r in raw:
            try:
                ec = [
                    EvidenceCheck(claim=e["claim"], verdict=e["verdict"], note=e["note"])
                    for e in r.get("evidence_check", [])
                ]
                verdicts.append(RecipeVerdict(
                    recipe_id=r["recipe_id"],
                    verdict=r["verdict"],
                    confidence=r["confidence"],
                    summary=r["summary"],
                    evidence_check=ec,
                    strengths=list(r.get("strengths", [])),
                    weaknesses=list(r.get("weaknesses", [])),
                    suggested_revision=r.get("suggested_revision"),
                ))
            except (KeyError, TypeError) as e:
                logger.warning("RecipeCritic: skipping malformed: %s", e)
                continue

        _log_usage(resp, elapsed, verdicts, artifact_ctx)
        return verdicts


def _log_usage(resp, elapsed_s, verdicts, ctx):
    from decision_log.logger import log_decision
    usage = resp.usage
    counts = {"ACCEPT": 0, "REVISE": 0, "REJECT": 0}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    log_decision(
        phase="inverse_design",
        decision=(
            f"RecipeCritic: {len(verdicts)} reviews "
            f"(A={counts['ACCEPT']}, R={counts['REVISE']}, X={counts['REJECT']})"
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
            "reviews": [asdict(v) for v in verdicts],
            "model_version": ctx.get("model_version"),
            "verdict_counts": counts,
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "cache_read": getattr(usage, "cache_read_input_tokens", 0),
                "cache_create": getattr(usage, "cache_creation_input_tokens", 0),
                "latency_s": round(elapsed_s, 2),
            },
        },
        author="recipe_critic",
        tags=["recipe_review", "sonnet-4-6"],
    )


def make_recipe_critic() -> RecipeCritic | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("anthropic package missing — RecipeCritic disabled")
        return None
    return RecipeCritic(client=Anthropic())
