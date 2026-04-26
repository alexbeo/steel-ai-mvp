"""
Recipe Designer — composition+process selection через Sonnet с PhD-уровневой
доказательной базой.

Sonnet получает baseline рецепт + target task + сводку артефакта модели,
возвращает 3-4 альтернативных рецепта с обязательной двойной evidence
(artifact-data + classical metallurgical mechanism). Каждое
изменение легирующего элемента или процесс-параметра должно быть
обосновано одновременно из двух источников — иначе проектировщик
обязан НЕ менять.

Архитектурно мирроринг hypothesis_generator/critic: ANTHROPIC_API_KEY
gates, prompt из gitignored prompts/recipe_designer.md, structured
output, persistence в decision_log под тагом "recipe_design".

Pipeline downstream:
  recipe_designer → ml+cost numerical verification → recipe_critic
                        ↓                                    ↓
                  TruthGate (XGBoost predict + cost)   PhD adversarial
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


@dataclass
class CompositionRecipe:
    id: str
    name: str
    composition: dict[str, float]
    process_params: dict[str, float]
    rationale: str
    evidence: list[str]
    expected_outcome: str
    risk_notes: str
    novelty: Novelty
    tags: list[str] = field(default_factory=list)


_SYSTEM_PROMPT_TEXT = load_prompt("recipe_designer")


_TOOL_SCHEMA = {
    "name": "report_recipes",
    "description": (
        "Submit composition+process recipes with evidence-grounded rationale"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "recipes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "composition": {"type": "object"},
                        "process_params": {"type": "object"},
                        "rationale": {"type": "string"},
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                        },
                        "expected_outcome": {"type": "string"},
                        "risk_notes": {"type": "string"},
                        "novelty": {
                            "type": "string",
                            "enum": ["LOW", "MEDIUM", "HIGH"],
                        },
                    },
                    "required": [
                        "name", "composition", "process_params",
                        "rationale", "evidence",
                        "expected_outcome", "risk_notes", "novelty",
                    ],
                },
                "minItems": 0,
                "maxItems": 4,
            },
        },
        "required": ["recipes"],
    },
}


def _build_user_payload(ctx: dict) -> str:
    importance = ctx.get("feature_importance") or {}
    top10 = dict(sorted(importance.items(), key=lambda kv: -kv[1])[:10])
    payload = {
        "task": ctx.get("task", "повысить целевое свойство и/или снизить cost vs baseline"),
        "steel_class": ctx.get("steel_class"),
        "target": ctx.get("target"),
        "data_source": ctx.get("data_source"),
        "metrics": {
            "r2_test": ctx.get("r2_test"),
            "mae_test": ctx.get("mae_test"),
            "coverage_90_ci": ctx.get("coverage_90_ci"),
            "conformal_correction_mpa": ctx.get("conformal_correction_mpa"),
        },
        "baseline_recipe": ctx.get("baseline_recipe", {}),
        "baseline_predicted_property": ctx.get("baseline_predicted_property"),
        "baseline_cost_per_ton": ctx.get("baseline_cost_per_ton"),
        "available_composition": ctx.get("available_composition", []),
        "available_process": ctx.get("available_process", []),
        "training_ranges": ctx.get("training_ranges") or {},
        "feature_importance_top10": top10,
        "target_distribution": ctx.get("target_distribution") or {},
    }
    return (
        "Задача проектирования рецепта:\n```json\n"
        + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        + "\n```"
    )


class RecipeDesigner:
    MODEL_ID = "claude-sonnet-4-6"
    MAX_TOKENS = 8192
    TIMEOUT_S = 180.0

    def __init__(self, client: Any, model: str | None = None):
        self.client = client
        self.model = model or self.MODEL_ID

    def design(self, context: dict) -> list[CompositionRecipe]:
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
                tool_choice={"type": "tool", "name": "report_recipes"},
                messages=[{"role": "user", "content": user_payload}],
                timeout=self.TIMEOUT_S,
            )
        except Exception as e:
            logger.warning("RecipeDesigner API call failed: %s", e)
            return []

        elapsed = time.monotonic() - start
        tool_block = next(
            (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            logger.warning("RecipeDesigner: no tool_use in response")
            return []

        try:
            raw = tool_block.input["recipes"]
        except (KeyError, TypeError) as e:
            stop_reason = getattr(resp, "stop_reason", "unknown")
            logger.warning(
                "RecipeDesigner: bad payload shape: %s (stop_reason=%s)",
                e, stop_reason,
            )
            return []

        recipes: list[CompositionRecipe] = []
        for r in raw:
            try:
                recipes.append(CompositionRecipe(
                    id=str(uuid.uuid4())[:8],
                    name=r["name"],
                    composition={k: float(v) for k, v in r["composition"].items()},
                    process_params={k: float(v) for k, v in r["process_params"].items()},
                    rationale=r["rationale"],
                    evidence=list(r["evidence"]),
                    expected_outcome=r["expected_outcome"],
                    risk_notes=r["risk_notes"],
                    novelty=r["novelty"],
                    tags=[
                        f"steel_class:{context.get('steel_class', 'unknown')}",
                        f"target:{context.get('target', 'unknown')}",
                    ],
                ))
            except (KeyError, TypeError, ValueError) as e:
                logger.warning("RecipeDesigner: skipping malformed: %s", e)
                continue

        _log_usage(resp, elapsed, recipes, context)
        return recipes


def _log_usage(resp, elapsed_s, recipes, context):
    from decision_log.logger import log_decision
    usage = resp.usage
    log_decision(
        phase="inverse_design",
        decision=f"RecipeDesigner: {len(recipes)} recipes",
        reasoning=(
            f"Model={getattr(resp, 'model', 'unknown')}, "
            f"input={getattr(usage, 'input_tokens', 0)} "
            f"(cache_read={getattr(usage, 'cache_read_input_tokens', 0)}, "
            f"cache_create={getattr(usage, 'cache_creation_input_tokens', 0)}), "
            f"output={getattr(usage, 'output_tokens', 0)}, "
            f"latency={elapsed_s:.2f}s"
        ),
        context={
            "recipes": [asdict(r) for r in recipes],
            "model_version": context.get("model_version"),
            "steel_class": context.get("steel_class"),
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "cache_read": getattr(usage, "cache_read_input_tokens", 0),
                "cache_create": getattr(usage, "cache_creation_input_tokens", 0),
                "latency_s": round(elapsed_s, 2),
            },
        },
        author="recipe_designer",
        tags=["recipe_design", "sonnet-4-6"],
    )


def make_recipe_designer() -> RecipeDesigner | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("anthropic package missing — RecipeDesigner disabled")
        return None
    return RecipeDesigner(client=Anthropic())
