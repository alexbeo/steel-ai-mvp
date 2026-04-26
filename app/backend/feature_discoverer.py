"""
Feature Discoverer — A1 from AI integration roadmap.

Asks Claude to propose new derived features (ratios, interactions,
transforms, binarizations, aggregates) over an existing dataset's
columns. Each proposal is a `pandas.DataFrame.eval()` formula plus
metadata. Verification is empirical: a separate evaluator applies each
formula, retrains the model with the new feature added, and measures
R² uplift on the test fold.

Architecturally mirrors hypothesis_generator and hypothesis_critic:
ANTHROPIC_API_KEY gates activation, prompt loaded from gitignored
prompts/, structured output via tool_use, persistence in Decision Log
under tag "feature_discovery".

Safety: only `pandas.DataFrame.eval()` is used to apply formulas — no
arbitrary Python execution. The DSL accepts arithmetic + log/sqrt/exp/
abs/where functions over whitelisted column names.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.backend.prompt_loader import load_prompt_optional

logger = logging.getLogger(__name__)

MechanismClass = Literal[
    "ratio", "interaction", "transform", "binarization", "aggregate",
]


@dataclass
class FeatureProposal:
    id: str
    name: str
    formula: str
    mechanism_class: MechanismClass
    rationale: str
    expected_uplift: str
    risk_notes: str
    tags: list[str] = field(default_factory=list)


_SYSTEM_PROMPT_TEXT = load_prompt_optional("feature_discoverer")


_TOOL_SCHEMA = {
    "name": "report_features",
    "description": (
        "Submit derived-feature proposals to add to the dataset"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "formula": {"type": "string"},
                        "mechanism_class": {
                            "type": "string",
                            "enum": [
                                "ratio", "interaction", "transform",
                                "binarization", "aggregate",
                            ],
                        },
                        "rationale": {"type": "string"},
                        "expected_uplift": {"type": "string"},
                        "risk_notes": {"type": "string"},
                    },
                    "required": [
                        "name", "formula", "mechanism_class",
                        "rationale", "expected_uplift", "risk_notes",
                    ],
                },
                "minItems": 0,
                "maxItems": 5,
            },
        },
        "required": ["features"],
    },
}


def _build_user_payload(ctx: dict) -> str:
    importance = ctx.get("feature_importance") or {}
    top10 = dict(sorted(importance.items(), key=lambda kv: -kv[1])[:10])
    payload = {
        "steel_class": ctx.get("steel_class"),
        "target": ctx.get("target"),
        "available_columns": ctx.get("available_columns", []),
        "current_feature_set": ctx.get("current_feature_set", []),
        "metrics": {
            "r2_test": ctx.get("r2_test"),
            "mae_test": ctx.get("mae_test"),
        },
        "feature_importance_top10": top10,
        "training_ranges": ctx.get("training_ranges") or {},
        "n_train": ctx.get("n_train"),
        "n_test": ctx.get("n_test"),
    }
    return (
        "Артефакт обученной модели и список колонок датасета:\n```json\n"
        + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        + "\n```"
    )


# ---------------------------------------------------------------------------
# Safe formula evaluator (pandas.DataFrame.eval wrapper)
# ---------------------------------------------------------------------------

class FormulaError(Exception):
    pass


def apply_formula(
    df: pd.DataFrame, formula: str, name: str,
) -> pd.Series:
    """Evaluate a single formula via pandas.DataFrame.eval and return the
    resulting Series. Raises FormulaError on any failure mode:
    - syntax error in the formula
    - reference to an unknown column
    - resulting Series contains NaN or inf

    pandas.DataFrame.eval is a restricted evaluator (NumExpr backend);
    it cannot execute arbitrary Python. The whitelist is enforced by
    pandas itself: only arithmetic and a small set of math functions
    work, and only existing columns are accessible.
    """
    try:
        result = df.eval(formula)
    except Exception as e:
        raise FormulaError(
            f"formula '{formula}' (name={name}) failed: {e}"
        ) from e
    if not isinstance(result, pd.Series):
        raise FormulaError(
            f"formula '{formula}' (name={name}) returned non-Series "
            f"({type(result).__name__})"
        )
    if result.isna().any():
        n_nan = int(result.isna().sum())
        raise FormulaError(
            f"formula '{formula}' (name={name}) produced {n_nan} NaN values"
        )
    if np.isinf(result).any():
        n_inf = int(np.isinf(result).sum())
        raise FormulaError(
            f"formula '{formula}' (name={name}) produced {n_inf} inf values"
        )
    return result.rename(name)


# ---------------------------------------------------------------------------
# LLM client wrapper
# ---------------------------------------------------------------------------

class FeatureDiscoverer:
    MODEL_ID = "claude-sonnet-4-6"
    MAX_TOKENS = 4096
    TIMEOUT_S = 120.0

    def __init__(self, client: Any, model: str | None = None):
        self.client = client
        self.model = model or self.MODEL_ID

    def discover(self, context: dict) -> list[FeatureProposal]:
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
                tool_choice={"type": "tool", "name": "report_features"},
                messages=[{"role": "user", "content": user_payload}],
                timeout=self.TIMEOUT_S,
            )
        except Exception as e:
            logger.warning("FeatureDiscoverer API call failed: %s", e)
            return []

        elapsed = time.monotonic() - start

        tool_block = next(
            (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            logger.warning("FeatureDiscoverer: no tool_use in response")
            return []

        try:
            raw = tool_block.input["features"]
        except (KeyError, TypeError) as e:
            stop_reason = getattr(resp, "stop_reason", "unknown")
            shape = (
                list(tool_block.input.keys())
                if isinstance(tool_block.input, dict)
                else type(tool_block.input).__name__
            )
            logger.warning(
                "FeatureDiscoverer: bad payload shape: %s "
                "(stop_reason=%s, tool_input_keys=%s)",
                e, stop_reason, shape,
            )
            return []

        proposals: list[FeatureProposal] = []
        for f in raw:
            try:
                proposals.append(FeatureProposal(
                    id=str(uuid.uuid4())[:8],
                    name=f["name"],
                    formula=f["formula"],
                    mechanism_class=f["mechanism_class"],
                    rationale=f["rationale"],
                    expected_uplift=f["expected_uplift"],
                    risk_notes=f["risk_notes"],
                    tags=[
                        f"steel_class:{context.get('steel_class', 'unknown')}",
                        f"target:{context.get('target', 'unknown')}",
                    ],
                ))
            except (KeyError, TypeError) as e:
                logger.warning("FeatureDiscoverer: skipping malformed entry: %s", e)
                continue

        _log_usage(resp, elapsed, proposals, context)
        return proposals


def _log_usage(
    resp: Any,
    elapsed_s: float,
    proposals: list[FeatureProposal],
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
        decision=f"FeatureDiscoverer: {len(proposals)} proposals",
        reasoning=(
            f"Model={model}, "
            f"input={input_tokens} (cache_read={cache_read}, "
            f"cache_create={cache_create}), "
            f"output={output_tokens}, latency={elapsed_s:.2f}s"
        ),
        context={
            "proposals": [asdict(p) for p in proposals],
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
        author="feature_discoverer",
        tags=["feature_discovery", "sonnet-4-6"],
    )


def make_feature_discoverer() -> FeatureDiscoverer | None:
    if _SYSTEM_PROMPT_TEXT is None:
        return None  # prompt missing on public clone
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("anthropic package missing — FeatureDiscoverer disabled")
        return None
    return FeatureDiscoverer(client=Anthropic())
