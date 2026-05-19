"""High-Al outlier cohort diagnoser — PhD root-cause analysis (PR 12 Block IV).

Находит плавки с аномально высоким расходом Al (vs predicted/percentile) и через
Sonnet ищет общие process-факторы (FeO реокисление, Ca-timing, basicity, etc.).

Discovery tool для R&D. CLI-only. R-003 pattern (load_prompt try/except,
factory gate, structured tool_use, cache_control ephemeral).

Cohort analysis ≠ single-record A3 (anomaly_explainer.py) — отдельный module.
"""
from __future__ import annotations

import json
import logging
import os
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.backend.prompt_loader import PromptNotFoundError, load_prompt

logger = logging.getLogger(__name__)

Confidence = Literal["HIGH", "MEDIUM", "LOW"]


@dataclass
class AlOverHypothesis:
    root_cause: str
    evidence_features: list[str]
    mechanism: str
    est_excess_al_pct: float
    confidence: Confidence
    suggested_action: str


@dataclass
class HighAlDiagnosis:
    summary: str
    hypotheses: list[AlOverHypothesis]
    cohort_severity: str
    n_outliers: int
    n_total: int


try:
    _SYSTEM_PROMPT_TEXT: str | None = load_prompt("high_al_diagnoser")
except PromptNotFoundError:
    _SYSTEM_PROMPT_TEXT = None
    logger.warning(
        "high_al_diagnoser prompt not found — make_high_al_diagnoser returns None"
    )


# Features compared between outlier и baseline cohorts. Все process-факторы,
# которые металлургически объясняют overconsumption Al.
_DIAGNOSTIC_FEATURES = [
    "slag_feo_pct", "slag_mno_pct", "slag_cao_pct", "slag_sio2_pct",
    "carry_over_slag_kg_per_t", "dt_to_al_min", "t_al_addition_c",
    "o_a_initial_ppm", "refractory_heat_count",
]


# ── Outlier detection (deterministic, no LLM) ─────────────────────────


def _o_a_band(o_a_after: float | None) -> str:
    """Band O_a target для стратификации (ultra-low / low / normal)."""
    if o_a_after is None:
        return "unknown"
    if o_a_after < 5:
        return "ultra_low"
    if o_a_after < 15:
        return "low"
    return "normal"


def identify_high_al_outliers(
    heats: list[dict],
    *,
    method: str = "percentile",
    threshold_pct: float = 120.0,
    min_cohort: int = 5,
) -> tuple[list[dict], list[dict]]:
    """Return (outliers, baseline). Каждый heat — dict с al_added_kg etc.

    method='percentile': top-20% specific consumption внутри страты
        (method_id, o_a band). Страты < min_cohort идут целиком в baseline.
    method='prediction': al_added_kg / expected_al_kg > threshold_pct/100.
    """
    if method == "prediction":
        outliers: list[dict] = []
        baseline: list[dict] = []
        for h in heats:
            expected = h.get("expected_al_kg")
            actual = h.get("al_added_kg")
            if expected and actual and expected > 0:
                ratio = actual / expected
                if ratio > threshold_pct / 100.0:
                    outliers.append({**h, "overconsumption_pct": (ratio - 1) * 100})
                else:
                    baseline.append(h)
            else:
                baseline.append(h)
        return outliers, baseline

    # percentile: внутри страты (method_id, o_a band)
    strata: dict[tuple, list[tuple[float, dict]]] = defaultdict(list)
    skipped: list[dict] = []
    for h in heats:
        if h.get("al_added_kg") and h.get("steel_mass_ton"):
            key = (h.get("method_id"), _o_a_band(h.get("o_a_after_ppm")))
            spec = h["al_added_kg"] / h["steel_mass_ton"]
            strata[key].append((spec, h))
        else:
            skipped.append(h)

    outliers, baseline = [], list(skipped)
    for _key, items in strata.items():
        if len(items) < min_cohort:
            baseline.extend(h for _, h in items)
            continue
        items.sort(key=lambda t: t[0])
        cutoff_idx = int(len(items) * 0.8)
        cutoff_spec = items[cutoff_idx][0]
        med = statistics.median(s for s, _ in items)
        for spec, h in items:
            if spec >= cutoff_spec:
                outliers.append(
                    {**h, "overconsumption_pct": (spec / med - 1) * 100 if med else 0.0}
                )
            else:
                baseline.append(h)
    return outliers, baseline


def _feature_stats(heats: list[dict], features: list[str]) -> dict[str, dict]:
    """Per-feature mean/median/p10/p90 over a cohort (None-tolerant)."""
    out: dict[str, dict] = {}
    for f in features:
        vals = [h[f] for h in heats if h.get(f) is not None]
        if not vals:
            continue
        vals.sort()
        n = len(vals)
        out[f] = {
            "mean": round(statistics.mean(vals), 4),
            "median": round(statistics.median(vals), 4),
            "p10": vals[int(n * 0.1)],
            "p90": vals[min(int(n * 0.9), n - 1)],
        }
    return out


def build_cohort_context(
    outliers: list[dict],
    baseline: list[dict],
    *,
    method: str,
    threshold_pct: float,
    cap_n: int = 40,
) -> dict:
    """Build deterministic cohort payload для LLM.

    feature_deltas — главный сигнал: Python считает (outlier_mean − baseline_mean)
    по каждому диагностическому фактору, сортирует по |delta_pct|, LLM ранжирует
    по mechanism plausibility.
    """
    outlier_stats = _feature_stats(outliers, _DIAGNOSTIC_FEATURES)
    baseline_stats = _feature_stats(baseline, _DIAGNOSTIC_FEATURES)
    deltas = []
    for f in _DIAGNOSTIC_FEATURES:
        if f in outlier_stats and f in baseline_stats:
            o_mean = outlier_stats[f]["mean"]
            b_mean = baseline_stats[f]["mean"]
            delta_pct = ((o_mean - b_mean) / b_mean * 100) if b_mean else 0.0
            deltas.append({
                "feature": f,
                "outlier_mean": round(o_mean, 3),
                "baseline_mean": round(b_mean, 3),
                "delta_pct": round(delta_pct, 1),
                "direction": "higher" if delta_pct > 0 else "lower",
            })
    deltas.sort(key=lambda d: abs(d["delta_pct"]), reverse=True)
    return {
        "n_total": len(outliers) + len(baseline),
        "n_outliers": len(outliers),
        "method": method,
        "threshold_pct": threshold_pct,
        "outliers": [
            {k: h.get(k) for k in (
                "heat_id", "plant_id", "method_id", "al_added_kg", "expected_al_kg",
                "overconsumption_pct", *_DIAGNOSTIC_FEATURES, "carrier_gas",
                "o_a_after_ppm",
            )}
            for h in outliers[:cap_n]
        ],
        "outlier_cohort_stats": outlier_stats,
        "baseline_cohort_stats": baseline_stats,
        "feature_deltas": deltas,
    }


# ── LLM diagnoser (R-003) ─────────────────────────────────────────────


_TOOL_SCHEMA = {
    "name": "report_high_al_diagnosis",
    "description": "Submit root-cause diagnosis of high-Al consumption cohort",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "cohort_severity": {"type": "string"},
            "hypotheses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "root_cause": {"type": "string"},
                        "evidence_features": {
                            "type": "array", "items": {"type": "string"},
                        },
                        "mechanism": {"type": "string"},
                        "est_excess_al_pct": {"type": "number"},
                        "confidence": {
                            "type": "string",
                            "enum": ["HIGH", "MEDIUM", "LOW"],
                        },
                        "suggested_action": {"type": "string"},
                    },
                    "required": [
                        "root_cause", "evidence_features", "mechanism",
                        "est_excess_al_pct", "confidence", "suggested_action",
                    ],
                },
            },
        },
        "required": ["summary", "cohort_severity", "hypotheses"],
    },
}


class HighAlDiagnoser:
    MODEL_ID = "claude-sonnet-4-6"
    MAX_TOKENS = 4096
    TIMEOUT_S = 120.0

    def __init__(self, client: Any, model: str | None = None):
        self.client = client
        self.model = model or self.MODEL_ID

    def diagnose(self, cohort_context: dict) -> HighAlDiagnosis | None:
        user_payload = (
            "Cohort high-Al плавок для root-cause анализа:\n```json\n"
            + json.dumps(cohort_context, indent=2, ensure_ascii=False, default=str)
            + "\n```"
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
                tool_choice={"type": "tool", "name": "report_high_al_diagnosis"},
                messages=[{"role": "user", "content": user_payload}],
                timeout=self.TIMEOUT_S,
            )
        except Exception as e:
            logger.warning("HighAlDiagnoser API call failed: %s", e)
            return None

        elapsed = time.monotonic() - start
        tool_block = next(
            (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            logger.warning("HighAlDiagnoser: no tool_use in response")
            return None

        try:
            r = tool_block.input
            hyps: list[AlOverHypothesis] = []
            for h in r.get("hypotheses", []):
                try:
                    hyps.append(AlOverHypothesis(
                        root_cause=h["root_cause"],
                        evidence_features=list(h["evidence_features"]),
                        mechanism=h["mechanism"],
                        est_excess_al_pct=float(h["est_excess_al_pct"]),
                        confidence=h["confidence"],
                        suggested_action=h["suggested_action"],
                    ))
                except (KeyError, TypeError, ValueError):
                    continue  # skip malformed hypothesis, keep the good ones
            diagnosis = HighAlDiagnosis(
                summary=r["summary"],
                hypotheses=hyps,
                cohort_severity=r["cohort_severity"],
                n_outliers=cohort_context.get("n_outliers", 0),
                n_total=cohort_context.get("n_total", 0),
            )
        except (KeyError, TypeError) as e:
            logger.warning("HighAlDiagnoser: bad payload: %s", e)
            return None

        _log_usage(resp, elapsed, diagnosis)
        return diagnosis


def _log_usage(resp, elapsed_s, diagnosis):
    try:
        from decision_log.logger import log_decision

        usage = getattr(resp, "usage", None)
        log_decision(
            phase="deoxidation",
            decision=(
                f"High-Al diagnosis: {len(diagnosis.hypotheses)} hypotheses, "
                f"severity={diagnosis.cohort_severity}, "
                f"{diagnosis.n_outliers}/{diagnosis.n_total} outliers"
            ),
            reasoning=(
                f"Model={getattr(resp, 'model', 'unknown')}, "
                f"input={getattr(usage, 'input_tokens', 0)} "
                f"(cache_read={getattr(usage, 'cache_read_input_tokens', 0)}, "
                f"cache_create={getattr(usage, 'cache_creation_input_tokens', 0)}), "
                f"output={getattr(usage, 'output_tokens', 0)}, "
                f"latency={elapsed_s:.2f}s"
            ),
            context={"diagnosis": asdict(diagnosis)},
            author="high_al_diagnoser",
            tags=["high_al_diagnoser", "sonnet-4-6"],
        )
    except Exception as exc:
        logger.warning("Decision Log save failed: %s", exc)


def make_high_al_diagnoser() -> HighAlDiagnoser | None:
    if _SYSTEM_PROMPT_TEXT is None:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("anthropic missing — HighAlDiagnoser disabled")
        return None
    return HighAlDiagnoser(client=Anthropic())


__all__ = [
    "AlOverHypothesis", "HighAlDiagnosis", "HighAlDiagnoser",
    "identify_high_al_outliers", "build_cohort_context", "make_high_al_diagnoser",
]
