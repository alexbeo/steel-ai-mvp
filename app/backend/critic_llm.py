"""
LLM-Critic v2 — Claude Sonnet 4.6 as exploratory reviewer on training phase.

Activated via ANTHROPIC_API_KEY env var; returns an empty observation
list on any API failure so the pipeline continues as if LLM-Critic
were not configured. Observations are informational — they do not
affect Verdict (Pattern Library remains the sole gate).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

Severity = Literal["HIGH", "MEDIUM", "LOW"]
Category = Literal["data", "model", "physics", "process"]


@dataclass
class LLMObservation:
    severity: Severity
    category: Category
    message: str
    rationale: str


_SYSTEM_PROMPT_TEXT = """\
Ты senior ML-инженер с 10-летним опытом в металлургии HSLA-сталей
(трубопроводные, K60-K65). Тебе на review поступает training-артефакт
от XGBoost-пайплайна: метрики, feature importance, training_ranges,
split/CV strategy, размеры выборок.

Твоя задача — выступить вторым мнением после детерминированной
Pattern Library. Ищи то, что правила не видят:

1. METRICS & CALIBRATION
   - Коэффициенты R² правдоподобны для текущего размера датасета?
   - Gap r2_train − r2_val / r2_test указывает на overfitting?
   - Coverage 90% CI в ожидаемом диапазоне 85-95%?
     Под-confidence (<85%) / сверх-confidence (>95%)?

2. FEATURE IMPORTANCE PHYSICS
   - Для pipe-HSLA с target σт / σв / δ ожидаются в top-10:
     c_pct, mn_pct, nb_pct, ti_pct, v_pct, rolling_finish_temp,
     cooling_rate, cev_iiw, pcm, microalloying_sum.
   - Если в top-5 «экзотика» (cu_pct, s_pct, n_ppm) без Nb/Ti —
     подозрение на spurious correlation или data leakage.
   - Суммарная доля одной фичи > 50% — возможна утечка target'а.

3. DATA LEAKAGE VIA SPLIT
   - Если split_strategy != "time_based" на данных с временной
     колонкой — high risk leakage.
   - Если cv_strategy != "group_kfold" на данных с groups —
     оптимистичный CV-score.

4. TRAINING_RANGES PHYSICAL SANITY
   - Диапазоны должны быть в типичных для pipe-HSLA пределах:
     C 0.03-0.15 %; Mn 0.8-1.8; Nb 0-0.06; Ti 0-0.03; Si 0.1-0.6;
     rolling_finish_temp 740-860 °C; cooling_rate 5-30 °C/s.
   - Выход за эти пределы → либо другой класс стали, либо ошибка
     генерации данных.

ФОРМАТ ОТВЕТА — через tool report_observations:
- До 5 observations (выбирай самые важные).
- severity: HIGH (стоп-сигнал для senior'а), MEDIUM (нужно
  выяснить), LOW (к сведению).
- category: data | model | physics | process.
- message и rationale на русском.
- Если всё чисто — верни пустой список. Не придумывай проблемы.
"""


_TOOL_SCHEMA = {
    "name": "report_observations",
    "description": "Report observations about training artifact quality",
    "input_schema": {
        "type": "object",
        "properties": {
            "observations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string",
                                     "enum": ["HIGH", "MEDIUM", "LOW"]},
                        "category": {"type": "string",
                                     "enum": ["data", "model",
                                              "physics", "process"]},
                        "message":   {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["severity", "category", "message", "rationale"],
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["observations"],
    },
}


class LLMCritic:
    MODEL_ID = "claude-sonnet-4-6"
    MAX_TOKENS = 1200
    TIMEOUT_S = 30.0

    def __init__(self, client: Any, model: str | None = None):
        self.client = client
        self.model = model or self.MODEL_ID

    def review_training(self, context: dict) -> list[LLMObservation]:
        """Stub — implemented in Task 3."""
        return []


def make_llm_critic() -> LLMCritic | None:
    """Return LLMCritic if ANTHROPIC_API_KEY is set, else None."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.warning("anthropic package missing — LLM-Critic disabled")
        return None
    return LLMCritic(client=Anthropic())
