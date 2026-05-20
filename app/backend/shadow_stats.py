"""Statistical test module для shadow validation (Phase 2 S2).

Гипотеза: AI снижает Al на ≥10% без потери качества. Per-heat paired:
Wilcoxon signed-rank + percentile bootstrap CI на median delta %, стратификация
plant×method. Stats по quality_pass плавкам (честная экономия от высокого η,
не от недодозировки — см. S1 review).

Aggregate = confirmatory hypothesis. Per-stratum p = exploratory (без MC коррекции).
"""
from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import wilcoxon

from app.backend.shadow_validation import ShadowComparison

logger = logging.getLogger(__name__)


@dataclass
class StratumStats:
    plant_id: str
    method_id: str | None
    n: int
    n_quality_pass: int
    median_delta_pct: float
    ci_low: float
    ci_high: float
    ci_method: str               # "percentile" | "insufficient_n"
    wilcoxon_p: float | None
    mean_al_saved_kg: float


@dataclass
class ShadowStats:
    n_total: int
    n_compared: int
    n_quality_pass: int
    median_delta_pct: float       # quality_pass only
    median_delta_pct_all: float   # все compared
    ci_low: float
    ci_high: float
    wilcoxon_p: float | None
    hypothesis_met: bool
    total_al_saved_kg: float
    n_literature_fallback: int
    n_resamples: int
    random_state: int
    per_stratum: list[StratumStats] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _bootstrap_median_ci(
    deltas: list[float], n_resamples: int, rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI на median."""
    if len(deltas) < 2:
        m = deltas[0] if deltas else float("nan")
        return m, m
    arr = np.array(deltas)
    medians = np.empty(n_resamples)
    n = len(arr)
    for i in range(n_resamples):
        sample = arr[rng.integers(0, n, size=n)]
        medians[i] = np.median(sample)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def _safe_wilcoxon(deltas: list[float]) -> float | None:
    """Wilcoxon signed-rank p-value, graceful на edge cases."""
    if len(deltas) < 1:
        return None
    # Все нули → нет вариации
    if all(d == 0 for d in deltas):
        return None
    try:
        # H0: median = 0. Тестируем сдвиг от нуля.
        _, p = wilcoxon(deltas)
        return float(p)
    except ValueError:
        return None


def compute_shadow_stats(
    comparisons: list[ShadowComparison],
    *,
    target_reduction_pct: float = 10.0,
    min_stratum_n: int = 10,
    bootstrap_resamples: int = 2000,
    random_state: int = 42,
    min_hypothesis_n: int = 30,
) -> ShadowStats:
    """Compute shadow validation statistics. Aggregate confirmatory test."""
    rng = np.random.default_rng(random_state)
    n_total = len(comparisons)
    compared = [c for c in comparisons if c.skip_reason is None]
    n_compared = len(compared)
    qpass = [c for c in compared if c.quality_pass]
    n_quality_pass = len(qpass)

    notes: list[str] = []

    # Literature fallback count
    n_lit = sum(1 for c in compared
                if c.eta_ai_source and "literature" in c.eta_ai_source.lower())
    if n_lit > 0:
        notes.append(
            f"{n_lit}/{n_compared} плавок на literature_fallback (нет plant-калибровки) "
            f"— η менее надёжна, down-weight при интерпретации."
        )

    # Edge: nothing compared
    if n_compared == 0:
        notes.append("Нет плавок для сравнения (все skip).")
        return ShadowStats(
            n_total=n_total, n_compared=0, n_quality_pass=0,
            median_delta_pct=float("nan"), median_delta_pct_all=float("nan"),
            ci_low=float("nan"), ci_high=float("nan"), wilcoxon_p=None,
            hypothesis_met=False, total_al_saved_kg=0.0,
            n_literature_fallback=0, n_resamples=bootstrap_resamples,
            random_state=random_state, notes=notes,
        )

    deltas_all = [c.delta_pct for c in compared if c.delta_pct is not None]
    median_all = statistics.median(deltas_all) if deltas_all else float("nan")

    # Primary stats: quality_pass
    if n_quality_pass == 0:
        notes.append("Ни одна плавка не прошла quality gate — экономия не подтверждена.")
        return ShadowStats(
            n_total=n_total, n_compared=n_compared, n_quality_pass=0,
            median_delta_pct=float("nan"), median_delta_pct_all=median_all,
            ci_low=float("nan"), ci_high=float("nan"), wilcoxon_p=None,
            hypothesis_met=False, total_al_saved_kg=0.0,
            n_literature_fallback=n_lit, n_resamples=bootstrap_resamples,
            random_state=random_state, notes=notes,
        )

    deltas = [c.delta_pct for c in qpass if c.delta_pct is not None]
    median_delta = statistics.median(deltas)
    ci_low, ci_high = _bootstrap_median_ci(deltas, bootstrap_resamples, rng)
    wp = _safe_wilcoxon(deltas)

    # Total Al saved (kg) — sum (actual - ai_p50) для quality_pass
    total_saved = sum(
        (c.al_actual_kg - c.al_ai_p50_kg)
        for c in qpass
        if c.al_actual_kg is not None and c.al_ai_p50_kg is not None
    )

    # Quality fail fraction note
    qfail_frac = 1 - n_quality_pass / n_compared if n_compared else 0
    if qfail_frac > 0.3:
        notes.append(
            f"Высокая доля quality_fail ({qfail_frac*100:.0f}%) — "
            f"проверьте residual spec / dose sufficiency."
        )

    # hypothesis_met: 4 условия
    hypothesis_met = (
        median_delta <= -target_reduction_pct
        and wp is not None and wp < 0.05
        and ci_high < 0
        and n_quality_pass >= min_hypothesis_n
    )
    if n_quality_pass < min_hypothesis_n:
        notes.append(
            f"n_quality_pass={n_quality_pass} < {min_hypothesis_n} — "
            f"выборка мала для подтверждения гипотезы."
        )

    # Per-stratum
    notes.append(
        "Per-stratum p-values exploratory (без multiple-comparison коррекции); "
        "только aggregate hypothesis confirmatory."
    )
    strata: dict[tuple[str, str | None], list[ShadowComparison]] = defaultdict(list)
    for c in qpass:
        strata[(c.plant_id, c.method_id)].append(c)
    per_stratum: list[StratumStats] = []
    for (plant, method), items in sorted(strata.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        s_deltas = [c.delta_pct for c in items if c.delta_pct is not None]
        if not s_deltas:
            continue
        s_median = statistics.median(s_deltas)
        s_saved_vals = [
            (c.al_actual_kg - c.al_ai_p50_kg) for c in items
            if c.al_actual_kg is not None and c.al_ai_p50_kg is not None
        ]
        s_saved = statistics.mean(s_saved_vals) if s_saved_vals else 0.0
        if len(s_deltas) >= min_stratum_n:
            s_ci_low, s_ci_high = _bootstrap_median_ci(s_deltas, bootstrap_resamples, rng)
            s_wp = _safe_wilcoxon(s_deltas)
            ci_method = "percentile"
        else:
            s_ci_low = s_ci_high = s_median
            s_wp = None
            ci_method = "insufficient_n"
        per_stratum.append(StratumStats(
            plant_id=plant, method_id=method, n=len(items),
            n_quality_pass=len(items), median_delta_pct=s_median,
            ci_low=s_ci_low, ci_high=s_ci_high, ci_method=ci_method,
            wilcoxon_p=s_wp, mean_al_saved_kg=s_saved,
        ))

    return ShadowStats(
        n_total=n_total, n_compared=n_compared, n_quality_pass=n_quality_pass,
        median_delta_pct=median_delta, median_delta_pct_all=median_all,
        ci_low=ci_low, ci_high=ci_high, wilcoxon_p=wp,
        hypothesis_met=hypothesis_met, total_al_saved_kg=total_saved,
        n_literature_fallback=n_lit, n_resamples=bootstrap_resamples,
        random_state=random_state, per_stratum=per_stratum, notes=notes,
    )


__all__ = ["StratumStats", "ShadowStats", "compute_shadow_stats"]
