"""Tests для shadow statistics (Phase 2 S2)."""
from __future__ import annotations

from app.backend.shadow_stats import compute_shadow_stats
from app.backend.shadow_validation import ShadowComparison


def _comp(delta=-12.0, quality=True, plant="P1", method="asis_shot",
          al_actual=200.0, al_ai=176.0, source="plant_only", skip=None):
    return ShadowComparison(
        heat_pk=None, heat_id=None, plant_id=plant, method_id=method,
        al_actual_kg=al_actual, al_ai_p50_kg=al_ai, al_ai_p90_kg=al_ai * 1.05,
        delta_pct=delta, delta_p90_pct=delta + 5,
        ai_target_o_a_ppm=5.0, actual_o_a_ppm=5.0,
        residual_in_spec=True, dose_sufficient=True, quality_pass=quality,
        eta_ai_source=source, skip_reason=skip,
    )


def test_stats_basic():
    comps = [_comp(delta=-12) for _ in range(40)]
    s = compute_shadow_stats(comps)
    assert s.n_compared == 40
    assert s.n_quality_pass == 40
    assert abs(s.median_delta_pct - (-12)) < 0.01


def test_wilcoxon_significant_negative():
    import random
    random.seed(1)
    comps = [_comp(delta=-15 + random.uniform(-2, 2)) for _ in range(40)]
    s = compute_shadow_stats(comps)
    assert s.wilcoxon_p is not None and s.wilcoxon_p < 0.05
    assert s.hypothesis_met is True


def test_wilcoxon_not_significant():
    import random
    random.seed(2)
    comps = [_comp(delta=random.uniform(-3, 3)) for _ in range(40)]
    s = compute_shadow_stats(comps)
    # median около 0 → hypothesis не met
    assert s.hypothesis_met is False


def test_bootstrap_ci_brackets_median():
    comps = [_comp(delta=-12) for _ in range(40)]
    s = compute_shadow_stats(comps)
    assert s.ci_low <= s.median_delta_pct <= s.ci_high


def test_stratification_groups():
    comps = ([_comp(plant="P1", method="asis_shot") for _ in range(15)]
             + [_comp(plant="P2", method="ingot") for _ in range(15)])
    s = compute_shadow_stats(comps)
    keys = {(st.plant_id, st.method_id) for st in s.per_stratum}
    assert ("P1", "asis_shot") in keys
    assert ("P2", "ingot") in keys


def test_small_stratum_no_pvalue():
    comps = [_comp(plant="P3", method="ingot") for _ in range(5)]  # n<10
    s = compute_shadow_stats(comps)
    st = next(st for st in s.per_stratum if st.plant_id == "P3")
    assert st.wilcoxon_p is None
    assert st.ci_method == "insufficient_n"


def test_quality_pass_filtering():
    # Достаточно quality_fail плавок (+5), чтобы сдвинуть all-median выше
    # quality_pass-median (-15) — иначе median по всем останется на -15.
    comps = ([_comp(delta=-15, quality=True) for _ in range(30)]
             + [_comp(delta=+5, quality=False) for _ in range(31)])
    s = compute_shadow_stats(comps)
    assert s.n_quality_pass == 30
    # median по quality_pass = -15, all (большинство +5) выше
    assert s.median_delta_pct < s.median_delta_pct_all


def test_hypothesis_met_requires_all_four():
    # median > -10 → not met
    s1 = compute_shadow_stats([_comp(delta=-5) for _ in range(40)])
    assert s1.hypothesis_met is False
    # n < 30 → not met даже при сильном сигнале
    s2 = compute_shadow_stats([_comp(delta=-15) for _ in range(20)])
    assert s2.hypothesis_met is False
    # все условия met
    s3 = compute_shadow_stats([_comp(delta=-15) for _ in range(40)])
    assert s3.hypothesis_met is True


def test_literature_fallback_counted():
    comps = ([_comp(source="plant_only") for _ in range(20)]
             + [_comp(source="literature_fallback") for _ in range(10)])
    s = compute_shadow_stats(comps)
    assert s.n_literature_fallback == 10
    assert any("literature" in n.lower() for n in s.notes)


def test_empty_comparisons():
    s = compute_shadow_stats([])
    assert s.n_compared == 0
    assert s.hypothesis_met is False


def test_all_skipped():
    comps = [_comp(skip="no_outcome") for _ in range(10)]
    s = compute_shadow_stats(comps)
    assert s.n_compared == 0
    assert s.hypothesis_met is False


def test_bootstrap_reproducible():
    comps = [_comp(delta=-12 + (i % 5)) for i in range(40)]
    s1 = compute_shadow_stats(comps, random_state=7)
    s2 = compute_shadow_stats(comps, random_state=7)
    assert s1.ci_low == s2.ci_low and s1.ci_high == s2.ci_high


def test_total_al_saved():
    comps = [_comp(al_actual=200, al_ai=176) for _ in range(10)]
    s = compute_shadow_stats(comps)
    # saved = (200-176)*10 = 240
    assert abs(s.total_al_saved_kg - 240) < 0.01
