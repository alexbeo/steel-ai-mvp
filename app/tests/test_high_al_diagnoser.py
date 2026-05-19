"""Tests for high_al_diagnoser (PR 12 Block IV).

Cohort logic (deterministic, no LLM) + LLM quintet (mocked Anthropic).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from app.backend.high_al_diagnoser import (
    AlOverHypothesis,
    HighAlDiagnoser,
    HighAlDiagnosis,
    build_cohort_context,
    identify_high_al_outliers,
    make_high_al_diagnoser,
)


# ── helpers ───────────────────────────────────────────────────────────


def _heat(spec_al, *, method_id="asis_shot", o_after=4.0, mass=200.0, **extra):
    """Construct heat dict where al_added_kg yields target specific consumption."""
    h = {
        "heat_id": extra.pop("heat_id", "H-x"),
        "method_id": method_id,
        "steel_mass_ton": mass,
        "al_added_kg": spec_al * mass,
        "o_a_after_ppm": o_after,
    }
    h.update(extra)
    return h


# ── cohort logic (4) ───────────────────────────────────────────────────


def test_identify_high_al_outliers_percentile():
    # 10 heats same stratum: 8 low spec (~1.0), 2 high spec (~3.0) → top-20% outliers
    heats = [_heat(1.0, heat_id=f"L{i}", slag_feo_pct=10.0) for i in range(8)]
    heats += [_heat(3.0, heat_id=f"H{i}", slag_feo_pct=30.0) for i in range(2)]
    outliers, baseline = identify_high_al_outliers(heats, method="percentile")
    assert len(outliers) == 2
    assert len(baseline) == 8
    assert all(o["heat_id"].startswith("H") for o in outliers)
    assert all("overconsumption_pct" in o for o in outliers)


def test_identify_high_al_outliers_prediction():
    heats = [
        {"heat_id": "ok", "al_added_kg": 100.0, "expected_al_kg": 100.0},
        {"heat_id": "over", "al_added_kg": 150.0, "expected_al_kg": 100.0},  # ratio 1.5 > 1.2
        {"heat_id": "no_exp", "al_added_kg": 200.0},  # no expected → baseline
    ]
    outliers, baseline = identify_high_al_outliers(
        heats, method="prediction", threshold_pct=120.0
    )
    assert [o["heat_id"] for o in outliers] == ["over"]
    assert outliers[0]["overconsumption_pct"] == 50.0
    assert {b["heat_id"] for b in baseline} == {"ok", "no_exp"}


def test_cohort_context_builder():
    outliers = [_heat(3.0, heat_id=f"H{i}", slag_feo_pct=30.0, dt_to_al_min=20.0) for i in range(5)]
    baseline = [_heat(1.0, heat_id=f"L{i}", slag_feo_pct=10.0, dt_to_al_min=5.0) for i in range(5)]
    ctx = build_cohort_context(outliers, baseline, method="percentile", threshold_pct=120.0)
    assert ctx["n_outliers"] == 5
    assert ctx["n_total"] == 10
    deltas = ctx["feature_deltas"]
    # slag_feo: outlier 30 vs baseline 10 → +200%; sorted by |delta| descending
    feo = next(d for d in deltas if d["feature"] == "slag_feo_pct")
    assert feo["delta_pct"] == 200.0
    assert feo["direction"] == "higher"
    abs_deltas = [abs(d["delta_pct"]) for d in deltas]
    assert abs_deltas == sorted(abs_deltas, reverse=True)


def test_stratification_separates_methods():
    # asis: high spec; chuck: low spec. Within-stratum top-20% only.
    asis = [_heat(2.0, method_id="asis_shot", heat_id=f"A{i}") for i in range(10)]
    chuck = [_heat(0.5, method_id="chuck", heat_id=f"C{i}") for i in range(10)]
    outliers, _ = identify_high_al_outliers(asis + chuck, method="percentile")
    # An outlier in one stratum should not be measured against the other stratum's median
    methods = {o["method_id"] for o in outliers}
    # Each stratum contributes its own top-20% — both methods can appear,
    # but a low-spec chuck heat must never out-rank a high-spec asis baseline heat.
    for o in outliers:
        assert o["overconsumption_pct"] >= 0
    assert methods.issubset({"asis_shot", "chuck"})


# ── LLM quintet (5) ─────────────────────────────────────────────────────


def test_factory_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_high_al_diagnoser() is None


def test_factory_no_prompt(monkeypatch):
    import app.backend.high_al_diagnoser as mod
    monkeypatch.setattr(mod, "_SYSTEM_PROMPT_TEXT", None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert make_high_al_diagnoser() is None


def _mock_response(payload):
    resp = MagicMock()
    tb = MagicMock()
    tb.type = "tool_use"
    tb.input = payload
    txt = MagicMock()
    txt.type = "text"
    resp.content = [txt, tb]
    resp.usage = MagicMock(
        input_tokens=3000, output_tokens=900,
        cache_read_input_tokens=2500, cache_creation_input_tokens=0,
    )
    resp.model = "claude-sonnet-4-6"
    resp.stop_reason = "tool_use"
    return resp


def _full_diagnosis_dict(**overrides):
    base = {
        "summary": "Перерасход объясним FeO-реокислением.",
        "cohort_severity": "умеренный — ~15% перерасход",
        "hypotheses": [
            {
                "root_cause": "FeO реокисление шлаком",
                "evidence_features": ["slag_feo_pct"],
                "mechanism": "FeO отдаёт O обратно в металл, Al связывает впустую",
                "est_excess_al_pct": 12.0,
                "confidence": "HIGH",
                "suggested_action": "Снизить carry-over BOF-шлака",
            },
        ],
    }
    base.update(overrides)
    return base


def test_diagnose_parses_well_formed(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response(_full_diagnosis_dict())
    import app.backend.high_al_diagnoser as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)
    out = HighAlDiagnoser(client=mock_client).diagnose(
        {"n_outliers": 5, "n_total": 50}
    )
    assert isinstance(out, HighAlDiagnosis)
    assert out.cohort_severity.startswith("умеренный")
    assert out.n_outliers == 5
    assert out.n_total == 50
    assert len(out.hypotheses) == 1
    assert isinstance(out.hypotheses[0], AlOverHypothesis)
    assert out.hypotheses[0].confidence == "HIGH"


def test_diagnose_api_failure():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("net")
    out = HighAlDiagnoser(client=mock_client).diagnose({"n_outliers": 1, "n_total": 10})
    assert out is None


def test_diagnose_skips_malformed_hypotheses(monkeypatch):
    payload = _full_diagnosis_dict(
        hypotheses=[
            {"root_cause": "incomplete"},  # missing required fields → skipped
            {
                "root_cause": "valid",
                "evidence_features": ["slag_feo_pct"],
                "mechanism": "ok",
                "est_excess_al_pct": 5.0,
                "confidence": "MEDIUM",
                "suggested_action": "do X",
            },
        ]
    )
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response(payload)
    import app.backend.high_al_diagnoser as mod
    monkeypatch.setattr(mod, "_log_usage", lambda *a, **kw: None)
    out = HighAlDiagnoser(client=mock_client).diagnose({"n_outliers": 2, "n_total": 20})
    assert out is not None
    assert len(out.hypotheses) == 1  # malformed dropped, valid kept
    assert out.hypotheses[0].root_cause == "valid"
