"""Unit tests for multi-class steel support."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.backend.steel_classes import (
    AVAILABLE_CLASS_IDS,
    SteelClassProfile,
    available_steel_classes,
    compute_features_for_class,
    get_synthetic_generator,
    load_steel_class,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_load_pipe_hsla_profile():
    p = load_steel_class("pipe_hsla")
    assert p.id == "pipe_hsla"
    assert p.standard.startswith("API 5L")
    assert "c_pct" in p.feature_set
    assert "nb_pct" in p.feature_set
    assert p.physical_bounds["c_pct"] == [0.04, 0.12]
    assert "yield_strength_mpa" in p.target_ids()


def test_load_en10083_profile():
    p = load_steel_class("en10083_qt")
    assert p.id == "en10083_qt"
    assert p.standard == "EN 10083-2"
    assert "tempering_temp" in p.feature_set
    assert "section_thickness_mm" in p.feature_set
    assert "nb_pct" not in p.feature_set
    assert p.physical_bounds["c_pct"] == [0.18, 0.65]
    assert "hardness_hrc" in p.target_ids()


def test_available_steel_classes_registry():
    profiles = available_steel_classes()
    assert len(profiles) == len(AVAILABLE_CLASS_IDS)
    ids = {p.id for p in profiles}
    assert ids == set(AVAILABLE_CLASS_IDS)
    for p in profiles:
        assert isinstance(p, SteelClassProfile)


def test_synthetic_generator_en10083_qt_physical_sanity():
    gen = get_synthetic_generator("en10083_qt")
    df = gen(n_samples=500, random_seed=1)

    for col in ("c_pct", "mn_pct", "tempering_temp",
                "section_thickness_mm", "hardness_hrc",
                "tensile_strength_mpa", "campaign_id", "heat_date"):
        assert col in df.columns, f"missing {col}"

    assert df["c_pct"].between(0.18, 0.65).all()
    assert df["tempering_temp"].between(150, 650).all()
    assert df["hardness_hrc"].between(15, 65).all()

    hard_mask = (df["c_pct"] > 0.55) & (df["tempering_temp"] < 250)
    soft_mask = (df["c_pct"] < 0.25) & (df["tempering_temp"] > 550)
    if hard_mask.sum() > 10 and soft_mask.sum() > 10:
        assert df.loc[hard_mask, "hardness_hrc"].mean() > \
               df.loc[soft_mask, "hardness_hrc"].mean() + 5


def test_compute_features_for_class_passthrough_for_qt():
    df = pd.DataFrame({"c_pct": [0.4], "mn_pct": [0.6]})
    out = compute_features_for_class(df, "en10083_qt")
    assert list(out.columns) == ["c_pct", "mn_pct"]


def test_compute_features_for_class_adds_derived_for_hsla():
    df = pd.DataFrame({
        "c_pct": [0.08], "mn_pct": [1.5], "si_pct": [0.3],
        "p_pct": [0.015], "s_pct": [0.005], "cr_pct": [0.1],
        "ni_pct": [0.1], "mo_pct": [0.02], "cu_pct": [0.2],
        "al_pct": [0.03], "v_pct": [0.03], "nb_pct": [0.04],
        "ti_pct": [0.02], "n_ppm": [50],
        "rolling_finish_temp": [820], "cooling_rate_c_per_s": [18],
    })
    out = compute_features_for_class(df, "pipe_hsla")
    assert "cev_iiw" in out.columns
    assert "pcm" in out.columns


def test_train_model_persists_steel_class(tmp_path, monkeypatch):
    """train_model stores steel_class in meta.json and TrainedModel."""
    import json
    from app.backend import model_trainer
    from app.backend.feature_eng import PIPE_HSLA_FEATURE_SET, compute_hsla_features

    monkeypatch.setattr(model_trainer, "MODELS_DIR", tmp_path)

    gen = get_synthetic_generator("pipe_hsla")
    df = gen(n_samples=500, random_seed=1)
    df_feat = compute_hsla_features(df)
    feat = [f for f in PIPE_HSLA_FEATURE_SET if f in df_feat.columns]

    trained = model_trainer.train_model(
        df_feat, "yield_strength_mpa", feat,
        n_optuna_trials=3, steel_class="pipe_hsla",
    )
    assert trained.steel_class == "pipe_hsla"
    assert trained.version.startswith("hsla_")

    meta_path = tmp_path / trained.version / "meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["steel_class"] == "pipe_hsla"

    bundle = model_trainer.load_model(trained.version)
    assert bundle["meta"]["steel_class"] == "pipe_hsla"


def test_train_model_en10083_qt_smoke(tmp_path, monkeypatch):
    """End-to-end Q&T training on synthetic data."""
    from app.backend import model_trainer
    from app.backend.steel_classes import load_steel_class

    monkeypatch.setattr(model_trainer, "MODELS_DIR", tmp_path)

    profile = load_steel_class("en10083_qt")
    gen = get_synthetic_generator("en10083_qt")
    df = gen(n_samples=1500, random_seed=7)

    trained = model_trainer.train_model(
        df, "hardness_hrc", profile.feature_set,
        n_optuna_trials=5, steel_class="en10083_qt",
    )
    assert trained.steel_class == "en10083_qt"
    assert trained.version.startswith("en10083qt_")
    assert trained.metrics.r2_test > 0.6


def test_pattern_m05_uses_ctx_expected_features_hsla():
    from pattern_library.patterns import run_all_patterns, Phase
    ctx = {
        "steel_class": "pipe_hsla",
        "expected_top_features": [
            "c_pct", "mn_pct", "nb_pct", "ti_pct", "v_pct",
            "rolling_finish_temp", "cooling_rate_c_per_s",
            "cev_iiw", "pcm", "microalloying_sum",
        ],
        "feature_importance": {
            "cu_pct": 0.40, "s_pct": 0.20, "n_ppm": 0.15,
            "p_pct": 0.10, "al_pct": 0.05,
        },
    }
    warnings = run_all_patterns(ctx, phase=Phase.TRAINING)
    ids = {w["pattern_id"] for w in warnings}
    assert "M05" in ids


def test_pattern_m05_uses_ctx_expected_features_en10083():
    from pattern_library.patterns import run_all_patterns, Phase
    ctx = {
        "steel_class": "en10083_qt",
        "expected_top_features": [
            "c_pct", "tempering_temp", "austenitizing_temp",
            "mn_pct", "section_thickness_mm",
        ],
        "feature_importance": {
            "c_pct": 0.35, "tempering_temp": 0.25,
            "austenitizing_temp": 0.15, "mn_pct": 0.10,
            "section_thickness_mm": 0.08, "cr_pct": 0.05,
        },
    }
    warnings = run_all_patterns(ctx, phase=Phase.TRAINING)
    ids = {w["pattern_id"] for w in warnings}
    assert "M05" not in ids


def test_engine_critic_context_gets_per_class_bounds_and_features():
    """_build_critic_context populates expected_top_features + physical_bounds per class."""
    from app.backend.engine import Orchestrator, PipelineState, AgentResult, Critic

    state = PipelineState(user_request={"task_type": "train"})
    state.features["training_ranges"] = {}
    agent_result = AgentResult(
        agent_name="model_trainer", success=True,
        output={
            "version": "en10083qt_hardness_hrc_xgb_test",
            "steel_class": "en10083_qt",
            "split_strategy": "time_based",
            "cv_strategy": "group_kfold",
            "has_uncertainty": True,
            "has_ood_detector": True,
            "feature_importance": {
                "c_pct": 0.3, "tempering_temp": 0.2,
                "austenitizing_temp": 0.15, "mn_pct": 0.1,
                "section_thickness_mm": 0.08,
            },
        },
    )
    orch = Orchestrator(agents={}, critic=Critic(use_llm=False))
    ctx = orch._build_critic_context("training", state, agent_result)

    assert ctx["steel_class"] == "en10083_qt"
    assert "c_pct" in ctx["expected_top_features"]
    assert "tempering_temp" in ctx["expected_top_features"]
    assert ctx["physical_bounds"]["c_pct"] == [0.18, 0.65]


def test_pattern_d07_uses_ctx_bounds_en10083():
    """D07 uses ctx['physical_bounds'] when provided (per-class)."""
    import pandas as pd
    from pattern_library.patterns import run_all_patterns, Phase
    df = pd.DataFrame({
        "c_pct": [0.25, 0.40, 0.75],    # last violates Q&T upper bound 0.65
        "tempering_temp": [200, 500, 700],   # last violates upper 650
    })
    ctx = {
        "dataframe": df,
        "physical_bounds": {
            "c_pct": [0.18, 0.65],
            "tempering_temp": [150.0, 650.0],
        },
    }
    warnings = run_all_patterns(ctx, phase=Phase.PREPROCESSING)
    ids = {w["pattern_id"] for w in warnings}
    assert "D07" in ids


def test_dataset_for_each_class_contains_full_feature_set():
    """Regression: build_context for HSLA crashed because raw parquet
    lacked engineered features (cev_iiw, pcm, ...). The dataset returned
    via the registry must be feature-engineered to match what models
    actually train on, otherwise inference at sample-prediction time
    raises a feature_names mismatch.
    """
    for class_id in AVAILABLE_CLASS_IDS:
        profile = load_steel_class(class_id)
        gen = get_synthetic_generator(profile.synthetic_generator_name)
        df_raw = gen()
        df_feat = compute_features_for_class(df_raw, class_id)

        missing = [f for f in profile.feature_set if f not in df_feat.columns]
        assert not missing, (
            f"class {class_id}: feature_set missing from dataset after "
            f"feature engineering: {missing}"
        )
