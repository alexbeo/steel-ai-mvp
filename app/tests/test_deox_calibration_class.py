"""Tests for virtual steel class deox_calibration (R-004 PR 4)."""
from __future__ import annotations

import pandas as pd

from app.backend.steel_classes import (
    AVAILABLE_CLASS_IDS,
    get_synthetic_generator,
    load_steel_class,
)


def test_in_registry():
    assert "deox_calibration" in AVAILABLE_CLASS_IDS


def test_yaml_loads():
    profile = load_steel_class("deox_calibration")
    assert profile.id == "deox_calibration"
    # target_properties — list[TargetProperty]; eta_al_effective должен быть id
    target_ids = [t.id for t in profile.target_properties]
    assert "eta_al_effective" in target_ids


def test_get_synthetic_generator_resolves():
    gen = get_synthetic_generator("deox_calibration")
    assert callable(gen)


def test_synthetic_generator_returns_dataframe():
    gen = get_synthetic_generator("deox_calibration")
    df = gen(n_heats=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100


def test_synthetic_target_in_bounds():
    gen = get_synthetic_generator("deox_calibration")
    df = gen(n_heats=200)
    assert (df["eta_al_effective"] >= 0.30).all()
    assert (df["eta_al_effective"] <= 0.99).all()


def test_synthetic_includes_campaign_and_date():
    gen = get_synthetic_generator("deox_calibration")
    df = gen(n_heats=50)
    assert "campaign_id" in df.columns
    assert "heat_date" in df.columns
    assert df["campaign_id"].nunique() >= 1
    assert df["heat_date"].is_monotonic_increasing


def test_synthetic_multi_plant():
    gen = get_synthetic_generator("deox_calibration")
    df = gen(n_heats=200)
    assert df["plant_id"].nunique() == 3
    assert set(df["plant_id"].unique()) == {"PLANT_A", "PLANT_B", "PLANT_C"}


def test_synthetic_uses_multiple_methods():
    gen = get_synthetic_generator("deox_calibration")
    df = gen(n_heats=200)
    # Из 5 методов в YAML каталоге, generator должен использовать >=4
    assert df["method_id"].nunique() >= 4


def test_feature_set_subset_of_dataframe_columns():
    profile = load_steel_class("deox_calibration")
    gen = get_synthetic_generator("deox_calibration")
    df = gen(n_heats=50)
    missing = [f for f in profile.feature_set if f not in df.columns]
    assert not missing, f"Features in profile but not in DataFrame: {missing}"


def test_physical_bounds_cover_feature_set():
    profile = load_steel_class("deox_calibration")
    missing = [f for f in profile.feature_set if f not in profile.physical_bounds]
    assert not missing, f"Features without physical_bounds: {missing}"


def test_feature_set_is_numeric():
    profile = load_steel_class("deox_calibration")
    gen = get_synthetic_generator("deox_calibration")
    df = gen(n_heats=50)
    numeric_subset = df[profile.feature_set].select_dtypes(include="number")
    assert numeric_subset.shape[1] == len(profile.feature_set), (
        f"Не все features numeric: "
        f"{set(profile.feature_set) - set(numeric_subset.columns)}"
    )


def test_eta_correlates_with_method_baseline():
    """Signal sanity — η_Al должен коррелировать с method baseline."""
    gen = get_synthetic_generator("deox_calibration")
    df = gen(n_heats=500)
    corr = df[["method_eta_baseline", "eta_al_effective"]].corr().iloc[0, 1]
    assert corr > 0.4, f"Method baseline corr = {corr:.3f}, expected > 0.4"


def test_eta_anticorrelates_with_feo():
    """FeO в шлаке должен снижать η_Al."""
    gen = get_synthetic_generator("deox_calibration")
    df = gen(n_heats=500)
    corr = df[["slag_feo_pct", "eta_al_effective"]].corr().iloc[0, 1]
    assert corr < -0.10, f"FeO corr = {corr:.3f}, expected < -0.10"


def test_expected_top_features_in_feature_set():
    profile = load_steel_class("deox_calibration")
    missing = [
        f for f in profile.expected_top_features if f not in profile.feature_set
    ]
    assert not missing, f"expected_top_features не в feature_set: {missing}"


def test_no_inverse_design_required():
    """target_o_activity_ppm не указан — должен быть None (поле опц.)."""
    profile = load_steel_class("deox_calibration")
    assert profile.target_o_activity_ppm is None


# ─────────────────────────────────────────────────────────────────────
# PR 8 — End-to-end Orchestrator integration tests
# ─────────────────────────────────────────────────────────────────────


def _build_orchestrator():
    """Helper: собирает Orchestrator с минимальным набором агентов
    (без inverse_designer/validator/reporter — process feedback classes их
    skip-ают).
    """
    from app.backend.engine import Orchestrator, Critic
    from app.backend.data_curator import DataCuratorAgent
    from app.backend.feature_eng import FeatureEngAgent
    from app.backend.model_trainer import ModelTrainerAgent

    agents = {
        "data_curator": DataCuratorAgent(),
        "feature_eng": FeatureEngAgent(),
        "model_trainer": ModelTrainerAgent(),
    }
    return Orchestrator(
        agents=agents,
        critic=Critic(use_llm=False),
        human_in_the_loop=False,
    )


def test_engine_trains_deox_calibration_model(tmp_path, monkeypatch):
    """End-to-end: Orchestrator обучает модель η_Al с R² > 0.5."""
    # Изоляция: DATA_DIR / MODELS_DIR должны указывать на tmp_path, иначе
    # тест мусорит в data/ и models/ репозитория.
    from app.backend import data_curator as dc_mod
    from app.backend import model_trainer as mt_mod

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(dc_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(mt_mod, "MODELS_DIR", models_dir)

    orch = _build_orchestrator()
    user_request = {
        "steel_class": "deox_calibration",
        "target_property": "eta_al_effective",
        "n_rows": 500,
        "n_optuna_trials": 3,  # fast для теста
    }
    state = orch.run_pipeline(user_request=user_request)

    assert state.model, "Orchestrator не обучил модель"
    r2_test = state.model.get("r2_test", 0.0)
    assert r2_test > 0.5, f"R² test = {r2_test:.3f}, ожидается > 0.5"

    # No BLOCK verdicts
    for report in state.critic_reports:
        verdict_name = (
            report.verdict.name
            if hasattr(report.verdict, "name")
            else str(report.verdict)
        )
        assert verdict_name != "BLOCK", (
            f"Critic BLOCK на фазе {report.phase}: {report.warnings}"
        )


def test_engine_skips_inverse_design_for_deox(tmp_path, monkeypatch):
    """Orchestrator не запускает inverse_design для deox_calibration."""
    from app.backend import data_curator as dc_mod
    from app.backend import model_trainer as mt_mod

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(dc_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(mt_mod, "MODELS_DIR", models_dir)

    orch = _build_orchestrator()
    # n_rows: synthetic generator группирует 40 heats / campaign,
    # GroupKFold нужно ≥6 campaign после 20% test hold-out.
    user_request = {
        "steel_class": "deox_calibration",
        "target_property": "eta_al_effective",
        "n_rows": 400,
        "n_optuna_trials": 2,
    }
    state = orch.run_pipeline(user_request=user_request)

    # Кандидатов inverse_design быть не должно (фаза не запускалась).
    assert state.candidates == [], (
        f"Ожидали 0 candidates, получили {len(state.candidates)}"
    )
    assert state.validated_candidates == []
    # Critic reports — только для запущенных фаз
    # (data_acquisition + feature_engineering + training).
    phases_reviewed = {r.phase for r in state.critic_reports}
    assert "inverse_design" not in phases_reviewed
    assert "validation" not in phases_reviewed
    assert "reporting" not in phases_reviewed
    assert "training" in phases_reviewed
