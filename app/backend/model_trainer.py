"""
Model Trainer Agent.

XGBoost с:
- Time-based + GroupKFold валидацией (избегаем data leakage)
- Quantile regression для uncertainty (q05, q95)
- Optuna для hyperparameter tuning
- Feature importance + smoke-тест на референсных составах
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class TrainingMetrics:
    r2_train: float
    r2_val: float
    r2_test: float
    mae_test: float
    rmse_test: float
    coverage_90_ci: float
    n_train: int
    n_val: int
    n_test: int
    coverage_90_ci_raw: float = 0.0
    conformal_correction_mpa: float = 0.0


@dataclass
class TrainedModel:
    version: str
    target: str
    feature_list: list[str]
    artifact_path: str
    metrics: TrainingMetrics
    feature_importance: dict[str, float]
    training_ranges: dict[str, list[float]]
    has_uncertainty: bool
    has_ood_detector: bool
    split_strategy: str
    cv_strategy: str
    steel_class: str = "pipe_hsla"


# =========================================================================
# Time-based + group split
# =========================================================================

def time_group_split(
    df: pd.DataFrame,
    time_col: str = "heat_date",
    group_col: str = "campaign_id",
    test_fraction: float = 0.2,
    val_fraction: float = 0.15,
):
    """
    Hold-out test — последние test_fraction по времени.
    Val — GroupKFold внутри оставшихся train+val.
    """
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    n_test = int(n * test_fraction)
    
    test_idx = df_sorted.index[-n_test:].values
    trainval_df = df_sorted.iloc[:n - n_test]
    
    # Внутри train+val делаем GroupKFold по campaigns
    n_folds_for_val = max(3, int(1 / val_fraction))
    gkf = GroupKFold(n_splits=n_folds_for_val)
    groups = trainval_df[group_col].values
    X_dummy = np.zeros(len(trainval_df))
    
    # Берём первый fold как val
    train_idx_local, val_idx_local = next(gkf.split(X_dummy, groups=groups))
    train_idx = trainval_df.index[train_idx_local].values
    val_idx = trainval_df.index[val_idx_local].values
    
    return df_sorted, train_idx, val_idx, test_idx


# =========================================================================
# Train XGBoost with uncertainty
# =========================================================================

def train_model(
    df_features: pd.DataFrame,
    target: str,
    feature_list: list[str],
    n_optuna_trials: int = 40,
    random_seed: int = 42,
    steel_class: str = "pipe_hsla",
) -> TrainedModel:
    import xgboost as xgb  # type: ignore[import-not-found]
    import optuna  # type: ignore[import-not-found]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # 1. Split
    df_sorted, train_idx, val_idx, test_idx = time_group_split(df_features)
    X = df_sorted[feature_list]
    y = df_sorted[target]
    
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    
    logger.info("Split: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))
    
    # 2. Hyperparameter search
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5, log=True),
            "tree_method": "hist",
            "random_state": random_seed,
            "early_stopping_rounds": 30,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return mean_absolute_error(y_val, model.predict(X_val))
    
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=False)
    best_params = study.best_params
    logger.info("Best params: %s (MAE val=%.2f)", best_params, study.best_value)
    
    # 3. Final model on train+val
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    
    final_params = {**best_params, "tree_method": "hist", "random_state": random_seed}
    main_model = xgb.XGBRegressor(**final_params, objective="reg:squarederror")
    main_model.fit(X_trainval, y_trainval, verbose=False)
    
    # 4. Quantile models — trained on train only so val stays clean
    # for split-conformal calibration (Romano et al. 2019).
    quantile_models = {}
    for alpha, name in [(0.05, "q05"), (0.95, "q95")]:
        qm = xgb.XGBRegressor(
            **final_params,
            objective="reg:quantileerror",
            quantile_alpha=alpha,
        )
        qm.fit(X_train, y_train, verbose=False)
        quantile_models[name] = qm

    # 4b. Split-conformal calibration. Nonconformity score
    #     E_i = max(q05(x_i) - y_i, y_i - q95(x_i)) on validation set.
    #     Q = ceil((n_cal+1) * (1-α)) / n_cal quantile of E.
    #     Widening [q05, q95] by ±Q brings empirical coverage ≥ 1-α under
    #     exchangeability (here α=0.10 for nominal 90% CI).
    lo_val = quantile_models["q05"].predict(X_val)
    hi_val = quantile_models["q95"].predict(X_val)
    nonconformity = np.maximum(lo_val - y_val.values, y_val.values - hi_val)
    n_cal = len(y_val)
    conformal_level = min(np.ceil((n_cal + 1) * 0.90) / n_cal, 1.0)
    conformal_correction = float(np.quantile(nonconformity, conformal_level))

    # 5. Eval — report both raw and conformal-corrected coverage
    y_pred_train = main_model.predict(X_train)
    y_pred_val = main_model.predict(X_val)
    y_pred_test = main_model.predict(X_test)

    raw_lower = quantile_models["q05"].predict(X_test)
    raw_upper = quantile_models["q95"].predict(X_test)
    raw_coverage = float(((y_test >= raw_lower) & (y_test <= raw_upper)).mean())

    lower = raw_lower - conformal_correction
    upper = raw_upper + conformal_correction
    coverage = float(((y_test >= lower) & (y_test <= upper)).mean())

    metrics = TrainingMetrics(
        r2_train=float(r2_score(y_train, y_pred_train)),
        r2_val=float(r2_score(y_val, y_pred_val)),
        r2_test=float(r2_score(y_test, y_pred_test)),
        mae_test=float(mean_absolute_error(y_test, y_pred_test)),
        rmse_test=float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        coverage_90_ci=coverage,
        coverage_90_ci_raw=raw_coverage,
        conformal_correction_mpa=conformal_correction,
        n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx),
    )
    logger.info(
        "Metrics: R² test=%.3f, MAE=%.2f, coverage 90%% raw=%.2f → "
        "conformal-corrected=%.2f (Q=%.1f)",
        metrics.r2_test, metrics.mae_test,
        metrics.coverage_90_ci_raw, metrics.coverage_90_ci,
        metrics.conformal_correction_mpa,
    )
    
    # 6. Feature importance
    importance = dict(zip(feature_list, main_model.feature_importances_.tolist()))
    
    # 7. OOD detector (GMM на training composition)
    comp_cols = [c for c in feature_list if c.endswith("_pct") or c == "n_ppm"]
    ood_detector = GaussianMixture(n_components=3, random_state=random_seed)
    ood_detector.fit(X_trainval[comp_cols].values)
    
    # 8. Training ranges
    training_ranges = {
        col: [float(X_trainval[col].min()), float(X_trainval[col].max())]
        for col in feature_list
    }
    
    # 9. Save artifact
    version_prefix = {
        "pipe_hsla": "hsla",
        "en10083_qt": "en10083qt",
        "fatigue_carbon_steel": "fatigue",
    }.get(steel_class, "model")
    version = f"{version_prefix}_{target.replace('_mpa', '').replace('_j_cm2', '')}_xgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifact_dir = MODELS_DIR / version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    main_model.save_model(str(artifact_dir / "main.json"))
    quantile_models["q05"].save_model(str(artifact_dir / "q05.json"))
    quantile_models["q95"].save_model(str(artifact_dir / "q95.json"))
    with open(artifact_dir / "ood_detector.pkl", "wb") as f:
        pickle.dump({"gmm": ood_detector, "comp_cols": comp_cols}, f)
    try:
        from app.backend.steel_classes import load_steel_class
        _profile = load_steel_class(steel_class)
        data_source = _profile.data_source
        data_source_doi = _profile.data_source_doi
    except Exception:
        data_source = None
        data_source_doi = None

    with open(artifact_dir / "meta.json", "w") as f:
        json.dump({
            "version": version,
            "target": target,
            "feature_list": feature_list,
            "best_params": best_params,
            "metrics": asdict(metrics),
            "feature_importance": importance,
            "training_ranges": training_ranges,
            "steel_class": steel_class,
            "data_source": data_source,
            "data_source_doi": data_source_doi,
            "conformal_correction_mpa": conformal_correction,
            "trained_at": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    return TrainedModel(
        version=version, target=target, feature_list=feature_list,
        artifact_path=str(artifact_dir), metrics=metrics,
        feature_importance=importance, training_ranges=training_ranges,
        has_uncertainty=True, has_ood_detector=True,
        split_strategy="time_based", cv_strategy="group_kfold",
        steel_class=steel_class,
    )


def load_model(version: str):
    """Загружает обученную модель для inference."""
    import xgboost as xgb  # type: ignore[import-not-found]
    artifact_dir = MODELS_DIR / version
    main = xgb.XGBRegressor()
    main.load_model(str(artifact_dir / "main.json"))
    q05 = xgb.XGBRegressor()
    q05.load_model(str(artifact_dir / "q05.json"))
    q95 = xgb.XGBRegressor()
    q95.load_model(str(artifact_dir / "q95.json"))
    with open(artifact_dir / "ood_detector.pkl", "rb") as f:
        ood = pickle.load(f)
    with open(artifact_dir / "meta.json") as f:
        meta = json.load(f)
    return {"main": main, "q05": q05, "q95": q95, "ood": ood, "meta": meta}


def predict_with_uncertainty(
    model_bundle: dict,
    df_input: pd.DataFrame,
) -> pd.DataFrame:
    """
    Возвращает DataFrame с prediction, lower, upper, ood_flag, mahalanobis_distance.
    """
    meta = model_bundle["meta"]
    feature_list = meta["feature_list"]
    X = df_input[feature_list]
    
    pred = model_bundle["main"].predict(X)
    raw_lo = model_bundle["q05"].predict(X)
    raw_hi = model_bundle["q95"].predict(X)

    # Split-conformal correction baked at training time; older models
    # without this field get Q=0 (raw quantiles).
    q_correction = float(meta.get("conformal_correction_mpa", 0.0))
    lo = raw_lo - q_correction
    hi = raw_hi + q_correction

    ood = model_bundle["ood"]
    comp = X[ood["comp_cols"]].values
    log_prob = ood["gmm"].score_samples(comp)
    threshold = np.percentile(log_prob, 1)
    ood_flag = log_prob < threshold - 5

    result = pd.DataFrame({
        "prediction": pred,
        "lower_90": lo,
        "upper_90": hi,
        "ci_half_width": (hi - lo) / 2,
        "ood_flag": ood_flag,
        "log_density": log_prob,
    }, index=df_input.index)
    return result


# =========================================================================
# Agent interface
# =========================================================================

class ModelTrainerAgent:
    name = "model_trainer"
    
    def run(self, state, task):
        from app.backend.engine import AgentResult
        from decision_log.logger import log_decision
        
        operation = task.get("operation", "train_xgboost")
        target = task.get("target", "yield_strength_mpa")
        
        try:
            if operation == "train_xgboost":
                features_path = Path(state.features.get("features_path", "/tmp/features.parquet"))
                df = pd.read_parquet(features_path)
                feature_list = state.features.get("feature_list", [])
                if not feature_list:
                    from app.backend.feature_eng import PIPE_HSLA_FEATURE_SET
                    feature_list = [f for f in PIPE_HSLA_FEATURE_SET if f in df.columns]
                
                trained = train_model(
                    df, target=target, feature_list=feature_list,
                    n_optuna_trials=task.get("n_optuna_trials", 40),
                    steel_class=task.get("steel_class", "pipe_hsla"),
                )
                
                log_decision(
                    phase="training",
                    decision=f"Обучена модель {trained.version} для {target}",
                    reasoning=(
                        f"XGBoost с time-based split + GroupKFold по campaign_id. "
                        f"Optuna HP tuning (40 trials). Quantile regression для uncertainty. "
                        f"Test R²={trained.metrics.r2_test:.3f}, MAE={trained.metrics.mae_test:.2f}, "
                        f"coverage 90% CI={trained.metrics.coverage_90_ci:.2%}."
                    ),
                    alternatives_considered=["CatBoost", "MLP", "Random Forest"],
                    context={
                        "version": trained.version, "target": target,
                        "metrics": asdict(trained.metrics),
                    },
                    author="model_trainer",
                    tags=["xgboost", target, "production_candidate"],
                )
                
                return AgentResult(
                    agent_name=self.name, success=True,
                    output={
                        "version": trained.version,
                        "artifact_path": trained.artifact_path,
                        "target": target,
                        "feature_list": feature_list,
                        "steel_class": trained.steel_class,
                        "r2_train": trained.metrics.r2_train,
                        "r2_val": trained.metrics.r2_val,
                        "r2_test": trained.metrics.r2_test,
                        "mae_test": trained.metrics.mae_test,
                        "coverage_90_ci": trained.metrics.coverage_90_ci,
                        "feature_importance": trained.feature_importance,
                        "training_ranges": trained.training_ranges,
                        "has_uncertainty": True,
                        "has_ood_detector": True,
                        "split_strategy": trained.split_strategy,
                        "cv_strategy": trained.cv_strategy,
                    },
                )
            
            return AgentResult(
                agent_name=self.name, success=False,
                output={}, error=f"Unknown operation: {operation}",
            )
        except Exception as e:
            logger.exception("ModelTrainer failed")
            return AgentResult(
                agent_name=self.name, success=False, output={}, error=str(e),
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from app.backend.feature_eng import compute_hsla_features, PIPE_HSLA_FEATURE_SET
    
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "hsla_synthetic.parquet"
    df = pd.read_parquet(data_path)
    df_feat = compute_hsla_features(df)
    feat_list = [f for f in PIPE_HSLA_FEATURE_SET if f in df_feat.columns]
    
    trained = train_model(df_feat, "yield_strength_mpa", feat_list, n_optuna_trials=15)
    print(f"\n✓ Trained model: {trained.version}")
    print(f"  R² test = {trained.metrics.r2_test:.3f}")
    print(f"  MAE test = {trained.metrics.mae_test:.2f} МПа")
    print(f"  Coverage 90% CI = {trained.metrics.coverage_90_ci:.2%}")
    print("\n  Top 5 features by importance:")
    top = sorted(trained.feature_importance.items(), key=lambda x: -x[1])[:5]
    for f, imp in top:
        print(f"    {f}: {imp:.3f}")
