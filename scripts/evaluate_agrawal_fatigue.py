"""
Path 2 · A · Step 2 / 2

Trains XGBoost on the Agrawal 2014 NIMS fatigue dataset (437 records) under
four configurations to answer: can our pipeline predict fatigue strength
of carbon/low-alloy steels from real NIMS records?

  A. all_classes       — train + test on all 437 records (paper-analog)
  B. carbon_la_only    — train + test on 338 carbon/low-alloy subset
  C. stratified_4fold  — train 3 sub-classes / test 4th (generalization check)
  D. composition_only  — drop all processing + inclusion features,
                         composition → fatigue (lower bound)

Mirrors the Phase 0.1 evaluation pattern (scripts/evaluate_public_data.py)
but targets fatigue_strength_mpa instead of yield_strength_mpa.

Run:
    PYTHONPATH=. .venv/bin/python scripts/evaluate_agrawal_fatigue.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "agrawal_nims_fatigue.parquet"
METRICS_OUT = PROJECT_ROOT / "docs" / "path2_agrawal_metrics.json"

TARGET = "fatigue_strength_mpa"
COMPOSITION = [
    "c_pct", "si_pct", "mn_pct", "p_pct", "s_pct",
    "ni_pct", "cr_pct", "cu_pct", "mo_pct",
]
PROCESSING = [
    "normalizing_temp_c",
    "through_hardening_temp_c", "through_hardening_time_min",
    "through_hardening_cooling_rate_c_per_s",
    "carburizing_temp_c", "carburizing_time_min",
    "diffusion_temp_c", "diffusion_time_min",
    "quenching_media_temp_c",
    "tempering_temp_c", "tempering_time_min",
    "tempering_cooling_rate_c_per_s",
    "reduction_ratio",
]
INCLUSIONS = [
    "inclusion_area_defect_a",
    "inclusion_area_defect_b",
    "inclusion_area_defect_c",
]
ALL_FEATURES = COMPOSITION + PROCESSING + INCLUSIONS
RANDOM_SEED = 42


def _fit_and_score(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    name: str, feature_list: list[str],
) -> dict:
    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train[feature_list], y_train)
    y_pred = model.predict(X_test[feature_list])

    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    print(f"[{name:30s}] n_train={len(X_train):4d} n_test={len(X_test):4d} "
          f"R²={r2:+.3f} MAE={mae:6.1f} MPa RMSE={rmse:6.1f} MPa")

    importances = dict(sorted(
        zip(feature_list, model.feature_importances_.tolist()),
        key=lambda kv: -kv[1],
    )[:10])

    return {
        "config": name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": len(feature_list),
        "r2": r2,
        "mae_mpa": mae,
        "rmse_mpa": rmse,
        "y_test_mean_mpa": float(y_test.mean()),
        "y_test_range_mpa": [float(y_test.min()), float(y_test.max())],
        "top10_importance": importances,
    }


def config_a_all(df: pd.DataFrame) -> dict:
    train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    return _fit_and_score(
        train, train[TARGET], test, test[TARGET],
        "A.all_classes", ALL_FEATURES,
    )


def config_b_carbon_la(df: pd.DataFrame) -> dict:
    subset = df[df["sub_class"] == "carbon_low_alloy"].reset_index(drop=True)
    train, test = train_test_split(subset, test_size=0.2, random_state=RANDOM_SEED)
    return _fit_and_score(
        train, train[TARGET], test, test[TARGET],
        "B.carbon_la_only", ALL_FEATURES,
    )


def config_c_stratified(df: pd.DataFrame) -> dict:
    """Train on 2 of 3 sub-classes, test on the 3rd (3 rotations, averaged)."""
    classes = df["sub_class"].unique().tolist()
    per_fold = []
    for held_out in classes:
        train_df = df[df["sub_class"] != held_out]
        test_df = df[df["sub_class"] == held_out]
        if len(test_df) < 10 or len(train_df) < 50:
            continue
        m = _fit_and_score(
            train_df, train_df[TARGET], test_df, test_df[TARGET],
            f"C.held_out={held_out}", ALL_FEATURES,
        )
        per_fold.append(m)
    avg_r2 = float(np.mean([m["r2"] for m in per_fold]))
    avg_mae = float(np.mean([m["mae_mpa"] for m in per_fold]))
    print(f"[C.stratified_avg             ] folds={len(per_fold)} "
          f"avg R²={avg_r2:+.3f} avg MAE={avg_mae:6.1f} MPa")
    return {
        "config": "C.stratified_cross_sub_class",
        "folds": per_fold,
        "avg_r2": avg_r2,
        "avg_mae_mpa": avg_mae,
    }


def config_d_composition_only(df: pd.DataFrame) -> dict:
    train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    return _fit_and_score(
        train, train[TARGET], test, test[TARGET],
        "D.composition_only", COMPOSITION,
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} missing — run scripts/fetch_agrawal_nims_fatigue.py first"
        )

    df = pd.read_parquet(DATA_PATH)
    print(f"[eval] loaded {len(df)} records, target range "
          f"{df[TARGET].min():.0f}-{df[TARGET].max():.0f} MPa\n")

    results = {
        "dataset": "agrawal_2014_nims_fatigue",
        "n_records": int(len(df)),
        "sub_class_counts": df["sub_class"].value_counts().to_dict(),
        "configs": {},
    }
    results["configs"]["A"] = config_a_all(df)
    print()
    results["configs"]["B"] = config_b_carbon_la(df)
    print()
    results["configs"]["C"] = config_c_stratified(df)
    print()
    results["configs"]["D"] = config_d_composition_only(df)

    METRICS_OUT.parent.mkdir(exist_ok=True)
    METRICS_OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=int))
    print(f"\n[eval] metrics written to {METRICS_OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
