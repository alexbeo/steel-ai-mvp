"""
Path 2 · B' · Step 2 / 2

Trains XGBoost on the MPEA dataset (1067 records with YS present, 630 unique
HEA formulas) under four configurations to stress-test whether our pipeline
architecture generalizes from steel (synthetic HSLA / matminer / Agrawal)
to an entirely different alloy family — multi-principal-element alloys.

  A. ys_composition_temp  — all 1067 YS records,
                             features = 15 mol_frac + test_temperature_c
  B. ys_tension_only      — restrict to tension tests (~200 records),
                             exclude compression (different strength regime)
  C. ys_bcc_phase_only    — BCC-only slice (often Fe-rich refractory HEAs,
                             closer in spirit to steel than FCC Cantor-type)
  D. hardness_from_comp   — predict hardness_hv from composition only
                             (different target, tests multi-property
                             generality of the feature space)

Mirrors Phase 0.1 / evaluate_agrawal pattern.

Run:
    PYTHONPATH=. .venv/bin/python scripts/evaluate_mpea.py
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
DATA_PATH = PROJECT_ROOT / "data" / "mpea_dataset.parquet"
METRICS_OUT = PROJECT_ROOT / "docs" / "path2_mpea_metrics.json"

MOL_FRAC_COLS = [
    "al_mol_frac", "co_mol_frac", "cr_mol_frac", "cu_mol_frac", "fe_mol_frac",
    "hf_mol_frac", "mn_mol_frac", "mo_mol_frac", "nb_mol_frac", "ni_mol_frac",
    "ta_mol_frac", "ti_mol_frac", "v_mol_frac", "w_mol_frac", "zr_mol_frac",
]
RANDOM_SEED = 42


def _fit_and_score(
    train: pd.DataFrame, test: pd.DataFrame,
    features: list[str], target: str, name: str,
) -> dict:
    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(train[features], train[target])
    y_pred = model.predict(test[features])
    y_test = test[target]

    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    print(f"[{name:30s}] n_train={len(train):4d} n_test={len(test):4d} "
          f"R²={r2:+.3f} MAE={mae:6.1f} RMSE={rmse:6.1f}")

    importances = dict(sorted(
        zip(features, model.feature_importances_.tolist()),
        key=lambda kv: -kv[1],
    )[:10])
    return {
        "config": name,
        "target": target,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": len(features),
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "y_test_mean": float(y_test.mean()),
        "y_test_range": [float(y_test.min()), float(y_test.max())],
        "top10_importance": importances,
    }


def config_a_ys_all(df: pd.DataFrame) -> dict:
    subset = df[df["yield_strength_mpa"].notna() & df["test_temperature_c"].notna()].copy()
    subset["test_temperature_c"] = subset["test_temperature_c"].fillna(25.0)
    features = MOL_FRAC_COLS + ["test_temperature_c"]
    train, test = train_test_split(subset, test_size=0.2, random_state=RANDOM_SEED)
    return _fit_and_score(train, test, features, "yield_strength_mpa", "A.ys_comp_plus_temp")


def config_b_ys_tension(df: pd.DataFrame) -> dict:
    subset = df[
        df["yield_strength_mpa"].notna()
        & df["test_temperature_c"].notna()
        & (df["test_type"] == "T")
    ].copy()
    features = MOL_FRAC_COLS + ["test_temperature_c"]
    train, test = train_test_split(subset, test_size=0.2, random_state=RANDOM_SEED)
    return _fit_and_score(train, test, features, "yield_strength_mpa", "B.ys_tension_only")


def config_c_ys_bcc(df: pd.DataFrame) -> dict:
    subset = df[
        df["yield_strength_mpa"].notna()
        & df["test_temperature_c"].notna()
        & df["phase"].str.contains("BCC", na=False, case=False)
    ].copy()
    features = MOL_FRAC_COLS + ["test_temperature_c"]
    train, test = train_test_split(subset, test_size=0.2, random_state=RANDOM_SEED)
    return _fit_and_score(train, test, features, "yield_strength_mpa", "C.ys_bcc_phase_only")


def config_d_hardness(df: pd.DataFrame) -> dict:
    subset = df[df["hardness_hv"].notna()].copy()
    features = MOL_FRAC_COLS
    train, test = train_test_split(subset, test_size=0.2, random_state=RANDOM_SEED)
    return _fit_and_score(train, test, features, "hardness_hv", "D.hardness_from_comp")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} missing — run scripts/fetch_mpea_dataset.py first"
        )
    df = pd.read_parquet(DATA_PATH)
    print(f"[eval] loaded {len(df)} records, {df['formula'].nunique()} unique formulas\n")

    results = {
        "dataset": "citrine_mpea_borg_2020",
        "n_records": int(len(df)),
        "n_unique_formulas": int(df["formula"].nunique()),
        "configs": {},
    }
    results["configs"]["A"] = config_a_ys_all(df)
    print()
    results["configs"]["B"] = config_b_ys_tension(df)
    print()
    results["configs"]["C"] = config_c_ys_bcc(df)
    print()
    results["configs"]["D"] = config_d_hardness(df)

    METRICS_OUT.parent.mkdir(exist_ok=True)
    METRICS_OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=int))
    print(f"\n[eval] metrics written to {METRICS_OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
