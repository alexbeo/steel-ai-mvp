"""
Public-data spike · Phase 0.1 · Step 2 / 3

Trains XGBoost regressors under three configurations and reports metrics:

  A. synthetic_only    — train + test on synthetic HSLA (baseline)
  B. public_only       — train + test on matminer public (312 records)
  C. merged_train_public_holdout
                       — train on (synthetic + half-public),
                         test on unseen half of public
                         (closest analog to cross-source generalization)
  D. synthetic_on_public
                       — model trained ONLY on synthetic, predicts public data
                         (measures how far apart the two steel-class populations are)

For each config we report R², MAE, RMSE on a held-out split.
Also emits JSON with metrics to docs/ for the final report.

Run:
    PYTHONPATH=. .venv/bin/python scripts/evaluate_public_data.py
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

from app.backend.data_curator import generate_synthetic_hsla_dataset

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
DOCS_DIR = Path(__file__).parent.parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# Feature set present in both synthetic HSLA and public matminer.
# Selected as intersection + zero-fill for class-specific alloys.
COMMON_FEATURES = [
    "c_pct", "mn_pct", "si_pct",
    "cr_pct", "ni_pct", "mo_pct",
    "v_pct", "nb_pct", "al_pct", "ti_pct",
    "co_pct", "w_pct", "n_pct",
]
TARGET = "yield_strength_mpa"
RANDOM_SEED = 42


def _align_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """Synthetic generator doesn't have Co, W, N%: add them as zero/derived."""
    df = df.copy()
    if "co_pct" not in df.columns:
        df["co_pct"] = 0.0
    if "w_pct" not in df.columns:
        df["w_pct"] = 0.0
    # synthetic has n_ppm; convert to wt% (1 ppm = 1e-4 wt%)
    if "n_pct" not in df.columns and "n_ppm" in df.columns:
        df["n_pct"] = df["n_ppm"] / 10000.0
    elif "n_pct" not in df.columns:
        df["n_pct"] = 0.005  # default plausible value
    return df


def _train_and_score(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame,  y_test: pd.Series,
    name: str,
) -> dict:
    """Train XGBoost, score on test, return metrics dict."""
    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "config": name,
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "r2":      float(r2_score(y_test, y_pred)),
        "mae":     float(mean_absolute_error(y_test, y_pred)),
        "rmse":    float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "y_test_mean":  float(np.mean(y_test)),
        "y_test_range": [float(np.min(y_test)), float(np.max(y_test))],
        "y_pred_mean":  float(np.mean(y_pred)),
        "y_pred_range": [float(np.min(y_pred)), float(np.max(y_pred))],
    }
    return metrics


def _print_row(m: dict) -> None:
    print(f"  {m['config']:<38} "
          f"n_train={m['n_train']:>4}  n_test={m['n_test']:>3}  "
          f"R²={m['r2']:+.3f}  MAE={m['mae']:>6.1f} MPa  "
          f"RMSE={m['rmse']:>6.1f}")


def main() -> None:
    print("Phase 0.1 · Step 2 of 3 — evaluate public data utility")
    print("=" * 80)

    # ----------------- Load -----------------
    print("\n[load] generating synthetic HSLA (2500 heats)...")
    syn = generate_synthetic_hsla_dataset(n_samples=2500, random_seed=RANDOM_SEED)
    syn = _align_synthetic(syn)
    print(f"[load] synthetic: {len(syn)} rows, target range "
          f"{syn[TARGET].min():.0f}..{syn[TARGET].max():.0f} MPa")

    pub_path = DATA_DIR / "public_matminer.parquet"
    if not pub_path.exists():
        raise SystemExit(
            f"Public data not found at {pub_path}. "
            "Run scripts/fetch_public_steel_data.py first."
        )
    pub = pd.read_parquet(pub_path)
    print(f"[load] public:    {len(pub)} rows, target range "
          f"{pub[TARGET].min():.0f}..{pub[TARGET].max():.0f} MPa")

    # Verify feature set
    for col in COMMON_FEATURES:
        if col not in syn.columns:
            syn[col] = 0.0
        if col not in pub.columns:
            pub[col] = 0.0

    # ----------------- Config A: synthetic only -----------------
    print("\n" + "=" * 80)
    print("Config A · synthetic_only (baseline)")
    print("-" * 80)
    Xa, ya = syn[COMMON_FEATURES], syn[TARGET]
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
        Xa, ya, test_size=0.2, random_state=RANDOM_SEED)
    m_a = _train_and_score(Xa_tr, ya_tr, Xa_te, ya_te, "A · synthetic_only")
    _print_row(m_a)

    # ----------------- Config B: public only -----------------
    print("\n" + "=" * 80)
    print("Config B · public_only (can XGBoost even fit this distribution?)")
    print("-" * 80)
    Xb, yb = pub[COMMON_FEATURES], pub[TARGET]
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
        Xb, yb, test_size=0.25, random_state=RANDOM_SEED)
    m_b = _train_and_score(Xb_tr, yb_tr, Xb_te, yb_te, "B · public_only")
    _print_row(m_b)

    # ----------------- Config C: merged train, public hold-out -----------------
    print("\n" + "=" * 80)
    print("Config C · merged_train, public_holdout (augmentation test)")
    print("-" * 80)
    # Split public first to keep held-out clean of any train leakage
    pub_train, pub_holdout = train_test_split(
        pub, test_size=0.3, random_state=RANDOM_SEED)
    merged_train = pd.concat(
        [syn[COMMON_FEATURES + [TARGET]], pub_train[COMMON_FEATURES + [TARGET]]],
        ignore_index=True)
    Xc_tr, yc_tr = merged_train[COMMON_FEATURES], merged_train[TARGET]
    Xc_te, yc_te = pub_holdout[COMMON_FEATURES], pub_holdout[TARGET]
    m_c = _train_and_score(Xc_tr, yc_tr, Xc_te, yc_te,
                           "C · merged_train / public_holdout")
    _print_row(m_c)

    # ----------------- Config D: synthetic-only, applied to public -----------------
    print("\n" + "=" * 80)
    print("Config D · synthetic_only, predicts_public (class-transfer test)")
    print("-" * 80)
    model_synth = XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        random_state=RANDOM_SEED, n_jobs=-1, tree_method="hist",
    )
    model_synth.fit(syn[COMMON_FEATURES], syn[TARGET])
    y_pred = model_synth.predict(pub[COMMON_FEATURES])
    y_true = pub[TARGET].values

    m_d = {
        "config": "D · synthetic_only → public_all",
        "n_train": int(len(syn)),
        "n_test":  int(len(pub)),
        "r2":      float(r2_score(y_true, y_pred)),
        "mae":     float(mean_absolute_error(y_true, y_pred)),
        "rmse":    float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "y_test_mean":  float(np.mean(y_true)),
        "y_test_range": [float(np.min(y_true)), float(np.max(y_true))],
        "y_pred_mean":  float(np.mean(y_pred)),
        "y_pred_range": [float(np.min(y_pred)), float(np.max(y_pred))],
    }
    _print_row(m_d)

    # ----------------- Summary -----------------
    print("\n" + "=" * 80)
    print("SUMMARY — R² comparison across configurations")
    print("=" * 80)
    for m in [m_a, m_b, m_c, m_d]:
        bar = "█" * max(0, int(m["r2"] * 40))
        flag = ""
        if m["r2"] < 0:
            flag = "  ← NEGATIVE: model worse than predicting mean"
        elif m["r2"] < 0.5:
            flag = "  ← weak"
        elif m["r2"] >= 0.80:
            flag = "  ← passes gate"
        print(f"  {m['config']:<40} R² = {m['r2']:+.3f}  {bar}{flag}")

    # Save metrics
    metrics_path = DOCS_DIR / "public_data_spike_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"configs": [m_a, m_b, m_c, m_d],
                   "target": TARGET,
                   "features": COMMON_FEATURES,
                   "random_seed": RANDOM_SEED}, f, indent=2)
    print(f"\n[save] Metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
