"""
Level 1 · Agrawal NIMS production model.

Trains a full-artifact model (main + q05/q95 + OOD detector + meta) on the
real 437-record Agrawal 2014 NIMS fatigue dataset using the project's
standard ModelTrainerAgent path. Unlike scripts/evaluate_agrawal_fatigue.py
which only prints metrics, this script persists the model to
models/<version>/ so that it becomes selectable in the Streamlit UI
sidebar dropdown.

Output meta.json carries `data_source` + `data_source_doi` from the
fatigue_carbon_steel profile — first model in the project's history that
is explicitly tagged as trained on real peer-reviewed data.

Run:
    PYTHONPATH=. .venv/bin/python scripts/train_agrawal_production.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from app.backend.data_curator import load_real_agrawal_fatigue_dataset
from app.backend.model_trainer import train_model
from app.backend.steel_classes import load_steel_class

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main(n_optuna_trials: int = 40) -> None:
    profile = load_steel_class("fatigue_carbon_steel")
    logger.info("Training %s (standard: %s)", profile.name, profile.standard)
    logger.info("Target: %s", profile.target_properties[0].id)
    logger.info("Features: %d", len(profile.feature_set))
    logger.info("Data source: %s", profile.data_source)

    df = load_real_agrawal_fatigue_dataset()
    logger.info("Loaded %d records (columns: %d)", len(df), df.shape[1])

    target = profile.target_properties[0].id
    feat = [f for f in profile.feature_set if f in df.columns]
    missing = set(profile.feature_set) - set(feat)
    if missing:
        raise RuntimeError(
            f"profile feature_set missing from parquet: {missing}"
        )

    trained = train_model(
        df_features=df,
        target=target,
        feature_list=feat,
        n_optuna_trials=n_optuna_trials,
        steel_class="fatigue_carbon_steel",
    )

    logger.info("✓ Training complete")
    logger.info("  version   : %s", trained.version)
    logger.info("  artifact  : %s", trained.artifact_path)
    logger.info("  R² test   : %.3f", trained.metrics.r2_test)
    logger.info("  MAE test  : %.1f MPa", trained.metrics.mae_test)
    logger.info("  coverage  : %.2f", trained.metrics.coverage_90_ci)
    logger.info("  n train/val/test: %d/%d/%d",
                trained.metrics.n_train,
                trained.metrics.n_val,
                trained.metrics.n_test)

    top5 = sorted(trained.feature_importance.items(), key=lambda kv: -kv[1])[:5]
    logger.info("  top-5 features:")
    for name, val in top5:
        logger.info("    %-40s %.3f", name, val)


if __name__ == "__main__":
    main()
