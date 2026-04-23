#!/usr/bin/env python
"""
Smoke test — проверяет, что все компоненты работают end-to-end.
Использует уменьшенные параметры для скорости.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("smoke")


def main():
    import pandas as pd
    from app.backend.data_curator import save_sample_dataset, clean_dataset
    from app.backend.feature_eng import compute_hsla_features, PIPE_HSLA_FEATURE_SET
    from app.backend.model_trainer import train_model
    from app.backend.inverse_designer import run_inverse_design
    from app.backend.validator import validate_batch
    from app.backend.reporter import render_html_report, save_report
    from pattern_library.patterns import run_all_patterns, Phase
    
    log.info("=" * 60)
    log.info("STEP 1/6: Data")
    log.info("=" * 60)
    data_path = PROJECT_ROOT / "data" / "hsla_synthetic.parquet"
    if not data_path.exists():
        save_sample_dataset()
    df_raw = pd.read_parquet(data_path)
    df_clean, cleaning_report = clean_dataset(df_raw)
    log.info("Cleaned %d -> %d rows", cleaning_report.input_rows, cleaning_report.output_rows)
    
    log.info("=" * 60)
    log.info("STEP 2/6: Features")
    log.info("=" * 60)
    df_feat = compute_hsla_features(df_clean)
    feat_list = [f for f in PIPE_HSLA_FEATURE_SET if f in df_feat.columns]
    log.info("Feature set: %d columns", len(feat_list))
    
    log.info("=" * 60)
    log.info("STEP 3/6: Model training (reduced Optuna for smoke test)")
    log.info("=" * 60)
    trained = train_model(df_feat, "yield_strength_mpa", feat_list, n_optuna_trials=10)
    log.info("Model: %s, R² test=%.3f", trained.version, trained.metrics.r2_test)
    
    log.info("=" * 60)
    log.info("STEP 4/6: Critic on training")
    log.info("=" * 60)
    critic_ctx = {
        "r2_train": trained.metrics.r2_train,
        "r2_val": trained.metrics.r2_val,
        "coverage_90_ci": trained.metrics.coverage_90_ci,
        "prediction_has_ci": True,
        "ood_detector_configured": True,
        "feature_importance": trained.feature_importance,
        "steel_class": "pipe_hsla",
        "has_time_column": True,
        "has_groups": True,
        "split_strategy": "time_based",
        "cv_strategy": "group_kfold",
    }
    warnings = run_all_patterns(critic_ctx, phase=Phase.TRAINING)
    log.info("Critic warnings: %d", len(warnings))
    for w in warnings:
        log.info("  [%s] %s: %s", w["severity"], w["pattern_id"], w["message"][:100])
    
    log.info("=" * 60)
    log.info("STEP 5/6: Inverse design (reduced)")
    log.info("=" * 60)
    design_result = run_inverse_design(
        model_version=trained.version,
        targets={"yield_strength_mpa": {"min": 485, "max": 580}},
        hard_constraints={"cev_iiw": {"max": 0.43}, "pcm": {"max": 0.22}},
        population_size=40, n_generations=30,
    )
    log.info("Pareto candidates: %d", design_result["n_candidates"])
    
    log.info("=" * 60)
    log.info("STEP 5b/6: Inverse design with cost-optimization (seed EUR prices)")
    log.info("=" * 60)
    from app.backend.cost_model import seed_snapshot
    snapshot = seed_snapshot()
    design_with_cost = run_inverse_design(
        model_version=trained.version,
        targets={"yield_strength_mpa": {"min": 485, "max": 580}},
        hard_constraints={"cev_iiw": {"max": 0.43}, "pcm": {"max": 0.22}},
        population_size=30, n_generations=20,
        price_snapshot=snapshot, cost_mode="full",
    )
    log.info("Candidates with cost: %d", design_with_cost["n_candidates"])
    if design_with_cost["pareto_candidates"]:
        c0 = design_with_cost["pareto_candidates"][0]
        log.info("  Top candidate cost: %.2f %s/т (%s mode)",
                 c0["cost"]["total_per_ton"],
                 c0["cost"]["currency"],
                 c0["cost"]["mode"])
        # Seed prices are in EUR; typical HSLA lands in 400-800 €/т range
        assert 200 <= c0["cost"]["total_per_ton"] <= 2000
    assert design_with_cost["price_snapshot_path"]

    log.info("=" * 60)
    log.info("STEP 6/6: Validation + Report")
    log.info("=" * 60)
    val_result = validate_batch(design_result["pareto_candidates"])
    log.info("Approved: %d, rejected: %d", len(val_result["approved"]), len(val_result["rejected"]))
    
    html = render_html_report(
        candidates=val_result["approved"] or design_result["pareto_candidates"][:5],
        model_info={
            "version": trained.version,
            "target": "yield_strength_mpa",
            "r2_train": trained.metrics.r2_train,
            "r2_val": trained.metrics.r2_val,
            "r2_test": trained.metrics.r2_test,
            "mae_test": trained.metrics.mae_test,
            "coverage_90_ci": trained.metrics.coverage_90_ci,
            "split_strategy": "time_based",
            "cv_strategy": "group_kfold",
        },
        user_request={
            "targets": {"yield_strength_mpa": {"min": 485, "max": 580}},
            "constraints": {"cev_iiw": {"max": 0.43}, "pcm": {"max": 0.22}},
        },
        critic_reports=[{"phase": "training", "verdict": "PASS_WITH_WARNINGS" if warnings else "PASS",
                        "warnings": warnings}],
    )
    report_path = save_report(html)
    log.info("Report: %s", report_path)
    
    log.info("=" * 60)
    log.info("✓ SMOKE TEST PASSED")
    log.info("=" * 60)
    log.info("Top 3 candidates:")
    top = (val_result["approved"] or design_result["pareto_candidates"])[:3]
    for i, c in enumerate(top, 1):
        comp = c.get("composition", {})
        log.info("  #%d: C=%.3f, Mn=%.2f, Nb=%.4f → σт=%.0f±%.0f МПа, CEV=%.3f",
                i, comp.get("c_pct", 0), comp.get("mn_pct", 0), comp.get("nb_pct", 0),
                c["predicted"]["mean"], c["predicted"]["ci_half_width"],
                c["derived"]["cev_iiw"])


if __name__ == "__main__":
    main()
