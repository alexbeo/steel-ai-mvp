#!/usr/bin/env python
"""
Главный CLI-скрипт — запускает полный pipeline.

Использование:
    python scripts/run_pipeline.py --full              # весь pipeline от A до Z
    python scripts/run_pipeline.py --step data         # только data
    python scripts/run_pipeline.py --step train        # только train
    python scripts/run_pipeline.py --step design \\
         --target-min 485 --target-max 580              # только inverse design
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.engine import Orchestrator, Critic, cli_checkpoint_handler
from app.backend.data_curator import DataCuratorAgent
from app.backend.feature_eng import FeatureEngAgent
from app.backend.model_trainer import ModelTrainerAgent
from app.backend.inverse_designer import InverseDesignerAgent
from app.backend.validator import ValidatorAgent
from app.backend.reporter import ReporterAgent


def make_orchestrator() -> Orchestrator:
    agents = {
        "data_curator": DataCuratorAgent(),
        "feature_eng": FeatureEngAgent(),
        "model_trainer": ModelTrainerAgent(),
        "inverse_designer": InverseDesignerAgent(),
        "validator": ValidatorAgent(),
        "reporter": ReporterAgent(),
    }
    return Orchestrator(
        agents=agents,
        critic=Critic(use_llm=True),
        human_in_the_loop=True,
        on_human_checkpoint=cli_checkpoint_handler,
    )


def run_full(
    target_min: float,
    target_max: float,
    max_cev: float,
    max_pcm: float,
    steel_class: str = "pipe_hsla",
    target_property: str | None = None,
    n_rows: int | None = None,
    n_optuna_trials: int = 40,
):
    orch = make_orchestrator()

    # Defaults per steel_class. Для process feedback классов
    # (deox_calibration) inverse_design фаза пропускается Orchestrator-ом,
    # так что targets/constraints не нужны.
    default_target = {
        "pipe_hsla": "yield_strength_mpa",
        "en10083_qt": "tensile_strength_mpa",
        "fatigue_carbon_steel": "fatigue_strength_mpa",
        "deox_calibration": "eta_al_effective",
    }.get(steel_class, "yield_strength_mpa")
    target = target_property or default_target

    user_request: dict = {
        "task_type": "train" if steel_class == "deox_calibration" else "inverse_design",
        "steel_class": steel_class,
        "target_property": target,
        "n_optuna_trials": n_optuna_trials,
    }
    if n_rows is not None:
        user_request["n_rows"] = n_rows

    # Inverse-design targets/constraints — только для HSLA pipeline.
    if steel_class == "pipe_hsla":
        user_request["targets"] = {
            target: {"min": target_min, "max": target_max},
        }
        user_request["constraints"] = {
            "cev_iiw": {"max": max_cev},
            "pcm": {"max": max_pcm},
        }

    state = orch.run_pipeline(user_request)
    
    print("\n" + "=" * 70)
    print("PIPELINE FINISHED")
    print("=" * 70)

    if state.report_paths.get("report_html_path"):
        print(f"\nОтчёт: {state.report_paths['report_html_path']}")

    # Для process feedback классов inverse_design фаза не запускалась —
    # вместо кандидатов выводим training metrics.
    if state.model:
        print(f"\nМодель: {state.model.get('version', '<unknown>')}")
        r2_test = state.model.get("r2_test")
        if r2_test is not None:
            print(f"  R² test = {r2_test:.3f}")
        mae = state.model.get("mae_test")
        if mae is not None:
            print(f"  MAE test = {mae:.4f}")

    n_approved = len(state.validated_candidates)
    if n_approved:
        print(f"Валидных кандидатов: {n_approved}")
        top = state.validated_candidates[0]
        print(f"\nТоп-кандидат:")
        comp = top["composition"]
        print(f"  Химия: C={comp.get('c_pct', 0):.3f}, Mn={comp.get('mn_pct', 0):.2f}, "
              f"Nb={comp.get('nb_pct', 0):.4f}, Ti={comp.get('ti_pct', 0):.4f}")
        print(f"  Прогноз σт = {top['predicted']['mean']:.0f} ± {top['predicted']['ci_half_width']:.0f} МПа")
        print(f"  CEV = {top['derived']['cev_iiw']:.3f}")

    print(f"\nCritic reports: {len(state.critic_reports)}")
    for r in state.critic_reports:
        v = r.verdict.value if hasattr(r.verdict, "value") else str(r.verdict)
        print(f"  [{v}] {r.phase}: {len(r.warnings)} warnings")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run full pipeline end-to-end")
    parser.add_argument("--step", choices=["data", "features", "train", "design", "report"])
    parser.add_argument(
        "--class", dest="steel_class", default="pipe_hsla",
        choices=["pipe_hsla", "en10083_qt", "fatigue_carbon_steel", "deox_calibration"],
        help="Steel class id (см. data/steel_classes/<id>.yaml).",
    )
    parser.add_argument(
        "--target-property", default=None,
        help="Override default target (например, eta_al_effective для deox_calibration).",
    )
    parser.add_argument(
        "--n-rows", type=int, default=None,
        help="Override число строк synthetic-датасета.",
    )
    parser.add_argument(
        "--n-optuna-trials", type=int, default=40,
        help="Optuna trials для XGBoost HP tuning.",
    )
    parser.add_argument("--target-min", type=float, default=485)
    parser.add_argument("--target-max", type=float, default=580)
    parser.add_argument("--max-cev", type=float, default=0.43)
    parser.add_argument("--max-pcm", type=float, default=0.22)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.full or not args.step:
        run_full(
            args.target_min, args.target_max, args.max_cev, args.max_pcm,
            steel_class=args.steel_class,
            target_property=args.target_property,
            n_rows=args.n_rows,
            n_optuna_trials=args.n_optuna_trials,
        )
    elif args.step == "data":
        from app.backend.data_curator import save_sample_dataset
        path = save_sample_dataset()
        print(f"✓ Dataset: {path}")
    elif args.step == "features":
        from app.backend.feature_eng import compute_hsla_features, PIPE_HSLA_FEATURE_SET
        import pandas as pd
        df = pd.read_parquet(PROJECT_ROOT / "data" / "hsla_synthetic.parquet")
        df_feat = compute_hsla_features(df)
        out = PROJECT_ROOT / "data" / "hsla_features.parquet"
        df_feat.to_parquet(out, index=False)
        print(f"✓ Features: {out} ({df_feat.shape[1]} columns)")
    elif args.step == "train":
        from app.backend.model_trainer import train_model
        from app.backend.feature_eng import compute_hsla_features, PIPE_HSLA_FEATURE_SET
        import pandas as pd
        df = pd.read_parquet(PROJECT_ROOT / "data" / "hsla_synthetic.parquet")
        df_feat = compute_hsla_features(df)
        feat = [f for f in PIPE_HSLA_FEATURE_SET if f in df_feat.columns]
        trained = train_model(df_feat, "yield_strength_mpa", feat, n_optuna_trials=40)
        print(f"✓ Model: {trained.version}")
        print(f"  R² test = {trained.metrics.r2_test:.3f}")


if __name__ == "__main__":
    main()
