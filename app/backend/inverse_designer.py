"""
Inverse Designer — NSGA-II multi-objective optimization.

Получает targets и constraints, возвращает Pareto-оптимальные кандидаты.
Работает поверх TrainedModel от model_trainer.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from app.backend.feature_eng import compute_hsla_features
from app.backend.model_trainer import load_model, predict_with_uncertainty
from app.backend.cost_model import (
    PriceSnapshot, CostMode, compute_cost, save_snapshot,
    validate_snapshot, required_elements_for_design, PriceSnapshotIncomplete,
)

logger = logging.getLogger(__name__)


# Variable bounds и element prices (€/kg при ~100₽/€)
VARIABLE_BOUNDS_HSLA = {
    "c_pct": [0.04, 0.12],
    "si_pct": [0.15, 0.55],
    "mn_pct": [0.9, 1.75],
    "p_pct": [0.005, 0.025],
    "s_pct": [0.002, 0.012],
    "cr_pct": [0.0, 0.30],
    "ni_pct": [0.0, 0.40],
    "mo_pct": [0.0, 0.10],
    "cu_pct": [0.05, 0.35],
    "al_pct": [0.020, 0.050],
    "v_pct": [0.0, 0.10],
    "nb_pct": [0.0, 0.06],
    "ti_pct": [0.0, 0.025],
    "n_ppm": [30.0, 80.0],
    "rolling_finish_temp": [750.0, 850.0],
    "cooling_rate_c_per_s": [8.0, 28.0],
}

ELEMENT_PRICES_EUR_PER_KG = {
    "c": 0.5, "si": 1.5, "mn": 2.5, "cr": 3.2, "ni": 22.0,
    "mo": 38.0, "cu": 7.8, "al": 1.8, "v": 32.0, "nb": 68.0,
    "ti": 18.0,
}


class HSLADesignProblem(ElementwiseProblem):
    
    def __init__(
        self,
        model_bundle: dict,
        targets: dict[str, dict],
        hard_constraints: dict[str, dict],
        variable_bounds: dict[str, list[float]],
        element_prices: dict[str, float],
        price_snapshot: PriceSnapshot | None = None,
        cost_mode: CostMode = "full",
    ):
        self.model_bundle = model_bundle
        self.targets = targets
        self.hard_constraints = hard_constraints
        self.variable_bounds = variable_bounds
        self.element_prices = element_prices
        self.price_snapshot = price_snapshot
        self.cost_mode = cost_mode

        self.var_names = list(variable_bounds.keys())
        xl = np.array([variable_bounds[n][0] for n in self.var_names])
        xu = np.array([variable_bounds[n][1] for n in self.var_names])

        # 3 objectives: distance_to_target, alloying_cost, prediction_uncertainty
        super().__init__(
            n_var=len(self.var_names),
            n_obj=3,
            n_ieq_constr=len(hard_constraints),
            xl=xl, xu=xu,
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        row = dict(zip(self.var_names, x))
        df_input = pd.DataFrame([row])
        df_with_features = compute_hsla_features(df_input)
        
        # Прогноз моделью
        pred = predict_with_uncertainty(self.model_bundle, df_with_features)
        pred_val = float(pred["prediction"].iloc[0])
        ci_width = float(pred["ci_half_width"].iloc[0])
        
        # Objective 1: distance to target
        f1 = 0.0
        for prop, spec in self.targets.items():
            # Для MVP один target — тот, что предсказывает модель
            target_val = pred_val if prop == self.model_bundle["meta"]["target"] else pred_val
            if "min" in spec and target_val < spec["min"]:
                f1 += ((spec["min"] - target_val) / max(abs(spec["min"]), 1)) ** 2
            if "max" in spec and target_val > spec["max"]:
                f1 += ((target_val - spec["max"]) / max(abs(spec["max"]), 1)) ** 2
        
        # Objective 2: alloying cost
        if self.price_snapshot is not None:
            composition_pct = {k: v for k, v in row.items() if k.endswith("_pct")}
            breakdown = compute_cost(
                composition_pct, self.price_snapshot, mode=self.cost_mode
            )
            f2 = breakdown.total_per_ton
        else:
            # Legacy fallback: simple element-weighted cost (deprecated).
            cost = 0.0
            for elem, price in self.element_prices.items():
                key = f"{elem}_pct"
                if key in row:
                    cost += row[key] * 10 * price  # %→кг/т
            f2 = cost
        
        # Objective 3: uncertainty
        f3 = ci_width
        
        out["F"] = [f1, f2, f3]
        
        # Constraints: g ≤ 0 = feasible
        g = []
        features = df_with_features.iloc[0]
        for cname, spec in self.hard_constraints.items():
            if cname in features.index:
                val = float(features[cname])
            elif cname in row:
                val = row[cname]
            else:
                val = 0
            if "max" in spec:
                g.append(val - spec["max"])
            elif "min" in spec:
                g.append(spec["min"] - val)
        out["G"] = g


def run_inverse_design(
    model_version: str,
    targets: dict,
    hard_constraints: dict | None = None,
    variable_bounds: dict | None = None,
    population_size: int = 80,
    n_generations: int = 60,
    random_seed: int = 42,
    price_snapshot: PriceSnapshot | None = None,
    cost_mode: CostMode = "full",
) -> dict:
    """
    Запускает NSGA-II.
    """
    bounds = variable_bounds or VARIABLE_BOUNDS_HSLA
    constraints = hard_constraints or {}

    # Pre-check prices (before loading model — fail fast)
    if price_snapshot is not None:
        required = required_elements_for_design(bounds)
        missing = validate_snapshot(price_snapshot, required)
        if missing:
            raise PriceSnapshotIncomplete(missing)
        if cost_mode == "full" and "scrap" not in price_snapshot.materials:
            raise PriceSnapshotIncomplete(["scrap (base material)"])

    model_bundle = load_model(model_version)

    problem = HSLADesignProblem(
        model_bundle=model_bundle,
        targets=targets,
        hard_constraints=constraints,
        variable_bounds=bounds,
        element_prices=ELEMENT_PRICES_EUR_PER_KG,
        price_snapshot=price_snapshot,
        cost_mode=cost_mode,
    )

    algorithm = NSGA2(
        pop_size=population_size,
        sampling=LHS(),
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem, algorithm,
        termination=get_termination("n_gen", n_generations),
        seed=random_seed, verbose=False,
    )

    # Save snapshot to decision_log for audit trail
    snapshot_path: str | None = None
    if price_snapshot is not None:
        from decision_log.logger import log_decision
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_dir = Path(__file__).resolve().parents[2] / "decision_log" / "price_snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snap_dir / f"{run_ts}.yaml"
        save_snapshot(price_snapshot, snap_path)
        snapshot_path = str(snap_path)
        log_decision(
            phase="inverse_design",
            decision=f"Inverse design с cost-optimization "
                     f"(snapshot {price_snapshot.date}, {price_snapshot.currency}, mode={cost_mode})",
            reasoning=f"Source: {price_snapshot.source}. "
                      f"{len(price_snapshot.materials)} материалов. "
                      f"Legacy ELEMENT_PRICES_EUR_PER_KG отключён для этого run.",
            context={
                "snapshot_path": snapshot_path,
                "currency": price_snapshot.currency,
                "n_materials": len(price_snapshot.materials),
                "cost_mode": cost_mode,
            },
            author="inverse_designer",
            tags=["cost_optimization", str(price_snapshot.date)],
        )

    if res.X is None:
        return {
            "pareto_candidates": [], "n_candidates": 0,
            "objectives_normalized": False, "n_objectives": 3,
            "variable_bounds": bounds,
            "cost_currency": price_snapshot.currency if price_snapshot else "EUR (legacy)",
            "cost_mode": cost_mode if price_snapshot else "legacy",
            "price_snapshot_path": snapshot_path,
        }

    # Build candidates
    candidates = []
    var_names = list(bounds.keys())
    for i, x in enumerate(res.X):
        row = dict(zip(var_names, x.tolist()))
        df = pd.DataFrame([row])
        df_feat = compute_hsla_features(df)
        pred = predict_with_uncertainty(model_bundle, df_feat)

        breakdown_dict = None
        if price_snapshot is not None:
            composition_pct = {k: v for k, v in row.items() if k.endswith("_pct")}
            breakdown_dict = asdict(
                compute_cost(composition_pct, price_snapshot, cost_mode)
            )

        candidates.append({
            "idx": i,
            "composition": {k: round(v, 4) for k, v in row.items() if k.endswith("_pct")},
            "processing": {k: round(v, 2) for k, v in row.items() if not k.endswith("_pct")},
            "predicted": {
                "mean": float(pred["prediction"].iloc[0]),
                "lower_90": float(pred["lower_90"].iloc[0]),
                "upper_90": float(pred["upper_90"].iloc[0]),
                "ci_half_width": float(pred["ci_half_width"].iloc[0]),
                "ood_flag": bool(pred["ood_flag"].iloc[0]),
            },
            "derived": {
                "cev_iiw": round(float(df_feat["cev_iiw"].iloc[0]), 4),
                "pcm": round(float(df_feat["pcm"].iloc[0]), 4),
                "cen": round(float(df_feat["cen"].iloc[0]), 4),
                "microalloying_sum": round(float(df_feat["microalloying_sum"].iloc[0]), 4),
            },
            "objectives": {
                "distance_to_target": float(res.F[i, 0]),
                "alloying_cost": float(res.F[i, 1]),
                "prediction_uncertainty": float(res.F[i, 2]),
            },
            "cost": breakdown_dict,
        })

    candidates.sort(key=lambda c: (
        c["objectives"]["distance_to_target"],
        c["objectives"]["alloying_cost"],
    ))

    return {
        "pareto_candidates": candidates,
        "n_candidates": len(candidates),
        "objectives_normalized": False,
        "n_objectives": 3,
        "variable_bounds": bounds,
        "training_variable_ranges": model_bundle["meta"].get("training_ranges", {}),
        "cost_currency": price_snapshot.currency if price_snapshot else "EUR (legacy)",
        "cost_mode": cost_mode if price_snapshot else "legacy",
        "price_snapshot_path": snapshot_path,
    }


# =========================================================================
# Agent interface
# =========================================================================

class InverseDesignerAgent:
    name = "inverse_designer"
    
    def run(self, state, task):
        from app.backend.engine import AgentResult
        from decision_log.logger import log_decision
        
        try:
            model_version = task.get("model_version") or state.model.get("version")
            if not model_version:
                return AgentResult(
                    agent_name=self.name, success=False,
                    output={}, error="No model_version provided",
                )
            
            targets = task.get("targets") or {"yield_strength_mpa": {"min": 450, "max": 600}}
            constraints = task.get("hard_constraints") or {"cev_iiw": {"max": 0.43}}
            
            result = run_inverse_design(
                model_version=model_version,
                targets=targets,
                hard_constraints=constraints,
                population_size=task.get("population_size", 80),
                n_generations=task.get("n_generations", 60),
                price_snapshot=task.get("price_snapshot"),
                cost_mode=task.get("cost_mode", "full"),
            )
            
            log_decision(
                phase="inverse_design",
                decision=f"NSGA-II: {result['n_candidates']} Pareto кандидатов",
                reasoning=(
                    f"Multi-objective оптимизация с 3 objectives: "
                    f"distance_to_target + alloying_cost + uncertainty. "
                    f"Targets: {targets}. Constraints: {constraints}. "
                    f"Population 80, gens 60."
                ),
                context={
                    "targets": targets, "constraints": constraints,
                    "n_candidates": result["n_candidates"],
                    "model_version": model_version,
                },
                author="inverse_designer",
                tags=["nsga2", "inverse_design"],
            )
            
            return AgentResult(
                agent_name=self.name, success=True, output=result,
            )
        except Exception as e:
            logger.exception("InverseDesigner failed")
            return AgentResult(
                agent_name=self.name, success=False, output={}, error=str(e),
            )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    # Берём последнюю обученную модель
    models_dir = Path(__file__).resolve().parent.parent.parent / "models"
    versions = sorted([d.name for d in models_dir.iterdir() if d.is_dir()])
    if not versions:
        print("No trained models found. Run model_trainer.py first.")
        sys.exit(1)
    
    version = versions[-1]
    print(f"Using model: {version}\n")
    
    result = run_inverse_design(
        model_version=version,
        targets={"yield_strength_mpa": {"min": 485, "max": 580}},
        hard_constraints={"cev_iiw": {"max": 0.43}, "pcm": {"max": 0.22}},
        population_size=60, n_generations=40,
    )
    
    print(f"Найдено кандидатов: {result['n_candidates']}")
    print("\nТоп-3 кандидата:")
    for c in result["pareto_candidates"][:3]:
        print(f"\n  #{c['idx']}:")
        print(f"    Состав: C={c['composition']['c_pct']}, Mn={c['composition']['mn_pct']}, "
              f"Nb={c['composition']['nb_pct']}, Ti={c['composition']['ti_pct']}")
        print(f"    Обработка: T_rolling={c['processing']['rolling_finish_temp']}, "
              f"cooling={c['processing']['cooling_rate_c_per_s']}")
        print(f"    σт = {c['predicted']['mean']:.0f} ± {c['predicted']['ci_half_width']:.0f} МПа")
        print(f"    CEV = {c['derived']['cev_iiw']}, Pcm = {c['derived']['pcm']}")
        print(f"    Стоимость = {c['objectives']['alloying_cost']:.1f} €/т")
