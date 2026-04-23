"""
Feature Engineering Agent.

Вычисляет физически-осмысленные фичи для HSLA сталей:
- CEV (IIW) — углеродный эквивалент для свариваемости
- Pcm — для низкоуглеродистых
- CEN (Yurioka) — для современных HSLA
- Соотношения и производные
"""
from __future__ import annotations

import math
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def nz(s):
    """Безопасная замена NaN на 0. Работает и с Series, и со скалярами."""
    if isinstance(s, pd.Series):
        return s.fillna(0)
    if s is None:
        return 0
    return s


def cev_iiw(df: pd.DataFrame) -> pd.Series:
    """CEV = C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15"""
    return (
        nz(df.get("c_pct", pd.Series(0, index=df.index)))
        + nz(df.get("mn_pct", pd.Series(0, index=df.index))) / 6
        + (nz(df.get("cr_pct", 0)) + nz(df.get("mo_pct", 0)) + nz(df.get("v_pct", 0))) / 5
        + (nz(df.get("ni_pct", 0)) + nz(df.get("cu_pct", 0))) / 15
    )


def pcm(df: pd.DataFrame) -> pd.Series:
    """Pcm = C + Si/30 + (Mn+Cu+Cr)/20 + Ni/60 + Mo/15 + V/10 + 5*B"""
    return (
        nz(df.get("c_pct", 0))
        + nz(df.get("si_pct", 0)) / 30
        + (nz(df.get("mn_pct", 0)) + nz(df.get("cu_pct", 0)) + nz(df.get("cr_pct", 0))) / 20
        + nz(df.get("ni_pct", 0)) / 60
        + nz(df.get("mo_pct", 0)) / 15
        + nz(df.get("v_pct", 0)) / 10
        + 5 * nz(df.get("b_pct", 0))
    )


def cen_yurioka(df: pd.DataFrame) -> pd.Series:
    """CEN = C + A(C) * [Si/24 + Mn/6 + ...]"""
    c = nz(df.get("c_pct", 0))
    a_c = 0.75 + 0.25 * np.tanh(20 * (c - 0.12))
    return c + a_c * (
        nz(df.get("si_pct", 0)) / 24
        + nz(df.get("mn_pct", 0)) / 6
        + nz(df.get("cu_pct", 0)) / 15
        + nz(df.get("ni_pct", 0)) / 20
        + (nz(df.get("cr_pct", 0)) + nz(df.get("mo_pct", 0)) 
           + nz(df.get("nb_pct", 0)) + nz(df.get("v_pct", 0))) / 5
        + 5 * nz(df.get("b_pct", 0))
    )


def compute_hsla_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет physically-informed features для HSLA.
    Возвращает новый DataFrame (не модифицирует исходный).
    """
    df = df.copy()
    
    # Carbon equivalents
    df["cev_iiw"] = cev_iiw(df)
    df["pcm"] = pcm(df)
    df["cen"] = cen_yurioka(df)
    
    # Ratios
    c = nz(df.get("c_pct", 0))
    mn = nz(df.get("mn_pct", 0))
    s = nz(df.get("s_pct", 0))
    
    df["mn_over_c"] = np.where(c > 0, mn / c.replace(0, np.nan), 0).astype(float)
    df["s_over_mn"] = np.where(mn > 0, s / mn.replace(0, np.nan), 0).astype(float)
    df["mn_over_c"] = df["mn_over_c"].fillna(0).clip(0, 100)
    df["s_over_mn"] = df["s_over_mn"].fillna(0).clip(0, 1)
    
    # Microalloying sum
    df["microalloying_sum"] = (
        nz(df.get("nb_pct", 0)) + nz(df.get("ti_pct", 0)) + nz(df.get("v_pct", 0))
    )
    
    # Ti/N balance (атомное соотношение для связывания N)
    ti = nz(df.get("ti_pct", 0))
    n_pct = nz(df.get("n_ppm", 0)) / 10000
    df["ti_over_n_atomic"] = np.where(
        n_pct > 0, (ti / 47.87) / (n_pct / 14.01).replace(0, np.nan), 0
    )
    df["ti_over_n_atomic"] = df["ti_over_n_atomic"].fillna(0).clip(0, 20)
    
    # Process parameter derivations
    if "rolling_finish_temp" in df.columns:
        # Ниже тем более мелкое зерно
        df["below_tnr_delta"] = 900 - df["rolling_finish_temp"].clip(700, 900)
    
    return df


PIPE_HSLA_FEATURE_SET = [
    # Base composition
    "c_pct", "si_pct", "mn_pct", "p_pct", "s_pct",
    "cr_pct", "ni_pct", "mo_pct", "cu_pct", "al_pct",
    "v_pct", "nb_pct", "ti_pct", "n_ppm",
    # Process
    "rolling_finish_temp", "cooling_rate_c_per_s",
    # Derived
    "cev_iiw", "pcm", "cen",
    "mn_over_c", "s_over_mn", "microalloying_sum", "ti_over_n_atomic",
    "below_tnr_delta",
]


# =========================================================================
# Agent interface
# =========================================================================

class FeatureEngAgent:
    name = "feature_eng"
    
    def run(self, state, task: dict):
        from app.backend.engine import AgentResult
        from decision_log.logger import log_decision
        
        try:
            clean_path = Path(task.get("dataset_path") or state.dataset.get("clean_path"))
            if not clean_path.exists():
                return AgentResult(
                    agent_name=self.name, success=False,
                    output={}, error=f"Dataset not found: {clean_path}",
                )
            
            df = pd.read_parquet(clean_path)
            df_feat = compute_hsla_features(df)
            
            features_path = clean_path.parent / "hsla_features.parquet"
            df_feat.to_parquet(features_path, index=False)
            
            # Посчитаем training ranges для inverse design bounds
            training_ranges = {}
            for col in PIPE_HSLA_FEATURE_SET:
                if col in df_feat.columns:
                    training_ranges[col] = [
                        float(df_feat[col].min()), 
                        float(df_feat[col].max())
                    ]
            
            log_decision(
                phase="feature_engineering",
                decision=f"Feature set 'pipe_hsla_v1' ({len(PIPE_HSLA_FEATURE_SET)} features)",
                reasoning=(
                    "Для HSLA использован feature set из 24 признаков: "
                    "14 базовых элементов + 2 процессных параметра + 8 derived. "
                    "Derived включают 3 углеродных эквивалента (CEV, Pcm, CEN) "
                    "и 5 ratios/interactions. Выбор обоснован публикациями по "
                    "сворачиваемости и прокаливаемости HSLA."
                ),
                alternatives_considered=[
                    "Только composition (без derived)",
                    "Расширенный set с 40 features",
                ],
                context={
                    "feature_set": "pipe_hsla_v1",
                    "n_features": len(PIPE_HSLA_FEATURE_SET),
                    "n_samples": len(df_feat),
                },
                author="feature_eng",
                tags=["features", "hsla"],
            )
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                output={
                    "features_path": str(features_path),
                    "feature_set_name": "pipe_hsla_v1",
                    "feature_list": PIPE_HSLA_FEATURE_SET,
                    "n_features": len(PIPE_HSLA_FEATURE_SET),
                    "training_ranges": training_ranges,
                },
            )
        except Exception as e:
            logger.exception("FeatureEng failed")
            return AgentResult(
                agent_name=self.name, success=False, output={}, error=str(e),
            )


if __name__ == "__main__":
    from pathlib import Path
    df = pd.read_parquet(Path(__file__).resolve().parent.parent.parent / "data" / "hsla_synthetic.parquet")
    df_feat = compute_hsla_features(df)
    print(f"Исходных колонок: {df.shape[1]}, после FE: {df_feat.shape[1]}")
    print("\nНовые derived features:")
    for col in ["cev_iiw", "pcm", "cen", "mn_over_c", "s_over_mn", "microalloying_sum"]:
        print(f"  {col}: mean={df_feat[col].mean():.4f}, std={df_feat[col].std():.4f}")
