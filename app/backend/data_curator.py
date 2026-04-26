"""
Data Curator Agent — загрузка и очистка NIMS HSLA данных.

Поскольку оригинальный NIMS MatNavi требует авторизации, для MVP используем
открытую подкомпозицию через Citrine Public Data или Kaggle Materials datasets.

В этом модуле:
1. download_sample_hsla() — загружает синтетический HSLA-like dataset для демо
2. clean_and_validate() — применяет Pattern Library checks
3. Основной agent.run() для интеграции с engine.py
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def generate_synthetic_hsla_dataset(n_samples: int = 2500, random_seed: int = 42) -> pd.DataFrame:
    """
    Генерирует синтетический HSLA-датасет для MVP-демо.
    
    В production заменить на реальную загрузку NIMS/Citrine. Для демо — 
    физически осмысленные данные с известными закономерностями, на которых
    модель должна обучиться.
    
    Закономерности (упрощённо, но физически правдоподобно):
    - σт растёт с C, Mn, Nb, Ti (grain refinement + precipitation)
    - σт уменьшается с S, P (включения, сегрегация)
    - KCV-60 растёт с Ni, убывает с C (выше C -> меньше вязкость)
    - σв коррелирует с σт + вклад от Nb
    - δ (elongation) обратно пропорционально prочности
    """
    rng = np.random.default_rng(random_seed)
    
    # Химические элементы — в реалистичных HSLA-диапазонах
    c = rng.uniform(0.04, 0.12, n_samples)
    si = rng.uniform(0.15, 0.55, n_samples)
    mn = rng.uniform(0.9, 1.75, n_samples)
    p = rng.uniform(0.005, 0.025, n_samples)
    s = rng.uniform(0.002, 0.012, n_samples)
    cr = rng.uniform(0.0, 0.30, n_samples)
    ni = rng.uniform(0.0, 0.40, n_samples)
    mo = rng.uniform(0.0, 0.10, n_samples)
    cu = rng.uniform(0.05, 0.35, n_samples)
    al = rng.uniform(0.020, 0.050, n_samples)
    v = rng.uniform(0.0, 0.10, n_samples)
    nb = rng.uniform(0.0, 0.06, n_samples)
    ti = rng.uniform(0.0, 0.025, n_samples)
    n_ppm = rng.uniform(30, 80, n_samples)
    
    # Процессные параметры
    rolling_finish_temp = rng.uniform(750, 850, n_samples)
    cooling_rate = rng.uniform(8, 28, n_samples)
    
    # Время — чтобы иметь time-based split (последние ~20% test)
    days_ago = rng.uniform(0, 1800, n_samples)  # 5 лет
    heat_date = pd.to_datetime("2026-04-01") - pd.to_timedelta(days_ago, unit="D")
    
    # Campaign ID для GroupKFold
    campaign_id = rng.integers(0, 150, n_samples)
    
    # =========================================================================
    # Синтетическая физика — yield strength (HSLA реалистичные диапазоны 380-680 МПа)
    # =========================================================================
    sigma_t = (
        320
        + 800 * c
        + 50 * mn
        + 900 * (nb + ti + v)  # micro-alloying strengthening
        + 2 * (cooling_rate - 8)  # cooling rate effect, умеренный
        + 0.3 * (900 - rolling_finish_temp)  # grain refinement от Tf
        + 30 * cr
        + 20 * ni
        - 400 * s
        - 150 * p
        + rng.normal(0, 14, n_samples)  # noise
    )
    
    # Tensile strength — коррелирует с σт + вклад от микролегирования
    sigma_b = sigma_t * 1.18 + 50 * mn + 1500 * nb + rng.normal(0, 20, n_samples)
    
    # Elongation — обратно прочности
    elongation = 38 - 0.025 * (sigma_t - 400) + 8 * al + rng.normal(0, 1.5, n_samples)
    elongation = np.clip(elongation, 12, 35)
    
    # KCV при -60°C
    kcv_neg60 = (
        80
        - 300 * c
        + 50 * ni
        + 20 * al
        - 0.02 * (sigma_t - 400)
        - 1000 * s
        + rng.normal(0, 8, n_samples)
    )
    kcv_neg60 = np.clip(kcv_neg60, 15, 150)
    
    # Округление к реалистичному precision
    df = pd.DataFrame({
        "heat_id": [f"H-{i:06d}" for i in range(n_samples)],
        "heat_date": heat_date,
        "campaign_id": [f"C-{cid:03d}" for cid in campaign_id],
        "c_pct": np.round(c, 4),
        "si_pct": np.round(si, 3),
        "mn_pct": np.round(mn, 3),
        "p_pct": np.round(p, 4),
        "s_pct": np.round(s, 4),
        "cr_pct": np.round(cr, 3),
        "ni_pct": np.round(ni, 3),
        "mo_pct": np.round(mo, 3),
        "cu_pct": np.round(cu, 3),
        "al_pct": np.round(al, 4),
        "v_pct": np.round(v, 4),
        "nb_pct": np.round(nb, 4),
        "ti_pct": np.round(ti, 4),
        "n_ppm": np.round(n_ppm, 1),
        "rolling_finish_temp": np.round(rolling_finish_temp, 1),
        "cooling_rate_c_per_s": np.round(cooling_rate, 2),
        "yield_strength_mpa": np.round(sigma_t, 1),
        "tensile_strength_mpa": np.round(sigma_b, 1),
        "elongation_pct": np.round(elongation, 2),
        "kcv_neg60_j_cm2": np.round(kcv_neg60, 2),
    })
    return df.sort_values("heat_date").reset_index(drop=True)


def save_sample_dataset(output_path: Path | None = None, n: int = 2500) -> Path:
    output_path = output_path or (DATA_DIR / "hsla_synthetic.parquet")
    df = generate_synthetic_hsla_dataset(n_samples=n)
    df.to_parquet(output_path, index=False)
    logger.info("Saved %d samples to %s", len(df), output_path)
    return output_path


def generate_synthetic_en10083_qt_dataset(
    n_samples: int = 2000, random_seed: int = 42
) -> pd.DataFrame:
    """
    EN 10083-2 Q&T carbon steels (C22/C35/C45/C60).

    Empirical physical model:
    - HRC_quenched = 20 + 85*C + 3*ln(Mn+0.5) − 0.05*thickness_mm
    - HRC_tempered = HRC_quenched − 0.4*((temper_T−150)/10)*ln(1 + temper_t/30)
    - tensile_strength ≈ 34.5*HRC*10 МПа (empirical Rockwell→UTS relation)
    """
    rng = np.random.default_rng(random_seed)
    c  = rng.uniform(0.18, 0.65, n_samples)
    si = rng.uniform(0.15, 0.40, n_samples)
    mn = rng.uniform(0.40, 0.80, n_samples)
    p  = rng.uniform(0.0, 0.035, n_samples)
    s  = rng.uniform(0.0, 0.035, n_samples)
    cr = rng.uniform(0.0, 0.40, n_samples)
    austenit_T = rng.uniform(820.0, 900.0, n_samples)
    temper_T   = rng.uniform(150.0, 650.0, n_samples)
    temper_t   = rng.uniform(30.0, 180.0, n_samples)
    thick_mm   = rng.uniform(10.0, 100.0, n_samples)

    hrc_q = 20 + 85 * c + 3 * np.log(mn + 0.5) - 0.05 * thick_mm
    hrc_q = np.clip(hrc_q, 20, 65)
    temper_loss = ((temper_T - 150) / 10) * np.log1p(temper_t / 30)
    hrc = hrc_q - 0.4 * temper_loss
    hrc = np.clip(hrc + rng.normal(0, 1.5, n_samples), 15, 65)
    tensile = 34.5 * hrc * 10 + rng.normal(0, 50, n_samples)
    tensile = np.clip(tensile, 400, 1600)

    campaign_id = rng.integers(1, 50, n_samples)
    heat_date = pd.date_range("2024-01-01", periods=n_samples, freq="2h")

    return pd.DataFrame({
        "c_pct": c, "si_pct": si, "mn_pct": mn,
        "p_pct": p, "s_pct": s, "cr_pct": cr,
        "austenitizing_temp": austenit_T,
        "tempering_temp": temper_T,
        "tempering_time_min": temper_t,
        "section_thickness_mm": thick_mm,
        "hardness_hrc": hrc,
        "tensile_strength_mpa": tensile,
        "campaign_id": campaign_id,
        "heat_date": heat_date,
    })


def save_sample_dataset_en10083_qt(path: Path | None = None) -> Path:
    """Symmetric to save_sample_dataset() but for Q&T class."""
    path = path or (DATA_DIR / "hsla_en10083_qt_synthetic.parquet")
    df = generate_synthetic_en10083_qt_dataset()
    df.to_parquet(path, index=False)
    return path


def load_real_agrawal_fatigue_dataset(
    n_samples: int = 437, random_seed: int = 42
) -> pd.DataFrame:
    """Real 437-record Agrawal 2014 NIMS fatigue dataset.

    The name mirrors the `generate_synthetic_*` pattern so the class registry
    can plug it into `get_synthetic_generator` without a parallel code path,
    but this loader serves REAL peer-reviewed data (NIMS MatNavi, Agrawal
    IMMI 3:8, 2014).

    Shuffles rows before assigning sequential `heat_date` because Agrawal's
    raw file groups carburizing records at the end — without shuffle,
    time_group_split's last-20% test hold-out inherits that class skew.

    `campaign_id` buckets heats into ~44 groups of ~10 records each so
    GroupKFold(n_splits=6) has enough distinct groups.
    """
    parquet_path = DATA_DIR / "agrawal_nims_fatigue.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} missing — run "
            "scripts/fetch_agrawal_nims_fatigue.py first"
        )
    df = pd.read_parquet(parquet_path)

    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    if n_samples < len(df):
        df = df.iloc[:n_samples].reset_index(drop=True)

    df["heat_date"] = pd.date_range("2020-01-01", periods=len(df), freq="1D")
    df["campaign_id"] = [f"C-{i // 10:03d}" for i in range(len(df))]
    return df


# =========================================================================
# Clean and validate
# =========================================================================

@dataclass
class CleaningReport:
    input_rows: int
    output_rows: int
    rejected_rows: int
    rejection_reasons: dict
    unit_conversions: int
    suspicious_flags: int


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Применяет базовые проверки очистки данных.
    Использует Pattern Library физ.границы.
    """
    rejection_reasons: dict[str, int] = {}
    n_in = len(df)
    
    # 1. Базовые hard filters
    bounds = {
        "c_pct": (0.02, 2.1),
        "mn_pct": (0.0, 20.0),
        "si_pct": (0.0, 5.0),
        "p_pct": (0.0, 0.15),
        "s_pct": (0.0, 0.15),
        "yield_strength_mpa": (100, 3000),
        "tensile_strength_mpa": (150, 3500),
        "elongation_pct": (0, 85),
        "kcv_neg60_j_cm2": (0, 400),
    }
    
    rejected_mask = pd.Series(False, index=df.index)
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            out_of_bounds = (df[col] < lo) | (df[col] > hi)
            n_bad = out_of_bounds.sum()
            if n_bad > 0:
                rejection_reasons[f"{col}_out_of_bounds"] = int(n_bad)
                rejected_mask |= out_of_bounds
    
    df_clean = df[~rejected_mask].reset_index(drop=True)
    
    # 2. Дубликаты по heat_id
    n_before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["heat_id"], keep="first")
    dup_removed = n_before_dedup - len(df_clean)
    if dup_removed > 0:
        rejection_reasons["duplicate_heat_id"] = dup_removed
    
    # 3. Statistical outlier flag (не удаляем, помечаем)
    suspicious_flags = 0
    if "yield_strength_mpa" in df_clean.columns:
        z = np.abs((df_clean["yield_strength_mpa"] - df_clean["yield_strength_mpa"].mean()) 
                   / df_clean["yield_strength_mpa"].std())
        df_clean["is_outlier"] = z > 4
        suspicious_flags = int(df_clean["is_outlier"].sum())
    
    report = CleaningReport(
        input_rows=n_in,
        output_rows=len(df_clean),
        rejected_rows=n_in - len(df_clean),
        rejection_reasons=rejection_reasons,
        unit_conversions=0,
        suspicious_flags=suspicious_flags,
    )
    return df_clean, report


# =========================================================================
# Agent interface
# =========================================================================

class DataCuratorAgent:
    name = "data_curator"
    
    def run(self, state, task: dict):
        from app.backend.engine import AgentResult
        from decision_log.logger import log_decision
        
        operation = task.get("operation")
        
        try:
            if operation == "download_nims_hsla":
                path = DATA_DIR / "hsla_synthetic.parquet"
                if not path.exists():
                    path = save_sample_dataset(path)
                df = pd.read_parquet(path)
                
                log_decision(
                    phase="data_acquisition",
                    decision="Использован синтетический HSLA датасет для MVP",
                    reasoning="NIMS MatNavi требует авторизации. Для MVP сгенерирован "
                              "synthetic dataset 2500 плавок с физ.осмысленной корреляцией. "
                              "В пилотной фазе заменить на реальные данные клиента.",
                    alternatives_considered=["NIMS MatNavi (требует auth)", 
                                             "Citrine Public Data",
                                             "Kaggle Steel datasets"],
                    context={"path": str(path), "n_rows": len(df)},
                    author="data_curator",
                    tags=["data_source", "mvp"],
                )
                
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    output={
                        "raw_path": str(path),
                        "n_rows": len(df),
                        "has_time_column": "heat_date" in df.columns,
                        "has_groups": "campaign_id" in df.columns,
                    },
                )
            
            if operation == "clean_and_validate":
                raw_path = Path(state.dataset.get("raw_path", DATA_DIR / "hsla_synthetic.parquet"))
                df = pd.read_parquet(raw_path)
                df_clean, report = clean_dataset(df)
                
                clean_path = DATA_DIR / "hsla_clean.parquet"
                df_clean.to_parquet(clean_path, index=False)
                
                log_decision(
                    phase="preprocessing",
                    decision=f"Очистка: {report.input_rows} → {report.output_rows} строк",
                    reasoning=f"Применены physical bounds из Pattern Library. "
                              f"Отброшено {report.rejected_rows} записей, причины: "
                              f"{report.rejection_reasons}. Outliers помечены: {report.suspicious_flags}.",
                    context={
                        "input": report.input_rows,
                        "output": report.output_rows,
                        "reasons": report.rejection_reasons,
                    },
                    author="data_curator",
                    tags=["cleaning", "physical_bounds"],
                )
                
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    output={
                        "clean_path": str(clean_path),
                        "n_rows": len(df_clean),
                        "rejected_rows": report.rejected_rows,
                        "rejection_reasons": report.rejection_reasons,
                        "suspicious_flags": report.suspicious_flags,
                        "has_time_column": True,
                        "has_groups": True,
                    },
                )
            
            return AgentResult(
                agent_name=self.name, success=False,
                output={}, error=f"Unknown operation: {operation}",
            )
        except Exception as e:
            logger.exception("DataCurator failed")
            return AgentResult(
                agent_name=self.name, success=False, output={}, error=str(e),
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = save_sample_dataset(n=2500)
    df = pd.read_parquet(path)
    print(f"Generated {len(df)} samples")
    print(df.describe())
    df_clean, report = clean_dataset(df)
    print(f"\nCleaning: {report.input_rows} → {report.output_rows}")
    print(f"Rejected: {report.rejection_reasons}")
