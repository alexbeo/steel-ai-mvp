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

    KNOWN LIMITATIONS (R-006 audit findings C-1, C-2):
      C-1: Compositional variables (C, Mn, Cr, Mo, V) sampled INDEPENDENTLY,
           no constraint that joint Pcm < 0.22 (HSLA weldability limit).
           Some records will have Pcm > 0.22 — Critic D08 will warn.
           Mitigation: D08 informational; production deploys should use real
           HSLA data from NIMS/mill where Pcm distribution is realistic.
      C-2: σ_t formula is purely linear-additive — no Mn×C synergy, no Nb
           solubility-product threshold, no Bs-line cooling rate non-linearity.
           This limits what symbolic regression (B1 capability) can discover
           because non-linear effects don't exist in the data.
           Mitigation: real Agrawal NIMS dataset (fatigue_carbon_steel class)
           does have non-linear effects; B1 demos work better there.
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
        steel_class = task.get("steel_class", "pipe_hsla")

        try:
            # ── PR 8: non-HSLA classes (deox_calibration, en10083_qt,
            #         fatigue_carbon_steel) — generic путь через registry.
            if operation == "download_nims_hsla" and steel_class != "pipe_hsla":
                from app.backend.steel_classes import (
                    get_synthetic_generator,
                    load_steel_class,
                )

                profile = load_steel_class(steel_class)
                gen = get_synthetic_generator(profile.synthetic_generator_name)
                n_rows = task.get("n_rows") or 500
                # Generator-specific kwarg: HSLA/Q&T → n_samples,
                # deox_calibration / fatigue loader → n_heats / n_samples.
                # Probe signature to pick the right kwarg.
                import inspect
                sig = inspect.signature(gen)
                if "n_heats" in sig.parameters:
                    df = gen(n_heats=n_rows)
                elif "n_samples" in sig.parameters:
                    df = gen(n_samples=n_rows)
                else:
                    df = gen()

                # Для process feedback классов (нет preprocessing) сразу
                # сохраняем как clean_path. Для остальных — как raw_path,
                # последующий preprocessing шаг сделает clean.
                file_stem = (
                    f"{steel_class}_clean" if steel_class == "deox_calibration"
                    else f"{steel_class}_synthetic"
                )
                out_path = DATA_DIR / f"{file_stem}.parquet"
                df.to_parquet(out_path, index=False)

                log_decision(
                    phase="data_acquisition",
                    decision=(
                        f"Сгенерирован синтетический датасет для класса "
                        f"{steel_class!r} ({len(df)} строк)"
                    ),
                    reasoning=(
                        f"Использован generator '{profile.synthetic_generator_name}' "
                        f"из registry steel_classes."
                    ),
                    context={
                        "path": str(out_path),
                        "n_rows": len(df),
                        "steel_class": steel_class,
                    },
                    author="data_curator",
                    tags=["data_source", steel_class],
                )

                # clean_path для process feedback (нет preprocessing фазы);
                # raw_path для всех — backwards-compat.
                output: dict = {
                    "raw_path": str(out_path),
                    "n_rows": len(df),
                    "has_time_column": "heat_date" in df.columns,
                    "has_groups": "campaign_id" in df.columns,
                    "steel_class": steel_class,
                }
                if steel_class == "deox_calibration":
                    output["clean_path"] = str(out_path)

                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    output=output,
                )

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


# ──────────────────────────────────────────────────────────────────────
# Synthetic generator для virtual class deox_calibration (R-004 PR 4)
# ──────────────────────────────────────────────────────────────────────

# Plant-specific смещения η_Al (multi-plant variability). Pre-агрегированы
# в `plant_offset_baseline` numeric feature — Pattern A из design-doc, без
# OneHot/категорий в feature_set профиля.
PLANT_OFFSETS = {"PLANT_A": +0.02, "PLANT_B": 0.00, "PLANT_C": -0.03}


def generate_synthetic_deox_calibration_dataset(
    n_heats: int = 500, random_seed: int = 42
) -> pd.DataFrame:
    """Synthetic LF deox heats для virtual class deox_calibration.

    Physics-based генератор для калибровки η_Al модели в Tier 2 capabilities
    (PR 8 ML, PR 11 symbolic regression, PR 12 anomaly explainer).

    Модель η_Al::

        η = method_baseline
            − 0.0030 × FeO% × (slag_mass / 2000)        # FeO re-oxidation
            − 0.0015 × max(0, T_Al − 1620°C)            # high-T burn-off
            + 0.05  × (vacuum_treatment == "VD")        # VD lowers pO2
            + 0.025 × (co_deox_FeSi > 50 kg)            # Si pre-deox helps
            − 0.0005 × refractory_heat_count            # lining age
            − 0.003  × max(0, dt_to_al_min − 5)         # late addition penalty
            + plant_offset
            + N(0, 0.04)
        η ∈ [0.30, 0.99] (clipped)

    method_eta_baseline / plant_offset_baseline pre-aggregated as **numeric**
    features (Option A) — категориальные method_id/plant_id остаются в
    DataFrame для трассировки и heats.db reference, но НЕ в profile.feature_set.

    Args:
        n_heats: число синтетических плавок. По умолчанию 500.
        random_seed: seed для reproducibility.

    Returns:
        DataFrame с колонками heat_id / heat_date / campaign_id (для M07
        GroupKFold), категориями (plant_id, method_id, addition_timing,
        carrier_gas, vacuum_treatment) для трассировки + всеми numeric
        features из feature_set профиля + target eta_al_effective.
    """
    from app.backend.slag_aware_deox import load_addition_methods

    rng = np.random.default_rng(random_seed)
    methods = load_addition_methods()
    method_ids = list(methods.keys())
    plants = list(PLANT_OFFSETS.keys())

    method_choice = rng.choice(method_ids, size=n_heats)
    plant_choice = rng.choice(plants, size=n_heats)

    # numeric process variables — физически правдоподобные диапазоны LF
    c_pct = rng.uniform(0.04, 0.50, n_heats)
    mn_pct = rng.uniform(0.4, 1.8, n_heats)
    si_pct = rng.uniform(0.15, 0.5, n_heats)
    s_pct = rng.uniform(0.003, 0.030, n_heats)
    p_pct = rng.uniform(0.005, 0.025, n_heats)
    slag_mass_kg = rng.uniform(1000, 3500, n_heats)
    slag_feo_pct = rng.uniform(2.0, 35.0, n_heats)
    slag_mno_pct = rng.uniform(0.5, 8.0, n_heats)
    slag_sio2_pct = rng.uniform(5.0, 22.0, n_heats)
    slag_cao_pct = rng.uniform(35.0, 60.0, n_heats)
    t_tap_c = rng.uniform(1610, 1680, n_heats)
    t_lf_arrival_c = rng.uniform(1560, 1620, n_heats)
    t_al_addition_c = rng.uniform(1580, 1660, n_heats)
    dt_to_al_min = rng.uniform(2.0, 25.0, n_heats)
    co_deox_fesi_kg = rng.uniform(0.0, 300.0, n_heats)
    ar_stir_nm3 = rng.uniform(0.0, 30.0, n_heats)
    refractory_heat_count = rng.integers(1, 200, n_heats).astype(float)
    steel_mass_ton = rng.uniform(150.0, 400.0, n_heats)
    o_a_initial_ppm = rng.uniform(200.0, 800.0, n_heats)

    # baselines (numeric features for feature_set)
    method_eta_baseline = np.array(
        [methods[m].eta_al_typical for m in method_choice]
    )
    plant_offset_baseline = np.array([PLANT_OFFSETS[p] for p in plant_choice])

    # категориальные — только для трассировки, не в feature_set
    vacuum_treatment = rng.choice(["none", "VD"], n_heats, p=[0.85, 0.15])
    carrier_gas = np.where(
        np.isin(method_choice, ["asis_shot"]),
        rng.choice(["Ar", "N2"], n_heats, p=[0.8, 0.2]),
        "none",
    )
    addition_timing = rng.choice(
        ["in_stream", "trim_after_lf_arrival", "split"], n_heats
    )

    # physics-based η_Al
    eta = method_eta_baseline.copy()
    eta -= 0.0030 * slag_feo_pct * (slag_mass_kg / 2000.0)
    eta -= 0.0015 * np.maximum(0.0, t_al_addition_c - 1620.0)
    eta += 0.05 * (vacuum_treatment == "VD")
    eta += 0.025 * (co_deox_fesi_kg > 50.0)
    eta -= 0.0005 * refractory_heat_count
    eta -= 0.003 * np.maximum(0.0, dt_to_al_min - 5.0)
    eta += plant_offset_baseline
    eta += rng.normal(0, 0.04, n_heats)
    eta = np.clip(eta, 0.30, 0.99)

    # campaign_id: 40 heats group (refractory campaign) для GroupKFold M07
    campaign_id = [f"C-{i // 40:03d}" for i in range(n_heats)]
    heat_date = pd.date_range("2024-06-01", periods=n_heats, freq="2h")

    return pd.DataFrame({
        "heat_id": [f"DH-{i:06d}" for i in range(n_heats)],
        "heat_date": heat_date,
        "campaign_id": campaign_id,
        "plant_id": plant_choice,
        "method_id": method_choice,
        "addition_timing": addition_timing,
        "carrier_gas": carrier_gas,
        "vacuum_treatment": vacuum_treatment,
        "c_pct": np.round(c_pct, 4),
        "mn_pct": np.round(mn_pct, 4),
        "si_pct": np.round(si_pct, 4),
        "s_pct": np.round(s_pct, 4),
        "p_pct": np.round(p_pct, 4),
        "slag_mass_kg": np.round(slag_mass_kg, 1),
        "slag_feo_pct": np.round(slag_feo_pct, 2),
        "slag_mno_pct": np.round(slag_mno_pct, 2),
        "slag_sio2_pct": np.round(slag_sio2_pct, 2),
        "slag_cao_pct": np.round(slag_cao_pct, 2),
        "t_tap_c": np.round(t_tap_c, 1),
        "t_lf_arrival_c": np.round(t_lf_arrival_c, 1),
        "t_al_addition_c": np.round(t_al_addition_c, 1),
        "dt_to_al_min": np.round(dt_to_al_min, 2),
        "co_deox_fesi_kg": np.round(co_deox_fesi_kg, 1),
        "ar_stir_nm3": np.round(ar_stir_nm3, 2),
        "refractory_heat_count": refractory_heat_count.astype(int),
        "steel_mass_ton": np.round(steel_mass_ton, 1),
        "o_a_initial_ppm": np.round(o_a_initial_ppm, 1),
        "method_eta_baseline": np.round(method_eta_baseline, 4),
        "plant_offset_baseline": np.round(plant_offset_baseline, 4),
        "eta_al_effective": np.round(eta, 4),
    })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = save_sample_dataset(n=2500)
    df = pd.read_parquet(path)
    print(f"Generated {len(df)} samples")
    print(df.describe())
    df_clean, report = clean_dataset(df)
    print(f"\nCleaning: {report.input_rows} → {report.output_rows}")
    print(f"Rejected: {report.rejection_reasons}")
