"""
Path 2 · A · Step 1 / 2

Loads the Agrawal et al. 2014 NIMS steel-fatigue dataset (437 records,
25 features + target), harmonizes column names to project snake_case
convention, tags provenance, and saves to data/agrawal_nims_fatigue.parquet.

Source: raw xlsx re-distributed via GitHub mirror
    hunterkimmett/Fatigue-Machine-Learning/fatigue_dataset.xlsx
Original paper:
    Agrawal et al. (2014) "Exploration of data science techniques to predict
    fatigue strength of steel from composition and processing parameters."
    Integrating Materials and Manufacturing Innovation 3:8
    https://link.springer.com/article/10.1186/2193-9772-3-8
Raw data: NIMS MatNavi public domain.

Coverage (per the paper): 371 carbon + low-alloy + 48 carburizing +
18 spring steels.

Run:
    PYTHONPATH=. .venv/bin/python scripts/fetch_agrawal_nims_fatigue.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RAW_XLSX = PROJECT_ROOT / "data" / "public_raw" / "agrawal_nims_fatigue_raw.xlsx"
OUTPUT = PROJECT_ROOT / "data" / "agrawal_nims_fatigue.parquet"
STATS_JSON = PROJECT_ROOT / "docs" / "agrawal_dataset_stats.json"

RENAME = {
    "Sl. No.": "heat_id",
    "NT": "normalizing_temp_c",
    "THT": "through_hardening_temp_c",
    "THt": "through_hardening_time_min",
    "THQCr": "through_hardening_cooling_rate_c_per_s",
    "CT": "carburizing_temp_c",
    "Ct": "carburizing_time_min",
    "DT": "diffusion_temp_c",
    "Dt": "diffusion_time_min",
    "QmT": "quenching_media_temp_c",
    "TT": "tempering_temp_c",
    "Tt": "tempering_time_min",
    "TCr": "tempering_cooling_rate_c_per_s",
    "C":  "c_pct",
    "Si": "si_pct",
    "Mn": "mn_pct",
    "P":  "p_pct",
    "S":  "s_pct",
    "Ni": "ni_pct",
    "Cr": "cr_pct",
    "Cu": "cu_pct",
    "Mo": "mo_pct",
    "RedRatio": "reduction_ratio",
    "dA": "inclusion_area_defect_a",
    "dB": "inclusion_area_defect_b",
    "dC": "inclusion_area_defect_c",
    "Fatigue": "fatigue_strength_mpa",
}


def _classify_heat(row: pd.Series) -> str:
    """Split 437 records into Agrawal's 3 sub-populations by processing signature.

    Heuristic from the paper:
      - carburizing steels: non-zero carburizing temp/time
      - spring steels: high C (>=0.50) and non-zero tempering
      - carbon/low-alloy: everything else
    """
    if row["carburizing_temp_c"] > 0 and row["carburizing_time_min"] > 0:
        return "carburizing"
    if row["c_pct"] >= 0.50 and row["tempering_temp_c"] > 0:
        return "spring"
    return "carbon_low_alloy"


def fetch_and_harmonize() -> pd.DataFrame:
    df = pd.read_excel(RAW_XLSX)
    print(f"[fetch] raw shape: {df.shape}")

    df = df.rename(columns=RENAME)
    missing = set(RENAME.values()) - set(df.columns)
    if missing:
        raise ValueError(f"missing expected columns after rename: {missing}")

    df["sub_class"] = df.apply(_classify_heat, axis=1)
    df["source"] = "agrawal_2014_nims_fatigue"
    df["source_doi"] = "10.1186/2193-9772-3-8"

    return df


def write_stats(df: pd.DataFrame) -> dict:
    sub_counts = df["sub_class"].value_counts().to_dict()
    stats = {
        "n_records": int(len(df)),
        "n_features": int(df.shape[1]) - 3,  # minus provenance & target
        "target": "fatigue_strength_mpa",
        "target_range_mpa": [
            float(df["fatigue_strength_mpa"].min()),
            float(df["fatigue_strength_mpa"].max()),
        ],
        "target_mean_mpa": float(df["fatigue_strength_mpa"].mean()),
        "sub_class_counts": {k: int(v) for k, v in sub_counts.items()},
        "composition_columns": [
            c for c in df.columns if c.endswith("_pct") or c == "reduction_ratio"
        ],
        "processing_columns": [
            c for c in df.columns
            if c.endswith("_temp_c") or c.endswith("_time_min")
            or c.endswith("_cooling_rate_c_per_s")
        ],
        "inclusion_columns": [c for c in df.columns if c.startswith("inclusion_")],
    }
    STATS_JSON.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def main() -> None:
    if not RAW_XLSX.exists():
        raise FileNotFoundError(
            f"raw xlsx not found at {RAW_XLSX}. "
            "Download via: curl -sL -o data/public_raw/agrawal_nims_fatigue_raw.xlsx "
            "https://raw.githubusercontent.com/hunterkimmett/Fatigue-Machine-Learning/main/fatigue_dataset.xlsx"
        )

    df = fetch_and_harmonize()
    df.to_parquet(OUTPUT, index=False)
    stats = write_stats(df)

    print(f"[fetch] saved {len(df)} records to {OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"[fetch] sub-class distribution: {stats['sub_class_counts']}")
    print(f"[fetch] fatigue_strength range: "
          f"{stats['target_range_mpa'][0]:.0f} - {stats['target_range_mpa'][1]:.0f} MPa "
          f"(mean {stats['target_mean_mpa']:.0f})")


if __name__ == "__main__":
    main()
