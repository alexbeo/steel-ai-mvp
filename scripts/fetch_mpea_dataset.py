"""
Path 2 · B' · Step 1 / 2

Loads the Citrine MPEA dataset (Borg et al., Scientific Data 2020,
1545 records / 630 unique multi-principal-element alloys), parses the
chemical-formula strings into mole fractions for 15 common elements,
harmonizes column names, and saves data/mpea_dataset.parquet.

Substitutes Citrine Conduit 800-steel dataset (Path 2 · B) which was
blocked: Citrination decommissioned, API returns 403 Access Denied,
and matminer does not wrap it. MPEA is strictly more useful for
cross-class robustness testing — HEA alloys are a completely different
composition-space family than matminer's maraging/tool steels.

Source: Apache 2.0 redistribution via CitrineInformatics/MPEA_dataset
    https://github.com/CitrineInformatics/MPEA_dataset (master/MPEA_dataset.csv)
Original paper: Borg et al., "Expanded dataset of mechanical properties
and observed phases of multi-principal element alloys",
Scientific Data 7:430 (2020). DOI: 10.1038/s41597-020-00768-9.

Run:
    PYTHONPATH=. .venv/bin/python scripts/fetch_mpea_dataset.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "public_raw" / "mpea_dataset_raw.csv"
OUTPUT = PROJECT_ROOT / "data" / "mpea_dataset.parquet"
STATS_JSON = PROJECT_ROOT / "docs" / "mpea_dataset_stats.json"

TRACKED_ELEMENTS = [
    "Al", "Co", "Cr", "Cu", "Fe", "Hf", "Mn", "Mo",
    "Nb", "Ni", "Ta", "Ti", "V", "W", "Zr",
]

FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)\s*([\d.]*)")


def parse_formula_to_mole_fractions(formula: str) -> dict[str, float]:
    """'Al0.25 Co1 Fe1 Ni1' → {'al_mol_frac': 0.077, 'co_mol_frac': 0.308, ...}."""
    if not isinstance(formula, str) or not formula.strip():
        return {}
    raw: dict[str, float] = {}
    for elem, coef in FORMULA_TOKEN_RE.findall(formula.strip()):
        if not elem:
            continue
        v = float(coef) if coef else 1.0
        raw[elem] = raw.get(elem, 0.0) + v
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {f"{e.lower()}_mol_frac": raw.get(e, 0.0) / total for e in TRACKED_ELEMENTS}


def fetch_and_harmonize() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    print(f"[fetch] raw shape: {df.shape}")

    keep_cols = {
        "IDENTIFIER: Reference ID": "reference_id",
        "FORMULA": "formula",
        "PROPERTY: Microstructure": "microstructure",
        "PROPERTY: Processing method": "processing_method",
        "PROPERTY: BCC/FCC/other": "phase",
        "PROPERTY: grain size ($\\mu$m)": "grain_size_um",
        "PROPERTY: HV": "hardness_hv",
        "PROPERTY: Type of test": "test_type",
        "PROPERTY: Test temperature ($^\\circ$C)": "test_temperature_c",
        "PROPERTY: YS (MPa)": "yield_strength_mpa",
        "PROPERTY: UTS (MPa)": "tensile_strength_mpa",
        "PROPERTY: Elongation (%)": "elongation_pct",
        "REFERENCE: doi": "reference_doi",
        "REFERENCE: year": "reference_year",
    }
    df = df[list(keep_cols.keys())].rename(columns=keep_cols)

    mole_fracs = df["formula"].apply(parse_formula_to_mole_fractions).apply(pd.Series)
    mole_fracs = mole_fracs.reindex(columns=[f"{e.lower()}_mol_frac" for e in TRACKED_ELEMENTS]).fillna(0.0)
    df = pd.concat([df, mole_fracs], axis=1)

    df["source"] = "citrine_mpea_dataset"
    df["source_doi"] = "10.1038/s41597-020-00768-9"
    return df


def write_stats(df: pd.DataFrame) -> dict:
    stats = {
        "n_records": int(len(df)),
        "n_unique_formulas": int(df["formula"].nunique()),
        "tracked_elements": TRACKED_ELEMENTS,
        "non_null_counts": {
            "yield_strength_mpa": int(df["yield_strength_mpa"].notna().sum()),
            "tensile_strength_mpa": int(df["tensile_strength_mpa"].notna().sum()),
            "elongation_pct": int(df["elongation_pct"].notna().sum()),
            "hardness_hv": int(df["hardness_hv"].notna().sum()),
        },
        "test_type_counts": df["test_type"].value_counts().dropna().to_dict(),
        "yield_range_mpa": [
            float(df["yield_strength_mpa"].min()),
            float(df["yield_strength_mpa"].max()),
        ],
        "test_temperature_c_range": [
            float(df["test_temperature_c"].min()),
            float(df["test_temperature_c"].max()),
        ],
    }
    STATS_JSON.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"raw csv not found at {RAW_CSV}. "
            "Download via: curl -sL -o data/public_raw/mpea_dataset_raw.csv "
            "https://raw.githubusercontent.com/CitrineInformatics/MPEA_dataset/master/MPEA_dataset.csv"
        )

    df = fetch_and_harmonize()
    df.to_parquet(OUTPUT, index=False)
    stats = write_stats(df)

    print(f"[fetch] saved {len(df)} records to {OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"[fetch] unique formulas: {stats['n_unique_formulas']}")
    print(f"[fetch] YS non-null: {stats['non_null_counts']['yield_strength_mpa']} records "
          f"(range {stats['yield_range_mpa'][0]:.0f}-{stats['yield_range_mpa'][1]:.0f} MPa)")


if __name__ == "__main__":
    main()
