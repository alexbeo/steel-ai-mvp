"""
Public-data spike · Phase 0.1 · Step 1 / 3

Fetches the steel_strength dataset via matminer, inspects the composition
space, and saves to data/public_matminer.parquet.

Finding (see docs/public_data_spike_report.md): the dataset covers
maraging / tool / high-strength steels, NOT HSLA pipe steels. The loader
still saves it — we use it downstream to prove pipeline generalizes to
another steel class, and to quantify OOD separation from our synthetic
HSLA generator.

Run:
    PYTHONPATH=. .venv/bin/python scripts/fetch_public_steel_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT = DATA_DIR / "public_matminer.parquet"


def fetch_and_harmonize() -> pd.DataFrame:
    """Load matminer steel_strength and harmonize column names to
    snake_case consistent with our synthetic generator."""
    from matminer.datasets import load_dataset

    df = load_dataset("steel_strength")
    print(f"[fetch] Loaded {len(df)} records, {df.shape[1]} columns")

    # Harmonize: matminer uses lowercase element symbols + spaces in target names.
    # Map to our schema.
    rename = {
        "c": "c_pct",
        "mn": "mn_pct",
        "si": "si_pct",
        "cr": "cr_pct",
        "ni": "ni_pct",
        "mo": "mo_pct",
        "v": "v_pct",
        "n": "n_pct",
        "nb": "nb_pct",
        "co": "co_pct",
        "w": "w_pct",
        "al": "al_pct",
        "ti": "ti_pct",
        "yield strength": "yield_strength_mpa",
        "tensile strength": "tensile_strength_mpa",
        "elongation": "elongation_pct",
    }
    df = df.rename(columns=rename)

    # Provenance: tag every row so we can do GroupKFold by source if needed.
    df["source"] = "matminer_steel_strength"
    df["source_dataset"] = "matminer/steel_strength"
    df["source_url"] = "https://figshare.com/articles/dataset/Steel_Strength_Data/7250453"
    df["source_license"] = "open (matminer curation, figshare)"

    # Drop the formula blob — too heavy; composition is already in wt% columns.
    if "formula" in df.columns:
        df = df.drop(columns=["formula"])
    if "composition" in df.columns:
        df = df.drop(columns=["composition"])

    return df


def describe_distribution(df: pd.DataFrame) -> dict:
    """Emit summary stats for comparison against synthetic HSLA."""
    stats = {}
    for col in ["c_pct", "mn_pct", "ni_pct", "cr_pct", "co_pct",
                "nb_pct", "ti_pct", "v_pct", "al_pct",
                "yield_strength_mpa", "tensile_strength_mpa"]:
        if col in df.columns:
            stats[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "n_nonzero": int((df[col] > 0).sum()),
            }
    return stats


def print_hsla_comparison(stats: dict) -> None:
    """Text comparison — is there overlap with HSLA range?"""
    hsla_bounds = {
        "c_pct":   (0.04, 0.12, "HSLA typical"),
        "mn_pct":  (0.9,  1.75, "HSLA typical"),
        "ni_pct":  (0.0,  0.40, "HSLA typical"),
        "cr_pct":  (0.0,  0.30, "HSLA typical"),
        "co_pct":  (0.0,  0.0,  "HSLA: none"),
        "nb_pct":  (0.0,  0.06, "HSLA micro"),
        "ti_pct":  (0.0,  0.025, "HSLA micro"),
        "yield_strength_mpa": (400, 600, "HSLA X60-X80"),
    }
    print()
    print("=" * 78)
    print("HSLA vs matminer-steel_strength composition / property ranges")
    print("=" * 78)
    print(f"{'Feature':<22} {'HSLA range':<22} {'Matminer range':<22} {'Overlap?':<10}")
    print("-" * 78)
    for k, (lo_h, hi_h, note) in hsla_bounds.items():
        if k not in stats:
            continue
        lo_m = stats[k]["min"]
        hi_m = stats[k]["max"]
        # Overlap: their range intersects ours?
        overlap = not (hi_m < lo_h or lo_m > hi_h)
        overlap_str = "✓" if overlap else "✗ NO"
        print(f"{k:<22} [{lo_h:>5}, {hi_h:>6}]     "
              f"[{lo_m:>5.2f}, {hi_m:>7.1f}]   {overlap_str}")
    print("=" * 78)


def main() -> None:
    print("Phase 0.1 · Step 1 of 3 — fetch public steel data")
    print()

    df = fetch_and_harmonize()
    print(f"[harmonize] {len(df)} records after column rename")
    print(f"[harmonize] Columns: {sorted(df.columns.tolist())}")

    stats = describe_distribution(df)
    print_hsla_comparison(stats)

    df.to_parquet(OUTPUT, index=False)
    print()
    print(f"[save] Wrote {len(df)} records -> {OUTPUT}")
    print(f"[save] File size: {OUTPUT.stat().st_size / 1024:.1f} KB")

    # Also dump stats as JSON for downstream steps
    stats_path = DATA_DIR / "public_matminer_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[save] Stats -> {stats_path}")


if __name__ == "__main__":
    main()
