"""Unit tests for app.backend.heat_history_etl (PR 3 — Excel ETL).

Uses tmp_path for synthetic xlsx files and a temporary heats.db. Verifies the
deterministic detector, LLM hook, dry-run, validation skipping, and CLI smoke.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

from app.backend import heat_records as _hr
from app.backend.heat_etl_critic import (
    EtlMappingProposal,
    ProposedMapping,
    UnmappedColumn,
)
from app.backend.heat_history_etl import (
    ColumnMatch,
    DEFAULT_ALIASES_PATH,
    apply_llm_critic,
    build_records,
    detect_columns,
    import_excel,
    load_aliases,
)


@pytest.fixture
def aliases():
    return load_aliases(DEFAULT_ALIASES_PATH)


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Redirect bulk_insert_heats to a temp DB to avoid polluting data/heats/heats.db."""
    tmp_db = tmp_path / "heats.db"
    monkeypatch.setattr(_hr, "DEFAULT_DB_PATH", tmp_db)
    # heat_history_etl imports bulk_insert_heats by name, so patch its binding too
    from app.backend import heat_history_etl as ehe

    original = ehe.bulk_insert_heats
    monkeypatch.setattr(
        ehe, "bulk_insert_heats", lambda recs: original(recs, db_path=tmp_db)
    )
    return tmp_db


# ---------- alias detector ----------

def test_alias_detector_exact_match(aliases):
    df = pd.DataFrame(
        {
            "plant_id": ["A"],
            "steel_mass_ton": [100.0],
            "o_a_initial_ppm": [500.0],
        }
    )
    matches = detect_columns(df, aliases=aliases)
    by_col = {m.excel_column: m for m in matches}
    assert by_col["plant_id"].heat_record_field == "plant_id"
    assert by_col["plant_id"].source == "alias_exact"
    assert by_col["plant_id"].confidence == 1.0
    assert by_col["steel_mass_ton"].heat_record_field == "steel_mass_ton"
    assert by_col["o_a_initial_ppm"].heat_record_field == "o_a_initial_ppm"


def test_alias_detector_normalizes_unicode_and_translit(aliases):
    # "Масса_стали, т" — cyrillic + bracket-less unit → after normalize → "massa_stali_t"
    # alias list contains "massa_stali" (translit pre-normalized) → substring match.
    df = pd.DataFrame({"Масса_стали, т": [120]})
    matches = detect_columns(df, aliases=aliases)
    m = matches[0]
    assert m.heat_record_field == "steel_mass_ton"
    assert m.source in ("alias_exact", "alias_substring", "alias_fuzzy")
    assert m.confidence > 0


def test_alias_detector_substring_match(aliases):
    # Adds units suffix → "o_initial_ppm" alias exact-matches after normalize drops "(ppm)"
    df = pd.DataFrame({"O_initial (ppm)": [500.0]})
    m = detect_columns(df, aliases=aliases)[0]
    assert m.heat_record_field == "o_a_initial_ppm"
    # exact since "(ppm)" is dropped
    assert m.source in ("alias_exact", "alias_substring")


def test_alias_detector_fuzzy_with_low_confidence(aliases):
    # "alimnium_added" — typo close to "aluminum_added"
    df = pd.DataFrame({"alimnium_added": [120]})
    m = detect_columns(df, aliases=aliases)[0]
    # Should be fuzzy match → al_added_kg, confidence 0.4
    assert m.heat_record_field == "al_added_kg"
    assert m.source == "alias_fuzzy"
    assert 0.0 < m.confidence < 0.7


def test_alias_detector_returns_unmapped_for_garbage(aliases):
    df = pd.DataFrame({"zzzz_unknown_xyz": [1, 2, 3]})
    m = detect_columns(df, aliases=aliases)[0]
    assert m.heat_record_field is None
    assert m.source == "unmapped"
    assert m.confidence == 0.0


def test_user_overrides_win_over_alias(aliases):
    df = pd.DataFrame({"oa_init_col": [500], "weight_col": [200]})
    matches = detect_columns(
        df,
        aliases=aliases,
        user_overrides={
            "oa_init_col": "o_a_initial_ppm",
            "weight_col": "steel_mass_ton",
        },
    )
    by_col = {m.excel_column: m for m in matches}
    assert by_col["oa_init_col"].heat_record_field == "o_a_initial_ppm"
    assert by_col["oa_init_col"].source == "user_override"
    assert by_col["oa_init_col"].confidence == 1.0
    assert by_col["weight_col"].heat_record_field == "steel_mass_ton"


# ---------- end-to-end via xlsx ----------

def _write_xlsx(path: Path, df: pd.DataFrame) -> None:
    df.to_excel(path, index=False, engine="openpyxl")


def test_load_xlsx_minimal(tmp_path, isolated_db):
    path = tmp_path / "heats.xlsx"
    df = pd.DataFrame(
        {
            "plant": ["A", "B", "C"],
            "steel_mass_ton": [371.0, 350.0, 380.0],
            "o_initial": [657.0, 600.0, 700.0],
            "Al_kg": [454.0, 400.0, 500.0],
            "method": ["asis_shot", "ingot", "asis_shot"],
            "T_al": [1605.0, 1610.0, 1600.0],
        }
    )
    _write_xlsx(path, df)
    report = import_excel(path, plant_id="ASIS_BOF")
    assert report.rows_total == 3
    assert report.rows_inserted == 3
    assert report.rows_failed == 0
    # plant_id from CLI must override the "plant" column values
    inserted = _hr.list_heats(plant_id="ASIS_BOF", db_path=isolated_db)
    assert len(inserted) == 3
    assert all(r.plant_id == "ASIS_BOF" for r in inserted)
    assert {r.al_added_kg for r in inserted} == {454.0, 400.0, 500.0}


def test_load_xlsx_dry_run_does_not_write(tmp_path, isolated_db):
    path = tmp_path / "heats.xlsx"
    _write_xlsx(
        path,
        pd.DataFrame(
            {
                "plant": ["A"],
                "steel_mass_ton": [100.0],
                "o_initial": [500.0],
            }
        ),
    )
    report = import_excel(path, plant_id="X", dry_run=True)
    assert report.dry_run is True
    assert report.rows_inserted == 0
    assert _hr.count_heats(db_path=isolated_db) == 0


def test_load_xlsx_skips_bound_violation_row(tmp_path, isolated_db):
    path = tmp_path / "heats.xlsx"
    df = pd.DataFrame(
        {
            "plant": ["A", "B"],
            "steel_mass_ton": [100.0, 600.0],  # second row exceeds bound (max 500)
            "o_initial": [500.0, 550.0],
        }
    )
    _write_xlsx(path, df)
    report = import_excel(path, plant_id="X")
    assert report.rows_inserted == 1
    assert report.rows_failed == 1
    assert report.errors[0].row_idx == 1
    assert "steel_mass_ton" in report.errors[0].reason


def test_load_xlsx_with_mock_llm_critic(tmp_path, isolated_db):
    path = tmp_path / "heats.xlsx"
    df = pd.DataFrame(
        {
            "plant": ["A"],
            "steel_mass_ton": [100.0],
            "o_initial": [500.0],
            "weird_xyz_col": [1.5],  # ambiguous → goes to LLM
        }
    )
    _write_xlsx(path, df)

    # Patch make_heat_etl_critic to return a fake critic
    fake_critic = MagicMock()
    fake_critic.propose_mappings.return_value = EtlMappingProposal(
        mappings=[],
        unmapped=[
            UnmappedColumn(
                excel_column="weird_xyz_col",
                reason="не входит в HeatRecord",
            )
        ],
        usage={},
    )
    import app.backend.heat_etl_critic as etl_critic_mod

    original_factory = etl_critic_mod.make_heat_etl_critic
    etl_critic_mod.make_heat_etl_critic = lambda: fake_critic  # type: ignore[assignment]
    try:
        report = import_excel(path, plant_id="X", use_llm_critic=True)
    finally:
        etl_critic_mod.make_heat_etl_critic = original_factory  # type: ignore[assignment]

    assert report.llm_used is True
    assert report.rows_inserted == 1
    # The weird column should be marked unmapped from LLM
    weird = next(m for m in report.matches if m.excel_column == "weird_xyz_col")
    assert weird.heat_record_field is None
    assert weird.reasoning is not None


def test_user_mapping_file_overrides_alias(tmp_path, isolated_db):
    path = tmp_path / "heats.xlsx"
    _write_xlsx(
        path,
        pd.DataFrame(
            {
                "factory_code": ["A", "B"],
                "ton_amount": [100.0, 150.0],
                "ox_init": [500.0, 600.0],
            }
        ),
    )
    mapping_file = tmp_path / "overrides.yaml"
    mapping_file.write_text(
        yaml.safe_dump(
            {
                "factory_code": "plant_id",
                "ton_amount": "steel_mass_ton",
                "ox_init": "o_a_initial_ppm",
            }
        ),
        encoding="utf-8",
    )
    report = import_excel(path, plant_id="X", mapping_file=mapping_file)
    assert report.rows_inserted == 2
    overrides_count = sum(1 for m in report.matches if m.source == "user_override")
    assert overrides_count == 3


def test_apply_llm_critic_returns_unchanged_when_critic_none():
    matches = [
        ColumnMatch(
            excel_column="x",
            heat_record_field=None,
            confidence=0.0,
            source="unmapped",
            sample_values=[],
        )
    ]
    out = apply_llm_critic(matches, critic=None)
    assert out == matches


def test_apply_llm_critic_promotes_mapping():
    matches = [
        ColumnMatch(
            excel_column="weird",
            heat_record_field=None,
            confidence=0.0,
            source="unmapped",
            sample_values=["500"],
        )
    ]
    fake = MagicMock()
    fake.propose_mappings.return_value = EtlMappingProposal(
        mappings=[
            ProposedMapping(
                excel_column="weird",
                heat_record_field="o_a_initial_ppm",
                confidence="HIGH",
                reasoning="sample looks like oxygen activity",
            )
        ],
        unmapped=[],
        usage={},
    )
    out = apply_llm_critic(matches, critic=fake)
    assert out[0].heat_record_field == "o_a_initial_ppm"
    assert out[0].source == "llm"
    assert out[0].confidence == 0.9
    assert out[0].reasoning is not None


def test_build_records_drops_low_confidence_columns():
    df = pd.DataFrame(
        {
            "plant": ["A"],
            "steel_mass_ton": [100.0],
            "o_initial": [500.0],
            "junk": [42.0],
        }
    )
    matches = [
        ColumnMatch("plant", "plant_id", 1.0, "alias_exact", ["A"]),
        ColumnMatch("steel_mass_ton", "steel_mass_ton", 1.0, "alias_exact", ["100"]),
        ColumnMatch("o_initial", "o_a_initial_ppm", 1.0, "alias_exact", ["500"]),
        ColumnMatch("junk", "method_id", 0.3, "alias_fuzzy", ["42"]),  # too low → dropped
    ]
    records, errors = build_records(df, matches, plant_id="X")
    assert len(records) == 1
    assert len(errors) == 0
    assert records[0].method_id is None  # low-confidence column was skipped


def test_cli_dry_run_smoke(tmp_path, monkeypatch):
    # Build synthetic xlsx
    xlsx = tmp_path / "qa_etl.xlsx"
    pd.DataFrame(
        {
            "plant": ["A", "B"],
            "steel_mass_ton": [100, 150],
            "o_initial": [500, 600],
            "Al_kg": [100, 150],
        }
    ).to_excel(xlsx, index=False, engine="openpyxl")

    project_root = Path(__file__).resolve().parents[2]
    env_pythonpath = str(project_root)
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "import_heats_from_excel.py"),
            "--file",
            str(xlsx),
            "--plant",
            "CLI_SMOKE",
            "--dry-run",
        ],
        env={**__import__("os").environ, "PYTHONPATH": env_pythonpath},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "DRY RUN" in result.stdout
    assert "Detected:" in result.stdout
