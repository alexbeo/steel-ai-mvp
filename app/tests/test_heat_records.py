"""Unit tests for app.backend.heat_records (PR 1 — HeatRecord + SQLite schema)."""
from __future__ import annotations

import sqlite3

import pytest
from pydantic import ValidationError

from app.backend.heat_records import (
    HeatRecord,
    _init_db,
    bulk_insert_heats,
    count_heats,
    delete_heat,
    get_heat_by_id,
    heats_session,
    insert_heat,
    list_heats,
    update_heat_outcome,
)


# ---------- helpers ----------

def _minimal_record(**overrides) -> HeatRecord:
    """Build smallest valid HeatRecord (only required fields), overrides for variation."""
    defaults: dict = dict(
        source="manual",
        plant_id="ASIS_BOF",
        steel_mass_ton=371.0,
        o_a_initial_ppm=657.0,
    )
    defaults.update(overrides)
    return HeatRecord(**defaults)


# ---------- schema init ----------

def test_init_db_creates_schema(tmp_path):
    db = tmp_path / "heats.db"
    _init_db(db)
    with sqlite3.connect(str(db)) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='heats'"
        )
        assert cur.fetchone() is not None
        cur = conn.execute("SELECT COUNT(*) FROM heats")
        assert cur.fetchone()[0] == 0


def test_init_db_idempotent(tmp_path):
    db = tmp_path / "heats.db"
    _init_db(db)
    _init_db(db)  # second call must not raise
    with sqlite3.connect(str(db)) as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_heats_plant'"
        )
        assert cur.fetchone()[0] == 1  # not duplicated


# ---------- insert / roundtrip ----------

def test_insert_minimal_record(tmp_path):
    db = tmp_path / "heats.db"
    pk = insert_heat(_minimal_record(), db_path=db)
    assert pk == 1
    assert count_heats(db_path=db) == 1


def test_insert_full_record_roundtrip(tmp_path):
    db = tmp_path / "heats.db"
    full = HeatRecord(
        source="excel_etl",
        plant_id="MMK_LF",
        heat_id="H-2026-042",
        steel_class_id="pipe_hsla",
        steel_mass_ton=350.0,
        o_a_initial_ppm=600.0,
        o_a_after_ppm=4.2,
        t_tap_c=1640.0,
        t_lf_arrival_c=1590.0,
        t_al_addition_c=1605.0,
        al_added_kg=420.5,
        al_residual_pct=0.025,
        slag_mass_kg=2100.0,
        carry_over_slag_kg_per_t=6.0,
        slag_feo_pct=17.0,
        slag_mno_pct=4.0,
        slag_sio2_pct=12.0,
        slag_cao_pct=48.0,
        slag_mgo_pct=8.0,
        slag_al2o3_pct=22.0,
        c_pct=0.08,
        mn_pct=1.4,
        si_pct=0.25,
        s_pct=0.004,
        p_pct=0.012,
        method_id="asis_shot",
        addition_timing="trim_after_lf_arrival",
        carrier_gas="Ar",
        co_deox_fesi_kg=120.0,
        dt_to_al_min=8.0,
        t_drying_c=180.0,
        ar_stir_nm3=4.5,
        vacuum_treatment="VD",
        refractory_heat_count=85,
        eta_al_effective=0.82,
        quality_flag="accept",
        notes="Sample roundtrip heat",
    )
    pk = insert_heat(full, db_path=db)
    back = get_heat_by_id(pk, db_path=db)
    assert back is not None
    # Compare every input field exactly (id + created_at are added by storage)
    for field in HeatRecord.model_fields:
        if field in ("id", "created_at"):
            continue
        assert getattr(back, field) == getattr(full, field), f"mismatch on {field}"
    assert back.id == pk
    assert back.created_at is not None  # auto-filled UTC


# ---------- pydantic bounds (defense layer 1) ----------

def test_bounds_o_a_initial_rejects_negative():
    with pytest.raises(ValidationError):
        _minimal_record(o_a_initial_ppm=-1.0)


def test_bounds_temperature_rejects_below_1400():
    with pytest.raises(ValidationError):
        _minimal_record(t_tap_c=1300.0)


def test_bounds_steel_mass_rejects_zero():
    with pytest.raises(ValidationError):
        _minimal_record(steel_mass_ton=0.0)


# ---------- read / filter ----------

def test_get_by_id_returns_none_for_missing(tmp_path):
    db = tmp_path / "heats.db"
    assert get_heat_by_id(999_999, db_path=db) is None


def test_list_filter_by_plant_id(tmp_path):
    db = tmp_path / "heats.db"
    insert_heat(_minimal_record(plant_id="ASIS_BOF"), db_path=db)
    insert_heat(_minimal_record(plant_id="ASIS_BOF"), db_path=db)
    insert_heat(_minimal_record(plant_id="MMK_LF"), db_path=db)
    asis = list_heats(plant_id="ASIS_BOF", db_path=db)
    assert len(asis) == 2
    assert all(r.plant_id == "ASIS_BOF" for r in asis)


def test_list_filter_by_has_outcome(tmp_path):
    db = tmp_path / "heats.db"
    # 2 in-progress (no o_a_after, no al_added)
    insert_heat(_minimal_record(), db_path=db)
    insert_heat(_minimal_record(), db_path=db)
    # 1 finished
    insert_heat(
        _minimal_record(o_a_after_ppm=5.0, al_added_kg=400.0),
        db_path=db,
    )
    finished = list_heats(has_outcome=True, db_path=db)
    assert len(finished) == 1
    in_progress = list_heats(has_outcome=False, db_path=db)
    assert len(in_progress) == 2


# ---------- update ----------

def test_update_heat_outcome_sets_eta(tmp_path):
    db = tmp_path / "heats.db"
    pk = insert_heat(_minimal_record(), db_path=db)
    update_heat_outcome(
        pk,
        o_a_after_ppm=5.0,
        eta_al_effective=0.82,
        quality_flag="accept",
        db_path=db,
    )
    back = get_heat_by_id(pk, db_path=db)
    assert back is not None
    assert back.o_a_after_ppm == 5.0
    assert back.eta_al_effective == 0.82
    assert back.quality_flag == "accept"


def test_update_heat_outcome_noop_if_all_none(tmp_path):
    db = tmp_path / "heats.db"
    pk = insert_heat(_minimal_record(), db_path=db)
    before = get_heat_by_id(pk, db_path=db)
    update_heat_outcome(pk, db_path=db)  # all kwargs None
    after = get_heat_by_id(pk, db_path=db)
    assert before is not None and after is not None
    assert before.o_a_after_ppm == after.o_a_after_ppm
    assert before.eta_al_effective == after.eta_al_effective
    assert before.quality_flag == after.quality_flag


# ---------- delete ----------

def test_delete_heat_returns_false_if_missing(tmp_path):
    db = tmp_path / "heats.db"
    _init_db(db)
    assert delete_heat(999_999, db_path=db) is False


def test_delete_heat_returns_true_and_removes(tmp_path):
    db = tmp_path / "heats.db"
    pk = insert_heat(_minimal_record(), db_path=db)
    assert delete_heat(pk, db_path=db) is True
    assert count_heats(db_path=db) == 0


# ---------- bulk ----------

def test_bulk_insert_returns_correct_ids(tmp_path):
    db = tmp_path / "heats.db"
    r1 = _minimal_record(plant_id="A")
    r2 = _minimal_record(plant_id="B")
    r3 = _minimal_record(plant_id="C")
    ids = bulk_insert_heats([r1, r2, r3], db_path=db)
    assert len(ids) == 3
    assert ids[1] == ids[0] + 1
    assert ids[2] == ids[0] + 2
    assert count_heats(db_path=db) == 3


def test_bulk_insert_empty_list_returns_empty(tmp_path):
    db = tmp_path / "heats.db"
    assert bulk_insert_heats([], db_path=db) == []


# ---------- extras (JSON column) ----------

def test_extras_roundtrip_dict(tmp_path):
    db = tmp_path / "heats.db"
    pk = insert_heat(
        _minimal_record(extras={"h_ppm": 1.5, "comment": "test"}),
        db_path=db,
    )
    back = get_heat_by_id(pk, db_path=db)
    assert back is not None
    assert back.extras == {"h_ppm": 1.5, "comment": "test"}


def test_extras_unicode(tmp_path):
    db = tmp_path / "heats.db"
    pk = insert_heat(
        _minimal_record(extras={"описание": "плавка ASIS №42"}),
        db_path=db,
    )
    back = get_heat_by_id(pk, db_path=db)
    assert back is not None
    assert back.extras == {"описание": "плавка ASIS №42"}


# ---------- transactional session ----------

def test_heats_session_commits_on_success(tmp_path):
    db = tmp_path / "heats.db"
    with heats_session(db_path=db) as conn:
        conn.execute(
            "INSERT INTO heats (created_at, source, plant_id, steel_mass_ton, o_a_initial_ppm)"
            " VALUES (?, ?, ?, ?, ?)",
            ("2026-05-19T00:00:00+00:00", "manual", "X", 100.0, 500.0),
        )
        conn.execute(
            "INSERT INTO heats (created_at, source, plant_id, steel_mass_ton, o_a_initial_ppm)"
            " VALUES (?, ?, ?, ?, ?)",
            ("2026-05-19T00:00:01+00:00", "manual", "X", 100.0, 500.0),
        )
    assert count_heats(db_path=db) == 2


def test_heats_session_rollback_on_exception(tmp_path):
    db = tmp_path / "heats.db"
    with pytest.raises(RuntimeError):
        with heats_session(db_path=db) as conn:
            conn.execute(
                "INSERT INTO heats (created_at, source, plant_id, steel_mass_ton, o_a_initial_ppm)"
                " VALUES (?, ?, ?, ?, ?)",
                ("2026-05-19T00:00:00+00:00", "manual", "Y", 100.0, 500.0),
            )
            raise RuntimeError("force rollback")
    assert count_heats(db_path=db) == 0


# ---------- count ----------

def test_count_heats_by_plant(tmp_path):
    db = tmp_path / "heats.db"
    insert_heat(_minimal_record(plant_id="A"), db_path=db)
    insert_heat(_minimal_record(plant_id="A"), db_path=db)
    insert_heat(_minimal_record(plant_id="B"), db_path=db)
    assert count_heats(db_path=db) == 3
    assert count_heats(plant_id="A", db_path=db) == 2
    assert count_heats(plant_id="B", db_path=db) == 1


# ---------- SQL CHECK constraints (defense layer 2: bypass pydantic) ----------

def test_check_constraint_o_a_after_via_raw_sql(tmp_path):
    db = tmp_path / "heats.db"
    _init_db(db)
    with pytest.raises(sqlite3.IntegrityError):
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                "INSERT INTO heats (created_at, source, plant_id, steel_mass_ton,"
                " o_a_initial_ppm, o_a_after_ppm)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                ("2026-05-19T00:00:00+00:00", "manual", "X", 100.0, 500.0, 3000.0),
            )
            conn.commit()


def test_check_constraint_addition_timing_enum(tmp_path):
    db = tmp_path / "heats.db"
    _init_db(db)
    with pytest.raises(sqlite3.IntegrityError):
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                "INSERT INTO heats (created_at, source, plant_id, steel_mass_ton,"
                " o_a_initial_ppm, addition_timing)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                ("2026-05-19T00:00:00+00:00", "manual", "X", 100.0, 500.0, "bad_value"),
            )
            conn.commit()
