"""Heat records storage — foundation для AI deox калибровки (PR 1).

Pydantic-модель HeatRecord + SQLite CRUD. Схема в data/heats/schema.sql.
БД gitignored (data/heats/heats.db), populated runtime: manual entry (PR 2),
Excel ETL (PR 3), CSV bulk (PR 3 CLI), synthetic generator (PR 4).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, Field

# Project layout — data/heats/heats.db (gitignored), data/heats/schema.sql (versioned)
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "heats" / "heats.db"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "data" / "heats" / "schema.sql"

Source = Literal["manual", "excel_etl", "csv_bulk", "synthetic"]
AdditionTiming = Literal["in_stream", "trim_after_lf_arrival", "split"]
CarrierGas = Literal["none", "Ar", "N2"]
VacuumTreatment = Literal["none", "VD", "RH"]
QualityFlag = Literal["accept", "out_of_spec", "unknown"]


class HeatRecord(BaseModel):
    """One ladle deoxidation heat record. Bounds aligned with app/api/routers/deox.py."""

    # Identification
    id: int | None = None  # set by SQLite AUTOINCREMENT after insert
    created_at: datetime | None = None  # auto-filled UTC at insert if None
    source: Source
    plant_id: str = Field(..., min_length=1, max_length=64)
    heat_id: str | None = Field(default=None, max_length=64)
    steel_class_id: str | None = Field(default=None, max_length=64)
    # Heat parameters
    steel_mass_ton: float = Field(..., ge=1.0, le=500.0)
    o_a_initial_ppm: float = Field(..., ge=0.0, le=2000.0)
    o_a_after_ppm: float | None = Field(default=None, ge=0.0, le=2000.0)
    t_tap_c: float | None = Field(default=None, ge=1400.0, le=1700.0)
    t_lf_arrival_c: float | None = Field(default=None, ge=1400.0, le=1700.0)
    t_al_addition_c: float | None = Field(default=None, ge=1400.0, le=1700.0)
    al_added_kg: float | None = Field(default=None, ge=0.0, le=5000.0)
    al_residual_pct: float | None = Field(default=None, ge=0.0, le=0.5)
    # Slag
    slag_mass_kg: float | None = Field(default=None, ge=0.0, le=10000.0)
    carry_over_slag_kg_per_t: float | None = Field(default=None, ge=0.0, le=50.0)
    slag_feo_pct: float | None = Field(default=None, ge=0.0, le=50.0)
    slag_mno_pct: float | None = Field(default=None, ge=0.0, le=20.0)
    slag_sio2_pct: float | None = Field(default=None, ge=0.0, le=30.0)
    slag_cao_pct: float | None = Field(default=None, ge=0.0, le=70.0)
    slag_mgo_pct: float | None = Field(default=None, ge=0.0, le=25.0)
    slag_al2o3_pct: float | None = Field(default=None, ge=0.0, le=50.0)
    # Composition
    c_pct: float | None = Field(default=None, ge=0.0, le=1.5)
    mn_pct: float | None = Field(default=None, ge=0.0, le=3.0)
    si_pct: float | None = Field(default=None, ge=0.0, le=2.5)
    s_pct: float | None = Field(default=None, ge=0.0, le=0.05)
    p_pct: float | None = Field(default=None, ge=0.0, le=0.05)
    # Method / process
    method_id: str | None = Field(default=None, max_length=64)
    addition_timing: AdditionTiming | None = None
    carrier_gas: CarrierGas | None = None
    co_deox_fesi_kg: float | None = Field(default=None, ge=0.0, le=5000.0)
    dt_to_al_min: float | None = Field(default=None, ge=0.0, le=120.0)
    t_drying_c: float | None = Field(default=None, ge=0.0, le=600.0)
    ar_stir_nm3: float | None = Field(default=None, ge=0.0, le=100.0)
    vacuum_treatment: VacuumTreatment | None = None
    refractory_heat_count: int | None = Field(default=None, ge=0, le=500)
    # Outcome
    eta_al_effective: float | None = Field(default=None, ge=0.0, le=1.5)
    quality_flag: QualityFlag | None = None
    notes: str | None = Field(default=None, max_length=4000)
    extras: dict | None = None  # serialized to JSON in extras column

    model_config = {"protected_namespaces": ()}


def _init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Initialize SQLite DB by executing schema.sql. Idempotent."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(ddl)
        conn.commit()


# Column order MUST match INSERT statement — single source of truth here
_COLUMNS = (
    "created_at", "source", "plant_id", "heat_id", "steel_class_id",
    "steel_mass_ton", "o_a_initial_ppm", "o_a_after_ppm",
    "t_tap_c", "t_lf_arrival_c", "t_al_addition_c",
    "al_added_kg", "al_residual_pct",
    "slag_mass_kg", "carry_over_slag_kg_per_t",
    "slag_feo_pct", "slag_mno_pct", "slag_sio2_pct",
    "slag_cao_pct", "slag_mgo_pct", "slag_al2o3_pct",
    "c_pct", "mn_pct", "si_pct", "s_pct", "p_pct",
    "method_id", "addition_timing", "carrier_gas",
    "co_deox_fesi_kg", "dt_to_al_min", "t_drying_c",
    "ar_stir_nm3", "vacuum_treatment", "refractory_heat_count",
    "eta_al_effective", "quality_flag", "notes", "extras",
)


def _record_to_row(record: HeatRecord) -> tuple:
    """Convert pydantic HeatRecord → tuple matching _COLUMNS order."""
    created = record.created_at or datetime.now(timezone.utc)
    if isinstance(created, datetime):
        if created.tzinfo is None:
            created_iso = created.replace(tzinfo=timezone.utc).isoformat()
        else:
            created_iso = created.astimezone(timezone.utc).isoformat()
    else:
        created_iso = str(created)
    extras_json = json.dumps(record.extras, ensure_ascii=False) if record.extras else None
    return (
        created_iso, record.source, record.plant_id, record.heat_id, record.steel_class_id,
        record.steel_mass_ton, record.o_a_initial_ppm, record.o_a_after_ppm,
        record.t_tap_c, record.t_lf_arrival_c, record.t_al_addition_c,
        record.al_added_kg, record.al_residual_pct,
        record.slag_mass_kg, record.carry_over_slag_kg_per_t,
        record.slag_feo_pct, record.slag_mno_pct, record.slag_sio2_pct,
        record.slag_cao_pct, record.slag_mgo_pct, record.slag_al2o3_pct,
        record.c_pct, record.mn_pct, record.si_pct, record.s_pct, record.p_pct,
        record.method_id, record.addition_timing, record.carrier_gas,
        record.co_deox_fesi_kg, record.dt_to_al_min, record.t_drying_c,
        record.ar_stir_nm3, record.vacuum_treatment, record.refractory_heat_count,
        record.eta_al_effective, record.quality_flag, record.notes, extras_json,
    )


def _row_to_record(row: sqlite3.Row) -> HeatRecord:
    """Convert SQLite Row → HeatRecord. Parse extras JSON, ISO ts → datetime."""
    d = dict(row)
    extras_raw = d.pop("extras", None)
    d["extras"] = json.loads(extras_raw) if extras_raw else None
    if d.get("created_at"):
        d["created_at"] = datetime.fromisoformat(d["created_at"])
    return HeatRecord(**d)


def insert_heat(record: HeatRecord, db_path: Path = DEFAULT_DB_PATH) -> int:
    """Insert one HeatRecord; returns lastrowid."""
    _init_db(db_path)
    placeholders = ", ".join("?" * len(_COLUMNS))
    cols = ", ".join(_COLUMNS)
    sql = f"INSERT INTO heats ({cols}) VALUES ({placeholders})"
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute(sql, _record_to_row(record))
        conn.commit()
        return int(cur.lastrowid)


def bulk_insert_heats(records: list[HeatRecord], db_path: Path = DEFAULT_DB_PATH) -> list[int]:
    """Insert N HeatRecords in single transaction; returns list of lastrowid (in order)."""
    if not records:
        return []
    _init_db(db_path)
    placeholders = ", ".join("?" * len(_COLUMNS))
    cols = ", ".join(_COLUMNS)
    sql = f"INSERT INTO heats ({cols}) VALUES ({placeholders})"
    ids: list[int] = []
    with sqlite3.connect(str(db_path)) as conn:
        for r in records:
            cur = conn.execute(sql, _record_to_row(r))
            ids.append(int(cur.lastrowid))
        conn.commit()
    return ids


def get_heat_by_id(heat_pk: int, db_path: Path = DEFAULT_DB_PATH) -> HeatRecord | None:
    """Lookup by primary key (id). Returns None if missing."""
    _init_db(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM heats WHERE id = ?", (heat_pk,))
        row = cur.fetchone()
    return _row_to_record(row) if row else None


def list_heats(
    *,
    plant_id: str | None = None,
    method_id: str | None = None,
    has_outcome: bool | None = None,
    limit: int = 1000,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[HeatRecord]:
    """Filter + paginate heat records. has_outcome=True → only finished heats."""
    _init_db(db_path)
    clauses: list[str] = []
    params: list = []
    if plant_id is not None:
        clauses.append("plant_id = ?")
        params.append(plant_id)
    if method_id is not None:
        clauses.append("method_id = ?")
        params.append(method_id)
    if has_outcome is True:
        clauses.append("o_a_after_ppm IS NOT NULL AND al_added_kg IS NOT NULL")
    elif has_outcome is False:
        clauses.append("(o_a_after_ppm IS NULL OR al_added_kg IS NULL)")
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    sql = f"SELECT * FROM heats {where} ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
    return [_row_to_record(r) for r in rows]


def update_heat_outcome(
    heat_pk: int,
    *,
    o_a_after_ppm: float | None = None,
    al_residual_pct: float | None = None,
    eta_al_effective: float | None = None,
    quality_flag: QualityFlag | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Update outcome fields of in-progress heat. Silently no-op if all args None."""
    updates: dict = {}
    if o_a_after_ppm is not None:
        updates["o_a_after_ppm"] = o_a_after_ppm
    if al_residual_pct is not None:
        updates["al_residual_pct"] = al_residual_pct
    if eta_al_effective is not None:
        updates["eta_al_effective"] = eta_al_effective
    if quality_flag is not None:
        updates["quality_flag"] = quality_flag
    if not updates:
        return
    _init_db(db_path)
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    sql = f"UPDATE heats SET {set_clause} WHERE id = ?"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(sql, list(updates.values()) + [heat_pk])
        conn.commit()


def delete_heat(heat_pk: int, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """Delete by id; returns True if row was deleted, False if missing."""
    _init_db(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute("DELETE FROM heats WHERE id = ?", (heat_pk,))
        conn.commit()
        return cur.rowcount > 0


def count_heats(
    *, plant_id: str | None = None, db_path: Path = DEFAULT_DB_PATH
) -> int:
    """Count rows (optionally filtered by plant_id)."""
    _init_db(db_path)
    if plant_id is None:
        sql, params = "SELECT COUNT(*) FROM heats", []
    else:
        sql, params = "SELECT COUNT(*) FROM heats WHERE plant_id = ?", [plant_id]
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute(sql, params)
        return int(cur.fetchone()[0])


@contextmanager
def heats_session(db_path: Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Context manager for bulk operations — single commit at exit, rollback on exc.

    Usage: with heats_session() as conn: conn.execute(...); conn.execute(...).
    Used by ETL (PR 3) for multi-1000-row imports.
    """
    _init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


__all__ = [
    "HeatRecord", "Source", "AdditionTiming", "CarrierGas",
    "VacuumTreatment", "QualityFlag",
    "DEFAULT_DB_PATH", "SCHEMA_PATH",
    "insert_heat", "bulk_insert_heats", "get_heat_by_id",
    "list_heats", "update_heat_outcome", "delete_heat", "count_heats",
    "heats_session",
]


if __name__ == "__main__":
    # Dry-run demo: create in-memory DB, insert sample record, query back
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "heats_demo.db"
    sample = HeatRecord(
        source="manual", plant_id="ASIS_BOF",
        steel_mass_ton=371.0, o_a_initial_ppm=657.0, o_a_after_ppm=4.5,
        t_al_addition_c=1605.0, al_added_kg=454.0, al_residual_pct=0.022,
        slag_mass_kg=2200.0, slag_feo_pct=18.0, method_id="asis_shot",
    )
    pk = insert_heat(sample, db_path=tmp)
    print(f"Inserted id={pk}, count={count_heats(db_path=tmp)}")
    print(f"Back: {get_heat_by_id(pk, db_path=tmp)}")
