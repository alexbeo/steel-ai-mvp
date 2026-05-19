"""Excel ETL для импорта исторических плавок в data/heats/heats.db.

Двухуровневый детектор колонок:
- Level 1 (детерминированный): alias-matching против data/heats/etl_aliases.yaml
  (exact / substring / difflib fuzzy)
- Level 2 (опциональный LLM): Sonnet-критик через heat_etl_critic.py для
  колонок с низкой уверенностью

CLI: scripts/import_heats_from_excel.py.
PR 3 (R-003) — bulk ingest для ASIS LF historical journals.
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import ValidationError

from app.backend.heat_records import (
    HeatRecord,
    Source,
    bulk_insert_heats,
)

logger = logging.getLogger(__name__)

DEFAULT_ALIASES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "heats" / "etl_aliases.yaml"
)

MatchSource = Literal[
    "alias_exact",
    "alias_substring",
    "alias_fuzzy",
    "user_override",
    "llm",
    "unmapped",
]


@dataclass(frozen=True)
class ColumnMatch:
    excel_column: str
    heat_record_field: str | None
    confidence: float
    source: MatchSource
    sample_values: list[str] = field(default_factory=list)
    reasoning: str | None = None


@dataclass(frozen=True)
class RowError:
    row_idx: int
    reason: str
    raw_row: dict


@dataclass
class ImportReport:
    file: str
    plant_id: str
    detected_cols: int
    mapped_cols: int
    skipped_cols: int
    rows_total: int
    rows_inserted: int
    rows_failed: int
    matches: list[ColumnMatch]
    errors: list[RowError]
    llm_used: bool
    dry_run: bool
    inserted_ids: list[int] = field(default_factory=list)


# ---------- aliases loading ----------

@lru_cache(maxsize=4)
def _load_aliases_cached(path_str: str) -> dict[str, list[str]]:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Aliases YAML not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("aliases", {})
    if not raw:
        raise ValueError(f"YAML at {p} has no 'aliases' section")
    # Pre-normalize alias strings so matching is symmetric with normalized columns.
    return {
        field_name: [_normalize_column_name(a) for a in alias_list]
        for field_name, alias_list in raw.items()
    }


def load_aliases(path: Path | None = None) -> dict[str, list[str]]:
    """Load alias map (with cache). Pass explicit path in tests; None → default YAML."""
    target = (path or DEFAULT_ALIASES_PATH).resolve()
    return _load_aliases_cached(str(target))


# ---------- normalization ----------

# Lightweight cyrillic transliteration (lowercase only — _normalize_column_name
# already lowercased input). Used to widen alias candidates without requiring
# external deps like Unidecode.
_CYRILLIC_TO_LATIN = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
        "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
        "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
        "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
        "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "iu", "я": "ia",
    }
)


def _normalize_column_name(name: str) -> str:
    """Lowercase + drop bracketed units + collapse non-alnum to underscore.

    Example: 'O_активность, ppm (start)' → 'o_aktivnost_ppm_start' (after translit).
    Keeps cyrillic letters at this stage; transliteration is done later when
    building candidates for matching.
    """
    s = str(name).lower().strip()
    # Drop bracketed annotations like (ppm), [°C], (start)
    s = re.sub(r"[\(\[][^\)\]]*[\)\]]", "", s)
    # Replace anything that is not lowercase ascii/cyrillic letter/digit with _
    s = re.sub(r"[^a-zа-яё0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _translit(s: str) -> str:
    return s.translate(_CYRILLIC_TO_LATIN)


# ---------- alias matching ----------

def _match_one_column(
    excel_col: str,
    normalized: str,
    aliases: dict[str, list[str]],
    user_overrides: dict[str, str] | None,
) -> tuple[str | None, float, MatchSource, str | None]:
    """Return (field_name, confidence, source, reasoning) for one excel column."""
    # User override wins unconditionally
    if user_overrides and excel_col in user_overrides:
        return (
            user_overrides[excel_col],
            1.0,
            "user_override",
            "user-provided mapping file",
        )

    translit = _translit(normalized)
    candidates = {normalized, translit}

    # Exact match against any alias
    for field_name, alias_list in aliases.items():
        for alias in alias_list:
            if alias in candidates:
                return (field_name, 1.0, "alias_exact", None)

    # Substring match — symmetric (alias contains candidate, or vice versa),
    # but require shorter side ≥4 chars to avoid false hits from very short
    # element aliases like 'c', 'mn', 'si' embedded in unrelated column names
    # ("alimnium_added" would otherwise match 'mn').
    _MIN_SUBSTR_LEN = 4
    for field_name, alias_list in aliases.items():
        for alias in alias_list:
            if not alias or len(alias) < _MIN_SUBSTR_LEN:
                continue
            for cand in candidates:
                if not cand or len(cand) < _MIN_SUBSTR_LEN:
                    continue
                if alias in cand or cand in alias:
                    return (field_name, 0.7, "alias_substring", None)

    # Fuzzy (difflib SequenceMatcher)
    best_field: str | None = None
    best_ratio: float = 0.0
    for field_name, alias_list in aliases.items():
        for alias in alias_list:
            for cand in candidates:
                if not alias or not cand:
                    continue
                ratio = difflib.SequenceMatcher(None, alias, cand).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_field = field_name
    if best_ratio >= 0.75 and best_field is not None:
        return (
            best_field,
            0.4,
            "alias_fuzzy",
            f"difflib ratio={best_ratio:.2f}",
        )

    return (None, 0.0, "unmapped", None)


def detect_columns(
    df: pd.DataFrame,
    aliases: dict[str, list[str]] | None = None,
    *,
    user_overrides: dict[str, str] | None = None,
) -> list[ColumnMatch]:
    """Run deterministic detector for every column in df. Order matches df.columns."""
    aliases = aliases or load_aliases()
    out: list[ColumnMatch] = []
    for col in df.columns:
        excel_col = str(col)
        normalized = _normalize_column_name(excel_col)
        field_name, conf, src, reasoning = _match_one_column(
            excel_col, normalized, aliases, user_overrides
        )
        samples = [str(v) for v in df[col].dropna().head(5).tolist()]
        out.append(
            ColumnMatch(
                excel_column=excel_col,
                heat_record_field=field_name,
                confidence=conf,
                source=src,
                sample_values=samples,
                reasoning=reasoning,
            )
        )
    return out


# ---------- LLM critic application ----------

_LLM_CONFIDENCE_MAP = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}


def apply_llm_critic(
    matches: list[ColumnMatch],
    critic: Any | None,
    *,
    confidence_threshold: float = 0.5,
) -> list[ColumnMatch]:
    """If critic provided, send ambiguous columns + samples for re-mapping.

    `critic` must expose `propose_mappings(unmapped: list[dict]) -> EtlMappingProposal|None`.
    On failure (None / exception) returns original matches unchanged.
    """
    if critic is None:
        return matches

    ambiguous_indices = [
        i
        for i, m in enumerate(matches)
        if m.heat_record_field is None or m.confidence < confidence_threshold
    ]
    if not ambiguous_indices:
        return matches

    payload = [
        {
            "excel_column": matches[i].excel_column,
            "sample_values": matches[i].sample_values,
        }
        for i in ambiguous_indices
    ]
    try:
        proposal = critic.propose_mappings(payload)
    except Exception as exc:  # noqa: BLE001 — best-effort fallback
        logger.warning("LLM critic failed: %s — keeping detector results", exc)
        return matches
    if proposal is None:
        return matches

    by_col = {m.excel_column: m for m in matches}
    for prop in proposal.mappings:
        if prop.excel_column in by_col:
            by_col[prop.excel_column] = ColumnMatch(
                excel_column=prop.excel_column,
                heat_record_field=prop.heat_record_field,
                confidence=_LLM_CONFIDENCE_MAP.get(prop.confidence, 0.3),
                source="llm",
                sample_values=by_col[prop.excel_column].sample_values,
                reasoning=prop.reasoning,
            )
    for unmapped in proposal.unmapped:
        if unmapped.excel_column in by_col:
            by_col[unmapped.excel_column] = ColumnMatch(
                excel_column=unmapped.excel_column,
                heat_record_field=None,
                confidence=0.0,
                source="unmapped",
                sample_values=by_col[unmapped.excel_column].sample_values,
                reasoning=unmapped.reason,
            )
    return [by_col[m.excel_column] for m in matches]


# ---------- row → HeatRecord ----------

def _clean_value(v: Any) -> Any:
    """Coerce numpy / pandas scalars → native Python primitives for pydantic."""
    if isinstance(v, (int, float, str, bool)):
        return v
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, AttributeError):
            return str(v)
    return str(v)


def build_records(
    df: pd.DataFrame,
    matches: list[ColumnMatch],
    *,
    plant_id: str,
    source: Source = "excel_etl",
    min_confidence: float = 0.5,
) -> tuple[list[HeatRecord], list[RowError]]:
    """Convert df rows → HeatRecord list using only matches with confidence ≥ threshold.

    Each row failing pydantic validation is reported in errors (skipped, not raised).
    `plant_id` always overrides any detected plant_id column to keep ingest atomic per plant.
    """
    mapping = {
        m.excel_column: m.heat_record_field
        for m in matches
        if m.heat_record_field and m.confidence >= min_confidence
    }
    records: list[HeatRecord] = []
    errors: list[RowError] = []

    for idx, row in df.iterrows():
        raw: dict[str, Any] = {}
        for excel_col, field_name in mapping.items():
            val = row[excel_col]
            if pd.isna(val):
                continue
            raw[field_name] = _clean_value(val)
        raw["source"] = source
        raw["plant_id"] = plant_id  # canonical plant override

        try:
            records.append(HeatRecord(**raw))
        except ValidationError as e:
            errors.append(
                RowError(row_idx=int(idx), reason=str(e), raw_row=raw)
            )

    return records, errors


# ---------- orchestrator ----------

def import_excel(
    path: Path,
    plant_id: str,
    *,
    sheet: str | int = 0,
    dry_run: bool = False,
    use_llm_critic: bool = False,
    mapping_file: Path | None = None,
    source: Source = "excel_etl",
    min_confidence: float = 0.5,
    aliases_path: Path | None = None,
) -> ImportReport:
    """Orchestrate Excel → HeatRecord → bulk insert.

    Returns ImportReport with column-match audit trail and per-row errors.
    If `dry_run=True`, validates everything but skips DB write.
    On LLM failure, silently degrades to detector-only mapping.
    """
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    aliases = load_aliases(aliases_path) if aliases_path else load_aliases()

    user_overrides: dict[str, str] | None = None
    if mapping_file:
        with open(mapping_file, encoding="utf-8") as f:
            user_overrides = yaml.safe_load(f) or {}
        if not isinstance(user_overrides, dict):
            raise ValueError(f"Mapping file must be a YAML dict: {mapping_file}")

    matches = detect_columns(
        df, aliases=aliases, user_overrides=user_overrides
    )

    llm_used = False
    if use_llm_critic:
        # Lazy import — heat_etl_critic pulls anthropic + prompt loader
        from app.backend.heat_etl_critic import make_heat_etl_critic

        critic = make_heat_etl_critic()
        if critic is not None:
            matches = apply_llm_critic(
                matches, critic, confidence_threshold=min_confidence
            )
            llm_used = True
        else:
            logger.info(
                "use_llm_critic=True but critic unavailable (no API key / no prompt); "
                "proceeding with detector results"
            )

    records, errors = build_records(
        df,
        matches,
        plant_id=plant_id,
        source=source,
        min_confidence=min_confidence,
    )

    inserted_ids: list[int] = []
    if not dry_run and records:
        inserted_ids = bulk_insert_heats(records)
        try:
            from decision_log.logger import log_decision

            log_decision(
                phase="data_acquisition",
                decision=(
                    f"Excel ETL imported {len(inserted_ids)} heats "
                    f"from plant {plant_id}"
                ),
                reasoning=(
                    f"File={path.name}, detected={len(matches)}, "
                    f"mapped={sum(1 for m in matches if m.heat_record_field)}, "
                    f"errors={len(errors)}, llm_used={llm_used}"
                ),
                context={
                    "file": str(path),
                    "plant_id": plant_id,
                    "llm_used": llm_used,
                    "source": source,
                },
                author="heat_history_etl",
                tags=["heat_etl", "excel_import"],
            )
        except Exception as exc:  # noqa: BLE001 — Decision Log must never block ingest
            logger.warning("Decision Log save failed: %s", exc)

    mapped = sum(
        1
        for m in matches
        if m.heat_record_field and m.confidence >= min_confidence
    )
    return ImportReport(
        file=str(path),
        plant_id=plant_id,
        detected_cols=len(matches),
        mapped_cols=mapped,
        skipped_cols=len(matches) - mapped,
        rows_total=len(df),
        rows_inserted=len(inserted_ids),
        rows_failed=len(errors),
        matches=matches,
        errors=errors,
        llm_used=llm_used,
        dry_run=dry_run,
        inserted_ids=inserted_ids,
    )


__all__ = [
    "ColumnMatch",
    "RowError",
    "ImportReport",
    "DEFAULT_ALIASES_PATH",
    "load_aliases",
    "detect_columns",
    "apply_llm_critic",
    "build_records",
    "import_excel",
]
