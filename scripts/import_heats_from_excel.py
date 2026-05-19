"""CLI: импорт Excel-журнала плавок в data/heats/heats.db.

Usage:
  python scripts/import_heats_from_excel.py --file heats.xlsx --plant ASIS_BOF
  python scripts/import_heats_from_excel.py --file heats.xlsx --plant ASIS_BOF --dry-run
  python scripts/import_heats_from_excel.py --file heats.xlsx --plant ASIS_BOF --use-llm-critic
  python scripts/import_heats_from_excel.py --file heats.xlsx --plant ASIS_BOF \
      --mapping-file overrides.yaml --log-file report.json

R-003 PR 3 — bulk ingest historical heats for ASIS deox calibration.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from app.backend.heat_history_etl import import_excel


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bulk import heats from Excel into data/heats/heats.db",
    )
    ap.add_argument("--file", required=True, type=Path, help="Path to .xlsx file")
    ap.add_argument("--plant", required=True, help="Plant identifier (overrides any plant_id column)")
    ap.add_argument("--sheet", default="0", help="Sheet name or index (default=0)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect + validate but skip DB write",
    )
    ap.add_argument(
        "--use-llm-critic",
        action="store_true",
        help="Use Sonnet for ambiguous columns (requires ANTHROPIC_API_KEY)",
    )
    ap.add_argument(
        "--mapping-file",
        type=Path,
        default=None,
        help="YAML with manual {excel_col: heat_record_field} overrides",
    )
    ap.add_argument(
        "--source",
        choices=["excel_etl", "csv_bulk", "manual", "synthetic"],
        default="excel_etl",
        help="Source tag stored in DB (default: excel_etl)",
    )
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum match confidence to accept a column (default: 0.5)",
    )
    ap.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Write structured JSON report to file (full match audit trail)",
    )
    args = ap.parse_args()

    if not args.file.exists():
        print(f"ERROR: file not found: {args.file}", file=sys.stderr)
        return 2

    sheet: str | int = (
        int(args.sheet) if args.sheet.isdigit() else args.sheet
    )

    print(
        f"Reading {args.file} (plant={args.plant}, sheet={sheet}, "
        f"dry_run={args.dry_run}, llm={args.use_llm_critic})..."
    )

    try:
        report = import_excel(
            path=args.file,
            plant_id=args.plant,
            sheet=sheet,
            dry_run=args.dry_run,
            use_llm_critic=args.use_llm_critic,
            mapping_file=args.mapping_file,
            source=args.source,
            min_confidence=args.min_confidence,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(
        f"\nDetected: {report.detected_cols} columns → "
        f"mapped: {report.mapped_cols} ({report.skipped_cols} skipped)"
    )
    print(
        f"Rows: {report.rows_total} total → "
        f"inserted: {report.rows_inserted} ({report.rows_failed} failed)"
    )
    print(f"LLM critic used: {report.llm_used}")
    if report.dry_run:
        print("\n*** DRY RUN — no DB writes ***")

    if report.errors:
        print(f"\nFirst {min(5, len(report.errors))} row errors:")
        for e in report.errors[:5]:
            reason = e.reason.replace("\n", " ")[:200]
            print(f"  row {e.row_idx}: {reason}")

    if report.matches:
        shown = min(20, len(report.matches))
        print(f"\nColumn matches (showing {shown}/{len(report.matches)}):")
        for m in report.matches[:20]:
            field_str = m.heat_record_field or "(unmapped)"
            print(
                f"  '{m.excel_column}' → {field_str} "
                f"[conf={m.confidence:.2f}, src={m.source}]"
            )

    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_file, "w", encoding="utf-8") as f:
            json.dump(
                asdict(report), f, ensure_ascii=False, indent=2, default=str
            )
        print(f"\nFull report → {args.log_file}")

    # Exit code: 0 if anything inserted or dry-run; 1 if zero rows ingested in non-dry-run
    if args.dry_run:
        return 0
    return 0 if report.rows_inserted > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
