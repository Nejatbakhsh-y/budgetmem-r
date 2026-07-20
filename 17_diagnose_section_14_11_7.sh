#!/usr/bin/env bash
set -Eeuo pipefail

# Section 14.11.7 — diagnose why no matched cell qualifies.
# Run from the budgetmem-r repository root under WSL/VS Code.

ROOT="$(pwd)"
CSV="$ROOT/reports/tables/section14_11_5_matched_comparison.csv"
TXT="$ROOT/reports/evidence/section14_11_5_confirmation.txt"
JSON="$ROOT/reports/evidence/section14_11_5_confirmation.json"
OUT="$ROOT/reports/evidence/section14_11_7_failure_diagnostic.txt"

printf '\n============================================================\n'
printf 'SECTION 14.11.7 — FAILURE DIAGNOSTIC\n'
printf '============================================================\n'

missing=0
for f in "$CSV" "$TXT" "$JSON"; do
    if [[ ! -f "$f" ]]; then
        printf 'MISSING: %s\n' "$f"
        missing=1
    else
        printf 'FOUND:   %s\n' "$f"
    fi
done

if [[ "$missing" -ne 0 ]]; then
    printf '\nRequired evidence is missing. Stop here.\n'
    exit 2
fi

python - "$CSV" "$TXT" "$JSON" "$OUT" <<'PY'
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

csv_path = Path(sys.argv[1])
txt_path = Path(sys.argv[2])
json_path = Path(sys.argv[3])
out_path = Path(sys.argv[4])

def is_number(value: str) -> bool:
    try:
        x = float(value)
        return math.isfinite(x)
    except (TypeError, ValueError):
        return False

def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "pass", "passed", "qualifies", "qualified"}:
        return True
    if text in {"false", "0", "no", "n", "fail", "failed", "not_qualified"}:
        return False
    return None

with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
    reader = csv.DictReader(fh)
    rows = list(reader)
    columns = reader.fieldnames or []

try:
    confirmation = json.loads(json_path.read_text(encoding="utf-8"))
except json.JSONDecodeError as exc:
    confirmation = {"json_parse_error": str(exc), "raw": json_path.read_text(encoding="utf-8")}

text_evidence = txt_path.read_text(encoding="utf-8", errors="replace")

candidate_flag_columns = [
    c for c in columns
    if any(token in c.lower() for token in ("qualif", "pass", "outperform", "win"))
]

numeric_columns = [
    c for c in columns
    if rows and all((not str(r.get(c, "")).strip()) or is_number(str(r.get(c, ""))) for r in rows)
]

lines: list[str] = []
lines.append("=" * 72)
lines.append("SECTION 14.11.7 — MATCHED-CELL FAILURE DIAGNOSTIC")
lines.append("=" * 72)
lines.append("")
lines.append(f"CSV:  {csv_path}")
lines.append(f"TXT:  {txt_path}")
lines.append(f"JSON: {json_path}")
lines.append("")
lines.append(f"Matched rows found: {len(rows)}")
lines.append(f"Columns ({len(columns)}): {', '.join(columns)}")
lines.append("")

if candidate_flag_columns:
    lines.append("Detected decision/status columns:")
    for col in candidate_flag_columns:
        true_count = sum(as_bool(r.get(col)) is True for r in rows)
        false_count = sum(as_bool(r.get(col)) is False for r in rows)
        other_count = len(rows) - true_count - false_count
        lines.append(
            f"  - {col}: true/pass={true_count}, false/fail={false_count}, other={other_count}"
        )
    lines.append("")

lines.append("Per-cell records")
lines.append("-" * 72)
for idx, row in enumerate(rows, start=1):
    lines.append(f"[ROW {idx}]")
    for col in columns:
        value = str(row.get(col, "")).strip()
        if value:
            lines.append(f"{col}: {value}")
    lines.append("")

lines.append("Numeric-column ranges")
lines.append("-" * 72)
for col in numeric_columns:
    vals = [float(r[col]) for r in rows if str(r.get(col, "")).strip() and is_number(str(r[col]))]
    if vals:
        lines.append(
            f"{col}: min={min(vals):.10g}, max={max(vals):.10g}, mean={sum(vals)/len(vals):.10g}"
        )
lines.append("")

lines.append("Existing text confirmation")
lines.append("-" * 72)
lines.append(text_evidence.rstrip())
lines.append("")

lines.append("Existing JSON confirmation")
lines.append("-" * 72)
lines.append(json.dumps(confirmation, indent=2, sort_keys=True))
lines.append("")

lines.append("Interpretation")
lines.append("-" * 72)
lines.append(
    "The parser has already accepted all six matched cells. Therefore, the earlier "
    "unrecognized-metric-column problem is resolved. A result of zero qualifying "
    "cells must now be explained by the actual per-cell comparison values, the "
    "qualification threshold, or the comparator-selection logic."
)
lines.append(
    "Do not rerun the full pilot until the per-cell deficits in this report have "
    "been reviewed."
)
lines.append("")

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print(f"\nDIAGNOSTIC SAVED: {out_path}")
PY

printf '\n============================================================\n'
printf 'NEXT FILE TO REVIEW OR ATTACH:\n%s\n' "$OUT"
printf '============================================================\n'
