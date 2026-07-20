#!/usr/bin/env bash
set -Eeuo pipefail

# Section 14.11.7 — inspect all matched-cell values and identify exact deficits.
# Run from the budgetmem-r repository root in the VS Code WSL terminal.

ROOT="$(pwd)"
CSV="$ROOT/reports/tables/section14_11_5_matched_comparison.csv"
OUT_TXT="$ROOT/reports/evidence/section14_11_7_cell_analysis.txt"
OUT_JSON="$ROOT/reports/evidence/section14_11_7_cell_analysis.json"

if [[ ! -f "$CSV" ]]; then
    echo "ERROR: Missing $CSV"
    exit 2
fi

python - "$CSV" "$OUT_TXT" "$OUT_JSON" <<'PY'
from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

csv_path = Path(sys.argv[1])
out_txt = Path(sys.argv[2])
out_json = Path(sys.argv[3])

def norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")

def finite_float(v: Any) -> float | None:
    try:
        x = float(str(v).strip())
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
    reader = csv.DictReader(fh)
    rows = list(reader)
    columns = reader.fieldnames or []

if not rows:
    raise SystemExit(f"ERROR: {csv_path} contains no data rows.")

ncols = {norm(c): c for c in columns}

def find_col(*tokens: str) -> str | None:
    wanted = [norm(t) for t in tokens]
    for nc, original in ncols.items():
        if all(t in nc for t in wanted):
            return original
    return None

def find_any(candidates: list[str]) -> str | None:
    for candidate in candidates:
        c = find_col(candidate)
        if c:
            return c
    return None

# Common identifier columns.
task_col = find_any(["task", "dataset"])
seq_col = find_any(["sequence_length", "seq_len", "length"])
budget_col = find_any(["memory_budget", "budget"])
seed_col = find_any(["seed"])
model_col = find_any(["model", "policy"])
metric_col = find_any(["metric"])
value_col = find_any(["score", "value", "token_accuracy", "accuracy", "recall"])

# First, produce a complete, compact rendering of the actual CSV.
widths = {}
for c in columns:
    widths[c] = min(
        40,
        max(len(c), *(len(str(r.get(c, ""))) for r in rows))
    )

def table_line(values: list[str]) -> str:
    cells = []
    for c, v in zip(columns, values):
        text = str(v)
        if len(text) > widths[c]:
            text = text[: max(0, widths[c] - 1)] + "…"
        cells.append(text.ljust(widths[c]))
    return " | ".join(cells)

lines: list[str] = []
lines.append("=" * 100)
lines.append("SECTION 14.11.7 — EXACT MATCHED-CELL ANALYSIS")
lines.append("=" * 100)
lines.append(f"Source: {csv_path}")
lines.append(f"Rows: {len(rows)}")
lines.append(f"Columns: {len(columns)}")
lines.append("")
lines.append(table_line(columns))
lines.append("-" * len(table_line(columns)))
for r in rows:
    lines.append(table_line([str(r.get(c, "")) for c in columns]))
lines.append("")

analysis: dict[str, Any] = {
    "source_csv": str(csv_path),
    "row_count": len(rows),
    "columns": columns,
    "mode": "unknown",
    "cells": [],
    "qualifying_cells": 0,
    "required_qualifying_cells": 1,
}

# Detect long format: one row per model.
long_format = bool(model_col and value_col and task_col and seq_col and budget_col)

if long_format:
    analysis["mode"] = "long"
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        key = (
            str(r.get(task_col, "")),
            str(r.get(seq_col, "")),
            str(r.get(budget_col, "")),
            str(r.get(seed_col, "")) if seed_col else "",
        )
        groups[key].append(r)

    lines.append("Computed comparison by matched cell")
    lines.append("-" * 100)

    for key, group in sorted(groups.items()):
        models: dict[str, float] = {}
        for r in group:
            score = finite_float(r.get(value_col))
            if score is not None:
                models[norm(r.get(model_col, ""))] = score

        bm_key = next((m for m in models if "budgetmem" in m), None)
        uniform_key = next((m for m in models if "uniform" in m), None)
        reservoir_key = next((m for m in models if "reservoir" in m), None)

        bm = models.get(bm_key) if bm_key else None
        uniform = models.get(uniform_key) if uniform_key else None
        reservoir = models.get(reservoir_key) if reservoir_key else None

        wins = 0
        if bm is not None and uniform is not None and bm > uniform:
            wins += 1
        if bm is not None and reservoir is not None and bm > reservoir:
            wins += 1

        qualifies = wins >= 2
        analysis["qualifying_cells"] += int(qualifies)

        cell = {
            "task": key[0],
            "sequence_length": key[1],
            "budget": key[2],
            "seed": key[3],
            "budgetmem_score": bm,
            "uniform_score": uniform,
            "reservoir_score": reservoir,
            "budgetmem_minus_uniform": None if bm is None or uniform is None else bm - uniform,
            "budgetmem_minus_reservoir": None if bm is None or reservoir is None else bm - reservoir,
            "wins_against_required_policies": wins,
            "qualifies": qualifies,
        }
        analysis["cells"].append(cell)

        lines.append(
            f"task={key[0]} seq={key[1]} budget={key[2]} seed={key[3] or 'NA'}"
        )
        lines.append(
            f"  BudgetMem-R={bm!r}; uniform={uniform!r}; reservoir={reservoir!r}"
        )
        lines.append(
            f"  deltas: vs_uniform={cell['budgetmem_minus_uniform']!r}; "
            f"vs_reservoir={cell['budgetmem_minus_reservoir']!r}; "
            f"wins={wins}/2; qualifies={qualifies}"
        )
else:
    # Wide or already-computed comparison format.
    analysis["mode"] = "wide_or_precomputed"
    lines.append("Detected wide/precomputed comparison format")
    lines.append("-" * 100)

    def score_cols_for(token: str) -> list[str]:
        result = []
        for c in columns:
            nc = norm(c)
            if token in nc and any(x in nc for x in ("score", "accuracy", "recall", "metric", "value")):
                result.append(c)
        return result

    bm_cols = score_cols_for("budgetmem")
    uniform_cols = score_cols_for("uniform")
    reservoir_cols = score_cols_for("reservoir")

    numeric_cols = []
    for c in columns:
        vals = [finite_float(r.get(c)) for r in rows if str(r.get(c, "")).strip()]
        if vals and all(v is not None for v in vals):
            numeric_cols.append(c)

    lines.append(f"BudgetMem-R score candidates: {bm_cols or 'none detected'}")
    lines.append(f"Uniform score candidates: {uniform_cols or 'none detected'}")
    lines.append(f"Reservoir score candidates: {reservoir_cols or 'none detected'}")
    lines.append(f"Numeric columns: {numeric_cols}")
    lines.append("")

    for i, r in enumerate(rows, start=1):
        bm_col = bm_cols[0] if bm_cols else None
        uniform_col = uniform_cols[0] if uniform_cols else None
        reservoir_col = reservoir_cols[0] if reservoir_cols else None

        bm = finite_float(r.get(bm_col)) if bm_col else None
        uniform = finite_float(r.get(uniform_col)) if uniform_col else None
        reservoir = finite_float(r.get(reservoir_col)) if reservoir_col else None

        wins = 0
        if bm is not None and uniform is not None and bm > uniform:
            wins += 1
        if bm is not None and reservoir is not None and bm > reservoir:
            wins += 1
        qualifies = wins >= 2 if bm is not None and uniform is not None and reservoir is not None else False
        analysis["qualifying_cells"] += int(qualifies)

        cell = {
            "row": i,
            "task": str(r.get(task_col, "")) if task_col else "",
            "sequence_length": str(r.get(seq_col, "")) if seq_col else "",
            "budget": str(r.get(budget_col, "")) if budget_col else "",
            "budgetmem_score_column": bm_col,
            "uniform_score_column": uniform_col,
            "reservoir_score_column": reservoir_col,
            "budgetmem_score": bm,
            "uniform_score": uniform,
            "reservoir_score": reservoir,
            "budgetmem_minus_uniform": None if bm is None or uniform is None else bm - uniform,
            "budgetmem_minus_reservoir": None if bm is None or reservoir is None else bm - reservoir,
            "wins_against_required_policies": wins,
            "qualifies": qualifies,
            "raw": r,
        }
        analysis["cells"].append(cell)

        lines.append(
            f"row={i} task={cell['task'] or 'NA'} "
            f"seq={cell['sequence_length'] or 'NA'} budget={cell['budget'] or 'NA'}"
        )
        if bm is not None or uniform is not None or reservoir is not None:
            lines.append(
                f"  BudgetMem-R={bm!r}; uniform={uniform!r}; reservoir={reservoir!r}"
            )
            lines.append(
                f"  deltas: vs_uniform={cell['budgetmem_minus_uniform']!r}; "
                f"vs_reservoir={cell['budgetmem_minus_reservoir']!r}; "
                f"wins={wins}/2; qualifies={qualifies}"
            )
        else:
            lines.append("  Automatic score-column mapping was not possible; use the full row printed above.")
        lines.append("")

lines.append("")
lines.append("FINAL DIAGNOSIS")
lines.append("-" * 100)
lines.append(f"Qualifying cells: {analysis['qualifying_cells']}")
lines.append(f"Required qualifying cells: {analysis['required_qualifying_cells']}")

if analysis["qualifying_cells"] >= analysis["required_qualifying_cells"]:
    lines.append(
        "The score values appear to satisfy the performance rule. The confirmation "
        "script likely has a comparator-mapping or qualification-logic defect."
    )
    analysis["diagnosis"] = "likely_confirmation_logic_defect"
else:
    mapped = [
        c for c in analysis["cells"]
        if c.get("budgetmem_score") is not None
        and c.get("uniform_score") is not None
        and c.get("reservoir_score") is not None
    ]
    if mapped:
        lines.append(
            "No matched cell beats both required cache policies. This is an actual "
            "performance shortfall under the strict higher-is-better rule."
        )
        analysis["diagnosis"] = "actual_performance_shortfall"
    else:
        lines.append(
            "The CSV was printed successfully, but its score columns could not be "
            "mapped automatically. Review the complete table at the top of this file."
        )
        analysis["diagnosis"] = "manual_column_mapping_required"

out_txt.parent.mkdir(parents=True, exist_ok=True)
out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
out_json.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print("\n".join(lines))
print("")
print(f"TEXT EVIDENCE: {out_txt}")
print(f"JSON EVIDENCE: {out_json}")
PY
