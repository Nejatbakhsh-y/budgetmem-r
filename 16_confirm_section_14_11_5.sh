#!/usr/bin/env bash
set -Eeuo pipefail

# Section 14.11.5: strict same-cell comparison of BudgetMem-R against
# GRU + uniform cache and GRU + reservoir cache.
#
# Run from the repository root:
#   chmod +x 16_confirm_section_14_11_5.sh
#   ./16_confirm_section_14_11_5.sh
#
# Optional overrides:
#   SOURCE_CSV=path/to/results.csv ./16_confirm_section_14_11_5.sh
#   CLEAR_MARGIN=0.01 ./16_confirm_section_14_11_5.sh

cd "$(dirname "${BASH_SOURCE[0]}")"

SOURCE_CSV="${SOURCE_CSV:-reports/tables/pilot_results.csv}"
LONG_RANGE_MIN="${LONG_RANGE_MIN:-1024}"
CLEAR_MARGIN="${CLEAR_MARGIN:-auto}"
REQUIRED_PASS_CELLS="${REQUIRED_PASS_CELLS:-1}"

TABLE_OUT="reports/tables/section14_11_5_matched_comparison.csv"
TEXT_OUT="reports/evidence/section14_11_5_confirmation.txt"
JSON_OUT="reports/evidence/section14_11_5_confirmation.json"

mkdir -p reports/tables reports/evidence

if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
elif [[ -x "venv/bin/python" ]]; then
    PYTHON="venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    echo "ERROR: Python was not found." >&2
    exit 2
fi

export SOURCE_CSV LONG_RANGE_MIN CLEAR_MARGIN REQUIRED_PASS_CELLS
export TABLE_OUT TEXT_OUT JSON_OUT

"$PYTHON" <<'PYTHON'
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

source = Path(os.environ["SOURCE_CSV"])
table_out = Path(os.environ["TABLE_OUT"])
text_out = Path(os.environ["TEXT_OUT"])
json_out = Path(os.environ["JSON_OUT"])
long_range_min = int(os.environ["LONG_RANGE_MIN"])
required_pass_cells = int(os.environ["REQUIRED_PASS_CELLS"])
margin_setting = os.environ["CLEAR_MARGIN"].strip().lower()


def fail(message):
    payload = {
        "section": "14.11.5",
        "decision": "FAIL",
        "reason": message,
        "source_csv": str(source),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    text_out.write_text(
        "Section 14.11.5 Same-Cell Policy Confirmation\n"
        "================================================\n\n"
        "OVERALL DECISION: FAIL\n\n"
        f"Reason: {message}\n",
        encoding="utf-8",
    )
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n============================================================")
    print("SECTION 14.11.5: FAIL")
    print("============================================================")
    print(message)
    sys.exit(2)


if not source.is_file():
    fail(f"Required source file was not found: {source}")

source_hash = hashlib.sha256(source.read_bytes()).hexdigest()


def normalized(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def text_key(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


def parse_integer(value):
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
        if number.is_integer():
            return int(number)
    except ValueError:
        pass
    match = re.search(r"(?<!\d)(\d+)(?!\d)", text)
    return int(match.group(1)) if match else None


def parse_seed(value):
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def parse_score(value):
    text = str(value).strip().replace(",", "")
    if text.endswith("%"):
        text = text[:-1].strip()
    if not text:
        return None
    try:
        score = float(text)
    except ValueError:
        return None
    return score if math.isfinite(score) else None


def model_role(value):
    name = normalized(value)
    if "budgetmem" in name:
        return "budgetmem_r"
    if "uniform" in name:
        return "gru_uniform_cache"
    if "reservoir" in name:
        return "gru_reservoir_cache"
    return None


with source.open("r", encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    fields = reader.fieldnames or []

if not rows:
    fail("The source CSV is empty.")

field_lookup = {normalized(field): field for field in fields}


def find_column(aliases, required=True):
    for alias in aliases:
        key = normalized(alias)
        if key in field_lookup:
            return field_lookup[key]
    if required:
        fail(
            "Required column not found. Expected one of: "
            + ", ".join(aliases)
            + ". Available columns: "
            + ", ".join(fields)
        )
    return None


task_col = find_column(["task", "task_name", "benchmark", "dataset"])
sequence_col = find_column(
    ["sequence_length", "seq_length", "seq_len", "length", "context_length"]
)
seed_col = find_column(["seed", "random_seed", "run_seed"])
budget_col = find_column(
    ["memory_budget", "budget", "mem_budget", "requested_budget"]
)
model_col = find_column(["model", "model_name", "method", "policy"])

requested_metric_column = os.environ.get("METRIC_COLUMN", "").strip()
if requested_metric_column:
    score_col = find_column([requested_metric_column])
else:
    score_col = find_column(
        [
            "token_accuracy",
            "exact_match_accuracy",
            "long_range_recall",
            "recall",
            "test_recall",
            "accuracy",
            "test_accuracy",
            "validation_accuracy",
            "val_accuracy",
            "exact_match",
            "primary_score",
            "score",
            "memory_recall",
            "successful_long_range_retrievals",
            "metric_value",
            "value",
        ]
    )

metric_name_col = find_column(
    ["metric_name", "evaluation_metric", "measure_name", "metric"],
    required=False,
)
selected_metric_name = None

if normalized(score_col) in {"metricvalue", "value"} and metric_name_col:
    available_metrics = sorted(
        {
            str(row.get(metric_name_col, "")).strip()
            for row in rows
            if model_role(row.get(model_col, ""))
            and str(row.get(metric_name_col, "")).strip()
        }
    )
    requested_metric_name = os.environ.get("METRIC_NAME", "").strip()
    if requested_metric_name:
        matches = [
            item
            for item in available_metrics
            if normalized(item) == normalized(requested_metric_name)
        ]
        if not matches:
            fail(
                f"METRIC_NAME={requested_metric_name!r} was not found. "
                f"Available metrics: {available_metrics}"
            )
        selected_metric_name = matches[0]
    else:
        priorities = [
            "longrangerecall",
            "recall",
            "testrecall",
            "accuracy",
            "testaccuracy",
            "exactmatch",
            "f1",
            "score",
        ]
        for priority in priorities:
            matches = [
                item for item in available_metrics if priority in normalized(item)
            ]
            if matches:
                selected_metric_name = sorted(matches)[0]
                break
        if selected_metric_name is None:
            if len(available_metrics) == 1:
                selected_metric_name = available_metrics[0]
            else:
                fail(
                    "Metric is ambiguous. Set METRIC_NAME explicitly. "
                    f"Available metrics: {available_metrics}"
                )

metric_label = selected_metric_name or score_col
direction_setting = os.environ.get("METRIC_DIRECTION", "").strip().lower()
if direction_setting:
    if direction_setting not in {"higher", "lower"}:
        fail("METRIC_DIRECTION must be either 'higher' or 'lower'.")
    direction = direction_setting
else:
    lower_terms = {"loss", "error", "rmse", "mae", "mse", "perplexity"}
    direction = (
        "lower"
        if any(term in normalized(metric_label) for term in lower_terms)
        else "higher"
    )

if margin_setting != "auto":
    try:
        manual_margin = float(margin_setting)
    except ValueError:
        fail("CLEAR_MARGIN must be 'auto' or a positive number.")
    if not math.isfinite(manual_margin) or manual_margin <= 0:
        fail("CLEAR_MARGIN must be greater than zero.")
else:
    manual_margin = None

cells = defaultdict(lambda: defaultdict(list))
display_tasks = {}
invalid_rows = []
target_rows = 0

for row_number, row in enumerate(rows, start=2):
    role = model_role(row.get(model_col, ""))
    if role is None:
        continue
    if selected_metric_name is not None:
        row_metric = str(row.get(metric_name_col, "")).strip()
        if normalized(row_metric) != normalized(selected_metric_name):
            continue

    target_rows += 1
    task = str(row.get(task_col, "")).strip()
    sequence_length = parse_integer(row.get(sequence_col, ""))
    seed = parse_seed(row.get(seed_col, ""))
    budget = parse_integer(row.get(budget_col, ""))
    score = parse_score(row.get(score_col, ""))
    problems = []
    if not task:
        problems.append("missing task")
    if sequence_length is None:
        problems.append("missing or invalid sequence length")
    if seed is None:
        problems.append("missing seed")
    if budget not in {16, 32}:
        problems.append("budget must be explicitly recorded as 16 or 32")
    if score is None:
        problems.append("missing, nonnumeric, or nonfinite score")

    if problems:
        invalid_rows.append(
            {
                "row": row_number,
                "model": row.get(model_col, ""),
                "problems": "; ".join(problems),
            }
        )
        continue
    if sequence_length < long_range_min:
        continue

    task_id = text_key(task)
    display_tasks.setdefault(task_id, task)
    key = (task_id, sequence_length, seed, budget)
    cells[key][role].append(score)

if target_rows == 0:
    fail("No BudgetMem-R, uniform-cache, or reservoir-cache rows were found.")
if not cells:
    fail(
        f"No valid comparison rows were found at sequence length "
        f"{long_range_min} or greater."
    )

comparison_rows = []
ambiguous_cells = 0
incomplete_cells = 0
qualifying_cells = 0
required_roles = [
    "budgetmem_r",
    "gru_uniform_cache",
    "gru_reservoir_cache",
]

for key in sorted(cells):
    task_id, sequence_length, seed, budget = key
    role_values = cells[key]
    resolved = {}
    ambiguous_roles = []
    missing_roles = []

    for role in required_roles:
        values = role_values.get(role, [])
        unique_values = []
        for value in values:
            if not any(
                math.isclose(value, prior, abs_tol=1e-12)
                for prior in unique_values
            ):
                unique_values.append(value)
        if not unique_values:
            missing_roles.append(role)
        elif len(unique_values) > 1:
            ambiguous_roles.append(role)
        else:
            resolved[role] = unique_values[0]

    if ambiguous_roles:
        ambiguous_cells += 1
        status = "AMBIGUOUS_DUPLICATES"
        threshold = ""
        uniform_margin = ""
        reservoir_margin = ""
        uniform_clear = False
        reservoir_clear = False
        policies_beaten = 0
    elif missing_roles:
        incomplete_cells += 1
        status = "INCOMPLETE"
        threshold = ""
        uniform_margin = ""
        reservoir_margin = ""
        uniform_clear = False
        reservoir_clear = False
        policies_beaten = 0
    else:
        scores = list(resolved.values())
        threshold = (
            manual_margin
            if manual_margin is not None
            else (0.02 if max(abs(value) for value in scores) <= 1.5 else 2.0)
        )
        budgetmem_score = resolved["budgetmem_r"]
        uniform_score = resolved["gru_uniform_cache"]
        reservoir_score = resolved["gru_reservoir_cache"]
        if direction == "higher":
            uniform_margin = budgetmem_score - uniform_score
            reservoir_margin = budgetmem_score - reservoir_score
        else:
            uniform_margin = uniform_score - budgetmem_score
            reservoir_margin = reservoir_score - budgetmem_score
        uniform_clear = uniform_margin + 1e-12 >= threshold
        reservoir_clear = reservoir_margin + 1e-12 >= threshold
        policies_beaten = int(uniform_clear) + int(reservoir_clear)
        if policies_beaten >= 2:
            status = "MATCHED_PASS"
            qualifying_cells += 1
        else:
            status = "MATCHED_NOT_CLEAR"

    comparison_rows.append(
        {
            "task": display_tasks.get(task_id, task_id),
            "sequence_length": sequence_length,
            "seed": seed,
            "memory_budget": budget,
            "metric": metric_label,
            "direction": direction,
            "clear_margin_required": threshold,
            "budgetmem_r_score": resolved.get("budgetmem_r", ""),
            "gru_uniform_cache_score": resolved.get("gru_uniform_cache", ""),
            "uniform_margin": uniform_margin,
            "uniform_clearly_beaten": uniform_clear,
            "gru_reservoir_cache_score": resolved.get(
                "gru_reservoir_cache", ""
            ),
            "reservoir_margin": reservoir_margin,
            "reservoir_clearly_beaten": reservoir_clear,
            "policies_clearly_beaten": policies_beaten,
            "missing_roles": ";".join(missing_roles),
            "ambiguous_roles": ";".join(ambiguous_roles),
            "status": status,
        }
    )

fieldnames = [
    "task",
    "sequence_length",
    "seed",
    "memory_budget",
    "metric",
    "direction",
    "clear_margin_required",
    "budgetmem_r_score",
    "gru_uniform_cache_score",
    "uniform_margin",
    "uniform_clearly_beaten",
    "gru_reservoir_cache_score",
    "reservoir_margin",
    "reservoir_clearly_beaten",
    "policies_clearly_beaten",
    "missing_roles",
    "ambiguous_roles",
    "status",
]

with table_out.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(comparison_rows)

complete_matched_cells = sum(
    row["status"] in {"MATCHED_PASS", "MATCHED_NOT_CLEAR"}
    for row in comparison_rows
)
overall_pass = (
    qualifying_cells >= required_pass_cells
    and incomplete_cells == 0
    and ambiguous_cells == 0
    and len(invalid_rows) == 0
)
decision = "PASS" if overall_pass else "FAIL"

payload = {
    "section": "14.11.5",
    "decision": decision,
    "source_csv": str(source),
    "source_sha256": source_hash,
    "metric": metric_label,
    "metric_direction": direction,
    "long_range_minimum": long_range_min,
    "clear_margin_setting": margin_setting,
    "required_qualifying_cells": required_pass_cells,
    "qualifying_cells": qualifying_cells,
    "complete_matched_cells": complete_matched_cells,
    "incomplete_cells": incomplete_cells,
    "ambiguous_cells": ambiguous_cells,
    "invalid_rows": invalid_rows,
    "generated_utc": datetime.now(timezone.utc).isoformat(),
}
json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = [
    "Section 14.11.5 Same-Cell Policy Confirmation",
    "================================================",
    "",
    f"OVERALL DECISION: {decision}",
    "",
    f"Source CSV: {source}",
    f"Source SHA-256: {source_hash}",
    f"Metric: {metric_label}",
    f"Direction: {direction} is better",
    f"Long-range minimum sequence length: {long_range_min}",
    f"Required qualifying matched cells: {required_pass_cells}",
    f"Qualifying matched cells: {qualifying_cells}",
    f"Complete matched cells: {complete_matched_cells}",
    f"Incomplete cells: {incomplete_cells}",
    f"Ambiguous cells: {ambiguous_cells}",
    f"Invalid source rows: {len(invalid_rows)}",
    "",
    "Matched-cell key:",
    "task + sequence length + seed + memory budget",
    "",
    "A cell passes only when BudgetMem-R clearly beats both:",
    "1. GRU + uniform cache",
    "2. GRU + reservoir cache",
    "",
    "Clear-margin rule:",
    "0.02 for a 0-1 scale or 2.0 for a 0-100 scale, unless",
    "CLEAR_MARGIN is supplied explicitly.",
    "",
    "This is an operational single-seed comparison and is not evidence",
    "of statistical significance across multiple random seeds.",
    "",
    "Cell results:",
]

for row in comparison_rows:
    lines.append(
        f"- {row['task']} | length={row['sequence_length']} | "
        f"seed={row['seed']} | budget={row['memory_budget']} | "
        f"status={row['status']} | policies clearly beaten="
        f"{row['policies_clearly_beaten']}"
    )

if invalid_rows:
    lines.extend(["", "Invalid rows:"])
    for item in invalid_rows:
        lines.append(
            f"- CSV row {item['row']}: {item['model']} - {item['problems']}"
        )

if overall_pass:
    lines.extend(
        [
            "",
            "CONFIRMATION:",
            "BudgetMem-R clearly outperforms both comparison policies within",
            "at least one identical long-range task/length/seed/budget cell.",
        ]
    )
else:
    lines.extend(
        [
            "",
            "CONFIRMATION NOT ESTABLISHED:",
            "Review the comparison CSV for missing, ambiguous, or",
            "insufficient-margin cells.",
        ]
    )

text_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

print()
print("============================================================")
print(f"SECTION 14.11.5: {decision}")
print("============================================================")
print(f"Qualifying matched cells : {qualifying_cells}")
print(f"Complete matched cells   : {complete_matched_cells}")
print(f"Incomplete cells         : {incomplete_cells}")
print(f"Ambiguous cells          : {ambiguous_cells}")
print(f"Invalid rows             : {len(invalid_rows)}")
print()
print(f"Evidence: {text_out}")
print(f"Table:    {table_out}")
print(f"JSON:     {json_out}")

sys.exit(0 if overall_pass else 2)
PYTHON
