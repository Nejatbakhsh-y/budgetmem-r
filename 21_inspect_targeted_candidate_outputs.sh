#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
OUTPUT="reports/evidence/targeted_associative_recall_output_inspection.txt"

mkdir -p reports/evidence

"$PYTHON" - <<'PY' | tee \
    reports/evidence/targeted_associative_recall_output_inspection.txt

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path.cwd()
REPORTS = ROOT / "reports"

TARGET_TASK = "associative_recall"
TARGET_SEQUENCE = 1024
TARGET_BUDGET = 16

MODEL_ALIASES = [
    "model",
    "model_name",
    "method",
    "architecture",
]

TASK_ALIASES = [
    "task",
    "task_name",
    "dataset",
]

SEQUENCE_ALIASES = [
    "sequence_length",
    "seq_length",
    "seq_len",
    "evaluation_sequence_length",
    "context_length",
]

BUDGET_ALIASES = [
    "memory_budget",
    "budget",
    "cache_size",
    "memory_size",
]

RECALL_ALIASES = [
    "memory_recall",
    "long_range_recall",
    "long_range_retrieval_recall",
    "relevant_state_retention_rate",
    "retrieval_recall",
    "recall",
]


def normalize(value: object) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def find_column(
    columns: Iterable[str],
    aliases: Iterable[str],
) -> str | None:
    lookup = {
        normalize(column): column
        for column in columns
    }

    for alias in aliases:
        if alias in lookup:
            return lookup[alias]

    return None


def identify_model(value: object) -> str | None:
    text = normalize(value)

    if "budgetmem" in text or "budget_mem" in text:
        return "budgetmem_r"

    if "uniform" in text:
        return "uniform_cache"

    if "reservoir" in text:
        return "reservoir_cache"

    return None


def extract_number(series: pd.Series) -> pd.Series:
    converted = pd.to_numeric(
        series,
        errors="coerce",
    )

    missing = converted.isna()

    if missing.any():
        extracted = (
            series.astype(str)
            .str.extract(
                r"([-+]?[0-9]*\.?[0-9]+)",
                expand=False,
            )
        )

        converted.loc[missing] = pd.to_numeric(
            extracted.loc[missing],
            errors="coerce",
        )

    return converted


print("=" * 100)
print("TARGETED ASSOCIATIVE-RECALL OUTPUT INSPECTION")
print("=" * 100)
print()

if not REPORTS.exists():
    raise SystemExit("ERROR: reports/ does not exist.")


candidate_files = sorted(
    [
        path
        for path in REPORTS.rglob("*.csv")
        if (
            "targeted_associative_recall" in str(path)
            or "pilot_tuned" in path.name
            or "candidate" in path.name
        )
    ],
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)


print("CSV files inspected:")
print()

if not candidate_files:
    print("- No candidate CSV files were found.")
else:
    for path in candidate_files:
        relative = path.relative_to(ROOT)

        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"- {relative}: unreadable ({exc})")
            continue

        print(f"- {relative}")
        print(f"  Rows: {len(frame)}")
        print(f"  Columns: {list(frame.columns)}")

        task_column = find_column(
            frame.columns,
            TASK_ALIASES,
        )
        model_column = find_column(
            frame.columns,
            MODEL_ALIASES,
        )
        sequence_column = find_column(
            frame.columns,
            SEQUENCE_ALIASES,
        )
        budget_column = find_column(
            frame.columns,
            BUDGET_ALIASES,
        )
        recall_column = find_column(
            frame.columns,
            RECALL_ALIASES,
        )

        if task_column:
            values = sorted(
                frame[task_column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            print(f"  Tasks: {values}")

        if model_column:
            values = sorted(
                frame[model_column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            print(f"  Models: {values}")

        if sequence_column:
            values = sorted(
                extract_number(
                    frame[sequence_column]
                )
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            print(f"  Sequence lengths: {values}")

        if budget_column:
            values = sorted(
                extract_number(
                    frame[budget_column]
                )
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            print(f"  Budgets: {values}")

        print(f"  Recall column: {recall_column}")
        print()


compatible_tables: list[
    tuple[Path, pd.DataFrame]
] = []


for path in candidate_files:
    try:
        frame = pd.read_csv(path)
    except Exception:
        continue

    task_column = find_column(
        frame.columns,
        TASK_ALIASES,
    )
    model_column = find_column(
        frame.columns,
        MODEL_ALIASES,
    )
    sequence_column = find_column(
        frame.columns,
        SEQUENCE_ALIASES,
    )
    budget_column = find_column(
        frame.columns,
        BUDGET_ALIASES,
    )
    recall_column = find_column(
        frame.columns,
        RECALL_ALIASES,
    )

    if not all(
        [
            task_column,
            model_column,
            sequence_column,
            budget_column,
            recall_column,
        ]
    ):
        continue

    standardized = pd.DataFrame(
        {
            "task": frame[task_column].map(normalize),
            "model": frame[model_column].map(
                identify_model
            ),
            "sequence_length": extract_number(
                frame[sequence_column]
            ),
            "memory_budget": extract_number(
                frame[budget_column]
            ),
            "memory_recall": extract_number(
                frame[recall_column]
            ),
        }
    )

    target = standardized[
        (
            standardized["task"]
            == TARGET_TASK
        )
        & (
            standardized["sequence_length"]
            == TARGET_SEQUENCE
        )
        & (
            standardized["memory_budget"]
            == TARGET_BUDGET
        )
    ].dropna(
        subset=[
            "model",
            "memory_recall",
        ]
    )

    models = set(target["model"])

    required_models = {
        "budgetmem_r",
        "uniform_cache",
        "reservoir_cache",
    }

    if required_models.issubset(models):
        compatible_tables.append(
            (
                path,
                target,
            )
        )


print("=" * 100)
print("TARGET-CELL RECOVERY")
print("=" * 100)
print()

if compatible_tables:
    source_path, target = compatible_tables[0]

    means = (
        target.groupby("model")["memory_recall"]
        .mean()
        .to_dict()
    )

    budgetmem = float(means["budgetmem_r"])
    uniform = float(means["uniform_cache"])
    reservoir = float(means["reservoir_cache"])

    gain_uniform = budgetmem - uniform
    gain_reservoir = budgetmem - reservoir

    print("STATUS: TARGET_CELL_FOUND")
    print(f"Source: {source_path.relative_to(ROOT)}")
    print(f"BudgetMem-R recall: {budgetmem:.6f}")
    print(f"Uniform recall:     {uniform:.6f}")
    print(f"Reservoir recall:   {reservoir:.6f}")
    print(f"Gain over uniform:  {gain_uniform:+.6f}")
    print(f"Gain over reservoir:{gain_reservoir:+.6f}")

    recovered_path = (
        ROOT
        / "reports/tables/"
        "recovered_targeted_associative_recall.csv"
    )

    target.to_csv(
        recovered_path,
        index=False,
    )

    print()
    print(
        "Recovered target table:"
    )
    print(
        recovered_path.relative_to(ROOT)
    )

else:
    print("STATUS: TARGET_CELL_NOT_FOUND")
    print()
    print(
        "No CSV contains associative_recall at sequence "
        "length 1024 and budget 16 for all three required models."
    )


print()
print("=" * 100)
print("RECENT JSON DECISIONS AND SUMMARIES")
print("=" * 100)
print()

json_files = sorted(
    [
        path
        for path in REPORTS.rglob("*.json")
        if (
            "targeted_associative_recall" in str(path)
            or "pilot_tuned" in path.name
            or "summary" in path.name
            or "decision" in path.name
        )
    ],
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)[:20]


def print_relevant_json(
    value: object,
    prefix: str = "",
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = normalize(key)

            relevant = any(
                token in normalized_key
                for token in (
                    "decision",
                    "long_range",
                    "recall",
                    "accuracy",
                    "gain",
                    "outperform",
                    "write_frequency",
                )
            )

            if relevant and not isinstance(
                child,
                (dict, list),
            ):
                print(f"  {prefix}{key}: {child}")

            print_relevant_json(
                child,
                prefix=f"{prefix}{key}.",
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            print_relevant_json(
                child,
                prefix=f"{prefix}{index}.",
            )


if not json_files:
    print("- No relevant JSON files found.")
else:
    for path in json_files:
        print(f"- {path.relative_to(ROOT)}")

        try:
            payload = json.loads(
                path.read_text(encoding="utf-8")
            )
            print_relevant_json(payload)
        except Exception as exc:
            print(f"  Unreadable: {exc}")

        print()


print("=" * 100)
print("FINAL DIAGNOSIS")
print("=" * 100)
print()

if compatible_tables:
    print(
        "A compatible target result exists. The failure was "
        "caused by the capture function, not by missing target data."
    )
    print(
        "Do not rerun training until the recovered comparison "
        "has been reviewed."
    )
else:
    sequence_values: set[int] = set()

    for path in candidate_files:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue

        sequence_column = find_column(
            frame.columns,
            SEQUENCE_ALIASES,
        )

        if sequence_column:
            values = (
                extract_number(
                    frame[sequence_column]
                )
                .dropna()
                .astype(int)
                .tolist()
            )
            sequence_values.update(values)

    if TARGET_SEQUENCE not in sequence_values:
        print("Diagnosis: SMOKE_MODE_INCOMPATIBLE")
        print()
        print(
            "The smoke runs did not generate sequence-length "
            "1024 result rows. They cannot directly screen the "
            "required long-range target cell."
        )
    else:
        print("Diagnosis: RESULT_SCHEMA_OR_MATRIX_MISMATCH")
        print()
        print(
            "Sequence length 1024 exists, but the required task, "
            "budget, model, or recall combination is missing."
        )

    print()
    print(
        "Do not rerun Script 20 unchanged. Its capture condition "
        "cannot be satisfied by the generated smoke outputs."
    )
PY

echo
echo "============================================================"
echo " Inspection complete"
echo "============================================================"
echo
echo "Saved diagnostic:"
echo "$OUTPUT"
echo
echo "Open in VS Code:"
echo "code $OUTPUT"
