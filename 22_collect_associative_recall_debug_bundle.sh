#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"

STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_ROOT="reports/evidence/section15_debug_bundle_${STAMP}"
ARCHIVE="reports/evidence/section15_debug_bundle_${STAMP}.tar.gz"
SUMMARY="reports/evidence/section15_debug_bundle_${STAMP}.txt"
FILE_LIST="$BUNDLE_ROOT/included_files.txt"

mkdir -p \
    "$BUNDLE_ROOT/repository_files" \
    "$BUNDLE_ROOT/diagnostics" \
    "$BUNDLE_ROOT/tests"

echo "============================================================"
echo " Section 15 Associative-Recall Debug Bundle"
echo "============================================================"
echo "Repository: $ROOT"
echo

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Project Python was not found:"
    echo "  $PYTHON"
    exit 1
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ------------------------------------------------------------
# Build a deduplicated list of relevant repository files.
# ------------------------------------------------------------

{
    find src tests scripts configs \
        -type f \
        \( \
            -name "*.py" \
            -o -name "*.yaml" \
            -o -name "*.yml" \
        \) \
        -print0 2>/dev/null \
        | xargs -0 -r grep -IlE \
            'associative_recall|BudgetMem|budgetmem_r|retrieval_k|memory_recall|write_rate|write_controller|successful_long_range_retrieval'

    find reports/evidence reports/tables reports/logs \
        -type f \
        \( \
            -iname "*section15*" \
            -o -iname "*pilot_tuned*" \
            -o -iname "*associative*" \
            -o -iname "*optimization*" \
            -o -iname "*capacity*" \
        \) \
        2>/dev/null

    find . -maxdepth 1 \
        -type f \
        \( \
            -name "15_*.sh" \
            -o -name "17_*.sh" \
            -o -name "18_*.sh" \
            -o -name "19_*.sh" \
            -o -name "20_*.sh" \
            -o -name "21_*.sh" \
        \) \
        -printf '%P\n' 2>/dev/null
} \
    | sed 's#^\./##' \
    | awk 'NF && !seen[$0]++' \
    | sort \
    > "$FILE_LIST"

echo "Collecting relevant files."

while IFS= read -r file; do
    [[ -f "$file" ]] || continue

    destination="$BUNDLE_ROOT/repository_files/$file"
    mkdir -p "$(dirname "$destination")"
    cp -f "$file" "$destination"
done < "$FILE_LIST"

# ------------------------------------------------------------
# Record repository state.
# ------------------------------------------------------------

{
    echo "SECTION 15 ASSOCIATIVE-RECALL DEBUG SUMMARY"
    echo "==========================================="
    echo
    echo "Generated:"
    date --iso-8601=seconds
    echo
    echo "Repository:"
    echo "$ROOT"
    echo
    echo "Branch:"
    git branch --show-current || true
    echo
    echo "Latest commit:"
    git log -1 --oneline || true
    echo
    echo "Git status:"
    git status --short || true
    echo
    echo "Python:"
    "$PYTHON" --version
    echo
    echo "PyTorch:"
    "$PYTHON" - <<'PY'
try:
    import torch

    print(f"version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"thread_count={torch.get_num_threads()}")
except Exception as exc:
    print(f"PyTorch inspection failed: {exc}")
PY
    echo
    echo "Included file count:"
    wc -l < "$FILE_LIST"
    echo
    echo "Included files:"
    cat "$FILE_LIST"
} > "$SUMMARY"

cp -f "$SUMMARY" "$BUNDLE_ROOT/diagnostics/overview.txt"

git status --short \
    > "$BUNDLE_ROOT/diagnostics/git_status.txt" \
    2>&1 || true

git diff \
    > "$BUNDLE_ROOT/diagnostics/unstaged_git_diff.patch" \
    2>&1 || true

git diff --cached \
    > "$BUNDLE_ROOT/diagnostics/staged_git_diff.patch" \
    2>&1 || true

# ------------------------------------------------------------
# Extract important configuration and source references.
# ------------------------------------------------------------

{
    echo "ASSOCIATIVE-RECALL SOURCE REFERENCES"
    echo "===================================="
    echo

    grep -RInE \
        'associative_recall|memory_recall|retrieval_k|write_rate|successful_long_range_retrieval|query|key|value|read_gate|write_gate' \
        src tests scripts configs \
        --include="*.py" \
        --include="*.yaml" \
        --include="*.yml" \
        2>/dev/null || true
} > "$BUNDLE_ROOT/diagnostics/source_reference_index.txt"

{
    echo "CURRENT DECISION EVIDENCE"
    echo "========================="
    echo

    for file in \
        reports/evidence/section15_final_go_decision.txt \
        reports/evidence/section15_no_go_diagnostic.txt \
        reports/evidence/targeted_associative_recall_output_inspection.txt \
        reports/tables/section15_final_go_comparison.csv \
        reports/tables/section15_final_go_matched_cells.csv; do

        if [[ -f "$file" ]]; then
            echo
            echo "------------------------------------------------------------"
            echo "FILE: $file"
            echo "------------------------------------------------------------"
            cat "$file"
        fi
    done
} > "$BUNDLE_ROOT/diagnostics/current_decision_evidence.txt"

# ------------------------------------------------------------
# Inspect all current result tables for the exact failed cell.
# ------------------------------------------------------------

"$PYTHON" - <<'PY' \
    > "$BUNDLE_ROOT/diagnostics/result_matrix_inventory.txt"
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path.cwd()


def normalize(value: object) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def find_column(columns, aliases):
    lookup = {
        normalize(column): column
        for column in columns
    }

    for alias in aliases:
        if alias in lookup:
            return lookup[alias]

    return None


aliases = {
    "task": [
        "task",
        "task_name",
        "dataset",
    ],
    "model": [
        "model",
        "model_name",
        "method",
        "architecture",
    ],
    "sequence": [
        "sequence_length",
        "seq_length",
        "seq_len",
        "context_length",
    ],
    "budget": [
        "memory_budget",
        "budget",
        "cache_size",
        "memory_size",
    ],
    "recall": [
        "memory_recall",
        "long_range_recall",
        "retrieval_recall",
        "relevant_state_retention_rate",
        "recall",
    ],
}


print("RESULT MATRIX INVENTORY")
print("=======================")

for path in sorted(Path("reports").rglob("*.csv")):
    try:
        frame = pd.read_csv(path)
    except Exception:
        continue

    detected = {
        key: find_column(frame.columns, values)
        for key, values in aliases.items()
    }

    if not any(detected.values()):
        continue

    print()
    print("-" * 100)
    print(path)
    print(f"rows={len(frame)}")
    print(f"columns={list(frame.columns)}")
    print(f"detected={detected}")

    for key in (
        "task",
        "model",
        "sequence",
        "budget",
    ):
        column = detected[key]

        if column is None:
            continue

        values = (
            frame[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        print(f"{key}_values={sorted(values)[:100]}")

    required = [
        detected["task"],
        detected["model"],
        detected["sequence"],
        detected["budget"],
    ]

    if not all(required):
        continue

    sequence = pd.to_numeric(
        frame[detected["sequence"]],
        errors="coerce",
    )

    budget = pd.to_numeric(
        frame[detected["budget"]],
        errors="coerce",
    )

    target = frame[
        (
            frame[detected["task"]]
            .astype(str)
            .map(normalize)
            == "associative_recall"
        )
        & (sequence == 1024)
        & (budget == 16)
    ]

    if not target.empty:
        print()
        print("TARGET CELL ROWS:")
        print(target.to_string(index=False))
PY

# ------------------------------------------------------------
# Run focused tests without stopping bundle creation on failure.
# ------------------------------------------------------------

echo "Running associative-recall tests."

set +e

mapfile -t TASK_TESTS < <(
    find tests \
        -type f \
        -iname "*associative*recall*.py" \
        2>/dev/null \
        | sort
)

if [[ "${#TASK_TESTS[@]}" -gt 0 ]]; then
    "$PYTHON" -m pytest \
        "${TASK_TESTS[@]}" \
        -q -vv \
        > "$BUNDLE_ROOT/tests/associative_recall_tests.txt" \
        2>&1

    echo $? \
        > "$BUNDLE_ROOT/tests/associative_recall_tests.exit_code"
else
    echo "No associative-recall-specific test file was found." \
        > "$BUNDLE_ROOT/tests/associative_recall_tests.txt"

    echo "127" \
        > "$BUNDLE_ROOT/tests/associative_recall_tests.exit_code"
fi

mapfile -t PILOT_TESTS < <(
    find tests \
        -type f \
        \( \
            -path "*/pilot/*" \
            -o -iname "*pilot*.py" \
        \) \
        2>/dev/null \
        | sort
)

if [[ "${#PILOT_TESTS[@]}" -gt 0 ]]; then
    "$PYTHON" -m pytest \
        "${PILOT_TESTS[@]}" \
        -q \
        > "$BUNDLE_ROOT/tests/pilot_tests.txt" \
        2>&1

    echo $? \
        > "$BUNDLE_ROOT/tests/pilot_tests.exit_code"
else
    echo "No pilot test file was found." \
        > "$BUNDLE_ROOT/tests/pilot_tests.txt"

    echo "127" \
        > "$BUNDLE_ROOT/tests/pilot_tests.exit_code"
fi

set -e

# ------------------------------------------------------------
# Create the final archive.
# ------------------------------------------------------------

tar -czf "$ARCHIVE" \
    -C "$BUNDLE_ROOT" \
    .

echo
echo "============================================================"
echo " Debug bundle complete"
echo "============================================================"
echo
echo "Summary:"
echo "  $SUMMARY"
echo
echo "Archive:"
echo "  $ARCHIVE"
echo
echo "No source files were modified."
echo "No commit or push was performed."
echo
