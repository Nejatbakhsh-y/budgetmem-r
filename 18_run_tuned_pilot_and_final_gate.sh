#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
CONFIG="configs/experiments/pilot_tuned.yaml"
FINAL_GATE="17_section15_final_go_decision.sh"

RESULTS="reports/tables/pilot_tuned_results.csv"
SUMMARY="reports/evidence/pilot_tuned_summary.json"
PILOT_GATE="reports/evidence/pilot_tuned_go_no_go.json"
REPORT="reports/pilot_tuned_report.md"

echo "============================================================"
echo " Section 14.10 and 14.11"
echo " Full Tuned Pilot and Final Recall Gate"
echo "============================================================"
echo "Repository: $ROOT"
echo

required=(
    "$PYTHON"
    "$CONFIG"
    "scripts/run_pilot.py"
    "$FINAL_GATE"
)

for path in "${required[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: Required path is missing:"
        echo "  $path"
        exit 1
    fi
done

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

timestamp="$(date +%Y%m%d_%H%M%S)"
backup="$ROOT/.section15_backup/full_tuned_recall_gate_$timestamp"

mkdir -p \
    "$backup/reports/tables" \
    "$backup/reports/evidence" \
    "$backup/reports"

echo "Backing up previous tuned artifacts."

for path in \
    "$RESULTS" \
    "$SUMMARY" \
    "$PILOT_GATE" \
    "$REPORT" \
    "reports/evidence/section15_final_go_decision.txt" \
    "reports/evidence/section15_final_go_decision.json" \
    "reports/tables/section15_final_go_comparison.csv" \
    "reports/tables/section15_final_go_matched_cells.csv"; do

    if [[ -f "$path" ]]; then
        mkdir -p "$backup/$(dirname "$path")"
        cp -f "$path" "$backup/$path"
    fi
done

if [[ -d "outputs/pilot_tuned" ]]; then
    mkdir -p "$backup/outputs"
    mv "outputs/pilot_tuned" "$backup/outputs/pilot_tuned"
fi

rm -f \
    "$RESULTS" \
    "$SUMMARY" \
    "$PILOT_GATE" \
    "$REPORT" \
    reports/evidence/section15_final_go_decision.txt \
    reports/evidence/section15_final_go_decision.json \
    reports/tables/section15_final_go_comparison.csv \
    reports/tables/section15_final_go_matched_cells.csv

echo
echo "Validating the selected tuned configuration."

"$PYTHON" - <<'PY'
from pathlib import Path

import yaml

path = Path("configs/experiments/pilot_tuned.yaml")
cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

expected_tasks = {
    "selective_copy",
    "associative_recall",
    "distractor_heavy_retrieval",
}
expected_lengths = {256, 512, 1024}
expected_budgets = {16, 32}
expected_models = {
    "gru",
    "gru_uniform_cache",
    "gru_reservoir_cache",
    "budgetmem_r",
}

matrix = cfg.get("matrix", {})

tasks = set(matrix.get("tasks", []))
lengths = {
    int(value)
    for value in matrix.get("evaluation_sequence_lengths", [])
}
budgets = {
    int(value)
    for value in matrix.get("memory_budgets", [])
}
models = set(matrix.get("models", []))

problems = []

if tasks != expected_tasks:
    problems.append(
        f"tasks={sorted(tasks)}, expected={sorted(expected_tasks)}"
    )

if lengths != expected_lengths:
    problems.append(
        f"lengths={sorted(lengths)}, expected={sorted(expected_lengths)}"
    )

if budgets != expected_budgets:
    problems.append(
        f"budgets={sorted(budgets)}, expected={sorted(expected_budgets)}"
    )

if models != expected_models:
    problems.append(
        f"models={sorted(models)}, expected={sorted(expected_models)}"
    )

if problems:
    raise SystemExit(
        "Invalid tuned pilot configuration:\n- "
        + "\n- ".join(problems)
    )

print("Configuration validation: PASS")
print(f"Tasks:    {sorted(tasks)}")
print(f"Lengths:  {sorted(lengths)}")
print(f"Budgets:  {sorted(budgets)}")
print(f"Models:   {sorted(models)}")
print()
print("Training settings:")
for key, value in cfg.get("training", {}).items():
    print(f"  {key}: {value}")
print()
print("Model settings:")
for key, value in cfg.get("model", {}).items():
    print(f"  {key}: {value}")
PY

echo
echo "Running focused Section 15 tests."

"$PYTHON" -m pytest tests/pilot -q

echo
echo "============================================================"
echo " Running the Full Tuned Section 15 Pilot"
echo " Expected matrix size: 72 result rows"
echo "============================================================"
echo

"$PYTHON" scripts/run_pilot.py \
    --config "$CONFIG"

echo
echo "Validating the completed pilot matrix."

"$PYTHON" - <<'PY'
from pathlib import Path

import pandas as pd

path = Path("reports/tables/pilot_tuned_results.csv")

if not path.exists():
    raise SystemExit(f"Missing tuned result table: {path}")

frame = pd.read_csv(path)

required_columns = {
    "task",
    "model",
    "sequence_length",
    "memory_budget",
    "memory_recall",
    "budget_pass",
    "resource_measurement_pass",
}

missing = required_columns - set(frame.columns)

if missing:
    raise SystemExit(
        f"Missing required result columns: {sorted(missing)}"
    )

expected_tasks = {
    "selective_copy",
    "associative_recall",
    "distractor_heavy_retrieval",
}
expected_models = {
    "gru",
    "gru_uniform_cache",
    "gru_reservoir_cache",
    "budgetmem_r",
}
expected_lengths = {256, 512, 1024}
expected_budgets = {16, 32}

actual_tasks = set(frame["task"].astype(str))
actual_models = set(frame["model"].astype(str))
actual_lengths = set(frame["sequence_length"].astype(int))
actual_budgets = set(frame["memory_budget"].astype(int))

expected_rows = (
    len(expected_tasks)
    * len(expected_models)
    * len(expected_lengths)
    * len(expected_budgets)
)

print(f"Result rows: {len(frame)}/{expected_rows}")
print(f"Tasks:       {sorted(actual_tasks)}")
print(f"Models:      {sorted(actual_models)}")
print(f"Lengths:     {sorted(actual_lengths)}")
print(f"Budgets:     {sorted(actual_budgets)}")

if len(frame) != expected_rows:
    raise SystemExit(
        f"Incomplete matrix: found {len(frame)}, expected {expected_rows}"
    )

if actual_tasks != expected_tasks:
    raise SystemExit("Task matrix is incomplete.")

if actual_models != expected_models:
    raise SystemExit("Model matrix is incomplete.")

if actual_lengths != expected_lengths:
    raise SystemExit("Sequence-length matrix is incomplete.")

if actual_budgets != expected_budgets:
    raise SystemExit("Memory-budget matrix is incomplete.")

if not frame["budget_pass"].astype(bool).all():
    raise SystemExit("At least one strict memory-budget check failed.")

if not frame["resource_measurement_pass"].astype(bool).all():
    raise SystemExit("At least one resource-measurement check failed.")

print("Complete 72-cell matrix: PASS")
print("Strict budget enforcement: PASS")
print("Resource measurement: PASS")
PY

echo
echo "Pilot-produced decision:"
echo

if [[ -f "$PILOT_GATE" ]]; then
    cat "$PILOT_GATE"
else
    echo "WARNING: $PILOT_GATE was not generated."
fi

echo
echo "============================================================"
echo " Running the Section 14.11 Recall-Based Final Gate"
echo "============================================================"
echo

set +e

RESULT_FILE="$RESULTS" \
AUTO_COMMIT=0 \
AUTO_PUSH=0 \
bash "$FINAL_GATE"

gate_status=$?

set -e

if [[ "$gate_status" -eq 0 ]]; then
    echo
    echo "============================================================"
    echo " FINAL DECISION: GO"
    echo "============================================================"
    echo
    echo "BudgetMem-R clearly outperformed at least two"
    echo "deterministic memory policies on matched-budget"
    echo "long-range recall."
    echo
    echo "Saving the successful evidence to Git."

    paths=(
        "18_run_tuned_pilot_and_final_gate.sh"
        "$FINAL_GATE"
        "$CONFIG"
        "$RESULTS"
        "$SUMMARY"
        "$PILOT_GATE"
        "$REPORT"
        "reports/evidence/section15_final_go_decision.txt"
        "reports/evidence/section15_final_go_decision.json"
        "reports/tables/section15_final_go_comparison.csv"
        "reports/tables/section15_final_go_matched_cells.csv"
    )

    for path in "${paths[@]}"; do
        if [[ -e "$path" ]]; then
            git add "$path"
        fi
    done

    git diff --check

    if git diff --cached --quiet; then
        echo "No new tracked changes require a commit."
    else
        git commit -m \
            "results: record Section 15 tuned GO decision"
    fi

    branch="$(git branch --show-current)"

    if [[ -n "$branch" ]] \
        && git remote get-url origin >/dev/null 2>&1; then

        git push -u origin "$branch"
        echo "Pushed branch: $branch"
    else
        echo "WARNING: Git push was skipped."
    fi

    echo
    echo "Section 14.10: COMPLETE"
    echo "Section 14.11: COMPLETE — GO"
    echo
    echo "Final evidence:"
    echo "  reports/evidence/section15_final_go_decision.txt"
    exit 0
fi

if [[ "$gate_status" -eq 2 ]]; then
    echo
    echo "============================================================"
    echo " FINAL DECISION: NO-GO"
    echo "============================================================"
    echo
    echo "The complete tuned pilot ran successfully, but"
    echo "BudgetMem-R still did not clearly outperform at least"
    echo "two deterministic memory policies on matched-budget"
    echo "long-range recall."
    echo
    echo "Section 14.10: COMPLETE"
    echo "Section 14.11: NOT COMPLETE — NO-GO"
    echo
    echo "No automatic commit or push was performed."
    echo
    echo "Inspect:"
    echo "  reports/evidence/section15_final_go_decision.txt"
    echo "  reports/tables/section15_final_go_comparison.csv"
    echo "  reports/tables/section15_final_go_matched_cells.csv"
    exit 2
fi

echo
echo "ERROR: The final gate failed with status $gate_status."
exit "$gate_status"
