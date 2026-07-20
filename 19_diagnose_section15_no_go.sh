#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

SUMMARY="reports/tables/section15_final_go_comparison.csv"
DETAILS="reports/tables/section15_final_go_matched_cells.csv"
DECISION="reports/evidence/section15_final_go_decision.txt"
OUTPUT="reports/evidence/section15_no_go_diagnostic.txt"

for file in "$SUMMARY" "$DETAILS" "$DECISION"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Missing required file: $file"
        exit 1
    fi
done

python - <<'PY' | tee reports/evidence/section15_no_go_diagnostic.txt
from pathlib import Path

import pandas as pd

summary_path = Path(
    "reports/tables/section15_final_go_comparison.csv"
)
details_path = Path(
    "reports/tables/section15_final_go_matched_cells.csv"
)
decision_path = Path(
    "reports/evidence/section15_final_go_decision.txt"
)

summary = pd.read_csv(summary_path)
details = pd.read_csv(details_path)

print("=" * 100)
print("SECTION 15 NO-GO DIAGNOSTIC")
print("=" * 100)
print()

print(decision_path.read_text(encoding="utf-8"))

print("=" * 100)
print("POLICY-LEVEL COMPARISON")
print("=" * 100)

policy_columns = [
    "policy",
    "matched_same_budget_cells",
    "budgetmem_mean_recall",
    "policy_mean_recall",
    "mean_absolute_gain",
    "minimum_absolute_gain",
    "clear_wins",
    "clear_win_rate",
    "qualifies_as_outperformed",
]

available_policy_columns = [
    column
    for column in policy_columns
    if column in summary.columns
]

print(
    summary[available_policy_columns]
    .sort_values(
        "mean_absolute_gain",
        ascending=False,
    )
    .to_string(index=False)
)

failed = summary[
    ~summary["qualifies_as_outperformed"].astype(bool)
].copy()

print()
print("=" * 100)
print("POLICIES THAT FAILED THE GO GATE")
print("=" * 100)

if failed.empty:
    print("No failed policies were detected.")
else:
    print(
        failed[available_policy_columns]
        .sort_values(
            "mean_absolute_gain",
            ascending=True,
        )
        .to_string(index=False)
    )

print()
print("=" * 100)
print("WORST MATCHED TASK/BUDGET CELLS")
print("=" * 100)

detail_columns = [
    "policy",
    "task",
    "sequence_length",
    "budget",
    "seed",
    "budgetmem_recall",
    "policy_recall",
    "absolute_gain",
    "clear_win",
]

available_detail_columns = [
    column
    for column in detail_columns
    if column in details.columns
]

worst = details.sort_values(
    "absolute_gain",
    ascending=True,
)

print(
    worst[available_detail_columns]
    .head(20)
    .to_string(index=False)
)

print()
print("=" * 100)
print("TASK-LEVEL SUMMARY")
print("=" * 100)

task_summary = (
    details.groupby("task", as_index=False)
    .agg(
        matched_cells=("absolute_gain", "size"),
        budgetmem_mean_recall=("budgetmem_recall", "mean"),
        policy_mean_recall=("policy_recall", "mean"),
        mean_absolute_gain=("absolute_gain", "mean"),
        minimum_absolute_gain=("absolute_gain", "min"),
        clear_win_rate=("clear_win", "mean"),
    )
    .sort_values(
        "mean_absolute_gain",
        ascending=True,
    )
)

print(task_summary.to_string(index=False))

print()
print("=" * 100)
print("BUDGET-LEVEL SUMMARY")
print("=" * 100)

budget_summary = (
    details.groupby("budget", as_index=False)
    .agg(
        matched_cells=("absolute_gain", "size"),
        budgetmem_mean_recall=("budgetmem_recall", "mean"),
        policy_mean_recall=("policy_recall", "mean"),
        mean_absolute_gain=("absolute_gain", "mean"),
        minimum_absolute_gain=("absolute_gain", "min"),
        clear_win_rate=("clear_win", "mean"),
    )
    .sort_values("budget")
)

print(budget_summary.to_string(index=False))

print()
print("=" * 100)
print("REQUIRED NEXT ACTION")
print("=" * 100)

weakest_task = task_summary.iloc[0]["task"]
weakest_task_gain = task_summary.iloc[0]["mean_absolute_gain"]

weakest_budget = int(
    budget_summary.sort_values(
        "mean_absolute_gain",
        ascending=True,
    ).iloc[0]["budget"]
)

print(
    f"Weakest task: {weakest_task} "
    f"(mean gain {weakest_task_gain:+.6f})"
)
print(f"Weakest budget: {weakest_budget}")
print()
print(
    "Do not rerun the unchanged full pilot. "
    "Target the weakest task and budget in the next tuning revision."
)
PY

echo
echo "============================================================"
echo " Diagnostic complete"
echo "============================================================"
echo
echo "Saved:"
echo "$OUTPUT"
echo
echo "Open in VS Code:"
echo "code $OUTPUT"
