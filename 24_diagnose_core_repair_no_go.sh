#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"

DECISION_JSON="reports/evidence/assoc_core_repair_screen_decision.json"
RESULTS_CSV="reports/tables/assoc_core_repair_screen_results.csv"
OUTPUT="reports/evidence/assoc_core_repair_no_go_diagnostic.txt"

for file in "$DECISION_JSON" "$RESULTS_CSV"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Missing required file:"
        echo "  $file"
        exit 1
    fi
done

"$PYTHON" - <<'PY' | tee \
    reports/evidence/assoc_core_repair_no_go_diagnostic.txt

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


decision_path = Path(
    "reports/evidence/"
    "assoc_core_repair_screen_decision.json"
)

results_path = Path(
    "reports/tables/"
    "assoc_core_repair_screen_results.csv"
)

decision = json.loads(
    decision_path.read_text(encoding="utf-8")
)

results = pd.read_csv(results_path)

print("=" * 100)
print("ASSOCIATIVE-RECALL CORE-REPAIR DIAGNOSTIC")
print("=" * 100)
print()

print(f"Recorded decision: {decision.get('decision')}")
print()

print("=" * 100)
print("AUXILIARY CRITERIA")
print("=" * 100)

criteria = decision.get("criteria", {})

for name, value in criteria.items():
    print(f"{name}: {value}")

print()
print("=" * 100)
print("PERFORMANCE GATE")
print("=" * 100)

print(
    f"Qualified policies: "
    f"{decision.get('qualified_policy_count', 'unknown')}/2"
)

policy_summary = pd.DataFrame(
    decision.get("policy_summary", [])
)

if not policy_summary.empty:
    print()
    print(policy_summary.to_string(index=False))

print()
print("=" * 100)
print("BUDGETMEM-R RESULT ROWS")
print("=" * 100)

budgetmem = results[
    results["model"].astype(str) == "budgetmem_r"
].copy()

important_columns = [
    "task",
    "model",
    "sequence_length",
    "memory_budget",
    "memory_recall",
    "token_accuracy",
    "write_frequency",
    "stability_pass",
    "budget_pass",
    "resource_measurement_pass",
    "memory_size_max",
    "configured_budget",
]

available = [
    column
    for column in important_columns
    if column in budgetmem.columns
]

print(
    budgetmem[available].to_string(index=False)
)

print()
print("=" * 100)
print("FAILED CONDITIONS")
print("=" * 100)

failures = []

boolean_checks = {
    "stability_pass": criteria.get("stability_pass"),
    "budget_pass": criteria.get("budget_pass"),
    "resource_pass": criteria.get("resource_pass"),
    "write_frequency_pass": criteria.get(
        "write_frequency_pass"
    ),
    "outperforms_both_policies": criteria.get(
        "outperforms_both_policies"
    ),
}

for name, value in boolean_checks.items():
    if value is not True:
        failures.append(name)
        print(f"FAIL: {name} = {value}")

if not failures:
    print(
        "No criterion is false. The decision-generation logic "
        "is inconsistent and must be corrected."
    )

print()
print("=" * 100)
print("INTERPRETATION")
print("=" * 100)

performance_pass = (
    criteria.get("outperforms_both_policies") is True
    and decision.get("qualified_policy_count", 0) >= 2
)

if performance_pass:
    print(
        "PASS: BudgetMem-R clearly outperformed both deterministic "
        "memory policies on matched-budget long-range recall."
    )
else:
    print(
        "FAIL: The required two-policy recall advantage was not met."
    )

if failures:
    print()
    print(
        "The recorded NO-GO was caused by the auxiliary condition(s):"
    )

    for failure in failures:
        print(f"- {failure}")

    print()
    print(
        "Do not rerun training until the failed condition is repaired "
        "or confirmed to be a reporting defect."
    )
else:
    print()
    print(
        "The NO-GO classification is a decision-script defect because "
        "all recorded conditions passed."
    )
PY

echo
echo "============================================================"
echo " Diagnostic complete"
echo "============================================================"
echo
echo "Saved:"
echo "  $OUTPUT"
echo
echo "Open in VS Code:"
echo "  code $OUTPUT"
