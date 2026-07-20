#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"

DECISION_JSON="reports/evidence/assoc_detached_memory_decision.json"
SUMMARY_JSON="reports/evidence/assoc_detached_memory_summary.json"
RESULTS_CSV="reports/tables/assoc_detached_memory_results.csv"
OUTPUT="reports/evidence/assoc_detached_memory_blocker.txt"

echo "============================================================"
echo " Detached-Memory Targeted Blocker Diagnostic"
echo "============================================================"

for required in \
    "$PYTHON" \
    "$DECISION_JSON" \
    "$SUMMARY_JSON" \
    "$RESULTS_CSV"; do

    if [[ ! -f "$required" ]]; then
        echo "ERROR: Missing required file:"
        echo "  $required"
        exit 1
    fi
done

"$PYTHON" - <<'PY' | tee \
    reports/evidence/assoc_detached_memory_blocker.txt

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


decision_path = Path(
    "reports/evidence/assoc_detached_memory_decision.json"
)

summary_path = Path(
    "reports/evidence/assoc_detached_memory_summary.json"
)

results_path = Path(
    "reports/tables/assoc_detached_memory_results.csv"
)

decision = json.loads(
    decision_path.read_text(encoding="utf-8")
)

summary = json.loads(
    summary_path.read_text(encoding="utf-8")
)

results = pd.read_csv(results_path)

criteria = decision.get("criteria", {})

records = summary.get("training_records", [])

budgetmem_records = [
    record
    for record in records
    if record.get("model") == "budgetmem_r"
]

budgetmem_record = (
    budgetmem_records[0]
    if budgetmem_records
    else {}
)

print("=" * 100)
print("DETACHED-MEMORY TARGETED BLOCKER")
print("=" * 100)
print()

print(f"Recorded decision: {decision.get('decision')}")
print(
    "Qualified deterministic policies: "
    f"{decision.get('qualified_policy_count', 0)}/2"
)

print()
print("=" * 100)
print("AUXILIARY CRITERIA")
print("=" * 100)

checks = {
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

for name, value in checks.items():
    status = "PASS" if value is True else "FAIL"
    print(f"{name:30s} {status:4s}  value={value}")

print()
print("=" * 100)
print("BUDGETMEM-R TRAINING RECORD")
print("=" * 100)

fields = [
    "task",
    "model",
    "first_loss",
    "final_loss",
    "maximum_gradient_norm",
    "finite_losses",
    "stability_pass",
    "checkpoint_path",
]

for field in fields:
    print(
        f"{field:30s} "
        f"{budgetmem_record.get(field, 'not recorded')}"
    )

print()
print("=" * 100)
print("BUDGETMEM-R EVALUATION ROWS")
print("=" * 100)

budgetmem_rows = results[
    results["model"].astype(str) == "budgetmem_r"
].copy()

columns = [
    "task",
    "sequence_length",
    "memory_budget",
    "memory_recall",
    "token_accuracy",
    "write_frequency",
    "stability_pass",
    "budget_pass",
    "resource_measurement_pass",
    "memory_size_max",
]

available = [
    column
    for column in columns
    if column in budgetmem_rows.columns
]

print(
    budgetmem_rows[available].to_string(index=False)
)

print()
print("=" * 100)
print("POLICY COMPARISONS")
print("=" * 100)

policy_summary = pd.DataFrame(
    decision.get("policy_summary", [])
)

if policy_summary.empty:
    print("No policy summary was recorded.")
else:
    print(policy_summary.to_string(index=False))

print()
print("=" * 100)
print("EXACT BLOCKERS")
print("=" * 100)

failed = [
    name
    for name, value in checks.items()
    if value is not True
]

if failed:
    for name in failed:
        print(f"- {name}")
else:
    print(
        "No criterion is false. The NO-GO result is caused "
        "by inconsistent decision logic."
    )

print()
print("=" * 100)
print("REQUIRED NEXT ACTION")
print("=" * 100)

if failed == ["stability_pass"]:
    print(
        "The recall, budget, resource, and write-frequency "
        "requirements passed."
    )
    print(
        "Only the gradient-stability criterion remains."
    )
    print(
        "Do not rerun the complete pilot. Inspect whether the "
        "reported maximum gradient is measured before or after "
        "gradient clipping."
    )

elif not failed:
    print(
        "Repair the decision-generation logic without retraining."
    )

else:
    print(
        "Repair the listed auxiliary conditions before any "
        "additional training or complete pilot."
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
echo "No training was run."
echo "No commit or push was performed."
