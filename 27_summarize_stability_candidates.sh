#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
OUTPUT="reports/evidence/assoc_stability_candidate_diagnostic.txt"

"$PYTHON" - <<'PY' | tee "$OUTPUT"
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


decision_dir = Path(
    "reports/evidence/assoc_stability_candidates"
)

rows = []

for decision_path in sorted(
    decision_dir.glob("*_decision.json")
):
    candidate = decision_path.name.removesuffix(
        "_decision.json"
    )

    decision = json.loads(
        decision_path.read_text(encoding="utf-8")
    )

    criteria = decision.get("criteria", {})

    summary_path = (
        decision_dir / f"{candidate}_summary.json"
    )

    maximum_gradient = None
    first_loss = None
    final_loss = None
    recorded_stability = None

    if summary_path.exists():
        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )

        records = summary.get(
            "training_records",
            [],
        )

        budgetmem_records = [
            record
            for record in records
            if record.get("model") == "budgetmem_r"
        ]

        if budgetmem_records:
            record = budgetmem_records[0]

            maximum_gradient = record.get(
                "maximum_gradient_norm"
            )
            first_loss = record.get("first_loss")
            final_loss = record.get("final_loss")
            recorded_stability = record.get(
                "stability_pass"
            )

    policy_summary = pd.DataFrame(
        decision.get("policy_summary", [])
    )

    minimum_policy_gain = None

    if (
        not policy_summary.empty
        and "minimum_gain" in policy_summary.columns
    ):
        minimum_policy_gain = float(
            policy_summary["minimum_gain"].min()
        )

    rows.append(
        {
            "candidate": candidate,
            "decision": decision.get("decision"),
            "stability_pass": criteria.get(
                "stability_pass"
            ),
            "budget_pass": criteria.get(
                "budget_pass"
            ),
            "resource_pass": criteria.get(
                "resource_pass"
            ),
            "write_frequency_pass": criteria.get(
                "write_frequency_pass"
            ),
            "qualified_policies": decision.get(
                "qualified_policy_count"
            ),
            "minimum_policy_gain": minimum_policy_gain,
            "maximum_gradient_norm": maximum_gradient,
            "first_loss": first_loss,
            "final_loss": final_loss,
            "recorded_stability": recorded_stability,
        }
    )


print("=" * 120)
print("SECTION 15 STABILITY-CANDIDATE DIAGNOSTIC")
print("=" * 120)
print()

if not rows:
    raise SystemExit(
        "No candidate decision files were found."
    )

frame = pd.DataFrame(rows)

print(frame.to_string(index=False))

print()
print("=" * 120)
print("FAILURE CLASSIFICATION")
print("=" * 120)

for row in rows:
    failures = []

    if row["stability_pass"] is not True:
        failures.append("stability")

    if row["qualified_policies"] is None:
        failures.append("missing recall comparison")
    elif int(row["qualified_policies"]) < 2:
        failures.append("two-policy recall")

    if row["budget_pass"] is not True:
        failures.append("memory budget")

    if row["resource_pass"] is not True:
        failures.append("resource measurement")

    if row["write_frequency_pass"] is not True:
        failures.append("write frequency")

    print(
        f"{row['candidate']}: "
        + (
            ", ".join(failures)
            if failures
            else "all checks passed"
        )
    )

print()
print("=" * 120)
print("NEXT DECISION")
print("=" * 120)

stable_candidates = [
    row
    for row in rows
    if row["stability_pass"] is True
]

recall_candidates = [
    row
    for row in rows
    if (
        row["qualified_policies"] is not None
        and int(row["qualified_policies"]) >= 2
    )
]

if not stable_candidates and recall_candidates:
    print(
        "All useful recall candidates remain unstable. "
        "The next repair must address the gradient computation "
        "or optimizer path, not the recall objective."
    )
elif stable_candidates and not recall_candidates:
    print(
        "At least one candidate is stable, but the recall advantage "
        "was lost. The next search must preserve stability while "
        "restoring the memory-recall gain."
    )
elif not stable_candidates and not recall_candidates:
    print(
        "The candidates failed both stability and recall. "
        "Inspect the implementation before another parameter sweep."
    )
else:
    print(
        "A candidate appears to pass both major gates. "
        "Its aggregate decision logic requires inspection."
    )
PY

echo
echo "Diagnostic saved:"
echo "  $OUTPUT"
