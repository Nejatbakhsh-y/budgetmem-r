#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"

SUMMARY_JSON="reports/evidence/assoc_core_repair_screen_summary.json"
DECISION_JSON="reports/evidence/assoc_core_repair_screen_decision.json"
CONFIG_FILE="configs/experiments/pilot_assoc_core_repair_screen.yaml"

OUTPUT_TXT="reports/evidence/assoc_core_repair_stability_diagnostic.txt"
OUTPUT_JSON="reports/evidence/assoc_core_repair_stability_diagnostic.json"

echo "============================================================"
echo " Section 15 Stability-Failure Diagnosis"
echo "============================================================"
echo

for required in \
    "$PYTHON" \
    "$SUMMARY_JSON" \
    "$DECISION_JSON" \
    "$CONFIG_FILE"; do

    if [[ ! -e "$required" ]]; then
        echo "ERROR: Missing required path:"
        echo "  $required"
        exit 1
    fi
done

mkdir -p reports/evidence

"$PYTHON" - <<'PY' | tee \
    reports/evidence/assoc_core_repair_stability_diagnostic.txt

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml


root = Path.cwd()

summary_path = Path(
    "reports/evidence/"
    "assoc_core_repair_screen_summary.json"
)

decision_path = Path(
    "reports/evidence/"
    "assoc_core_repair_screen_decision.json"
)

config_path = Path(
    "configs/experiments/"
    "pilot_assoc_core_repair_screen.yaml"
)

output_json_path = Path(
    "reports/evidence/"
    "assoc_core_repair_stability_diagnostic.json"
)


summary = json.loads(
    summary_path.read_text(encoding="utf-8")
)

decision = json.loads(
    decision_path.read_text(encoding="utf-8")
)

config = yaml.safe_load(
    config_path.read_text(encoding="utf-8")
)


records = summary.get("training_records", [])

budgetmem_records = [
    record
    for record in records
    if str(record.get("model", "")).lower()
    == "budgetmem_r"
]

if not budgetmem_records:
    raise SystemExit(
        "ERROR: No BudgetMem-R training record was found."
    )


record = budgetmem_records[0]

first_loss = float(
    record.get("first_loss", float("nan"))
)

final_loss = float(
    record.get("final_loss", float("nan"))
)

maximum_gradient_norm = float(
    record.get(
        "maximum_gradient_norm",
        float("nan"),
    )
)

finite_losses_recorded = bool(
    record.get("finite_losses", False)
)

recorded_stability = bool(
    record.get("stability_pass", False)
)

maximum_allowed_gradient = float(
    config["training"][
        "maximum_acceptable_gradient_norm"
    ]
)

gradient_clip_norm = float(
    config["training"]["gradient_clip_norm"]
)

learning_rate = float(
    config["training"]["learning_rate"]
)

epochs = int(
    config["training"]["epochs"]
)

loss_growth_limit = max(
    first_loss * 2.0,
    first_loss + 1.0,
)


finite_first = math.isfinite(first_loss)
finite_final = math.isfinite(final_loss)
finite_gradient = math.isfinite(
    maximum_gradient_norm
)

finite_condition = bool(
    finite_losses_recorded
    and finite_first
    and finite_final
    and finite_gradient
)

gradient_condition = bool(
    finite_gradient
    and maximum_gradient_norm
    <= maximum_allowed_gradient
)

loss_growth_condition = bool(
    finite_first
    and finite_final
    and final_loss <= loss_growth_limit
)


checkpoint_value = record.get(
    "checkpoint_path",
    "",
)

checkpoint_path = (
    root / str(checkpoint_value)
    if checkpoint_value
    else None
)

checkpoint_exists = bool(
    checkpoint_path
    and checkpoint_path.exists()
)

history: list[float] = []
checkpoint_epoch: int | None = None
checkpoint_config_sha256: str | None = None

if checkpoint_exists and checkpoint_path is not None:
    payload: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    history = [
        float(value)
        for value in payload.get("history", [])
    ]

    checkpoint_epoch = int(
        payload.get("epoch", -1)
    )

    checkpoint_config_sha256 = str(
        payload.get("config_sha256", "")
    )


history_finite = bool(
    history
    and all(
        math.isfinite(value)
        for value in history
    )
)

history_min = (
    min(history)
    if history
    else float("nan")
)

history_max = (
    max(history)
    if history
    else float("nan")
)

best_epoch = (
    history.index(history_min) + 1
    if history
    else None
)

final_to_first_ratio = (
    final_loss / first_loss
    if first_loss != 0.0
    and finite_first
    and finite_final
    else float("nan")
)


failed_conditions: list[str] = []

if not finite_condition:
    failed_conditions.append(
        "NONFINITE_OR_INVALID_LOSS"
    )

if not gradient_condition:
    failed_conditions.append(
        "EXCESSIVE_PRECLIP_GRADIENT_NORM"
    )

if not loss_growth_condition:
    failed_conditions.append(
        "FINAL_LOSS_GROWTH"
    )


recomputed_stability = bool(
    finite_condition
    and gradient_condition
    and loss_growth_condition
)


if (
    not failed_conditions
    and recorded_stability is False
):
    diagnosis = "STABILITY_REPORTING_DEFECT"

elif failed_conditions == [
    "EXCESSIVE_PRECLIP_GRADIENT_NORM"
]:
    diagnosis = "GRADIENT_NORM_FAILURE"

elif failed_conditions == [
    "FINAL_LOSS_GROWTH"
]:
    diagnosis = "LOSS_GROWTH_FAILURE"

elif set(failed_conditions) == {
    "EXCESSIVE_PRECLIP_GRADIENT_NORM",
    "FINAL_LOSS_GROWTH",
}:
    diagnosis = "GRADIENT_AND_LOSS_FAILURE"

elif "NONFINITE_OR_INVALID_LOSS" in failed_conditions:
    diagnosis = "NONFINITE_TRAINING_FAILURE"

else:
    diagnosis = "MULTIPLE_STABILITY_FAILURES"


print("=" * 100)
print("ASSOCIATIVE-RECALL STABILITY DIAGNOSTIC")
print("=" * 100)
print()

print("Performance status:")
print(
    "- Policies clearly outperformed: "
    f"{decision.get('qualified_policy_count', 'unknown')}/2"
)
print(
    "- Performance requirement: "
    + (
        "PASS"
        if decision.get(
            "qualified_policy_count",
            0,
        )
        >= 2
        else "FAIL"
    )
)

print()
print("=" * 100)
print("BUDGETMEM-R TRAINING RECORD")
print("=" * 100)

print(
    f"Task:                         "
    f"{record.get('task')}"
)
print(
    f"Model:                        "
    f"{record.get('model')}"
)
print(
    f"Epochs configured:            "
    f"{epochs}"
)
print(
    f"Learning rate:                "
    f"{learning_rate:.8f}"
)
print(
    f"Gradient clipping norm:       "
    f"{gradient_clip_norm:.6f}"
)
print(
    f"First epoch loss:              "
    f"{first_loss:.10f}"
)
print(
    f"Final epoch loss:              "
    f"{final_loss:.10f}"
)
print(
    f"Allowed final-loss maximum:   "
    f"{loss_growth_limit:.10f}"
)
print(
    f"Final/first loss ratio:        "
    f"{final_to_first_ratio:.6f}"
)
print(
    f"Maximum pre-clip gradient:     "
    f"{maximum_gradient_norm:.10f}"
)
print(
    f"Allowed gradient maximum:     "
    f"{maximum_allowed_gradient:.10f}"
)
print(
    f"Finite losses recorded:       "
    f"{finite_losses_recorded}"
)
print(
    f"Recorded stability pass:      "
    f"{recorded_stability}"
)
print(
    f"Recomputed stability pass:    "
    f"{recomputed_stability}"
)

print()
print("=" * 100)
print("CHECKPOINT HISTORY")
print("=" * 100)

print(
    f"Checkpoint:                    "
    f"{checkpoint_value or 'not recorded'}"
)
print(
    f"Checkpoint exists:             "
    f"{checkpoint_exists}"
)
print(
    f"Checkpoint epoch:              "
    f"{checkpoint_epoch}"
)
print(
    f"History entries:               "
    f"{len(history)}"
)
print(
    f"All history values finite:     "
    f"{history_finite}"
)
print(
    f"Minimum epoch loss:            "
    f"{history_min:.10f}"
)
print(
    f"Maximum epoch loss:            "
    f"{history_max:.10f}"
)
print(
    f"Best epoch:                    "
    f"{best_epoch}"
)

if history:
    print()
    print("Epoch loss history:")

    for index, value in enumerate(
        history,
        start=1,
    ):
        marker = (
            "  <-- best"
            if index == best_epoch
            else ""
        )

        print(
            f"- epoch {index:02d}: "
            f"{value:.10f}{marker}"
        )

print()
print("=" * 100)
print("STABILITY CONDITIONS")
print("=" * 100)

print(
    "Finite-loss condition:        "
    + (
        "PASS"
        if finite_condition
        else "FAIL"
    )
)

print(
    "Gradient-norm condition:      "
    + (
        "PASS"
        if gradient_condition
        else "FAIL"
    )
)

print(
    "Loss-growth condition:        "
    + (
        "PASS"
        if loss_growth_condition
        else "FAIL"
    )
)

print()
print("=" * 100)
print("FINAL DIAGNOSIS")
print("=" * 100)

print(f"Diagnosis: {diagnosis}")

if failed_conditions:
    print()
    print("Failed condition identifiers:")

    for condition in failed_conditions:
        print(f"- {condition}")

print()
print("Required next action:")

if diagnosis == "GRADIENT_NORM_FAILURE":
    print(
        "- Stabilize the optimizer or model initialization."
    )
    print(
        "- Do not raise the acceptable-gradient threshold."
    )
    print(
        "- Rerun only the targeted associative-recall screen."
    )

elif diagnosis == "LOSS_GROWTH_FAILURE":
    print(
        "- Add best-checkpoint selection or early stopping."
    )
    print(
        "- Restore the lowest-loss finite checkpoint rather "
        "than using the final epoch automatically."
    )
    print(
        "- Rerun only the targeted associative-recall screen."
    )

elif diagnosis == "GRADIENT_AND_LOSS_FAILURE":
    print(
        "- Apply optimizer stabilization and best-checkpoint "
        "selection together."
    )
    print(
        "- Rerun only the targeted associative-recall screen."
    )

elif diagnosis == "NONFINITE_TRAINING_FAILURE":
    print(
        "- Inspect the first non-finite batch and loss term."
    )
    print(
        "- Do not proceed to another pilot."
    )

elif diagnosis == "STABILITY_REPORTING_DEFECT":
    print(
        "- All stability subconditions pass."
    )
    print(
        "- Repair the decision/reporting code without retraining."
    )

else:
    print(
        "- Review all failed stability conditions before "
        "another training run."
    )


payload = {
    "diagnosis": diagnosis,
    "performance_requirement_pass": (
        decision.get(
            "qualified_policy_count",
            0,
        )
        >= 2
    ),
    "recorded_stability_pass": (
        recorded_stability
    ),
    "recomputed_stability_pass": (
        recomputed_stability
    ),
    "failed_conditions": (
        failed_conditions
    ),
    "training_record": {
        "first_loss": first_loss,
        "final_loss": final_loss,
        "loss_growth_limit": (
            loss_growth_limit
        ),
        "final_to_first_ratio": (
            final_to_first_ratio
        ),
        "maximum_gradient_norm": (
            maximum_gradient_norm
        ),
        "maximum_allowed_gradient_norm": (
            maximum_allowed_gradient
        ),
        "gradient_clip_norm": (
            gradient_clip_norm
        ),
        "learning_rate": learning_rate,
        "epochs": epochs,
        "finite_losses": (
            finite_losses_recorded
        ),
    },
    "checkpoint": {
        "path": checkpoint_value,
        "exists": checkpoint_exists,
        "epoch": checkpoint_epoch,
        "config_sha256": (
            checkpoint_config_sha256
        ),
        "history_length": len(history),
        "history_finite": history_finite,
        "minimum_loss": history_min,
        "maximum_loss": history_max,
        "best_epoch": best_epoch,
        "history": history,
    },
}

output_json_path.write_text(
    json.dumps(
        payload,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print()
print("=" * 100)
print("FILES WRITTEN")
print("=" * 100)

print(
    "reports/evidence/"
    "assoc_core_repair_stability_diagnostic.txt"
)

print(
    "reports/evidence/"
    "assoc_core_repair_stability_diagnostic.json"
)
PY

echo
echo "============================================================"
echo " Stability diagnosis complete"
echo "============================================================"
echo
echo "No training was run."
echo "No source files were changed."
echo "No commit or push was performed."
echo
echo "Open the diagnostic with:"
echo
echo "code $OUTPUT_TXT"
