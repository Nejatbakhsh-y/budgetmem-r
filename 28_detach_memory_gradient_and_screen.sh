#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"

PILOT_SOURCE="src/budgetmem/experiments/pilot.py"
BASE_CONFIG="configs/experiments/assoc_stability_candidates/stable_96_lr100.yaml"
SCREEN_CONFIG="configs/experiments/pilot_assoc_core_repair_screen.yaml"
CANDIDATE_CONFIG="configs/experiments/pilot_assoc_detached_memory.yaml"
SCREEN_RUNNER="scripts/run_assoc_core_repair_screen.py"

GENERIC_RESULTS="reports/tables/assoc_core_repair_screen_results.csv"
GENERIC_SUMMARY="reports/evidence/assoc_core_repair_screen_summary.json"
GENERIC_DECISION_JSON="reports/evidence/assoc_core_repair_screen_decision.json"
GENERIC_DECISION_TXT="reports/evidence/assoc_core_repair_screen_decision.txt"

FINAL_RESULTS="reports/tables/assoc_detached_memory_results.csv"
FINAL_SUMMARY="reports/evidence/assoc_detached_memory_summary.json"
FINAL_DECISION_JSON="reports/evidence/assoc_detached_memory_decision.json"
FINAL_DECISION_TXT="reports/evidence/assoc_detached_memory_decision.txt"
FINAL_DIAGNOSTIC="reports/evidence/assoc_detached_memory_diagnostic.txt"
LOG_FILE="reports/logs/assoc_detached_memory_screen.log"

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP=".section15_backup/detached_memory_${STAMP}"

echo "============================================================"
echo " Section 15 Detached-Memory Gradient Repair"
echo "============================================================"
echo "Repository: $ROOT"
echo

for required in \
    "$PYTHON" \
    "$PILOT_SOURCE" \
    "$BASE_CONFIG" \
    "$SCREEN_CONFIG" \
    "$SCREEN_RUNNER"; do

    if [[ ! -e "$required" ]]; then
        echo "ERROR: Missing required path:"
        echo "  $required"
        exit 1
    fi
done

export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p \
    "$BACKUP" \
    reports/tables \
    reports/evidence \
    reports/logs \
    configs/experiments

cp -f "$PILOT_SOURCE" "$BACKUP/pilot.py"
cp -f "$SCREEN_CONFIG" "$BACKUP/pilot_assoc_core_repair_screen.yaml"

for artifact in \
    "$GENERIC_RESULTS" \
    "$GENERIC_SUMMARY" \
    "$GENERIC_DECISION_JSON" \
    "$GENERIC_DECISION_TXT"; do

    if [[ -f "$artifact" ]]; then
        mkdir -p "$BACKUP/$(dirname "$artifact")"
        cp -f "$artifact" "$BACKUP/$artifact"
    fi
done

git status --short > "$BACKUP/git_status_before.txt"
git diff > "$BACKUP/git_diff_before.patch"

echo "Backup created:"
echo "  $BACKUP"
echo

restore_screen_config() {
    if [[ -f "$BACKUP/pilot_assoc_core_repair_screen.yaml" ]]; then
        cp -f \
            "$BACKUP/pilot_assoc_core_repair_screen.yaml" \
            "$SCREEN_CONFIG"
    fi
}

trap restore_screen_config EXIT

# ============================================================
# PATCH THE PILOT ADAPTER
# ============================================================

echo "Patching the BudgetMem-R pilot adapter."

"$PYTHON" - <<'PY'
from pathlib import Path

path = Path("src/budgetmem/experiments/pilot.py")
text = path.read_text(encoding="utf-8")

new_entry = (
    '"detach_memory_writes": bool('
    'model_cfg.get("detach_memory_writes", False)'
    '),'
)

if new_entry in text:
    print("Adapter already supports detach_memory_writes.")

else:
    old = '''            "detach_memory": False,
'''

    new = '''            "detach_memory": bool(
                model_cfg.get("detach_memory", False)
            ),
            "detach_memory_writes": bool(
                model_cfg.get("detach_memory_writes", False)
            ),
'''

    count = text.count(old)

    if count != 1:
        raise SystemExit(
            "ERROR: Could not locate the adapter detach-memory entry. "
            f"Occurrences found: {count}"
        )

    text = text.replace(old, new, 1)

    path.write_text(
        text,
        encoding="utf-8",
    )

    print(f"Patched: {path}")
PY

# ============================================================
# CREATE THE ISOLATED CANDIDATE
# ============================================================

echo
echo "Creating the detached-memory candidate configuration."

"$PYTHON" - <<'PY'
from pathlib import Path

import yaml


source = Path(
    "configs/experiments/"
    "assoc_stability_candidates/"
    "stable_96_lr100.yaml"
)

destination = Path(
    "configs/experiments/"
    "pilot_assoc_detached_memory.yaml"
)

cfg = yaml.safe_load(
    source.read_text(encoding="utf-8")
)

cfg["experiment_name"] = (
    "section15_assoc_detached_memory"
)

model_cfg = cfg.setdefault("model", {})
model_cfg["detach_memory_writes"] = True

# Preserve the closest successful recall candidate.
cfg["matrix"]["train_sequence_length"] = 96
cfg["training"]["epochs"] = 24
cfg["training"]["learning_rate"] = 0.0001
cfg["training"]["gradient_clip_norm"] = 0.25

# The acceptance threshold remains unchanged.
cfg["training"][
    "maximum_acceptable_gradient_norm"
] = 100.0

cfg["artifacts"] = {
    "output_root": (
        "outputs/assoc_detached_memory"
    ),
    "results_csv": (
        "reports/tables/"
        "assoc_detached_memory_results.csv"
    ),
    "summary_json": (
        "reports/evidence/"
        "assoc_detached_memory_summary.json"
    ),
    "gate_json": (
        "reports/evidence/"
        "assoc_detached_memory_decision.json"
    ),
    "report_markdown": (
        "reports/"
        "assoc_detached_memory_report.md"
    ),
    "checkpoint_root": (
        "outputs/assoc_detached_memory/checkpoints"
    ),
}

destination.write_text(
    yaml.safe_dump(
        cfg,
        sort_keys=False,
    ),
    encoding="utf-8",
)

print(f"Created: {destination}")
print(
    "detach_memory_writes: "
    f"{cfg['model']['detach_memory_writes']}"
)
print(
    "train_sequence_length: "
    f"{cfg['matrix']['train_sequence_length']}"
)
print(
    "learning_rate: "
    f"{cfg['training']['learning_rate']}"
)
print(
    "gradient_clip_norm: "
    f"{cfg['training']['gradient_clip_norm']}"
)
print(
    "maximum_acceptable_gradient_norm: "
    f"{cfg['training']['maximum_acceptable_gradient_norm']}"
)
PY

# ============================================================
# VERIFY THAT THE OPTION REACHES THE CORE MODEL
# ============================================================

echo
echo "Verifying the repaired constructor path."

"$PYTHON" - <<'PY'
from pathlib import Path

import yaml

from budgetmem.experiments.pilot import BudgetMemRAdapter


path = Path(
    "configs/experiments/"
    "pilot_assoc_detached_memory.yaml"
)

cfg = yaml.safe_load(
    path.read_text(encoding="utf-8")
)

adapter = BudgetMemRAdapter(cfg)

actual = bool(
    adapter.core.detach_memory_writes
)

print(
    "Core detach_memory_writes: "
    f"{actual}"
)

if actual is not True:
    raise SystemExit(
        "ERROR: detach_memory_writes did not reach "
        "the BudgetMem-R core."
    )
PY

# ============================================================
# ADD A REGRESSION TEST
# ============================================================

cat > tests/pilot/test_detached_memory_adapter.py <<'PY'
"""Regression test for the Section 15 detached-memory repair."""

from __future__ import annotations

from pathlib import Path

import yaml

from budgetmem.experiments.pilot import BudgetMemRAdapter


def test_pilot_adapter_passes_detach_memory_writes() -> None:
    config_path = Path(
        "configs/experiments/"
        "pilot_assoc_detached_memory.yaml"
    )

    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    adapter = BudgetMemRAdapter(config)

    assert adapter.core.detach_memory_writes is True
PY

echo
echo "Running focused regression tests."

"$PYTHON" -m pytest \
    tests/pilot/test_detached_memory_adapter.py \
    tests/models/test_budgetmem_r_straight_through_repair.py \
    tests/tasks/test_pilot_oracle_alignment.py \
    tests/pilot/test_controller_calibration.py \
    -q

echo "Regression tests: PASS"

# ============================================================
# RUN THE TARGETED SCREEN
# ============================================================

cp -f "$CANDIDATE_CONFIG" "$SCREEN_CONFIG"

rm -rf \
    outputs/assoc_core_repair_screen \
    outputs/assoc_detached_memory

rm -f \
    "$GENERIC_RESULTS" \
    "$GENERIC_SUMMARY" \
    "$GENERIC_DECISION_JSON" \
    "$GENERIC_DECISION_TXT" \
    "$FINAL_RESULTS" \
    "$FINAL_SUMMARY" \
    "$FINAL_DECISION_JSON" \
    "$FINAL_DECISION_TXT" \
    "$FINAL_DIAGNOSTIC" \
    "$LOG_FILE"

echo
echo "============================================================"
echo " Running Detached-Memory Associative-Recall Screen"
echo "============================================================"
echo
echo "Evaluation task:      associative_recall"
echo "Evaluation length:    1024"
echo "Budgets:              16 and 32"
echo "Gradient threshold:   unchanged at 100.0"
echo "Full pilot:           not started"
echo

set +e

"$PYTHON" "$SCREEN_RUNNER" \
    2>&1 | tee "$LOG_FILE"

RUN_STATUS="${PIPESTATUS[0]}"

set -e

restore_screen_config
trap - EXIT

for pair in \
    "$GENERIC_RESULTS:$FINAL_RESULTS" \
    "$GENERIC_SUMMARY:$FINAL_SUMMARY" \
    "$GENERIC_DECISION_JSON:$FINAL_DECISION_JSON" \
    "$GENERIC_DECISION_TXT:$FINAL_DECISION_TXT"; do

    source="${pair%%:*}"
    destination="${pair#*:}"

    if [[ -f "$source" ]]; then
        cp -f "$source" "$destination"
    fi
done

if [[ ! -f "$FINAL_SUMMARY" ]] \
    || [[ ! -f "$FINAL_DECISION_JSON" ]]; then

    echo
    echo "ERROR: The targeted screen did not produce"
    echo "the required summary and decision files."
    echo
    echo "Inspect:"
    echo "  $LOG_FILE"
    exit 1
fi

# ============================================================
# PRODUCE THE FINAL DIAGNOSTIC
# ============================================================

SUMMARY_PATH="$FINAL_SUMMARY" \
DECISION_PATH="$FINAL_DECISION_JSON" \
CONFIG_PATH="$CANDIDATE_CONFIG" \
"$PYTHON" - <<'PY' | tee \
    reports/evidence/assoc_detached_memory_diagnostic.txt

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import yaml


summary_path = Path(os.environ["SUMMARY_PATH"])
decision_path = Path(os.environ["DECISION_PATH"])
config_path = Path(os.environ["CONFIG_PATH"])

summary = json.loads(
    summary_path.read_text(encoding="utf-8")
)

decision = json.loads(
    decision_path.read_text(encoding="utf-8")
)

config = yaml.safe_load(
    config_path.read_text(encoding="utf-8")
)

records = summary.get(
    "training_records",
    []
)

budgetmem_records = [
    record
    for record in records
    if record.get("model") == "budgetmem_r"
]

if not budgetmem_records:
    raise SystemExit(
        "ERROR: Missing BudgetMem-R training record."
    )

record = budgetmem_records[0]
criteria = decision.get("criteria", {})

maximum_gradient = float(
    record["maximum_gradient_norm"]
)

allowed_gradient = float(
    config["training"][
        "maximum_acceptable_gradient_norm"
    ]
)

qualified_policies = int(
    decision.get(
        "qualified_policy_count",
        0,
    )
)

policy_summary = pd.DataFrame(
    decision.get(
        "policy_summary",
        [],
    )
)

print("=" * 100)
print("DETACHED-MEMORY TARGETED SCREEN")
print("=" * 100)
print()

print(
    f"Decision:                    "
    f"{decision.get('decision')}"
)

print(
    f"Maximum gradient norm:       "
    f"{maximum_gradient:.6f}"
)

print(
    f"Allowed gradient norm:       "
    f"{allowed_gradient:.6f}"
)

print(
    f"Gradient margin:             "
    f"{allowed_gradient - maximum_gradient:+.6f}"
)

print(
    f"Stability pass:              "
    f"{criteria.get('stability_pass')}"
)

print(
    f"Qualified policies:          "
    f"{qualified_policies}/2"
)

print(
    f"Strict budget pass:          "
    f"{criteria.get('budget_pass')}"
)

print(
    f"Resource measurement pass:   "
    f"{criteria.get('resource_pass')}"
)

print(
    f"Write-frequency pass:        "
    f"{criteria.get('write_frequency_pass')}"
)

print(
    f"Mean write frequency:        "
    f"{criteria.get('write_frequency')}"
)

print(
    f"detach_memory_writes:        "
    f"{config['model']['detach_memory_writes']}"
)

print()

if not policy_summary.empty:
    print("Policy comparison:")
    print(
        policy_summary.to_string(
            index=False
        )
    )
    print()

all_pass = bool(
    decision.get("decision")
    == "TARGETED_GO"
    and criteria.get("stability_pass") is True
    and qualified_policies >= 2
    and criteria.get("budget_pass") is True
    and criteria.get("resource_pass") is True
    and criteria.get(
        "write_frequency_pass"
    ) is True
)

print("=" * 100)
print("FINAL TARGETED STATUS")
print("=" * 100)

if all_pass:
    print("TARGETED DECISION: GO")
    print()
    print(
        "The detached-memory gradient repair passed:"
    )
    print("- Gradient stability")
    print("- Recall over uniform cache")
    print("- Recall over reservoir cache")
    print("- Strict memory budget")
    print("- Resource measurement")
    print("- Nontrivial controller writing")
else:
    print("TARGETED DECISION: NO-GO")
    print()
    print(
        "The detached-memory repair did not satisfy "
        "all targeted conditions."
    )
    print(
        "Do not begin the complete pilot."
    )
PY

FINAL_STATUS="$(
    DECISION_PATH="$FINAL_DECISION_JSON" \
    "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(
    Path(
        os.environ["DECISION_PATH"]
    ).read_text(encoding="utf-8")
)

print(payload.get("decision", "UNKNOWN"))
PY
)"

echo
echo "============================================================"

if [[ "$RUN_STATUS" -eq 0 ]] \
    && [[ "$FINAL_STATUS" == "TARGETED_GO" ]]; then

    echo " TARGETED DECISION: GO"
    echo "============================================================"
    echo
    echo "The gradient-path repair passed the targeted gate."
    echo
    echo "Section 14.11 is not yet complete."
    echo "The next step will be one isolated full Section 15 pilot."
    echo
    echo "Do not commit or push yet."
    echo
    echo "Review:"
    echo "  $FINAL_DIAGNOSTIC"
    echo "  $FINAL_DECISION_TXT"
    echo "  $FINAL_RESULTS"
    exit 0
fi

echo " TARGETED DECISION: NO-GO"
echo "============================================================"
echo
echo "The complete pilot was not started."
echo "No commit or push was performed."
echo
echo "Review:"
echo "  $FINAL_DIAGNOSTIC"
echo "  $FINAL_DECISION_TXT"
echo "  $LOG_FILE"
echo
echo "Section 14.11 remains incomplete."
exit 2
