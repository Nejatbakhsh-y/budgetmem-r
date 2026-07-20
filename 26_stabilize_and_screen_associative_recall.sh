#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"

SCREEN_CONFIG="configs/experiments/pilot_assoc_core_repair_screen.yaml"
SCREEN_RUNNER="scripts/run_assoc_core_repair_screen.py"
STABILITY_DIAGNOSTIC="reports/evidence/assoc_core_repair_stability_diagnostic.json"

CANDIDATE_DIR="configs/experiments/assoc_stability_candidates"
RESULT_DIR="reports/tables/assoc_stability_candidates"
EVIDENCE_DIR="reports/evidence/assoc_stability_candidates"
LOG_DIR="reports/logs/assoc_stability_candidates"

GENERIC_RESULTS="reports/tables/assoc_core_repair_screen_results.csv"
GENERIC_SUMMARY="reports/evidence/assoc_core_repair_screen_summary.json"
GENERIC_DECISION_JSON="reports/evidence/assoc_core_repair_screen_decision.json"
GENERIC_DECISION_TXT="reports/evidence/assoc_core_repair_screen_decision.txt"

SELECTED_CONFIG="configs/experiments/pilot_assoc_stable.yaml"
SELECTED_RESULTS="reports/tables/assoc_stability_selected_results.csv"
SELECTED_DECISION_JSON="reports/evidence/assoc_stability_selected_decision.json"
SELECTED_DECISION_TXT="reports/evidence/assoc_stability_selected_decision.txt"
SWEEP_SUMMARY="reports/evidence/assoc_stability_sweep_summary.txt"

echo "============================================================"
echo " Section 15 Associative-Recall Stability Sweep"
echo "============================================================"
echo "Repository: $ROOT"
echo

for required in \
    "$PYTHON" \
    "$SCREEN_CONFIG" \
    "$SCREEN_RUNNER" \
    "$STABILITY_DIAGNOSTIC"; do

    if [[ ! -e "$required" ]]; then
        echo "ERROR: Missing required path:"
        echo "  $required"
        exit 1
    fi
done

export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p \
    "$CANDIDATE_DIR" \
    "$RESULT_DIR" \
    "$EVIDENCE_DIR" \
    "$LOG_DIR"

BACKUP_CONFIG="$(mktemp)"

cp -f "$SCREEN_CONFIG" "$BACKUP_CONFIG"

restore_original_config() {
    if [[ -f "$BACKUP_CONFIG" ]]; then
        cp -f "$BACKUP_CONFIG" "$SCREEN_CONFIG"
        rm -f "$BACKUP_CONFIG"
    fi
}

trap restore_original_config EXIT

echo "Confirming the recorded failure."

"$PYTHON" - <<'PY'
import json
from pathlib import Path

path = Path(
    "reports/evidence/"
    "assoc_core_repair_stability_diagnostic.json"
)

payload = json.loads(
    path.read_text(encoding="utf-8")
)

diagnosis = payload.get("diagnosis")
performance_pass = payload.get(
    "performance_requirement_pass"
)

print(f"Diagnosis: {diagnosis}")
print(
    "Recall performance requirement: "
    f"{'PASS' if performance_pass else 'FAIL'}"
)

if diagnosis != "GRADIENT_NORM_FAILURE":
    raise SystemExit(
        "ERROR: This automation is only for "
        "GRADIENT_NORM_FAILURE."
    )

if performance_pass is not True:
    raise SystemExit(
        "ERROR: The two-policy recall requirement "
        "has not passed."
    )
PY

echo
echo "Creating isolated stabilization candidates."

"$PYTHON" - <<'PY'
from __future__ import annotations

import copy
from pathlib import Path

import yaml


source = Path(
    "configs/experiments/"
    "pilot_assoc_core_repair_screen.yaml"
)

destination = Path(
    "configs/experiments/"
    "assoc_stability_candidates"
)

base = yaml.safe_load(
    source.read_text(encoding="utf-8")
)

destination.mkdir(
    parents=True,
    exist_ok=True,
)

original_threshold = float(
    base["training"][
        "maximum_acceptable_gradient_norm"
    ]
)

candidates = [
    {
        "name": "stable_128_lr250",
        "train_sequence_length": 128,
        "epochs": 16,
        "learning_rate": 0.00025,
        "gradient_clip_norm": 0.5,
    },
    {
        "name": "stable_128_lr125",
        "train_sequence_length": 128,
        "epochs": 20,
        "learning_rate": 0.000125,
        "gradient_clip_norm": 0.5,
    },
    {
        "name": "stable_96_lr100",
        "train_sequence_length": 96,
        "epochs": 24,
        "learning_rate": 0.0001,
        "gradient_clip_norm": 0.25,
    },
]

for candidate in candidates:
    cfg = copy.deepcopy(base)
    name = candidate["name"]

    cfg["experiment_name"] = (
        f"section15_assoc_{name}"
    )

    cfg["matrix"]["train_sequence_length"] = (
        candidate["train_sequence_length"]
    )

    cfg["training"]["epochs"] = (
        candidate["epochs"]
    )

    cfg["training"]["learning_rate"] = (
        candidate["learning_rate"]
    )

    cfg["training"]["gradient_clip_norm"] = (
        candidate["gradient_clip_norm"]
    )

    # The acceptance threshold is deliberately unchanged.
    cfg["training"][
        "maximum_acceptable_gradient_norm"
    ] = original_threshold

    cfg["artifacts"] = {
        "output_root": (
            f"outputs/assoc_stability_candidates/{name}"
        ),
        "results_csv": (
            f"reports/tables/"
            f"assoc_stability_candidates/{name}_results.csv"
        ),
        "summary_json": (
            f"reports/evidence/"
            f"assoc_stability_candidates/{name}_summary.json"
        ),
        "gate_json": (
            f"reports/evidence/"
            f"assoc_stability_candidates/{name}_decision.json"
        ),
        "report_markdown": (
            f"reports/"
            f"assoc_stability_candidates/{name}_report.md"
        ),
        "checkpoint_root": (
            f"outputs/assoc_stability_candidates/"
            f"{name}/checkpoints"
        ),
    }

    path = destination / f"{name}.yaml"

    path.write_text(
        yaml.safe_dump(
            cfg,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Created {path}: "
        f"train_length={candidate['train_sequence_length']}, "
        f"epochs={candidate['epochs']}, "
        f"lr={candidate['learning_rate']}, "
        f"clip={candidate['gradient_clip_norm']}, "
        f"maximum_allowed_gradient={original_threshold}"
    )
PY

rm -f "$SWEEP_SUMMARY"

{
    echo "SECTION 15 ASSOCIATIVE-RECALL STABILITY SWEEP"
    echo "============================================="
    echo
    echo "The maximum acceptable gradient threshold was not changed."
    echo
} > "$SWEEP_SUMMARY"

selected_candidate=""

for candidate_config in \
    "$CANDIDATE_DIR/stable_128_lr250.yaml" \
    "$CANDIDATE_DIR/stable_128_lr125.yaml" \
    "$CANDIDATE_DIR/stable_96_lr100.yaml"; do

    candidate="$(
        basename "$candidate_config" .yaml
    )"

    candidate_results="$RESULT_DIR/${candidate}_results.csv"
    candidate_summary="$EVIDENCE_DIR/${candidate}_summary.json"
    candidate_decision_json="$EVIDENCE_DIR/${candidate}_decision.json"
    candidate_decision_txt="$EVIDENCE_DIR/${candidate}_decision.txt"
    candidate_log="$LOG_DIR/${candidate}.log"

    echo
    echo "============================================================"
    echo " Running candidate: $candidate"
    echo "============================================================"

    cp -f "$candidate_config" "$SCREEN_CONFIG"

    rm -rf \
        "outputs/assoc_core_repair_screen" \
        "outputs/assoc_stability_candidates/$candidate"

    rm -f \
        "$GENERIC_RESULTS" \
        "$GENERIC_SUMMARY" \
        "$GENERIC_DECISION_JSON" \
        "$GENERIC_DECISION_TXT" \
        "$candidate_results" \
        "$candidate_summary" \
        "$candidate_decision_json" \
        "$candidate_decision_txt" \
        "$candidate_log"

    set +e

    "$PYTHON" "$SCREEN_RUNNER" \
        2>&1 | tee "$candidate_log"

    candidate_status="${PIPESTATUS[0]}"

    set -e

    [[ -f "$GENERIC_RESULTS" ]] && \
        cp -f "$GENERIC_RESULTS" "$candidate_results"

    [[ -f "$GENERIC_SUMMARY" ]] && \
        cp -f "$GENERIC_SUMMARY" "$candidate_summary"

    [[ -f "$GENERIC_DECISION_JSON" ]] && \
        cp -f "$GENERIC_DECISION_JSON" "$candidate_decision_json"

    [[ -f "$GENERIC_DECISION_TXT" ]] && \
        cp -f "$GENERIC_DECISION_TXT" "$candidate_decision_txt"

    if [[ ! -f "$candidate_decision_json" ]]; then
        echo
        echo "Candidate did not produce a decision file."

        {
            echo
            echo "$candidate"
            echo "  runner_status: $candidate_status"
            echo "  decision: ERROR"
        } >> "$SWEEP_SUMMARY"

        continue
    fi

    echo
    echo "Candidate result:"

    CANDIDATE_JSON="$candidate_decision_json" \
    "$PYTHON" - <<'PY' | tee -a "$SWEEP_SUMMARY"
import json
import os
from pathlib import Path

path = Path(os.environ["CANDIDATE_JSON"])

payload = json.loads(
    path.read_text(encoding="utf-8")
)

criteria = payload.get("criteria", {})

print()
print(path.stem)
print(f"  decision: {payload.get('decision')}")
print(
    "  qualified_policies: "
    f"{payload.get('qualified_policy_count')}/2"
)
print(
    "  stability_pass: "
    f"{criteria.get('stability_pass')}"
)
print(
    "  budget_pass: "
    f"{criteria.get('budget_pass')}"
)
print(
    "  resource_pass: "
    f"{criteria.get('resource_pass')}"
)
print(
    "  write_frequency_pass: "
    f"{criteria.get('write_frequency_pass')}"
)
print(
    "  write_frequency: "
    f"{criteria.get('write_frequency')}"
)
PY

    decision="$(
        CANDIDATE_JSON="$candidate_decision_json" \
        "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(
    Path(
        os.environ["CANDIDATE_JSON"]
    ).read_text(encoding="utf-8")
)

print(payload.get("decision", "UNKNOWN"))
PY
    )"

    if [[ "$candidate_status" -eq 0 ]] \
        && [[ "$decision" == "TARGETED_GO" ]]; then

        selected_candidate="$candidate"
        break
    fi
done

restore_original_config
trap - EXIT

echo
echo "============================================================"
echo " Stability Sweep Result"
echo "============================================================"

if [[ -z "$selected_candidate" ]]; then
    echo
    echo "TARGETED DECISION: NO-GO"
    echo
    echo "None of the stabilization candidates passed both:"
    echo "- the unchanged gradient-norm stability gate; and"
    echo "- the two-policy long-range recall gate."
    echo
    echo "The complete pilot was not started."
    echo "No commit or push was performed."
    echo
    echo "Review:"
    echo "  $SWEEP_SUMMARY"
    echo "  $EVIDENCE_DIR"
    echo "  $LOG_DIR"
    echo
    exit 2
fi

echo
echo "TARGETED DECISION: GO"
echo "Selected candidate: $selected_candidate"

cp -f \
    "$CANDIDATE_DIR/${selected_candidate}.yaml" \
    "$SELECTED_CONFIG"

cp -f \
    "$RESULT_DIR/${selected_candidate}_results.csv" \
    "$SELECTED_RESULTS"

cp -f \
    "$EVIDENCE_DIR/${selected_candidate}_decision.json" \
    "$SELECTED_DECISION_JSON"

cp -f \
    "$EVIDENCE_DIR/${selected_candidate}_decision.txt" \
    "$SELECTED_DECISION_TXT"

{
    echo
    echo "Selected candidate: $selected_candidate"
    echo "Selected configuration: $SELECTED_CONFIG"
    echo "Selected results: $SELECTED_RESULTS"
    echo "Selected decision: $SELECTED_DECISION_TXT"
} >> "$SWEEP_SUMMARY"

echo
echo "The targeted associative-recall screen now passes:"
echo "- stability;"
echo "- strict memory-budget enforcement;"
echo "- resource measurement;"
echo "- nontrivial writing;"
echo "- recall advantage over uniform cache;"
echo "- recall advantage over reservoir cache."
echo
echo "Saved selected configuration:"
echo "  $SELECTED_CONFIG"
echo
echo "No full pilot was started."
echo "No commit or push was performed."
echo
echo "Review:"
echo "  $SELECTED_DECISION_TXT"
echo "  $SWEEP_SUMMARY"
