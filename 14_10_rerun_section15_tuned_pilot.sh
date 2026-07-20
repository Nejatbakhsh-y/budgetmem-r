#!/usr/bin/env bash
set -Eeuo pipefail

trap 'printf "\nERROR at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR

DEFAULT_REPO_ROOT="/mnt/c/Users/nejat/OneDrive/Desktop/UN/Skills/GitHub 2026/budgetmem-r"

if git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    REPO_ROOT="$git_root"
else
    REPO_ROOT="$DEFAULT_REPO_ROOT"
fi

if [[ ! -d "$REPO_ROOT/.git" ]]; then
    printf 'ERROR: Git repository not found: %s\n' "$REPO_ROOT" >&2
    exit 1
fi

cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
CONFIG="configs/experiments/pilot_tuned.yaml"
BASE_CONFIG="configs/experiments/pilot.yaml"
TUNING_DECISION="reports/evidence/pilot_tuning/performance_tuning_decision.json"
TUNING_LEADERBOARD="reports/tables/pilot_tuning/leaderboard.csv"
RUN_LOG="reports/evidence/section14_10_tuned_pilot_terminal.log"

required_paths=(
    "$PYTHON"
    "$BASE_CONFIG"
    "$TUNING_DECISION"
    "$TUNING_LEADERBOARD"
    "scripts/run_pilot.py"
    "src/budgetmem/experiments/pilot.py"
    "src/budgetmem/models/budgetmem_r.py"
    "tests/pilot"
)

for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
        printf 'ERROR: Required path is missing: %s\n' "$path" >&2
        exit 1
    fi
done

export PYTHONPATH="$REPO_ROOT/src"
export PYTHONHASHSEED=2026
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

timestamp="$(date +%Y%m%d_%H%M%S)"
backup_root="$REPO_ROOT/.section15_backup/section14_10_rerun_$timestamp"

mkdir -p \
    "$backup_root/configs/experiments" \
    "$backup_root/reports/evidence/pilot_tuning" \
    "$backup_root/reports/tables/pilot_tuning" \
    "$backup_root/reports/evidence" \
    "$backup_root/reports/tables" \
    reports/evidence \
    reports/tables \
    configs/experiments

printf '\n============================================================\n'
printf '14.10 - RERUN SECTION 15 WITH IMPROVED CONFIGURATION\n'
printf '============================================================\n\n'

printf '1. Preserving the existing tuning evidence and artifacts.\n'

for path in \
    "$CONFIG" \
    "$TUNING_DECISION" \
    "$TUNING_LEADERBOARD" \
    reports/tables/pilot_tuned_results.csv \
    reports/evidence/pilot_tuned_summary.json \
    reports/evidence/pilot_tuned_go_no_go.json \
    reports/pilot_tuned_report.md \
    "$RUN_LOG"; do

    if [[ -f "$path" ]]; then
        mkdir -p "$backup_root/$(dirname "$path")"
        cp -f "$path" "$backup_root/$path"
    fi
done

if [[ -d outputs/pilot_tuned ]]; then
    mkdir -p "$backup_root/outputs"
    mv outputs/pilot_tuned "$backup_root/outputs/pilot_tuned"
fi

printf 'Backup directory:\n%s\n\n' "$backup_root"

printf '2. Preparing and validating pilot_tuned.yaml.\n'

"$PYTHON" - <<'PY'
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

base_path = Path("configs/experiments/pilot.yaml")
config_path = Path("configs/experiments/pilot_tuned.yaml")
decision_path = Path(
    "reports/evidence/pilot_tuning/performance_tuning_decision.json"
)


def deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value


decision = json.loads(decision_path.read_text(encoding="utf-8"))

if not config_path.exists():
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    selected = decision["selected_candidate"]["settings"]

    config = copy.deepcopy(base)
    deep_update(config["training"], selected["training"])
    deep_update(config["model"], selected["model"])

    config["experiment_name"] = "section15_pilot_tuned"
    config["matrix"]["tasks"] = [
        "selective_copy",
        "associative_recall",
        "distractor_heavy_retrieval",
    ]
    config["matrix"]["evaluation_sequence_lengths"] = [256, 512, 1024]
    config["matrix"]["memory_budgets"] = [16, 32]
    config["matrix"]["models"] = [
        "gru",
        "gru_uniform_cache",
        "gru_reservoir_cache",
        "budgetmem_r",
    ]
    config["artifacts"] = {
        "output_root": "outputs/pilot_tuned",
        "results_csv": "reports/tables/pilot_tuned_results.csv",
        "summary_json": "reports/evidence/pilot_tuned_summary.json",
        "gate_json": "reports/evidence/pilot_tuned_go_no_go.json",
        "report_markdown": "reports/pilot_tuned_report.md",
        "checkpoint_root": "outputs/pilot_tuned/checkpoints",
    }

    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    print("Recreated missing pilot_tuned.yaml from the tuning decision.")

config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
matrix = config["matrix"]

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

checks = {
    "tasks": set(matrix["tasks"]) == expected_tasks,
    "sequence_lengths": {
        int(value)
        for value in matrix["evaluation_sequence_lengths"]
    }
    == expected_lengths,
    "memory_budgets": {
        int(value) for value in matrix["memory_budgets"]
    }
    == expected_budgets,
    "models": set(matrix["models"]) == expected_models,
}

print("Tuning screen status:", decision.get("status", "UNKNOWN"))
print(
    "Selected candidate:",
    decision.get("selected_candidate", {}).get("candidate", "UNKNOWN"),
)

for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")

if not all(checks.values()):
    raise SystemExit(
        "pilot_tuned.yaml does not contain the required Section 15 matrix."
    )

print("Improved configuration validation passed.")
PY

printf '\n3. Ensuring runtime outputs remain excluded from Git.\n'

gitignore_marker="# Section 15 tuned-pilot runtime artifacts"

touch .gitignore

if ! grep -Fq "$gitignore_marker" .gitignore; then
    cat >> .gitignore <<'GITIGNORE'

# Section 15 tuned-pilot runtime artifacts
outputs/pilot_tuned/
GITIGNORE
fi

printf '\n4. Compiling and testing the Section 15 implementation.\n'

"$PYTHON" -m compileall -q \
    src/budgetmem/experiments \
    src/budgetmem/models/budgetmem_r.py \
    scripts/run_pilot.py \
    tests/pilot

if "$PYTHON" -c \
    "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('ruff') else 1)"
then
    "$PYTHON" -m ruff check \
        src/budgetmem/experiments \
        src/budgetmem/models/budgetmem_r.py \
        scripts/run_pilot.py \
        tests/pilot
else
    printf 'WARNING: Ruff is unavailable; Ruff validation was skipped.\n'
fi

"$PYTHON" -m pytest tests/pilot -q

printf '\n5. Removing only the previous tuned-pilot runtime artifacts.\n'

rm -rf outputs/pilot_tuned

rm -f \
    reports/tables/pilot_tuned_results.csv \
    reports/evidence/pilot_tuned_summary.json \
    reports/evidence/pilot_tuned_go_no_go.json \
    reports/pilot_tuned_report.md \
    "$RUN_LOG"

printf '\n6. Running the complete four-model tuned Section 15 pilot.\n'
printf 'This is a CPU-intensive run. Leave VS Code and this terminal open.\n\n'

set +e

"$PYTHON" scripts/run_pilot.py \
    --config "$CONFIG" \
    2>&1 | tee "$RUN_LOG"

pilot_return_code="${PIPESTATUS[0]}"

set -e

if [[ "$pilot_return_code" -ne 0 ]]; then
    printf '\nERROR: The tuned pilot process returned code %s.\n' \
        "$pilot_return_code" >&2
    printf 'Inspect: %s\n' "$RUN_LOG" >&2
    exit "$pilot_return_code"
fi

printf '\n7. Validating the complete result matrix and final evidence.\n'

"$PYTHON" - <<'PY'
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

config_path = Path("configs/experiments/pilot_tuned.yaml")
decision_path = Path(
    "reports/evidence/pilot_tuning/performance_tuning_decision.json"
)
results_path = Path("reports/tables/pilot_tuned_results.csv")
summary_path = Path("reports/evidence/pilot_tuned_summary.json")
gate_path = Path("reports/evidence/pilot_tuned_go_no_go.json")
report_path = Path("reports/pilot_tuned_report.md")
log_path = Path("reports/evidence/section14_10_tuned_pilot_terminal.log")

required_artifacts = [
    config_path,
    decision_path,
    results_path,
    summary_path,
    gate_path,
    report_path,
    log_path,
]

missing_artifacts = [
    str(path)
    for path in required_artifacts
    if not path.exists() or path.stat().st_size == 0
]

if missing_artifacts:
    raise SystemExit(
        "Missing or empty required artifacts:\n"
        + "\n".join(missing_artifacts)
    )

config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
decision = json.loads(decision_path.read_text(encoding="utf-8"))
gate = json.loads(gate_path.read_text(encoding="utf-8"))
results = pd.read_csv(results_path)

required_columns = {
    "task",
    "model",
    "sequence_length",
    "memory_budget",
    "token_accuracy",
    "budget_pass",
    "resource_measurement_pass",
}

missing_columns = required_columns.difference(results.columns)

if missing_columns:
    raise SystemExit(
        f"Results CSV is missing required columns: {sorted(missing_columns)}"
    )

tasks = [
    "selective_copy",
    "associative_recall",
    "distractor_heavy_retrieval",
]
models = [
    "gru",
    "gru_uniform_cache",
    "gru_reservoir_cache",
    "budgetmem_r",
]
lengths = [256, 512, 1024]
budgets = [16, 32]

expected = set(
    itertools.product(tasks, models, lengths, budgets)
)

observed = {
    (
        str(row.task),
        str(row.model),
        int(row.sequence_length),
        int(row.memory_budget),
    )
    for row in results.itertuples(index=False)
}

missing_combinations = sorted(expected.difference(observed))
extra_combinations = sorted(observed.difference(expected))
matrix_complete = (
    not missing_combinations
    and not extra_combinations
    and len(results) == len(expected)
)

if not matrix_complete:
    print("Missing combinations:", missing_combinations)
    print("Extra combinations:", extra_combinations)
    print("Expected rows:", len(expected))
    print("Observed rows:", len(results))
    raise SystemExit("The complete Section 15 matrix was not produced.")

config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
pilot_gate_status = str(gate.get("status", "UNKNOWN"))

completion = {
    "status": "COMPLETE",
    "requirement": (
        "Rerun the Section 15 pilot with the improved configuration."
    ),
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "configuration": str(config_path),
    "configuration_sha256": config_sha256,
    "selected_candidate": decision.get(
        "selected_candidate", {}
    ).get("candidate"),
    "tuning_screen_status": decision.get("status"),
    "pilot_gate_status": pilot_gate_status,
    "pilot_decision": gate.get("decision"),
    "expected_matrix_rows": len(expected),
    "observed_matrix_rows": len(results),
    "matrix_complete": matrix_complete,
    "all_budget_checks_passed": bool(results["budget_pass"].all()),
    "all_resource_checks_passed": bool(
        results["resource_measurement_pass"].all()
    ),
    "results_csv": str(results_path),
    "summary_json": str(summary_path),
    "gate_json": str(gate_path),
    "report_markdown": str(report_path),
    "terminal_log": str(log_path),
    "interpretation": (
        "The Section 14.10 rerun requirement is complete. "
        "The separate research GO/NO_GO decision is preserved exactly "
        "as reported by the pilot gate."
    ),
}

completion_json = Path(
    "reports/evidence/section14_10_rerun_completion.json"
)
completion_json.write_text(
    json.dumps(completion, indent=2) + "\n",
    encoding="utf-8",
)

completion_md = Path(
    "reports/evidence/section14_10_rerun_completion.md"
)
completion_md.write_text(
    "\n".join(
        [
            "# Section 14.10 Completion",
            "",
            "**Requirement:** Rerun the Section 15 pilot with the improved configuration.",
            "",
            "**Rerun status:** COMPLETE",
            f"**Tuning screen status:** {decision.get('status', 'UNKNOWN')}",
            f"**Final pilot gate:** {pilot_gate_status}",
            f"**Matrix rows:** {len(results)} of {len(expected)}",
            f"**All budget checks passed:** {bool(results['budget_pass'].all())}",
            (
                "**All resource checks passed:** "
                f"{bool(results['resource_measurement_pass'].all())}"
            ),
            "",
            (
                "The rerun requirement is complete independently of whether "
                "the final research decision is GO or NO_GO."
            ),
            "",
        ]
    ),
    encoding="utf-8",
)

print()
print("============================================================")
print("SECTION 14.10 VALIDATION")
print("============================================================")
print("RERUN REQUIREMENT: COMPLETE")
print("RESULT MATRIX:", f"{len(results)}/{len(expected)} rows")
print("TUNING SCREEN:", decision.get("status", "UNKNOWN"))
print("FINAL PILOT GATE:", pilot_gate_status)
print("BUDGET CHECKS:", bool(results["budget_pass"].all()))
print(
    "RESOURCE CHECKS:",
    bool(results["resource_measurement_pass"].all()),
)
print("COMPLETION EVIDENCE:", completion_json)
print("============================================================")
PY

printf '\n8. Checking repository integrity and final status.\n'

git diff --check
git status --short

printf '\n============================================================\n'
printf 'SECTION 14.10 AUTOMATION FINISHED\n'
printf 'RERUN REQUIREMENT: COMPLETE\n'
printf 'Completion evidence:\n'
printf '  reports/evidence/section14_10_rerun_completion.json\n'
printf '  reports/evidence/section14_10_rerun_completion.md\n'
printf 'Final research decision:\n'
printf '  reports/evidence/pilot_tuned_go_no_go.json\n'
printf 'Terminal log:\n'
printf '  reports/evidence/section14_10_tuned_pilot_terminal.log\n'
printf 'Backup:\n'
printf '  %s\n' "$backup_root"
printf '============================================================\n'
