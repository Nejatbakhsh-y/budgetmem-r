#!/usr/bin/env bash
set -Eeuo pipefail

# Section 16.15.1 / Research Protocol Section 15
# Single Bash automation for the configuration-driven BudgetMem-R pilot.
# Intended environment: VS Code connected to WSL, CPU-only execution.
#
# Usage:
#   ./16_15_1_section15_pilot.sh full     # Configure, test, run, and gate
#   ./16_15_1_section15_pilot.sh resume   # Resume from existing checkpoints
#   ./16_15_1_section15_pilot.sh verify   # Verify existing result artifacts only
#   ./16_15_1_section15_pilot.sh smoke    # Infrastructure smoke run; not a GO decision

DEFAULT_REPO_ROOT="/mnt/c/Users/nejat/OneDrive/Desktop/UN/Skills/GitHub 2026/budgetmem-r"
MODE="${1:-full}"

case "$MODE" in
    full|resume|verify|smoke) ;;
    *)
        printf 'ERROR: Unsupported mode: %s\n' "$MODE" >&2
        printf 'Use: full, resume, verify, or smoke.\n' >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
elif git rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git rev-parse --show-toplevel)"
else
    REPO_ROOT="${BUDGETMEM_REPO_ROOT:-$DEFAULT_REPO_ROOT}"
fi

if [[ ! -d "$REPO_ROOT/.git" ]]; then
    printf 'ERROR: budgetmem-r repository not found: %s\n' "$REPO_ROOT" >&2
    printf 'Open the repository in VS Code/WSL or set BUDGETMEM_REPO_ROOT.\n' >&2
    exit 2
fi
cd "$REPO_ROOT"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    printf 'ERROR: Python was not found. Expected .venv/bin/python.\n' >&2
    exit 2
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="2026"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export TOKENIZERS_PARALLELISM="false"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="reports/evidence/logs"
BACKUP_DIR="reports/evidence/backups/section15_${STAMP}"
LOG_FILE="$LOG_DIR/section15_automation_${STAMP}.log"
CONFIG_FILE="configs/experiments/pilot.yaml"
RESULTS_FILE="reports/tables/pilot_results.csv"
SUMMARY_FILE="reports/evidence/pilot_summary.json"
RUNNER_GATE_FILE="reports/evidence/pilot_go_no_go.json"
REPORT_FILE="reports/pilot_report.md"
AUTOMATION_GATE_JSON="reports/evidence/section15_automation_gate.json"
AUTOMATION_GATE_TXT="reports/evidence/section15_automation_gate.txt"
COMPARISON_FILE="reports/tables/section15_long_range_comparison.csv"

mkdir -p "$LOG_DIR" "$BACKUP_DIR" reports/tables reports/evidence configs/experiments
exec > >(tee -a "$LOG_FILE") 2>&1

heading() {
    printf '\n============================================================\n'
    printf '%s\n' "$1"
    printf '============================================================\n'
}

fail() {
    local message="$1"
    local code="${2:-1}"
    printf '\nERROR: %s\n' "$message" >&2
    printf 'Log: %s\n' "$LOG_FILE" >&2
    exit "$code"
}

backup_if_present() {
    local path="$1"
    if [[ -e "$path" ]]; then
        mkdir -p "$BACKUP_DIR/$(dirname "$path")"
        cp -a "$path" "$BACKUP_DIR/$path"
    fi
}

heading "SECTION 15 — PILOT EXPERIMENT"
printf 'Repository: %s\n' "$REPO_ROOT"
printf 'Mode:       %s\n' "$MODE"
printf 'Python:     %s\n' "$PYTHON"
printf 'Log:        %s\n' "$LOG_FILE"
printf 'Backup:     %s\n' "$BACKUP_DIR"

heading "1. VERIFY SECTION 14 PRETRAINING GATE"
PRETRAINING_GATE="reports/evidence/pretraining_gate_report.json"
[[ -f "$PRETRAINING_GATE" ]] || fail "Missing $PRETRAINING_GATE. Complete Section 14 before training."

"$PYTHON" - "$PRETRAINING_GATE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
status = data.get("status") or data.get("overall_status") or data.get("gate_status")
if str(status).strip().upper() != "PASS":
    raise SystemExit(f"Pretraining gate is {status!r}; Section 15 must not start.")
print("Pretraining gate: PASS")
PY

heading "2. PREPARE THE CONTROLLED PILOT CONFIGURATION"
backup_if_present "$CONFIG_FILE"

"$PYTHON" - "$CONFIG_FILE" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
if path.exists():
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = loaded if isinstance(loaded, dict) else {}
else:
    cfg = {}

cfg["schema_version"] = "1.0"
cfg["experiment_name"] = "section15_pilot"
cfg["seed"] = 2026
cfg["device"] = "cpu"
cfg["pretraining_gate"] = {
    "report": "reports/evidence/pretraining_gate_report.json",
    "required_status": "PASS",
}

matrix = cfg.setdefault("matrix", {})
matrix["tasks"] = [
    "selective_copy",
    "associative_recall",
    "distractor_heavy_retrieval",
]
matrix["train_sequence_length"] = int(matrix.get("train_sequence_length", 256))
matrix["evaluation_sequence_lengths"] = [256, 512, 1024]
matrix["memory_budgets"] = [16, 32]
matrix["models"] = [
    "gru",
    "gru_uniform_cache",
    "gru_reservoir_cache",
    "budgetmem_r",
]

model = cfg.setdefault("model", {})
model.setdefault("vocabulary_size", 192)
model.setdefault("embedding_dim", 32)
model.setdefault("hidden_dim", 64)
model.setdefault("key_dim", 32)
model.setdefault("retrieval_k", 4)
model.setdefault("max_target_length", 12)
model.setdefault("dropout", 0.0)

training = cfg.setdefault("training", {})
training.setdefault("train_samples", 128)
training.setdefault("validation_samples", 48)
training.setdefault("batch_size", 8)
training.setdefault("epochs", 4)
training.setdefault("learning_rate", 0.001)
training.setdefault("weight_decay", 0.0001)
training.setdefault("gradient_clip_norm", 1.0)
training.setdefault("maximum_acceptable_gradient_norm", 100.0)
training.setdefault("budget_sampling", "alternating")
training.setdefault("write_rate_target", 0.15)
training.setdefault("write_rate_penalty", 0.05)
training.setdefault("budget_violation_penalty", 10.0)
training.setdefault("num_workers", 0)

evaluation = cfg.setdefault("evaluation", {})
evaluation.setdefault("batch_size", 8)
evaluation["deterministic"] = True
evaluation.setdefault("recent_overlap_failure_threshold", 0.90)
evaluation.setdefault("nontrivial_write_frequency_min", 0.01)
evaluation.setdefault("nontrivial_write_frequency_max", 0.95)
evaluation.setdefault("random_retention_margin", 0.01)
evaluation.setdefault("minimum_clear_accuracy_gain", 0.02)
evaluation["long_range_sequence_length"] = 1024
evaluation.setdefault("resource_sample_interval_seconds", 0.01)

cfg["artifacts"] = {
    "output_root": "outputs/pilot",
    "results_csv": "reports/tables/pilot_results.csv",
    "summary_json": "reports/evidence/pilot_summary.json",
    "gate_json": "reports/evidence/pilot_go_no_go.json",
    "report_markdown": "reports/pilot_report.md",
    "checkpoint_root": "outputs/pilot/checkpoints",
}

path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"WROTE {path}")
print("Matrix: 3 tasks × 3 lengths × 2 budgets × 4 models × 1 seed = 72 cells")
PY

heading "3. VERIFY PILOT IMPLEMENTATION"
required_files=(
    "scripts/run_pilot.py"
    "src/budgetmem/experiments/pilot.py"
    "src/budgetmem/models/budgetmem_r.py"
    "src/budgetmem/memory/budget_state.py"
    "scripts/data/generate_synthetic.py"
    "configs/data/synthetic.yaml"
)

missing=()
for path in "${required_files[@]}"; do
    [[ -f "$path" ]] || missing+=("$path")
done

if (( ${#missing[@]} > 0 )) && [[ -f "15_pilot_experiment.sh" ]]; then
    printf 'Core pilot files are missing. Running the established Section 15 installer.\n'
    bash 15_pilot_experiment.sh install-only
    missing=()
    for path in "${required_files[@]}"; do
        [[ -f "$path" ]] || missing+=("$path")
    done
fi

if (( ${#missing[@]} > 0 )); then
    printf 'Missing files:\n' >&2
    printf '  - %s\n' "${missing[@]}" >&2
    fail "The Section 15 pilot implementation is incomplete."
fi

"$PYTHON" -m py_compile scripts/run_pilot.py src/budgetmem/experiments/pilot.py
printf 'Pilot implementation syntax: PASS\n'

heading "4. RUN FOCUSED TESTS"
if [[ "$MODE" != "verify" ]]; then
    test_targets=()
    [[ -d tests/pilot ]] && test_targets+=("tests/pilot")
    [[ -f tests/test_memory_budget.py ]] && test_targets+=("tests/test_memory_budget.py")
    [[ -f tests/test_determinism.py ]] && test_targets+=("tests/test_determinism.py")

    if (( ${#test_targets[@]} == 0 )); then
        fail "No Section 15 or memory-budget tests were found."
    fi

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest \
        -q -o addopts='' "${test_targets[@]}"
else
    printf 'Verify mode: test execution skipped.\n'
fi

heading "5. RUN THE PILOT"
RUNNER_RC=0
if [[ "$MODE" == "full" || "$MODE" == "smoke" ]]; then
    for path in "$RESULTS_FILE" "$SUMMARY_FILE" "$RUNNER_GATE_FILE" "$REPORT_FILE"; do
        backup_if_present "$path"
    done
    backup_if_present "outputs/pilot"

    rm -rf outputs/pilot
    rm -f "$RESULTS_FILE" "$SUMMARY_FILE" "$RUNNER_GATE_FILE" "$REPORT_FILE"

    set +e
    if [[ "$MODE" == "smoke" ]]; then
        "$PYTHON" scripts/run_pilot.py --config "$CONFIG_FILE" --smoke
    else
        "$PYTHON" scripts/run_pilot.py --config "$CONFIG_FILE"
    fi
    RUNNER_RC=$?
    set -e
elif [[ "$MODE" == "resume" ]]; then
    set +e
    "$PYTHON" scripts/run_pilot.py --config "$CONFIG_FILE" --resume
    RUNNER_RC=$?
    set -e
else
    printf 'Verify mode: using existing pilot artifacts.\n'
fi

if [[ "$MODE" == "smoke" ]]; then
    printf '\nSMOKE RUN COMPLETED. This mode cannot produce the Section 15 research GO decision.\n'
    printf 'Runner exit code: %s\n' "$RUNNER_RC"
    printf 'Log: %s\n' "$LOG_FILE"
    exit "$RUNNER_RC"
fi

if [[ ! -f "$RESULTS_FILE" ]]; then
    fail "The pilot runner did not produce $RESULTS_FILE (runner exit code: $RUNNER_RC)." 4
fi

heading "6. INDEPENDENTLY VERIFY EVERY SECTION 15 REQUIREMENT"
set +e
"$PYTHON" - \
    "$CONFIG_FILE" \
    "$RESULTS_FILE" \
    "$SUMMARY_FILE" \
    "$RUNNER_GATE_FILE" \
    "$AUTOMATION_GATE_JSON" \
    "$AUTOMATION_GATE_TXT" \
    "$COMPARISON_FILE" \
    "$MODE" \
    "$RUNNER_RC" <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

(
    config_raw,
    results_raw,
    summary_raw,
    runner_gate_raw,
    gate_json_raw,
    gate_txt_raw,
    comparison_raw,
    mode,
    runner_rc_raw,
) = sys.argv[1:]

config_path = Path(config_raw)
results_path = Path(results_raw)
summary_path = Path(summary_raw)
runner_gate_path = Path(runner_gate_raw)
gate_json_path = Path(gate_json_raw)
gate_txt_path = Path(gate_txt_raw)
comparison_path = Path(comparison_raw)
runner_rc = int(runner_rc_raw)

cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(cfg, dict):
    raise SystemExit("Pilot configuration is not a mapping.")

df = pd.read_csv(results_path)
if df.empty:
    raise SystemExit("Pilot result table is empty.")


def first_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "pass", "passed"}


def finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def recursive_values(value: Any, keys: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in keys:
                found.append(item)
            found.extend(recursive_values(item, keys))
    elif isinstance(value, list):
        for item in value:
            found.extend(recursive_values(item, keys))
    return found


def normalize_model(value: Any) -> str:
    raw = str(value).strip().lower().replace("-", "_").replace("+", "_")
    raw = "_".join(raw.split())
    aliases = {
        "budgetmemr": "budgetmem_r",
        "budget_mem_r": "budgetmem_r",
        "gru_uniform": "gru_uniform_cache",
        "gru_uniform_checkpoint": "gru_uniform_cache",
        "uniform": "gru_uniform_cache",
        "uniform_cache": "gru_uniform_cache",
        "gru_reservoir": "gru_reservoir_cache",
        "reservoir": "gru_reservoir_cache",
        "reservoir_cache": "gru_reservoir_cache",
    }
    return aliases.get(raw, raw)


def normalize_task(value: Any) -> str:
    raw = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "distractor_retrieval": "distractor_heavy_retrieval",
        "distractor": "distractor_heavy_retrieval",
    }
    return aliases.get(raw, raw)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

required = {
    "task": ["task"],
    "model": ["model", "model_name"],
    "sequence_length": ["sequence_length", "seq_len", "length"],
    "memory_budget": ["memory_budget", "budget"],
}
columns: dict[str, str] = {}
for logical, aliases in required.items():
    column = first_column(df, aliases)
    if column is None:
        raise SystemExit(f"Required result column is missing: {logical}")
    columns[logical] = column

df = df.copy()
df["__task"] = df[columns["task"]].map(normalize_task)
df["__model"] = df[columns["model"]].map(normalize_model)
df["__length"] = pd.to_numeric(df[columns["sequence_length"]], errors="coerce")
df["__budget"] = pd.to_numeric(df[columns["memory_budget"]], errors="coerce")
seed_column = first_column(df, ["seed", "random_seed", "experiment_seed"])
if seed_column is None:
    df["__seed"] = int(cfg.get("seed", 2026))
else:
    df["__seed"] = pd.to_numeric(df[seed_column], errors="coerce").fillna(int(cfg.get("seed", 2026)))

expected_tasks = {"selective_copy", "associative_recall", "distractor_heavy_retrieval"}
expected_models = {"gru", "gru_uniform_cache", "gru_reservoir_cache", "budgetmem_r"}
expected_lengths = {256, 512, 1024}
expected_budgets = {16, 32}
expected_seed = int(cfg.get("seed", 2026))
expected_cells = {
    (task, model, length, budget, expected_seed)
    for task in expected_tasks
    for model in expected_models
    for length in expected_lengths
    for budget in expected_budgets
}
observed_cells = {
    (str(task), str(model), int(length), int(budget), int(seed))
    for task, model, length, budget, seed in df[
        ["__task", "__model", "__length", "__budget", "__seed"]
    ].itertuples(index=False, name=None)
    if pd.notna(length) and pd.notna(budget) and pd.notna(seed)
}
missing_cells = sorted(expected_cells - observed_cells)
unexpected_cells = sorted(observed_cells - expected_cells)
unique_required_rows = df.drop_duplicates(["__task", "__model", "__length", "__budget", "__seed"])
matrix_complete = (
    len(df) == 72
    and len(unique_required_rows) == 72
    and not missing_cells
    and not unexpected_cells
)

budgetmem = df[df["__model"].eq("budgetmem_r")].copy()

# Stability gate.
stability_column = first_column(df, ["stability_pass", "training_stability_pass", "finite_losses"])
gradient_column = first_column(
    df,
    ["maximum_gradient_norm", "max_gradient_norm", "gradient_norm_max"],
)
loss_columns = [
    column
    for column in [
        first_column(df, ["first_loss", "initial_loss"]),
        first_column(df, ["final_loss", "training_loss", "loss"]),
    ]
    if column is not None
]
stability_signals: list[bool] = []
if stability_column is not None:
    stability_signals.append(df[stability_column].map(truthy).all())
if gradient_column is not None:
    gradients = finite_series(df, gradient_column)
    gradient_limit = float(cfg.get("training", {}).get("maximum_acceptable_gradient_norm", 100.0))
    stability_signals.append(gradients.notna().all() and bool((gradients <= gradient_limit + 1e-12).all()))
for column in loss_columns:
    values = finite_series(df, column)
    stability_signals.append(values.notna().all() and bool(values.map(math.isfinite).all()))
stability_pass = bool(stability_signals) and all(stability_signals)

# Strict budget gate.
budget_pass_column = first_column(df, ["budget_pass", "memory_budget_pass", "budget_respected"])
max_memory_column = first_column(df, ["max_memory_size", "maximum_memory_size", "memory_size_max"])
budget_signals: list[bool] = []
if budget_pass_column is not None:
    budget_signals.append(df[budget_pass_column].map(truthy).all())
if max_memory_column is not None:
    maximum_sizes = finite_series(df, max_memory_column)
    budget_signals.append(
        maximum_sizes.notna().all()
        and bool((maximum_sizes <= df["__budget"] + 1e-12).all())
    )
budget_enforcement_pass = bool(budget_signals) and all(budget_signals)

# Controller behavior gates.
write_column = first_column(df, ["write_frequency", "write_rate", "controller_write_frequency"])
recent_column = first_column(df, ["recent_state_overlap", "recent_overlap", "recency_overlap"])
relevant_column = first_column(
    df,
    ["relevant_state_retention_rate", "relevant_retention_rate", "memory_recall"],
)
random_column = first_column(df, ["random_retention_rate", "random_selection_retention_rate"])

write_min = float(cfg.get("evaluation", {}).get("nontrivial_write_frequency_min", 0.01))
write_max = float(cfg.get("evaluation", {}).get("nontrivial_write_frequency_max", 0.95))
recent_limit = float(cfg.get("evaluation", {}).get("recent_overlap_failure_threshold", 0.90))
random_margin = float(cfg.get("evaluation", {}).get("random_retention_margin", 0.01))

if write_column is not None and not budgetmem.empty:
    writes = finite_series(budgetmem, write_column)
    nontrivial_writes_pass = (
        writes.notna().all()
        and bool((writes >= write_min - 1e-12).all())
        and bool((writes <= write_max + 1e-12).all())
    )
else:
    writes = pd.Series(dtype=float)
    nontrivial_writes_pass = False

if recent_column is not None and not budgetmem.empty:
    recent_values = finite_series(budgetmem, recent_column)
    non_recent_only_pass = recent_values.notna().all() and bool((recent_values < recent_limit).all())
else:
    recent_values = pd.Series(dtype=float)
    non_recent_only_pass = False

retention_gains: pd.Series
if relevant_column is not None and not budgetmem.empty:
    relevant_values = finite_series(budgetmem, relevant_column)
    if random_column is not None:
        random_values = finite_series(budgetmem, random_column)
    else:
        random_values = budgetmem["__budget"] / budgetmem["__length"]
    retention_gains = relevant_values - random_values
    retention_over_random_pass = (
        relevant_values.notna().all()
        and random_values.notna().all()
        and float(retention_gains.mean()) >= random_margin - 1e-12
    )
else:
    relevant_values = pd.Series(dtype=float)
    retention_gains = pd.Series(dtype=float)
    retention_over_random_pass = False

# Resource measurement gate. Require at least one positive wall-time column and
# at least one positive memory column, with no invalid values in either family.
wall_columns = [
    column
    for column in [
        first_column(df, ["train_wall_seconds", "training_wall_seconds"]),
        first_column(df, ["eval_wall_seconds", "evaluation_wall_seconds"]),
        first_column(df, ["wall_seconds", "wall_time_seconds"]),
    ]
    if column is not None
]
memory_columns = [
    column
    for column in [
        first_column(df, ["train_peak_rss_mb", "training_peak_rss_mb"]),
        first_column(df, ["eval_peak_rss_mb", "evaluation_peak_rss_mb"]),
        first_column(df, ["peak_rss_mb", "peak_memory_mb"]),
    ]
    if column is not None
]
resource_signals: list[bool] = []
for column in wall_columns + memory_columns:
    values = finite_series(df, column)
    resource_signals.append(values.notna().all() and bool((values > 0).all()))
resource_measurements_pass = bool(wall_columns) and bool(memory_columns) and all(resource_signals)

# Checkpoint resumption gate from either the CSV or nested summary evidence.
resume_column = first_column(df, ["checkpoint_resume_pass", "resume_pass", "resumption_pass"])
resume_signals: list[bool] = []
if resume_column is not None:
    resume_signals.append(df[resume_column].map(truthy).all())
summary: dict[str, Any] = {}
if summary_path.exists():
    loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if isinstance(loaded_summary, dict):
        summary = loaded_summary
    nested_resume = recursive_values(
        loaded_summary,
        {"checkpoint_resume_pass", "resume_pass", "resumption_pass"},
    )
    if nested_resume:
        resume_signals.append(all(truthy(value) for value in nested_resume))
checkpoint_resumption_pass = bool(resume_signals) and all(resume_signals)

# Configuration provenance gate.
config_sha = hash_file(config_path)
config_hash_column = first_column(df, ["config_sha256", "configuration_sha256", "config_hash"])
config_path_column = first_column(df, ["config_path", "configuration_path"])
provenance_signals: list[bool] = []
if config_hash_column is not None:
    observed_hashes = {str(value).strip().lower() for value in df[config_hash_column].dropna()}
    provenance_signals.append(observed_hashes == {config_sha.lower()})
if config_path_column is not None:
    observed_paths = {Path(str(value)).name for value in df[config_path_column].dropna()}
    provenance_signals.append(observed_paths == {config_path.name})
summary_hashes = {
    str(value).strip().lower()
    for value in recursive_values(summary, {"config_sha256", "configuration_sha256", "config_hash"})
    if value is not None
}
if summary_hashes:
    provenance_signals.append(summary_hashes == {config_sha.lower()})
configuration_provenance_pass = bool(provenance_signals) and all(provenance_signals)

# Final long-range, same-budget policy comparison.
metric_candidates = [
    "memory_recall",
    "relevant_state_retention_rate",
    "successful_long_range_retrievals",
    "token_accuracy",
    "exact_match_accuracy",
]
metric_column = first_column(df, metric_candidates)
comparison_rows: list[dict[str, Any]] = []
policy_aggregate: dict[str, list[float]] = defaultdict(list)
cell_passes = 0
matched_cells = 0
clear_margin = 0.02

if metric_column is not None:
    metric_values = pd.to_numeric(df[metric_column], errors="coerce")
    finite_metric = metric_values.dropna()
    if not finite_metric.empty and float(finite_metric.abs().max()) > 1.5:
        clear_margin = 2.0
    df["__metric"] = metric_values
    long_df = df[df["__length"].eq(1024)].copy()
    keys = ["__task", "__length", "__budget", "__seed"]
    policies = ["gru_uniform_cache", "gru_reservoir_cache"]
    for key, group in long_df.groupby(keys, dropna=False):
        model_values = (
            group.groupby("__model", dropna=False)["__metric"]
            .mean()
            .to_dict()
        )
        if "budgetmem_r" not in model_values or any(policy not in model_values for policy in policies):
            continue
        if not all(math.isfinite(float(model_values[name])) for name in ["budgetmem_r", *policies]):
            continue
        matched_cells += 1
        budgetmem_value = float(model_values["budgetmem_r"])
        cell_policy_passes: list[bool] = []
        for policy in policies:
            policy_value = float(model_values[policy])
            gain = budgetmem_value - policy_value
            passed = gain >= clear_margin - 1e-12
            policy_aggregate[policy].append(gain)
            cell_policy_passes.append(passed)
            comparison_rows.append(
                {
                    "task": key[0],
                    "sequence_length": int(key[1]),
                    "memory_budget": int(key[2]),
                    "seed": int(key[3]),
                    "metric": metric_column,
                    "budgetmem_r": budgetmem_value,
                    "policy": policy,
                    "policy_value": policy_value,
                    "gain": gain,
                    "required_clear_gain": clear_margin,
                    "cell_policy_pass": passed,
                }
            )
        if all(cell_policy_passes):
            cell_passes += 1

policy_mean_gains = {
    policy: (sum(values) / len(values) if values else None)
    for policy, values in policy_aggregate.items()
}
required_policy_count = 2
outperformed_policies = [
    policy
    for policy, gain in policy_mean_gains.items()
    if gain is not None and gain >= clear_margin - 1e-12
]
# Require both deterministic policies to be beaten on their mean exactly-matched
# long-range cells, and require at least one cell to beat both policies directly.
performance_go = (
    matched_cells > 0
    and len(outperformed_policies) >= required_policy_count
    and cell_passes > 0
)

comparison_path.parent.mkdir(parents=True, exist_ok=True)
with comparison_path.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "task",
        "sequence_length",
        "memory_budget",
        "seed",
        "metric",
        "budgetmem_r",
        "policy",
        "policy_value",
        "gain",
        "required_clear_gain",
        "cell_policy_pass",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(comparison_rows)

checks = {
    "pilot_matrix_has_exactly_72_unique_required_cells": matrix_complete,
    "model_trains_without_instability": stability_pass,
    "memory_budget_never_violated": budget_enforcement_pass,
    "controller_writes_nontrivially": nontrivial_writes_pass,
    "controller_does_not_retain_every_recent_state": non_recent_only_pass,
    "relevant_state_retention_exceeds_random": retention_over_random_pass,
    "resource_measurements_are_valid": resource_measurements_pass,
    "checkpoint_resumption_works": checkpoint_resumption_pass,
    "results_are_configuration_driven": configuration_provenance_pass,
    "budgetmem_r_beats_two_deterministic_policies": performance_go,
}
operational_checks = {key: value for key, value in checks.items() if key != "budgetmem_r_beats_two_deterministic_policies"}
operational_pass = all(operational_checks.values())
final_go = operational_pass and performance_go

runner_gate: dict[str, Any] | None = None
if runner_gate_path.exists():
    loaded_gate = json.loads(runner_gate_path.read_text(encoding="utf-8"))
    if isinstance(loaded_gate, dict):
        runner_gate = loaded_gate

payload = {
    "section": "15",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "mode": mode,
    "runner_exit_code": runner_rc,
    "status": "GO" if final_go else "NO_GO",
    "operational_status": "PASS" if operational_pass else "FAIL",
    "scientific_performance_status": "PASS" if performance_go else "FAIL",
    "decision": (
        "Proceed to the full experiment matrix."
        if final_go
        else "Do not begin the full experiment matrix. Correct failed pilot criteria and rerun Section 15."
    ),
    "expected_result_rows": 72,
    "observed_result_rows": int(len(df)),
    "unique_required_cells": int(len(unique_required_rows)),
    "missing_cells": [list(cell) for cell in missing_cells],
    "unexpected_cells": [list(cell) for cell in unexpected_cells],
    "config_path": str(config_path),
    "config_sha256": config_sha,
    "result_table": str(results_path),
    "summary_json": str(summary_path),
    "runner_gate_json": str(runner_gate_path),
    "runner_gate": runner_gate,
    "checks": checks,
    "metrics": {
        "gradient_column": gradient_column,
        "maximum_gradient_norm": (
            float(finite_series(df, gradient_column).max()) if gradient_column else None
        ),
        "max_memory_column": max_memory_column,
        "maximum_memory_size": (
            float(finite_series(df, max_memory_column).max()) if max_memory_column else None
        ),
        "write_frequency_column": write_column,
        "budgetmem_write_frequency_min": (float(writes.min()) if not writes.empty else None),
        "budgetmem_write_frequency_mean": (float(writes.mean()) if not writes.empty else None),
        "budgetmem_write_frequency_max": (float(writes.max()) if not writes.empty else None),
        "recent_overlap_column": recent_column,
        "budgetmem_recent_overlap_max": (
            float(recent_values.max()) if not recent_values.empty else None
        ),
        "relevant_retention_column": relevant_column,
        "mean_retention_gain_over_random": (
            float(retention_gains.mean()) if not retention_gains.empty else None
        ),
        "resource_wall_columns": wall_columns,
        "resource_memory_columns": memory_columns,
        "long_range_comparison_metric": metric_column,
        "clear_gain_required": clear_margin,
        "matched_long_range_cells": matched_cells,
        "cells_beating_both_policies": cell_passes,
        "policy_mean_gains": policy_mean_gains,
        "outperformed_policies": outperformed_policies,
    },
    "comparison_csv": str(comparison_path),
}

gate_json_path.parent.mkdir(parents=True, exist_ok=True)
gate_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

lines = [
    "Section 15 Pilot Experiment — Independent Automation Gate",
    "=========================================================",
    f"Generated UTC: {payload['generated_at_utc']}",
    f"Mode: {mode}",
    f"Result rows: {len(df)} / 72",
    f"Operational status: {payload['operational_status']}",
    f"Scientific performance status: {payload['scientific_performance_status']}",
    "",
]
for name, passed in checks.items():
    lines.append(f"{'PASS' if passed else 'FAIL'}  {name}")
lines.extend(
    [
        "",
        f"Long-range comparison metric: {metric_column or 'NOT FOUND'}",
        f"Clear-gain requirement: {clear_margin}",
        f"Matched long-range cells: {matched_cells}",
        f"Cells beating both policies: {cell_passes}",
        f"Uniform-cache mean gain: {policy_mean_gains.get('gru_uniform_cache')}",
        f"Reservoir-cache mean gain: {policy_mean_gains.get('gru_reservoir_cache')}",
        "",
        f"Final decision: {'GO' if final_go else 'NO_GO'}",
        payload["decision"],
        f"JSON evidence: {gate_json_path}",
        f"Comparison table: {comparison_path}",
    ]
)
gate_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
raise SystemExit(0 if final_go else 3)
PY
GATE_RC=$?
set -e

heading "7. FINAL RESULT"
cat "$AUTOMATION_GATE_TXT"
printf '\nArtifacts:\n'
printf '  %s\n' "$CONFIG_FILE"
printf '  %s\n' "$RESULTS_FILE"
printf '  %s\n' "$SUMMARY_FILE"
printf '  %s\n' "$RUNNER_GATE_FILE"
printf '  %s\n' "$REPORT_FILE"
printf '  %s\n' "$AUTOMATION_GATE_JSON"
printf '  %s\n' "$AUTOMATION_GATE_TXT"
printf '  %s\n' "$COMPARISON_FILE"
printf '  %s\n' "$LOG_FILE"
printf '  %s\n' "$BACKUP_DIR"

printf '\nRepository status:\n'
git status --short

if [[ "$GATE_RC" -eq 0 ]]; then
    printf '\nSECTION 15: COMPLETE — FINAL DECISION GO\n'
    exit 0
fi

printf '\nSECTION 15: NOT COMPLETE — FINAL DECISION NO_GO\n'
printf 'Review the FAIL lines in %s.\n' "$AUTOMATION_GATE_TXT"
exit "$GATE_RC"
