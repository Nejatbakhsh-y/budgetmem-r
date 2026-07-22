#!/usr/bin/env bash
# Section 18 — Main Experiment Matrix
# Run from the budgetmem-r repository root in the VS Code WSL terminal.
#
# Safe default:
#   ./21_section18_main_experiment_matrix.sh
# creates and validates the complete matrix without starting training.
#
# Execute:
#   ./21_section18_main_experiment_matrix.sh --execute
#
# Common controls:
#   MAX_PARALLEL_RUNS=1 ./21_section18_main_experiment_matrix.sh --execute
#   PHASE=synthetic-primary ./21_section18_main_experiment_matrix.sh --execute
#   FULL_SECONDARY=1 ./21_section18_main_experiment_matrix.sh --execute
#   RETRY_FAILED=1 ./21_section18_main_experiment_matrix.sh --execute
#
# If automatic runner detection fails:
#   BUDGETMEM_RUNNER='python -m budgetmem.experiments.run --config {config}' \
#     ./21_section18_main_experiment_matrix.sh --execute
#
# The runner template must contain {config}. Optional placeholders:
# {run_dir}, {run_id}, {phase}, {model}, {task}, {dataset},
# {sequence_length}, {budget}, {seed}, {retrieval_k}.

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

SCRIPT_VERSION="1.0.0"
SECTION="18"
AUTOMATION_NAME="21_section18_main_experiment_matrix.sh"

# ----------------------------- CLI defaults -----------------------------
ACTION="plan"
PHASE="${PHASE:-all}"
MAX_PARALLEL_RUNS="${MAX_PARALLEL_RUNS:-1}"
FULL_SECONDARY="${FULL_SECONDARY:-0}"
RETRY_FAILED="${RETRY_FAILED:-0}"
FAIL_FAST="${FAIL_FAST:-0}"
COMMIT_CONTROLLED="${COMMIT_CONTROLLED:-0}"
PUSH_CONTROLLED="${PUSH_CONTROLLED:-0}"
RUNNER_TEMPLATE="${BUDGETMEM_RUNNER:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PRIMARY_METRIC="${PRIMARY_METRIC:-auto}"

# Section 17 fair-comparison defaults.
PRIMARY_BUDGETS="${PRIMARY_BUDGETS:-32 64}"
RETRIEVAL_K="${RETRIEVAL_K:-4}"
SEEDS="${SEEDS:-2026 2027 2028 2029 2030}"
SYNTHETIC_LENGTHS="${SYNTHETIC_LENGTHS:-256 512 1024 2048 4096 8192 16384}"
IMDB_LENGTHS="${IMDB_LENGTHS:-1024 2048 4096}"
SECONDARY_LENGTHS="${SECONDARY_LENGTHS:-1024 4096 16384}"
SECONDARY_TASK_LIMIT="${SECONDARY_TASK_LIMIT:-3}"
HDFS_LENGTH="${HDFS_LENGTH:-1024}"
BGL_LENGTH="${BGL_LENGTH:-4096}"
BGL_BUDGET="${BGL_BUDGET:-64}"

PRIMARY_MODELS="${PRIMARY_MODELS:-gru gru_uniform gru_reservoir memory_caching budgetmem_r}"
SECONDARY_MODELS="${SECONDARY_MODELS:-lstm transformer mamba rmt}"
REAL_MODELS="${REAL_MODELS:-gru lstm transformer mamba gru_uniform memory_caching budgetmem_r}"
NON_MEMORY_CANDIDATES="${NON_MEMORY_CANDIDATES:-gru lstm transformer mamba}"
EXISTING_MEMORY_CANDIDATES="${EXISTING_MEMORY_CANDIDATES:-gru_uniform gru_reservoir memory_caching rmt}"

# Fallback names are used only when six synthetic task names cannot be discovered
# from existing YAML configuration files. Override with SYNTHETIC_TASKS.
FALLBACK_SYNTHETIC_TASKS="selective_copy associative_recall distractor_retrieval key_value_retrieval variable_delay_recall needle_in_haystack"
SYNTHETIC_TASKS="${SYNTHETIC_TASKS:-}"

# ----------------------------- Paths ------------------------------------
ROOT=""
CONFIG_ROOT=""
RUN_ROOT=""
LOG_ROOT=""
TABLE_ROOT=""
EVIDENCE_ROOT=""
DOC_ROOT=""
STATE_ROOT=""
MATRIX_CSV=""
STATUS_CSV=""
FAILURES_CSV=""
COMMANDS_FILE=""
PROTOCOL_MD=""
SUMMARY_TXT=""
RUNNER_RECORD=""
TASK_RECORD=""
START_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
HOSTNAME_VALUE="$(hostname 2>/dev/null || printf 'unknown')"
GIT_COMMIT="unknown"

# ----------------------------- Utilities --------------------------------
usage() {
  cat <<'EOF'
Section 18 Main Experiment Matrix

Usage:
  ./21_section18_main_experiment_matrix.sh [options]

Actions:
  --plan                 Generate and validate matrix only. Default.
  --execute              Execute selected phase, resuming completed runs.
  --summarize            Rebuild status summaries and BGL selections only.
  --validate             Validate repository, matrix, configurations, and runner.

Options:
  --phase NAME           all | synthetic-primary | synthetic-secondary |
                         hdfs | imdb | bgl
  --jobs N               Maximum parallel runs. Default: 1.
  --full-secondary       Run the complete secondary matrix instead of reduced.
  --retry-failed         Retry runs previously marked FAILED or OOM.
  --fail-fast            Stop after the first failed run.
  --commit               Commit controlled Section 18 artifacts.
  --push                 Commit and push controlled artifacts.
  -h, --help             Show this help.

Environment overrides:
  SYNTHETIC_TASKS="task1 ... task6"
  PRIMARY_BUDGETS="32 64"
  SEEDS="2026 2027 2028 2029 2030"
  SYNTHETIC_LENGTHS="256 512 1024 2048 4096 8192 16384"
  BUDGETMEM_RUNNER='python -m ... --config {config}'
  MAX_PARALLEL_RUNS=1
  FULL_SECONDARY=0
  RETRY_FAILED=0
EOF
}

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

warn() {
  log "WARNING: $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

trim() {
  local x="$*"
  x="${x#"${x%%[![:space:]]*}"}"
  x="${x%"${x##*[![:space:]]}"}"
  printf '%s' "$x"
}

csv_escape() {
  local s="${1//\"/\"\"}"
  printf '"%s"' "$s"
}

contains_word() {
  local needle="$1"; shift
  local x
  for x in "$@"; do
    [[ "$x" == "$needle" ]] && return 0
  done
  return 1
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    "$PYTHON_BIN" - "$1" <<'PY'
import hashlib, pathlib, sys
p = pathlib.Path(sys.argv[1])
print(hashlib.sha256(p.read_bytes()).hexdigest())
PY
  fi
}

repo_root() {
  if git rev-parse --show-toplevel >/dev/null 2>&1; then
    git rev-parse --show-toplevel
  else
    pwd
  fi
}

normalize_name() {
  local s="${1,,}"
  s="${s// /_}"
  s="${s//-/_}"
  s="${s//./_}"
  s="${s//\//_}"
  while [[ "$s" == *"__"* ]]; do s="${s//__/_}"; done
  s="${s#_}"
  s="${s%_}"
  printf '%s' "$s"
}

model_requires_budget() {
  case "$1" in
    gru_uniform|gru_reservoir|memory_caching|budgetmem_r|rmt) return 0 ;;
    *) return 1 ;;
  esac
}

phase_selected() {
  local candidate="$1"
  [[ "$PHASE" == "all" || "$PHASE" == "$candidate" ]]
}

# ----------------------------- CLI parsing -------------------------------
while (($#)); do
  case "$1" in
    --plan) ACTION="plan" ;;
    --execute) ACTION="execute" ;;
    --summarize) ACTION="summarize" ;;
    --validate) ACTION="validate" ;;
    --phase)
      shift
      (($#)) || die "--phase requires a value"
      PHASE="$1"
      ;;
    --jobs)
      shift
      (($#)) || die "--jobs requires a value"
      MAX_PARALLEL_RUNS="$1"
      ;;
    --full-secondary) FULL_SECONDARY=1 ;;
    --retry-failed) RETRY_FAILED=1 ;;
    --fail-fast) FAIL_FAST=1 ;;
    --commit) COMMIT_CONTROLLED=1 ;;
    --push)
      COMMIT_CONTROLLED=1
      PUSH_CONTROLLED=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "Unknown option: $1" ;;
  esac
  shift
done

case "$ACTION" in plan|execute|summarize|validate) ;; *) die "Invalid action: $ACTION" ;; esac
case "$PHASE" in all|synthetic-primary|synthetic-secondary|hdfs|imdb|bgl) ;;
  *) die "Invalid phase: $PHASE" ;;
esac
is_positive_int "$MAX_PARALLEL_RUNS" || die "MAX_PARALLEL_RUNS must be a positive integer"
[[ "$FULL_SECONDARY" =~ ^[01]$ ]] || die "FULL_SECONDARY must be 0 or 1"
[[ "$RETRY_FAILED" =~ ^[01]$ ]] || die "RETRY_FAILED must be 0 or 1"
[[ "$FAIL_FAST" =~ ^[01]$ ]] || die "FAIL_FAST must be 0 or 1"

# ----------------------------- Repository setup --------------------------
require_cmd git
require_cmd "$PYTHON_BIN"
ROOT="$(repo_root)"
cd "$ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || printf unknown)"

[[ -d src/budgetmem || -d budgetmem ]] || \
  warn "Expected package directory src/budgetmem or budgetmem was not found. Continuing for matrix generation."

CONFIG_ROOT="$ROOT/configs/experiments/section18"
RUN_ROOT="$ROOT/artifacts/section18/runs"
LOG_ROOT="$ROOT/artifacts/section18/logs"
STATE_ROOT="$ROOT/artifacts/section18/state"
TABLE_ROOT="$ROOT/reports/tables/section18"
EVIDENCE_ROOT="$ROOT/reports/evidence/section18"
DOC_ROOT="$ROOT/docs"
MATRIX_CSV="$TABLE_ROOT/main_experiment_matrix.csv"
STATUS_CSV="$TABLE_ROOT/main_experiment_status.csv"
FAILURES_CSV="$TABLE_ROOT/main_experiment_failures.csv"
COMMANDS_FILE="$EVIDENCE_ROOT/commands.sh"
PROTOCOL_MD="$DOC_ROOT/section18_main_experiment_matrix.md"
SUMMARY_TXT="$EVIDENCE_ROOT/section18_summary.txt"
RUNNER_RECORD="$EVIDENCE_ROOT/runner_detection.txt"
TASK_RECORD="$EVIDENCE_ROOT/synthetic_task_discovery.txt"

mkdir -p "$CONFIG_ROOT" "$RUN_ROOT" "$LOG_ROOT" "$STATE_ROOT" \
         "$TABLE_ROOT" "$EVIDENCE_ROOT" "$DOC_ROOT"

# Activate a local virtual environment when present.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  for activate in "$ROOT/.venv/bin/activate" "$ROOT/venv/bin/activate"; do
    if [[ -f "$activate" ]]; then
      # shellcheck disable=SC1090
      source "$activate"
      log "Activated virtual environment: $activate"
      break
    fi
  done
fi

# ----------------------------- Git controls ------------------------------
ensure_gitignore() {
  local begin="# BEGIN SECTION 18 RUNTIME OUTPUTS"
  local end="# END SECTION 18 RUNTIME OUTPUTS"
  touch .gitignore
  if ! grep -Fq "$begin" .gitignore; then
    cat >> .gitignore <<'EOF'

# BEGIN SECTION 18 RUNTIME OUTPUTS
artifacts/section18/
reports/evidence/section18/logs/
*.section18.tmp
# END SECTION 18 RUNTIME OUTPUTS
EOF
  fi
}

ensure_branch() {
  local branch="${SECTION18_BRANCH:-feature/18-main-experiment-matrix}"
  local current
  current="$(git branch --show-current 2>/dev/null || true)"
  if [[ -z "$current" ]]; then
    warn "Detached HEAD; branch creation skipped."
    return
  fi
  if [[ "$current" == "$branch" ]]; then
    return
  fi
  if git show-ref --verify --quiet "refs/heads/$branch"; then
    git switch "$branch"
  else
    git switch -c "$branch"
  fi
}

ensure_gitignore
if [[ "${CREATE_SECTION18_BRANCH:-1}" == "1" && "$ACTION" != "summarize" ]]; then
  ensure_branch
fi

# ----------------------------- Task discovery ----------------------------
discover_tasks() {
  if [[ -n "$SYNTHETIC_TASKS" ]]; then
    printf '%s\n' "$SYNTHETIC_TASKS"
    return
  fi

  if [[ -f "$TASK_RECORD" ]]; then
    local recorded
    recorded="$(sed -n 's/^Tasks: //p' "$TASK_RECORD" | head -n 1)"
    if [[ -n "$recorded" ]]; then
      printf '%s\n' "$recorded"
      return
    fi
  fi

  "$PYTHON_BIN" - "$ROOT" "$FALLBACK_SYNTHETIC_TASKS" <<'PY'
from pathlib import Path
import re, sys

root = Path(sys.argv[1])
fallback = sys.argv[2].split()
real = {"hdfs", "imdb", "bgl"}
found = []

# Prefer explicit task config filenames.
for p in sorted((root / "configs").glob("**/*.y*ml")) if (root / "configs").exists() else []:
    if "section18" in {part.lower() for part in p.parts}:
        continue
    stem = re.sub(r"[^a-z0-9]+", "_", p.stem.lower()).strip("_")
    if stem in real or any(x in stem for x in ("hdfs", "imdb", "bgl")):
        continue
    if any(k in stem for k in (
        "selective", "associative", "distractor", "recall", "copy",
        "needle", "retrieval", "delay", "key_value"
    )):
        found.append(stem)

# Extract task names from YAML text without requiring PyYAML.
task_pattern = re.compile(r"^\s*(?:name|task|task_name)\s*:\s*['\"]?([A-Za-z0-9_.-]+)", re.M)
for p in sorted((root / "configs").glob("**/*.y*ml")) if (root / "configs").exists() else []:
    if "section18" in {part.lower() for part in p.parts}:
        continue
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    if not re.search(r"(?im)^\s*(?:task|task_name)\s*:", text):
        continue
    for value in task_pattern.findall(text):
        name = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        if name and name not in real:
            found.append(name)

ordered = []
for name in found:
    if name not in ordered:
        ordered.append(name)

for name in fallback:
    if len(ordered) >= 6:
        break
    if name not in ordered:
        ordered.append(name)

print(" ".join(ordered[:6]))
PY
}

SYNTHETIC_TASKS="$(trim "$(discover_tasks)")"
IFS=' ' read -r -a SYNTH_TASK_ARRAY <<< "$SYNTHETIC_TASKS"
((${#SYNTH_TASK_ARRAY[@]} == 6)) || \
  die "Exactly six synthetic tasks are required; discovered ${#SYNTH_TASK_ARRAY[@]}: $SYNTHETIC_TASKS"

{
  printf 'Section 18 synthetic task discovery\n'
  printf 'Timestamp UTC: %s\n' "$START_UTC"
  printf 'Tasks: %s\n' "$SYNTHETIC_TASKS"
  printf 'Override source: %s\n' "$([[ -n "${SYNTHETIC_TASKS_OVERRIDE:-}" ]] && echo environment || echo automatic/default)"
  printf 'Requirement: exactly six tasks\n'
} > "$TASK_RECORD"

# ----------------------------- Runner detection --------------------------
module_exists() {
  "$PYTHON_BIN" - "$1" <<'PY' >/dev/null 2>&1
import importlib.util, sys
raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
}

detect_runner() {
  if [[ -n "$RUNNER_TEMPLATE" ]]; then
    [[ "$RUNNER_TEMPLATE" == *"{config}"* ]] || \
      die "BUDGETMEM_RUNNER must contain the placeholder {config}"
    printf '%s' "$RUNNER_TEMPLATE"
    return
  fi

  local candidate
  local -a candidates=()

  module_exists budgetmem.experiments.run && \
    candidates+=("$PYTHON_BIN -m budgetmem.experiments.run --config {config}")
  module_exists budgetmem.cli.run && \
    candidates+=("$PYTHON_BIN -m budgetmem.cli.run --config {config}")
  module_exists budgetmem.cli.train && \
    candidates+=("$PYTHON_BIN -m budgetmem.cli.train --config {config}")
  module_exists budgetmem.train && \
    candidates+=("$PYTHON_BIN -m budgetmem.train --config {config}")

  [[ -f scripts/run_experiment.py ]] && \
    candidates+=("$PYTHON_BIN scripts/run_experiment.py --config {config}")
  [[ -f scripts/train.py ]] && \
    candidates+=("$PYTHON_BIN scripts/train.py --config {config}")
  [[ -f train.py ]] && \
    candidates+=("$PYTHON_BIN train.py --config {config}")

  for candidate in "${candidates[@]}"; do
    # Use the first structurally available candidate. Full validation occurs
    # before execution by invoking its help command.
    printf '%s' "$candidate"
    return
  done

  printf ''
}

RUNNER_TEMPLATE="$(detect_runner)"
{
  printf 'Section 18 runner detection\n'
  printf 'Timestamp UTC: %s\n' "$START_UTC"
  printf 'Python: %s\n' "$("$PYTHON_BIN" --version 2>&1)\n"
  printf 'Runner template: %s\n' "${RUNNER_TEMPLATE:-NOT DETECTED}"
} > "$RUNNER_RECORD"

validate_runner() {
  if [[ -z "$RUNNER_TEMPLATE" ]]; then
    die "No experiment runner was detected. Set BUDGETMEM_RUNNER with a {config} placeholder."
  fi
  [[ "$RUNNER_TEMPLATE" == *"{config}"* ]] || \
    die "Runner template does not contain {config}: $RUNNER_TEMPLATE"

  local help_cmd="${RUNNER_TEMPLATE%%--config*}--help"
  # A nonzero help status is recorded as a warning because some project CLIs
  # initialize configuration before showing help.
  if ! timeout 30s bash -lc "$help_cmd" >> "$RUNNER_RECORD" 2>&1; then
    warn "Runner help check was inconclusive. Execution will still use: $RUNNER_TEMPLATE"
  fi
}

# ----------------------------- Config generation -------------------------
write_config() {
  local config_path="$1"
  local run_id="$2"
  local phase="$3"
  local task="$4"
  local dataset="$5"
  local model="$6"
  local sequence_length="$7"
  local budget="$8"
  local seed="$9"
  local run_dir="${10}"
  local external_role="${11:-}"

  cat > "$config_path" <<EOF
# Generated by $AUTOMATION_NAME v$SCRIPT_VERSION
schema_version: "1.0"
section: 18

experiment:
  name: "section18_main_experiment_matrix"
  phase: "$phase"
  run_id: "$run_id"
  output_dir: "$run_dir"
  generated_utc: "$START_UTC"
  host: "$HOSTNAME_VALUE"
  external_validation_role: "$external_role"

task:
  name: "$task"
  dataset: "$dataset"
  sequence_length: $sequence_length

data:
  dataset: "$dataset"
  sequence_length: $sequence_length
  split_seed: $seed

model:
  name: "$model"

memory:
  budget: $budget
  retrieval_k: $RETRIEVAL_K
  strict_budget_enforcement: true

training:
  seed: $seed
  deterministic: true
  resume: true
  output_dir: "$run_dir"

evaluation:
  seed: $seed
  sequence_length: $sequence_length
  save_predictions: false
  save_metrics: true

fair_comparison:
  parameter_regime: "from_frozen_protocol"
  fixed_training_tokens: true
  fixed_max_optimization_steps: true
  fixed_gradient_accumulation: true
  fixed_precision: true
  fixed_hardware: true
  fixed_hyperparameter_search_trials: true
  fixed_evaluation_frequency: true
  fixed_batch_control: true
  latency_batch_sizes: [1, 8]

provenance:
  automation: "$AUTOMATION_NAME"
  automation_version: "$SCRIPT_VERSION"
  git_commit: "$GIT_COMMIT"
EOF
}

matrix_header() {
  cat > "$MATRIX_CSV" <<'EOF'
run_id,phase,task,dataset,model,sequence_length,memory_budget,retrieval_k,seed,config_path,run_dir,external_role
EOF
}

add_matrix_row() {
  local phase="$1"
  local task="$2"
  local dataset="$3"
  local model="$4"
  local length="$5"
  local budget="$6"
  local seed="$7"
  local external_role="${8:-}"

  local run_id
  run_id="$(normalize_name "${phase}__${task}__${dataset}__${model}__l${length}__b${budget}__s${seed}")"
  local config_path="$CONFIG_ROOT/$phase/$run_id.yaml"
  local run_dir="$RUN_ROOT/$phase/$run_id"

  write_config "$config_path" "$run_id" "$phase" "$task" "$dataset" \
               "$model" "$length" "$budget" "$seed" "$run_dir" "$external_role"

  {
    csv_escape "$run_id"; printf ','
    csv_escape "$phase"; printf ','
    csv_escape "$task"; printf ','
    csv_escape "$dataset"; printf ','
    csv_escape "$model"; printf ','
    printf '%s,%s,%s,%s,' "$length" "$budget" "$RETRIEVAL_K" "$seed"
    csv_escape "$config_path"; printf ','
    csv_escape "$run_dir"; printf ','
    csv_escape "$external_role"; printf '\n'
  } >> "$MATRIX_CSV"
}

build_matrix() {
  mkdir -p \
    "$CONFIG_ROOT/synthetic-primary" \
    "$CONFIG_ROOT/synthetic-secondary" \
    "$CONFIG_ROOT/hdfs" \
    "$CONFIG_ROOT/imdb" \
    "$CONFIG_ROOT/bgl"

  SECTION18_AUTOMATION="$AUTOMATION_NAME" \
  SECTION18_VERSION="$SCRIPT_VERSION" \
  SECTION18_START_UTC="$START_UTC" \
  SECTION18_HOST="$HOSTNAME_VALUE" \
  SECTION18_GIT_COMMIT="$GIT_COMMIT" \
  SECTION18_TASKS="$SYNTHETIC_TASKS" \
  SECTION18_PRIMARY_MODELS="$PRIMARY_MODELS" \
  SECTION18_SECONDARY_MODELS="$SECONDARY_MODELS" \
  SECTION18_REAL_MODELS="$REAL_MODELS" \
  SECTION18_BUDGETS="$PRIMARY_BUDGETS" \
  SECTION18_RETRIEVAL_K="$RETRIEVAL_K" \
  SECTION18_SEEDS="$SEEDS" \
  SECTION18_SYNTH_LENGTHS="$SYNTHETIC_LENGTHS" \
  SECTION18_IMDB_LENGTHS="$IMDB_LENGTHS" \
  SECTION18_SECONDARY_LENGTHS="$SECONDARY_LENGTHS" \
  SECTION18_SECONDARY_TASK_LIMIT="$SECONDARY_TASK_LIMIT" \
  SECTION18_FULL_SECONDARY="$FULL_SECONDARY" \
  SECTION18_HDFS_LENGTH="$HDFS_LENGTH" \
  "$PYTHON_BIN" - "$MATRIX_CSV" "$CONFIG_ROOT" "$RUN_ROOT" <<'PY'
import csv
import json
import os
import re
import sys
from pathlib import Path

matrix_path = Path(sys.argv[1])
config_root = Path(sys.argv[2])
run_root = Path(sys.argv[3])

automation = os.environ["SECTION18_AUTOMATION"]
version = os.environ["SECTION18_VERSION"]
start_utc = os.environ["SECTION18_START_UTC"]
host = os.environ["SECTION18_HOST"]
git_commit = os.environ["SECTION18_GIT_COMMIT"]
tasks = os.environ["SECTION18_TASKS"].split()
primary_models = os.environ["SECTION18_PRIMARY_MODELS"].split()
secondary_models = os.environ["SECTION18_SECONDARY_MODELS"].split()
real_models = os.environ["SECTION18_REAL_MODELS"].split()
budgets = [int(x) for x in os.environ["SECTION18_BUDGETS"].split()]
retrieval_k = int(os.environ["SECTION18_RETRIEVAL_K"])
seeds = [int(x) for x in os.environ["SECTION18_SEEDS"].split()]
synth_lengths = [int(x) for x in os.environ["SECTION18_SYNTH_LENGTHS"].split()]
imdb_lengths = [int(x) for x in os.environ["SECTION18_IMDB_LENGTHS"].split()]
secondary_lengths = [int(x) for x in os.environ["SECTION18_SECONDARY_LENGTHS"].split()]
secondary_task_limit = int(os.environ["SECTION18_SECONDARY_TASK_LIMIT"])
full_secondary = bool(int(os.environ["SECTION18_FULL_SECONDARY"]))
hdfs_length = int(os.environ["SECTION18_HDFS_LENGTH"])

if len(tasks) != 6:
    raise SystemExit(f"exactly six synthetic tasks required, found {len(tasks)}")
if len(synth_lengths) != 7:
    raise SystemExit(f"exactly seven synthetic lengths required, found {len(synth_lengths)}")
if len(primary_models) != 5:
    raise SystemExit(f"exactly five primary models required, found {len(primary_models)}")
if len(seeds) != 5:
    raise SystemExit(f"exactly five seeds required, found {len(seeds)}")
if budgets != [32, 64]:
    raise SystemExit(f"primary budgets must be 32 and 64, found {budgets}")

def norm(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")

def q(value) -> str:
    return json.dumps(str(value))

rows = []

def add(phase, task, dataset, model, length, budget, seed, role=""):
    run_id = norm(
        f"{phase}__{task}__{dataset}__{model}__l{length}__b{budget}__s{seed}"
    )
    config_path = config_root / phase / f"{run_id}.yaml"
    run_dir = run_root / phase / run_id
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = f"""# Generated by {automation} v{version}
schema_version: "1.0"
section: 18

experiment:
  name: "section18_main_experiment_matrix"
  phase: {q(phase)}
  run_id: {q(run_id)}
  output_dir: {q(run_dir)}
  generated_utc: {q(start_utc)}
  host: {q(host)}
  external_validation_role: {q(role)}

task:
  name: {q(task)}
  dataset: {q(dataset)}
  sequence_length: {length}

data:
  dataset: {q(dataset)}
  sequence_length: {length}
  split_seed: {seed}

model:
  name: {q(model)}

memory:
  budget: {budget}
  retrieval_k: {retrieval_k}
  strict_budget_enforcement: true

training:
  seed: {seed}
  deterministic: true
  resume: true
  output_dir: {q(run_dir)}

evaluation:
  seed: {seed}
  sequence_length: {length}
  save_predictions: false
  save_metrics: true

fair_comparison:
  parameter_regime: "from_frozen_protocol"
  fixed_training_tokens: true
  fixed_max_optimization_steps: true
  fixed_gradient_accumulation: true
  fixed_precision: true
  fixed_hardware: true
  fixed_hyperparameter_search_trials: true
  fixed_evaluation_frequency: true
  fixed_batch_control: true
  latency_batch_sizes: [1, 8]

provenance:
  automation: {q(automation)}
  automation_version: {q(version)}
  git_commit: {q(git_commit)}
"""
    config_path.write_text(config, encoding="utf-8", newline="\n")
    rows.append({
        "run_id": run_id,
        "phase": phase,
        "task": task,
        "dataset": dataset,
        "model": model,
        "sequence_length": length,
        "memory_budget": budget,
        "retrieval_k": retrieval_k,
        "seed": seed,
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "external_role": role,
    })

for task in tasks:
    for length in synth_lengths:
        for model in primary_models:
            for seed in seeds:
                for budget in budgets:
                    add("synthetic-primary", task, "synthetic", model, length, budget, seed)

selected_tasks = tasks if full_secondary else tasks[:secondary_task_limit]
selected_lengths = synth_lengths if full_secondary else secondary_lengths
for task in selected_tasks:
    for length in selected_lengths:
        for model in secondary_models:
            for seed in seeds:
                for budget in budgets:
                    add("synthetic-secondary", task, "synthetic", model, length, budget, seed)

for model in real_models:
    for seed in seeds:
        for budget in budgets:
            add("hdfs", "hdfs_log_anomaly", "hdfs", model, hdfs_length, budget, seed)

for length in imdb_lengths:
    for model in real_models:
        for seed in seeds:
            for budget in budgets:
                add("imdb", "imdb_sentiment", "imdb", model, length, budget, seed)

fieldnames = [
    "run_id", "phase", "task", "dataset", "model", "sequence_length",
    "memory_budget", "retrieval_k", "seed", "config_path", "run_dir",
    "external_role",
]
matrix_path.parent.mkdir(parents=True, exist_ok=True)
with matrix_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
}

# ----------------------------- Documentation -----------------------------
write_protocol() {
  local total primary_count secondary_count hdfs_count imdb_count
  total="$(($(wc -l < "$MATRIX_CSV") - 1))"
  primary_count="$(awk -F, '$2=="\"synthetic-primary\"" {n++} END{print n+0}' "$MATRIX_CSV")"
  secondary_count="$(awk -F, '$2=="\"synthetic-secondary\"" {n++} END{print n+0}' "$MATRIX_CSV")"
  hdfs_count="$(awk -F, '$2=="\"hdfs\"" {n++} END{print n+0}' "$MATRIX_CSV")"
  imdb_count="$(awk -F, '$2=="\"imdb\"" {n++} END{print n+0}' "$MATRIX_CSV")"

  cat > "$PROTOCOL_MD" <<EOF
# Section 18 — Main Experiment Matrix

Generated by \`$AUTOMATION_NAME\` version $SCRIPT_VERSION.

## Primary synthetic experiments

- Tasks: $SYNTHETIC_TASKS
- Sequence lengths: $SYNTHETIC_LENGTHS
- Models: $PRIMARY_MODELS
- Seeds: $SEEDS
- Primary memory budgets: $PRIMARY_BUDGETS
- Retrieval budget: k=$RETRIEVAL_K
- Planned runs: $primary_count

The primary synthetic matrix is the Cartesian product of six tasks, seven
sequence lengths, five primary models, five seeds, and two primary budgets.

## Secondary synthetic experiments

- Models: $SECONDARY_MODELS
- Full matrix enabled: $FULL_SECONDARY
- Reduced sequence lengths: $SECONDARY_LENGTHS
- Reduced task count: $SECONDARY_TASK_LIMIT
- Planned runs: $secondary_count

The reduced matrix is used by default because full Transformer, Mamba, and RMT
coverage across all synthetic cells may be computationally excessive. The
selection retains short-to-long coverage and multiple task types. Set
\`FULL_SECONDARY=1\` to expand the secondary matrix to all six tasks and all
seven sequence lengths.

## Real-world experiments

### HDFS

- Models: $REAL_MODELS
- Seeds: $SEEDS
- Sequence length: $HDFS_LENGTH
- Budgets: $PRIMARY_BUDGETS
- Planned runs: $hdfs_count

### IMDb

- Models: $REAL_MODELS
- Seeds: $SEEDS
- Sequence lengths: $IMDB_LENGTHS
- Budgets: $PRIMARY_BUDGETS
- Planned runs: $imdb_count

## External validation on BGL

After HDFS metrics are available, the automation selects:

1. Strongest non-memory baseline from: $NON_MEMORY_CANDIDATES
2. Strongest existing memory baseline from: $EXISTING_MEMORY_CANDIDATES
3. BudgetMem-R

Selection is based on mean HDFS validation performance over completed seeds and
matched settings. If no parseable HDFS metric is available, the controlled
fallbacks are GRU and Memory Caching, and that fallback is explicitly recorded.
BGL uses five seeds, sequence length $BGL_LENGTH, and budget $BGL_BUDGET.

## Execution controls

- Each run has an immutable generated YAML configuration.
- Completed runs are skipped using \`.done\` markers.
- Failed and OOM runs are recorded separately.
- \`RETRY_FAILED=1\` permits targeted retries.
- Raw logs, checkpoints, and runtime state remain under \`artifacts/section18/\`
  and are excluded from Git.
- Controlled matrices, documentation, and summaries are written under
  \`configs/experiments/section18/\`, \`reports/\`, and \`docs/\`.
- Maximum parallel runs: $MAX_PARALLEL_RUNS
- Current runner template: ${RUNNER_TEMPLATE:-NOT DETECTED}
- Planned non-BGL runs: $total

## Completion gates

- Six-task primary synthetic matrix generated.
- Seven sequence lengths represented.
- Five primary models represented.
- Five seeds represented.
- Budgets 32 and 64 represented.
- HDFS and IMDb real-world matrices generated.
- BGL strongest-three selection recorded.
- Every attempted run has PASS, FAILED, OOM, or SKIPPED status.
- No memory-budget violation may be accepted.
EOF
}

# ----------------------------- Matrix validation -------------------------
validate_matrix() {
  "$PYTHON_BIN" - "$MATRIX_CSV" "$FULL_SECONDARY" <<'PY'
import csv, sys
from collections import Counter

path = sys.argv[1]
full_secondary = bool(int(sys.argv[2]))
rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
if not rows:
    raise SystemExit("matrix is empty")

required = {
    "run_id", "phase", "task", "dataset", "model", "sequence_length",
    "memory_budget", "retrieval_k", "seed", "config_path", "run_dir"
}
missing = required - set(rows[0])
if missing:
    raise SystemExit(f"missing columns: {sorted(missing)}")

run_ids = [r["run_id"] for r in rows]
if len(run_ids) != len(set(run_ids)):
    dupes = [k for k,v in Counter(run_ids).items() if v > 1]
    raise SystemExit(f"duplicate run ids: {dupes[:10]}")

primary = [r for r in rows if r["phase"] == "synthetic-primary"]
assert len({r["task"] for r in primary}) == 6, "primary matrix must contain six tasks"
assert len({r["sequence_length"] for r in primary}) == 7, "primary matrix must contain seven lengths"
assert len({r["model"] for r in primary}) == 5, "primary matrix must contain five models"
assert len({r["seed"] for r in primary}) == 5, "primary matrix must contain five seeds"
assert set(r["memory_budget"] for r in primary) == {"32", "64"}, "primary budgets must be 32 and 64"
assert len(primary) == 6 * 7 * 5 * 5 * 2, f"expected 2100 primary rows, found {len(primary)}"

hdfs = [r for r in rows if r["phase"] == "hdfs"]
imdb = [r for r in rows if r["phase"] == "imdb"]
assert len({r["model"] for r in hdfs}) == 7, "HDFS must contain seven requested models"
assert len({r["seed"] for r in hdfs}) == 5, "HDFS must contain five seeds"
assert set(r["memory_budget"] for r in hdfs) == {"32", "64"}, "HDFS budgets must be 32 and 64"
assert len({r["model"] for r in imdb}) == 7, "IMDb must contain seven requested models"
assert len({r["seed"] for r in imdb}) == 5, "IMDb must contain five seeds"
assert set(r["sequence_length"] for r in imdb) == {"1024", "2048", "4096"}, \
    "IMDb lengths must be 1024, 2048, and 4096"

for r in rows:
    if not __import__("pathlib").Path(r["config_path"]).is_file():
        raise SystemExit(f"missing config: {r['config_path']}")

print(f"Matrix validation: PASS ({len(rows)} non-BGL runs)")
PY
}

# ----------------------------- Command rendering -------------------------
shell_quote() {
  printf '%q' "$1"
}

render_command() {
  local template="$1"
  local config="$2"
  local run_dir="$3"
  local run_id="$4"
  local phase="$5"
  local model="$6"
  local task="$7"
  local dataset="$8"
  local length="$9"
  local budget="${10}"
  local seed="${11}"

  local command="$template"
  command="${command//\{config\}/$(shell_quote "$config")}"
  command="${command//\{run_dir\}/$(shell_quote "$run_dir")}"
  command="${command//\{run_id\}/$(shell_quote "$run_id")}"
  command="${command//\{phase\}/$(shell_quote "$phase")}"
  command="${command//\{model\}/$(shell_quote "$model")}"
  command="${command//\{task\}/$(shell_quote "$task")}"
  command="${command//\{dataset\}/$(shell_quote "$dataset")}"
  command="${command//\{sequence_length\}/$(shell_quote "$length")}"
  command="${command//\{budget\}/$(shell_quote "$budget")}"
  command="${command//\{seed\}/$(shell_quote "$seed")}"
  command="${command//\{retrieval_k\}/$(shell_quote "$RETRIEVAL_K")}"
  printf '%s' "$command"
}

# ----------------------------- Run execution -----------------------------
init_status_files() {
  if [[ ! -f "$STATUS_CSV" ]]; then
    cat > "$STATUS_CSV" <<'EOF'
run_id,phase,model,task,dataset,sequence_length,memory_budget,seed,status,exit_code,start_utc,end_utc,duration_seconds,config_sha256,log_path,metric_name,metric_value
EOF
  fi
  if [[ ! -f "$FAILURES_CSV" ]]; then
    cat > "$FAILURES_CSV" <<'EOF'
run_id,phase,model,task,dataset,sequence_length,memory_budget,seed,status,exit_code,log_path,reason
EOF
  fi
  : > "$COMMANDS_FILE"
  chmod +x "$COMMANDS_FILE"
  printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n\n' > "$COMMANDS_FILE"
}

status_for_run() {
  local run_id="$1"
  [[ -f "$STATUS_CSV" ]] || return 0
  awk -F, -v id="\"$run_id\"" '$1==id {gsub(/"/,"",$9); s=$9} END{print s}' "$STATUS_CSV"
}

extract_metric() {
  local run_dir="$1"
  "$PYTHON_BIN" - "$run_dir" "$PRIMARY_METRIC" <<'PY'
from pathlib import Path
import csv, json, math, re, sys

root = Path(sys.argv[1])
requested = sys.argv[2]
priority = [
    "long_range_recall", "memory_recall", "successful_long_range_retrievals",
    "token_accuracy", "exact_match_accuracy", "accuracy", "macro_f1", "f1",
    "roc_auc", "auroc", "validation_score", "eval_score", "score"
]
if requested != "auto":
    priority = [requested] + [x for x in priority if x != requested]

values = {}

def add(key, value):
    key = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(value):
        values.setdefault(key, []).append(value)

for p in root.rglob("*.json"):
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            for k,v in x.items():
                if isinstance(v, (dict,list)):
                    stack.append(v)
                else:
                    add(k,v)
        elif isinstance(x, list):
            stack.extend(x)

for p in root.rglob("*.csv"):
    try:
        with p.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for k,v in row.items():
                    add(k,v)
    except Exception:
        continue

for name in priority:
    if name in values:
        print(f"{name}\t{values[name][-1]}")
        raise SystemExit(0)
print("\t")
PY
}

append_status() {
  local run_id="$1" phase="$2" model="$3" task="$4" dataset="$5"
  local length="$6" budget="$7" seed="$8" status="$9" exit_code="${10}"
  local start="${11}" end="${12}" duration="${13}" config_hash="${14}"
  local log_path="${15}" metric_name="${16}" metric_value="${17}"

  {
    csv_escape "$run_id"; printf ','
    csv_escape "$phase"; printf ','
    csv_escape "$model"; printf ','
    csv_escape "$task"; printf ','
    csv_escape "$dataset"; printf ','
    printf '%s,%s,%s,' "$length" "$budget" "$seed"
    csv_escape "$status"; printf ',%s,' "$exit_code"
    csv_escape "$start"; printf ','
    csv_escape "$end"; printf ',%s,' "$duration"
    csv_escape "$config_hash"; printf ','
    csv_escape "$log_path"; printf ','
    csv_escape "$metric_name"; printf ','
    csv_escape "$metric_value"; printf '\n'
  } >> "$STATUS_CSV"
}

append_failure() {
  local run_id="$1" phase="$2" model="$3" task="$4" dataset="$5"
  local length="$6" budget="$7" seed="$8" status="$9" exit_code="${10}"
  local log_path="${11}" reason="${12}"
  {
    csv_escape "$run_id"; printf ','
    csv_escape "$phase"; printf ','
    csv_escape "$model"; printf ','
    csv_escape "$task"; printf ','
    csv_escape "$dataset"; printf ','
    printf '%s,%s,%s,' "$length" "$budget" "$seed"
    csv_escape "$status"; printf ',%s,' "$exit_code"
    csv_escape "$log_path"; printf ','
    csv_escape "$reason"; printf '\n'
  } >> "$FAILURES_CSV"
}

run_one() {
  local run_id="$1" phase="$2" task="$3" dataset="$4" model="$5"
  local length="$6" budget="$7" seed="$8" config="$9" run_dir="${10}"

  local done_marker="$run_dir/.done"
  local failed_marker="$run_dir/.failed"
  local oom_marker="$run_dir/.oom"
  local running_marker="$run_dir/.running"
  local log_path="$LOG_ROOT/$phase/$run_id.log"
  mkdir -p "$run_dir" "$(dirname "$log_path")"

  if [[ -f "$done_marker" ]]; then
    log "SKIP completed: $run_id"
    return 0
  fi

  local prior_status
  prior_status="$(status_for_run "$run_id")"
  if [[ "$RETRY_FAILED" != "1" && ( "$prior_status" == "FAILED" || "$prior_status" == "OOM" ) ]]; then
    log "SKIP previous $prior_status: $run_id"
    return 0
  fi

  rm -f "$failed_marker" "$oom_marker"
  printf '%s\n' "$START_UTC" > "$running_marker"

  local command
  command="$(render_command "$RUNNER_TEMPLATE" "$config" "$run_dir" "$run_id" \
            "$phase" "$model" "$task" "$dataset" "$length" "$budget" "$seed")"
  printf '%s\n' "$command" >> "$COMMANDS_FILE"

  local start_epoch end_epoch duration start_iso end_iso exit_code status reason
  start_epoch="$(date +%s)"
  start_iso="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  log "RUN $run_id"

  set +e
  (
    export BUDGETMEM_RUN_ID="$run_id"
    export BUDGETMEM_PHASE="$phase"
    export BUDGETMEM_MODEL="$model"
    export BUDGETMEM_TASK="$task"
    export BUDGETMEM_DATASET="$dataset"
    export BUDGETMEM_SEQUENCE_LENGTH="$length"
    export BUDGETMEM_MEMORY_BUDGET="$budget"
    export BUDGETMEM_SEED="$seed"
    export BUDGETMEM_RETRIEVAL_K="$RETRIEVAL_K"
    export BUDGETMEM_OUTPUT_DIR="$run_dir"
    eval "$command"
  ) > "$log_path" 2>&1
  exit_code=$?
  set -e
  cat "$log_path"

  end_epoch="$(date +%s)"
  end_iso="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  duration="$((end_epoch - start_epoch))"
  rm -f "$running_marker"

  status="PASS"
  reason=""
  if ((exit_code != 0)); then
    if grep -Eqi 'out of memory|cuda.*oom|cannot allocate memory|std::bad_alloc|killed process' "$log_path"; then
      status="OOM"
      reason="Out-of-memory signature detected"
      touch "$oom_marker"
    else
      status="FAILED"
      reason="Runner returned nonzero exit code"
      touch "$failed_marker"
    fi
  elif grep -Eqi 'memory[ _-]*budget.*(violation|exceeded|fail)|budget correctness.*fail' "$log_path"; then
    status="FAILED"
    reason="Memory-budget violation detected in log"
    exit_code=90
    touch "$failed_marker"
  else
    touch "$done_marker"
  fi

  local metric metric_name metric_value
  metric="$(extract_metric "$run_dir")"
  metric_name="${metric%%$'\t'*}"
  metric_value="${metric#*$'\t'}"
  [[ "$metric" == *$'\t'* ]] || { metric_name=""; metric_value=""; }

  append_status "$run_id" "$phase" "$model" "$task" "$dataset" "$length" \
                "$budget" "$seed" "$status" "$exit_code" "$start_iso" "$end_iso" \
                "$duration" "$(sha256_file "$config")" "$log_path" \
                "$metric_name" "$metric_value"

  if [[ "$status" != "PASS" ]]; then
    append_failure "$run_id" "$phase" "$model" "$task" "$dataset" "$length" \
                   "$budget" "$seed" "$status" "$exit_code" "$log_path" "$reason"
    log "$status $run_id"
    [[ "$FAIL_FAST" == "1" ]] && return "$exit_code"
  else
    log "PASS $run_id"
  fi
  return 0
}

execute_matrix_phase() {
  local selected_phase="$1"
  validate_runner
  init_status_files

  local active=0 failures=0
  while IFS=, read -r q_run_id q_phase q_task q_dataset q_model length budget k seed q_config q_run_dir q_role; do
    [[ "$q_run_id" == "run_id" ]] && continue

    local run_id="${q_run_id%\"}"; run_id="${run_id#\"}"
    local phase="${q_phase%\"}"; phase="${phase#\"}"
    local task="${q_task%\"}"; task="${task#\"}"
    local dataset="${q_dataset%\"}"; dataset="${dataset#\"}"
    local model="${q_model%\"}"; model="${model#\"}"
    local config="${q_config%\"}"; config="${config#\"}"
    local run_dir="${q_run_dir%\"}"; run_dir="${run_dir#\"}"

    [[ "$phase" == "$selected_phase" ]] || continue

    if ((MAX_PARALLEL_RUNS == 1)); then
      run_one "$run_id" "$phase" "$task" "$dataset" "$model" "$length" \
              "$budget" "$seed" "$config" "$run_dir" || failures=$((failures + 1))
    else
      run_one "$run_id" "$phase" "$task" "$dataset" "$model" "$length" \
              "$budget" "$seed" "$config" "$run_dir" &
      active=$((active + 1))
      if ((active >= MAX_PARALLEL_RUNS)); then
        if ! wait -n; then failures=$((failures + 1)); fi
        active=$((active - 1))
      fi
    fi
  done < "$MATRIX_CSV"

  while ((active > 0)); do
    if ! wait -n; then failures=$((failures + 1)); fi
    active=$((active - 1))
  done

  ((failures == 0)) || warn "$failures worker process(es) returned nonzero status."
}

# ----------------------------- BGL selection -----------------------------
select_bgl_methods() {
  "$PYTHON_BIN" - "$STATUS_CSV" "$NON_MEMORY_CANDIDATES" \
    "$EXISTING_MEMORY_CANDIDATES" "$EVIDENCE_ROOT/bgl_method_selection.txt" <<'PY'
import csv, math, statistics, sys
from collections import defaultdict
from pathlib import Path

status_path = Path(sys.argv[1])
non_memory = sys.argv[2].split()
existing_memory = sys.argv[3].split()
out_path = Path(sys.argv[4])

scores = defaultdict(list)
metric_names = defaultdict(set)
if status_path.exists():
    for row in csv.DictReader(status_path.open(newline="", encoding="utf-8")):
        if row.get("phase") != "hdfs" or row.get("status") != "PASS":
            continue
        try:
            value = float(row.get("metric_value", ""))
        except ValueError:
            continue
        if math.isfinite(value):
            scores[row["model"]].append(value)
            if row.get("metric_name"):
                metric_names[row["model"]].add(row["metric_name"])

def choose(candidates, fallback):
    valid = [(statistics.mean(scores[m]), m) for m in candidates if scores[m]]
    if not valid:
        return fallback, None
    valid.sort(reverse=True)
    mean, model = valid[0]
    return model, mean

non_model, non_score = choose(non_memory, "gru")
mem_model, mem_score = choose(existing_memory, "memory_caching")

lines = [
    "Section 18 BGL strongest-three selection",
    f"strongest_non_memory={non_model}",
    f"strongest_non_memory_mean={non_score if non_score is not None else 'FALLBACK'}",
    f"strongest_existing_memory={mem_model}",
    f"strongest_existing_memory_mean={mem_score if mem_score is not None else 'FALLBACK'}",
    "budgetmem_method=budgetmem_r",
]
for model in sorted(m for m, values in scores.items() if values):
    lines.append(
        f"hdfs_{model}_mean={statistics.mean(scores[model]):.12g};"
        f"n={len(scores[model])};metrics={','.join(sorted(metric_names[model]))}"
    )
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(non_model)
print(mem_model)
PY
}

append_bgl_matrix() {
  local selection_file="$EVIDENCE_ROOT/bgl_method_selection.txt"
  local selected
  selected="$(select_bgl_methods)"
  local strongest_non_memory strongest_memory
  strongest_non_memory="$(printf '%s\n' "$selected" | sed -n '1p')"
  strongest_memory="$(printf '%s\n' "$selected" | sed -n '2p')"

  # Remove previously generated BGL rows and configs, then recreate.
  "$PYTHON_BIN" - "$MATRIX_CSV" <<'PY'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
rows = [r for r in rows if r["phase"] != "bgl"]
with p.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=[
        "run_id","phase","task","dataset","model","sequence_length",
        "memory_budget","retrieval_k","seed","config_path","run_dir","external_role"
    ])
    w.writeheader()
    w.writerows(rows)
PY
  rm -rf "$CONFIG_ROOT/bgl"
  mkdir -p "$CONFIG_ROOT/bgl"

  local -a seeds
  IFS=' ' read -r -a seeds <<< "$SEEDS"
  local seed
  for seed in "${seeds[@]}"; do
    add_matrix_row "bgl" "bgl_log_anomaly" "bgl" "$strongest_non_memory" \
                   "$BGL_LENGTH" "$BGL_BUDGET" "$seed" "strongest_non_memory"
    add_matrix_row "bgl" "bgl_log_anomaly" "bgl" "$strongest_memory" \
                   "$BGL_LENGTH" "$BGL_BUDGET" "$seed" "strongest_existing_memory"
    add_matrix_row "bgl" "bgl_log_anomaly" "bgl" "budgetmem_r" \
                   "$BGL_LENGTH" "$BGL_BUDGET" "$seed" "budgetmem_r"
  done
  log "BGL methods: $strongest_non_memory, $strongest_memory, budgetmem_r"
}

# ----------------------------- Summaries ---------------------------------
deduplicate_status() {
  [[ -f "$STATUS_CSV" ]] || return 0
  "$PYTHON_BIN" - "$STATUS_CSV" <<'PY'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
latest = {}
order = []
for row in rows:
    rid = row["run_id"]
    if rid not in latest:
        order.append(rid)
    latest[rid] = row
with p.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [
        "run_id","phase","model","task","dataset","sequence_length",
        "memory_budget","seed","status","exit_code","start_utc","end_utc",
        "duration_seconds","config_sha256","log_path","metric_name","metric_value"
    ])
    w.writeheader()
    for rid in order:
        w.writerow(latest[rid])
PY
}

write_summary() {
  deduplicate_status
  "$PYTHON_BIN" - "$MATRIX_CSV" "$STATUS_CSV" "$FAILURES_CSV" "$SUMMARY_TXT" <<'PY'
import csv, sys
from collections import Counter, defaultdict
from pathlib import Path

matrix_path, status_path, failure_path, out_path = map(Path, sys.argv[1:])
matrix = list(csv.DictReader(matrix_path.open(newline="", encoding="utf-8")))
status = list(csv.DictReader(status_path.open(newline="", encoding="utf-8"))) if status_path.exists() else []
latest = {r["run_id"]: r for r in status}

phase_counts = Counter(r["phase"] for r in matrix)
status_counts = Counter(r["status"] for r in latest.values())
phase_status = defaultdict(Counter)
for r in latest.values():
    phase_status[r["phase"]][r["status"]] += 1

attempted = len(latest)
planned = len(matrix)
passed = status_counts["PASS"]
failed = status_counts["FAILED"]
oom = status_counts["OOM"]
remaining = sum(1 for r in matrix if r["run_id"] not in latest or latest[r["run_id"]]["status"] != "PASS")

lines = [
    "Section 18 Main Experiment Matrix Summary",
    f"planned_runs={planned}",
    f"attempted_unique_runs={attempted}",
    f"pass={passed}",
    f"failed={failed}",
    f"oom={oom}",
    f"remaining_not_passed={remaining}",
]
for phase in sorted(phase_counts):
    counts = phase_status[phase]
    lines.append(
        f"phase={phase};planned={phase_counts[phase]};"
        f"pass={counts['PASS']};failed={counts['FAILED']};oom={counts['OOM']}"
    )

required_primary = [r for r in matrix if r["phase"] == "synthetic-primary"]
primary_complete = all(latest.get(r["run_id"], {}).get("status") == "PASS" for r in required_primary)
hdfs_rows = [r for r in matrix if r["phase"] == "hdfs"]
imdb_rows = [r for r in matrix if r["phase"] == "imdb"]
bgl_rows = [r for r in matrix if r["phase"] == "bgl"]

lines += [
    f"primary_synthetic_complete={'PASS' if primary_complete else 'INCOMPLETE'}",
    f"hdfs_complete={'PASS' if hdfs_rows and all(latest.get(r['run_id'],{}).get('status') == 'PASS' for r in hdfs_rows) else 'INCOMPLETE'}",
    f"imdb_complete={'PASS' if imdb_rows and all(latest.get(r['run_id'],{}).get('status') == 'PASS' for r in imdb_rows) else 'INCOMPLETE'}",
    f"bgl_complete={'PASS' if bgl_rows and all(latest.get(r['run_id'],{}).get('status') == 'PASS' for r in bgl_rows) else 'INCOMPLETE'}",
    f"memory_budget_violations={'FAIL' if any(r.get('exit_code') == '90' for r in latest.values()) else 'PASS'}",
]
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
}

# ----------------------------- Controlled commit -------------------------
commit_controlled_artifacts() {
  [[ "$COMMIT_CONTROLLED" == "1" ]] || return 0

  git add \
    "$CONFIG_ROOT" \
    "$TABLE_ROOT" \
    "$PROTOCOL_MD" \
    "$TASK_RECORD" \
    "$RUNNER_RECORD" \
    "$EVIDENCE_ROOT/bgl_method_selection.txt" \
    "$SUMMARY_TXT" \
    .gitignore \
    "$AUTOMATION_NAME" 2>/dev/null || true

  if git diff --cached --quiet; then
    log "No controlled Section 18 changes to commit."
  else
    git commit -m "Run Section 18 main experiment matrix"
  fi

  if [[ "$PUSH_CONTROLLED" == "1" ]]; then
    local branch
    branch="$(git branch --show-current)"
    git push -u origin "$branch"
  fi
}

# ----------------------------- Main --------------------------------------
log "Repository: $ROOT"
log "Action: $ACTION"
log "Phase: $PHASE"
log "Synthetic tasks: $SYNTHETIC_TASKS"

build_matrix
write_protocol
validate_matrix

if [[ "$ACTION" == "validate" ]]; then
  validate_runner
  log "Section 18 validation: PASS"
  exit 0
fi

if [[ "$ACTION" == "plan" ]]; then
  append_bgl_matrix
  write_protocol
  validate_matrix
  write_summary
  commit_controlled_artifacts
  cat <<EOF

Section 18 plan generated successfully.

Matrix:
  $MATRIX_CSV

Protocol:
  $PROTOCOL_MD

Runner:
  ${RUNNER_TEMPLATE:-NOT DETECTED}

No training was started.
Execute all phases with:
  ./$AUTOMATION_NAME --execute

Execute one phase with:
  ./$AUTOMATION_NAME --execute --phase synthetic-primary
EOF
  exit 0
fi

if [[ "$ACTION" == "summarize" ]]; then
  append_bgl_matrix
  write_protocol
  write_summary
  commit_controlled_artifacts
  exit 0
fi

# Execution ordering ensures BGL selection uses completed HDFS evidence.
if phase_selected "synthetic-primary"; then
  execute_matrix_phase "synthetic-primary"
  write_summary
fi
if phase_selected "synthetic-secondary"; then
  execute_matrix_phase "synthetic-secondary"
  write_summary
fi
if phase_selected "hdfs"; then
  execute_matrix_phase "hdfs"
  write_summary
fi
if phase_selected "imdb"; then
  execute_matrix_phase "imdb"
  write_summary
fi

append_bgl_matrix
write_protocol
validate_matrix

if phase_selected "bgl"; then
  execute_matrix_phase "bgl"
fi

write_summary
commit_controlled_artifacts

log "Section 18 automation finished."
log "Summary: $SUMMARY_TXT"
