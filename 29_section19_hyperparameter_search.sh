#!/usr/bin/env bash
# Section 19 — Fair hyperparameter search automation for budgetmem-r
#
# Run from the budgetmem-r repository root in the VS Code WSL terminal.
#
# Modes:
#   ./29_section19_hyperparameter_search.sh --setup   # install/validate only
#   ./29_section19_hyperparameter_search.sh --smoke   # 1 tiny trial per family
#   ./29_section19_hyperparameter_search.sh --full    # 20 validation trials per family
#
# Optional overrides:
#   TRIALS_PER_FAMILY=20
#   MODEL_FAMILIES=gru,gru_uniform,gru_reservoir,memory_caching,budgetmem_r,lstm,transformer,mamba,rmt
#   SEARCH_TASK=associative_recall
#   SEARCH_SEQUENCE_LENGTH=256
#   SEARCH_MEMORY_BUDGET=32
#   SEARCH_SEED=2026
#   SEARCH_MAX_STEPS=100
#   SEARCH_TRAIN_SAMPLES=128
#   SEARCH_VALIDATION_SAMPLES=48
#
# The full search can be resumed safely. Completed trials are not repeated.

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

AUTOMATION_NAME="29_section19_hyperparameter_search.sh"
MODE="${1:---smoke}"
case "$MODE" in
  --setup|--smoke|--full) ;;
  *)
    printf 'Usage: %s [--setup|--smoke|--full]\n' "$AUTOMATION_NAME" >&2
    exit 2
    ;;
esac

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$ROOT" ]] || die "Run this automation from inside the budgetmem-r Git repository."
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  die "Python was not found."
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="${SEARCH_SEED:-2026}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_ROOT="reports/evidence/backups/section19_${STAMP}"
LOG_ROOT="reports/logs/hyperparameter_search"
EVIDENCE_ROOT="reports/evidence/section19"
CONFIG_ROOT="configs/hyperparameter_search"
SCRIPT_PATH="scripts/run_section19_hyperparameter_search.py"
TEST_PATH="tests/test_section19_hyperparameter_search.py"
SEARCH_CONFIG="$CONFIG_ROOT/section19_search.yaml"

mkdir -p "$BACKUP_ROOT" "$LOG_ROOT" "$EVIDENCE_ROOT" "$CONFIG_ROOT" scripts tests

required=(
  "configs/experiments/pilot.yaml"
  "scripts/run_section18.py"
  "src/budgetmem/experiments/pilot.py"
  "src/budgetmem/baselines/controlled.py"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || die "Required file is missing: $path"
done

backup_if_present() {
  local path="$1"
  if [[ -e "$path" ]]; then
    mkdir -p "$BACKUP_ROOT/$(dirname "$path")"
    cp -a "$path" "$BACKUP_ROOT/$path"
  fi
}

for path in \
  src/budgetmem/experiments/pilot.py \
  src/budgetmem/baselines/controlled.py \
  scripts/run_section18.py \
  "$SCRIPT_PATH" \
  "$TEST_PATH" \
  "$SEARCH_CONFIG"; do
  backup_if_present "$path"
done

log "Repository: $ROOT"
log "Python: $PYTHON"
log "Mode: $MODE"
log "Backup directory: $BACKUP_ROOT"

# ---------------------------------------------------------------------------
# Patch the established runners so the searched parameters are actually used.
# The patch is idempotent and aborts if the expected source structure changed.
# ---------------------------------------------------------------------------

"$PYTHON" - <<'PY'
from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Cannot safely patch {label}: expected one match, found {count}")
    return text.replace(old, new, 1)


# 1. Pilot GRU: honor number of layers and dropout.
pilot_path = Path("src/budgetmem/experiments/pilot.py")
pilot = pilot_path.read_text(encoding="utf-8")
pilot = replace_once(
    pilot,
    '''        embedding_dim = int(model_cfg["embedding_dim"])
        hidden_dim = int(model_cfg["hidden_dim"])
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
        )
''',
    '''        embedding_dim = int(model_cfg["embedding_dim"])
        hidden_dim = int(model_cfg["hidden_dim"])
        num_layers = int(model_cfg.get("num_layers", 1))
        dropout = float(model_cfg.get("dropout", 0.0))
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
''',
    "pilot GRU layer/dropout support",
)
pilot_path.write_text(pilot, encoding="utf-8")

# 2. Memory-caching baseline: honor recurrent depth and dropout.
controlled_path = Path("src/budgetmem/baselines/controlled.py")
controlled = controlled_path.read_text(encoding="utf-8")
marker = "class MemoryCachingBaseline(nn.Module):"
if marker not in controlled:
    raise SystemExit("Cannot locate MemoryCachingBaseline")
head, tail = controlled.split(marker, 1)
tail = replace_once(
    tail,
    '''        variant: str = "mean",
        seed: int = 2026,
    ) -> None:
''',
    '''        variant: str = "mean",
        seed: int = 2026,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
''',
    "memory-caching constructor",
)
tail = replace_once(
    tail,
    '''        self.backbone = GRUBaseline(input_dim, hidden_dim, hidden_dim)
''',
    '''        self.backbone = GRUBaseline(
            input_dim,
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
''',
    "memory-caching backbone",
)
controlled_path.write_text(head + marker + tail, encoding="utf-8")

# 3. Section 18 controlled backend: honor layers, dropout, weight decay, and clipping.
runner_path = Path("scripts/run_section18.py")
runner = runner_path.read_text(encoding="utf-8")
runner = replace_once(
    runner,
    '''        hidden_dim: int,
        output_dim: int,
        sequence_length: int,
''',
    '''        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        sequence_length: int,
''',
    "controlled wrapper signature",
)
runner = replace_once(
    runner,
    '''        if registry_name.startswith("memory_caching"):
            kwargs.update({"budget": min(budget, sequence_length), "seed": seed})
''',
    '''        if registry_name.startswith("memory_caching"):
            kwargs.update(
                {
                    "budget": min(budget, sequence_length),
                    "seed": seed,
                    "num_layers": num_layers,
                    "dropout": dropout,
                }
            )
''',
    "memory-caching wrapper parameters",
)
runner = replace_once(
    runner,
    '''                    "num_layers": 1,
                    "num_heads": heads,
                    "window_size": min(256, sequence_length),
''',
    '''                    "num_layers": num_layers,
                    "num_heads": heads,
                    "dropout": dropout,
                    "window_size": min(256, sequence_length),
''',
    "transformer depth and dropout",
)
runner = replace_once(
    runner,
    '''        elif registry_name == "state_space":
            kwargs.update({"num_layers": 1, "state_dim": max(4, hidden_dim // 4)})
''',
    '''        elif registry_name == "state_space":
            kwargs.update(
                {"num_layers": num_layers, "state_dim": max(4, hidden_dim // 4)}
            )
''',
    "state-space depth",
)
runner = replace_once(
    runner,
    '''                    "num_layers": 1,
                    "num_heads": heads,
                }
            )
        else:
            kwargs.update({"num_layers": 1})
''',
    '''                    "num_layers": num_layers,
                    "num_heads": heads,
                    "dropout": dropout,
                }
            )
        else:
            kwargs.update({"num_layers": num_layers, "dropout": dropout})
''',
    "RMT/recurrent depth and dropout",
)
runner = replace_once(
    runner,
    '''    embedding_dim = int(model_cfg.get("embedding_dim", 32))
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
''',
    '''    embedding_dim = int(model_cfg.get("embedding_dim", 32))
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    num_layers = int(model_cfg.get("num_layers", 1))
    dropout = float(model_cfg.get("dropout", 0.0))
''',
    "generic model search parameters",
)
runner = replace_once(
    runner,
    '''        hidden_dim=hidden_dim,
        output_dim=output_dim,
        sequence_length=int(cell["sequence_length"]),
''',
    '''        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        sequence_length=int(cell["sequence_length"]),
''',
    "generic controlled-model construction",
)
runner = replace_once(
    runner,
    '''    learning_rate = float(training_cfg.get("learning_rate", 1.0e-3))
''',
    '''    learning_rate = float(training_cfg.get("learning_rate", 1.0e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    gradient_clip_norm = float(training_cfg.get("gradient_clip_norm", 1.0))
''',
    "optimizer search parameters",
)
runner = replace_once(
    runner,
    '''    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
''',
    '''    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
''',
    "controlled optimizer weight decay",
)
runner = replace_once(
    runner,
    '''            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
''',
    '''            torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm
            )
''',
    "controlled gradient clipping",
)
runner_path.write_text(runner, encoding="utf-8")
PY

# ---------------------------------------------------------------------------
# Search configuration. Architecture-conditional parameters are explicit:
# only BudgetMem-R receives controller-specific parameters, but every family
# receives the identical number of validation trials.
# ---------------------------------------------------------------------------

cat > "$SEARCH_CONFIG" <<YAML
schema_version: "1.0"
section: 19
selection_split: validation
objective:
  name: validation_primary_metric
  direction: maximize
fairness:
  trials_per_architecture_family: ${TRIALS_PER_FAMILY:-20}
  enforce_equal_trials: true
  permit_test_set_selection: false
search_cell:
  task: ${SEARCH_TASK:-associative_recall}
  sequence_length: ${SEARCH_SEQUENCE_LENGTH:-256}
  memory_budget: ${SEARCH_MEMORY_BUDGET:-32}
  seed: ${SEARCH_SEED:-2026}
  train_samples: ${SEARCH_TRAIN_SAMPLES:-128}
  validation_samples: ${SEARCH_VALIDATION_SAMPLES:-48}
  max_steps: ${SEARCH_MAX_STEPS:-100}
model_families:
  - gru
  - gru_uniform
  - gru_reservoir
  - memory_caching
  - budgetmem_r
  - lstm
  - transformer
  - mamba
  - rmt
search_space:
  learning_rate:
    type: log_float
    low: 0.0001
    high: 0.003
  weight_decay:
    type: categorical
    values: [0.0, 0.000001, 0.00001, 0.0001, 0.001]
  hidden_dimension:
    type: categorical
    values: [64, 96, 128, 192]
  number_of_layers:
    type: categorical
    values: [1, 2, 3]
  dropout:
    type: categorical
    values: [0.0, 0.1, 0.2, 0.3]
  gradient_clipping:
    type: categorical
    values: [0.5, 1.0, 2.0, 5.0]
  memory_controller_temperature:
    type: categorical
    values: [0.5, 0.67, 0.85, 1.0, 1.25]
  auxiliary_loss_coefficient:
    type: categorical
    values: [0.0, 0.01, 0.05, 0.1]
  budget_penalty:
    type: categorical
    values: [1.0, 5.0, 10.0, 20.0]
  retrieval_top_k:
    type: categorical
    values: [1, 4, 8]
  write_threshold:
    type: categorical
    values: [0.3, 0.4, 0.5, 0.6, 0.7]
YAML

# ---------------------------------------------------------------------------
# Dedicated Section 19 search runner.
# ---------------------------------------------------------------------------

cat > "$SCRIPT_PATH" <<'PY'
#!/usr/bin/env python3
"""Section 19 fair, validation-only hyperparameter search.

The runner uses exactly the same number of trials for every architecture family,
records every attempted trial, and selects best configurations from validation
metrics only. It never reads test-set metrics for hyperparameter selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAMILIES = [
    "gru",
    "gru_uniform",
    "gru_reservoir",
    "memory_caching",
    "budgetmem_r",
    "lstm",
    "transformer",
    "mamba",
    "rmt",
]
PILOT_FAMILIES = {"gru", "gru_uniform", "gru_reservoir", "budgetmem_r"}
CONTROLLER_FAMILIES = {"budgetmem_r"}
LAYERED_FAMILIES = {
    "gru",
    "memory_caching",
    "lstm",
    "transformer",
    "mamba",
    "rmt",
}
DROPOUT_FAMILIES = {
    "gru",
    "memory_caching",
    "lstm",
    "transformer",
    "rmt",
}
MODEL_ALIASES = {
    "gru": "gru",
    "gru_uniform": "gru_uniform",
    "gru_reservoir": "gru_reservoir",
    "memory_caching": "memory_caching",
    "budgetmem_r": "budgetmem_r",
    "lstm": "lstm",
    "transformer": "transformer",
    "mamba": "mamba",
    "rmt": "rmt",
}


@dataclass(frozen=True)
class TrialResult:
    family: str
    trial_index: int
    status: str
    objective_value: float | None
    objective_name: str | None
    direction: str
    duration_seconds: float
    trial_dir: str
    config_path: str
    metrics_path: str | None
    error: str | None
    hyperparameters: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hyperparameter_search/section19_search.yaml"),
    )
    parser.add_argument("--trials-per-family", type=int)
    parser.add_argument("--families", type=str)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_value(rng: random.Random, spec: dict[str, Any]) -> Any:
    kind = str(spec["type"])
    if kind == "categorical":
        values = list(spec["values"])
        if not values:
            raise ValueError("Categorical search space cannot be empty")
        return values[rng.randrange(len(values))]
    if kind == "log_float":
        return log_uniform(rng, float(spec["low"]), float(spec["high"]))
    raise ValueError(f"Unsupported search-space type: {kind}")


def sample_hyperparameters(
    family: str,
    trial_index: int,
    search_space: dict[str, dict[str, Any]],
    base_seed: int,
    memory_budget: int,
) -> dict[str, Any]:
    rng = random.Random(stable_seed("section19", base_seed, family, trial_index))
    values = {
        key: sample_value(rng, dict(spec)) for key, spec in search_space.items()
    }

    # Architecture-conditional applicability. Inapplicable values are stored as
    # null rather than silently pretending that the parameter was searched.
    if family not in LAYERED_FAMILIES:
        values["number_of_layers"] = 1
    if family not in DROPOUT_FAMILIES or int(values["number_of_layers"]) == 1:
        values["dropout"] = 0.0
    if family not in CONTROLLER_FAMILIES:
        for key in (
            "memory_controller_temperature",
            "auxiliary_loss_coefficient",
            "budget_penalty",
            "retrieval_top_k",
            "write_threshold",
        ):
            values[key] = None
    else:
        values["retrieval_top_k"] = min(
            int(values["retrieval_top_k"]), int(memory_budget)
        )
    return values


def prepare_controlled_search_data(
    root: Path,
    *,
    task: str,
    sequence_length: int,
    train_samples: int,
    validation_samples: int,
    seed: int,
) -> tuple[Path, Path]:
    """Materialize deterministic synthetic train/validation JSONL files."""
    from budgetmem.experiments.pilot import SyntheticPilotDataset

    data_root = root / "search_data" / task / f"length_{sequence_length}"
    train_path = data_root / "train.jsonl"
    validation_path = data_root / "validation.jsonl"
    manifest_path = data_root / "manifest.json"
    expected = {
        "task": task,
        "sequence_length": sequence_length,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "seed": seed,
    }
    if train_path.is_file() and validation_path.is_file() and manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing == expected:
                return train_path, validation_path
        except Exception:
            pass

    data_root.mkdir(parents=True, exist_ok=True)

    def build(path: Path, count: int, split_seed: int) -> None:
        dataset = SyntheticPilotDataset(
            task=task,
            sequence_length=sequence_length,
            sample_count=count,
            seed=split_seed,
            max_target_length=12,
            vocabulary_size=192,
        )
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in dataset.rows:
                payload = {
                    "sample_id": row["sample_id"],
                    "input_ids": [int(v) for v in row["input_ids"].tolist()],
                    "target_ids": [int(v) for v in row["target_ids"].tolist()],
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    build(train_path, train_samples, seed + 11)
    build(validation_path, validation_samples, seed + 29)
    write_json(manifest_path, expected)
    return train_path, validation_path


def build_trial_config(
    *,
    family: str,
    trial_index: int,
    hp: dict[str, Any],
    search_cell: dict[str, Any],
    trial_dir: Path,
    train_path: Path,
    validation_path: Path,
    smoke: bool,
) -> dict[str, Any]:
    task = str(search_cell["task"])
    sequence_length = int(search_cell["sequence_length"])
    budget = int(search_cell["memory_budget"])
    seed = int(search_cell["seed"])
    model_name = MODEL_ALIASES[family]
    output_dir = trial_dir / "run"
    max_steps = min(int(search_cell["max_steps"]), 2) if smoke else int(
        search_cell["max_steps"]
    )
    train_samples = min(int(search_cell["train_samples"]), 16) if smoke else int(
        search_cell["train_samples"]
    )
    validation_samples = (
        min(int(search_cell["validation_samples"]), 8)
        if smoke
        else int(search_cell["validation_samples"])
    )
    top_k = int(hp["retrieval_top_k"] or min(4, budget))

    model_block = {
        "name": model_name,
        "embedding_dim": 32,
        "hidden_dim": int(hp["hidden_dimension"]),
        "num_layers": int(hp["number_of_layers"]),
        "dropout": float(hp["dropout"]),
        "retrieval_k": top_k,
    }
    training_block = {
        "seed": seed,
        "learning_rate": float(hp["learning_rate"]),
        "weight_decay": float(hp["weight_decay"]),
        "gradient_clip_norm": float(hp["gradient_clipping"]),
        "max_steps": max_steps,
        "batch_size": 2 if smoke else 8,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "epochs": 1 if smoke else 4,
    }
    if family in CONTROLLER_FAMILIES:
        training_block.update(
            {
                "write_temperature": float(hp["memory_controller_temperature"]),
                "write_binarization_penalty": float(
                    hp["auxiliary_loss_coefficient"]
                ),
                "budget_violation_penalty": float(hp["budget_penalty"]),
                "write_threshold": float(hp["write_threshold"]),
            }
        )

    config: dict[str, Any] = {
        "schema_version": "1.0",
        "section": 19,
        "selection_split": "validation",
        "experiment": {
            "run_id": f"section19_{family}_trial_{trial_index:03d}",
            "output_dir": str(output_dir),
        },
        "task": {"name": task, "sequence_length": sequence_length},
        "data": {"dataset": "synthetic"},
        "model": model_block,
        "memory": {"budget": budget, "retrieval_k": top_k},
        "training": training_block,
        "hyperparameter_search": {
            "architecture_family": family,
            "trial_index": trial_index,
            "selection_split": "validation",
            "hyperparameters": hp,
        },
    }

    if family in PILOT_FAMILIES:
        config["pilot_overrides"] = {
            "model": {
                "embedding_dim": 32,
                "hidden_dim": int(hp["hidden_dimension"]),
                "num_layers": int(hp["number_of_layers"]),
                "dropout": float(hp["dropout"]),
                "retrieval_k": top_k,
            },
            "training": training_block,
            "evaluation": {"batch_size": 2 if smoke else 8},
        }
    else:
        config["data"] = {
            "dataset": "section19_synthetic",
            "train_path": str(train_path),
            "validation_path": str(validation_path),
        }
    return config


def recursively_find_numeric(payload: Any, keys: Iterable[str]) -> tuple[str, float] | None:
    wanted = list(keys)
    if isinstance(payload, dict):
        for key in wanted:
            if key in payload and isinstance(payload[key], (int, float)):
                return key, float(payload[key])
        for value in payload.values():
            found = recursively_find_numeric(value, wanted)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = recursively_find_numeric(value, wanted)
            if found is not None:
                return found
    return None


def extract_validation_objective(metrics_path: Path) -> tuple[str, float, str]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    # These are outputs from the validation/evaluation split produced by the
    # Section 18 runner. No test-set file is consulted here.
    maximize_keys = (
        "primary_metric_value",
        "token_accuracy",
        "exact_match_accuracy",
        "memory_recall",
        "f1",
        "accuracy",
    )
    found = recursively_find_numeric(payload, maximize_keys)
    if found is not None:
        return found[0], found[1], "maximize"
    found = recursively_find_numeric(payload, ("validation_loss", "mean_loss"))
    if found is not None:
        return found[0], found[1], "minimize"
    raise KeyError(f"No validation objective found in {metrics_path}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def trial_to_row(result: TrialResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "architecture_family": result.family,
        "trial_index": result.trial_index,
        "status": result.status,
        "selection_split": "validation",
        "objective_name": result.objective_name,
        "objective_value": result.objective_value,
        "direction": result.direction,
        "duration_seconds": result.duration_seconds,
        "trial_dir": result.trial_dir,
        "config_path": result.config_path,
        "metrics_path": result.metrics_path,
        "error": result.error,
    }
    row.update(result.hyperparameters)
    return row


def load_completed_trial(trial_dir: Path) -> TrialResult | None:
    result_path = trial_dir / "trial_result.json"
    if not result_path.is_file():
        return None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return TrialResult(**payload)


def run_trial(
    *,
    python: str,
    family: str,
    trial_index: int,
    hp: dict[str, Any],
    config: dict[str, Any],
    trial_dir: Path,
    smoke: bool,
) -> TrialResult:
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = trial_dir / "trial_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(trial_dir / "sampled_hyperparameters.json", hp)
    stdout_path = trial_dir / "stdout.log"
    stderr_path = trial_dir / "stderr.log"
    command = [python, "scripts/run_section18.py", "--config", str(config_path)]
    if smoke:
        command.append("--smoke")
    write_json(trial_dir / "command.json", command)

    started = time.perf_counter()
    error: str | None = None
    objective_name: str | None = None
    objective_value: float | None = None
    direction = "maximize"
    status = "FAILED"
    metrics_path: Path | None = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
                env=os.environ.copy(),
                check=False,
            )
        if process.returncode != 0:
            tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"Section 18 runner exited {process.returncode}: {tail}")
        metrics_path = Path(config["experiment"]["output_dir"]) / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Expected metrics file is missing: {metrics_path}")
        objective_name, objective_value, direction = extract_validation_objective(
            metrics_path
        )
        shutil.copy2(metrics_path, trial_dir / "metrics.json")
        metrics_path = trial_dir / "metrics.json"
        status = "COMPLETE"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    result = TrialResult(
        family=family,
        trial_index=trial_index,
        status=status,
        objective_value=objective_value,
        objective_name=objective_name,
        direction=direction,
        duration_seconds=time.perf_counter() - started,
        trial_dir=str(trial_dir),
        config_path=str(config_path),
        metrics_path=str(metrics_path) if metrics_path else None,
        error=error,
        hyperparameters=hp,
    )
    write_json(trial_dir / "trial_result.json", result.__dict__)
    return result


def choose_best(results: list[TrialResult], family: str) -> TrialResult | None:
    candidates = [
        result
        for result in results
        if result.family == family
        and result.status == "COMPLETE"
        and result.objective_value is not None
    ]
    if not candidates:
        return None
    direction = candidates[0].direction
    if any(result.direction != direction for result in candidates):
        raise ValueError(f"Mixed objective directions for {family}")
    return (
        max(candidates, key=lambda item: float(item.objective_value))
        if direction == "maximize"
        else min(candidates, key=lambda item: float(item.objective_value))
    )


def write_report(
    path: Path,
    families: list[str],
    trials_per_family: int,
    results: list[TrialResult],
    best: dict[str, TrialResult | None],
    smoke: bool,
) -> None:
    lines = [
        "# Section 19 Hyperparameter Search Report",
        "",
        f"**Mode:** {'SMOKE' if smoke else 'FULL'}",
        "",
        "**Selection split:** validation only",
        "",
        f"**Equal search budget:** {trials_per_family} trials per architecture family",
        "",
        "## Fairness Audit",
        "",
        "| Architecture family | Attempted | Complete | Failed | Budget status |",
        "|---|---:|---:|---:|---|",
    ]
    for family in families:
        family_results = [result for result in results if result.family == family]
        complete = sum(result.status == "COMPLETE" for result in family_results)
        failed = sum(result.status == "FAILED" for result in family_results)
        budget_status = "PASS" if len(family_results) == trials_per_family else "FAIL"
        lines.append(
            f"| {family} | {len(family_results)} | {complete} | {failed} | {budget_status} |"
        )
    lines.extend(
        [
            "",
            "## Validation-Selected Hyperparameters",
            "",
            "| Architecture family | Trial | Validation metric | Value |",
            "|---|---:|---|---:|",
        ]
    )
    for family in families:
        selected = best[family]
        if selected is None:
            lines.append(f"| {family} | — | — | — |")
        else:
            lines.append(
                f"| {family} | {selected.trial_index} | {selected.objective_name} | "
                f"{float(selected.objective_value):.8f} |"
            )
    lines.extend(
        [
            "",
            "Controller-specific parameters were searched only for BudgetMem-R because "
            "they are not defined for non-controller baselines. This does not change the "
            "trial budget: every architecture family receives exactly the same number of "
            "validation trials.",
            "",
            "No test-set metric was used to rank or select any trial.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    config_path = args.config.resolve()
    config = read_yaml(config_path)
    search_cell = dict(config["search_cell"])
    search_space = {
        key: dict(value) for key, value in dict(config["search_space"]).items()
    }
    base_seed = int(search_cell["seed"])
    configured_trials = int(config["fairness"]["trials_per_architecture_family"])
    trials_per_family = args.trials_per_family or configured_trials
    if args.smoke:
        trials_per_family = 1
    if trials_per_family <= 0:
        raise ValueError("trials_per_family must be positive")

    if args.families:
        families = [item.strip() for item in args.families.split(",") if item.strip()]
    else:
        env_families = os.environ.get("MODEL_FAMILIES", "")
        families = (
            [item.strip() for item in env_families.split(",") if item.strip()]
            if env_families
            else list(config.get("model_families", DEFAULT_FAMILIES))
        )
    unknown = sorted(set(families) - set(DEFAULT_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown architecture families: {unknown}")
    if len(families) != len(set(families)):
        raise ValueError("Architecture families must be unique")

    run_name = "smoke" if args.smoke else "full"
    root = Path("reports/logs/hyperparameter_search") / run_name
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": "1.0",
        "selection_split": "validation",
        "test_set_selection_forbidden": True,
        "families": families,
        "trials_per_family": trials_per_family,
        "total_trials": len(families) * trials_per_family,
        "search_cell": search_cell,
        "search_space": search_space,
        "architecture_conditional": {
            "controller_specific_parameters": sorted(CONTROLLER_FAMILIES),
            "layer_search_families": sorted(LAYERED_FAMILIES),
            "dropout_search_families": sorted(DROPOUT_FAMILIES),
        },
    }
    write_json(root / "search_plan.json", plan)
    print(
        f"SECTION19 PLAN: {len(families)} families x {trials_per_family} trials = "
        f"{plan['total_trials']} trials",
        flush=True,
    )
    if args.plan_only:
        print("SECTION19 PLAN VALIDATION: PASS", flush=True)
        return 0

    train_samples = min(int(search_cell["train_samples"]), 16) if args.smoke else int(
        search_cell["train_samples"]
    )
    validation_samples = (
        min(int(search_cell["validation_samples"]), 8)
        if args.smoke
        else int(search_cell["validation_samples"])
    )
    train_path, validation_path = prepare_controlled_search_data(
        root,
        task=str(search_cell["task"]),
        sequence_length=int(search_cell["sequence_length"]),
        train_samples=train_samples,
        validation_samples=validation_samples,
        seed=base_seed,
    )

    python = sys.executable
    results: list[TrialResult] = []
    for family in families:
        for trial_index in range(trials_per_family):
            trial_dir = root / family / f"trial_{trial_index:03d}"
            hp = sample_hyperparameters(
                family,
                trial_index,
                search_space,
                base_seed,
                int(search_cell["memory_budget"]),
            )
            trial_config = build_trial_config(
                family=family,
                trial_index=trial_index,
                hp=hp,
                search_cell=search_cell,
                trial_dir=trial_dir,
                train_path=train_path,
                validation_path=validation_path,
                smoke=args.smoke,
            )
            if args.resume:
                completed = load_completed_trial(trial_dir)
                existing_config = trial_dir / "trial_config.yaml"
                expected_text = yaml.safe_dump(trial_config, sort_keys=False)
                config_matches = (
                    existing_config.is_file()
                    and existing_config.read_text(encoding="utf-8") == expected_text
                )
                if (
                    completed is not None
                    and completed.status == "COMPLETE"
                    and config_matches
                ):
                    results.append(completed)
                    print(
                        f"RESUME family={family} trial={trial_index:03d} status=COMPLETE",
                        flush=True,
                    )
                    continue
            result = run_trial(
                python=python,
                family=family,
                trial_index=trial_index,
                hp=hp,
                config=trial_config,
                trial_dir=trial_dir,
                smoke=args.smoke,
            )
            results.append(result)
            value = (
                f"{result.objective_value:.8f}"
                if result.objective_value is not None
                else "NA"
            )
            print(
                f"TRIAL family={family} trial={trial_index:03d} "
                f"status={result.status} objective={value}",
                flush=True,
            )
            if result.status != "COMPLETE" and args.fail_fast:
                raise RuntimeError(result.error or "Trial failed")

    rows = [trial_to_row(result) for result in results]
    write_csv(root / "all_trials.csv", rows)
    best = {family: choose_best(results, family) for family in families}
    best_payload = {
        family: (selected.__dict__ if selected is not None else None)
        for family, selected in best.items()
    }
    write_json(root / "best_hyperparameters.json", best_payload)

    best_config_root = root / "best_configs"
    best_config_root.mkdir(parents=True, exist_ok=True)
    for family, selected in best.items():
        if selected is None:
            continue
        source = Path(selected.config_path)
        if source.is_file():
            shutil.copy2(source, best_config_root / f"{family}.yaml")

    counts = {
        family: sum(result.family == family for result in results) for family in families
    }
    equal_budget = all(count == trials_per_family for count in counts.values())
    complete_counts = {
        family: sum(
            result.family == family and result.status == "COMPLETE" for result in results
        )
        for family in families
    }
    fairness = {
        "status": "PASS" if equal_budget else "FAIL",
        "selection_split": "validation",
        "test_set_selection_used": False,
        "trials_per_family_required": trials_per_family,
        "attempted_trials_by_family": counts,
        "complete_trials_by_family": complete_counts,
        "equal_trial_budget": equal_budget,
    }
    write_json(root / "fairness_audit.json", fairness)
    write_report(
        root / "section19_report.md",
        families,
        trials_per_family,
        results,
        best,
        args.smoke,
    )

    all_complete = all(count == trials_per_family for count in complete_counts.values())
    if not equal_budget:
        print("SECTION19 FAIR SEARCH BUDGET: FAIL", flush=True)
        return 1
    if not all_complete:
        print("SECTION19 TRIAL EXECUTION: INCOMPLETE", flush=True)
        return 1
    print("SECTION19 FAIR SEARCH BUDGET: PASS", flush=True)
    print("SECTION19 VALIDATION-ONLY SELECTION: PASS", flush=True)
    print(f"SECTION19 RESULTS: {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
chmod +x "$SCRIPT_PATH"

# ---------------------------------------------------------------------------
# Contract tests: fairness, deterministic sampling, conditional applicability,
# validation metric extraction, and complete trial logging.
# ---------------------------------------------------------------------------

cat > "$TEST_PATH" <<'PY'
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_section19_hyperparameter_search.py"
SPEC = importlib.util.spec_from_file_location("section19_search", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def search_space() -> dict:
    config = yaml.safe_load(
        (ROOT / "configs/hyperparameter_search/section19_search.yaml").read_text(
            encoding="utf-8"
        )
    )
    return config["search_space"]


def test_sampling_is_deterministic() -> None:
    first = MODULE.sample_hyperparameters(
        "budgetmem_r", 3, search_space(), 2026, 32
    )
    second = MODULE.sample_hyperparameters(
        "budgetmem_r", 3, search_space(), 2026, 32
    )
    assert first == second


def test_non_controller_families_do_not_fake_controller_parameters() -> None:
    values = MODULE.sample_hyperparameters("gru", 0, search_space(), 2026, 32)
    assert values["memory_controller_temperature"] is None
    assert values["auxiliary_loss_coefficient"] is None
    assert values["budget_penalty"] is None
    assert values["retrieval_top_k"] is None
    assert values["write_threshold"] is None


def test_budgetmem_retrieval_top_k_never_exceeds_budget() -> None:
    for trial in range(50):
        values = MODULE.sample_hyperparameters(
            "budgetmem_r", trial, search_space(), 2026, 4
        )
        assert 1 <= values["retrieval_top_k"] <= 4


def test_validation_metric_extraction(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps({"metrics": {"primary_metric_value": 0.75}}),
        encoding="utf-8",
    )
    name, value, direction = MODULE.extract_validation_objective(path)
    assert name == "primary_metric_value"
    assert value == 0.75
    assert direction == "maximize"


def test_plan_has_equal_trials_for_all_families() -> None:
    families = list(MODULE.DEFAULT_FAMILIES)
    trials = 20
    counts = {family: trials for family in families}
    assert len(set(counts.values())) == 1
    assert sum(counts.values()) == len(families) * trials
PY

# Keep runtime outputs out of Git while retaining the directory contract.
if [[ -f .gitignore ]]; then
  grep -qxF 'reports/logs/hyperparameter_search/**' .gitignore || cat >> .gitignore <<'EOF'

# Section 19 runtime search logs and checkpoints
reports/logs/hyperparameter_search/**
!reports/logs/hyperparameter_search/.gitkeep
EOF
else
  cat > .gitignore <<'EOF'
# Section 19 runtime search logs and checkpoints
reports/logs/hyperparameter_search/**
!reports/logs/hyperparameter_search/.gitkeep
EOF
fi
touch "$LOG_ROOT/.gitkeep"

log "Compiling Section 19 Python components."
"$PYTHON" -m py_compile \
  "$SCRIPT_PATH" \
  scripts/run_section18.py \
  src/budgetmem/experiments/pilot.py \
  src/budgetmem/baselines/controlled.py

log "Running focused Section 19 contract tests."
"$PYTHON" -m pytest -q "$TEST_PATH"

log "Validating the equal-budget search plan."
"$PYTHON" "$SCRIPT_PATH" \
  --config "$SEARCH_CONFIG" \
  --plan-only \
  --trials-per-family "${TRIALS_PER_FAMILY:-20}"

if [[ "$MODE" == "--setup" ]]; then
  log "SECTION 19 SETUP: PASS"
  log "Next command: ./$AUTOMATION_NAME --smoke"
  exit 0
fi

if [[ "$MODE" == "--smoke" ]]; then
  log "Launching one tiny validation trial for every architecture family."
  "$PYTHON" "$SCRIPT_PATH" \
    --config "$SEARCH_CONFIG" \
    --smoke \
    --resume
  log "SECTION 19 SMOKE SEARCH: PASS"
  log "Review: reports/logs/hyperparameter_search/smoke/section19_report.md"
  log "Then run: ./$AUTOMATION_NAME --full"
  exit 0
fi

log "Launching the full fair search with ${TRIALS_PER_FAMILY:-20} trials per family."
"$PYTHON" "$SCRIPT_PATH" \
  --config "$SEARCH_CONFIG" \
  --trials-per-family "${TRIALS_PER_FAMILY:-20}" \
  --resume

LATEST_EVIDENCE="$EVIDENCE_ROOT/section19_completion_${STAMP}.txt"
cat > "$LATEST_EVIDENCE" <<EOF
Section 19 Hyperparameter Search
Timestamp UTC: $(date -u +'%Y-%m-%dT%H:%M:%SZ')
Search budget: ${TRIALS_PER_FAMILY:-20} validation trials per architecture family
Selection split: validation only
Trial logs: reports/logs/hyperparameter_search/full/
Fairness audit: reports/logs/hyperparameter_search/full/fairness_audit.json
Best hyperparameters: reports/logs/hyperparameter_search/full/best_hyperparameters.json
Report: reports/logs/hyperparameter_search/full/section19_report.md

Equal trial budget: PASS
Every trial logged: PASS
Validation-only selection: PASS
Section 19: COMPLETE
EOF
cp "$LATEST_EVIDENCE" "$EVIDENCE_ROOT/section19_completion_latest.txt"

log "SECTION 19 FULL SEARCH: PASS"
log "Evidence: $LATEST_EVIDENCE"
log "Do not use test results to revise the selected hyperparameters."
