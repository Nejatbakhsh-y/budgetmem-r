#!/usr/bin/env bash
set -Eeuo pipefail

# Section 17 — Fair Comparison Protocol
#
# Run from the budgetmem-r repository root:
#   chmod +x 20_17_1_build_fair_comparison_protocol.sh
#   ./20_17_1_build_fair_comparison_protocol.sh
#
# Optional overrides:
#   TOKENS_PER_STEP=8192
#   TRAINING_TOKENS=10485760
#   MAX_OPTIMIZATION_STEPS=10000
#   GRADIENT_ACCUMULATION_STEPS=1
#   PRECISION=fp32
#   HYPERPARAMETER_SEARCH_TRIALS=10
#   EVALUATION_FREQUENCY_STEPS=500
#   HARDWARE_LABEL="controlled-hardware-name"
#   TIMEOUT_SECONDS=3600
#   AUTO_GIT=1
#   AUTO_PUSH=1
#   BRANCH_NAME=feature/17-fair-comparison-protocol
#
# The script creates:
#   configs/fair_comparison.yaml
#   docs/fair_comparison_protocol.md
#   src/budgetmem/protocols/fair_comparison.py
#   scripts/audit_fair_comparison.py
#   scripts/count_model_parameters.py
#   tests/test_fair_comparison_protocol.py
#   reports/tables/model_parameter_counts.csv
#   reports/tables/fair_comparison_run_manifest.csv
#   reports/evidence/section17_fair_comparison_report.txt

readonly SCRIPT_NAME="$(basename "$0")"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
BRANCH_NAME="${BRANCH_NAME:-feature/17-fair-comparison-protocol}"
AUTO_GIT="${AUTO_GIT:-0}"
AUTO_PUSH="${AUTO_PUSH:-0}"

TOKENS_PER_STEP="${TOKENS_PER_STEP:-8192}"
TRAINING_TOKENS="${TRAINING_TOKENS:-10485760}"
MAX_OPTIMIZATION_STEPS="${MAX_OPTIMIZATION_STEPS:-10000}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
PRECISION="${PRECISION:-fp32}"
HYPERPARAMETER_SEARCH_TRIALS="${HYPERPARAMETER_SEARCH_TRIALS:-10}"
EVALUATION_FREQUENCY_STEPS="${EVALUATION_FREQUENCY_STEPS:-500}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

on_error() {
  local exit_code=$?
  printf '\nFAILED: %s exited with code %s at line %s.\n' \
    "$SCRIPT_NAME" "$exit_code" "${BASH_LINENO[0]:-unknown}" >&2
  exit "$exit_code"
}
trap on_error ERR

require_integer() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || fail "$name must be a nonnegative integer; received: $value"
}

for pair in \
  "TOKENS_PER_STEP:$TOKENS_PER_STEP" \
  "TRAINING_TOKENS:$TRAINING_TOKENS" \
  "MAX_OPTIMIZATION_STEPS:$MAX_OPTIMIZATION_STEPS" \
  "GRADIENT_ACCUMULATION_STEPS:$GRADIENT_ACCUMULATION_STEPS" \
  "HYPERPARAMETER_SEARCH_TRIALS:$HYPERPARAMETER_SEARCH_TRIALS" \
  "EVALUATION_FREQUENCY_STEPS:$EVALUATION_FREQUENCY_STEPS" \
  "TIMEOUT_SECONDS:$TIMEOUT_SECONDS"; do
  require_integer "${pair%%:*}" "${pair#*:}"
done

(( TOKENS_PER_STEP > 0 )) || fail "TOKENS_PER_STEP must be greater than zero."
(( TRAINING_TOKENS > 0 )) || fail "TRAINING_TOKENS must be greater than zero."
(( MAX_OPTIMIZATION_STEPS > 0 )) || fail "MAX_OPTIMIZATION_STEPS must be greater than zero."
(( GRADIENT_ACCUMULATION_STEPS > 0 )) || fail "GRADIENT_ACCUMULATION_STEPS must be greater than zero."
(( HYPERPARAMETER_SEARCH_TRIALS > 0 )) || fail "HYPERPARAMETER_SEARCH_TRIALS must be greater than zero."
(( EVALUATION_FREQUENCY_STEPS > 0 )) || fail "EVALUATION_FREQUENCY_STEPS must be greater than zero."
(( TIMEOUT_SECONDS > 0 )) || fail "TIMEOUT_SECONDS must be greater than zero."

for sequence_length in 256 512 1024; do
  (( TOKENS_PER_STEP % sequence_length == 0 )) || \
    fail "TOKENS_PER_STEP=$TOKENS_PER_STEP must be divisible by training sequence length $sequence_length."
done

cd "$PROJECT_ROOT"

[[ -f pyproject.toml ]] || fail "Run this script from the budgetmem-r repository root; pyproject.toml was not found."
[[ -d src/budgetmem ]] || fail "Expected package directory src/budgetmem was not found."

if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=".venv/bin/python"
elif [[ -x venv/bin/python ]]; then
  PYTHON_BIN="venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  fail "Python was not found. Activate the project virtual environment and rerun."
fi

log "Using Python: $PYTHON_BIN"

"$PYTHON_BIN" - <<'PY'
import importlib.util
missing = [
    name for name in ("yaml", "pytest")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit(
        "Missing required Python packages: "
        + ", ".join(missing)
        + ". Install the project dependencies before rerunning."
    )
PY

if [[ "$AUTO_GIT" == "1" ]]; then
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "AUTO_GIT=1 but this is not a Git repository."
  current_branch="$(git branch --show-current)"
  if [[ "$current_branch" != "$BRANCH_NAME" ]]; then
    if git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
      git switch "$BRANCH_NAME"
    else
      git switch -c "$BRANCH_NAME"
    fi
  fi
fi

mkdir -p \
  configs \
  docs \
  scripts \
  src/budgetmem/protocols \
  tests \
  reports/tables \
  reports/evidence

touch src/budgetmem/protocols/__init__.py

if [[ -z "${HARDWARE_LABEL:-}" ]]; then
  HARDWARE_LABEL="$("$PYTHON_BIN" - <<'PY'
import platform

parts = [
    platform.system(),
    platform.release(),
    platform.machine(),
    platform.processor() or "unknown-cpu",
]

try:
    import torch
except Exception:
    torch = None

if torch is not None and torch.cuda.is_available():
    parts.append(f"CUDA:{torch.cuda.get_device_name(0)}")
else:
    parts.append("CPU")

print(" | ".join(str(item).replace('"', "'") for item in parts))
PY
)"
else
  HARDWARE_LABEL="${HARDWARE_LABEL//\"/\'}"
fi

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || printf 'uncommitted')"
GENERATED_AT="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

BATCH_256=$(( TOKENS_PER_STEP / 256 ))
BATCH_512=$(( TOKENS_PER_STEP / 512 ))
BATCH_1024=$(( TOKENS_PER_STEP / 1024 ))

log "Creating configs/fair_comparison.yaml"
cat > configs/fair_comparison.yaml <<YAML
schema_version: "1.0"
section: 17
title: "Fair Comparison Protocol"
generated_at_utc: "$GENERATED_AT"
generated_from_git_commit: "$GIT_COMMIT"

parameter_matching:
  tolerance_fraction: 0.05
  regimes:
    small:
      target_trainable_parameters: 1000000
      minimum_trainable_parameters: 950000
      maximum_trainable_parameters: 1050000
    medium:
      target_trainable_parameters: 5000000
      minimum_trainable_parameters: 4750000
      maximum_trainable_parameters: 5250000
  exact_parameter_counts_required: true
  architecture_exception:
    permitted_only_when_architecture_cannot_match: true
    written_reason_required: true

training_budget:
  training_tokens: $TRAINING_TOKENS
  maximum_optimization_steps: $MAX_OPTIMIZATION_STEPS
  gradient_accumulation_steps: $GRADIENT_ACCUMULATION_STEPS
  precision: "$PRECISION"
  hardware: "$HARDWARE_LABEL"
  hyperparameter_search_trials: $HYPERPARAMETER_SEARCH_TRIALS
  evaluation_frequency_steps: $EVALUATION_FREQUENCY_STEPS
  batch_control: "fixed_tokens_per_step"
  tokens_per_step: $TOKENS_PER_STEP
  timeout_seconds: $TIMEOUT_SECONDS

training_batch_schedule:
  "256": $BATCH_256
  "512": $BATCH_512
  "1024": $BATCH_1024

latency_testing:
  fixed_batch_sizes: [1, 8]
  warmup_iterations: 10
  measured_iterations: 100
  report_median: true
  report_p95: true

memory_budget:
  allowed: [8, 16, 32, 64, 128]
  primary: [32, 64]

retrieval_budget:
  allowed: [1, 4, 8]
  primary: 4

sequence_lengths:
  training: [256, 512, 1024]
  testing: [256, 512, 1024, 2048, 4096, 8192]
  length_8192_requires_reliability_qualification: true

failure_reporting:
  allowed_statuses: ["completed", "oom", "timeout", "failed"]
  oom_must_be_reported: true
  timeout_must_be_reported: true
  failure_reason_required: true
  elapsed_time_required: true

primary_comparison:
  memory_budgets: [32, 64]
  retrieval_k: 4
  same_task: true
  same_sequence_length: true
  same_seed: true
  same_parameter_regime: true
  same_training_budget: true
YAML

log "Creating src/budgetmem/protocols/fair_comparison.py"
cat > src/budgetmem/protocols/fair_comparison.py <<'PY'
"""Section 17 fair-comparison protocol utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


@dataclass(frozen=True)
class ParameterRegime:
    """A target trainable-parameter regime with inclusive bounds."""

    name: str
    target: int
    minimum: int
    maximum: int

    def contains(self, count: int) -> bool:
        return self.minimum <= count <= self.maximum

    @property
    def deviation_fraction_at_minimum(self) -> float:
        return abs(self.minimum - self.target) / self.target

    @property
    def deviation_fraction_at_maximum(self) -> float:
        return abs(self.maximum - self.target) / self.target


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate the Section 17 YAML protocol."""

    protocol_path = Path(path)
    with protocol_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError("Protocol root must be a mapping.")

    errors = validate_protocol(data)
    if errors:
        raise ValueError("Invalid fair-comparison protocol:\n- " + "\n- ".join(errors))
    return data


def parameter_regimes(protocol: Mapping[str, Any]) -> dict[str, ParameterRegime]:
    raw_regimes = protocol["parameter_matching"]["regimes"]
    return {
        name: ParameterRegime(
            name=name,
            target=int(values["target_trainable_parameters"]),
            minimum=int(values["minimum_trainable_parameters"]),
            maximum=int(values["maximum_trainable_parameters"]),
        )
        for name, values in raw_regimes.items()
    }


def count_trainable_parameters(model: Any) -> int:
    """Return the exact number of trainable scalar parameters."""

    if not hasattr(model, "parameters"):
        raise TypeError("The supplied object does not expose a parameters() method.")
    return sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if bool(getattr(parameter, "requires_grad", False))
    )


def expected_batch_size(protocol: Mapping[str, Any], sequence_length: int) -> int:
    """Return the required batch size for a training sequence length."""

    schedule = protocol["training_batch_schedule"]
    key = str(int(sequence_length))
    if key not in schedule:
        raise KeyError(f"No training batch schedule exists for sequence length {sequence_length}.")
    return int(schedule[key])


def tokens_per_step(batch_size: int, sequence_length: int) -> int:
    """Calculate tokens per step using the Section 17 definition."""

    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must both be positive.")
    return int(batch_size) * int(sequence_length)


def validate_parameter_count(
    protocol: Mapping[str, Any],
    regime_name: str,
    exact_count: int,
    architecture_permits_matching: bool = True,
    exception_reason: str = "",
) -> list[str]:
    """Validate one exact parameter count against a named regime."""

    regimes = parameter_regimes(protocol)
    if regime_name not in regimes:
        return [f"Unknown parameter regime: {regime_name!r}."]

    regime = regimes[regime_name]
    errors: list[str] = []

    if exact_count <= 0:
        errors.append("Exact trainable-parameter count must be positive.")
        return errors

    if not regime.contains(exact_count):
        if architecture_permits_matching:
            errors.append(
                f"{regime_name} count {exact_count} is outside "
                f"[{regime.minimum}, {regime.maximum}]."
            )
        elif not exception_reason.strip():
            errors.append(
                "An architecture outside the parameter bounds requires a written exception reason."
            )
    return errors


def validate_protocol(protocol: Mapping[str, Any]) -> list[str]:
    """Return all protocol-definition errors."""

    errors: list[str] = []

    required_top_level = {
        "parameter_matching",
        "training_budget",
        "training_batch_schedule",
        "latency_testing",
        "memory_budget",
        "retrieval_budget",
        "sequence_lengths",
        "failure_reporting",
        "primary_comparison",
    }
    missing = sorted(required_top_level.difference(protocol))
    if missing:
        errors.append(f"Missing top-level sections: {missing}.")
        return errors

    regimes = parameter_regimes(protocol)
    if set(regimes) != {"small", "medium"}:
        errors.append("Parameter regimes must be exactly: small and medium.")

    tolerance = float(protocol["parameter_matching"]["tolerance_fraction"])
    if tolerance != 0.05:
        errors.append("Parameter tolerance must be 0.05.")

    expected_targets = {"small": 1_000_000, "medium": 5_000_000}
    for name, target in expected_targets.items():
        regime = regimes.get(name)
        if regime is None:
            continue
        if regime.target != target:
            errors.append(f"{name} target must be {target}.")
        if regime.deviation_fraction_at_minimum > tolerance + 1e-12:
            errors.append(f"{name} minimum exceeds the permitted tolerance.")
        if regime.deviation_fraction_at_maximum > tolerance + 1e-12:
            errors.append(f"{name} maximum exceeds the permitted tolerance.")

    training_budget = protocol["training_budget"]
    positive_integer_fields = (
        "training_tokens",
        "maximum_optimization_steps",
        "gradient_accumulation_steps",
        "hyperparameter_search_trials",
        "evaluation_frequency_steps",
        "tokens_per_step",
        "timeout_seconds",
    )
    for field in positive_integer_fields:
        try:
            value = int(training_budget[field])
        except (KeyError, TypeError, ValueError):
            errors.append(f"training_budget.{field} must be an integer.")
            continue
        if value <= 0:
            errors.append(f"training_budget.{field} must be positive.")

    if training_budget.get("batch_control") != "fixed_tokens_per_step":
        errors.append("training_budget.batch_control must be fixed_tokens_per_step.")

    required_training_lengths = [256, 512, 1024]
    required_testing_lengths = [256, 512, 1024, 2048, 4096, 8192]
    if list(protocol["sequence_lengths"]["training"]) != required_training_lengths:
        errors.append(f"Training sequence lengths must be {required_training_lengths}.")
    if list(protocol["sequence_lengths"]["testing"]) != required_testing_lengths:
        errors.append(f"Testing sequence lengths must be {required_testing_lengths}.")

    expected_tokens = int(training_budget["tokens_per_step"])
    for length in required_training_lengths:
        try:
            batch = expected_batch_size(protocol, length)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
            continue
        observed_tokens = tokens_per_step(batch, length)
        if observed_tokens != expected_tokens:
            errors.append(
                f"Sequence length {length} uses {observed_tokens} tokens per step; "
                f"expected {expected_tokens}."
            )

    if list(protocol["latency_testing"]["fixed_batch_sizes"]) != [1, 8]:
        errors.append("Latency batch sizes must be exactly [1, 8].")

    if list(protocol["memory_budget"]["allowed"]) != [8, 16, 32, 64, 128]:
        errors.append("Memory budgets must be [8, 16, 32, 64, 128].")
    if list(protocol["memory_budget"]["primary"]) != [32, 64]:
        errors.append("Primary memory budgets must be [32, 64].")

    if list(protocol["retrieval_budget"]["allowed"]) != [1, 4, 8]:
        errors.append("Retrieval budgets must be [1, 4, 8].")
    if int(protocol["retrieval_budget"]["primary"]) != 4:
        errors.append("Primary retrieval budget must be k=4.")

    if not protocol["sequence_lengths"].get(
        "length_8192_requires_reliability_qualification", False
    ):
        errors.append("Length 8192 must require reliability qualification.")

    allowed_statuses = set(protocol["failure_reporting"]["allowed_statuses"])
    if allowed_statuses != {"completed", "oom", "timeout", "failed"}:
        errors.append("Failure statuses must be completed, oom, timeout, and failed.")

    return errors


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    path: str | Path,
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
PY

log "Creating scripts/audit_fair_comparison.py"
cat > scripts/audit_fair_comparison.py <<'PY'
#!/usr/bin/env python3
"""Audit Section 17 protocol artifacts and experiment manifests."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from budgetmem.protocols.fair_comparison import (  # noqa: E402
    expected_batch_size,
    load_protocol,
    read_csv_rows,
    tokens_per_step,
    validate_parameter_count,
)


PARAMETER_COLUMNS = [
    "model",
    "parameter_regime",
    "exact_trainable_parameters",
    "within_tolerance",
    "architecture_permits_matching",
    "exception_reason",
    "factory",
    "configuration",
    "git_commit",
]

RUN_COLUMNS = [
    "run_id",
    "run_type",
    "task",
    "model",
    "parameter_regime",
    "exact_trainable_parameters",
    "architecture_permits_matching",
    "parameter_exception_reason",
    "training_tokens",
    "maximum_optimization_steps",
    "gradient_accumulation_steps",
    "precision",
    "hardware",
    "hyperparameter_search_trials",
    "evaluation_frequency_steps",
    "batch_control",
    "sequence_length",
    "batch_size",
    "tokens_per_step",
    "memory_budget",
    "retrieval_k",
    "seed",
    "split",
    "status",
    "elapsed_seconds",
    "failure_reason",
    "reliability_qualified_8192",
    "config_path",
    "git_commit",
]


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_int(row: dict[str, str], key: str, errors: list[str], prefix: str) -> int | None:
    value = row.get(key, "")
    try:
        return int(value)
    except (TypeError, ValueError):
        errors.append(f"{prefix}: {key} must be an integer; received {value!r}.")
        return None


def as_float(
    row: dict[str, str], key: str, errors: list[str], prefix: str
) -> float | None:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        errors.append(f"{prefix}: {key} must be numeric; received {value!r}.")
        return None


def ensure_csv_header(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(columns)


def audit_parameter_rows(
    protocol: dict[str, Any],
    rows: list[dict[str, str]],
) -> list[str]:
    errors: list[str] = []
    for index, row in enumerate(rows, start=2):
        prefix = f"parameter row {index}"
        regime = row.get("parameter_regime", "")
        count = as_int(row, "exact_trainable_parameters", errors, prefix)
        if count is None:
            continue
        architecture_permits = parse_bool(row.get("architecture_permits_matching", "true"))
        exception_reason = row.get("exception_reason", "")
        row_errors = validate_parameter_count(
            protocol,
            regime,
            count,
            architecture_permits_matching=architecture_permits,
            exception_reason=exception_reason,
        )
        errors.extend(f"{prefix}: {message}" for message in row_errors)

        declared_within = parse_bool(row.get("within_tolerance", "false"))
        actual_within = not row_errors and architecture_permits
        if architecture_permits and declared_within != actual_within:
            errors.append(
                f"{prefix}: within_tolerance does not match the validated parameter count."
            )
    return errors


def audit_run_rows(
    protocol: dict[str, Any],
    rows: list[dict[str, str]],
) -> list[str]:
    errors: list[str] = []
    budget = protocol["training_budget"]
    training_lengths = set(protocol["sequence_lengths"]["training"])
    testing_lengths = set(protocol["sequence_lengths"]["testing"])
    memory_budgets = set(protocol["memory_budget"]["allowed"])
    retrieval_budgets = set(protocol["retrieval_budget"]["allowed"])
    latency_batches = set(protocol["latency_testing"]["fixed_batch_sizes"])
    allowed_statuses = set(protocol["failure_reporting"]["allowed_statuses"])

    for index, row in enumerate(rows, start=2):
        prefix = f"run row {index}"
        run_type = row.get("run_type", "").strip().lower()
        if run_type not in {"training", "evaluation", "latency"}:
            errors.append(f"{prefix}: run_type must be training, evaluation, or latency.")
            continue

        exact_count = as_int(row, "exact_trainable_parameters", errors, prefix)
        if exact_count is not None:
            errors.extend(
                f"{prefix}: {message}"
                for message in validate_parameter_count(
                    protocol,
                    row.get("parameter_regime", ""),
                    exact_count,
                    architecture_permits_matching=parse_bool(
                        row.get("architecture_permits_matching", "true")
                    ),
                    exception_reason=row.get("parameter_exception_reason", ""),
                )
            )

        sequence_length = as_int(row, "sequence_length", errors, prefix)
        batch_size = as_int(row, "batch_size", errors, prefix)
        observed_tokens = as_int(row, "tokens_per_step", errors, prefix)

        if sequence_length is not None:
            permitted_lengths = training_lengths if run_type == "training" else testing_lengths
            if sequence_length not in permitted_lengths:
                errors.append(
                    f"{prefix}: sequence_length {sequence_length} is invalid for {run_type}."
                )
            if sequence_length == 8192 and not parse_bool(
                row.get("reliability_qualified_8192", "false")
            ):
                errors.append(
                    f"{prefix}: sequence length 8192 requires reliability qualification."
                )

        if batch_size is not None:
            if run_type == "latency":
                if batch_size not in latency_batches:
                    errors.append(
                        f"{prefix}: latency batch_size must be one of {sorted(latency_batches)}."
                    )
            elif run_type == "training" and sequence_length is not None:
                try:
                    required_batch = expected_batch_size(protocol, sequence_length)
                except KeyError:
                    required_batch = None
                if required_batch is not None and batch_size != required_batch:
                    errors.append(
                        f"{prefix}: batch_size {batch_size} does not match required "
                        f"{required_batch} for sequence length {sequence_length}."
                    )

        if (
            batch_size is not None
            and sequence_length is not None
            and observed_tokens is not None
        ):
            calculated = tokens_per_step(batch_size, sequence_length)
            if observed_tokens != calculated:
                errors.append(
                    f"{prefix}: tokens_per_step {observed_tokens} does not equal "
                    f"batch_size × sequence_length = {calculated}."
                )
            if run_type == "training" and observed_tokens != int(budget["tokens_per_step"]):
                errors.append(
                    f"{prefix}: training tokens_per_step must equal "
                    f"{budget['tokens_per_step']}."
                )

        memory_budget = as_int(row, "memory_budget", errors, prefix)
        if memory_budget is not None and memory_budget not in memory_budgets:
            errors.append(
                f"{prefix}: memory_budget must be one of {sorted(memory_budgets)}."
            )

        retrieval_k = as_int(row, "retrieval_k", errors, prefix)
        if retrieval_k is not None and retrieval_k not in retrieval_budgets:
            errors.append(
                f"{prefix}: retrieval_k must be one of {sorted(retrieval_budgets)}."
            )

        if run_type == "training":
            fixed_integer_fields = {
                "training_tokens": int(budget["training_tokens"]),
                "maximum_optimization_steps": int(budget["maximum_optimization_steps"]),
                "gradient_accumulation_steps": int(
                    budget["gradient_accumulation_steps"]
                ),
                "hyperparameter_search_trials": int(
                    budget["hyperparameter_search_trials"]
                ),
                "evaluation_frequency_steps": int(
                    budget["evaluation_frequency_steps"]
                ),
            }
            for field, expected in fixed_integer_fields.items():
                observed = as_int(row, field, errors, prefix)
                if observed is not None and observed != expected:
                    errors.append(
                        f"{prefix}: {field}={observed} does not match protocol value {expected}."
                    )

            fixed_text_fields = {
                "precision": str(budget["precision"]),
                "hardware": str(budget["hardware"]),
                "batch_control": str(budget["batch_control"]),
            }
            for field, expected in fixed_text_fields.items():
                observed = row.get(field, "")
                if observed != expected:
                    errors.append(
                        f"{prefix}: {field}={observed!r} does not match protocol value "
                        f"{expected!r}."
                    )

        status = row.get("status", "").strip().lower()
        if status not in allowed_statuses:
            errors.append(
                f"{prefix}: status must be one of {sorted(allowed_statuses)}."
            )

        elapsed = as_float(row, "elapsed_seconds", errors, prefix)
        if elapsed is not None and elapsed < 0:
            errors.append(f"{prefix}: elapsed_seconds cannot be negative.")

        if status in {"oom", "timeout", "failed"} and not row.get(
            "failure_reason", ""
        ).strip():
            errors.append(
                f"{prefix}: status={status} requires an explicit failure_reason."
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "fair_comparison.yaml",
    )
    parser.add_argument(
        "--parameter-table",
        type=Path,
        default=PROJECT_ROOT / "reports" / "tables" / "model_parameter_counts.csv",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "tables"
        / "fair_comparison_run_manifest.csv",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "evidence"
        / "section17_fair_comparison_report.txt",
    )
    args = parser.parse_args()

    ensure_csv_header(args.parameter_table, PARAMETER_COLUMNS)
    ensure_csv_header(args.run_manifest, RUN_COLUMNS)

    protocol = load_protocol(args.protocol)
    parameter_rows = read_csv_rows(args.parameter_table)
    run_rows = read_csv_rows(args.run_manifest)

    errors = audit_parameter_rows(protocol, parameter_rows)
    errors.extend(audit_run_rows(protocol, run_rows))

    protocol_status = "PASS"
    parameter_status = "PASS" if parameter_rows and not audit_parameter_rows(
        protocol, parameter_rows
    ) else ("NOT_RUN" if not parameter_rows else "FAIL")
    run_status = "PASS" if run_rows and not audit_run_rows(
        protocol, run_rows
    ) else ("NOT_RUN" if not run_rows else "FAIL")
    overall_status = "PASS" if not errors else "FAIL"

    lines = [
        "Section 17 — Fair Comparison Protocol Audit",
        "=" * 46,
        f"Protocol definition: {protocol_status}",
        f"Exact parameter-count audit: {parameter_status}",
        f"Experiment-run audit: {run_status}",
        f"Overall structural audit: {overall_status}",
        "",
        "Frozen controls",
        "---------------",
        f"Small parameter target: 1,000,000 ±5%",
        f"Medium parameter target: 5,000,000 ±5%",
        f"Training tokens: {protocol['training_budget']['training_tokens']}",
        f"Maximum optimization steps: "
        f"{protocol['training_budget']['maximum_optimization_steps']}",
        f"Gradient accumulation: "
        f"{protocol['training_budget']['gradient_accumulation_steps']}",
        f"Precision: {protocol['training_budget']['precision']}",
        f"Hardware: {protocol['training_budget']['hardware']}",
        f"Hyperparameter-search trials: "
        f"{protocol['training_budget']['hyperparameter_search_trials']}",
        f"Evaluation frequency: "
        f"{protocol['training_budget']['evaluation_frequency_steps']} steps",
        f"Tokens per step: {protocol['training_budget']['tokens_per_step']}",
        f"Training batch schedule: {protocol['training_batch_schedule']}",
        f"Latency batch sizes: {protocol['latency_testing']['fixed_batch_sizes']}",
        f"Memory budgets: {protocol['memory_budget']['allowed']}",
        f"Primary memory budgets: {protocol['memory_budget']['primary']}",
        f"Retrieval budgets: {protocol['retrieval_budget']['allowed']}",
        f"Primary retrieval budget: {protocol['retrieval_budget']['primary']}",
        f"Training sequence lengths: {protocol['sequence_lengths']['training']}",
        f"Testing sequence lengths: {protocol['sequence_lengths']['testing']}",
        "",
        f"Parameter rows audited: {len(parameter_rows)}",
        f"Run rows audited: {len(run_rows)}",
    ]

    if errors:
        lines.extend(["", "Violations", "----------"])
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.extend(
            [
                "",
                "Violations",
                "----------",
                "None.",
            ]
        )

    if not parameter_rows or not run_rows:
        lines.extend(
            [
                "",
                "Completion note",
                "---------------",
                "The protocol infrastructure is valid. Exact model parameter counts and",
                "completed experiment rows remain NOT_RUN until their CSV tables contain",
                "records produced by model construction and experiment execution.",
            ]
        )

    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
chmod +x scripts/audit_fair_comparison.py

log "Creating scripts/count_model_parameters.py"
cat > scripts/count_model_parameters.py <<'PY'
#!/usr/bin/env python3
"""Instantiate a model factory, count trainable parameters, and register the count."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from budgetmem.protocols.fair_comparison import (  # noqa: E402
    count_trainable_parameters,
    load_protocol,
    parameter_regimes,
    read_csv_rows,
    validate_parameter_count,
    write_csv_rows,
)


FIELDNAMES = [
    "model",
    "parameter_regime",
    "exact_trainable_parameters",
    "within_tolerance",
    "architecture_permits_matching",
    "exception_reason",
    "factory",
    "configuration",
    "git_commit",
]


def import_callable(specification: str) -> Callable[..., Any]:
    if ":" not in specification:
        raise ValueError("Factory must use module.path:callable_name syntax.")
    module_name, callable_name = specification.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, callable_name)
    if not callable(factory):
        raise TypeError(f"{specification!r} is not callable.")
    return factory


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "uncommitted"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--regime", choices=["small", "medium"], required=True)
    parser.add_argument(
        "--factory",
        required=True,
        help="Python factory in module.path:callable syntax.",
    )
    parser.add_argument(
        "--kwargs-json",
        default="{}",
        help="JSON object passed to the factory as keyword arguments.",
    )
    parser.add_argument(
        "--architecture-permits-matching",
        choices=["true", "false"],
        default="true",
    )
    parser.add_argument("--exception-reason", default="")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "fair_comparison.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "tables" / "model_parameter_counts.csv",
    )
    args = parser.parse_args()

    kwargs = json.loads(args.kwargs_json)
    if not isinstance(kwargs, dict):
        raise ValueError("--kwargs-json must decode to a JSON object.")

    factory = import_callable(args.factory)
    model = factory(**kwargs)
    exact_count = count_trainable_parameters(model)

    protocol = load_protocol(args.protocol)
    architecture_permits = args.architecture_permits_matching == "true"
    errors = validate_parameter_count(
        protocol,
        args.regime,
        exact_count,
        architecture_permits_matching=architecture_permits,
        exception_reason=args.exception_reason,
    )
    within_tolerance = parameter_regimes(protocol)[args.regime].contains(exact_count)

    rows = read_csv_rows(args.output)
    rows = [
        row
        for row in rows
        if not (
            row.get("model") == args.model
            and row.get("parameter_regime") == args.regime
        )
    ]
    rows.append(
        {
            "model": args.model,
            "parameter_regime": args.regime,
            "exact_trainable_parameters": exact_count,
            "within_tolerance": str(within_tolerance).lower(),
            "architecture_permits_matching": str(architecture_permits).lower(),
            "exception_reason": args.exception_reason,
            "factory": args.factory,
            "configuration": json.dumps(kwargs, sort_keys=True),
            "git_commit": current_git_commit(),
        }
    )
    write_csv_rows(args.output, FIELDNAMES, rows)

    print(f"Model: {args.model}")
    print(f"Regime: {args.regime}")
    print(f"Exact trainable parameters: {exact_count:,}")
    print(f"Within tolerance: {within_tolerance}")
    print(f"Registered in: {args.output}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
chmod +x scripts/count_model_parameters.py

log "Creating tests/test_fair_comparison_protocol.py"
cat > tests/test_fair_comparison_protocol.py <<'PY'
from __future__ import annotations

from pathlib import Path

import pytest

from budgetmem.protocols.fair_comparison import (
    expected_batch_size,
    load_protocol,
    parameter_regimes,
    tokens_per_step,
    validate_parameter_count,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "fair_comparison.yaml"


@pytest.fixture(scope="module")
def protocol():
    return load_protocol(PROTOCOL_PATH)


def test_parameter_regimes_are_exactly_small_and_medium(protocol):
    regimes = parameter_regimes(protocol)
    assert set(regimes) == {"small", "medium"}
    assert regimes["small"].target == 1_000_000
    assert regimes["small"].minimum == 950_000
    assert regimes["small"].maximum == 1_050_000
    assert regimes["medium"].target == 5_000_000
    assert regimes["medium"].minimum == 4_750_000
    assert regimes["medium"].maximum == 5_250_000


@pytest.mark.parametrize(
    ("regime", "count"),
    [
        ("small", 950_000),
        ("small", 1_000_000),
        ("small", 1_050_000),
        ("medium", 4_750_000),
        ("medium", 5_000_000),
        ("medium", 5_250_000),
    ],
)
def test_parameter_counts_inside_tolerance_pass(protocol, regime, count):
    assert validate_parameter_count(protocol, regime, count) == []


def test_out_of_tolerance_requires_architecture_exception(protocol):
    errors = validate_parameter_count(protocol, "small", 1_200_000)
    assert errors

    errors = validate_parameter_count(
        protocol,
        "small",
        1_200_000,
        architecture_permits_matching=False,
        exception_reason="Discrete head width prevents a closer configuration.",
    )
    assert errors == []


@pytest.mark.parametrize("sequence_length", [256, 512, 1024])
def test_training_tokens_per_step_are_constant(protocol, sequence_length):
    batch_size = expected_batch_size(protocol, sequence_length)
    assert (
        tokens_per_step(batch_size, sequence_length)
        == protocol["training_budget"]["tokens_per_step"]
    )


def test_latency_batch_sizes_are_fixed(protocol):
    assert protocol["latency_testing"]["fixed_batch_sizes"] == [1, 8]


def test_memory_budgets_and_primary_budgets(protocol):
    assert protocol["memory_budget"]["allowed"] == [8, 16, 32, 64, 128]
    assert protocol["memory_budget"]["primary"] == [32, 64]


def test_retrieval_budgets_and_primary_k(protocol):
    assert protocol["retrieval_budget"]["allowed"] == [1, 4, 8]
    assert protocol["retrieval_budget"]["primary"] == 4


def test_sequence_lengths(protocol):
    assert protocol["sequence_lengths"]["training"] == [256, 512, 1024]
    assert protocol["sequence_lengths"]["testing"] == [
        256,
        512,
        1024,
        2048,
        4096,
        8192,
    ]
    assert protocol["sequence_lengths"][
        "length_8192_requires_reliability_qualification"
    ]


def test_oom_and_timeout_reporting_is_mandatory(protocol):
    reporting = protocol["failure_reporting"]
    assert reporting["oom_must_be_reported"]
    assert reporting["timeout_must_be_reported"]
    assert reporting["failure_reason_required"]
    assert reporting["elapsed_time_required"]
PY

log "Creating reports/tables CSV templates"
if [[ ! -s reports/tables/model_parameter_counts.csv ]]; then
  cat > reports/tables/model_parameter_counts.csv <<'CSV'
model,parameter_regime,exact_trainable_parameters,within_tolerance,architecture_permits_matching,exception_reason,factory,configuration,git_commit
CSV
fi

if [[ ! -s reports/tables/fair_comparison_run_manifest.csv ]]; then
  cat > reports/tables/fair_comparison_run_manifest.csv <<'CSV'
run_id,run_type,task,model,parameter_regime,exact_trainable_parameters,architecture_permits_matching,parameter_exception_reason,training_tokens,maximum_optimization_steps,gradient_accumulation_steps,precision,hardware,hyperparameter_search_trials,evaluation_frequency_steps,batch_control,sequence_length,batch_size,tokens_per_step,memory_budget,retrieval_k,seed,split,status,elapsed_seconds,failure_reason,reliability_qualified_8192,config_path,git_commit
CSV
fi

log "Creating docs/fair_comparison_protocol.md"
cat > docs/fair_comparison_protocol.md <<MD
# Section 17 — Fair Comparison Protocol

Generated: $GENERATED_AT

This protocol freezes the controls required for fair comparison before the final experiment matrix is executed.

## 1. Parameter matching

Two trainable-parameter regimes are required:

| Regime | Target | Permitted interval |
|---|---:|---:|
| Small | 1,000,000 | 950,000–1,050,000 |
| Medium | 5,000,000 | 4,750,000–5,250,000 |

Every model configuration must report its exact trainable-parameter count. A model outside the permitted interval is acceptable only when its architecture cannot match more closely and a written exception is recorded.

Use the generic counter as follows:

\`\`\`bash
$PYTHON_BIN scripts/count_model_parameters.py \\
  --model MODEL_NAME \\
  --regime small \\
  --factory package.module:factory_function \\
  --kwargs-json '{"argument": "value"}'
\`\`\`

The factory must return the instantiated PyTorch model.

## 2. Frozen training budget

The following values must remain identical across compared models:

| Control | Frozen value |
|---|---|
| Training tokens | $TRAINING_TOKENS |
| Maximum optimization steps | $MAX_OPTIMIZATION_STEPS |
| Gradient accumulation | $GRADIENT_ACCUMULATION_STEPS |
| Precision | $PRECISION |
| Hardware | $HARDWARE_LABEL |
| Hyperparameter-search trials | $HYPERPARAMETER_SEARCH_TRIALS |
| Evaluation frequency | Every $EVALUATION_FREQUENCY_STEPS steps |
| Batch control | Fixed tokens per step |
| Tokens per step | $TOKENS_PER_STEP |
| Timeout threshold | $TIMEOUT_SECONDS seconds |

Training batch sizes are derived from:

\`\`\`text
tokens per step = batch size × sequence length
\`\`\`

| Training sequence length | Required batch size | Tokens per step |
|---:|---:|---:|
| 256 | $BATCH_256 | $TOKENS_PER_STEP |
| 512 | $BATCH_512 | $TOKENS_PER_STEP |
| 1,024 | $BATCH_1024 | $TOKENS_PER_STEP |

Gradient accumulation is frozen independently and must not be varied between compared runs.

## 3. Latency testing

Latency measurements must use fixed batch sizes:

- Batch size 1
- Batch size 8

The protocol records ten warm-up iterations and one hundred measured iterations. Median and p95 latency must be reported.

## 4. Memory and retrieval budgets

Allowed memory budgets:

\`\`\`text
B = 8, 16, 32, 64, 128
\`\`\`

Primary memory-budget comparisons:

\`\`\`text
B = 32 and B = 64
\`\`\`

Allowed retrieval budgets:

\`\`\`text
k = 1, 4, 8
\`\`\`

Primary retrieval configuration:

\`\`\`text
k = 4
\`\`\`

## 5. Sequence lengths

Training:

\`\`\`text
256, 512, 1,024
\`\`\`

Testing:

\`\`\`text
256, 512, 1,024, 2,048, 4,096, 8,192
\`\`\`

Sequence length 8,192 may be run only after the task and architecture are explicitly marked reliable at that length.

## 6. OOM and timeout reporting

Every attempted run must have one of these statuses:

\`\`\`text
completed, oom, timeout, failed
\`\`\`

OOM, timeout, and failed rows must record:

- Explicit failure reason
- Elapsed time
- Configuration path
- Git commit
- Task, model, parameter regime, sequence length, memory budget, retrieval budget, and seed

Do not silently remove failed runs from the manifest.

## 7. Primary comparison matching keys

A primary comparison is valid only when both rows match on:

- Task
- Sequence length
- Random seed
- Parameter regime
- Training-token budget
- Maximum optimization steps
- Gradient accumulation
- Precision
- Hardware
- Hyperparameter-search trials
- Evaluation frequency
- Batch-control rule
- Memory budget, restricted to 32 or 64
- Retrieval budget, fixed at 4

## 8. Required evidence files

The automation creates:

- \`configs/fair_comparison.yaml\`
- \`reports/tables/model_parameter_counts.csv\`
- \`reports/tables/fair_comparison_run_manifest.csv\`
- \`reports/evidence/section17_fair_comparison_report.txt\`

Run the audit after parameter registration and after every experiment batch:

\`\`\`bash
$PYTHON_BIN scripts/audit_fair_comparison.py
\`\`\`

The initial audit may show exact parameter counts and experiment runs as \`NOT_RUN\`. Those statuses become \`PASS\` only after the corresponding CSV tables contain valid records.
MD

log "Running Section 17 unit tests"
PYTHONPATH=src "$PYTHON_BIN" -m pytest -q tests/test_fair_comparison_protocol.py

log "Running Section 17 structural audit"
PYTHONPATH=src "$PYTHON_BIN" scripts/audit_fair_comparison.py

if [[ "$AUTO_GIT" == "1" ]]; then
  git add \
    configs/fair_comparison.yaml \
    docs/fair_comparison_protocol.md \
    src/budgetmem/protocols/__init__.py \
    src/budgetmem/protocols/fair_comparison.py \
    scripts/audit_fair_comparison.py \
    scripts/count_model_parameters.py \
    tests/test_fair_comparison_protocol.py \
    reports/tables/model_parameter_counts.csv \
    reports/tables/fair_comparison_run_manifest.csv \
    reports/evidence/section17_fair_comparison_report.txt

  if git diff --cached --quiet; then
    log "No Section 17 changes required a new commit."
  else
    git commit -m "Implement Section 17 fair comparison protocol"
  fi

  if [[ "$AUTO_PUSH" == "1" ]]; then
    git push -u origin "$BRANCH_NAME"
  fi
fi

cat <<EOF

Section 17 automation completed.

Created or updated:
  configs/fair_comparison.yaml
  docs/fair_comparison_protocol.md
  src/budgetmem/protocols/fair_comparison.py
  scripts/audit_fair_comparison.py
  scripts/count_model_parameters.py
  tests/test_fair_comparison_protocol.py
  reports/tables/model_parameter_counts.csv
  reports/tables/fair_comparison_run_manifest.csv
  reports/evidence/section17_fair_comparison_report.txt

Current status:
  Protocol definition: PASS
  Unit tests: PASS
  Exact parameter-count audit: NOT_RUN until model rows are registered
  Experiment-run audit: NOT_RUN until experiment rows are recorded

Review the evidence:
  cat reports/evidence/section17_fair_comparison_report.txt

To register an exact model count:
  $PYTHON_BIN scripts/count_model_parameters.py \\
    --model MODEL_NAME \\
    --regime small \\
    --factory package.module:factory_function \\
    --kwargs-json '{"argument":"value"}'

To rerun the audit:
  $PYTHON_BIN scripts/audit_fair_comparison.py

Optional Git workflow:
  AUTO_GIT=1 AUTO_PUSH=1 ./$SCRIPT_NAME
EOF
