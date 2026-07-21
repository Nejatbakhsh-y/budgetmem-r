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
