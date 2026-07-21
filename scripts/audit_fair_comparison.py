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
