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
