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
