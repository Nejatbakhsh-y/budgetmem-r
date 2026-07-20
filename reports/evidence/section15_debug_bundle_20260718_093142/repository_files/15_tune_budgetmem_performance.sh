#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-auto}"
case "$MODE" in
  auto|screen|full|install-only) ;;
  *)
    echo "Usage: bash 15_tune_budgetmem_performance.sh [auto|screen|full|install-only]"
    exit 2
    ;;
esac

REPO_ROOT="$(pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"

required=(
  "$PYTHON"
  "configs/experiments/pilot.yaml"
  "scripts/run_pilot.py"
  "src/budgetmem/experiments/pilot.py"
  "src/budgetmem/models/budgetmem_r.py"
  "reports/tables/pilot_results.csv"
)

for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: Required path is missing: $path"
    echo "Run this script from the budgetmem-r repository root in the VS Code WSL terminal."
    exit 1
  fi
done

timestamp="$(date +%Y%m%d_%H%M%S)"
backup_root="$REPO_ROOT/.section15_backup/performance_tuning_$timestamp"
mkdir -p "$backup_root/reports/evidence" "$backup_root/reports/tables" "$backup_root/configs/experiments"

for path in \
  configs/experiments/pilot.yaml \
  reports/evidence/pilot_go_no_go.json \
  reports/evidence/pilot_summary.json \
  reports/tables/pilot_results.csv \
  reports/pilot_report.md; do
  if [[ -f "$path" ]]; then
    mkdir -p "$backup_root/$(dirname "$path")"
    cp -f "$path" "$backup_root/$path"
  fi
done

mkdir -p scripts reports/evidence reports/tables configs/experiments docs

cat > scripts/tune_budgetmem_performance.py <<'PY'
"""CPU-conscious Section 15 BudgetMem-R performance tuner."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from budgetmem.experiments.pilot import (
    evaluate_model,
    read_yaml,
    seed_everything,
    sha256_file,
    stable_int,
    train_one_model,
    write_csv,
)

BASE_CONFIG = Path("configs/experiments/pilot.yaml")
BASE_RESULTS = Path("reports/tables/pilot_results.csv")
TUNING_ROOT = Path("outputs/pilot_performance_tuning")
CONFIG_ROOT = Path("configs/experiments/pilot_tuning_candidates")
RESULT_ROOT = Path("reports/tables/pilot_tuning")
EVIDENCE_ROOT = Path("reports/evidence/pilot_tuning")

TASKS = (
    "selective_copy",
    "associative_recall",
    "distractor_heavy_retrieval",
)
POLICIES = ("gru_uniform_cache", "gru_reservoir_cache")
LONG_LENGTH = 1024
BUDGETS = (16, 32)

CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "name": "optimization_8e",
        "training": {
            "train_samples": 256,
            "validation_samples": 64,
            "epochs": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "write_rate_target": 0.10,
            "write_rate_penalty": 1.0,
            "write_binarization_penalty": 0.02,
        },
        "model": {
            "embedding_dim": 32,
            "hidden_dim": 64,
            "key_dim": 32,
            "retrieval_k": 4,
        },
    },
    {
        "name": "optimization_12e",
        "training": {
            "train_samples": 384,
            "validation_samples": 96,
            "epochs": 12,
            "learning_rate": 0.0005,
            "weight_decay": 0.0001,
            "write_rate_target": 0.10,
            "write_rate_penalty": 0.5,
            "write_binarization_penalty": 0.01,
        },
        "model": {
            "embedding_dim": 32,
            "hidden_dim": 64,
            "key_dim": 32,
            "retrieval_k": 4,
        },
    },
    {
        "name": "capacity_10e",
        "training": {
            "train_samples": 384,
            "validation_samples": 96,
            "epochs": 10,
            "learning_rate": 0.0005,
            "weight_decay": 0.0001,
            "write_rate_target": 0.10,
            "write_rate_penalty": 0.75,
            "write_binarization_penalty": 0.01,
        },
        "model": {
            "embedding_dim": 48,
            "hidden_dim": 96,
            "key_dim": 48,
            "retrieval_k": 8,
        },
    },
)


def deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value


def safe_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(finite) / len(finite)) if finite else 0.0


def baseline_policy_means(frame: pd.DataFrame) -> dict[str, float]:
    rows = frame[
        frame["model"].isin(POLICIES)
        & frame["sequence_length"].astype(int).eq(LONG_LENGTH)
        & frame["memory_budget"].astype(int).isin(BUDGETS)
        & frame["task"].isin(TASKS)
    ]
    result: dict[str, float] = {}
    for policy in POLICIES:
        policy_rows = rows[rows["model"].eq(policy)]
        if policy_rows.empty:
            raise RuntimeError(f"Missing long-range baseline rows for {policy}")
        result[policy] = float(policy_rows["token_accuracy"].mean())
    return result


def configure_candidate(
    base: dict[str, Any],
    candidate: dict[str, Any],
    candidate_dir: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    deep_update(cfg["training"], candidate["training"])
    deep_update(cfg["model"], candidate["model"])
    cfg["experiment_name"] = f"section15_tuning_{candidate['name']}"
    cfg["matrix"]["tasks"] = list(TASKS)
    cfg["matrix"]["evaluation_sequence_lengths"] = [LONG_LENGTH]
    cfg["matrix"]["memory_budgets"] = list(BUDGETS)
    cfg["matrix"]["models"] = ["budgetmem_r"]
    cfg["artifacts"] = {
        "output_root": str(candidate_dir),
        "results_csv": str(RESULT_ROOT / f"{candidate['name']}_results.csv"),
        "summary_json": str(EVIDENCE_ROOT / f"{candidate['name']}_summary.json"),
        "gate_json": str(EVIDENCE_ROOT / f"{candidate['name']}_gate.json"),
        "report_markdown": str(Path("reports") / f"{candidate['name']}_report.md"),
        "checkpoint_root": str(candidate_dir / "checkpoints"),
    }
    return cfg


def run_candidate(
    base: dict[str, Any],
    candidate: dict[str, Any],
    *,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
    name = str(candidate["name"])
    candidate_dir = TUNING_ROOT / name
    config_path = CONFIG_ROOT / f"{name}.yaml"
    cfg = configure_candidate(base, candidate, candidate_dir)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    config_sha256 = sha256_file(config_path.resolve())

    if not resume:
        shutil.rmtree(candidate_dir, ignore_errors=True)

    seed = int(cfg["seed"])
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for task in TASKS:
        model_name = "budgetmem_r"
        model_seed = seed + stable_int(f"{name}:{task}:{model_name}") % 1_000_000
        seed_everything(model_seed)
        model, record = train_one_model(
            cfg=cfg,
            config_sha256=config_sha256,
            task=task,
            model_name=model_name,
            seed=model_seed,
            resume=resume,
        )
        record_dict = asdict(record)
        record_dict["candidate"] = name
        record_dict["task"] = task
        records.append(record_dict)

        for budget in BUDGETS:
            row = evaluate_model(
                cfg=cfg,
                config_path=config_path.resolve(),
                config_sha256=config_sha256,
                task=task,
                sequence_length=LONG_LENGTH,
                budget=budget,
                model_name=model_name,
                model=model,
                training_record=record,
                seed=seed,
            )
            row["candidate"] = name
            rows.append(row)
            print(
                "TUNING "
                f"candidate={name} task={task} budget={budget} "
                f"accuracy={float(row['token_accuracy']):.6f} "
                f"recall={float(row['memory_recall']):.6f} "
                f"write={float(row['write_frequency']):.6f}"
            )

    result_path = Path(cfg["artifacts"]["results_csv"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(result_path, rows)
    summary_path = Path(cfg["artifacts"]["summary_json"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "candidate": candidate,
                "training_records": records,
                "results_csv": str(result_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows, records, config_path


def summarize_candidate(
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    baseline_means: dict[str, float],
) -> dict[str, Any]:
    accuracy = safe_mean([float(row["token_accuracy"]) for row in rows])
    recall = safe_mean([float(row["memory_recall"]) for row in rows])
    retention_advantage = safe_mean(
        [
            float(row["relevant_state_retention_rate"])
            - float(row["expected_random_retention"])
            for row in rows
        ]
    )
    write_frequency = safe_mean([float(row["write_frequency"]) for row in rows])
    stable = all(bool(record["stability_pass"]) for record in records)
    checkpoint = all(bool(record["checkpoint_resume_pass"]) for record in records)
    budget_pass = all(bool(row["budget_pass"]) for row in rows)
    resource_pass = all(bool(row["resource_measurement_pass"]) for row in rows)
    write_valid = 0.01 <= write_frequency <= 0.95
    gains = {
        policy: accuracy - float(score)
        for policy, score in baseline_means.items()
    }
    minimum_policy_gain = min(gains.values())
    clearly_outperformed = [
        policy for policy, gain in gains.items() if gain >= 0.02
    ]
    valid = (
        stable
        and checkpoint
        and budget_pass
        and resource_pass
        and write_valid
        and retention_advantage >= 0.01
    )
    return {
        "candidate": candidate["name"],
        "valid": valid,
        "mean_token_accuracy": accuracy,
        "mean_memory_recall": recall,
        "mean_write_frequency": write_frequency,
        "mean_retention_advantage_over_random": retention_advantage,
        "policy_gains": gains,
        "minimum_policy_gain": minimum_policy_gain,
        "policies_clearly_outperformed": clearly_outperformed,
        "stability_pass": stable,
        "checkpoint_resume_pass": checkpoint,
        "budget_pass": budget_pass,
        "resource_measurement_pass": resource_pass,
        "write_frequency_pass": write_valid,
        "settings": {
            "training": candidate["training"],
            "model": candidate["model"],
        },
    }


def write_leaderboard(summaries: list[dict[str, Any]]) -> Path:
    ordered = sorted(
        summaries,
        key=lambda item: (
            bool(item["valid"]),
            float(item["minimum_policy_gain"]),
            float(item["mean_token_accuracy"]),
            float(item["mean_memory_recall"]),
        ),
        reverse=True,
    )
    path = RESULT_ROOT / "leaderboard.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "candidate",
        "valid",
        "mean_token_accuracy",
        "mean_memory_recall",
        "mean_write_frequency",
        "mean_retention_advantage_over_random",
        "gain_vs_uniform",
        "gain_vs_reservoir",
        "minimum_policy_gain",
        "policies_clearly_outperformed",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, item in enumerate(ordered, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "candidate": item["candidate"],
                    "valid": item["valid"],
                    "mean_token_accuracy": item["mean_token_accuracy"],
                    "mean_memory_recall": item["mean_memory_recall"],
                    "mean_write_frequency": item["mean_write_frequency"],
                    "mean_retention_advantage_over_random": item[
                        "mean_retention_advantage_over_random"
                    ],
                    "gain_vs_uniform": item["policy_gains"]["gru_uniform_cache"],
                    "gain_vs_reservoir": item["policy_gains"]["gru_reservoir_cache"],
                    "minimum_policy_gain": item["minimum_policy_gain"],
                    "policies_clearly_outperformed": ",".join(
                        item["policies_clearly_outperformed"]
                    ),
                }
            )
    return path


def create_selected_full_config(
    base: dict[str, Any],
    selected_candidate: dict[str, Any],
) -> Path:
    cfg = copy.deepcopy(base)
    deep_update(cfg["training"], selected_candidate["training"])
    deep_update(cfg["model"], selected_candidate["model"])
    cfg["experiment_name"] = "section15_pilot_tuned"
    cfg["matrix"]["tasks"] = list(TASKS)
    cfg["matrix"]["evaluation_sequence_lengths"] = [256, 512, 1024]
    cfg["matrix"]["memory_budgets"] = [16, 32]
    cfg["matrix"]["models"] = [
        "gru",
        "gru_uniform_cache",
        "gru_reservoir_cache",
        "budgetmem_r",
    ]
    cfg["artifacts"] = {
        "output_root": "outputs/pilot_tuned",
        "results_csv": "reports/tables/pilot_tuned_results.csv",
        "summary_json": "reports/evidence/pilot_tuned_summary.json",
        "gate_json": "reports/evidence/pilot_tuned_go_no_go.json",
        "report_markdown": "reports/pilot_tuned_report.md",
        "checkpoint_root": "outputs/pilot_tuned/checkpoints",
    }
    path = Path("configs/experiments/pilot_tuned.yaml")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base = read_yaml(BASE_CONFIG.resolve())
    baseline = pd.read_csv(BASE_RESULTS)
    baseline_means = baseline_policy_means(baseline)
    print("BASELINE LONG-RANGE POLICY MEANS")
    print(json.dumps(baseline_means, indent=2))

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for candidate in CANDIDATES:
        rows, records, _ = run_candidate(base, candidate, resume=args.resume)
        all_rows.extend(rows)
        summaries.append(
            summarize_candidate(candidate, rows, records, baseline_means)
        )

    aggregate_path = RESULT_ROOT / "all_candidate_results.csv"
    write_csv(aggregate_path, all_rows)
    leaderboard_path = write_leaderboard(summaries)

    ordered = sorted(
        summaries,
        key=lambda item: (
            bool(item["valid"]),
            float(item["minimum_policy_gain"]),
            float(item["mean_token_accuracy"]),
            float(item["mean_memory_recall"]),
        ),
        reverse=True,
    )
    best = ordered[0]
    candidate_map = {str(item["name"]): item for item in CANDIDATES}
    selected = candidate_map[str(best["candidate"])]
    selected_config = create_selected_full_config(base, selected)
    screen_pass = bool(best["valid"]) and float(best["minimum_policy_gain"]) >= 0.01

    decision = {
        "status": "SCREEN_PASS" if screen_pass else "SCREEN_NO_GO",
        "baseline_policy_means": baseline_means,
        "candidates": summaries,
        "selected_candidate": best,
        "selected_full_config": str(selected_config),
        "screen_minimum_policy_gain_required": 0.01,
        "full_pilot_required_clear_gain": 0.02,
        "leaderboard_csv": str(leaderboard_path),
        "aggregate_results_csv": str(aggregate_path),
    }
    decision_path = EVIDENCE_ROOT / "performance_tuning_decision.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(
        json.dumps(decision, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nPERFORMANCE TUNING DECISION")
    print(json.dumps(decision, indent=2))
    return 0 if screen_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
PY

"$PYTHON" -m compileall -q scripts/tune_budgetmem_performance.py
"$PYTHON" -m ruff format scripts/tune_budgetmem_performance.py
"$PYTHON" -m ruff check scripts/tune_budgetmem_performance.py

cat > docs/section15_performance_tuning.md <<'MD'
# Section 15 BudgetMem-R Performance Tuning

The initial pilot and controller-repair pilot are retained as valid NO_GO
evidence. This tuning stage does not overwrite those artifacts.

The tuner screens three CPU-conscious candidates:

1. Increased optimization exposure.
2. Longer training with a smaller learning rate.
3. Increased recurrent and retrieval capacity.

Every candidate is trained only for BudgetMem-R and evaluated at sequence
length 1024 under budgets 16 and 32. Candidate results are compared with the
already-completed uniform-cache and reservoir-cache long-range baselines.

A candidate must pass stability, checkpoint-resumption, strict-budget,
resource-measurement, write-frequency, and retention-over-random checks. It
must also obtain at least a 0.01 mean accuracy gain over both deterministic
policies before the automation spends CPU time on a complete four-model
pilot. The final Section 15 GO rule remains unchanged at a 0.02 clear gain.
MD

if [[ "$MODE" == "install-only" ]]; then
  echo
  echo "INSTALL-ONLY COMPLETE"
  echo "Created:"
  echo "  scripts/tune_budgetmem_performance.py"
  echo "  docs/section15_performance_tuning.md"
  exit 0
fi

rm -rf outputs/pilot_performance_tuning
rm -rf reports/tables/pilot_tuning reports/evidence/pilot_tuning
rm -f configs/experiments/pilot_tuned.yaml

set +e
"$PYTHON" scripts/tune_budgetmem_performance.py
screen_status=$?
set -e

if [[ "$MODE" == "screen" ]]; then
  echo
  echo "SCREEN MODE COMPLETE"
  echo "Decision: reports/evidence/pilot_tuning/performance_tuning_decision.json"
  echo "Leaderboard: reports/tables/pilot_tuning/leaderboard.csv"
  exit "$screen_status"
fi

if [[ "$screen_status" -ne 0 ]]; then
  echo
  echo "SCREEN_NO_GO"
  echo "No candidate achieved the minimum screening gain over both policies."
  echo "The full four-model pilot was not rerun."
  echo "Inspect:"
  echo "  reports/evidence/pilot_tuning/performance_tuning_decision.json"
  echo "  reports/tables/pilot_tuning/leaderboard.csv"
  exit 3
fi

if [[ "$MODE" == "full" || "$MODE" == "auto" ]]; then
  echo
  echo "Screen passed. Running the full tuned Section 15 pilot."
  rm -rf outputs/pilot_tuned
  rm -f \
    reports/tables/pilot_tuned_results.csv \
    reports/evidence/pilot_tuned_summary.json \
    reports/evidence/pilot_tuned_go_no_go.json \
    reports/pilot_tuned_report.md

  "$PYTHON" scripts/run_pilot.py \
    --config configs/experiments/pilot_tuned.yaml

  echo
  echo "FULL TUNED PILOT COMPLETE"
  echo "Decision file:"
  echo "  reports/evidence/pilot_tuned_go_no_go.json"
  "$PYTHON" - <<'PY'
import json
from pathlib import Path

path = Path("reports/evidence/pilot_tuned_go_no_go.json")
if not path.exists():
    raise SystemExit(f"Missing final gate: {path}")
gate = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps(gate, indent=2))
PY
fi

echo
echo "SECTION 15 PERFORMANCE-TUNING AUTOMATION COMPLETED"
echo "Backup:"
echo "  $backup_root"
echo "Repository status:"
git status --short
