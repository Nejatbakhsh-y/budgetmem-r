#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"

CONFIG="configs/experiments/pilot_assoc_detached_memory.yaml"
SUMMARY="reports/evidence/assoc_detached_memory_summary.json"

OUTPUT_TXT="reports/evidence/assoc_gradient_source_profile.txt"
OUTPUT_CSV="reports/tables/assoc_gradient_source_profile.csv"
OUTPUT_JSON="reports/evidence/assoc_gradient_source_profile.json"

echo "============================================================"
echo " Associative-Recall Gradient-Source Profiler"
echo "============================================================"
echo "Repository: $ROOT"
echo

for required in \
    "$PYTHON" \
    "$CONFIG" \
    "$SUMMARY" \
    "src/budgetmem/experiments/pilot.py"; do

    if [[ ! -e "$required" ]]; then
        echo "ERROR: Missing required path:"
        echo "  $required"
        exit 1
    fi
done

export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p reports/evidence reports/tables

CONFIG_PATH="$CONFIG" \
SUMMARY_PATH="$SUMMARY" \
OUTPUT_CSV="$OUTPUT_CSV" \
OUTPUT_JSON="$OUTPUT_JSON" \
"$PYTHON" - <<'PY' | tee \
    reports/evidence/assoc_gradient_source_profile.txt

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import yaml

from budgetmem.experiments.pilot import (
    _move_batch,
    build_model,
    make_loader,
    stable_int,
    task_loss,
    total_training_loss,
)


ROOT = Path.cwd()

CONFIG_PATH = Path(os.environ["CONFIG_PATH"])
SUMMARY_PATH = Path(os.environ["SUMMARY_PATH"])
OUTPUT_CSV = Path(os.environ["OUTPUT_CSV"])
OUTPUT_JSON = Path(os.environ["OUTPUT_JSON"])

TASK = "associative_recall"
MODEL_NAME = "budgetmem_r"


def total_gradient_norm(
    model: torch.nn.Module,
) -> float:
    """Calculate the current global L2 gradient norm."""
    squared_norm = 0.0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        gradient_norm = float(
            parameter.grad.detach().norm(2).cpu().item()
        )

        squared_norm += gradient_norm * gradient_norm

    return math.sqrt(squared_norm)


def module_gradient_norm(
    parameters,
) -> float:
    squared_norm = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient_norm = float(
            parameter.grad.detach().norm(2).cpu().item()
        )

        squared_norm += gradient_norm * gradient_norm

    return math.sqrt(squared_norm)


cfg = yaml.safe_load(
    CONFIG_PATH.read_text(encoding="utf-8")
)

summary = json.loads(
    SUMMARY_PATH.read_text(encoding="utf-8")
)

training_records = summary.get("training_records", [])

budgetmem_records = [
    record
    for record in training_records
    if record.get("model") == MODEL_NAME
    and record.get("task") == TASK
]

if not budgetmem_records:
    raise SystemExit(
        "ERROR: The BudgetMem-R associative-recall "
        "training record was not found."
    )

training_record = budgetmem_records[0]

checkpoint_value = training_record.get(
    "checkpoint_path"
)

if not checkpoint_value:
    raise SystemExit(
        "ERROR: The training record has no checkpoint path."
    )

checkpoint_path = ROOT / checkpoint_value

if not checkpoint_path.exists():
    raise SystemExit(
        f"ERROR: Checkpoint does not exist: {checkpoint_path}"
    )

device = torch.device("cpu")

seed = int(cfg["seed"])
model_seed = (
    seed
    + stable_int(f"{TASK}:{MODEL_NAME}")
    % 1_000_000
)

model, model_source = build_model(
    MODEL_NAME,
    cfg,
    seed=model_seed,
)

model.to(device)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model.train()

training_cfg = cfg["training"]
matrix_cfg = cfg["matrix"]

train_length = int(
    matrix_cfg["train_sequence_length"]
)

batch_size = int(
    training_cfg["batch_size"]
)

loader = make_loader(
    cfg=cfg,
    task=TASK,
    sequence_length=train_length,
    sample_count=max(batch_size, 8),
    seed=model_seed + 29,
    batch_size=batch_size,
    shuffle=False,
)

batch = _move_batch(
    next(iter(loader)),
    device,
)

budgets = [
    int(value)
    for value in matrix_cfg["memory_budgets"]
]

clip_norm = float(
    training_cfg["gradient_clip_norm"]
)

allowed_raw_norm = float(
    training_cfg[
        "maximum_acceptable_gradient_norm"
    ]
)

write_rate_penalty = float(
    training_cfg.get(
        "write_rate_penalty",
        0.0,
    )
)

write_binarization_penalty = float(
    training_cfg.get(
        "write_binarization_penalty",
        0.0,
    )
)

budget_violation_penalty = float(
    training_cfg.get(
        "budget_violation_penalty",
        0.0,
    )
)

write_rate_target = float(
    training_cfg.get(
        "write_rate_target",
        0.0,
    )
)

write_threshold = float(
    training_cfg.get(
        "write_threshold",
        0.5,
    )
)


def calculate_component(
    budget: int,
    component: str,
) -> tuple[float, float, dict[str, float]]:
    model.zero_grad(set_to_none=True)

    output = model(
        batch.input_ids,
        budget=budget,
    )

    component_values: dict[str, torch.Tensor] = {}

    component_values["task_loss"] = task_loss(
        output.logits,
        batch.target_ids,
    )

    zero = component_values[
        "task_loss"
    ].new_zeros(())

    write_rate_loss = zero
    binarization_loss = zero
    budget_loss = zero

    if output.write_probabilities is not None:
        probabilities = (
            output.write_probabilities.float()
        )

        hard = (
            probabilities >= write_threshold
        ).to(probabilities.dtype)

        straight_through = (
            hard.detach()
            - probabilities.detach()
            + probabilities
        )

        per_sample_rate = (
            straight_through.mean(dim=1)
        )

        write_rate_loss = (
            write_rate_penalty
            * (
                per_sample_rate
                - write_rate_target
            )
            .square()
            .mean()
        )

        binarization_loss = (
            write_binarization_penalty
            * (
                probabilities
                * (1.0 - probabilities)
            )
            .mean()
        )

    if output.memory_sizes is not None:
        overflow = torch.relu(
            output.memory_sizes.float()
            - float(budget)
        ).mean()

        budget_loss = (
            budget_violation_penalty
            * overflow
        )

    component_values[
        "write_rate_penalty"
    ] = write_rate_loss

    component_values[
        "write_binarization_penalty"
    ] = binarization_loss

    component_values[
        "budget_violation_penalty"
    ] = budget_loss

    component_values[
        "total_training_loss"
    ] = total_training_loss(
        output,
        batch.target_ids,
        model_name=MODEL_NAME,
        budget=budget,
        cfg=cfg,
    )

    if component not in component_values:
        raise KeyError(component)

    loss = component_values[component]

    if loss.requires_grad:
        loss.backward()

        global_norm = total_gradient_norm(
            model
        )
    else:
        global_norm = 0.0

    group_norms: dict[str, float] = {}

    for group_name in (
        "write_controller",
        "utility_controller",
        "core",
    ):
        module = getattr(
            model,
            group_name,
            None,
        )

        if module is None:
            module = getattr(
                getattr(model, "core", None),
                group_name,
                None,
            )

        if module is not None:
            group_norms[group_name] = (
                module_gradient_norm(
                    module.parameters()
                )
            )

    return (
        float(loss.detach().cpu().item()),
        global_norm,
        group_norms,
    )


components = [
    "task_loss",
    "write_rate_penalty",
    "write_binarization_penalty",
    "budget_violation_penalty",
    "total_training_loss",
]

rows: list[dict[str, object]] = []

for budget in budgets:
    for component in components:
        try:
            (
                loss_value,
                gradient_norm,
                group_norms,
            ) = calculate_component(
                budget,
                component,
            )

            row = {
                "budget": budget,
                "component": component,
                "loss_value": loss_value,
                "raw_gradient_norm": gradient_norm,
                "exceeds_raw_threshold": (
                    gradient_norm
                    > allowed_raw_norm
                ),
                "applied_norm_after_clipping": (
                    min(
                        gradient_norm,
                        clip_norm,
                    )
                ),
                "write_controller_gradient_norm": (
                    group_norms.get(
                        "write_controller",
                        0.0,
                    )
                ),
                "utility_controller_gradient_norm": (
                    group_norms.get(
                        "utility_controller",
                        0.0,
                    )
                ),
                "core_gradient_norm": (
                    group_norms.get(
                        "core",
                        0.0,
                    )
                ),
            }

            rows.append(row)

        except Exception as exc:
            rows.append(
                {
                    "budget": budget,
                    "component": component,
                    "loss_value": float("nan"),
                    "raw_gradient_norm": float("nan"),
                    "exceeds_raw_threshold": False,
                    "applied_norm_after_clipping": (
                        float("nan")
                    ),
                    "write_controller_gradient_norm": (
                        float("nan")
                    ),
                    "utility_controller_gradient_norm": (
                        float("nan")
                    ),
                    "core_gradient_norm": (
                        float("nan")
                    ),
                    "error": str(exc),
                }
            )


frame = pd.DataFrame(rows)

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

frame.to_csv(
    OUTPUT_CSV,
    index=False,
)

valid = frame[
    frame["raw_gradient_norm"].notna()
].copy()

valid = valid.sort_values(
    "raw_gradient_norm",
    ascending=False,
)

dominant = valid.iloc[0]

print("=" * 120)
print("ASSOCIATIVE-RECALL GRADIENT-SOURCE PROFILE")
print("=" * 120)
print()

print(f"Model source:                 {model_source}")
print(f"Checkpoint:                   {checkpoint_value}")
print(f"Training sequence length:     {train_length}")
print(f"Gradient clipping norm:       {clip_norm:.6f}")
print(f"Maximum acceptable raw norm:  {allowed_raw_norm:.6f}")
print(
    "Recorded maximum raw norm:   "
    f"{training_record.get('maximum_gradient_norm')}"
)

print()
print("=" * 120)
print("COMPONENT RESULTS")
print("=" * 120)
print()

columns = [
    "budget",
    "component",
    "loss_value",
    "raw_gradient_norm",
    "exceeds_raw_threshold",
    "applied_norm_after_clipping",
    "write_controller_gradient_norm",
    "utility_controller_gradient_norm",
    "core_gradient_norm",
]

print(
    valid[columns].to_string(
        index=False,
    )
)

print()
print("=" * 120)
print("DOMINANT GRADIENT SOURCE")
print("=" * 120)
print()

print(
    f"Budget:                       "
    f"{int(dominant['budget'])}"
)
print(
    f"Component:                    "
    f"{dominant['component']}"
)
print(
    f"Raw gradient norm:            "
    f"{dominant['raw_gradient_norm']:.6f}"
)
print(
    f"Threshold excess:             "
    f"{dominant['raw_gradient_norm'] - allowed_raw_norm:+.6f}"
)

component = str(
    dominant["component"]
)

print()
print("Required repair category:")

if component == "task_loss":
    recommendation = (
        "TASK_GRAPH_STABILIZATION"
    )

    print(
        "- The supervised task-loss graph is the "
        "dominant source."
    )
    print(
        "- Stabilize recurrent backpropagation, "
        "normalization, or initialization."
    )
    print(
        "- Do not weaken the memory-recall objective."
    )

elif component == "write_rate_penalty":
    recommendation = (
        "WRITE_RATE_PENALTY_NORMALIZATION"
    )

    print(
        "- The write-rate regularizer is the "
        "dominant source."
    )
    print(
        "- Normalize or rescale this regularizer "
        "without changing the recall gate."
    )

elif component == "write_binarization_penalty":
    recommendation = (
        "BINARIZATION_PENALTY_NORMALIZATION"
    )

    print(
        "- The write-binarization regularizer is "
        "the dominant source."
    )
    print(
        "- Normalize or rescale its gradient "
        "contribution."
    )

elif component == "budget_violation_penalty":
    recommendation = (
        "BUDGET_PENALTY_PATH_REPAIR"
    )

    print(
        "- The budget penalty is the dominant source."
    )
    print(
        "- Inspect why a hard-budget model produces "
        "a differentiable overflow penalty."
    )

else:
    recommendation = (
        "COMBINED_LOSS_INTERACTION"
    )

    print(
        "- No individual component fully explains "
        "the total gradient."
    )
    print(
        "- Inspect interactions among loss components."
    )

payload = {
    "recorded_maximum_gradient_norm": (
        training_record.get(
            "maximum_gradient_norm"
        )
    ),
    "gradient_clip_norm": clip_norm,
    "maximum_acceptable_raw_norm": (
        allowed_raw_norm
    ),
    "dominant_budget": int(
        dominant["budget"]
    ),
    "dominant_component": component,
    "dominant_raw_gradient_norm": float(
        dominant["raw_gradient_norm"]
    ),
    "recommended_repair_category": (
        recommendation
    ),
    "rows": frame.to_dict(
        orient="records"
    ),
}

OUTPUT_JSON.write_text(
    json.dumps(
        payload,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print()
print("=" * 120)
print("FILES WRITTEN")
print("=" * 120)
print(OUTPUT_CSV)
print(OUTPUT_JSON)
PY

echo
echo "============================================================"
echo " Gradient profiling complete"
echo "============================================================"
echo
echo "No training was run."
echo "No thresholds were changed."
echo "No source files were modified."
echo "No commit or push was performed."
echo
echo "Saved:"
echo "  $OUTPUT_TXT"
echo "  $OUTPUT_CSV"
echo "  $OUTPUT_JSON"
