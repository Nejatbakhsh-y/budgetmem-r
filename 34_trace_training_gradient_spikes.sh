#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
CONFIG="configs/experiments/pilot_assoc_detached_memory.yaml"

OUTPUT_CSV="reports/tables/assoc_gradient_training_trace.csv"
OUTPUT_JSON="reports/evidence/assoc_gradient_training_trace.json"
OUTPUT_TXT="reports/evidence/assoc_gradient_training_trace.txt"

echo "============================================================"
echo " Section 15 Training-Wide Gradient Trace"
echo "============================================================"
echo "Repository: $ROOT"
echo

for required in \
    "$PYTHON" \
    "$CONFIG" \
    "src/budgetmem/experiments/pilot.py"; do

    if [[ ! -e "$required" ]]; then
        echo "ERROR: Missing required path:"
        echo "  $required"
        exit 1
    fi
done

export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p reports/tables reports/evidence

CONFIG_PATH="$CONFIG" \
OUTPUT_CSV="$OUTPUT_CSV" \
OUTPUT_JSON="$OUTPUT_JSON" \
"$PYTHON" - <<'PY' | tee \
    reports/evidence/assoc_gradient_training_trace.txt

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from budgetmem.experiments.pilot import (
    _move_batch,
    build_model,
    make_loader,
    seed_everything,
    stable_int,
    task_loss,
    total_training_loss,
)


CONFIG_PATH = Path(os.environ["CONFIG_PATH"])
OUTPUT_CSV = Path(os.environ["OUTPUT_CSV"])
OUTPUT_JSON = Path(os.environ["OUTPUT_JSON"])

TASK = "associative_recall"
MODEL_NAME = "budgetmem_r"


def global_gradient_norm(model: torch.nn.Module) -> float:
    squared_total = 0.0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        norm = float(
            parameter.grad.detach().norm(2).cpu().item()
        )

        squared_total += norm * norm

    return math.sqrt(squared_total)


def parameter_gradient_table(
    model: torch.nn.Module,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        norm = float(
            parameter.grad.detach().norm(2).cpu().item()
        )

        rows.append(
            {
                "parameter": name,
                "gradient_norm": norm,
                "parameter_count": parameter.numel(),
            }
        )

    return sorted(
        rows,
        key=lambda row: float(row["gradient_norm"]),
        reverse=True,
    )


def component_losses(
    output,
    target_ids: torch.Tensor,
    *,
    budget: int,
    cfg: dict,
) -> dict[str, torch.Tensor]:
    training_cfg = cfg["training"]

    supervised = task_loss(
        output.logits,
        target_ids,
    )

    zero = supervised.new_zeros(())

    write_rate = zero
    binarization = zero
    budget_penalty = zero

    if output.write_probabilities is not None:
        probabilities = output.write_probabilities.float()

        threshold = float(
            training_cfg.get(
                "write_threshold",
                0.5,
            )
        )

        hard = (
            probabilities >= threshold
        ).to(probabilities.dtype)

        straight_through = (
            hard.detach()
            - probabilities.detach()
            + probabilities
        )

        observed_rate = straight_through.mean(dim=1)

        target_rate = float(
            training_cfg["write_rate_target"]
        )

        write_rate = (
            float(
                training_cfg["write_rate_penalty"]
            )
            * (
                observed_rate - target_rate
            )
            .square()
            .mean()
        )

        binarization = (
            float(
                training_cfg.get(
                    "write_binarization_penalty",
                    0.0,
                )
            )
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

        budget_penalty = (
            float(
                training_cfg[
                    "budget_violation_penalty"
                ]
            )
            * overflow
        )

    return {
        "task_loss": supervised,
        "write_rate_penalty": write_rate,
        "write_binarization_penalty": binarization,
        "budget_violation_penalty": budget_penalty,
        "total_training_loss": (
            supervised
            + write_rate
            + binarization
            + budget_penalty
        ),
    }


def profile_spike_components(
    model: torch.nn.Module,
    batch,
    *,
    budget: int,
    cfg: dict,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for component_name in (
        "task_loss",
        "write_rate_penalty",
        "write_binarization_penalty",
        "budget_violation_penalty",
        "total_training_loss",
    ):
        model.zero_grad(set_to_none=True)

        output = model(
            batch.input_ids,
            budget=budget,
        )

        components = component_losses(
            output,
            batch.target_ids,
            budget=budget,
            cfg=cfg,
        )

        component = components[component_name]

        if component.requires_grad:
            component.backward()
            norm = global_gradient_norm(model)
            top_parameters = parameter_gradient_table(model)[:10]
        else:
            norm = 0.0
            top_parameters = []

        rows.append(
            {
                "component": component_name,
                "loss_value": float(
                    component.detach().cpu().item()
                ),
                "raw_gradient_norm": norm,
                "top_parameters": top_parameters,
            }
        )

    model.zero_grad(set_to_none=True)

    return rows


cfg = yaml.safe_load(
    CONFIG_PATH.read_text(encoding="utf-8")
)

training_cfg = cfg["training"]
matrix_cfg = cfg["matrix"]

seed = int(cfg["seed"])
model_seed = (
    seed
    + stable_int(f"{TASK}:{MODEL_NAME}")
    % 1_000_000
)

seed_everything(seed)

train_length = int(
    matrix_cfg["train_sequence_length"]
)

train_samples = int(
    training_cfg["train_samples"]
)

batch_size = int(
    training_cfg["batch_size"]
)

epochs = int(
    training_cfg["epochs"]
)

learning_rate = float(
    training_cfg["learning_rate"]
)

weight_decay = float(
    training_cfg["weight_decay"]
)

clip_norm = float(
    training_cfg["gradient_clip_norm"]
)

allowed_raw_norm = float(
    training_cfg[
        "maximum_acceptable_gradient_norm"
    ]
)

budgets = [
    int(value)
    for value in matrix_cfg["memory_budgets"]
]

loader = make_loader(
    cfg=cfg,
    task=TASK,
    sequence_length=train_length,
    sample_count=train_samples,
    seed=model_seed + 11,
    batch_size=batch_size,
    shuffle=True,
)

model, model_source = build_model(
    MODEL_NAME,
    cfg,
    seed=model_seed,
)

model.to(torch.device("cpu"))
model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

trace_rows: list[dict[str, object]] = []
spike_payload: dict[str, object] | None = None

start_time = time.perf_counter()
step = 0

print("=" * 100)
print("TRAINING-WIDE GRADIENT TRACE")
print("=" * 100)
print()
print(f"Model source:                 {model_source}")
print(f"Task:                         {TASK}")
print(f"Training sequence length:     {train_length}")
print(f"Epochs:                       {epochs}")
print(f"Training samples:             {train_samples}")
print(f"Batch size:                   {batch_size}")
print(f"Learning rate:                {learning_rate}")
print(f"Gradient clipping norm:       {clip_norm}")
print(f"Maximum acceptable raw norm:  {allowed_raw_norm}")
print(f"detach_memory_writes:         {cfg['model'].get('detach_memory_writes')}")
print()

for epoch in range(epochs):
    epoch_maximum = 0.0

    for batch_index, batch in enumerate(loader):
        step += 1

        batch = _move_batch(
            batch,
            torch.device("cpu"),
        )

        budget = budgets[
            (epoch + batch_index) % len(budgets)
        ]

        optimizer.zero_grad(set_to_none=True)

        output = model(
            batch.input_ids,
            budget=budget,
        )

        loss = total_training_loss(
            output,
            batch.target_ids,
            model_name=MODEL_NAME,
            budget=budget,
            cfg=cfg,
        )

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                "Non-finite loss at "
                f"epoch={epoch + 1}, "
                f"batch={batch_index + 1}"
            )

        loss.backward()

        raw_norm = global_gradient_norm(model)
        epoch_maximum = max(
            epoch_maximum,
            raw_norm,
        )

        top_parameters = parameter_gradient_table(
            model
        )

        trace_rows.append(
            {
                "step": step,
                "epoch": epoch + 1,
                "batch": batch_index + 1,
                "budget": budget,
                "loss": float(
                    loss.detach().cpu().item()
                ),
                "raw_gradient_norm": raw_norm,
                "exceeds_threshold": (
                    raw_norm > allowed_raw_norm
                ),
                "largest_parameter": (
                    top_parameters[0]["parameter"]
                    if top_parameters
                    else ""
                ),
                "largest_parameter_gradient": (
                    top_parameters[0]["gradient_norm"]
                    if top_parameters
                    else 0.0
                ),
            }
        )

        if raw_norm > allowed_raw_norm:
            spike_payload = {
                "status": "SPIKE_REPRODUCED",
                "step": step,
                "epoch": epoch + 1,
                "batch": batch_index + 1,
                "budget": budget,
                "loss": float(
                    loss.detach().cpu().item()
                ),
                "raw_gradient_norm": raw_norm,
                "allowed_raw_gradient_norm": allowed_raw_norm,
                "threshold_excess": (
                    raw_norm - allowed_raw_norm
                ),
                "top_parameter_gradients": (
                    top_parameters[:20]
                ),
                "component_profile": (
                    profile_spike_components(
                        model,
                        batch,
                        budget=budget,
                        cfg=cfg,
                    )
                ),
            }

            print(
                "SPIKE FOUND: "
                f"epoch={epoch + 1}, "
                f"batch={batch_index + 1}, "
                f"budget={budget}, "
                f"loss={loss.item():.6f}, "
                f"raw_norm={raw_norm:.6f}"
            )

            break

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=clip_norm,
            error_if_nonfinite=True,
        )

        optimizer.step()

    print(
        f"Epoch {epoch + 1:02d}: "
        f"maximum raw gradient="
        f"{epoch_maximum:.6f}"
    )

    if spike_payload is not None:
        break


elapsed = time.perf_counter() - start_time

trace_frame = pd.DataFrame(
    trace_rows
)

trace_frame.to_csv(
    OUTPUT_CSV,
    index=False,
)

if spike_payload is None:
    status = "NO_SPIKE_REPRODUCED"

    maximum_row = (
        trace_frame.sort_values(
            "raw_gradient_norm",
            ascending=False,
        )
        .iloc[0]
        .to_dict()
    )

    payload = {
        "status": status,
        "elapsed_seconds": elapsed,
        "maximum_observed_step": maximum_row,
        "allowed_raw_gradient_norm": allowed_raw_norm,
        "trace_csv": str(OUTPUT_CSV),
    }

else:
    status = "SPIKE_REPRODUCED"

    payload = {
        **spike_payload,
        "elapsed_seconds": elapsed,
        "trace_csv": str(OUTPUT_CSV),
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
print("=" * 100)
print("FINAL TRACE STATUS")
print("=" * 100)
print(f"Status: {status}")
print(f"Elapsed seconds: {elapsed:.2f}")
print()

if status == "SPIKE_REPRODUCED":
    print(
        "The training-wide instability was reproduced."
    )
    print(
        f"First offending step: epoch "
        f"{payload['epoch']}, batch {payload['batch']}"
    )
    print(
        f"Raw gradient norm: "
        f"{payload['raw_gradient_norm']:.6f}"
    )
    print()
    print("Component profile:")

    for component in payload["component_profile"]:
        print(
            f"- {component['component']}: "
            f"loss={component['loss_value']:.6f}, "
            f"raw_norm="
            f"{component['raw_gradient_norm']:.6f}"
        )

else:
    print(
        "The full targeted training trace did not "
        "reproduce a raw gradient above 100."
    )
    print(
        "The earlier stability failure may belong to a "
        "different configuration, stale artifact, or "
        "non-reproducible execution path."
    )

print()
print("Files:")
print(f"- {OUTPUT_CSV}")
print(f"- {OUTPUT_JSON}")
PY

echo
echo "============================================================"
echo " Gradient trace complete"
echo "============================================================"
echo
echo "No source files were modified."
echo "No thresholds were changed."
echo "No full pilot was run."
echo "No commit or push was performed."
echo
echo "Saved:"
echo "  $OUTPUT_TXT"
echo "  $OUTPUT_CSV"
echo "  $OUTPUT_JSON"
