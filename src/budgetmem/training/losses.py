"""Composite BudgetMem-R objective with auditable component losses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor

from budgetmem.memory.state import BudgetMemoryState


@dataclass(frozen=True)
class BudgetMemLossWeights:
    budget: float = 1.0
    write: float = 0.01
    auxiliary: float = 0.1
    diversity: float = 0.01


def _diversity_loss(state: BudgetMemoryState) -> Tensor:
    losses: list[Tensor] = []
    for batch_index in range(state.batch_size):
        resident = state.values[batch_index, state.valid[batch_index]]
        if resident.shape[0] < 2:
            continue
        normalized = functional.normalize(resident, dim=-1)
        similarity = normalized @ normalized.transpose(0, 1)
        off_diagonal = ~torch.eye(
            similarity.shape[0], device=similarity.device, dtype=torch.bool
        )
        losses.append(similarity[off_diagonal].pow(2).mean())
    if not losses:
        return state.values.new_zeros(())
    return torch.stack(losses).mean()


def budgetmem_objective(
    outputs: dict[str, Tensor | BudgetMemoryState],
    *,
    task_targets: Tensor,
    task_loss: Callable[[Tensor, Tensor], Tensor],
    inputs: Tensor,
    weights: BudgetMemLossWeights = BudgetMemLossWeights(),
    target_write_rate: float = 0.25,
) -> dict[str, Tensor]:
    """Compute task, budget, write, self-supervised, and diversity losses.

    Final task targets are used only by ``task_loss``. They are never supplied to
    the model or to its write, eviction, or retrieval controllers.
    """

    logits = outputs["logits"]
    auxiliary_predictions = outputs["auxiliary_predictions"]
    write_probabilities = outputs["write_probabilities"]
    memory_sizes = outputs["memory_sizes"]
    budgets = outputs["budgets"]
    final_memory = outputs["final_memory"]
    if not isinstance(logits, Tensor):
        raise TypeError("outputs['logits'] must be a tensor")
    if not isinstance(auxiliary_predictions, Tensor):
        raise TypeError("outputs['auxiliary_predictions'] must be a tensor")
    if not isinstance(write_probabilities, Tensor):
        raise TypeError("outputs['write_probabilities'] must be a tensor")
    if not isinstance(memory_sizes, Tensor) or not isinstance(budgets, Tensor):
        raise TypeError("memory sizes and budgets must be tensors")
    if not isinstance(final_memory, BudgetMemoryState):
        raise TypeError("outputs['final_memory'] must be BudgetMemoryState")

    primary = task_loss(logits, task_targets)
    overflow = torch.relu(memory_sizes.to(inputs.dtype) - budgets.unsqueeze(1).to(inputs.dtype))
    budget_loss = overflow.pow(2).mean()
    write_loss = torch.relu(write_probabilities.mean() - target_write_rate).pow(2)
    if inputs.shape[1] > 1:
        auxiliary_loss = functional.mse_loss(
            auxiliary_predictions[:, :-1], inputs[:, 1:]
        )
    else:
        auxiliary_loss = inputs.new_zeros(())
    diversity_loss = _diversity_loss(final_memory)
    total = (
        primary
        + weights.budget * budget_loss
        + weights.write * write_loss
        + weights.auxiliary * auxiliary_loss
        + weights.diversity * diversity_loss
    )
    return {
        "total": total,
        "task": primary,
        "budget": budget_loss,
        "write": write_loss,
        "auxiliary": auxiliary_loss,
        "diversity": diversity_loss,
    }
