"""Composite training objective for BudgetMem-R."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from budgetmem.models.budgetmem_r import BudgetMemROutput

TaskKind = Literal["classification", "regression"]


@dataclass(frozen=True)
class BudgetMemRLoss:
    """Named loss terms for optimization and experiment logging."""

    total: Tensor
    task: Tensor
    budget: Tensor
    write: Tensor
    auxiliary: Tensor
    diversity: Tensor


def _task_loss(
    output: BudgetMemROutput,
    target: Tensor,
    *,
    task_kind: TaskKind,
    sequence_task: bool,
) -> Tensor:
    predictions = output.sequence_logits if sequence_task else output.logits
    if task_kind == "classification":
        if sequence_task:
            if target.shape != predictions.shape[:2]:
                raise ValueError(
                    "sequence classification target must have shape [batch, sequence]"
                )
            return F.cross_entropy(
                predictions.reshape(-1, predictions.shape[-1]),
                target.reshape(-1).long(),
            )
        if target.shape != predictions.shape[:1]:
            raise ValueError("classification target must have shape [batch]")
        return F.cross_entropy(predictions, target.long())

    if task_kind == "regression":
        expected_shape = predictions.shape
        if target.shape == predictions.shape[:-1] and predictions.shape[-1] == 1:
            target = target.unsqueeze(-1)
        if target.shape != expected_shape:
            raise ValueError("regression target shape must match model predictions")
        return F.mse_loss(predictions, target.to(predictions.dtype))

    raise ValueError(f"unsupported task kind: {task_kind}")


def _auxiliary_loss(output: BudgetMemROutput) -> Tensor:
    if output.inputs.shape[1] < 2:
        return output.logits.new_zeros(())
    target = output.inputs[:, 1:]
    mean = output.auxiliary_mean[:, :-1]
    log_variance = output.auxiliary_log_variance[:, :-1]
    inverse_variance = torch.exp(-log_variance)
    gaussian_nll = 0.5 * (inverse_variance * (target - mean).pow(2) + log_variance)
    return gaussian_nll.mean()


def _diversity_loss(output: BudgetMemROutput) -> Tensor:
    values = output.final_memory.values
    valid = output.final_memory.valid
    normalized = F.normalize(values, p=2, dim=-1, eps=1.0e-8)
    similarity = torch.matmul(normalized, normalized.transpose(1, 2))
    pair_mask = valid.unsqueeze(1) & valid.unsqueeze(2)
    diagonal = torch.eye(
        values.shape[1],
        device=values.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    pair_mask = pair_mask & ~diagonal
    pair_count = pair_mask.sum()
    if int(pair_count.item()) == 0:
        return values.new_zeros(())
    return torch.relu(similarity[pair_mask]).mean()


def compute_budgetmem_r_loss(
    output: BudgetMemROutput,
    target: Tensor,
    *,
    task_kind: TaskKind = "classification",
    sequence_task: bool = False,
    lambda_budget: float = 1.0,
    lambda_write: float = 0.01,
    lambda_auxiliary: float = 0.1,
    lambda_diversity: float = 0.01,
    max_write_fraction: float = 0.25,
) -> BudgetMemRLoss:
    """Compute the Section 13.7 composite objective.

    ``target`` is used only by the primary task loss after the model forward
    pass. It is never supplied to the write, eviction, or retrieval policy.
    """
    if not 0.0 <= max_write_fraction <= 1.0:
        raise ValueError("max_write_fraction must be in [0, 1]")

    task = _task_loss(
        output,
        target,
        task_kind=task_kind,
        sequence_task=sequence_task,
    )
    over_budget = torch.relu(
        output.memory_sizes.to(output.logits.dtype) - output.budgets.unsqueeze(1)
    )
    budget = over_budget.pow(2).mean()
    write_rate = output.write_probabilities.mean(dim=1)
    write = torch.relu(write_rate - max_write_fraction).pow(2).mean()
    auxiliary = _auxiliary_loss(output)
    diversity = _diversity_loss(output)
    total = (
        task
        + lambda_budget * budget
        + lambda_write * write
        + lambda_auxiliary * auxiliary
        + lambda_diversity * diversity
    )
    return BudgetMemRLoss(
        total=total,
        task=task,
        budget=budget,
        write=write,
        auxiliary=auxiliary,
        diversity=diversity,
    )
