"""Regression tests for the Section 15 write-gradient repair."""

from __future__ import annotations

import torch

from budgetmem.models.budgetmem_r import BudgetMemR


def _model() -> BudgetMemR:
    return BudgetMemR(
        input_dim=6,
        hidden_dim=12,
        output_dim=4,
        max_budget=4,
        training_budgets=(2, 4),
        key_dim=8,
        value_dim=10,
        retrieval_k=2,
        write_threshold=1.0,
    )


def test_empty_memory_is_filled_before_replacement_control() -> None:
    torch.manual_seed(2026)

    model = _model().eval()
    inputs = torch.randn(3, 9, 6)

    with torch.no_grad():
        output = model(inputs, budget=2)

    assert torch.equal(
        output.final_memory.sizes(),
        torch.full((3,), 2, dtype=torch.long),
    )

    assert torch.all(
        output.memory_sizes <= 2
    )

    assert torch.all(
        output.hard_writes[:, :2] == 1
    )


def test_task_path_reaches_write_controller_after_memory_is_full() -> None:
    torch.manual_seed(2026)

    model = _model().train()
    inputs = torch.randn(3, 10, 6)

    output = model(inputs, budget=2)

    loss = output.logits.square().mean()
    loss.backward()

    gradient_sum = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.write_controller.parameters()
        if parameter.grad is not None
    )

    assert gradient_sum > 0.0
