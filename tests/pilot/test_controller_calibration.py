"""Regression tests for deterministic BudgetMem-R write calibration."""

from __future__ import annotations

import torch

from budgetmem.experiments.pilot import PilotOutput, total_training_loss
from budgetmem.models.budgetmem_r import BudgetMemR


def test_training_and_evaluation_use_the_same_hard_write_decision() -> None:
    model = BudgetMemR(
        input_dim=4,
        hidden_dim=8,
        output_dim=3,
        max_budget=4,
        allowed_budgets=(2, 4),
        write_threshold=0.5,
    )
    logits = torch.tensor([-1.0, 1.0])
    model.train()
    _, training_hard, _ = model._write_gate(logits)
    model.eval()
    _, evaluation_hard, _ = model._write_gate(logits)
    assert torch.equal(training_hard, evaluation_hard)


def test_write_rate_loss_pushes_collapsed_probabilities_upward() -> None:
    probabilities = torch.full((2, 8), 0.10, requires_grad=True)
    logits = torch.zeros(2, 12, 192, requires_grad=True)
    target = torch.zeros(2, 12, dtype=torch.long)
    output = PilotOutput(logits=logits, write_probabilities=probabilities)
    cfg = {
        "training": {
            "write_threshold": 0.5,
            "write_rate_target": 0.15,
            "write_rate_penalty": 5.0,
            "write_binarization_penalty": 0.05,
            "budget_violation_penalty": 10.0,
        }
    }
    loss = total_training_loss(
        output,
        target,
        model_name="budgetmem_r",
        budget=16,
        cfg=cfg,
    )
    loss.backward()
    assert probabilities.grad is not None
    assert float(probabilities.grad.mean()) < 0.0
