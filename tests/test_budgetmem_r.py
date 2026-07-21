from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as functional

from budgetmem.memory.budget import DEFAULT_TRAINING_BUDGETS, sample_training_budgets
from budgetmem.memory.state import BudgetMemoryState
from budgetmem.models.budgetmem_r import BudgetMemR
from budgetmem.training.losses import budgetmem_objective


def _model(*, fusion_mode: str = "gated", max_budget: int = 16) -> BudgetMemR:
    return BudgetMemR(
        input_dim=6,
        hidden_dim=12,
        output_dim=3,
        key_dim=8,
        value_dim=10,
        budget_embedding_dim=5,
        controller_dim=16,
        max_budget=max_budget,
        training_budgets=(4, 8, 16),
        top_k=3,
        fusion_mode=fusion_mode,
    )


def _force_writes(model: BudgetMemR) -> None:
    with torch.no_grad():
        for parameter in model.write_controller.parameters():
            parameter.zero_()
        final_layer = model.write_controller.network[-1]
        assert isinstance(final_layer, torch.nn.Linear)
        final_layer.bias.fill_(20.0)


def test_training_budget_sampling_uses_only_controlled_choices() -> None:
    generator = torch.Generator().manual_seed(2026)
    sampled = sample_training_budgets(
        256,
        device=torch.device("cpu"),
        choices=DEFAULT_TRAINING_BUDGETS,
        generator=generator,
    )
    assert set(sampled.tolist()).issubset(set(DEFAULT_TRAINING_BUDGETS))
    assert len(set(sampled.tolist())) > 1


def test_hard_budget_is_never_violated_even_when_every_step_writes() -> None:
    model = _model().eval()
    _force_writes(model)
    inputs = torch.randn(2, 24, 6)
    budgets = torch.tensor([4, 8])
    with torch.no_grad():
        outputs = model(inputs, budget=budgets)
    sizes = outputs["memory_sizes"]
    assert isinstance(sizes, torch.Tensor)
    assert torch.all(sizes <= budgets.unsqueeze(1))
    assert sizes[:, -1].tolist() == [4, 8]
    assert int(outputs["budget_violations"].item()) == 0
    final_memory = outputs["final_memory"]
    assert isinstance(final_memory, BudgetMemoryState)
    final_memory.assert_within_budget()


def test_forward_api_prevents_final_label_leakage() -> None:
    parameters = inspect.signature(BudgetMemR.forward).parameters
    forbidden = {"label", "labels", "target", "targets", "sentiment", "anomaly"}
    assert forbidden.isdisjoint(parameters)


def test_all_fusion_comparison_modes_execute() -> None:
    inputs = torch.randn(2, 5, 6)
    for mode in ("concatenation", "residual", "gated", "attention"):
        model = _model(fusion_mode=mode).eval()
        with torch.no_grad():
            outputs = model(inputs, budget=4)
        assert outputs["logits"].shape == (2, 3)
        assert outputs["retrieval_weights"].shape == (2, 5, 3)


def test_composite_objective_is_finite_and_controller_receives_gradient() -> None:
    torch.manual_seed(7)
    model = _model().train()
    inputs = torch.randn(3, 7, 6)
    targets = torch.tensor([0, 1, 2])
    outputs = model(inputs, budget=torch.tensor([4, 8, 16]))
    losses = budgetmem_objective(
        outputs,
        task_targets=targets,
        task_loss=functional.cross_entropy,
        inputs=inputs,
    )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    gradients = [
        parameter.grad
        for parameter in model.write_controller.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_invalid_budget_fails_fast() -> None:
    model = _model(max_budget=16)
    with pytest.raises(ValueError, match="Budgets must be within"):
        model(torch.randn(1, 3, 6), budget=17)
