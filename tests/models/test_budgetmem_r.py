from __future__ import annotations

import inspect

import pytest
import torch

from budgetmem.models.budgetmem_r import BudgetMemR
from budgetmem.training.budgetmem_loss import compute_budgetmem_r_loss

torch.set_num_threads(1)


def _model(**overrides: object) -> BudgetMemR:
    config: dict[str, object] = {
        "input_dim": 6,
        "hidden_dim": 12,
        "output_dim": 3,
        "max_budget": 5,
        "allowed_budgets": (1, 2, 3, 4, 5),
        "key_dim": 8,
        "value_dim": 10,
        "retrieval_k": 2,
        "write_threshold": 0.0,
    }
    config.update(overrides)
    return BudgetMemR(**config)


def test_strict_budget_is_never_violated() -> None:
    torch.manual_seed(2026)
    model = _model().eval()
    inputs = torch.randn(5, 11, 6)
    budgets = torch.tensor([1, 2, 3, 4, 5])

    output = model(inputs, budget=budgets)

    assert torch.all(output.memory_sizes <= budgets.unsqueeze(1))
    assert torch.equal(output.final_memory.sizes(), budgets)
    output.final_memory.assert_invariant()


def test_eval_is_deterministic() -> None:
    torch.manual_seed(2026)
    model = _model(write_threshold=0.5).eval()
    inputs = torch.randn(3, 7, 6)

    first = model(inputs, budget=3)
    second = model(inputs, budget=3)

    assert torch.equal(first.hard_writes, second.hard_writes)
    assert torch.equal(first.write_slots, second.write_slots)
    assert torch.allclose(first.sequence_logits, second.sequence_logits)


def test_forward_api_does_not_accept_task_labels() -> None:
    parameters = inspect.signature(BudgetMemR.forward).parameters
    forbidden = {"target", "targets", "label", "labels", "y"}
    assert forbidden.isdisjoint(parameters)


def test_training_loss_backpropagates_to_write_controller() -> None:
    torch.manual_seed(2026)
    model = _model(write_threshold=0.0).train()
    inputs = torch.randn(4, 8, 6)
    targets = torch.tensor([0, 1, 2, 1])

    output = model(inputs, budget=torch.tensor([1, 2, 3, 4]))
    losses = compute_budgetmem_r_loss(output, targets)
    losses.total.backward()

    write_gradient_sum = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.write_controller.parameters()
        if parameter.grad is not None
    )
    utility_gradient_sum = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.utility_controller.parameters()
        if parameter.grad is not None
    )
    assert write_gradient_sum > 0.0
    assert utility_gradient_sum > 0.0
    assert losses.budget.item() == pytest.approx(0.0)


def test_budget_sampler_uses_only_allowed_values() -> None:
    torch.manual_seed(2026)
    model = _model().train()
    sampled = model.sample_budgets(200, torch.device("cpu"))
    assert set(sampled.tolist()).issubset(set(model.allowed_budgets))


@pytest.mark.parametrize(
    "fusion",
    ["concatenation", "residual", "gated", "attention"],
)
def test_all_fusion_modes_produce_expected_shapes(fusion: str) -> None:
    model = _model(fusion=fusion).eval()
    output = model(torch.randn(2, 5, 6), budget=3)
    assert output.logits.shape == (2, 3)
    assert output.sequence_logits.shape == (2, 5, 3)
    assert output.retrieval_weights.shape == (2, 5, model.retrieval_k)


@pytest.mark.parametrize("backbone", ["rnn", "gru", "lstm"])
def test_all_recurrent_backbones_run(backbone: str) -> None:
    model = _model(backbone=backbone).eval()
    output = model(torch.randn(2, 5, 6), budget=2)
    assert output.hidden_states.shape == (2, 5, 12)


def test_invalid_budget_is_rejected() -> None:
    model = _model().eval()
    inputs = torch.randn(2, 4, 6)
    with pytest.raises(ValueError, match="Budgets must be within"):
        model(inputs, budget=6)
