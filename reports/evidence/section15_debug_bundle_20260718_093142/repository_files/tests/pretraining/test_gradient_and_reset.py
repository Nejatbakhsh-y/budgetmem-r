"""Gradient-connectivity and memory-reset tests for BudgetMem-R."""

from __future__ import annotations

import copy

import torch
from torch import nn

from tests.pretraining.conftest import (
    allowed_budgets,
    build_budgetmem_model,
    make_inputs,
)


def _force_controller_to_write(model: nn.Module) -> None:
    controller = getattr(model, "write_controller")
    linear_layers = [
        module for module in controller.modules() if isinstance(module, nn.Linear)
    ]
    if not linear_layers:
        raise AssertionError("Write controller contains no Linear layer")

    final_layer = linear_layers[-1]
    with torch.no_grad():
        final_layer.weight.zero_()
        if final_layer.bias is None:
            raise AssertionError("Write-controller output layer has no bias")
        final_layer.bias.fill_(50.0)


def _has_nonzero_gradient(module: nn.Module) -> bool:
    trainable = [
        parameter for parameter in module.parameters() if parameter.requires_grad
    ]
    assert trainable, "Expected trainable controller parameters"
    return any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in trainable
    )


def test_memory_controllers_receive_gradients_and_graph_policy_is_explicit() -> None:
    # Test the write controller under its normal, nonsaturated initialization.
    torch.manual_seed(12000)
    write_model = build_budgetmem_model().train()

    inputs = make_inputs(
        write_model,
        batch_size=2,
        sequence_length=10,
        seed=12001,
    )
    budget = allowed_budgets(write_model)[0]
    write_output = write_model(inputs, budget=budget)

    write_loss = write_output.write_probabilities.mean()
    write_loss.backward()

    assert _has_nonzero_gradient(write_model.write_controller)

    # Use a separate model to force writes and evictions. The forced controller
    # is intentionally saturated, so its gradient is not tested in this pass.
    torch.manual_seed(12000)
    memory_model = build_budgetmem_model().train()
    _force_controller_to_write(memory_model)

    memory_output = memory_model(inputs, budget=budget)

    assert bool(torch.all(memory_output.hard_writes))
    assert bool(torch.any(memory_output.eviction_flags))

    utility_loss = (
        memory_output.sequence_logits.square().mean()
        + 0.01 * memory_output.final_memory.utility.sum()
    )
    utility_loss.backward()

    assert _has_nonzero_gradient(memory_model.utility_controller)

    assert memory_output.final_memory.keys.requires_grad
    assert memory_output.final_memory.values.requires_grad
    assert memory_output.final_memory.utility.requires_grad

    assert not memory_output.final_memory.retrieval_count.requires_grad
    assert not memory_output.final_memory.valid.requires_grad
    assert not memory_output.final_memory.last_write_step.requires_grad
    assert not memory_output.final_memory.budgets.requires_grad


def test_memory_is_reset_between_unrelated_forward_calls() -> None:
    model = build_budgetmem_model().eval()
    fresh_model = copy.deepcopy(model).eval()
    budget = allowed_budgets(model)[0]

    unrelated = make_inputs(
        model,
        batch_size=2,
        sequence_length=7,
        seed=13001,
    )
    target = make_inputs(
        model,
        batch_size=2,
        sequence_length=9,
        seed=13002,
    )

    with torch.no_grad():
        model(unrelated, budget=budget)
        after_unrelated = model(target, budget=budget)
        fresh = fresh_model(target, budget=budget)

    for name in (
        "logits",
        "sequence_logits",
        "hidden_states",
        "write_probabilities",
        "hard_writes",
        "write_slots",
        "eviction_flags",
        "retrieval_weights",
        "memory_masks",
        "memory_sizes",
        "budgets",
    ):
        assert torch.equal(
            getattr(after_unrelated, name),
            getattr(fresh, name),
        )

    for name in (
        "keys",
        "values",
        "utility",
        "age",
        "retrieval_count",
        "valid",
        "last_write_step",
        "budgets",
    ):
        assert torch.equal(
            getattr(after_unrelated.final_memory, name),
            getattr(fresh.final_memory, name),
        )
