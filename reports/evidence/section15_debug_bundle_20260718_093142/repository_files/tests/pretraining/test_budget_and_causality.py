"""Hard-budget and strict-causality tests for BudgetMem-R."""

from __future__ import annotations

import torch

from tests.pretraining.conftest import (
    allowed_budgets,
    build_budgetmem_model,
    make_inputs,
)


def test_memory_never_exceeds_configured_budget_at_any_forward_step() -> None:
    model = build_budgetmem_model().eval()
    inputs = make_inputs(model, batch_size=3, sequence_length=12)

    with torch.no_grad():
        for budget in allowed_budgets(model):
            output = model(inputs, budget=budget)
            expected_budgets = output.budgets.unsqueeze(1).expand_as(
                output.memory_sizes
            )

            assert torch.all(output.memory_sizes <= expected_budgets)
            assert torch.equal(
                output.memory_masks.sum(dim=-1),
                output.memory_sizes,
            )
            assert torch.all(output.final_memory.sizes() <= output.final_memory.budgets)
            output.final_memory.assert_invariant()


def test_future_tokens_do_not_change_earlier_memory_decisions() -> None:
    model = build_budgetmem_model().eval()
    inputs = make_inputs(model, batch_size=2, sequence_length=11, seed=7001)
    changed = inputs.clone()
    cutoff = 6

    generator = torch.Generator(device="cpu")
    generator.manual_seed(7002)
    changed[:, cutoff:, :] = torch.randn(
        changed[:, cutoff:, :].shape,
        generator=generator,
    )

    budget = allowed_budgets(model)[0]
    with torch.no_grad():
        original = model(inputs, budget=budget)
        perturbed = model(changed, budget=budget)

    floating_prefixes = (
        (original.sequence_logits, perturbed.sequence_logits),
        (original.hidden_states, perturbed.hidden_states),
        (original.write_probabilities, perturbed.write_probabilities),
        (original.retrieval_weights, perturbed.retrieval_weights),
    )
    for left, right in floating_prefixes:
        torch.testing.assert_close(
            left[:, :cutoff],
            right[:, :cutoff],
            rtol=0.0,
            atol=0.0,
        )

    exact_prefixes = (
        (original.hard_writes, perturbed.hard_writes),
        (original.write_slots, perturbed.write_slots),
        (original.eviction_flags, perturbed.eviction_flags),
        (original.memory_masks, perturbed.memory_masks),
        (original.memory_sizes, perturbed.memory_sizes),
    )
    for left, right in exact_prefixes:
        assert torch.equal(left[:, :cutoff], right[:, :cutoff])
