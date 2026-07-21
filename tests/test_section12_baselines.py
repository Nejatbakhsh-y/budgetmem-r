"""Contract tests for the controlled Section 12 baseline suite."""

from __future__ import annotations

import pytest
import torch

from budgetmem.baselines.controlled import (
    BASELINE_REGISTRY,
    POLICY_REGISTRY,
    DiagonalSSMBaseline,
    GRUBaseline,
    LSTMBaseline,
    MemoryCachingBaseline,
    OfficialMambaOrSSMBaseline,
    RecurrentMemoryTransformer,
    TransformerBaseline,
    VanillaRNNBaseline,
    assert_parameter_budget,
    parameter_count,
)


@pytest.mark.parametrize(
    "model",
    [
        VanillaRNNBaseline(8, 16, 4),
        GRUBaseline(8, 16, 4),
        LSTMBaseline(8, 16, 4),
    ],
)
def test_recurrent_baselines_shape_and_gradient(model: torch.nn.Module) -> None:
    x = torch.randn(2, 12, 8)
    output = model(x)
    assert output.shape == (2, 12, 4)
    output.mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("name", sorted(POLICY_REGISTRY))
def test_cache_policies_respect_budget(name: str) -> None:
    states = torch.randn(3, 17, 8)
    kwargs = {"seed": 11} if name in {"random", "reservoir"} else {}
    indices = POLICY_REGISTRY[name](**kwargs).select_indices(states, budget=5)
    assert indices.shape == (3, 5)
    assert torch.all(indices >= 0)
    assert torch.all(indices < 17)
    assert all(len(set(row.tolist())) == 5 for row in indices)


def test_seeded_policies_are_reproducible() -> None:
    states = torch.randn(2, 19, 8)
    for name in ("random", "reservoir"):
        first = POLICY_REGISTRY[name](seed=7).select_indices(states, 6)
        second = POLICY_REGISTRY[name](seed=7).select_indices(states, 6)
        assert torch.equal(first, second)


@pytest.mark.parametrize("mode", ["full", "sliding"])
def test_attention_baselines(mode: str) -> None:
    model = TransformerBaseline(
        8,
        16,
        4,
        num_layers=1,
        num_heads=4,
        attention_mode=mode,
        window_size=4,
        max_length=64,
    )
    assert model(torch.randn(2, 14, 8)).shape == (2, 14, 4)


def test_state_space_baseline_cpu_fallback() -> None:
    model = OfficialMambaOrSSMBaseline(
        8, 12, 4, backend="s4d_reference", num_layers=1, state_dim=4
    )
    output = model(torch.randn(2, 10, 8))
    assert output.shape == (2, 10, 4)
    assert model.backend == "s4d_reference"


def test_recurrent_memory_transformer_preserves_length() -> None:
    model = RecurrentMemoryTransformer(
        8,
        16,
        4,
        segment_length=5,
        memory_tokens=2,
        num_layers=1,
        num_heads=4,
    )
    assert model(torch.randn(2, 13, 8)).shape == (2, 13, 4)


@pytest.mark.parametrize("variant", ["mean", "gated", "sparse_selective"])
def test_memory_caching_variants(variant: str) -> None:
    model = MemoryCachingBaseline(
        8, 12, 4, budget=4, policy="uniform", variant=variant
    )
    x = torch.randn(2, 15, 8)
    assert model(x).shape == (2, 15, 4)
    assert model.cache_indices(x).shape == (2, 4)


def test_registries_cover_all_required_baselines() -> None:
    assert {
        "vanilla_rnn",
        "gru",
        "lstm",
        "transformer_full",
        "transformer_sliding",
        "state_space",
        "recurrent_memory_transformer",
        "memory_caching_mean",
        "memory_caching_gated",
        "memory_caching_sparse_selective",
    }.issubset(BASELINE_REGISTRY)


def test_parameter_budget_guard() -> None:
    model = DiagonalSSMBaseline(8, 12, 4, num_layers=1, state_dim=4)
    target = parameter_count(model)
    assert_parameter_budget(model, target, tolerance=0.0)
    with pytest.raises(ValueError):
        assert_parameter_budget(model, target * 2, tolerance=0.1)
