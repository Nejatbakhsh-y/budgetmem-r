"""Determinism tests for data, initialization, ordering, and evaluation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from budgetmem.utils.reproducibility import seed_everything, seeded_generator
from tests.pretraining.conftest import (
    allowed_budgets,
    build_budgetmem_model,
    make_inputs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_GENERATOR = REPO_ROOT / "scripts" / "data" / "generate_synthetic.py"


def _load_synthetic_generator() -> ModuleType:
    if not SYNTHETIC_GENERATOR.exists():
        raise FileNotFoundError(
            f"Synthetic generator is missing: {SYNTHETIC_GENERATOR}"
        )

    spec = importlib.util.spec_from_file_location(
        "_budgetmem_synthetic_generator",
        SYNTHETIC_GENERATOR,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SYNTHETIC_GENERATOR}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_same_seed_and_configuration_produce_identical_synthetic_example() -> None:
    module = _load_synthetic_generator()
    generator = module.GENERATORS["delayed_xor"]
    config = {
        "sequence_length": 40,
        "vocabulary_size": 32,
        "number_keys": 2,
        "number_queries": 1,
        "delay_length": 12,
        "distractor_percentage": 90,
        "number_relevant_events": 2,
        "random_seed": 104,
    }

    first = generator(config, np.random.default_rng(44001))
    second = generator(config, np.random.default_rng(44001))
    different = generator(config, np.random.default_rng(44002))

    assert first == second
    assert first != different


def test_same_seed_produces_identical_model_initialization_and_evaluation() -> None:
    seed_everything(2026)
    first_model = build_budgetmem_model().eval()
    first_state = {
        name: tensor.detach().clone()
        for name, tensor in first_model.state_dict().items()
    }

    seed_everything(2026)
    second_model = build_budgetmem_model().eval()
    second_state = second_model.state_dict()

    assert first_state.keys() == second_state.keys()
    for name in first_state:
        assert torch.equal(first_state[name], second_state[name])

    inputs = make_inputs(first_model, batch_size=2, sequence_length=10, seed=8080)
    budget = allowed_budgets(first_model)[0]
    with torch.no_grad():
        first_output = first_model(inputs, budget=budget)
        second_output = second_model(inputs, budget=budget)

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
            getattr(first_output, name),
            getattr(second_output, name),
        )


def _loader_order(seed: int) -> list[int]:
    dataset = TensorDataset(torch.arange(32))
    loader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=True,
        num_workers=0,
        generator=seeded_generator(seed),
    )
    return [int(value) for batch in loader for value in batch[0].tolist()]


def test_same_seed_produces_identical_training_order() -> None:
    assert _loader_order(91001) == _loader_order(91001)
    assert _loader_order(91001) != _loader_order(91002)
