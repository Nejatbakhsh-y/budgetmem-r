"""Shared helpers for the Section 14 pretraining-gate tests."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
SECTION13_TEST_FILE = REPO_ROOT / "tests" / "models" / "test_budgetmem_r.py"


@lru_cache(maxsize=1)
def _load_section13_test_module() -> ModuleType:
    if not SECTION13_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Section 13 model tests are missing: {SECTION13_TEST_FILE}"
        )

    spec = importlib.util.spec_from_file_location(
        "_budgetmem_section13_model_tests",
        SECTION13_TEST_FILE,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SECTION13_TEST_FILE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_budgetmem_model(**overrides: object) -> nn.Module:
    """Build the exact small BudgetMem-R model used by Section 13 tests."""
    module = _load_section13_test_module()
    factory = getattr(module, "_model", None)
    if factory is None:
        raise AttributeError(
            "tests/models/test_budgetmem_r.py must expose the Section 13 "
            "_model test factory"
        )

    try:
        model = factory(**overrides)
    except TypeError:
        if overrides:
            raise
        model = factory()

    if not isinstance(model, nn.Module):
        raise TypeError("Section 13 _model() did not return torch.nn.Module")
    return model


def model_input_dim(model: nn.Module) -> int:
    value = getattr(model, "input_dim", None)
    if isinstance(value, int):
        return value

    for module in model.modules():
        if isinstance(module, (nn.RNNCell, nn.GRUCell, nn.LSTMCell)):
            return int(module.input_size)

    for name in ("input_projection", "input_encoder", "encoder"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Linear):
            return int(module.in_features)

    return 6


def allowed_budgets(model: nn.Module) -> tuple[int, ...]:
    values = getattr(model, "allowed_budgets", None)
    if values is None:
        maximum = int(getattr(model, "max_budget", 4))
        return tuple(range(1, maximum + 1))

    budgets = tuple(sorted({int(value) for value in values}))
    if not budgets:
        raise AssertionError("BudgetMem-R has no allowed budgets")
    return budgets


def make_inputs(
    model: nn.Module,
    *,
    batch_size: int = 2,
    sequence_length: int = 9,
    seed: int = 2026,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(
        batch_size,
        sequence_length,
        model_input_dim(model),
        generator=generator,
    )
