"""Deterministic execution utilities used by training and evaluation."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch for reproducible CPU execution."""
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seed


def seeded_generator(seed: int, *, device: str = "cpu") -> torch.Generator:
    """Return a PyTorch generator with a fixed seed."""
    if seed < 0:
        raise ValueError("seed must be non-negative")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
