"""Uniform checkpoint-selection policy."""

from budgetmem.baselines.controlled import UniformCheckpointPolicy

UniformPolicy = UniformCheckpointPolicy

__all__ = ["UniformCheckpointPolicy", "UniformPolicy"]
