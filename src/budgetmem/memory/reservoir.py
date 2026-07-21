"""Reservoir-sampling cache policy."""

from budgetmem.baselines.controlled import ReservoirPolicy

ReservoirSamplingPolicy = ReservoirPolicy

__all__ = ["ReservoirPolicy", "ReservoirSamplingPolicy"]
