"""Seeded random-replacement cache policy."""

from budgetmem.baselines.controlled import RandomReplacementPolicy

RandomPolicy = RandomReplacementPolicy

__all__ = ["RandomReplacementPolicy", "RandomPolicy"]
