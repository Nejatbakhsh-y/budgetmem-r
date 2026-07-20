"""Segment-recurrent Transformer baseline."""

from budgetmem.baselines.controlled import RecurrentMemoryTransformer

RMTBaseline = RecurrentMemoryTransformer

__all__ = ["RecurrentMemoryTransformer", "RMTBaseline"]
