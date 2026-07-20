"""FIFO cache policy."""

from budgetmem.baselines.controlled import FIFOPolicy

FIFO = FIFOPolicy

__all__ = ["FIFOPolicy", "FIFO"]
