"""Least-recently-used cache policy."""

from budgetmem.baselines.controlled import LRUPolicy

LRU = LRUPolicy

__all__ = ["LRUPolicy", "LRU"]
