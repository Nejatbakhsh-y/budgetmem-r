"""Novelty-only cache policy."""

from budgetmem.baselines.controlled import NoveltyPolicy

NoveltyOnlyPolicy = NoveltyPolicy

__all__ = ["NoveltyPolicy", "NoveltyOnlyPolicy"]
