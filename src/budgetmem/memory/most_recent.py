"""Most-recent-state cache policy."""

from budgetmem.baselines.controlled import MostRecentPolicy

MostRecentCachePolicy = MostRecentPolicy

__all__ = ["MostRecentPolicy", "MostRecentCachePolicy"]
