"""Surprise-only cache policy."""

from budgetmem.baselines.controlled import SurprisePolicy

SurpriseOnlyPolicy = SurprisePolicy

__all__ = ["SurprisePolicy", "SurpriseOnlyPolicy"]
