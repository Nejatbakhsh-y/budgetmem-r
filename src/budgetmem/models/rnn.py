"""Vanilla RNN baseline."""

from budgetmem.baselines.controlled import VanillaRNNBaseline

RNNBaseline = VanillaRNNBaseline
RNNModel = VanillaRNNBaseline

__all__ = ["VanillaRNNBaseline", "RNNBaseline", "RNNModel"]
