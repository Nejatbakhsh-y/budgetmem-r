"""Tensorized, per-sample bounded external-memory state."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class BudgetMemoryState:
    """Batched external memory with a hard per-sample deployment budget."""

    keys: Tensor
    values: Tensor
    utility: Tensor
    age: Tensor
    retrieval_count: Tensor
    valid: Tensor
    budgets: Tensor
    last_write_step: Tensor

    @classmethod
    def empty(
        cls,
        *,
        batch_size: int,
        capacity: int,
        key_dim: int,
        value_dim: int,
        budgets: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "BudgetMemoryState":
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if budgets.shape != (batch_size,):
            raise ValueError("budgets must contain one value per batch item")
        return cls(
            keys=torch.zeros(batch_size, capacity, key_dim, device=device, dtype=dtype),
            values=torch.zeros(
                batch_size, capacity, value_dim, device=device, dtype=dtype
            ),
            utility=torch.zeros(batch_size, capacity, device=device, dtype=dtype),
            age=torch.zeros(batch_size, capacity, device=device, dtype=torch.long),
            retrieval_count=torch.zeros(
                batch_size, capacity, device=device, dtype=dtype
            ),
            valid=torch.zeros(batch_size, capacity, device=device, dtype=torch.bool),
            budgets=budgets.to(device=device, dtype=torch.long),
            last_write_step=torch.full(
                (batch_size,), -1, device=device, dtype=torch.long
            ),
        )

    @property
    def capacity(self) -> int:
        return int(self.keys.shape[1])

    @property
    def batch_size(self) -> int:
        return int(self.keys.shape[0])

    def sizes(self) -> Tensor:
        return self.valid.sum(dim=1)

    def within_budget(self) -> Tensor:
        return self.sizes() <= self.budgets

    def assert_within_budget(self) -> None:
        if not bool(torch.all(self.within_budget())):
            sizes = self.sizes().detach().cpu().tolist()
            budgets = self.budgets.detach().cpu().tolist()
            raise RuntimeError(
                f"Hard memory budget violated: sizes={sizes}, budgets={budgets}"
            )

    def assert_invariant(self) -> None:
        """Validate all hard memory-state invariants.

        This compatibility method currently delegates to the strict
        per-sample budget invariant.
        """
        self.assert_within_budget()

