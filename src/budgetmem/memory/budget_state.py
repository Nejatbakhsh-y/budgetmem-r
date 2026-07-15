"""Tensor container for the strict-budget external memory used by BudgetMem-R."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ExternalMemoryState:
    """Per-batch external-memory tensors.

    Every tensor reserves ``max_budget`` physical slots, while ``budgets``
    defines the strict logical budget for each sample. A slot is usable only
    when ``valid`` is true. BudgetMem-R never marks more than ``budgets[b]``
    slots valid for sample ``b``.
    """

    keys: Tensor
    values: Tensor
    utility: Tensor
    age: Tensor
    retrieval_count: Tensor
    valid: Tensor
    last_write_step: Tensor
    budgets: Tensor

    @classmethod
    def empty(
        cls,
        *,
        batch_size: int,
        max_budget: int,
        key_dim: int,
        value_dim: int,
        budgets: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "ExternalMemoryState":
        if budgets.shape != (batch_size,):
            raise ValueError("budgets must have shape [batch_size]")
        if torch.any(budgets < 1):
            raise ValueError("every budget must be at least 1")
        if torch.any(budgets > max_budget):
            raise ValueError("a requested budget exceeds max_budget")

        return cls(
            keys=torch.zeros(
                batch_size, max_budget, key_dim, device=device, dtype=dtype
            ),
            values=torch.zeros(
                batch_size,
                max_budget,
                value_dim,
                device=device,
                dtype=dtype,
            ),
            utility=torch.zeros(batch_size, max_budget, device=device, dtype=dtype),
            age=torch.zeros(batch_size, max_budget, device=device, dtype=dtype),
            retrieval_count=torch.zeros(
                batch_size,
                max_budget,
                device=device,
                dtype=dtype,
            ),
            valid=torch.zeros(batch_size, max_budget, device=device, dtype=torch.bool),
            last_write_step=torch.full(
                (batch_size,),
                fill_value=-1,
                device=device,
                dtype=torch.long,
            ),
            budgets=budgets.to(device=device, dtype=torch.long),
        )

    @property
    def batch_size(self) -> int:
        return int(self.keys.shape[0])

    @property
    def max_budget(self) -> int:
        return int(self.keys.shape[1])

    def sizes(self) -> Tensor:
        """Return the number of valid slots for every sample."""
        return self.valid.sum(dim=1)

    def allowed_slot_mask(self) -> Tensor:
        """Return slots that belong to each sample's logical budget."""
        slot_ids = torch.arange(self.max_budget, device=self.keys.device)
        return slot_ids.unsqueeze(0) < self.budgets.unsqueeze(1)

    def increment_age(self) -> "ExternalMemoryState":
        """Increase age for valid entries without changing invalid entries."""
        self.age = self.age + self.valid.to(self.age.dtype)
        return self

    def assert_invariant(self) -> None:
        """Raise if the strict per-sample budget invariant is violated."""
        sizes = self.sizes()
        if torch.any(sizes > self.budgets):
            raise RuntimeError("strict memory-budget invariant violated")
        if torch.any(self.valid & ~self.allowed_slot_mask()):
            raise RuntimeError("a slot outside a sample's logical budget is valid")
