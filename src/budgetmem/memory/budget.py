"""Budget encoding and randomized training-budget sampling for BudgetMem-R."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

DEFAULT_TRAINING_BUDGETS: tuple[int, ...] = (8, 16, 32, 64, 128)


class BudgetConditioner(nn.Module):
    """Encode a requested memory budget as a normalized learned representation."""

    def __init__(self, max_budget: int = 128, embedding_dim: int = 16) -> None:
        super().__init__()
        if max_budget <= 0:
            raise ValueError("max_budget must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.max_budget = int(max_budget)
        self.embedding_dim = int(embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def normalize(self, budgets: Tensor) -> Tensor:
        budgets = budgets.to(dtype=torch.float32)
        if torch.any(budgets < 1):
            raise ValueError("Every requested budget must be at least 1")
        if torch.any(budgets > self.max_budget):
            raise ValueError(
                f"Requested budget exceeds configured maximum {self.max_budget}"
            )
        return budgets / float(self.max_budget)

    def forward(self, budgets: Tensor) -> Tensor:
        if budgets.ndim == 0:
            budgets = budgets.unsqueeze(0)
        if budgets.ndim != 1:
            raise ValueError("budgets must be a scalar or one-dimensional tensor")
        normalized = self.normalize(budgets).unsqueeze(-1)
        return self.encoder(normalized)


def sample_training_budgets(
    batch_size: int,
    *,
    device: torch.device,
    choices: Sequence[int] = DEFAULT_TRAINING_BUDGETS,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one deployment budget per sample from the controlled training set."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not choices:
        raise ValueError("choices cannot be empty")
    choice_tensor = torch.tensor(tuple(int(value) for value in choices), device=device)
    if torch.any(choice_tensor <= 0):
        raise ValueError("All budget choices must be positive")
    indices = torch.randint(
        0,
        len(choices),
        (batch_size,),
        device=device,
        generator=generator,
    )
    return choice_tensor[indices]
