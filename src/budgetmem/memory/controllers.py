"""Write and future-utility eviction controllers for BudgetMem-R."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class WriteController(nn.Module):
    """Estimate whether the current hidden state should be written to memory."""

    FEATURE_COUNT = 6

    def __init__(
        self,
        *,
        hidden_dim: int,
        budget_embedding_dim: int,
        controller_dim: int = 64,
    ) -> None:
        super().__init__()
        input_dim = hidden_dim + budget_embedding_dim + self.FEATURE_COUNT
        self.network = nn.Sequential(
            nn.Linear(input_dim, controller_dim),
            nn.SiLU(),
            nn.Linear(controller_dim, controller_dim),
            nn.SiLU(),
            nn.Linear(controller_dim, 1),
        )

    def forward(
        self,
        hidden: Tensor,
        *,
        novelty: Tensor,
        surprise: Tensor,
        uncertainty: Tensor,
        occupancy: Tensor,
        time_since_write: Tensor,
        retrieved_agreement: Tensor,
        budget_embedding: Tensor,
    ) -> Tensor:
        scalars = torch.stack(
            (
                novelty,
                surprise,
                uncertainty,
                occupancy,
                time_since_write,
                retrieved_agreement,
            ),
            dim=-1,
        )
        features = torch.cat((hidden, scalars, budget_embedding), dim=-1)
        return torch.sigmoid(self.network(features).squeeze(-1))

    @staticmethod
    def differentiable_gate(
        probability: Tensor,
        *,
        training: bool,
        threshold: float,
        temperature: float,
    ) -> Tensor:
        # Boundary values are valid controlled modes:
        # 0.0 means always write; 1.0 means write only at probability one.
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within [0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        if threshold <= 0.0:
            hard = torch.ones_like(probability)
            return (
                hard + probability - probability.detach()
                if training
                else hard
            )

        if threshold >= 1.0:
            hard = (probability >= 1.0).to(probability.dtype)
            return (
                hard + probability - probability.detach()
                if training
                else hard
            )

        if not training:
            return (probability >= threshold).to(probability.dtype)

        eps = torch.finfo(probability.dtype).eps
        clipped = probability.clamp(min=eps, max=1.0 - eps)
        logistic = torch.log(clipped) - torch.log1p(-clipped)
        uniform = torch.rand_like(clipped).clamp(
            min=eps,
            max=1.0 - eps,
        )
        noise = torch.log(uniform) - torch.log1p(-uniform)
        relaxed = torch.sigmoid((logistic + noise) / temperature)
        hard = (relaxed >= threshold).to(relaxed.dtype)
        return hard + relaxed - relaxed.detach()



class EvictionController(nn.Module):
    """Predict each resident slot's future utility and select the minimum."""

    METADATA_COUNT = 3

    def __init__(
        self,
        *,
        value_dim: int,
        hidden_dim: int,
        budget_embedding_dim: int,
        controller_dim: int = 64,
    ) -> None:
        super().__init__()
        input_dim = (
            value_dim
            + hidden_dim
            + budget_embedding_dim
            + self.METADATA_COUNT
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, controller_dim),
            nn.SiLU(),
            nn.Linear(controller_dim, controller_dim),
            nn.SiLU(),
            nn.Linear(controller_dim, 1),
        )

    def future_utility(
        self,
        values: Tensor,
        *,
        stored_utility: Tensor,
        age: Tensor,
        retrieval_count: Tensor,
        hidden: Tensor,
        budget_embedding: Tensor,
    ) -> Tensor:
        slot_count = values.shape[1]
        hidden_expanded = hidden.unsqueeze(1).expand(-1, slot_count, -1)
        budget_expanded = budget_embedding.unsqueeze(1).expand(-1, slot_count, -1)
        age_scaled = age.to(values.dtype) / (age.amax(dim=1, keepdim=True) + 1).to(
            values.dtype
        )
        retrieval_scaled = retrieval_count / (
            retrieval_count.amax(dim=1, keepdim=True) + 1.0
        )
        metadata = torch.stack(
            (stored_utility, age_scaled, retrieval_scaled), dim=-1
        )
        features = torch.cat(
            (values, hidden_expanded, metadata, budget_expanded), dim=-1
        )
        return self.network(features).squeeze(-1)
