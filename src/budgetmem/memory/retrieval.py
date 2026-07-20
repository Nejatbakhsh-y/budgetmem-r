"""Top-k content retrieval and recurrent-memory fusion."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn


@dataclass
class RetrievalResult:
    retrieved: Tensor
    indices: Tensor
    weights: Tensor


class TopKRetriever(nn.Module):
    """Retrieve a weighted top-k value mixture from valid memory slots."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        key_dim: int,
        value_dim: int,
        budget_embedding_dim: int = 0,
        top_k: int = 4,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.query_projection = nn.Linear(
            hidden_dim + budget_embedding_dim, key_dim, bias=False
        )
        self.budget_embedding_dim = int(budget_embedding_dim)
        self.value_dim = int(value_dim)
        self.top_k = int(top_k)
        self.scale = sqrt(float(key_dim))

    def forward(
        self,
        hidden: Tensor,
        keys: Tensor,
        values: Tensor,
        valid: Tensor,
        budget_embedding: Tensor | None = None,
    ) -> RetrievalResult:
        batch_size = hidden.shape[0]
        if self.budget_embedding_dim:
            if budget_embedding is None:
                raise ValueError("budget_embedding is required for budget-conditioned retrieval")
            query_input = torch.cat((hidden, budget_embedding), dim=-1)
        else:
            query_input = hidden
        query = self.query_projection(query_input)
        retrieved_rows: list[Tensor] = []
        index_rows: list[Tensor] = []
        weight_rows: list[Tensor] = []

        for batch_index in range(batch_size):
            valid_indices = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                retrieved_rows.append(values.new_zeros(self.value_dim))
                index_rows.append(
                    torch.full(
                        (self.top_k,),
                        -1,
                        device=hidden.device,
                        dtype=torch.long,
                    )
                )
                weight_rows.append(values.new_zeros(self.top_k))
                continue

            slot_keys = keys[batch_index, valid_indices]
            scores = torch.mv(slot_keys, query[batch_index]) / self.scale
            selected_count = min(self.top_k, int(valid_indices.numel()))
            top_scores, relative_indices = torch.topk(scores, selected_count)
            selected_indices = valid_indices[relative_indices]
            selected_weights = torch.softmax(top_scores, dim=0)
            selected_values = values[batch_index, selected_indices]
            retrieved_rows.append(
                torch.sum(selected_weights.unsqueeze(-1) * selected_values, dim=0)
            )

            padded_indices = torch.full(
                (self.top_k,), -1, device=hidden.device, dtype=torch.long
            )
            padded_weights = values.new_zeros(self.top_k)
            padded_indices[:selected_count] = selected_indices
            padded_weights[:selected_count] = selected_weights
            index_rows.append(padded_indices)
            weight_rows.append(padded_weights)

        return RetrievalResult(
            retrieved=torch.stack(retrieved_rows, dim=0),
            indices=torch.stack(index_rows, dim=0),
            weights=torch.stack(weight_rows, dim=0),
        )


class MemoryFusion(nn.Module):
    """Fuse recurrent and retrieved states using a controlled comparison mode."""

    MODES = {"concatenation", "residual", "gated", "attention"}

    def __init__(self, *, hidden_dim: int, value_dim: int, mode: str = "gated") -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported fusion mode {mode!r}; expected one of {sorted(self.MODES)}")
        self.mode = mode
        self.value_projection = nn.Linear(value_dim, hidden_dim)
        self.concat_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden: Tensor, retrieved: Tensor) -> Tensor:
        projected = self.value_projection(retrieved)
        if self.mode == "concatenation":
            return torch.tanh(self.concat_projection(torch.cat((hidden, projected), dim=-1)))
        if self.mode == "residual":
            return hidden + projected
        if self.mode == "gated":
            gate = torch.sigmoid(self.gate_projection(torch.cat((hidden, projected), dim=-1)))
            return gate * hidden + (1.0 - gate) * projected

        candidates = torch.stack((hidden, projected), dim=1)
        weights = torch.softmax(self.attention_score(torch.tanh(candidates)), dim=1)
        return torch.sum(weights * candidates, dim=1)
