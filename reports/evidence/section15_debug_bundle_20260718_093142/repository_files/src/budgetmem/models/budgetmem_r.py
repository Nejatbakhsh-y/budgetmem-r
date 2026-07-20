"""Budget-conditioned recurrent model with strict external-memory control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from budgetmem.memory.budget_state import ExternalMemoryState

BackboneName = Literal["rnn", "gru", "lstm"]
FusionName = Literal["concatenation", "residual", "gated", "attention"]


@dataclass
class BudgetMemROutput:
    """Model outputs and diagnostics required by the benchmark."""

    logits: Tensor
    sequence_logits: Tensor
    hidden_states: Tensor
    write_probabilities: Tensor
    hard_writes: Tensor
    write_slots: Tensor
    eviction_flags: Tensor
    retrieval_weights: Tensor
    memory_masks: Tensor
    memory_sizes: Tensor
    budgets: Tensor
    auxiliary_mean: Tensor
    auxiliary_log_variance: Tensor
    inputs: Tensor
    final_memory: ExternalMemoryState


class BudgetConditioner(nn.Module):
    """Encode a normalized requested budget into a dense representation."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
        )

    def forward(self, normalized_budget: Tensor) -> Tensor:
        return self.network(normalized_budget.unsqueeze(-1))


class WriteController(nn.Module):
    """Estimate whether the current state should be written to memory."""

    def __init__(self, hidden_dim: int, budget_embedding_dim: int) -> None:
        super().__init__()
        feature_dim = hidden_dim + budget_embedding_dim + 6
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        hidden: Tensor,
        novelty: Tensor,
        surprise: Tensor,
        uncertainty: Tensor,
        occupancy: Tensor,
        recency: Tensor,
        agreement: Tensor,
        budget_embedding: Tensor,
    ) -> Tensor:
        features = torch.cat(
            [
                hidden,
                novelty.unsqueeze(-1),
                surprise.unsqueeze(-1),
                uncertainty.unsqueeze(-1),
                occupancy.unsqueeze(-1),
                recency.unsqueeze(-1),
                agreement.unsqueeze(-1),
                budget_embedding,
            ],
            dim=-1,
        )
        return self.network(features).squeeze(-1)


class UtilityController(nn.Module):
    """Predict future utility for every occupied memory slot."""

    def __init__(
        self,
        *,
        key_dim: int,
        value_dim: int,
        hidden_dim: int,
        budget_embedding_dim: int,
    ) -> None:
        super().__init__()
        feature_dim = key_dim + value_dim + hidden_dim + budget_embedding_dim + 2
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        memory: ExternalMemoryState,
        hidden: Tensor,
        budget_embedding: Tensor,
        sequence_length: int,
    ) -> Tensor:
        slots = memory.max_budget
        hidden_expanded = hidden.unsqueeze(1).expand(-1, slots, -1)
        budget_expanded = budget_embedding.unsqueeze(1).expand(-1, slots, -1)
        age = memory.age / max(float(sequence_length), 1.0)
        retrieval_frequency = memory.retrieval_count / max(float(sequence_length), 1.0)
        features = torch.cat(
            [
                memory.keys,
                memory.values,
                hidden_expanded,
                budget_expanded,
                age.unsqueeze(-1),
                retrieval_frequency.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.network(features).squeeze(-1)


class MemoryFusion(nn.Module):
    """Fuse retrieved memory with the recurrent state."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        value_dim: int,
        mode: FusionName,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.memory_projection = nn.Linear(value_dim, hidden_dim)
        self.concatenate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_score = nn.Linear(hidden_dim, 1)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: Tensor, retrieved: Tensor) -> Tensor:
        projected = self.memory_projection(retrieved)
        if self.mode == "concatenation":
            return torch.tanh(self.concatenate(torch.cat([hidden, projected], dim=-1)))
        if self.mode == "residual":
            return self.normalization(hidden + projected)
        if self.mode == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([hidden, projected], dim=-1)))
            return self.normalization(gate * projected + (1.0 - gate) * hidden)
        if self.mode == "attention":
            candidates = torch.stack([hidden, projected], dim=1)
            weights = torch.softmax(self.attention_score(candidates).squeeze(-1), dim=1)
            return self.normalization((weights.unsqueeze(-1) * candidates).sum(dim=1))
        raise ValueError(f"unsupported fusion mode: {self.mode}")


class BudgetMemR(nn.Module):
    """BudgetMem-R with write, utility-based eviction, retrieval, and fusion.

    The forward method intentionally accepts no final classification or anomaly
    label. All write-controller signals are computed from the observed input
    sequence, recurrent state, external memory, and auxiliary next-input model.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_budget: int = 128,
        allowed_budgets: Sequence[int] = (8, 16, 32, 64, 128),
        key_dim: int | None = None,
        value_dim: int | None = None,
        budget_embedding_dim: int = 16,
        retrieval_k: int = 4,
        backbone: BackboneName = "gru",
        fusion: FusionName = "gated",
        write_threshold: float = 0.5,
        write_temperature: float = 0.67,
        detach_memory_writes: bool = False,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or output_dim < 1:
            raise ValueError("input_dim, hidden_dim, and output_dim must be positive")
        if max_budget < 1:
            raise ValueError("max_budget must be positive")
        if retrieval_k < 1:
            raise ValueError("retrieval_k must be positive")
        if not 0.0 <= write_threshold <= 1.0:
            raise ValueError("write_threshold must be in [0, 1]")
        if write_temperature <= 0.0:
            raise ValueError("write_temperature must be positive")
        normalized_budgets = tuple(sorted({int(value) for value in allowed_budgets}))
        if not normalized_budgets:
            raise ValueError("allowed_budgets must not be empty")
        if normalized_budgets[0] < 1 or normalized_budgets[-1] > max_budget:
            raise ValueError("allowed_budgets must be within [1, max_budget]")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_budget = max_budget
        self.allowed_budgets = normalized_budgets
        self.key_dim = key_dim or hidden_dim
        self.value_dim = value_dim or hidden_dim
        self.retrieval_k = min(retrieval_k, max_budget)
        self.backbone_name = backbone
        self.write_threshold = write_threshold
        self.write_temperature = write_temperature
        self.detach_memory_writes = detach_memory_writes

        if backbone == "rnn":
            self.recurrent_cell: nn.Module = nn.RNNCell(input_dim, hidden_dim)
        elif backbone == "gru":
            self.recurrent_cell = nn.GRUCell(input_dim, hidden_dim)
        elif backbone == "lstm":
            self.recurrent_cell = nn.LSTMCell(input_dim, hidden_dim)
        else:
            raise ValueError(f"unsupported recurrent backbone: {backbone}")

        self.key_projection = nn.Linear(hidden_dim, self.key_dim)
        self.value_projection = nn.Linear(hidden_dim, self.value_dim)
        self.query_projection = nn.Linear(hidden_dim, self.key_dim)
        self.budget_conditioner = BudgetConditioner(budget_embedding_dim)
        self.write_controller = WriteController(hidden_dim, budget_embedding_dim)
        self.utility_controller = UtilityController(
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            hidden_dim=hidden_dim,
            budget_embedding_dim=budget_embedding_dim,
        )
        self.fusion = MemoryFusion(
            hidden_dim=hidden_dim,
            value_dim=self.value_dim,
            mode=fusion,
        )
        self.output_head = nn.Linear(hidden_dim, output_dim)
        self.auxiliary_head = nn.Linear(hidden_dim, input_dim * 2)

    def sample_budgets(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample one allowed deployment budget per sample."""
        choices = torch.tensor(self.allowed_budgets, device=device, dtype=torch.long)
        indices = torch.randint(
            0, len(self.allowed_budgets), (batch_size,), device=device
        )
        return choices[indices]

    def _resolve_budgets(
        self,
        budget: int | Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if budget is None:
            if self.training:
                budgets = self.sample_budgets(batch_size, device)
            else:
                budgets = torch.full(
                    (batch_size,),
                    fill_value=self.allowed_budgets[-1],
                    device=device,
                    dtype=torch.long,
                )
        elif isinstance(budget, int):
            budgets = torch.full(
                (batch_size,),
                fill_value=budget,
                device=device,
                dtype=torch.long,
            )
        else:
            budgets = budget.to(device=device, dtype=torch.long)
            if budgets.ndim == 0:
                budgets = budgets.repeat(batch_size)
        if budgets.shape != (batch_size,):
            raise ValueError(
                "budget must be an integer, scalar tensor, or [batch] tensor"
            )
        if torch.any(budgets < 1) or torch.any(budgets > self.max_budget):
            raise ValueError("every requested budget must be within [1, max_budget]")
        return budgets

    def _recurrent_step(
        self,
        input_t: Tensor,
        hidden: Tensor,
        cell: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        if self.backbone_name == "lstm":
            assert cell is not None
            next_hidden, next_cell = self.recurrent_cell(input_t, (hidden, cell))
            return next_hidden, next_cell
        next_hidden = self.recurrent_cell(input_t, hidden)
        return next_hidden, cell

    def _retrieve(
        self,
        *,
        hidden: Tensor,
        memory: ExternalMemoryState,
    ) -> tuple[Tensor, Tensor, ExternalMemoryState]:
        query = self.query_projection(hidden)
        scores = torch.einsum("bd,bsd->bs", query, memory.keys)
        scores = scores / math.sqrt(float(self.key_dim))
        scores = scores.masked_fill(~memory.valid, -1.0e9)

        top_scores, top_indices = torch.topk(scores, k=self.retrieval_k, dim=1)
        top_valid = torch.gather(memory.valid, dim=1, index=top_indices)
        stable_scores = torch.where(
            top_valid, top_scores, torch.full_like(top_scores, -1.0e9)
        )
        weights = torch.softmax(stable_scores, dim=1) * top_valid.to(scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)

        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, self.value_dim)
        top_values = torch.gather(memory.values, dim=1, index=gather_index)
        retrieved = (weights.unsqueeze(-1) * top_values).sum(dim=1)

        dense_weights = torch.zeros_like(scores)
        dense_weights = dense_weights.scatter_add(1, top_indices, weights)
        memory.retrieval_count = memory.retrieval_count + dense_weights.detach()
        return retrieved, dense_weights, memory

    def _novelty(self, key: Tensor, memory: ExternalMemoryState) -> Tensor:
        normalized_key = F.normalize(key, p=2, dim=-1, eps=1.0e-8)
        normalized_memory = F.normalize(memory.keys, p=2, dim=-1, eps=1.0e-8)
        similarities = torch.einsum("bd,bsd->bs", normalized_key, normalized_memory)
        similarities = similarities.masked_fill(~memory.valid, -1.0)
        maximum_similarity = similarities.max(dim=1).values
        has_memory = memory.valid.any(dim=1)
        novelty = (1.0 - maximum_similarity).clamp(min=0.0, max=2.0)
        return torch.where(has_memory, novelty, torch.ones_like(novelty))

    def _agreement(self, hidden: Tensor, retrieved: Tensor) -> Tensor:
        projected = self.fusion.memory_projection(retrieved)
        similarity = F.cosine_similarity(hidden, projected, dim=-1, eps=1.0e-8)
        return (similarity + 1.0) * 0.5

    def _write_gate(self, logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Training and evaluation must use the same deterministic hard decision.
        # The relaxed value is retained only as the straight-through gradient path.
        probabilities = torch.sigmoid(logits)
        hard = (probabilities >= self.write_threshold).to(logits.dtype)
        if self.training:
            relaxed = torch.sigmoid(logits / self.write_temperature)
            straight_through = hard.detach() - relaxed.detach() + relaxed
            return probabilities, hard, straight_through
        return probabilities, hard, hard

    def _choose_write_slots(
        self,
        *,
        hard_write: Tensor,
        memory: ExternalMemoryState,
        utility_scores: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = memory.batch_size
        allowed = memory.allowed_slot_mask()
        free = allowed & ~memory.valid
        has_free = free.any(dim=1)
        first_free = free.to(torch.int64).argmax(dim=1)
        free_selection = F.one_hot(
            first_free,
            num_classes=memory.max_budget,
        ).to(utility_scores.dtype)

        eviction_logits = (-utility_scores).masked_fill(~memory.valid, -1.0e9)
        if self.training:
            eviction_selection = F.gumbel_softmax(
                eviction_logits,
                tau=self.write_temperature,
                hard=True,
                dim=1,
            )
        else:
            eviction_slot = utility_scores.masked_fill(
                ~memory.valid,
                torch.inf,
            ).argmin(dim=1)
            eviction_selection = F.one_hot(
                eviction_slot,
                num_classes=memory.max_budget,
            ).to(utility_scores.dtype)

        selection = torch.where(
            has_free.unsqueeze(1),
            free_selection,
            eviction_selection,
        )
        selection = selection * hard_write.unsqueeze(1)
        selected = selection.argmax(dim=1)
        no_write_value = torch.full_like(selected, -1)
        selected = torch.where(
            hard_write.to(torch.bool),
            selected,
            no_write_value,
        )
        evicted = hard_write.to(torch.bool) & ~has_free
        if selected.shape != (batch_size,):
            raise RuntimeError("internal write-slot selection failure")
        return selected, evicted, selection

    def _apply_writes(
        self,
        *,
        memory: ExternalMemoryState,
        candidate_key: Tensor,
        candidate_value: Tensor,
        write_slots: Tensor,
        selection_weights: Tensor,
        hard_write: Tensor,
        straight_through_gate: Tensor,
        step: int,
    ) -> ExternalMemoryState:
        active = write_slots >= 0
        safe_slots = write_slots.clamp_min(0)
        hard_selected_weights = F.one_hot(
            safe_slots,
            num_classes=memory.max_budget,
        ).to(dtype=memory.keys.dtype)
        hard_selected_weights = hard_selected_weights * active.unsqueeze(1).to(
            hard_selected_weights.dtype
        )
        gate = selection_weights * straight_through_gate.unsqueeze(1)
        gate_3d = gate.unsqueeze(-1)

        source_key = (
            candidate_key.detach() if self.detach_memory_writes else candidate_key
        )
        source_value = (
            candidate_value.detach() if self.detach_memory_writes else candidate_value
        )
        memory.keys = memory.keys * (1.0 - gate_3d) + source_key.unsqueeze(1) * gate_3d
        memory.values = (
            memory.values * (1.0 - gate_3d) + source_value.unsqueeze(1) * gate_3d
        )

        hard_selected = hard_selected_weights.to(torch.bool) & hard_write.to(
            torch.bool
        ).unsqueeze(1)
        memory.valid = memory.valid | hard_selected
        memory.utility = memory.utility * (1.0 - hard_selected_weights)
        memory.age = memory.age * (1.0 - hard_selected_weights)
        memory.retrieval_count = memory.retrieval_count * (1.0 - hard_selected_weights)
        memory.last_write_step = torch.where(
            hard_write.to(torch.bool),
            torch.full_like(memory.last_write_step, step),
            memory.last_write_step,
        )
        memory.assert_invariant()
        return memory

    def forward(
        self,
        inputs: Tensor,
        budget: int | Tensor | None = None,
    ) -> BudgetMemROutput:
        """Process ``inputs`` without exposing final task labels to memory control."""
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, sequence, input_dim]")
        if inputs.shape[-1] != self.input_dim:
            raise ValueError("the last input dimension does not match input_dim")

        batch_size, sequence_length, _ = inputs.shape
        if sequence_length < 1:
            raise ValueError("sequence length must be positive")
        budgets = self._resolve_budgets(
            budget,
            batch_size=batch_size,
            device=inputs.device,
        )
        normalized_budget = budgets.to(inputs.dtype) / float(self.max_budget)
        budget_embedding = self.budget_conditioner(normalized_budget)

        memory = ExternalMemoryState.empty(
            batch_size=batch_size,
            max_budget=self.max_budget,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            budgets=budgets,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        hidden = torch.zeros(
            batch_size, self.hidden_dim, device=inputs.device, dtype=inputs.dtype
        )
        cell = torch.zeros_like(hidden) if self.backbone_name == "lstm" else None
        previous_aux_mean: Tensor | None = None
        previous_aux_log_variance: Tensor | None = None

        hidden_history: list[Tensor] = []
        logits_history: list[Tensor] = []
        write_probability_history: list[Tensor] = []
        hard_write_history: list[Tensor] = []
        write_slot_history: list[Tensor] = []
        eviction_history: list[Tensor] = []
        retrieval_weight_history: list[Tensor] = []
        memory_mask_history: list[Tensor] = []
        memory_size_history: list[Tensor] = []
        auxiliary_mean_history: list[Tensor] = []
        auxiliary_log_variance_history: list[Tensor] = []

        for step in range(sequence_length):
            input_t = inputs[:, step]
            memory.increment_age()
            hidden, cell = self._recurrent_step(input_t, hidden, cell)

            retrieved, retrieval_weights, memory = self._retrieve(
                hidden=hidden,
                memory=memory,
            )
            fused_hidden = self.fusion(hidden, retrieved)
            key = self.key_projection(hidden)
            value = self.value_projection(hidden)

            if previous_aux_mean is None or previous_aux_log_variance is None:
                surprise = torch.zeros(
                    batch_size, device=inputs.device, dtype=inputs.dtype
                )
                uncertainty = torch.zeros_like(surprise)
            else:
                inverse_variance = torch.exp(-previous_aux_log_variance)
                squared_error = (input_t - previous_aux_mean).pow(2)
                surprise = 0.5 * (
                    inverse_variance * squared_error + previous_aux_log_variance
                ).mean(dim=-1)
                uncertainty = torch.exp(previous_aux_log_variance).mean(dim=-1)

            novelty = self._novelty(key, memory)
            occupancy = memory.sizes().to(inputs.dtype) / budgets.to(inputs.dtype)
            recency = (step - memory.last_write_step.clamp_min(0)).to(
                inputs.dtype
            ) / float(max(sequence_length, 1))
            agreement = self._agreement(hidden, retrieved)

            write_logits = self.write_controller(
                hidden=hidden,
                novelty=novelty,
                surprise=surprise,
                uncertainty=uncertainty,
                occupancy=occupancy,
                recency=recency,
                agreement=agreement,
                budget_embedding=budget_embedding,
            )
            write_probability, hard_write, straight_through_gate = self._write_gate(
                write_logits
            )
            utility_scores = self.utility_controller(
                memory=memory,
                hidden=hidden,
                budget_embedding=budget_embedding,
                sequence_length=sequence_length,
            )
            memory.utility = utility_scores
            write_slots, evicted, selection_weights = self._choose_write_slots(
                hard_write=hard_write,
                memory=memory,
                utility_scores=utility_scores,
            )
            memory = self._apply_writes(
                memory=memory,
                candidate_key=key,
                candidate_value=value,
                write_slots=write_slots,
                selection_weights=selection_weights,
                hard_write=hard_write,
                straight_through_gate=straight_through_gate,
                step=step,
            )

            auxiliary_parameters = self.auxiliary_head(fused_hidden)
            auxiliary_mean, auxiliary_log_variance = auxiliary_parameters.chunk(
                2, dim=-1
            )
            auxiliary_log_variance = auxiliary_log_variance.clamp(min=-8.0, max=8.0)
            previous_aux_mean = auxiliary_mean
            previous_aux_log_variance = auxiliary_log_variance

            step_logits = self.output_head(fused_hidden)
            hidden = fused_hidden

            hidden_history.append(fused_hidden)
            logits_history.append(step_logits)
            write_probability_history.append(write_probability)
            hard_write_history.append(hard_write)
            write_slot_history.append(write_slots)
            eviction_history.append(evicted)
            retrieval_weight_history.append(retrieval_weights)
            memory_mask_history.append(memory.valid.clone())
            memory_size_history.append(memory.sizes())
            auxiliary_mean_history.append(auxiliary_mean)
            auxiliary_log_variance_history.append(auxiliary_log_variance)

        sequence_logits = torch.stack(logits_history, dim=1)
        memory_sizes = torch.stack(memory_size_history, dim=1)
        if torch.any(memory_sizes > budgets.unsqueeze(1)):
            raise RuntimeError("strict memory-budget invariant violated during forward")

        return BudgetMemROutput(
            logits=sequence_logits[:, -1],
            sequence_logits=sequence_logits,
            hidden_states=torch.stack(hidden_history, dim=1),
            write_probabilities=torch.stack(write_probability_history, dim=1),
            hard_writes=torch.stack(hard_write_history, dim=1),
            write_slots=torch.stack(write_slot_history, dim=1),
            eviction_flags=torch.stack(eviction_history, dim=1),
            retrieval_weights=torch.stack(retrieval_weight_history, dim=1),
            memory_masks=torch.stack(memory_mask_history, dim=1),
            memory_sizes=memory_sizes,
            budgets=budgets,
            auxiliary_mean=torch.stack(auxiliary_mean_history, dim=1),
            auxiliary_log_variance=torch.stack(
                auxiliary_log_variance_history,
                dim=1,
            ),
            inputs=inputs,
            final_memory=memory,
        )
