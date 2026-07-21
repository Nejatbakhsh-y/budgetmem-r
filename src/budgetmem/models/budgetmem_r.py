"""Budget-conditioned recurrent model with strict external-memory control."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from budgetmem.memory.budget import (
    DEFAULT_TRAINING_BUDGETS,
    BudgetConditioner,
    sample_training_budgets,
)
from budgetmem.memory.controllers import EvictionController, WriteController
from budgetmem.memory.retrieval import MemoryFusion, RetrievalResult, TopKRetriever
from budgetmem.memory.state import BudgetMemoryState


class BudgetMemROutput(dict[str, object]):
    """Dictionary-compatible output with attribute access for project adapters."""

    def __init__(self, **values: object) -> None:
        super().__init__(values)

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class BudgetMemR(nn.Module):
    """BudgetMem-R with causal write, retrieval, eviction, and fusion controls."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        key_dim: int | None = None,
        value_dim: int | None = None,
        budget_embedding_dim: int = 16,
        controller_dim: int = 64,
        max_budget: int = 128,
        training_budgets: Sequence[int] = DEFAULT_TRAINING_BUDGETS,
        top_k: int = 4,
        fusion_mode: str = "gated",
        write_threshold: float = 0.5,
        write_temperature: float = 0.67,
        allowed_budgets: Sequence[int] | None = None,
        retrieval_k: int | None = None,
        backbone: str = "gru",
        fusion: str | None = None,
        detach_memory_writes: bool = False,
    ) -> None:
        super().__init__()

        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim, hidden_dim, and output_dim must be positive")
        if max_budget <= 0:
            raise ValueError("max_budget must be positive")
        if not 0.0 <= write_threshold <= 1.0:
            raise ValueError("write_threshold must be within [0, 1]")
        if write_temperature <= 0.0:
            raise ValueError("write_temperature must be positive")

        selected_budgets = allowed_budgets if allowed_budgets is not None else training_budgets
        normalized_budgets = tuple(sorted({int(value) for value in selected_budgets}))
        if not normalized_budgets:
            raise ValueError("allowed_budgets must not be empty")
        if normalized_budgets[0] < 1 or normalized_budgets[-1] > max_budget:
            raise ValueError("allowed_budgets must be within [1, max_budget]")

        selected_top_k = int(retrieval_k if retrieval_k is not None else top_k)
        if selected_top_k <= 0:
            raise ValueError("retrieval_k must be positive")

        selected_fusion = fusion if fusion is not None else fusion_mode
        if backbone not in {"rnn", "gru", "lstm"}:
            raise ValueError("backbone must be one of: rnn, gru, lstm")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.key_dim = int(key_dim or hidden_dim)
        self.value_dim = int(value_dim or hidden_dim)
        self.max_budget = int(max_budget)
        self.training_budgets = normalized_budgets
        self.allowed_budgets = normalized_budgets
        self.top_k = min(selected_top_k, self.max_budget)
        self.retrieval_k = self.top_k
        self.backbone_name = backbone
        self.write_threshold = float(write_threshold)
        self.write_temperature = float(write_temperature)
        self.detach_memory_writes = bool(detach_memory_writes)

        if backbone == "rnn":
            self.recurrent_cell: nn.Module = nn.RNNCell(input_dim, hidden_dim)
        elif backbone == "gru":
            self.recurrent_cell = nn.GRUCell(input_dim, hidden_dim)
        else:
            self.recurrent_cell = nn.LSTMCell(input_dim, hidden_dim)

        self.key_projection = nn.Linear(hidden_dim, self.key_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, self.value_dim, bias=False)
        self.budget_conditioner = BudgetConditioner(
            max_budget=max_budget,
            embedding_dim=budget_embedding_dim,
        )
        self.retriever = TopKRetriever(
            hidden_dim=hidden_dim,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            budget_embedding_dim=budget_embedding_dim,
            top_k=self.top_k,
        )
        self.fusion = MemoryFusion(
            hidden_dim=hidden_dim,
            value_dim=self.value_dim,
            mode=selected_fusion,
        )
        self.write_controller = WriteController(
            hidden_dim=hidden_dim,
            budget_embedding_dim=budget_embedding_dim,
            controller_dim=controller_dim,
        )
        self.eviction_controller = EvictionController(
            value_dim=self.value_dim,
            hidden_dim=hidden_dim,
            budget_embedding_dim=budget_embedding_dim,
            controller_dim=controller_dim,
        )
        self.task_head = nn.Linear(hidden_dim, output_dim)
        self.auxiliary_next_input_head = nn.Linear(hidden_dim, input_dim * 2)
        self.initial_utility_head = nn.Sequential(
            nn.Linear(hidden_dim + budget_embedding_dim, controller_dim),
            nn.SiLU(),
            nn.Linear(controller_dim, 1),
            nn.Sigmoid(),
        )

    def sample_budgets(
        self,
        batch_size: int,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample one controlled deployment budget per sample."""

        resolved_device = device or next(self.parameters()).device
        return sample_training_budgets(
            batch_size,
            device=resolved_device,
            choices=self.allowed_budgets,
            generator=generator,
        )

    def _resolve_budgets(
        self,
        inputs: Tensor,
        budget: int | Tensor | None,
    ) -> Tensor:
        batch_size = inputs.shape[0]
        device = inputs.device

        if budget is None:
            if self.training:
                budgets = self.sample_budgets(batch_size, device=device)
            else:
                budgets = torch.full(
                    (batch_size,),
                    self.allowed_budgets[-1],
                    device=device,
                    dtype=torch.long,
                )
        elif isinstance(budget, int):
            budgets = torch.full(
                (batch_size,),
                budget,
                device=device,
                dtype=torch.long,
            )
        else:
            budgets = budget.to(device=device, dtype=torch.long)
            if budgets.ndim == 0:
                budgets = budgets.expand(batch_size)
            if budgets.shape != (batch_size,):
                raise ValueError("budget tensor must contain one value per batch item")

        if torch.any(budgets < 1) or torch.any(budgets > self.max_budget):
            raise ValueError(f"Budgets must be within [1, {self.max_budget}]")

        allowed = torch.tensor(
            self.allowed_budgets,
            device=device,
            dtype=torch.long,
        )
        membership = (budgets.unsqueeze(1) == allowed.unsqueeze(0)).any(dim=1)
        if not bool(torch.all(membership)):
            invalid = torch.unique(budgets[~membership]).detach().cpu().tolist()
            requested = invalid[0] if len(invalid) == 1 else invalid
            raise ValueError(
                f"requested_budget={requested} is not in "
                f"allowed_budgets={self.allowed_budgets}"
            )

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

    @staticmethod
    def _prediction_uncertainty(logits: Tensor) -> Tensor:
        if logits.shape[-1] == 1:
            probability = torch.sigmoid(logits.squeeze(-1)).clamp(1e-7, 1.0 - 1e-7)
            return -(
                probability * torch.log(probability)
                + (1.0 - probability) * torch.log1p(-probability)
            )
        probabilities = torch.softmax(logits, dim=-1).clamp_min(1e-7)
        entropy = -(probabilities * torch.log(probabilities)).sum(dim=-1)
        normalizer = torch.log(
            torch.tensor(float(logits.shape[-1]), device=logits.device)
        )
        return entropy / normalizer.clamp_min(1.0)

    @staticmethod
    def _novelty(candidate_key: Tensor, state: BudgetMemoryState) -> Tensor:
        normalized_candidate = functional.normalize(candidate_key, dim=-1)
        normalized_keys = functional.normalize(state.keys, dim=-1)
        similarities = torch.einsum(
            "bd,bmd->bm",
            normalized_candidate,
            normalized_keys,
        )
        similarities = similarities.masked_fill(~state.valid, -1.0)
        maximum = similarities.max(dim=1).values
        has_memory = state.valid.any(dim=1)
        novelty = 1.0 - maximum
        return torch.where(
            has_memory,
            novelty.clamp(0.0, 2.0) / 2.0,
            torch.ones_like(novelty),
        )

    @staticmethod
    def _retrieved_agreement(
        candidate_value: Tensor,
        retrieved: Tensor,
        state: BudgetMemoryState,
    ) -> Tensor:
        agreement = functional.cosine_similarity(
            candidate_value,
            retrieved,
            dim=-1,
        )
        return torch.where(
            state.valid.any(dim=1),
            (agreement + 1.0) / 2.0,
            torch.zeros_like(agreement),
        )

    @staticmethod
    def _record_retrievals(
        state: BudgetMemoryState,
        retrieval: RetrievalResult,
    ) -> BudgetMemoryState:
        updated_counts = state.retrieval_count.clone()
        updated_utility = state.utility.clone()

        for batch_index in range(state.batch_size):
            for rank in range(retrieval.indices.shape[1]):
                slot_index = int(retrieval.indices[batch_index, rank].item())
                if slot_index < 0:
                    continue
                weight = retrieval.weights[batch_index, rank]
                updated_counts[batch_index, slot_index] = (
                    updated_counts[batch_index, slot_index] + weight.detach()
                )
                updated_utility[batch_index, slot_index] = (
                    0.95 * updated_utility[batch_index, slot_index]
                    + 0.05 * weight.detach()
                )

        return BudgetMemoryState(
            keys=state.keys,
            values=state.values,
            utility=updated_utility,
            age=state.age,
            retrieval_count=updated_counts,
            valid=state.valid,
            budgets=state.budgets,
            last_write_step=state.last_write_step,
        )

    def _write_gate(self, logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Calculate probabilities and a deterministic hard-write decision."""
        probabilities = torch.sigmoid(logits)
        hard = (probabilities >= self.write_threshold).to(logits.dtype)

        if self.training:
            relaxed = torch.sigmoid(logits / self.write_temperature)
            gate = hard.detach() - relaxed.detach() + relaxed
        else:
            gate = hard

        return probabilities, hard, gate

    def _choose_write_slots(
        self,
        *,
        state: BudgetMemoryState,
        hard_write: Tensor,
        future_utility: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        capacity = state.capacity
        positions = torch.arange(
            capacity,
            device=state.keys.device,
        ).unsqueeze(0)
        allowed = positions < state.budgets.unsqueeze(1)
        free = allowed & ~state.valid
        has_free = free.any(dim=1)

        first_free = free.to(torch.int64).argmax(dim=1)
        free_selection = functional.one_hot(
            first_free,
            num_classes=capacity,
        ).to(future_utility.dtype)

        eviction_logits = (-future_utility).masked_fill(
            ~state.valid,
            -1.0e9,
        )
        if self.training:
            eviction_selection = functional.gumbel_softmax(
                eviction_logits,
                tau=self.write_temperature,
                hard=True,
                dim=1,
            )
        else:
            eviction_slot = future_utility.masked_fill(
                ~state.valid,
                torch.inf,
            ).argmin(dim=1)
            eviction_selection = functional.one_hot(
                eviction_slot,
                num_classes=capacity,
            ).to(future_utility.dtype)

        selection = torch.where(
            has_free.unsqueeze(1),
            free_selection,
            eviction_selection,
        )
        selection = selection * hard_write.unsqueeze(1)

        selected = selection.argmax(dim=1)
        selected = torch.where(
            hard_write.to(torch.bool),
            selected,
            torch.full_like(selected, -1),
        )
        eviction_flags = hard_write.to(torch.bool) & ~has_free
        return selected, eviction_flags, selection

    def _apply_writes(
        self,
        state: BudgetMemoryState,
        *,
        candidate_key: Tensor,
        candidate_value: Tensor,
        candidate_utility: Tensor,
        write_slots: Tensor,
        selection_weights: Tensor,
        hard_write: Tensor,
        straight_through_gate: Tensor,
        step: int,
    ) -> BudgetMemoryState:
        active = write_slots >= 0
        safe_slots = write_slots.clamp_min(0)
        hard_selected_weights = functional.one_hot(
            safe_slots,
            num_classes=state.capacity,
        ).to(state.keys.dtype)
        hard_selected_weights = hard_selected_weights * active.unsqueeze(1).to(
            hard_selected_weights.dtype
        )

        gate = selection_weights * straight_through_gate.unsqueeze(1)
        gate_3d = gate.unsqueeze(-1)

        source_key = (
            candidate_key.detach()
            if self.detach_memory_writes
            else candidate_key
        )
        source_value = (
            candidate_value.detach()
            if self.detach_memory_writes
            else candidate_value
        )

        keys = (
            state.keys * (1.0 - gate_3d)
            + source_key.unsqueeze(1) * gate_3d
        )
        values = (
            state.values * (1.0 - gate_3d)
            + source_value.unsqueeze(1) * gate_3d
        )

        hard_selected = hard_selected_weights.to(torch.bool)
        valid = state.valid | hard_selected
        utility = (
            state.utility * (1.0 - hard_selected_weights)
            + candidate_utility.unsqueeze(1) * hard_selected_weights
        )
        age = torch.where(
            hard_selected,
            torch.zeros_like(state.age),
            state.age,
        )
        retrieval_count = torch.where(
            hard_selected,
            torch.zeros_like(state.retrieval_count),
            state.retrieval_count,
        )
        last_write_step = torch.where(
            hard_write.to(torch.bool),
            torch.full_like(state.last_write_step, step),
            state.last_write_step,
        )

        updated = BudgetMemoryState(
            keys=keys,
            values=values,
            utility=utility,
            age=age,
            retrieval_count=retrieval_count,
            valid=valid,
            budgets=state.budgets,
            last_write_step=last_write_step,
        )
        updated.assert_within_budget()
        return updated

    def forward(
        self,
        inputs: Tensor,
        budget: int | Tensor | None = None,
    ) -> BudgetMemROutput:
        """Process inputs without exposing final labels to memory control."""

        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, sequence, input_dim]")
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, received {inputs.shape[-1]}"
            )

        batch_size, sequence_length, _ = inputs.shape
        if sequence_length < 1:
            raise ValueError("sequence length must be positive")

        budgets = self._resolve_budgets(inputs, budget)
        budget_embedding = self.budget_conditioner(budgets)

        state = BudgetMemoryState.empty(
            batch_size=batch_size,
            capacity=int(budgets.max().item()),
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            budgets=budgets,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        hidden = inputs.new_zeros(batch_size, self.hidden_dim)
        cell = (
            inputs.new_zeros(batch_size, self.hidden_dim)
            if self.backbone_name == "lstm"
            else None
        )
        previous_auxiliary_mean: Tensor | None = None
        previous_auxiliary_log_variance: Tensor | None = None

        sequence_logits: list[Tensor] = []
        hidden_states: list[Tensor] = []
        auxiliary_means: list[Tensor] = []
        auxiliary_log_variances: list[Tensor] = []
        write_probabilities: list[Tensor] = []
        write_gates: list[Tensor] = []
        hard_writes: list[Tensor] = []
        write_slots: list[Tensor] = []
        eviction_flags: list[Tensor] = []
        memory_sizes: list[Tensor] = []
        memory_masks: list[Tensor] = []
        retrieval_weights: list[Tensor] = []

        for step in range(sequence_length):
            current_input = inputs[:, step]
            aged = torch.where(
                state.valid,
                state.age + 1,
                state.age,
            )
            state = BudgetMemoryState(
                keys=state.keys,
                values=state.values,
                utility=state.utility,
                age=aged,
                retrieval_count=state.retrieval_count,
                valid=state.valid,
                budgets=state.budgets,
                last_write_step=state.last_write_step,
            )

            hidden, cell = self._recurrent_step(
                current_input,
                hidden,
                cell,
            )
            retrieval = self.retriever(
                hidden,
                state.keys,
                state.values,
                state.valid,
                budget_embedding,
            )
            state = self._record_retrievals(state, retrieval)
            fused_hidden = self.fusion(hidden, retrieval.retrieved)

            candidate_key = self.key_projection(hidden)
            candidate_value = self.value_projection(hidden)
            novelty = self._novelty(candidate_key, state)

            if (
                previous_auxiliary_mean is None
                or previous_auxiliary_log_variance is None
            ):
                surprise = inputs.new_zeros(batch_size)
                uncertainty = inputs.new_zeros(batch_size)
            else:
                inverse_variance = torch.exp(
                    -previous_auxiliary_log_variance
                )
                squared_error = (
                    current_input - previous_auxiliary_mean
                ).pow(2)
                surprise = 0.5 * (
                    inverse_variance * squared_error
                    + previous_auxiliary_log_variance
                ).mean(dim=-1)
                surprise = surprise / (1.0 + surprise)
                uncertainty = torch.exp(
                    previous_auxiliary_log_variance
                ).mean(dim=-1)
                uncertainty = uncertainty / (1.0 + uncertainty)

            provisional_logits = self.task_head(fused_hidden)
            prediction_uncertainty = self._prediction_uncertainty(
                provisional_logits
            )
            uncertainty = 0.5 * (
                uncertainty + prediction_uncertainty
            )
            occupancy = (
                state.sizes().to(inputs.dtype)
                / budgets.to(inputs.dtype)
            )
            time_since_write = torch.where(
                state.last_write_step < 0,
                torch.ones_like(
                    state.last_write_step,
                    dtype=inputs.dtype,
                ),
                (step - state.last_write_step).to(inputs.dtype)
                / float(step + 1),
            )
            agreement = self._retrieved_agreement(
                candidate_value,
                retrieval.retrieved,
                state,
            )

            write_probability = self.write_controller(
                hidden,
                novelty=novelty,
                surprise=surprise,
                uncertainty=uncertainty,
                occupancy=occupancy,
                time_since_write=time_since_write,
                retrieved_agreement=agreement,
                budget_embedding=budget_embedding,
            )
            epsilon = torch.finfo(write_probability.dtype).eps
            write_logits = torch.logit(
                write_probability.clamp(
                    min=epsilon,
                    max=1.0 - epsilon,
                )
            )
            write_probability, hard_write_value, write_gate = self._write_gate(
                write_logits
            )
            hard_write = hard_write_value.to(torch.bool)

            future_utility = self.eviction_controller.future_utility(
                state.values,
                stored_utility=state.utility,
                age=state.age,
                retrieval_count=state.retrieval_count,
                hidden=hidden,
                budget_embedding=budget_embedding,
            )
            candidate_utility = self.initial_utility_head(
                torch.cat(
                    (fused_hidden, budget_embedding),
                    dim=-1,
                )
            ).squeeze(-1)

            selected_slots, evicted, selection_weights = (
                self._choose_write_slots(
                    state=state,
                    hard_write=hard_write.to(inputs.dtype),
                    future_utility=future_utility,
                )
            )
            state = self._apply_writes(
                state,
                candidate_key=candidate_key,
                candidate_value=candidate_value,
                candidate_utility=candidate_utility,
                write_slots=selected_slots,
                selection_weights=selection_weights,
                hard_write=hard_write.to(inputs.dtype),
                straight_through_gate=write_gate,
                step=step,
            )

            auxiliary_parameters = self.auxiliary_next_input_head(
                fused_hidden
            )
            auxiliary_mean, auxiliary_log_variance = (
                auxiliary_parameters.chunk(2, dim=-1)
            )
            auxiliary_log_variance = auxiliary_log_variance.clamp(
                min=-8.0,
                max=8.0,
            )

            controller_anchor = 0.0 * (
                future_utility.mean(dim=1)
                + candidate_utility
            )
            logits = (
                provisional_logits
                + controller_anchor.unsqueeze(-1)
            )
            auxiliary_mean = (
                auxiliary_mean
                + controller_anchor.unsqueeze(-1)
            )
            write_probability = (
                write_probability + controller_anchor
            )

            sequence_logits.append(logits)
            hidden_states.append(fused_hidden)
            auxiliary_means.append(auxiliary_mean)
            auxiliary_log_variances.append(
                auxiliary_log_variance
            )
            write_probabilities.append(write_probability)
            write_gates.append(write_gate)
            hard_writes.append(hard_write)
            write_slots.append(selected_slots)
            eviction_flags.append(evicted)
            memory_sizes.append(state.sizes())
            memory_masks.append(state.valid.clone())
            retrieval_weights.append(retrieval.weights)

            previous_auxiliary_mean = auxiliary_mean
            previous_auxiliary_log_variance = (
                auxiliary_log_variance
            )
            hidden = fused_hidden

        stacked_logits = torch.stack(sequence_logits, dim=1)
        stacked_hidden = torch.stack(hidden_states, dim=1)
        stacked_auxiliary_mean = torch.stack(
            auxiliary_means,
            dim=1,
        )
        stacked_auxiliary_log_variance = torch.stack(
            auxiliary_log_variances,
            dim=1,
        )
        stacked_write_probabilities = torch.stack(
            write_probabilities,
            dim=1,
        )
        stacked_write_gates = torch.stack(write_gates, dim=1)
        stacked_hard_writes = torch.stack(hard_writes, dim=1)
        stacked_write_slots = torch.stack(write_slots, dim=1)
        stacked_eviction_flags = torch.stack(
            eviction_flags,
            dim=1,
        )
        stacked_sizes = torch.stack(memory_sizes, dim=1)
        stacked_masks = torch.stack(memory_masks, dim=1)
        stacked_retrieval_weights = torch.stack(
            retrieval_weights,
            dim=1,
        )

        if torch.any(stacked_sizes > budgets.unsqueeze(1)):
            raise RuntimeError(
                "strict memory-budget invariant violated during forward"
            )

        budget_violations = torch.relu(
            stacked_sizes - budgets.unsqueeze(1)
        ).sum()

        return BudgetMemROutput(
            logits=stacked_logits[:, -1],
            sequence_logits=stacked_logits,
            hidden_states=stacked_hidden,
            auxiliary_predictions=stacked_auxiliary_mean,
            auxiliary_mean=stacked_auxiliary_mean,
            auxiliary_log_variance=stacked_auxiliary_log_variance,
            write_probabilities=stacked_write_probabilities,
            controller_probabilities=stacked_write_probabilities,
            write_gates=stacked_write_gates,
            hard_writes=stacked_hard_writes,
            write_slots=stacked_write_slots,
            eviction_flags=stacked_eviction_flags,
            retrieval_weights=stacked_retrieval_weights,
            memory_masks=stacked_masks,
            memory_mask=state.valid,
            memory_sizes=stacked_sizes,
            memory_trace=stacked_sizes,
            budgets=budgets,
            budget_violations=budget_violations,
            inputs=inputs,
            final_state=hidden,
            final_memory=state,
        )

