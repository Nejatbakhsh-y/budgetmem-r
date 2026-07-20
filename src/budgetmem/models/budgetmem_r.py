"""Budget-conditioned recurrent model with strictly bounded external memory."""

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
    """GRU-based BudgetMem-R implementation.

    The forward API deliberately accepts no task labels. Write decisions use only
    causal hidden-state, memory, budget, and self-supervised prediction signals.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        key_dim: int = 64,
        value_dim: int | None = None,
        budget_embedding_dim: int = 16,
        controller_dim: int = 64,
        max_budget: int = 128,
        training_budgets: Sequence[int] = DEFAULT_TRAINING_BUDGETS,
        top_k: int = 4,
        fusion_mode: str = "gated",
        write_threshold: float = 0.5,
        write_temperature: float = 0.67,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim, hidden_dim, and output_dim must be positive")
        if max_budget <= 0:
            raise ValueError("max_budget must be positive")
        if not training_budgets:
            raise ValueError("training_budgets cannot be empty")
        if max(training_budgets) > max_budget:
            raise ValueError("Every training budget must be <= max_budget")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim or hidden_dim)
        self.max_budget = int(max_budget)
        self.training_budgets = tuple(int(value) for value in training_budgets)
        self.write_threshold = float(write_threshold)
        self.write_temperature = float(write_temperature)

        self.recurrent_cell = nn.GRUCell(input_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, key_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, self.value_dim, bias=False)
        self.budget_conditioner = BudgetConditioner(
            max_budget=max_budget,
            embedding_dim=budget_embedding_dim,
        )
        self.retriever = TopKRetriever(
            hidden_dim=hidden_dim,
            key_dim=key_dim,
            value_dim=self.value_dim,
            budget_embedding_dim=budget_embedding_dim,
            top_k=top_k,
        )
        self.fusion = MemoryFusion(
            hidden_dim=hidden_dim,
            value_dim=self.value_dim,
            mode=fusion_mode,
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
        self.auxiliary_next_input_head = nn.Linear(hidden_dim, input_dim)
        self.initial_utility_head = nn.Sequential(
            nn.Linear(hidden_dim + budget_embedding_dim, controller_dim),
            nn.SiLU(),
            nn.Linear(controller_dim, 1),
            nn.Sigmoid(),
        )

    def _resolve_budgets(self, inputs: Tensor, budget: int | Tensor | None) -> Tensor:
        batch_size = inputs.shape[0]
        device = inputs.device
        if budget is None:
            if self.training:
                budgets = sample_training_budgets(
                    batch_size,
                    device=device,
                    choices=self.training_budgets,
                )
            else:
                budgets = torch.full(
                    (batch_size,), self.max_budget, device=device, dtype=torch.long
                )
        elif isinstance(budget, int):
            budgets = torch.full(
                (batch_size,), budget, device=device, dtype=torch.long
            )
        else:
            budgets = budget.to(device=device, dtype=torch.long)
            if budgets.ndim == 0:
                budgets = budgets.expand(batch_size)
            if budgets.shape != (batch_size,):
                raise ValueError("budget tensor must contain one value per batch item")

        if torch.any(budgets < 1) or torch.any(budgets > self.max_budget):
            raise ValueError(
                f"Budgets must be within [1, {self.max_budget}]"
            )
        return budgets

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
            "bd,bmd->bm", normalized_candidate, normalized_keys
        )
        similarities = similarities.masked_fill(~state.valid, -1.0)
        maximum = similarities.max(dim=1).values
        has_memory = state.valid.any(dim=1)
        novelty = 1.0 - maximum
        return torch.where(has_memory, novelty.clamp(0.0, 2.0) / 2.0, torch.ones_like(novelty))

    @staticmethod
    def _retrieved_agreement(candidate_value: Tensor, retrieved: Tensor, state: BudgetMemoryState) -> Tensor:
        agreement = functional.cosine_similarity(candidate_value, retrieved, dim=-1)
        return torch.where(
            state.valid.any(dim=1),
            (agreement + 1.0) / 2.0,
            torch.zeros_like(agreement),
        )

    @staticmethod
    def _record_retrievals(
        state: BudgetMemoryState, retrieval: RetrievalResult
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

    def _write_memory(
        self,
        state: BudgetMemoryState,
        *,
        candidate_key: Tensor,
        candidate_value: Tensor,
        candidate_utility: Tensor,
        write_gate: Tensor,
        hidden: Tensor,
        budget_embedding: Tensor,
        step: int,
    ) -> BudgetMemoryState:
        aged = torch.where(state.valid, state.age + 1, state.age)
        keys = state.keys.clone()
        values = state.values.clone()
        utility = state.utility.clone()
        age = aged.clone()
        retrieval_count = state.retrieval_count.clone()
        valid = state.valid.clone()
        last_write_step = state.last_write_step.clone()

        requires_eviction = (
            (state.sizes() >= state.budgets)
            & (write_gate.detach() >= 0.5)
        )
        if bool(torch.any(requires_eviction)):
            future_utility = self.eviction_controller.future_utility(
                state.values,
                stored_utility=state.utility,
                age=aged,
                retrieval_count=state.retrieval_count,
                hidden=hidden,
                budget_embedding=budget_embedding,
            )
        else:
            future_utility = state.utility

        for batch_index in range(state.batch_size):
            hard_write = bool(write_gate[batch_index].detach().item() >= 0.5)
            if not hard_write:
                continue

            budget_value = int(state.budgets[batch_index].item())
            current_valid = valid[batch_index]
            current_size = int(current_valid.sum().item())
            if current_size < budget_value:
                invalid_indices = torch.nonzero(
                    ~current_valid, as_tuple=False
                ).flatten()
                slot_index = int(invalid_indices[0].item())
            else:
                valid_indices = torch.nonzero(
                    current_valid, as_tuple=False
                ).flatten()
                scores = future_utility[batch_index, valid_indices]
                slot_index = int(valid_indices[torch.argmin(scores)].item())

            gate = write_gate[batch_index]
            previous_key = keys[batch_index, slot_index].clone()
            previous_value = values[batch_index, slot_index].clone()
            keys[batch_index, slot_index] = (
                gate * candidate_key[batch_index]
                + (1.0 - gate) * previous_key
            )
            values[batch_index, slot_index] = (
                gate * candidate_value[batch_index]
                + (1.0 - gate) * previous_value
            )
            utility[batch_index, slot_index] = candidate_utility[batch_index]
            age[batch_index, slot_index] = 0
            retrieval_count[batch_index, slot_index] = 0.0
            valid[batch_index, slot_index] = True
            last_write_step[batch_index] = step

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
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, sequence, input_dim]")
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, received {inputs.shape[-1]}"
            )

        batch_size, sequence_length, _ = inputs.shape
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
        previous_auxiliary_prediction: Tensor | None = None

        sequence_logits: list[Tensor] = []
        auxiliary_predictions: list[Tensor] = []
        write_probabilities: list[Tensor] = []
        write_gates: list[Tensor] = []
        memory_sizes: list[Tensor] = []
        retrieval_weights: list[Tensor] = []

        for step in range(sequence_length):
            current_input = inputs[:, step]
            hidden = self.recurrent_cell(current_input, hidden)
            retrieval = self.retriever(
                hidden,
                state.keys,
                state.values,
                state.valid,
                budget_embedding,
            )
            state = self._record_retrievals(state, retrieval)
            fused_hidden = self.fusion(hidden, retrieval.retrieved)
            logits = self.task_head(fused_hidden)
            auxiliary_prediction = self.auxiliary_next_input_head(fused_hidden)

            candidate_key = self.key_projection(hidden)
            candidate_value = self.value_projection(hidden)
            novelty = self._novelty(candidate_key, state)
            if previous_auxiliary_prediction is None:
                surprise = inputs.new_zeros(batch_size)
            else:
                surprise = functional.mse_loss(
                    previous_auxiliary_prediction,
                    current_input,
                    reduction="none",
                ).mean(dim=-1)
                surprise = surprise / (1.0 + surprise)
            uncertainty = self._prediction_uncertainty(logits)
            occupancy = state.sizes().to(inputs.dtype) / budgets.to(inputs.dtype)
            time_since_write = torch.where(
                state.last_write_step < 0,
                torch.ones_like(state.last_write_step, dtype=inputs.dtype),
                (step - state.last_write_step).to(inputs.dtype) / float(step + 1),
            )
            retrieved_agreement = self._retrieved_agreement(
                candidate_value, retrieval.retrieved, state
            )
            write_probability = self.write_controller(
                hidden,
                novelty=novelty,
                surprise=surprise,
                uncertainty=uncertainty,
                occupancy=occupancy,
                time_since_write=time_since_write,
                retrieved_agreement=retrieved_agreement,
                budget_embedding=budget_embedding,
            )
            write_gate = self.write_controller.differentiable_gate(
                write_probability,
                training=self.training,
                threshold=self.write_threshold,
                temperature=self.write_temperature,
            )
            candidate_utility = self.initial_utility_head(
                torch.cat((fused_hidden, budget_embedding), dim=-1)
            ).squeeze(-1)
            state = self._write_memory(
                state,
                candidate_key=candidate_key,
                candidate_value=candidate_value,
                candidate_utility=candidate_utility,
                write_gate=write_gate,
                hidden=hidden,
                budget_embedding=budget_embedding,
                step=step,
            )

            sequence_logits.append(logits)
            auxiliary_predictions.append(auxiliary_prediction)
            write_probabilities.append(write_probability)
            write_gates.append(write_gate)
            memory_sizes.append(state.sizes())
            retrieval_weights.append(retrieval.weights)
            previous_auxiliary_prediction = auxiliary_prediction
            hidden = fused_hidden

        stacked_logits = torch.stack(sequence_logits, dim=1)
        stacked_sizes = torch.stack(memory_sizes, dim=1)
        budget_violations = torch.relu(
            stacked_sizes - budgets.unsqueeze(1)
        ).sum()
        stacked_write_probabilities = torch.stack(write_probabilities, dim=1)
        stacked_write_gates = torch.stack(write_gates, dim=1)
        return BudgetMemROutput(
            logits=stacked_logits[:, -1],
            sequence_logits=stacked_logits,
            auxiliary_predictions=torch.stack(auxiliary_predictions, dim=1),
            write_probabilities=stacked_write_probabilities,
            controller_probabilities=stacked_write_probabilities,
            write_gates=stacked_write_gates,
            hard_writes=stacked_write_gates >= self.write_threshold,
            memory_sizes=stacked_sizes,
            memory_trace=stacked_sizes,
            budgets=budgets,
            budget_violations=budget_violations,
            retrieval_weights=torch.stack(retrieval_weights, dim=1),
            final_state=hidden,
            final_memory=state,
            memory_mask=state.valid,
        )
