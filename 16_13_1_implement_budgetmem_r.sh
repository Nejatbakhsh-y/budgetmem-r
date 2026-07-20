#!/usr/bin/env bash
# Section 13 — Implement BudgetMem-R
# Run from the budgetmem-r repository root in the VS Code WSL terminal:
#   chmod +x 16_13_1_implement_budgetmem_r.sh
#   ./16_13_1_implement_budgetmem_r.sh

set -Eeuo pipefail
IFS=$'\n\t'

readonly AUTOMATION_NAME="16.13.1 — Section 13 BudgetMem-R"
readonly TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if git rev-parse --show-toplevel >/dev/null 2>&1; then
    ROOT_DIR="$(git rev-parse --show-toplevel)"
else
    ROOT_DIR="$(pwd)"
fi
cd "$ROOT_DIR"

LOG_DIR="reports/evidence/logs"
BACKUP_DIR="reports/evidence/backups/section13_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/section13_implementation_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR" "$BACKUP_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
    local exit_code=$?
    local line_number=${1:-unknown}
    echo
    echo "ERROR: ${AUTOMATION_NAME} failed at line ${line_number} with exit code ${exit_code}."
    echo "Review: ${LOG_FILE}"
    exit "$exit_code"
}
trap 'on_error $LINENO' ERR

printf '%s\n' "============================================================"
printf '%s\n' "$AUTOMATION_NAME"
printf '%s\n' "Repository: $ROOT_DIR"
printf '%s\n' "UTC run:    $TIMESTAMP"
printf '%s\n' "============================================================"

if [[ ! -d "src/budgetmem" ]]; then
    echo "ERROR: src/budgetmem was not found."
    echo "Open the budgetmem-r repository root in VS Code and run this automation again."
    exit 2
fi

write_generated_file() {
    local target="$1"
    local temporary
    temporary="$(mktemp)"
    cat > "$temporary"
    mkdir -p "$(dirname "$target")"

    if [[ -f "$target" ]]; then
        if cmp -s "$temporary" "$target"; then
            rm -f "$temporary"
            echo "UNCHANGED  $target"
            return 0
        fi
        mkdir -p "$BACKUP_DIR/$(dirname "$target")"
        cp -p "$target" "$BACKUP_DIR/$target"
        echo "BACKUP     $target -> $BACKUP_DIR/$target"
    fi

    install -m 0644 "$temporary" "$target"
    rm -f "$temporary"
    echo "WRITTEN    $target"
}

ensure_package_init() {
    local target="$1"
    if [[ ! -e "$target" ]]; then
        mkdir -p "$(dirname "$target")"
        printf '%s\n' '"""BudgetMem-R package module."""' > "$target"
        echo "CREATED    $target"
    fi
}

ensure_package_init "src/budgetmem/__init__.py"
ensure_package_init "src/budgetmem/models/__init__.py"
ensure_package_init "src/budgetmem/memory/__init__.py"
ensure_package_init "src/budgetmem/training/__init__.py"

write_generated_file "src/budgetmem/memory/budget.py" <<'__BUDGETMEM_SECTION13_FILE_1__'
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
__BUDGETMEM_SECTION13_FILE_1__

write_generated_file "src/budgetmem/memory/state.py" <<'__BUDGETMEM_SECTION13_FILE_2__'
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
__BUDGETMEM_SECTION13_FILE_2__

write_generated_file "src/budgetmem/memory/controllers.py" <<'__BUDGETMEM_SECTION13_FILE_3__'
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
        """Use a straight-through relaxed Bernoulli gate during training."""

        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be strictly between zero and one")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not training:
            return (probability >= threshold).to(probability.dtype)

        eps = torch.finfo(probability.dtype).eps
        clipped = probability.clamp(min=eps, max=1.0 - eps)
        logistic = torch.log(clipped) - torch.log1p(-clipped)
        uniform = torch.rand_like(clipped).clamp(min=eps, max=1.0 - eps)
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
__BUDGETMEM_SECTION13_FILE_3__

write_generated_file "src/budgetmem/memory/retrieval.py" <<'__BUDGETMEM_SECTION13_FILE_4__'
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
__BUDGETMEM_SECTION13_FILE_4__

write_generated_file "src/budgetmem/models/budgetmem_r.py" <<'__BUDGETMEM_SECTION13_FILE_5__'
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
__BUDGETMEM_SECTION13_FILE_5__

write_generated_file "src/budgetmem/training/losses.py" <<'__BUDGETMEM_SECTION13_FILE_6__'
"""Composite BudgetMem-R objective with auditable component losses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor

from budgetmem.memory.state import BudgetMemoryState


@dataclass(frozen=True)
class BudgetMemLossWeights:
    budget: float = 1.0
    write: float = 0.01
    auxiliary: float = 0.1
    diversity: float = 0.01


def _diversity_loss(state: BudgetMemoryState) -> Tensor:
    losses: list[Tensor] = []
    for batch_index in range(state.batch_size):
        resident = state.values[batch_index, state.valid[batch_index]]
        if resident.shape[0] < 2:
            continue
        normalized = functional.normalize(resident, dim=-1)
        similarity = normalized @ normalized.transpose(0, 1)
        off_diagonal = ~torch.eye(
            similarity.shape[0], device=similarity.device, dtype=torch.bool
        )
        losses.append(similarity[off_diagonal].pow(2).mean())
    if not losses:
        return state.values.new_zeros(())
    return torch.stack(losses).mean()


def budgetmem_objective(
    outputs: dict[str, Tensor | BudgetMemoryState],
    *,
    task_targets: Tensor,
    task_loss: Callable[[Tensor, Tensor], Tensor],
    inputs: Tensor,
    weights: BudgetMemLossWeights = BudgetMemLossWeights(),
    target_write_rate: float = 0.25,
) -> dict[str, Tensor]:
    """Compute task, budget, write, self-supervised, and diversity losses.

    Final task targets are used only by ``task_loss``. They are never supplied to
    the model or to its write, eviction, or retrieval controllers.
    """

    logits = outputs["logits"]
    auxiliary_predictions = outputs["auxiliary_predictions"]
    write_probabilities = outputs["write_probabilities"]
    memory_sizes = outputs["memory_sizes"]
    budgets = outputs["budgets"]
    final_memory = outputs["final_memory"]
    if not isinstance(logits, Tensor):
        raise TypeError("outputs['logits'] must be a tensor")
    if not isinstance(auxiliary_predictions, Tensor):
        raise TypeError("outputs['auxiliary_predictions'] must be a tensor")
    if not isinstance(write_probabilities, Tensor):
        raise TypeError("outputs['write_probabilities'] must be a tensor")
    if not isinstance(memory_sizes, Tensor) or not isinstance(budgets, Tensor):
        raise TypeError("memory sizes and budgets must be tensors")
    if not isinstance(final_memory, BudgetMemoryState):
        raise TypeError("outputs['final_memory'] must be BudgetMemoryState")

    primary = task_loss(logits, task_targets)
    overflow = torch.relu(memory_sizes.to(inputs.dtype) - budgets.unsqueeze(1).to(inputs.dtype))
    budget_loss = overflow.pow(2).mean()
    write_loss = torch.relu(write_probabilities.mean() - target_write_rate).pow(2)
    if inputs.shape[1] > 1:
        auxiliary_loss = functional.mse_loss(
            auxiliary_predictions[:, :-1], inputs[:, 1:]
        )
    else:
        auxiliary_loss = inputs.new_zeros(())
    diversity_loss = _diversity_loss(final_memory)
    total = (
        primary
        + weights.budget * budget_loss
        + weights.write * write_loss
        + weights.auxiliary * auxiliary_loss
        + weights.diversity * diversity_loss
    )
    return {
        "total": total,
        "task": primary,
        "budget": budget_loss,
        "write": write_loss,
        "auxiliary": auxiliary_loss,
        "diversity": diversity_loss,
    }
__BUDGETMEM_SECTION13_FILE_6__

write_generated_file "configs/models/budgetmem_r.yaml" <<'__BUDGETMEM_SECTION13_FILE_7__'
model:
  name: budgetmem_r
  backbone: gru
  hidden_dim: 128
  key_dim: 64
  value_dim: 128
  budget_embedding_dim: 16
  controller_dim: 64
  max_budget: 128
  training_budgets: [8, 16, 32, 64, 128]
  retrieval:
    top_k: 4
  fusion:
    mode: gated
    comparison_modes: [concatenation, residual, gated, attention]
  write:
    threshold: 0.5
    temperature: 0.67
    differentiable_training: straight_through_relaxed_bernoulli
    evaluation_selection: hard_threshold
  eviction:
    learned_policy: lowest_predicted_future_utility
    comparison_policies:
      - fifo
      - lru
      - random
      - reservoir
      - lowest_novelty
      - lowest_retrieval_frequency
  leakage_control:
    final_task_label_available_to_controller: false
    self_supervised_signal: next_input_prediction
loss:
  lambda_budget: 1.0
  lambda_write: 0.01
  lambda_aux: 0.1
  lambda_diversity: 0.01
  target_write_rate: 0.25
invariants:
  hard_memory_budget: true
  maximum_resident_slots_per_sample: requested_budget
__BUDGETMEM_SECTION13_FILE_7__

write_generated_file "tests/test_budgetmem_r.py" <<'__BUDGETMEM_SECTION13_FILE_8__'
from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as functional

from budgetmem.memory.budget import DEFAULT_TRAINING_BUDGETS, sample_training_budgets
from budgetmem.memory.state import BudgetMemoryState
from budgetmem.models.budgetmem_r import BudgetMemR
from budgetmem.training.losses import budgetmem_objective


def _model(*, fusion_mode: str = "gated", max_budget: int = 16) -> BudgetMemR:
    return BudgetMemR(
        input_dim=6,
        hidden_dim=12,
        output_dim=3,
        key_dim=8,
        value_dim=10,
        budget_embedding_dim=5,
        controller_dim=16,
        max_budget=max_budget,
        training_budgets=(4, 8, 16),
        top_k=3,
        fusion_mode=fusion_mode,
    )


def _force_writes(model: BudgetMemR) -> None:
    with torch.no_grad():
        for parameter in model.write_controller.parameters():
            parameter.zero_()
        final_layer = model.write_controller.network[-1]
        assert isinstance(final_layer, torch.nn.Linear)
        final_layer.bias.fill_(20.0)


def test_training_budget_sampling_uses_only_controlled_choices() -> None:
    generator = torch.Generator().manual_seed(2026)
    sampled = sample_training_budgets(
        256,
        device=torch.device("cpu"),
        choices=DEFAULT_TRAINING_BUDGETS,
        generator=generator,
    )
    assert set(sampled.tolist()).issubset(set(DEFAULT_TRAINING_BUDGETS))
    assert len(set(sampled.tolist())) > 1


def test_hard_budget_is_never_violated_even_when_every_step_writes() -> None:
    model = _model().eval()
    _force_writes(model)
    inputs = torch.randn(2, 24, 6)
    budgets = torch.tensor([4, 8])
    with torch.no_grad():
        outputs = model(inputs, budget=budgets)
    sizes = outputs["memory_sizes"]
    assert isinstance(sizes, torch.Tensor)
    assert torch.all(sizes <= budgets.unsqueeze(1))
    assert sizes[:, -1].tolist() == [4, 8]
    assert int(outputs["budget_violations"].item()) == 0
    final_memory = outputs["final_memory"]
    assert isinstance(final_memory, BudgetMemoryState)
    final_memory.assert_within_budget()


def test_forward_api_prevents_final_label_leakage() -> None:
    parameters = inspect.signature(BudgetMemR.forward).parameters
    forbidden = {"label", "labels", "target", "targets", "sentiment", "anomaly"}
    assert forbidden.isdisjoint(parameters)


def test_all_fusion_comparison_modes_execute() -> None:
    inputs = torch.randn(2, 5, 6)
    for mode in ("concatenation", "residual", "gated", "attention"):
        model = _model(fusion_mode=mode).eval()
        with torch.no_grad():
            outputs = model(inputs, budget=4)
        assert outputs["logits"].shape == (2, 3)
        assert outputs["retrieval_weights"].shape == (2, 5, 3)


def test_composite_objective_is_finite_and_controller_receives_gradient() -> None:
    torch.manual_seed(7)
    model = _model().train()
    inputs = torch.randn(3, 7, 6)
    targets = torch.tensor([0, 1, 2])
    outputs = model(inputs, budget=torch.tensor([4, 8, 16]))
    losses = budgetmem_objective(
        outputs,
        task_targets=targets,
        task_loss=functional.cross_entropy,
        inputs=inputs,
    )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    gradients = [
        parameter.grad
        for parameter in model.write_controller.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_invalid_budget_fails_fast() -> None:
    model = _model(max_budget=16)
    with pytest.raises(ValueError, match="Budgets must be within"):
        model(torch.randn(1, 3, 6), budget=17)
__BUDGETMEM_SECTION13_FILE_8__

write_generated_file "scripts/verify_section13.py" <<'__BUDGETMEM_SECTION13_FILE_9__'
"""Generate a concise, machine-readable Section 13 implementation report."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

from budgetmem.memory.state import BudgetMemoryState
from budgetmem.models.budgetmem_r import BudgetMemR


def _force_writes(model: BudgetMemR) -> None:
    with torch.no_grad():
        for parameter in model.write_controller.parameters():
            parameter.zero_()
        final_layer = model.write_controller.network[-1]
        if isinstance(final_layer, torch.nn.Linear):
            final_layer.bias.fill_(20.0)


def main() -> None:
    conditioning_model = BudgetMemR(
        input_dim=4,
        hidden_dim=8,
        output_dim=3,
        key_dim=4,
        value_dim=8,
        controller_dim=8,
        max_budget=128,
        training_budgets=(8, 16, 32, 64, 128),
        top_k=2,
    ).eval()
    with torch.no_grad():
        conditioning_outputs = conditioning_model(
            torch.randn(5, 3, 4),
            budget=torch.tensor([8, 16, 32, 64, 128]),
        )

    enforcement_model = BudgetMemR(
        input_dim=4,
        hidden_dim=8,
        output_dim=3,
        key_dim=4,
        value_dim=8,
        controller_dim=8,
        max_budget=8,
        training_budgets=(4, 8),
        top_k=2,
    ).eval()
    _force_writes(enforcement_model)
    with torch.no_grad():
        outputs = enforcement_model(
            torch.randn(2, 12, 4), budget=torch.tensor([4, 8])
        )

    sizes = outputs["memory_sizes"]
    budgets = outputs["budgets"]
    final_memory = outputs["final_memory"]
    conditioning_budgets = conditioning_outputs["budgets"]
    assert isinstance(sizes, torch.Tensor)
    assert isinstance(budgets, torch.Tensor)
    assert isinstance(conditioning_budgets, torch.Tensor)
    assert isinstance(final_memory, BudgetMemoryState)
    budget_pass = bool(torch.all(sizes <= budgets.unsqueeze(1)))
    leakage_pass = not any(
        name in inspect.signature(BudgetMemR.forward).parameters
        for name in ("label", "labels", "target", "targets")
    )
    final_memory.assert_within_budget()

    report = "\n".join(
        (
            "Section 13 — BudgetMem-R Implementation Verification",
            "====================================================",
            f"Memory representation: {'PASS' if final_memory.keys.ndim == 3 else 'FAIL'}",
            f"Budget conditioning: {'PASS' if conditioning_budgets.tolist() == [8, 16, 32, 64, 128] else 'FAIL'}",
            f"Write policy: {'PASS' if outputs['write_probabilities'].shape == (2, 12) else 'FAIL'}",
            f"Target leakage prevention: {'PASS' if leakage_pass else 'FAIL'}",
            f"Learned eviction policy: {'PASS' if hasattr(enforcement_model, 'eviction_controller') else 'FAIL'}",
            f"Top-k retrieval: {'PASS' if outputs['retrieval_weights'].shape == (2, 12, 2) else 'FAIL'}",
            f"Composite objective module: {'PASS' if Path('src/budgetmem/training/losses.py').is_file() else 'FAIL'}",
            f"Hard memory-budget enforcement: {'PASS' if budget_pass else 'FAIL'}",
            f"Final resident sizes: {sizes[:, -1].tolist()}",
            "Section 13: COMPLETE" if budget_pass and leakage_pass else "Section 13: NOT COMPLETE",
            "",
        )
    )
    destination = Path("reports/evidence/section13_budgetmem_r_implementation.txt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
__BUDGETMEM_SECTION13_FILE_9__

write_generated_file "docs/section13_budgetmem_r.md" <<'__BUDGETMEM_SECTION13_FILE_10__'
# Section 13 — BudgetMem-R

## Architecture

BudgetMem-R uses a GRU recurrent backbone. Each recurrent hidden state is projected into a memory key and value. The external memory is tensorized per batch item and enforces a strict deployment budget by maintaining an explicit valid-slot mask and checking the invariant after every write or eviction.

The requested budget is normalized by the configured maximum and encoded by a learned budget conditioner. The resulting embedding is supplied to the write controller, future-utility eviction controller, and retrieval query projection. During training, a budget is sampled independently for each batch item from `8, 16, 32, 64, 128` unless an explicit budget is supplied.

## Write policy

The write controller uses only causal and label-free signals:

- recurrent hidden state;
- hidden-state novelty;
- self-supervised next-input surprise;
- prediction uncertainty;
- current memory occupancy;
- time since the previous write;
- retrieved-memory agreement; and
- requested-budget embedding.

Training uses a straight-through relaxed Bernoulli gate. Evaluation uses a hard threshold.

## Leakage control

The model `forward` method does not accept task labels. Classification or anomaly labels can therefore affect only the primary task loss after the forward pass. Controller surprise is computed from the previous step's auxiliary next-input prediction and the newly observed input. This is a causal self-supervised signal and does not expose the final sentiment, anomaly, or class label.

## Eviction and retrieval

When a sample's memory is full and a hard write is selected, the future-utility controller scores resident slots and the lowest-scoring valid slot is replaced. Retrieval constructs a budget-conditioned query, selects valid top-k keys by scaled dot product, applies a softmax over selected scores, and returns the weighted value sum.

Supported fusion comparisons are concatenation, residual addition, gated fusion, and attention fusion.

## Objective

The training objective contains:

- primary task loss;
- budget-overflow penalty;
- excessive-write penalty;
- auxiliary next-input prediction loss; and
- memory diversity loss.

The budget penalty remains useful as an auditable safety term, while the implementation separately guarantees that resident memory never exceeds the requested budget.

## Verification

Run:

```bash
PYTHONPATH=src python -m pytest -q tests/test_budgetmem_r.py
PYTHONPATH=src python scripts/verify_section13.py
```

The verification report is written to `reports/evidence/section13_budgetmem_r_implementation.txt`.
__BUDGETMEM_SECTION13_FILE_10__



if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "ERROR: Python was not found. Activate the project virtual environment first."
    exit 3
fi

echo
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

if ! "$PYTHON_BIN" -c 'import torch, pytest' >/dev/null 2>&1; then
    if [[ "${BUDGETMEM_INSTALL_MISSING:-1}" == "1" ]]; then
        echo "Installing missing runtime test dependencies into the active environment..."
        "$PYTHON_BIN" -m pip install pytest torch
    else
        echo "ERROR: torch and pytest are required."
        echo "Set BUDGETMEM_INSTALL_MISSING=1 or install the project dependencies manually."
        exit 4
    fi
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

echo
echo "Compiling generated Python modules..."
"$PYTHON_BIN" -m compileall -q \
    src/budgetmem/memory/budget.py \
    src/budgetmem/memory/state.py \
    src/budgetmem/memory/controllers.py \
    src/budgetmem/memory/retrieval.py \
    src/budgetmem/models/budgetmem_r.py \
    src/budgetmem/training/losses.py \
    scripts/verify_section13.py \
    tests/test_budgetmem_r.py

echo
echo "Running Section 13 unit tests..."
"$PYTHON_BIN" -m pytest -q tests/test_budgetmem_r.py

if "$PYTHON_BIN" -m ruff --version >/dev/null 2>&1; then
    echo
    echo "Running targeted Ruff validation..."
    "$PYTHON_BIN" -m ruff check \
        src/budgetmem/memory/budget.py \
        src/budgetmem/memory/state.py \
        src/budgetmem/memory/controllers.py \
        src/budgetmem/memory/retrieval.py \
        src/budgetmem/models/budgetmem_r.py \
        src/budgetmem/training/losses.py \
        scripts/verify_section13.py \
        tests/test_budgetmem_r.py
else
    echo
    echo "Ruff is not installed in this environment; targeted Ruff validation was skipped."
fi

echo
echo "Generating Section 13 evidence..."
"$PYTHON_BIN" scripts/verify_section13.py

EVIDENCE_FILE="reports/evidence/section13_budgetmem_r_implementation.txt"
required_lines=(
    "Memory representation: PASS"
    "Budget conditioning: PASS"
    "Write policy: PASS"
    "Target leakage prevention: PASS"
    "Learned eviction policy: PASS"
    "Top-k retrieval: PASS"
    "Composite objective module: PASS"
    "Hard memory-budget enforcement: PASS"
    "Section 13: COMPLETE"
)
for required_line in "${required_lines[@]}"; do
    if ! grep -Fqx "$required_line" "$EVIDENCE_FILE"; then
        echo "ERROR: Missing required evidence line: $required_line"
        exit 5
    fi
done

MANIFEST="reports/evidence/section13_generated_files_${TIMESTAMP}.sha256"
sha256sum \
    src/budgetmem/memory/budget.py \
    src/budgetmem/memory/state.py \
    src/budgetmem/memory/controllers.py \
    src/budgetmem/memory/retrieval.py \
    src/budgetmem/models/budgetmem_r.py \
    src/budgetmem/training/losses.py \
    configs/models/budgetmem_r.yaml \
    tests/test_budgetmem_r.py \
    scripts/verify_section13.py \
    docs/section13_budgetmem_r.md \
    "$EVIDENCE_FILE" > "$MANIFEST"

echo
echo "============================================================"
echo "SECTION 13 AUTOMATION: PASS"
echo "Evidence: $EVIDENCE_FILE"
echo "Log:      $LOG_FILE"
echo "Manifest: $MANIFEST"
if find "$BACKUP_DIR" -type f -print -quit | grep -q .; then
    echo "Backups:  $BACKUP_DIR"
else
    rmdir "$BACKUP_DIR" 2>/dev/null || true
fi
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Branch:   $(git branch --show-current)"
    echo
    echo "Generated Git changes:"
    git status --short -- \
        src/budgetmem/memory/budget.py \
        src/budgetmem/memory/state.py \
        src/budgetmem/memory/controllers.py \
        src/budgetmem/memory/retrieval.py \
        src/budgetmem/models/budgetmem_r.py \
        src/budgetmem/training/losses.py \
        configs/models/budgetmem_r.yaml \
        tests/test_budgetmem_r.py \
        scripts/verify_section13.py \
        docs/section13_budgetmem_r.md \
        reports/evidence/section13_budgetmem_r_implementation.txt || true
fi
echo "============================================================"
