"""Reference implementations for the controlled Section 12 baseline suite.

The module is deliberately self-contained and CPU-testable. Existing project
modules may wrap these classes or retain their own compatible implementations.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def parameter_count(module: nn.Module, trainable_only: bool = True) -> int:
    """Return the number of scalar parameters in ``module``."""
    parameters = module.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


def assert_parameter_budget(
    module: nn.Module,
    target: int,
    tolerance: float = 0.15,
) -> None:
    """Raise when a module is outside a symmetric parameter-count tolerance."""
    if target <= 0:
        raise ValueError("target must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    actual = parameter_count(module)
    relative_error = abs(actual - target) / target
    if relative_error > tolerance:
        raise ValueError(
            f"parameter budget violated: actual={actual}, target={target}, "
            f"relative_error={relative_error:.4f}, tolerance={tolerance:.4f}"
        )


class RecurrentBaseline(nn.Module):
    """Shared implementation for vanilla RNN, GRU, and LSTM baselines."""

    CORE_TYPES: dict[str, type[nn.RNNBase]] = {
        "rnn": nn.RNN,
        "gru": nn.GRU,
        "lstm": nn.LSTM,
    }

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        core: str,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if core not in self.CORE_TYPES:
            raise ValueError(f"unsupported recurrent core: {core}")
        if min(input_dim, hidden_dim, output_dim, num_layers) <= 0:
            raise ValueError("dimensions and num_layers must be positive")
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.core_name = core
        self.core = self.CORE_TYPES[core](
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def encode(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("input must have shape [batch, sequence, features]")
        projected = self.input_proj(x)
        states, _ = self.core(projected)
        return self.norm(states)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encode(x))


class VanillaRNNBaseline(RecurrentBaseline):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            input_dim,
            hidden_dim,
            output_dim,
            core="rnn",
            num_layers=num_layers,
            dropout=dropout,
        )


class GRUBaseline(RecurrentBaseline):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            input_dim,
            hidden_dim,
            output_dim,
            core="gru",
            num_layers=num_layers,
            dropout=dropout,
        )


class LSTMBaseline(RecurrentBaseline):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            input_dim,
            hidden_dim,
            output_dim,
            core="lstm",
            num_layers=num_layers,
            dropout=dropout,
        )


class CachePolicy:
    """Base class for deterministic or seeded state-selection policies."""

    def _shape(self, states: Tensor, budget: int) -> tuple[int, int, int]:
        if states.ndim != 3:
            raise ValueError("states must have shape [batch, sequence, hidden]")
        if budget <= 0:
            raise ValueError("budget must be positive")
        batch, length, hidden = states.shape
        if length <= 0:
            raise ValueError("sequence length must be positive")
        return batch, length, hidden

    def select_indices(
        self,
        states: Tensor,
        budget: int,
        *,
        scores: Tensor | None = None,
        metadata: dict[str, Tensor] | None = None,
    ) -> Tensor:
        raise NotImplementedError


class UniformCheckpointPolicy(CachePolicy):
    def select_indices(
        self,
        states: Tensor,
        budget: int,
        *,
        scores: Tensor | None = None,
        metadata: dict[str, Tensor] | None = None,
    ) -> Tensor:
        batch, length, _ = self._shape(states, budget)
        keep = min(budget, length)
        points = torch.linspace(0, length - 1, keep, device=states.device)
        indices = points.round().long().unique(sorted=True)
        if indices.numel() < keep:
            all_indices = torch.arange(length, device=states.device)
            missing = all_indices[~torch.isin(all_indices, indices)][: keep - indices.numel()]
            indices = torch.cat([indices, missing]).sort().values
        return indices.unsqueeze(0).expand(batch, -1)


class MostRecentPolicy(CachePolicy):
    def select_indices(
        self,
        states: Tensor,
        budget: int,
        *,
        scores: Tensor | None = None,
        metadata: dict[str, Tensor] | None = None,
    ) -> Tensor:
        batch, length, _ = self._shape(states, budget)
        keep = min(budget, length)
        indices = torch.arange(length - keep, length, device=states.device)
        return indices.unsqueeze(0).expand(batch, -1)


class FIFOPolicy(MostRecentPolicy):
    """Final contents of a bounded online FIFO cache."""


class LRUPolicy(CachePolicy):
    def select_indices(
        self,
        states: Tensor,
        budget: int,
        *,
        scores: Tensor | None = None,
        metadata: dict[str, Tensor] | None = None,
    ) -> Tensor:
        batch, length, _ = self._shape(states, budget)
        keep = min(budget, length)
        if metadata is None or "last_access" not in metadata:
            return MostRecentPolicy().select_indices(states, budget)
        last_access = metadata["last_access"].to(states.device)
        if last_access.shape != (batch, length):
            raise ValueError("metadata['last_access'] must have shape [batch, sequence]")
        return last_access.topk(keep, dim=1, largest=True).indices.sort(dim=1).values


class RandomReplacementPolicy(CachePolicy):
    def __init__(self, seed: int = 2026) -> None:
        self.seed = int(seed)

    def select_indices(
        self,
        states: Tensor,
        budget: int,
        *,
        scores: Tensor | None = None,
        metadata: dict[str, Tensor] | None = None,
    ) -> Tensor:
        batch, length, _ = self._shape(states, budget)
        keep = min(budget, length)
        rows: list[Tensor] = []
        for row in range(batch):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + row)
            picked = torch.randperm(length, generator=generator)[:keep].sort().values
            rows.append(picked.to(states.device))
        return torch.stack(rows)


class ReservoirPolicy(CachePolicy):
    def __init__(self, seed: int = 2026) -> None:
        self.seed = int(seed)

    def select_indices(
        self,
        states: Tensor,
        budget: int,
        *,
        scores: Tensor | None = None,
        metadata: dict[str, Tensor] | None = None,
    ) -> Tensor:
        batch, length, _ = self._shape(states, budget)
        keep = min(budget, length)
        rows: list[Tensor] = []
        for row in range(batch):
            reservoir = list(range(keep))
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + row)
            for index in range(keep, length):
                candidate = int(
                    torch.randint(0, index + 1, (1,), generator=generator).item()
                )
                if candidate < keep:
                    reservoir[candidate] = index
            rows.append(torch.tensor(sorted(reservoir), device=states.device))
        return torch.stack(rows)


class NoveltyPolicy(CachePolicy):
    """Greedy farthest-state selection under cosine distance."""

    def select_indices(
        self,
        states: Tensor,
        budget: int,
        *,
        scores: Tensor | None = None,
        metadata: dict[str, Tensor] | None = None,
    ) -> Tensor:
        batch, length, _ = self._shape(states, budget)
        keep = min(budget, length)
        normalized = F.normalize(states.detach(), dim=-1, eps=1e-8)
        rows: list[Tensor] = []
        for row in range(batch):
            selected = [0]
            while len(selected) < keep:
                similarity = normalized[row] @ normalized[row, selected].T
                max_similarity = similarity.max(dim=1).values
                max_similarity[selected] = float("inf")
                selected.append(int(max_similarity.argmin().item()))
            rows.append(torch.tensor(sorted(selected), device=states.device))
        return torch.stack(rows)


class SurprisePolicy(CachePolicy):
    """Select states with the largest change from the preceding hidden state."""

    def select_indices(
        self,
        states: Tensor,
        budget: int,
        *,
        scores: Tensor | None = None,
        metadata: dict[str, Tensor] | None = None,
    ) -> Tensor:
        batch, length, _ = self._shape(states, budget)
        keep = min(budget, length)
        if scores is None:
            delta = torch.zeros(batch, length, device=states.device, dtype=states.dtype)
            if length > 1:
                delta[:, 1:] = (states[:, 1:] - states[:, :-1]).norm(dim=-1)
            scores = delta
        if scores.shape != (batch, length):
            raise ValueError("scores must have shape [batch, sequence]")
        return scores.topk(keep, dim=1, largest=True).indices.sort(dim=1).values


def gather_states(states: Tensor, indices: Tensor) -> Tensor:
    if indices.ndim != 2 or indices.shape[0] != states.shape[0]:
        raise ValueError("indices must have shape [batch, selected]")
    expanded = indices.unsqueeze(-1).expand(-1, -1, states.shape[-1])
    return states.gather(1, expanded)


class TransformerBaseline(nn.Module):
    """Standard full-attention or sliding-window Transformer encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
        attention_mode: str = "full",
        window_size: int = 128,
        max_length: int = 4096,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if attention_mode not in {"full", "sliding"}:
            raise ValueError("attention_mode must be 'full' or 'sliding'")
        if window_size <= 0 or max_length <= 0:
            raise ValueError("window_size and max_length must be positive")
        self.attention_mode = attention_mode
        self.window_size = window_size
        self.causal = causal
        self.max_length = max_length
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim or 4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def _mask(self, length: int, device: torch.device) -> Tensor | None:
        if not self.causal and self.attention_mode == "full":
            return None
        row = torch.arange(length, device=device).unsqueeze(1)
        col = torch.arange(length, device=device).unsqueeze(0)
        blocked = torch.zeros(length, length, dtype=torch.bool, device=device)
        if self.causal:
            blocked |= col > row
        if self.attention_mode == "sliding":
            blocked |= col < (row - self.window_size + 1)
        return blocked

    def encode(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("input must have shape [batch, sequence, features]")
        length = x.shape[1]
        if length > self.max_length:
            raise ValueError(f"sequence length {length} exceeds max_length={self.max_length}")
        hidden = self.input_proj(x) + self.position[:, :length]
        return self.norm(self.encoder(hidden, mask=self._mask(length, x.device)))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encode(x))


class S4DReferenceLayer(nn.Module):
    """CPU-safe diagonal state-space layer using an S4D-style recurrence."""

    def __init__(self, hidden_dim: int, state_dim: int = 8) -> None:
        super().__init__()
        if hidden_dim <= 0 or state_dim <= 0:
            raise ValueError("hidden_dim and state_dim must be positive")
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.log_dt = nn.Parameter(torch.full((hidden_dim, 1), -2.0))
        self.log_decay = nn.Parameter(torch.zeros(hidden_dim, state_dim))
        frequencies = torch.arange(state_dim, dtype=torch.float32) * math.pi
        self.frequency = nn.Parameter(frequencies.expand(hidden_dim, -1).clone())
        self.b_real = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.02)
        self.b_imag = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.02)
        self.c_real = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.02)
        self.c_imag = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.02)
        self.skip = nn.Parameter(torch.ones(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.hidden_dim:
            raise ValueError("x must have shape [batch, sequence, hidden_dim]")
        dtype = torch.complex64 if x.dtype in {torch.float16, torch.float32} else torch.complex128
        a = torch.complex(-self.log_decay.exp(), self.frequency).to(dtype)
        dt = self.log_dt.exp().to(a.dtype)
        abar = torch.exp(dt * a)
        b = torch.complex(self.b_real, self.b_imag).to(dtype)
        c = torch.complex(self.c_real, self.c_imag).to(dtype)
        bbar = torch.where(a.abs() > 1e-8, (abar - 1.0) / a * b, dt * b)
        state = torch.zeros(
            x.shape[0],
            self.hidden_dim,
            self.state_dim,
            dtype=dtype,
            device=x.device,
        )
        outputs: list[Tensor] = []
        for step in range(x.shape[1]):
            state = abar.unsqueeze(0) * state + bbar.unsqueeze(0) * x[:, step].unsqueeze(-1)
            projected = 2.0 * (c.unsqueeze(0) * state).real.sum(dim=-1)
            outputs.append(projected + self.skip * x[:, step])
        y = torch.stack(outputs, dim=1).to(x.dtype)
        return self.norm(y + x)


class DiagonalSSMBaseline(nn.Module):
    """Established diagonal state-space fallback suitable for CPU verification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        num_layers: int = 2,
        state_dim: int = 8,
    ) -> None:
        super().__init__()
        self.backend = "s4d_reference"
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [S4DReferenceLayer(hidden_dim, state_dim=state_dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def encode(self, x: Tensor) -> Tensor:
        hidden = self.input_proj(x)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encode(x))


class OfficialMambaOrSSMBaseline(nn.Module):
    """Use official Mamba on supported CUDA systems, otherwise the S4D fallback."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        backend: str = "auto",
        num_layers: int = 2,
        state_dim: int = 16,
    ) -> None:
        super().__init__()
        if backend not in {"auto", "official_mamba", "s4d_reference"}:
            raise ValueError("invalid backend")
        use_official = False
        mamba_class: type[nn.Module] | None = None
        if backend in {"auto", "official_mamba"} and torch.cuda.is_available():
            try:
                from mamba_ssm import Mamba  # type: ignore

                mamba_class = Mamba
                use_official = True
            except Exception:
                if backend == "official_mamba":
                    raise
        elif backend == "official_mamba":
            raise RuntimeError("official Mamba requires a supported CUDA environment")

        if use_official and mamba_class is not None:
            self.backend = "official_mamba"
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.layers = nn.ModuleList(
                [
                    mamba_class(
                        d_model=hidden_dim,
                        d_state=state_dim,
                        d_conv=4,
                        expand=2,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, output_dim)
            self.fallback = None
        else:
            self.backend = "s4d_reference"
            self.fallback = DiagonalSSMBaseline(
                input_dim,
                hidden_dim,
                output_dim,
                num_layers=num_layers,
                state_dim=min(state_dim, 16),
            )

    def encode(self, x: Tensor) -> Tensor:
        if self.fallback is not None:
            return self.fallback.encode(x)
        hidden = self.input_proj(x)
        for layer in self.layers:
            hidden = hidden + layer(hidden)
        return self.norm(hidden)

    def forward(self, x: Tensor) -> Tensor:
        if self.fallback is not None:
            return self.fallback(x)
        return self.head(self.encode(x))


class RecurrentMemoryTransformer(nn.Module):
    """Segment-recurrent Transformer with learned memory tokens."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        segment_length: int = 128,
        memory_tokens: int = 8,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
        detach_memory: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if segment_length <= 0 or memory_tokens <= 0:
            raise ValueError("segment_length and memory_tokens must be positive")
        self.segment_length = segment_length
        self.memory_token_count = memory_tokens
        self.detach_memory = detach_memory
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.memory = nn.Parameter(torch.randn(memory_tokens, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim or 4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def encode(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("input must have shape [batch, sequence, features]")
        hidden = self.input_proj(x)
        memory = self.memory.unsqueeze(0).expand(x.shape[0], -1, -1)
        outputs: list[Tensor] = []
        for start in range(0, hidden.shape[1], self.segment_length):
            segment = hidden[:, start : start + self.segment_length]
            encoded = self.encoder(torch.cat([memory, segment], dim=1))
            memory = encoded[:, : self.memory_token_count]
            if self.detach_memory:
                memory = memory.detach()
            outputs.append(encoded[:, self.memory_token_count :])
        return self.norm(torch.cat(outputs, dim=1))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encode(x))


POLICY_REGISTRY: dict[str, Callable[..., CachePolicy]] = {
    "uniform": UniformCheckpointPolicy,
    "most_recent": MostRecentPolicy,
    "fifo": FIFOPolicy,
    "lru": LRUPolicy,
    "random": RandomReplacementPolicy,
    "reservoir": ReservoirPolicy,
    "novelty": NoveltyPolicy,
    "surprise": SurprisePolicy,
}


def build_policy(name: str, **kwargs: Any) -> CachePolicy:
    try:
        factory = POLICY_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown cache policy: {name}") from exc
    return factory(**kwargs)


class MemoryCachingBaseline(nn.Module):
    """Recurrent-state caching with mean, gated, or sparse-selective aggregation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        budget: int,
        policy: str = "uniform",
        variant: str = "mean",
        seed: int = 2026,
    ) -> None:
        super().__init__()
        if budget <= 0:
            raise ValueError("budget must be positive")
        if variant not in {"mean", "gated", "sparse_selective"}:
            raise ValueError("unsupported memory-caching variant")
        policy_kwargs = {"seed": seed} if policy in {"random", "reservoir"} else {}
        self.policy_name = policy
        self.policy = build_policy(policy, **policy_kwargs)
        self.variant = variant
        self.budget = budget
        self.backbone = GRUBaseline(input_dim, hidden_dim, hidden_dim)
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fuse = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def encode(self, x: Tensor) -> Tensor:
        states = self.backbone.encode(x)
        indices = self.policy.select_indices(states, self.budget)
        cache = gather_states(states, indices)
        if self.variant == "mean":
            context = cache.mean(dim=1, keepdim=True).expand_as(states)
        elif self.variant == "gated":
            summary = cache.mean(dim=1, keepdim=True).expand_as(states)
            gate = torch.sigmoid(self.gate(torch.cat([states, summary], dim=-1)))
            context = gate * summary + (1.0 - gate) * states
        else:
            scale = states.shape[-1] ** -0.5
            weights = torch.softmax(states @ cache.transpose(1, 2) * scale, dim=-1)
            context = weights @ cache
        fused = torch.tanh(self.fuse(torch.cat([states, context], dim=-1)))
        return self.norm(states + fused)

    def cache_indices(self, x: Tensor) -> Tensor:
        states = self.backbone.encode(x)
        return self.policy.select_indices(states, self.budget)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encode(x))


BASELINE_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "vanilla_rnn": VanillaRNNBaseline,
    "gru": GRUBaseline,
    "lstm": LSTMBaseline,
    "transformer_full": lambda **kwargs: TransformerBaseline(
        attention_mode="full", **kwargs
    ),
    "transformer_sliding": lambda **kwargs: TransformerBaseline(
        attention_mode="sliding", **kwargs
    ),
    "state_space": OfficialMambaOrSSMBaseline,
    "recurrent_memory_transformer": RecurrentMemoryTransformer,
    "memory_caching_mean": lambda **kwargs: MemoryCachingBaseline(
        variant="mean", **kwargs
    ),
    "memory_caching_gated": lambda **kwargs: MemoryCachingBaseline(
        variant="gated", **kwargs
    ),
    "memory_caching_sparse_selective": lambda **kwargs: MemoryCachingBaseline(
        variant="sparse_selective", **kwargs
    ),
}


def build_baseline(name: str, **kwargs: Any) -> nn.Module:
    try:
        factory = BASELINE_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown baseline: {name}") from exc
    return factory(**kwargs)
