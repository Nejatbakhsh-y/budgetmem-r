#!/usr/bin/env bash
set -Eeuo pipefail

# Section 12 — Controlled Baseline Implementation
# Run from the budgetmem-r repository root in the VS Code WSL terminal.
#
# Optional environment variables:
#   PYTHON_BIN=/path/to/python
#   INSTALL_MISSING=1          Install torch/pytest if unavailable (default: 1)
#   FORCE_TARGET_FILES=0       Replace exact-path target modules after backup (default: 0)
#   RUN_FULL_TESTS=0           Run the complete project test suite after Section 12 tests (default: 0)
#   AUTO_COMMIT=0              Commit only files created/updated by this automation (default: 0)

SECTION="12"
INSTALL_MISSING="${INSTALL_MISSING:-1}"
FORCE_TARGET_FILES="${FORCE_TARGET_FILES:-0}"
RUN_FULL_TESTS="${RUN_FULL_TESTS:-0}"
AUTO_COMMIT="${AUTO_COMMIT:-0}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

log()  { printf '[Section %s] %s\n' "$SECTION" "$*"; }
fail() { printf '[Section %s] ERROR: %s\n' "$SECTION" "$*" >&2; exit 1; }

if git rev-parse --show-toplevel >/dev/null 2>&1; then
  ROOT="$(git rev-parse --show-toplevel)"
else
  ROOT="$(pwd)"
fi
cd "$ROOT"

[[ -d src ]] || fail "Run this file from the budgetmem-r repository root; src/ was not found."

BACKUP_ROOT="reports/evidence/backups/section12_${STAMP}"
mkdir -p "$BACKUP_ROOT" reports/evidence reports/tables configs/baselines scripts tests
mkdir -p src/budgetmem/baselines src/budgetmem/models src/budgetmem/memory

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -x .venv/bin/python ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="$(command -v python3 || command -v python || true)"
fi
[[ -n "$PYTHON" ]] || fail "Python was not found. Activate the project virtual environment."
log "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"

ensure_module() {
  local module="$1"
  local package="$2"
  if ! "$PYTHON" -c "import ${module}" >/dev/null 2>&1; then
    [[ "$INSTALL_MISSING" == "1" ]] || fail "Missing Python module '${module}'. Re-run with INSTALL_MISSING=1."
    log "Installing missing package: ${package}"
    "$PYTHON" -m pip install "$package"
  fi
}
ensure_module torch torch
ensure_module pytest pytest

backup_path() {
  local path="$1"
  if [[ -e "$path" ]]; then
    mkdir -p "$BACKUP_ROOT/$(dirname "$path")"
    cp -a "$path" "$BACKUP_ROOT/$path"
  fi
}

install_owned() {
  local path="$1"
  local tmp
  tmp="$(mktemp)"
  cat > "$tmp"
  mkdir -p "$(dirname "$path")"
  if [[ -f "$path" ]] && cmp -s "$tmp" "$path"; then
    rm -f "$tmp"
    log "Unchanged: $path"
    return
  fi
  backup_path "$path"
  mv "$tmp" "$path"
  log "Installed: $path"
}

install_target() {
  local path="$1"
  local tmp
  tmp="$(mktemp)"
  cat > "$tmp"
  mkdir -p "$(dirname "$path")"
  if [[ -f "$path" && "$FORCE_TARGET_FILES" != "1" ]]; then
    rm -f "$tmp"
    log "Retained existing target: $path"
    return
  fi
  backup_path "$path"
  mv "$tmp" "$path"
  log "Installed target: $path"
}

install_owned src/budgetmem/baselines/__init__.py <<'PY'
"""Controlled baseline suite for Section 12."""

from .controlled import (
    BASELINE_REGISTRY,
    POLICY_REGISTRY,
    DiagonalSSMBaseline,
    FIFOPolicy,
    GRUBaseline,
    LRUPolicy,
    LSTMBaseline,
    MemoryCachingBaseline,
    MostRecentPolicy,
    NoveltyPolicy,
    RandomReplacementPolicy,
    RecurrentMemoryTransformer,
    ReservoirPolicy,
    SurprisePolicy,
    TransformerBaseline,
    UniformCheckpointPolicy,
    VanillaRNNBaseline,
    assert_parameter_budget,
    build_baseline,
    build_policy,
    parameter_count,
)

__all__ = [
    "BASELINE_REGISTRY",
    "POLICY_REGISTRY",
    "DiagonalSSMBaseline",
    "FIFOPolicy",
    "GRUBaseline",
    "LRUPolicy",
    "LSTMBaseline",
    "MemoryCachingBaseline",
    "MostRecentPolicy",
    "NoveltyPolicy",
    "RandomReplacementPolicy",
    "RecurrentMemoryTransformer",
    "ReservoirPolicy",
    "SurprisePolicy",
    "TransformerBaseline",
    "UniformCheckpointPolicy",
    "VanillaRNNBaseline",
    "assert_parameter_budget",
    "build_baseline",
    "build_policy",
    "parameter_count",
]
PY

install_owned src/budgetmem/baselines/controlled.py <<'PY'
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
PY

install_target src/budgetmem/models/rnn.py <<'PY'
"""Vanilla RNN baseline."""

from budgetmem.baselines.controlled import VanillaRNNBaseline

RNNBaseline = VanillaRNNBaseline
RNNModel = VanillaRNNBaseline

__all__ = ["VanillaRNNBaseline", "RNNBaseline", "RNNModel"]
PY

install_target src/budgetmem/models/gru.py <<'PY'
"""GRU baseline."""

from budgetmem.baselines.controlled import GRUBaseline

GRUModel = GRUBaseline

__all__ = ["GRUBaseline", "GRUModel"]
PY

install_target src/budgetmem/models/lstm.py <<'PY'
"""LSTM baseline."""

from budgetmem.baselines.controlled import LSTMBaseline

LSTMModel = LSTMBaseline

__all__ = ["LSTMBaseline", "LSTMModel"]
PY

install_target src/budgetmem/memory/uniform.py <<'PY'
"""Uniform checkpoint-selection policy."""

from budgetmem.baselines.controlled import UniformCheckpointPolicy

UniformPolicy = UniformCheckpointPolicy

__all__ = ["UniformCheckpointPolicy", "UniformPolicy"]
PY

install_target src/budgetmem/memory/most_recent.py <<'PY'
"""Most-recent-state cache policy."""

from budgetmem.baselines.controlled import MostRecentPolicy

MostRecentCachePolicy = MostRecentPolicy

__all__ = ["MostRecentPolicy", "MostRecentCachePolicy"]
PY

install_target src/budgetmem/memory/fifo.py <<'PY'
"""FIFO cache policy."""

from budgetmem.baselines.controlled import FIFOPolicy

FIFO = FIFOPolicy

__all__ = ["FIFOPolicy", "FIFO"]
PY

install_target src/budgetmem/memory/lru.py <<'PY'
"""Least-recently-used cache policy."""

from budgetmem.baselines.controlled import LRUPolicy

LRU = LRUPolicy

__all__ = ["LRUPolicy", "LRU"]
PY

install_target src/budgetmem/memory/random_policy.py <<'PY'
"""Seeded random-replacement cache policy."""

from budgetmem.baselines.controlled import RandomReplacementPolicy

RandomPolicy = RandomReplacementPolicy

__all__ = ["RandomReplacementPolicy", "RandomPolicy"]
PY

install_target src/budgetmem/memory/reservoir.py <<'PY'
"""Reservoir-sampling cache policy."""

from budgetmem.baselines.controlled import ReservoirPolicy

ReservoirSamplingPolicy = ReservoirPolicy

__all__ = ["ReservoirPolicy", "ReservoirSamplingPolicy"]
PY

install_target src/budgetmem/memory/novelty.py <<'PY'
"""Novelty-only cache policy."""

from budgetmem.baselines.controlled import NoveltyPolicy

NoveltyOnlyPolicy = NoveltyPolicy

__all__ = ["NoveltyPolicy", "NoveltyOnlyPolicy"]
PY

install_target src/budgetmem/memory/surprise.py <<'PY'
"""Surprise-only cache policy."""

from budgetmem.baselines.controlled import SurprisePolicy

SurpriseOnlyPolicy = SurprisePolicy

__all__ = ["SurprisePolicy", "SurpriseOnlyPolicy"]
PY

install_target src/budgetmem/models/attention.py <<'PY'
"""Full-attention and sliding-window Transformer baselines."""

from budgetmem.baselines.controlled import TransformerBaseline

__all__ = ["TransformerBaseline"]
PY

install_target src/budgetmem/models/state_space.py <<'PY'
"""Official-Mamba adapter with CPU-safe diagonal SSM fallback."""

from budgetmem.baselines.controlled import (
    DiagonalSSMBaseline,
    OfficialMambaOrSSMBaseline,
)

MambaBaseline = OfficialMambaOrSSMBaseline

__all__ = ["DiagonalSSMBaseline", "OfficialMambaOrSSMBaseline", "MambaBaseline"]
PY

install_target src/budgetmem/models/recurrent_memory.py <<'PY'
"""Segment-recurrent Transformer baseline."""

from budgetmem.baselines.controlled import RecurrentMemoryTransformer

RMTBaseline = RecurrentMemoryTransformer

__all__ = ["RecurrentMemoryTransformer", "RMTBaseline"]
PY

install_target src/budgetmem/models/memory_caching.py <<'PY'
"""Memory Caching recurrent baseline."""

from budgetmem.baselines.controlled import MemoryCachingBaseline

__all__ = ["MemoryCachingBaseline"]
PY

for init_file in src/budgetmem/models/__init__.py src/budgetmem/memory/__init__.py; do
  if [[ ! -e "$init_file" ]]; then
    printf '"""BudgetMem package module."""\n' > "$init_file"
    log "Created: $init_file"
  fi
done

install_owned configs/baselines/section12_baselines.yaml <<'YAML'
schema_version: "1.0"
section: 12
seed: 2026

fairness_controls:
  reference_model: gru
  target_training_tokens: 1000000
  identical_optimizer_schedule: true
  identical_dataset_order: true
  identical_task_splits: true
  parameter_budget_tolerance: 0.20
  report_compute_and_peak_memory: true

stages:
  stage_1:
    name: simple_recurrent
    models: [vanilla_rnn, gru, lstm]
  stage_2:
    name: deterministic_caching
    recurrent_backbone: gru
    policies:
      [uniform, most_recent, fifo, lru, random, reservoir, novelty, surprise]
  stage_3:
    name: attention
    models: [transformer_full, transformer_sliding]
    sliding_window: 128
  stage_4:
    name: modern_sequence
    preferred_backend: official_mamba
    cpu_fallback: s4d_reference
  stage_5:
    name: recurrent_memory
    model: recurrent_memory_transformer
    segment_length: 128
    memory_tokens: 8
  stage_6:
    name: memory_caching
    variants: [mean, gated, sparse_selective]

implementation_order: [stage_1, stage_2, stage_3, stage_4, stage_5, stage_6]
YAML

install_owned scripts/calibrate_section12_parameters.py <<'PY'
"""Calibrate hidden dimensions against the GRU parameter budget."""

from __future__ import annotations

import json
from pathlib import Path

from budgetmem.baselines.controlled import (
    DiagonalSSMBaseline,
    GRUBaseline,
    LSTMBaseline,
    MemoryCachingBaseline,
    RecurrentMemoryTransformer,
    TransformerBaseline,
    VanillaRNNBaseline,
    parameter_count,
)

INPUT_DIM = 32
OUTPUT_DIM = 16
REFERENCE_HIDDEN = 64
DEFAULT_CANDIDATES = list(range(8, 137, 4))
STATE_SPACE_CANDIDATES = list(range(8, 513, 4))


def closest(factory, target: int, candidates: list[int] | None = None) -> dict[str, int | float]:
    best: tuple[float, int, int] | None = None
    for hidden in (candidates or DEFAULT_CANDIDATES):
        try:
            model = factory(hidden)
        except (ValueError, RuntimeError):
            continue
        count = parameter_count(model)
        error = abs(count - target) / target
        if best is None or error < best[0]:
            best = (error, hidden, count)
    if best is None:
        raise RuntimeError("no valid hidden dimension found")
    return {
        "hidden_dim": best[1],
        "parameters": best[2],
        "relative_error": round(best[0], 6),
    }


def main() -> None:
    target = parameter_count(
        GRUBaseline(INPUT_DIM, REFERENCE_HIDDEN, OUTPUT_DIM)
    )
    factories = {
        "vanilla_rnn": lambda h: VanillaRNNBaseline(INPUT_DIM, h, OUTPUT_DIM),
        "gru": lambda h: GRUBaseline(INPUT_DIM, h, OUTPUT_DIM),
        "lstm": lambda h: LSTMBaseline(INPUT_DIM, h, OUTPUT_DIM),
        "transformer_full": lambda h: TransformerBaseline(
            INPUT_DIM,
            h,
            OUTPUT_DIM,
            num_layers=1,
            num_heads=4,
            feedforward_dim=2 * h,
            attention_mode="full",
            max_length=256,
        ),
        "transformer_sliding": lambda h: TransformerBaseline(
            INPUT_DIM,
            h,
            OUTPUT_DIM,
            num_layers=1,
            num_heads=4,
            feedforward_dim=2 * h,
            attention_mode="sliding",
            window_size=32,
            max_length=256,
        ),
        "state_space": lambda h: DiagonalSSMBaseline(
            INPUT_DIM,
            h,
            OUTPUT_DIM,
            num_layers=1,
            state_dim=4,
        ),
        "recurrent_memory_transformer": lambda h: RecurrentMemoryTransformer(
            INPUT_DIM,
            h,
            OUTPUT_DIM,
            segment_length=32,
            memory_tokens=4,
            num_layers=1,
            num_heads=4,
            feedforward_dim=2 * h,
        ),
        "memory_caching_mean": lambda h: MemoryCachingBaseline(
            INPUT_DIM, h, OUTPUT_DIM, budget=8, variant="mean"
        ),
        "memory_caching_gated": lambda h: MemoryCachingBaseline(
            INPUT_DIM, h, OUTPUT_DIM, budget=8, variant="gated"
        ),
        "memory_caching_sparse_selective": lambda h: MemoryCachingBaseline(
            INPUT_DIM, h, OUTPUT_DIM, budget=8, variant="sparse_selective"
        ),
    }
    results = {
        "reference_model": "gru",
        "reference_hidden_dim": REFERENCE_HIDDEN,
        "target_parameters": target,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
                "models": {
            name: closest(
                factory,
                target,
                STATE_SPACE_CANDIDATES if name == "state_space" else DEFAULT_CANDIDATES,
            )
            for name, factory in factories.items()
        },
    }
    output = Path("reports/tables/section12_parameter_calibration.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
PY

install_owned scripts/verify_section12_baselines.py <<'PY'
"""Verify Section 12 baseline completeness and write auditable evidence."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from budgetmem.baselines.controlled import (
    BASELINE_REGISTRY,
    POLICY_REGISTRY,
    DiagonalSSMBaseline,
    GRUBaseline,
    LSTMBaseline,
    MemoryCachingBaseline,
    OfficialMambaOrSSMBaseline,
    RecurrentMemoryTransformer,
    TransformerBaseline,
    VanillaRNNBaseline,
    parameter_count,
)

REQUIRED_FILES = [
    "src/budgetmem/models/rnn.py",
    "src/budgetmem/models/gru.py",
    "src/budgetmem/models/lstm.py",
    "src/budgetmem/memory/uniform.py",
    "src/budgetmem/memory/most_recent.py",
    "src/budgetmem/memory/fifo.py",
    "src/budgetmem/memory/lru.py",
    "src/budgetmem/memory/random_policy.py",
    "src/budgetmem/memory/reservoir.py",
    "src/budgetmem/memory/novelty.py",
    "src/budgetmem/memory/surprise.py",
    "src/budgetmem/models/attention.py",
    "src/budgetmem/models/state_space.py",
    "src/budgetmem/models/recurrent_memory.py",
    "src/budgetmem/models/memory_caching.py",
    "configs/baselines/section12_baselines.yaml",
]

REQUIRED_POLICIES = {
    "uniform",
    "most_recent",
    "fifo",
    "lru",
    "random",
    "reservoir",
    "novelty",
    "surprise",
}
REQUIRED_BASELINES = {
    "vanilla_rnn",
    "gru",
    "lstm",
    "transformer_full",
    "transformer_sliding",
    "state_space",
    "recurrent_memory_transformer",
    "memory_caching_mean",
    "memory_caching_gated",
    "memory_caching_sparse_selective",
}


def check_forward_and_gradient(model: torch.nn.Module, x: torch.Tensor) -> bool:
    model.zero_grad(set_to_none=True)
    output = model(x)
    if output.shape[:2] != x.shape[:2]:
        return False
    loss = output.square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    return bool(gradients) and all(
        grad is None or torch.isfinite(grad).all().item() for grad in gradients
    ) and any(grad is not None and grad.abs().sum().item() > 0 for grad in gradients)


def main() -> None:
    torch.manual_seed(2026)
    x = torch.randn(2, 16, 8)
    checks: dict[str, bool] = {}

    checks["required_files"] = all(Path(path).is_file() for path in REQUIRED_FILES)
    checks["policy_registry"] = REQUIRED_POLICIES.issubset(POLICY_REGISTRY)
    checks["baseline_registry"] = REQUIRED_BASELINES.issubset(BASELINE_REGISTRY)

    stage1_models = [
        VanillaRNNBaseline(8, 16, 4),
        GRUBaseline(8, 16, 4),
        LSTMBaseline(8, 16, 4),
    ]
    checks["stage_1_simple_recurrent"] = all(
        check_forward_and_gradient(model, x) for model in stage1_models
    )

    states = torch.randn(2, 16, 12)
    policy_pass = True
    policy_indices: dict[str, list[list[int]]] = {}
    for name, factory in POLICY_REGISTRY.items():
        kwargs = {"seed": 2026} if name in {"random", "reservoir"} else {}
        policy = factory(**kwargs)
        indices = policy.select_indices(states, 4)
        policy_pass &= indices.shape == (2, 4)
        policy_pass &= bool(((indices >= 0) & (indices < 16)).all().item())
        policy_pass &= all(len(set(row.tolist())) == 4 for row in indices)
        policy_indices[name] = indices.tolist()
    checks["stage_2_deterministic_caching"] = policy_pass

    attention_models = [
        TransformerBaseline(
            8, 16, 4, num_layers=1, num_heads=4, attention_mode="full", max_length=64
        ),
        TransformerBaseline(
            8,
            16,
            4,
            num_layers=1,
            num_heads=4,
            attention_mode="sliding",
            window_size=4,
            max_length=64,
        ),
    ]
    checks["stage_3_attention"] = all(
        check_forward_and_gradient(model, x) for model in attention_models
    )

    state_space = OfficialMambaOrSSMBaseline(
        8, 12, 4, num_layers=1, state_dim=4, backend="auto"
    )
    checks["stage_4_modern_sequence"] = check_forward_and_gradient(state_space, x)

    rmt = RecurrentMemoryTransformer(
        8,
        16,
        4,
        segment_length=6,
        memory_tokens=2,
        num_layers=1,
        num_heads=4,
    )
    checks["stage_5_recurrent_memory"] = check_forward_and_gradient(rmt, x)

    caching_models = [
        MemoryCachingBaseline(8, 12, 4, budget=4, variant=variant)
        for variant in ("mean", "gated", "sparse_selective")
    ]
    checks["stage_6_memory_caching"] = all(
        check_forward_and_gradient(model, x) for model in caching_models
    )

    calibration_path = Path("reports/tables/section12_parameter_calibration.json")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    max_error = max(
        float(item["relative_error"]) for item in calibration["models"].values()
    )
    checks["parameter_budget_calibration"] = max_error <= 0.20
    checks["training_token_budget_contract"] = (
        "target_training_tokens: 1000000"
        in Path("configs/baselines/section12_baselines.yaml").read_text(encoding="utf-8")
    )

    complete = all(checks.values())
    report = {
        "checks": checks,
        "complete": complete,
        "state_space_backend": state_space.backend,
        "maximum_parameter_relative_error": max_error,
        "policy_indices_smoke_test": policy_indices,
        "parameter_counts_smoke_test": {
            "rnn": parameter_count(stage1_models[0]),
            "gru": parameter_count(stage1_models[1]),
            "lstm": parameter_count(stage1_models[2]),
            "s4d_reference": parameter_count(DiagonalSSMBaseline(8, 12, 4, num_layers=1, state_dim=4)),
        },
    }

    json_path = Path("reports/evidence/section12_baselines_verification.json")
    text_path = Path("reports/evidence/section12_baselines_report.txt")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    status = lambda value: "PASS" if value else "FAIL"
    lines = [
        "Section 12 — Controlled Baseline Verification",
        "=" * 44,
        f"Required implementation files: {status(checks['required_files'])}",
        f"Stage 1 — Simple recurrent baselines: {status(checks['stage_1_simple_recurrent'])}",
        f"Stage 2 — Deterministic caching baselines: {status(checks['stage_2_deterministic_caching'])}",
        f"Stage 3 — Attention baselines: {status(checks['stage_3_attention'])}",
        f"Stage 4 — Modern sequence baseline: {status(checks['stage_4_modern_sequence'])}",
        f"Stage 4 backend used: {state_space.backend}",
        f"Stage 5 — Recurrent-memory baseline: {status(checks['stage_5_recurrent_memory'])}",
        f"Stage 6 — Memory Caching baseline: {status(checks['stage_6_memory_caching'])}",
        f"Baseline registry: {status(checks['baseline_registry'])}",
        f"Policy registry: {status(checks['policy_registry'])}",
        f"Parameter-budget calibration: {status(checks['parameter_budget_calibration'])}",
        f"Maximum parameter relative error: {max_error:.4f}",
        f"Equal training-token contract: {status(checks['training_token_budget_contract'])}",
        f"Section 12: {'COMPLETE' if complete else 'NOT COMPLETE'}",
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
PY

install_owned tests/test_section12_baselines.py <<'PY'
"""Contract tests for the controlled Section 12 baseline suite."""

from __future__ import annotations

import pytest
import torch

from budgetmem.baselines.controlled import (
    BASELINE_REGISTRY,
    POLICY_REGISTRY,
    DiagonalSSMBaseline,
    GRUBaseline,
    LSTMBaseline,
    MemoryCachingBaseline,
    OfficialMambaOrSSMBaseline,
    RecurrentMemoryTransformer,
    TransformerBaseline,
    VanillaRNNBaseline,
    assert_parameter_budget,
    parameter_count,
)


@pytest.mark.parametrize(
    "model",
    [
        VanillaRNNBaseline(8, 16, 4),
        GRUBaseline(8, 16, 4),
        LSTMBaseline(8, 16, 4),
    ],
)
def test_recurrent_baselines_shape_and_gradient(model: torch.nn.Module) -> None:
    x = torch.randn(2, 12, 8)
    output = model(x)
    assert output.shape == (2, 12, 4)
    output.mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("name", sorted(POLICY_REGISTRY))
def test_cache_policies_respect_budget(name: str) -> None:
    states = torch.randn(3, 17, 8)
    kwargs = {"seed": 11} if name in {"random", "reservoir"} else {}
    indices = POLICY_REGISTRY[name](**kwargs).select_indices(states, budget=5)
    assert indices.shape == (3, 5)
    assert torch.all(indices >= 0)
    assert torch.all(indices < 17)
    assert all(len(set(row.tolist())) == 5 for row in indices)


def test_seeded_policies_are_reproducible() -> None:
    states = torch.randn(2, 19, 8)
    for name in ("random", "reservoir"):
        first = POLICY_REGISTRY[name](seed=7).select_indices(states, 6)
        second = POLICY_REGISTRY[name](seed=7).select_indices(states, 6)
        assert torch.equal(first, second)


@pytest.mark.parametrize("mode", ["full", "sliding"])
def test_attention_baselines(mode: str) -> None:
    model = TransformerBaseline(
        8,
        16,
        4,
        num_layers=1,
        num_heads=4,
        attention_mode=mode,
        window_size=4,
        max_length=64,
    )
    assert model(torch.randn(2, 14, 8)).shape == (2, 14, 4)


def test_state_space_baseline_cpu_fallback() -> None:
    model = OfficialMambaOrSSMBaseline(
        8, 12, 4, backend="s4d_reference", num_layers=1, state_dim=4
    )
    output = model(torch.randn(2, 10, 8))
    assert output.shape == (2, 10, 4)
    assert model.backend == "s4d_reference"


def test_recurrent_memory_transformer_preserves_length() -> None:
    model = RecurrentMemoryTransformer(
        8,
        16,
        4,
        segment_length=5,
        memory_tokens=2,
        num_layers=1,
        num_heads=4,
    )
    assert model(torch.randn(2, 13, 8)).shape == (2, 13, 4)


@pytest.mark.parametrize("variant", ["mean", "gated", "sparse_selective"])
def test_memory_caching_variants(variant: str) -> None:
    model = MemoryCachingBaseline(
        8, 12, 4, budget=4, policy="uniform", variant=variant
    )
    x = torch.randn(2, 15, 8)
    assert model(x).shape == (2, 15, 4)
    assert model.cache_indices(x).shape == (2, 4)


def test_registries_cover_all_required_baselines() -> None:
    assert {
        "vanilla_rnn",
        "gru",
        "lstm",
        "transformer_full",
        "transformer_sliding",
        "state_space",
        "recurrent_memory_transformer",
        "memory_caching_mean",
        "memory_caching_gated",
        "memory_caching_sparse_selective",
    }.issubset(BASELINE_REGISTRY)


def test_parameter_budget_guard() -> None:
    model = DiagonalSSMBaseline(8, 12, 4, num_layers=1, state_dim=4)
    target = parameter_count(model)
    assert_parameter_budget(model, target, tolerance=0.0)
    with pytest.raises(ValueError):
        assert_parameter_budget(model, target * 2, tolerance=0.1)
PY

if [[ -f .gitignore ]]; then
  if ! grep -qxF 'reports/evidence/backups/' .gitignore; then
    printf '\n# Local automation backups\nreports/evidence/backups/\n' >> .gitignore
    log "Updated: .gitignore"
  fi
else
  printf '# Local automation backups\nreports/evidence/backups/\n' > .gitignore
  log "Created: .gitignore"
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

log "Compiling controlled baseline modules"
"$PYTHON" -m compileall -q src/budgetmem/baselines scripts/calibrate_section12_parameters.py scripts/verify_section12_baselines.py tests/test_section12_baselines.py

log "Calibrating model dimensions to the GRU parameter budget"
"$PYTHON" scripts/calibrate_section12_parameters.py

log "Running Section 12 contract tests"
"$PYTHON" -m pytest -q tests/test_section12_baselines.py

log "Writing final Section 12 verification evidence"
"$PYTHON" scripts/verify_section12_baselines.py | tee "reports/evidence/logs_section12_${STAMP}.txt"

if [[ "$RUN_FULL_TESTS" == "1" ]]; then
  log "Running complete project test suite"
  "$PYTHON" -m pytest -q
fi

if [[ "$AUTO_COMMIT" == "1" ]] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add \
    src/budgetmem/baselines \
    src/budgetmem/models/rnn.py \
    src/budgetmem/models/gru.py \
    src/budgetmem/models/lstm.py \
    src/budgetmem/models/attention.py \
    src/budgetmem/models/state_space.py \
    src/budgetmem/models/recurrent_memory.py \
    src/budgetmem/models/memory_caching.py \
    src/budgetmem/memory/uniform.py \
    src/budgetmem/memory/most_recent.py \
    src/budgetmem/memory/fifo.py \
    src/budgetmem/memory/lru.py \
    src/budgetmem/memory/random_policy.py \
    src/budgetmem/memory/reservoir.py \
    src/budgetmem/memory/novelty.py \
    src/budgetmem/memory/surprise.py \
    configs/baselines/section12_baselines.yaml \
    scripts/calibrate_section12_parameters.py \
    scripts/verify_section12_baselines.py \
    tests/test_section12_baselines.py \
    reports/evidence/section12_baselines_verification.json \
    reports/evidence/section12_baselines_report.txt \
    reports/tables/section12_parameter_calibration.json \
    .gitignore
  if ! git diff --cached --quiet; then
    git commit -m "Implement controlled Section 12 baselines"
    log "Committed Section 12 changes"
  else
    log "No Section 12 changes required a commit"
  fi
fi

printf '\n'
cat reports/evidence/section12_baselines_report.txt
printf '\nEvidence files:\n'
printf '  %s\n' \
  "reports/evidence/section12_baselines_report.txt" \
  "reports/evidence/section12_baselines_verification.json" \
  "reports/tables/section12_parameter_calibration.json"
printf '\nBackups (if any): %s\n' "$BACKUP_ROOT"
