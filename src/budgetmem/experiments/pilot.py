"""Configuration-driven Section 15 pilot experiment for BudgetMem-R.

The pilot trains on sequence length 256 and evaluates the same checkpoint at
256, 512, and 1,024. All execution is CPU-only and all research decisions are
written to machine-readable artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import random
import resource
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import psutil
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
IGNORE_INDEX = -100


@dataclass
class PilotBatch:
    input_ids: Tensor
    target_ids: Tensor
    relevant_positions: list[list[int]]
    query_positions: list[list[int]]
    sample_ids: list[str]


@dataclass
class PilotOutput:
    logits: Tensor
    memory_sizes: Tensor | None = None
    write_probabilities: Tensor | None = None
    hard_writes: Tensor | None = None
    eviction_flags: Tensor | None = None
    retained_positions: list[list[int]] = field(default_factory=list)
    retention_ages: list[list[float]] = field(default_factory=list)
    write_counts: Tensor | None = None
    eviction_counts: Tensor | None = None


@dataclass
class TrainingRecord:
    task: str
    model: str
    model_source: str
    checkpoint_path: str
    config_sha256: str
    first_loss: float
    final_loss: float
    maximum_gradient_norm: float
    finite_losses: bool
    stability_pass: bool
    checkpoint_resume_pass: bool
    checkpoint_epoch: int
    train_wall_seconds: float
    train_peak_rss_mb: float
    parameter_count: int


class ResourceMonitor:
    """Sample process RSS while measuring wall and CPU time."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = max(float(interval_seconds), 0.001)
        self.process = psutil.Process(os.getpid())
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.wall_start = 0.0
        self.cpu_start: Any = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_rss = max(
                    self.peak_rss,
                    int(self.process.memory_info().rss),
                )
            except psutil.Error:
                pass
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "ResourceMonitor":
        self.wall_start = time.perf_counter()
        self.cpu_start = self.process.cpu_times()
        self.peak_rss = int(self.process.memory_info().rss)
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.wall_seconds = max(time.perf_counter() - self.wall_start, 0.0)
        cpu_end = self.process.cpu_times()
        self.cpu_user_seconds = max(cpu_end.user - self.cpu_start.user, 0.0)
        self.cpu_system_seconds = max(
            cpu_end.system - self.cpu_start.system,
            0.0,
        )
        linux_peak_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        self.peak_rss_mb = max(
            self.peak_rss / (1024.0**2),
            linux_peak_kb / 1024.0,
        )


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return payload


def load_generator_module() -> ModuleType:
    path = REPO_ROOT / "scripts" / "data" / "generate_synthetic.py"
    if not path.exists():
        raise FileNotFoundError(
            f"The Section 9 synthetic generator is required: {path}"
        )
    spec = importlib.util.spec_from_file_location(
        "_budgetmem_pilot_synthetic_generator",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "GENERATORS"):
        raise AttributeError(f"{path} does not expose GENERATORS")
    return module


class SyntheticPilotDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        task: str,
        sequence_length: int,
        sample_count: int,
        seed: int,
        max_target_length: int,
        vocabulary_size: int,
    ) -> None:
        super().__init__()
        module = load_generator_module()
        data_config = read_yaml(REPO_ROOT / "configs" / "data" / "synthetic.yaml")
        task_config = dict(data_config["tasks"][task])
        task_config["sequence_length"] = int(sequence_length)
        task_config["vocabulary_size"] = max(
            int(task_config["vocabulary_size"]),
            int(vocabulary_size),
        )
        generator = module.GENERATORS[task]

        task_seed = stable_int(task) % 1_000_000
        rows: list[dict[str, Any]] = []
        for index in range(int(sample_count)):
            example_seed = int(seed) * 1_000_003 + task_seed + index
            rng = np.random.default_rng(example_seed)
            generated = generator(task_config, rng)
            if len(generated) != 5:
                raise ValueError(
                    f"Generator {task} returned {len(generated)} values, expected 5"
                )
            input_ids, target_ids, relevant, query, _metadata = generated
            if len(input_ids) != sequence_length:
                raise AssertionError(
                    f"{task} produced length {len(input_ids)}, expected {sequence_length}"
                )
            if len(target_ids) > max_target_length:
                raise ValueError(
                    f"Target length {len(target_ids)} exceeds max_target_length="
                    f"{max_target_length}"
                )
            if max(input_ids, default=0) >= vocabulary_size:
                raise ValueError(
                    f"Input token exceeds vocabulary_size={vocabulary_size}"
                )
            if max(target_ids, default=0) >= vocabulary_size:
                raise ValueError(
                    f"Target token exceeds vocabulary_size={vocabulary_size}"
                )
            padded = [IGNORE_INDEX] * max_target_length
            padded[: len(target_ids)] = list(map(int, target_ids))
            rows.append(
                {
                    "sample_id": f"{task}-{sequence_length}-{index:06d}",
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "target_ids": torch.tensor(padded, dtype=torch.long),
                    "relevant_positions": list(map(int, relevant)),
                    "query_positions": list(map(int, query)),
                }
            )
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate_batch(rows: Sequence[dict[str, Any]]) -> PilotBatch:
    return PilotBatch(
        input_ids=torch.stack([row["input_ids"] for row in rows]),
        target_ids=torch.stack([row["target_ids"] for row in rows]),
        relevant_positions=[row["relevant_positions"] for row in rows],
        query_positions=[row["query_positions"] for row in rows],
        sample_ids=[str(row["sample_id"]) for row in rows],
    )


class GRUPilotModel(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        self.vocab_size = int(model_cfg["vocabulary_size"])
        self.max_target_length = int(model_cfg["max_target_length"])
        embedding_dim = int(model_cfg["embedding_dim"])
        hidden_dim = int(model_cfg["hidden_dim"])
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
        )
        self.decoder = nn.Linear(
            hidden_dim,
            self.max_target_length * self.vocab_size,
        )

    def forward(self, input_ids: Tensor, *, budget: int) -> PilotOutput:
        del budget
        embedded = self.embedding(input_ids)
        outputs, _hidden = self.gru(embedded)
        logits = self.decoder(outputs[:, -1]).reshape(
            input_ids.shape[0],
            self.max_target_length,
            self.vocab_size,
        )
        return PilotOutput(logits=logits)


class CachedGRUPilotModel(nn.Module):
    def __init__(self, cfg: Mapping[str, Any], *, policy: str, seed: int) -> None:
        super().__init__()
        if policy not in {"uniform", "reservoir"}:
            raise ValueError(policy)
        model_cfg = cfg["model"]
        matrix_cfg = cfg["matrix"]
        self.policy = policy
        self.seed = int(seed)
        self.vocab_size = int(model_cfg["vocabulary_size"])
        self.max_target_length = int(model_cfg["max_target_length"])
        self.max_budget = max(map(int, matrix_cfg["memory_budgets"]))
        embedding_dim = int(model_cfg["embedding_dim"])
        hidden_dim = int(model_cfg["hidden_dim"])
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.cell = nn.GRUCell(embedding_dim, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder = nn.Linear(
            hidden_dim,
            self.max_target_length * self.vocab_size,
        )

    @staticmethod
    def _reservoir_slot(seed: int, sample: int, step: int) -> int:
        value = (
            (seed + 1) * 1_103_515_245
            + (sample + 1) * 12_345
            + (step + 1) * 2_654_435_761
        ) & 0x7FFFFFFF
        return int(value % (step + 1))

    @staticmethod
    def _uniform_schedule(length: int, budget: int) -> dict[int, int]:
        positions = np.linspace(0, length - 1, num=budget, dtype=int).tolist()
        unique: list[int] = []
        for position in positions:
            position = int(position)
            if position not in unique:
                unique.append(position)
        cursor = 0
        while len(unique) < budget and cursor < length:
            if cursor not in unique:
                unique.append(cursor)
            cursor += 1
        unique = sorted(unique[:budget])
        return {position: slot for slot, position in enumerate(unique)}

    def forward(self, input_ids: Tensor, *, budget: int) -> PilotOutput:
        if not 1 <= int(budget) <= self.max_budget:
            raise ValueError(f"budget must be in [1, {self.max_budget}]")
        budget = int(budget)
        embedded = self.embedding(input_ids)
        batch_size, sequence_length, _ = embedded.shape
        device = embedded.device
        dtype = embedded.dtype

        hidden = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        cache = torch.zeros(
            batch_size,
            budget,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )
        valid = torch.zeros(batch_size, budget, device=device, dtype=torch.bool)
        positions = torch.full(
            (batch_size, budget),
            -1,
            device=device,
            dtype=torch.long,
        )
        write_counts = torch.zeros(batch_size, device=device, dtype=torch.long)
        eviction_counts = torch.zeros(batch_size, device=device, dtype=torch.long)
        memory_sizes: list[Tensor] = []
        hard_writes: list[Tensor] = []
        eviction_flags: list[Tensor] = []
        schedule = self._uniform_schedule(sequence_length, budget)
        fused = hidden

        for step in range(sequence_length):
            hidden = self.cell(embedded[:, step], hidden)
            scores = torch.einsum("bh,bsh->bs", hidden, cache) / math.sqrt(
                float(self.hidden_dim)
            )
            scores = scores.masked_fill(~valid, -1.0e9)
            weights = torch.softmax(scores, dim=1)
            weights = torch.where(valid, weights, torch.zeros_like(weights))
            normalizer = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
            weights = weights / normalizer
            context = torch.einsum("bs,bsh->bh", weights, cache)
            fused = torch.tanh(self.fusion(torch.cat([hidden, context], dim=-1)))

            wrote = torch.zeros(batch_size, device=device, dtype=torch.bool)
            evicted = torch.zeros(batch_size, device=device, dtype=torch.bool)
            if self.policy == "uniform":
                if step in schedule:
                    slot = int(schedule[step])
                    evicted = valid[:, slot].clone()
                    cache = cache.clone()
                    positions = positions.clone()
                    valid = valid.clone()
                    cache[:, slot] = hidden
                    positions[:, slot] = step
                    valid[:, slot] = True
                    wrote[:] = True
            else:
                if step < budget:
                    slot = step
                    cache = cache.clone()
                    positions = positions.clone()
                    valid = valid.clone()
                    cache[:, slot] = hidden
                    positions[:, slot] = step
                    valid[:, slot] = True
                    wrote[:] = True
                else:
                    selected_slots = torch.tensor(
                        [
                            self._reservoir_slot(self.seed, sample, step)
                            for sample in range(batch_size)
                        ],
                        device=device,
                        dtype=torch.long,
                    )
                    replace_mask = selected_slots < budget
                    for slot in range(budget):
                        mask = replace_mask & (selected_slots == slot)
                        if bool(mask.any()):
                            cache = cache.clone()
                            positions = positions.clone()
                            valid = valid.clone()
                            evicted[mask] = valid[mask, slot]
                            cache[mask, slot] = hidden[mask]
                            positions[mask, slot] = step
                            valid[mask, slot] = True
                            wrote[mask] = True

            write_counts += wrote.long()
            eviction_counts += evicted.long()
            hard_writes.append(wrote)
            eviction_flags.append(evicted)
            memory_sizes.append(valid.sum(dim=1))

        logits = self.decoder(fused).reshape(
            batch_size,
            self.max_target_length,
            self.vocab_size,
        )
        retained = [
            sorted(map(int, positions[row, valid[row]].detach().cpu().tolist()))
            for row in range(batch_size)
        ]
        ages = [
            [float(sequence_length - 1 - position) for position in row]
            for row in retained
        ]
        return PilotOutput(
            logits=logits,
            memory_sizes=torch.stack(memory_sizes, dim=1),
            hard_writes=torch.stack(hard_writes, dim=1),
            eviction_flags=torch.stack(eviction_flags, dim=1),
            retained_positions=retained,
            retention_ages=ages,
            write_counts=write_counts,
            eviction_counts=eviction_counts,
        )


class BudgetMemRAdapter(nn.Module):
    """Adapt the Section 13 BudgetMemR implementation to token tasks."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        matrix_cfg = cfg["matrix"]
        self.vocab_size = int(model_cfg["vocabulary_size"])
        self.embedding_dim = int(model_cfg["embedding_dim"])
        self.hidden_dim = int(model_cfg["hidden_dim"])
        self.key_dim = int(model_cfg["key_dim"])
        self.retrieval_k = int(model_cfg["retrieval_k"])
        self.max_target_length = int(model_cfg["max_target_length"])
        self.allowed_budgets = tuple(map(int, matrix_cfg["memory_budgets"]))
        self.max_budget = max(self.allowed_budgets)
        self.decoder_dim = self.max_target_length * self.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        module = importlib.import_module("budgetmem.models.budgetmem_r")
        budgetmem_class = getattr(module, "BudgetMemR")
        self.core_signature = str(inspect.signature(budgetmem_class))
        kwargs = self._constructor_kwargs(budgetmem_class)
        self.core = budgetmem_class(**kwargs)

    def _constructor_kwargs(self, budgetmem_class: type[nn.Module]) -> dict[str, Any]:
        signature = inspect.signature(budgetmem_class)
        values: dict[str, Any] = {
            "input_dim": self.embedding_dim,
            "input_size": self.embedding_dim,
            "feature_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_size": self.hidden_dim,
            "state_dim": self.hidden_dim,
            "output_dim": self.decoder_dim,
            "output_size": self.decoder_dim,
            "num_classes": self.decoder_dim,
            "max_budget": self.max_budget,
            "allowed_budgets": self.allowed_budgets,
            "memory_budget": self.max_budget,
            "memory_size": self.max_budget,
            "capacity": self.max_budget,
            "key_dim": self.key_dim,
            "value_dim": self.hidden_dim,
            "budget_embedding_dim": 8,
            "budget_dim": 8,
            "controller_hidden_dim": self.hidden_dim,
            "retrieval_k": min(self.retrieval_k, self.max_budget),
            "top_k": min(self.retrieval_k, self.max_budget),
            "read_k": min(self.retrieval_k, self.max_budget),
            "recurrent_type": "gru",
            "cell_type": "gru",
            "fusion": "gated",
            "fusion_type": "gated",
            "write_threshold": 0.5,
            "temperature": 1.0,
            "dropout": 0.0,
            "num_layers": 1,
            "detach_memory": False,
        }
        kwargs: dict[str, Any] = {}
        unresolved: list[str] = []
        for name, parameter in signature.parameters.items():
            if name in {"self", "args", "kwargs"}:
                continue
            if name in values:
                kwargs[name] = values[name]
            elif parameter.default is inspect.Parameter.empty:
                unresolved.append(name)
        if unresolved:
            raise TypeError(
                "Cannot construct BudgetMemR. Unmapped required parameters: "
                f"{unresolved}. Signature: {signature}"
            )
        return kwargs

    @staticmethod
    def _output_tensor(output: Any) -> Tensor:
        if isinstance(output, Tensor):
            return output
        if hasattr(output, "logits"):
            return output.logits
        if (
            isinstance(output, (tuple, list))
            and output
            and isinstance(output[0], Tensor)
        ):
            return output[0]
        raise TypeError("BudgetMemR output does not expose logits")

    def _call_core(self, embedded: Tensor, budget: int) -> Any:
        signature = inspect.signature(self.core.forward)
        params = signature.parameters
        budget_tensor = torch.full(
            (embedded.shape[0],),
            int(budget),
            device=embedded.device,
            dtype=torch.long,
        )
        kwargs: dict[str, Any] = {}
        if "budget" in params:
            kwargs["budget"] = budget_tensor
        elif "budgets" in params:
            kwargs["budgets"] = budget_tensor
        elif "memory_budget" in params:
            kwargs["memory_budget"] = budget_tensor
        elif "memory_budgets" in params:
            kwargs["memory_budgets"] = budget_tensor
        else:
            raise TypeError(
                "BudgetMemR.forward must accept budget information. Signature: "
                f"{signature}"
            )
        return self.core(embedded, **kwargs)

    def forward(self, input_ids: Tensor, *, budget: int) -> PilotOutput:
        embedded = self.embedding(input_ids)
        raw = self._call_core(embedded, int(budget))
        raw_logits = self._output_tensor(raw)
        if raw_logits.ndim == 3:
            final_logits = raw_logits[:, -1]
        elif raw_logits.ndim == 2:
            final_logits = raw_logits
        else:
            raise ValueError(
                f"Unsupported BudgetMemR logits shape: {tuple(raw_logits.shape)}"
            )
        if final_logits.shape[-1] != self.decoder_dim:
            raise ValueError(
                "BudgetMemR output dimension mismatch. Expected "
                f"{self.decoder_dim}, received {final_logits.shape[-1]}. "
                f"Constructor signature: {self.core_signature}"
            )
        logits = final_logits.reshape(
            input_ids.shape[0],
            self.max_target_length,
            self.vocab_size,
        )

        memory_sizes = getattr(raw, "memory_sizes", None)
        write_probabilities = getattr(raw, "write_probabilities", None)
        hard_writes = getattr(raw, "hard_writes", None)
        eviction_flags = getattr(raw, "eviction_flags", None)
        final_memory = getattr(raw, "final_memory", None)
        if memory_sizes is None or hard_writes is None or final_memory is None:
            raise AttributeError(
                "BudgetMemROutput must expose memory_sizes, hard_writes, and "
                "final_memory for the Section 15 diagnostics."
            )

        valid = final_memory.valid
        ages_tensor = final_memory.age
        sequence_length = int(input_ids.shape[1])
        retained: list[list[int]] = []
        ages: list[list[float]] = []
        for row in range(input_ids.shape[0]):
            row_ages = ages_tensor[row, valid[row]].detach().cpu().tolist()
            row_positions = [
                max(0, min(sequence_length - 1, sequence_length - 1 - int(round(age))))
                for age in row_ages
            ]
            retained.append(sorted(row_positions))
            ages.append([float(age) for age in row_ages])

        write_counts = hard_writes.long().sum(dim=1)
        if eviction_flags is None:
            eviction_counts = torch.zeros_like(write_counts)
        else:
            eviction_counts = eviction_flags.long().sum(dim=1)
        return PilotOutput(
            logits=logits,
            memory_sizes=memory_sizes,
            write_probabilities=write_probabilities,
            hard_writes=hard_writes,
            eviction_flags=eviction_flags,
            retained_positions=retained,
            retention_ages=ages,
            write_counts=write_counts,
            eviction_counts=eviction_counts,
        )


def build_model(
    model_name: str,
    cfg: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[nn.Module, str]:
    torch.manual_seed(seed)
    if model_name == "gru":
        return GRUPilotModel(cfg), "pilot_standard_gru"
    if model_name == "gru_uniform_cache":
        return (
            CachedGRUPilotModel(cfg, policy="uniform", seed=seed),
            "pilot_uniform_checkpoint_cache",
        )
    if model_name == "gru_reservoir_cache":
        return (
            CachedGRUPilotModel(cfg, policy="reservoir", seed=seed),
            "pilot_reservoir_cache",
        )
    if model_name == "budgetmem_r":
        return BudgetMemRAdapter(cfg), "src/budgetmem/models/budgetmem_r.py"
    raise KeyError(model_name)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def task_loss(logits: Tensor, target_ids: Tensor) -> Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def total_training_loss(
    output: PilotOutput,
    target_ids: Tensor,
    *,
    model_name: str,
    budget: int,
    cfg: Mapping[str, Any],
) -> Tensor:
    loss = task_loss(output.logits, target_ids)
    if model_name != "budgetmem_r":
        return loss
    training_cfg = cfg["training"]
    if output.write_probabilities is not None:
        write_rate = output.write_probabilities.float().mean()
        target_rate = float(training_cfg["write_rate_target"])
        loss = (
            loss
            + float(training_cfg["write_rate_penalty"])
            * (write_rate - target_rate).square()
        )
    if output.memory_sizes is not None:
        overflow = torch.relu(output.memory_sizes.float() - float(budget)).mean()
        loss = loss + float(training_cfg["budget_violation_penalty"]) * overflow
    return loss


def make_loader(
    *,
    cfg: Mapping[str, Any],
    task: str,
    sequence_length: int,
    sample_count: int,
    seed: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[PilotBatch]:
    dataset = SyntheticPilotDataset(
        task=task,
        sequence_length=sequence_length,
        sample_count=sample_count,
        seed=seed,
        max_target_length=int(cfg["model"]["max_target_length"]),
        vocabulary_size=int(cfg["model"]["vocabulary_size"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(cfg["training"]["num_workers"]),
        collate_fn=collate_batch,
        generator=generator,
        drop_last=False,
    )


def _move_batch(batch: PilotBatch, device: torch.device) -> PilotBatch:
    return PilotBatch(
        input_ids=batch.input_ids.to(device),
        target_ids=batch.target_ids.to(device),
        relevant_positions=batch.relevant_positions,
        query_positions=batch.query_positions,
        sample_ids=batch.sample_ids,
    )


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[float],
    cfg: Mapping[str, Any],
    config_sha256: str,
    task: str,
    model_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "task": task,
        "model_name": model_name,
        "epoch": int(epoch),
        "history": list(map(float, history)),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "config": dict(cfg),
        "config_sha256": config_sha256,
    }


def validate_checkpoint_resume(
    *,
    checkpoint_path: Path,
    model_name: str,
    cfg: Mapping[str, Any],
    seed: int,
    batch: PilotBatch,
    device: torch.device,
) -> bool:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restored, _source = build_model(model_name, cfg, seed=seed)
    restored.to(device)
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    restored.eval()

    optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if int(payload["epoch"]) < 0:
        return False

    batch = _move_batch(batch, device)
    budget = min(map(int, cfg["matrix"]["memory_budgets"]))
    original, _source = build_model(model_name, cfg, seed=seed)
    original.to(device)
    original.load_state_dict(payload["model_state_dict"], strict=True)
    original.eval()
    with torch.no_grad():
        reference = original(batch.input_ids, budget=budget).logits
        resumed = restored(batch.input_ids, budget=budget).logits
    return bool(torch.allclose(reference, resumed, atol=0.0, rtol=0.0))


def train_one_model(
    *,
    cfg: Mapping[str, Any],
    config_sha256: str,
    task: str,
    model_name: str,
    seed: int,
    resume: bool,
) -> tuple[nn.Module, TrainingRecord]:
    device = torch.device("cpu")
    training_cfg = cfg["training"]
    artifacts_cfg = cfg["artifacts"]
    train_length = int(cfg["matrix"]["train_sequence_length"])
    loader = make_loader(
        cfg=cfg,
        task=task,
        sequence_length=train_length,
        sample_count=int(training_cfg["train_samples"]),
        seed=seed + 11,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
    )
    validation_loader = make_loader(
        cfg=cfg,
        task=task,
        sequence_length=train_length,
        sample_count=max(8, min(16, int(training_cfg["validation_samples"]))),
        seed=seed + 29,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
    )
    validation_batch = next(iter(validation_loader))

    model, source = build_model(model_name, cfg, seed=seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    checkpoint_path = (
        REPO_ROOT
        / str(artifacts_cfg["checkpoint_root"])
        / task
        / model_name
        / "last.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[float] = []
    start_epoch = 0

    if resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload.get("config_sha256") != config_sha256:
            raise RuntimeError(
                f"Refusing to resume {checkpoint_path}: configuration hash changed"
            )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        history = list(map(float, payload.get("history", [])))
        start_epoch = int(payload["epoch"]) + 1

    budgets = list(map(int, cfg["matrix"]["memory_budgets"]))
    finite_losses = True
    maximum_gradient_norm = 0.0
    with ResourceMonitor(
        float(cfg["evaluation"]["resource_sample_interval_seconds"])
    ) as monitor:
        for epoch in range(start_epoch, int(training_cfg["epochs"])):
            model.train()
            epoch_losses: list[float] = []
            for batch_index, batch in enumerate(loader):
                batch = _move_batch(batch, device)
                budget = budgets[(epoch + batch_index) % len(budgets)]
                optimizer.zero_grad(set_to_none=True)
                output = model(batch.input_ids, budget=budget)
                loss = total_training_loss(
                    output,
                    batch.target_ids,
                    model_name=model_name,
                    budget=budget,
                    cfg=cfg,
                )
                if not bool(torch.isfinite(loss)):
                    finite_losses = False
                    raise FloatingPointError(
                        f"Non-finite loss for {task}/{model_name} at epoch {epoch}"
                    )
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(training_cfg["gradient_clip_norm"]),
                    error_if_nonfinite=True,
                )
                maximum_gradient_norm = max(
                    maximum_gradient_norm,
                    float(grad_norm.detach().cpu().item()),
                )
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            history.append(float(np.mean(epoch_losses)))
            torch.save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    history=history,
                    cfg=cfg,
                    config_sha256=config_sha256,
                    task=task,
                    model_name=model_name,
                ),
                checkpoint_path,
            )

    if not history:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        history = list(map(float, payload["history"]))
        model.load_state_dict(payload["model_state_dict"], strict=True)

    first_loss = float(history[0])
    final_loss = float(history[-1])
    maximum_allowed = float(training_cfg["maximum_acceptable_gradient_norm"])
    stability_pass = bool(
        finite_losses
        and math.isfinite(first_loss)
        and math.isfinite(final_loss)
        and maximum_gradient_norm <= maximum_allowed
        and final_loss <= max(first_loss * 2.0, first_loss + 1.0)
    )
    resume_pass = validate_checkpoint_resume(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        cfg=cfg,
        seed=seed,
        batch=validation_batch,
        device=device,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    record = TrainingRecord(
        task=task,
        model=model_name,
        model_source=source,
        checkpoint_path=str(checkpoint_path.relative_to(REPO_ROOT)),
        config_sha256=config_sha256,
        first_loss=first_loss,
        final_loss=final_loss,
        maximum_gradient_norm=maximum_gradient_norm,
        finite_losses=finite_losses,
        stability_pass=stability_pass,
        checkpoint_resume_pass=resume_pass,
        checkpoint_epoch=int(payload["epoch"]),
        train_wall_seconds=float(monitor.wall_seconds),
        train_peak_rss_mb=float(monitor.peak_rss_mb),
        parameter_count=parameter_count(model),
    )
    return model, record


def _safe_mean(values: Iterable[float]) -> float:
    data = list(values)
    return float(np.mean(data)) if data else 0.0


def evaluate_model(
    *,
    cfg: Mapping[str, Any],
    config_path: Path,
    config_sha256: str,
    task: str,
    sequence_length: int,
    budget: int,
    model_name: str,
    model: nn.Module,
    training_record: TrainingRecord,
    seed: int,
) -> dict[str, Any]:
    device = torch.device("cpu")
    loader = make_loader(
        cfg=cfg,
        task=task,
        sequence_length=sequence_length,
        sample_count=int(cfg["training"]["validation_samples"]),
        seed=seed + 101 + sequence_length + budget,
        batch_size=int(cfg["evaluation"]["batch_size"]),
        shuffle=False,
    )
    model.eval()
    losses: list[float] = []
    correct_tokens = 0
    total_tokens = 0
    exact_matches = 0
    sample_count = 0
    relevant_retention: list[float] = []
    irrelevant_retention: list[float] = []
    memory_precision: list[float] = []
    recent_overlap: list[float] = []
    retention_ages: list[float] = []
    write_count = 0
    eviction_count = 0
    external_steps = 0
    max_memory_size = 0
    successful_long_range_retrievals = 0
    token_count = 0

    with ResourceMonitor(
        float(cfg["evaluation"]["resource_sample_interval_seconds"])
    ) as monitor:
        with torch.no_grad():
            for batch in loader:
                batch = _move_batch(batch, device)
                output = model(batch.input_ids, budget=int(budget))
                loss = task_loss(output.logits, batch.target_ids)
                losses.append(float(loss.detach().cpu().item()))
                predictions = output.logits.argmax(dim=-1)
                valid_targets = batch.target_ids != IGNORE_INDEX
                matches = (predictions == batch.target_ids) & valid_targets
                correct_tokens += int(matches.sum().item())
                total_tokens += int(valid_targets.sum().item())
                per_sample_exact = (matches | ~valid_targets).all(
                    dim=1
                ) & valid_targets.any(dim=1)
                exact_matches += int(per_sample_exact.sum().item())
                sample_count += int(batch.input_ids.shape[0])
                token_count += int(batch.input_ids.numel())

                if output.memory_sizes is not None:
                    max_memory_size = max(
                        max_memory_size,
                        int(output.memory_sizes.max().detach().cpu().item()),
                    )
                    external_steps += int(output.memory_sizes.numel())
                if output.write_counts is not None:
                    write_count += int(output.write_counts.sum().detach().cpu().item())
                elif output.hard_writes is not None:
                    write_count += int(output.hard_writes.sum().detach().cpu().item())
                if output.eviction_counts is not None:
                    eviction_count += int(
                        output.eviction_counts.sum().detach().cpu().item()
                    )
                elif output.eviction_flags is not None:
                    eviction_count += int(
                        output.eviction_flags.sum().detach().cpu().item()
                    )

                if output.retained_positions:
                    for row, retained_list in enumerate(output.retained_positions):
                        retained = set(map(int, retained_list))
                        relevant = set(map(int, batch.relevant_positions[row]))
                        relevant_hits = len(retained & relevant)
                        relevant_retention.append(relevant_hits / max(1, len(relevant)))
                        irrelevant_hits = len(retained - relevant)
                        irrelevant_total = max(1, sequence_length - len(relevant))
                        irrelevant_retention.append(irrelevant_hits / irrelevant_total)
                        memory_precision.append(relevant_hits / max(1, len(retained)))
                        recent = set(
                            range(max(0, sequence_length - budget), sequence_length)
                        )
                        recent_overlap.append(
                            len(retained & recent) / max(1, len(retained))
                        )
                        if row < len(output.retention_ages):
                            retention_ages.extend(output.retention_ages[row])
                        query_positions = batch.query_positions[row]
                        long_range = bool(
                            relevant
                            and query_positions
                            and min(query_positions) - min(relevant)
                            >= sequence_length // 2
                        )
                        if long_range and bool(per_sample_exact[row].item()):
                            successful_long_range_retrievals += 1

    wall_seconds = float(monitor.wall_seconds)
    tokens_per_second = token_count / max(wall_seconds, 1.0e-12)
    write_frequency = (
        write_count / max(1, external_steps) if model_name != "gru" else 0.0
    )
    expected_random_retention = min(1.0, float(budget) / float(sequence_length))
    resource_measurement_pass = bool(
        wall_seconds > 0.0
        and monitor.peak_rss_mb > 0.0
        and tokens_per_second > 0.0
        and monitor.cpu_user_seconds >= 0.0
    )

    return {
        "schema_version": "1.0",
        "experiment_name": cfg["experiment_name"],
        "task": task,
        "model": model_name,
        "model_source": training_record.model_source,
        "sequence_length": int(sequence_length),
        "memory_budget": int(budget),
        "seed": int(cfg["seed"]),
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": config_sha256,
        "checkpoint_path": training_record.checkpoint_path,
        "checkpoint_resume_pass": training_record.checkpoint_resume_pass,
        "stability_pass": training_record.stability_pass,
        "mean_loss": _safe_mean(losses),
        "token_accuracy": correct_tokens / max(1, total_tokens),
        "exact_match_accuracy": exact_matches / max(1, sample_count),
        "relevant_state_retention_rate": _safe_mean(relevant_retention),
        "irrelevant_state_retention_rate": _safe_mean(irrelevant_retention),
        "memory_precision": _safe_mean(memory_precision),
        "memory_recall": _safe_mean(relevant_retention),
        "average_retention_duration": _safe_mean(retention_ages),
        "successful_long_range_retrievals": successful_long_range_retrievals,
        "cache_turnover": int(eviction_count),
        "eviction_errors": int(
            round(sum(max(0.0, 1.0 - value) for value in relevant_retention))
        ),
        "write_frequency": float(write_frequency),
        "recent_state_overlap": _safe_mean(recent_overlap),
        "expected_random_retention": expected_random_retention,
        "max_memory_size": int(max_memory_size),
        "budget_pass": bool(max_memory_size <= budget),
        "wall_seconds": wall_seconds,
        "cpu_user_seconds": float(monitor.cpu_user_seconds),
        "cpu_system_seconds": float(monitor.cpu_system_seconds),
        "peak_rss_mb": float(monitor.peak_rss_mb),
        "tokens_per_second": float(tokens_per_second),
        "parameter_count": int(training_record.parameter_count),
        "resource_measurement_pass": resource_measurement_pass,
        "train_first_loss": training_record.first_loss,
        "train_final_loss": training_record.final_loss,
        "maximum_gradient_norm": training_record.maximum_gradient_norm,
    }


def summarize_go_no_go(
    rows: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    *,
    smoke: bool,
) -> dict[str, Any]:
    evaluation_cfg = cfg["evaluation"]
    long_length = int(evaluation_cfg["long_range_sequence_length"])
    long_rows = [row for row in rows if int(row["sequence_length"]) == long_length]

    def model_mean(model: str, metric: str) -> float:
        return _safe_mean(
            float(row[metric]) for row in long_rows if row["model"] == model
        )

    budgetmem_accuracy = model_mean("budgetmem_r", "token_accuracy")
    policy_scores = {
        "gru_uniform_cache": model_mean("gru_uniform_cache", "token_accuracy"),
        "gru_reservoir_cache": model_mean("gru_reservoir_cache", "token_accuracy"),
    }
    minimum_gain = float(evaluation_cfg["minimum_clear_accuracy_gain"])
    gains = {
        policy: budgetmem_accuracy - score for policy, score in policy_scores.items()
    }
    policies_clearly_outperformed = [
        policy for policy, gain in gains.items() if gain >= minimum_gain
    ]

    budget_rows = [row for row in rows if row["model"] != "gru"]
    budgetmem_rows = [row for row in rows if row["model"] == "budgetmem_r"]
    budgetmem_long_rows = [
        row for row in budgetmem_rows if int(row["sequence_length"]) == long_length
    ]
    write_frequency = _safe_mean(
        float(row["write_frequency"]) for row in budgetmem_rows
    )
    recent_overlap = _safe_mean(
        float(row["recent_state_overlap"]) for row in budgetmem_long_rows
    )
    retention_advantage = _safe_mean(
        float(row["relevant_state_retention_rate"])
        - float(row["expected_random_retention"])
        for row in budgetmem_long_rows
    )

    criteria = {
        "training_stability": all(bool(row["stability_pass"]) for row in rows),
        "strict_memory_budget": all(bool(row["budget_pass"]) for row in budget_rows),
        "nontrivial_controller_writes": (
            float(evaluation_cfg["nontrivial_write_frequency_min"])
            <= write_frequency
            <= float(evaluation_cfg["nontrivial_write_frequency_max"])
        ),
        "not_recent_state_copying": recent_overlap
        < float(evaluation_cfg["recent_overlap_failure_threshold"]),
        "retention_exceeds_random": retention_advantage
        >= float(evaluation_cfg["random_retention_margin"]),
        "resource_measurements_valid": all(
            bool(row["resource_measurement_pass"]) for row in rows
        ),
        "checkpoint_resumption": all(
            bool(row["checkpoint_resume_pass"]) for row in rows
        ),
        "configuration_provenance": len({str(row["config_sha256"]) for row in rows})
        == 1,
        "outperforms_two_memory_policies": len(policies_clearly_outperformed) >= 2,
    }
    if smoke:
        status = "SMOKE_ONLY"
    else:
        status = "GO" if all(criteria.values()) else "NO_GO"
    return {
        "schema_version": "1.0",
        "status": status,
        "decision": (
            "Proceed to the full experiment matrix."
            if status == "GO"
            else (
                "Smoke validation only; do not use this result for the pilot decision."
                if status == "SMOKE_ONLY"
                else "Do not begin the full experiment matrix."
            )
        ),
        "criteria": criteria,
        "long_range_sequence_length": long_length,
        "long_range_token_accuracy": {
            "budgetmem_r": budgetmem_accuracy,
            **policy_scores,
        },
        "accuracy_gains": gains,
        "minimum_clear_accuracy_gain": minimum_gain,
        "policies_clearly_outperformed": policies_clearly_outperformed,
        "budgetmem_mean_write_frequency": write_frequency,
        "budgetmem_mean_recent_state_overlap": recent_overlap,
        "budgetmem_mean_retention_advantage_over_random": retention_advantage,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No pilot rows were produced")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    training_records: Sequence[TrainingRecord],
    gate: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Section 15 Pilot Experiment Report",
        "",
        f"**Decision:** `{gate['status']}`",
        "",
        str(gate["decision"]),
        "",
        "## Go/No-Go Criteria",
        "",
        "| Criterion | Result |",
        "|---|---:|",
    ]
    for criterion, value in gate["criteria"].items():
        lines.append(f"| {criterion} | {'PASS' if value else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Long-Range Accuracy",
            "",
            "| Model | Token accuracy at length 1,024 |",
            "|---|---:|",
        ]
    )
    for model, value in gate["long_range_token_accuracy"].items():
        lines.append(f"| {model} | {float(value):.6f} |")
    lines.extend(
        [
            "",
            "## Training Records",
            "",
            "| Task | Model | First loss | Final loss | Stable | Resume |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for record in training_records:
        lines.append(
            f"| {record.task} | {record.model} | {record.first_loss:.6f} | "
            f"{record.final_loss:.6f} | {record.stability_pass} | "
            f"{record.checkpoint_resume_pass} |"
        )
    lines.extend(
        [
            "",
            "## Evaluation Matrix",
            "",
            "| Task | Model | Length | Budget | Token accuracy | Memory recall | Write frequency | Budget pass |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['model']} | {row['sequence_length']} | "
            f"{row['memory_budget']} | {float(row['token_accuracy']):.6f} | "
            f"{float(row['memory_recall']):.6f} | "
            f"{float(row['write_frequency']):.6f} | {row['budget_pass']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_pretraining_gate(cfg: Mapping[str, Any]) -> dict[str, Any]:
    gate_cfg = cfg["pretraining_gate"]
    path = REPO_ROOT / str(gate_cfg["report"])
    if not path.exists():
        raise FileNotFoundError(
            f"Section 14 gate report is missing: {path}. Complete Section 14 first."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    status = (
        report.get("status")
        or report.get("overall_status")
        or report.get("gate_status")
    )
    required = str(gate_cfg["required_status"])
    if str(status).upper() != required.upper():
        raise RuntimeError(
            f"Section 14 gate status is {status!r}; required status is {required!r}."
        )
    return report


def validate_config(cfg: Mapping[str, Any]) -> None:
    expected_tasks = {
        "selective_copy",
        "associative_recall",
        "distractor_heavy_retrieval",
    }
    if set(cfg["matrix"]["tasks"]) != expected_tasks:
        raise ValueError("Section 15 requires exactly the three pilot tasks")
    if list(map(int, cfg["matrix"]["evaluation_sequence_lengths"])) != [
        256,
        512,
        1024,
    ]:
        raise ValueError("Section 15 requires lengths 256, 512, and 1024")
    if list(map(int, cfg["matrix"]["memory_budgets"])) != [16, 32]:
        raise ValueError("Section 15 requires memory budgets 16 and 32")
    expected_models = {
        "gru",
        "gru_uniform_cache",
        "gru_reservoir_cache",
        "budgetmem_r",
    }
    if set(cfg["matrix"]["models"]) != expected_models:
        raise ValueError("Section 15 model matrix is incomplete")
    if str(cfg["device"]).lower() != "cpu":
        raise ValueError("This laptop pilot must use device: cpu")


def smoke_config(cfg: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(cfg))
    copied["matrix"]["tasks"] = ["selective_copy"]
    copied["matrix"]["evaluation_sequence_lengths"] = [256]
    copied["matrix"]["memory_budgets"] = [16]
    copied["training"]["train_samples"] = 16
    copied["training"]["validation_samples"] = 8
    copied["training"]["epochs"] = 1
    copied["training"]["batch_size"] = 4
    copied["evaluation"]["batch_size"] = 4
    return copied


def run_pilot(
    config_path: Path,
    *,
    smoke: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    base_cfg = read_yaml(config_path)
    validate_config(base_cfg)
    validate_pretraining_gate(base_cfg)
    cfg = smoke_config(base_cfg) if smoke else base_cfg
    seed = int(cfg["seed"])
    seed_everything(seed)
    config_sha256 = sha256_file(config_path)

    output_root = REPO_ROOT / str(cfg["artifacts"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "effective_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )

    trained: dict[tuple[str, str], nn.Module] = {}
    training_records: list[TrainingRecord] = []
    for task in cfg["matrix"]["tasks"]:
        for model_index, model_name in enumerate(cfg["matrix"]["models"]):
            model_seed = seed + stable_int(f"{task}:{model_name}") % 1_000_000
            model, record = train_one_model(
                cfg=cfg,
                config_sha256=config_sha256,
                task=str(task),
                model_name=str(model_name),
                seed=model_seed,
                resume=resume,
            )
            trained[(str(task), str(model_name))] = model
            training_records.append(record)
            print(
                f"TRAINED task={task} model={model_name} "
                f"loss={record.final_loss:.6f} stable={record.stability_pass} "
                f"resume={record.checkpoint_resume_pass}",
                flush=True,
            )

    rows: list[dict[str, Any]] = []
    records_by_key = {
        (record.task, record.model): record for record in training_records
    }
    for task in cfg["matrix"]["tasks"]:
        for model_name in cfg["matrix"]["models"]:
            model = trained[(str(task), str(model_name))]
            record = records_by_key[(str(task), str(model_name))]
            for sequence_length in cfg["matrix"]["evaluation_sequence_lengths"]:
                for budget in cfg["matrix"]["memory_budgets"]:
                    row = evaluate_model(
                        cfg=cfg,
                        config_path=config_path,
                        config_sha256=config_sha256,
                        task=str(task),
                        sequence_length=int(sequence_length),
                        budget=int(budget),
                        model_name=str(model_name),
                        model=model,
                        training_record=record,
                        seed=seed,
                    )
                    rows.append(row)
                    print(
                        f"EVALUATED task={task} model={model_name} "
                        f"length={sequence_length} budget={budget} "
                        f"accuracy={row['token_accuracy']:.6f} "
                        f"budget_pass={row['budget_pass']}",
                        flush=True,
                    )

    gate = summarize_go_no_go(rows, cfg, smoke=smoke)
    artifacts = cfg["artifacts"]
    results_path = REPO_ROOT / str(artifacts["results_csv"])
    summary_path = REPO_ROOT / str(artifacts["summary_json"])
    gate_path = REPO_ROOT / str(artifacts["gate_json"])
    report_path = REPO_ROOT / str(artifacts["report_markdown"])
    write_csv(results_path, rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "config_path": str(config_path.relative_to(REPO_ROOT)),
                "config_sha256": config_sha256,
                "smoke": smoke,
                "training_records": [asdict(record) for record in training_records],
                "result_count": len(rows),
                "results_csv": str(results_path.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, rows, training_records, gate)
    return gate
