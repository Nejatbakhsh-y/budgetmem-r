#!/usr/bin/env python3
"""Dedicated single-cell runner for the Section 18 experiment matrix.

Backends
--------
1. The established Section 15 pilot backend is used for supported synthetic
   recurrent cells. This preserves the tested BudgetMem-R adapters, metrics,
   checkpointing, and strict memory-budget checks.
2. The controlled Section 12 baseline registry is used for LSTM, Transformer,
   Mamba/S4D, RMT, uniform/reservoir cache, and Memory Caching cells over
   prepared JSONL, Parquet, or CSV datasets.

The script executes exactly one configuration file per process. The matrix
orchestrator remains responsible for enumerating and resuming cells.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import inspect
import json
import shutil
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from budgetmem.baselines.controlled import build_baseline, parameter_count
from budgetmem.experiments.pilot import (
    evaluate_model,
    read_yaml,
    seed_everything,
    sha256_file,
    stable_int,
    train_one_model,
    write_csv,
)
from budgetmem.models.budgetmem_r import BudgetMemR


PILOT_MODELS = {
    "gru": "gru",
    "gru_uniform": "gru_uniform_cache",
    "gru_uniform_cache": "gru_uniform_cache",
    "uniform_cache": "gru_uniform_cache",
    "gru_reservoir": "gru_reservoir_cache",
    "gru_reservoir_cache": "gru_reservoir_cache",
    "reservoir_cache": "gru_reservoir_cache",
    "budgetmem_r": "budgetmem_r",
    "budgetmem-r": "budgetmem_r",
}

CONTROLLED_MODELS = {
    "gru": ("gru", {}),
    "lstm": ("lstm", {}),
    "transformer": ("transformer_sliding", {}),
    "transformer_full": ("transformer_full", {}),
    "mamba": ("state_space", {"backend": "s4d_reference"}),
    "state_space": ("state_space", {"backend": "s4d_reference"}),
    "rmt": ("recurrent_memory_transformer", {}),
    "recurrent_memory_transformer": ("recurrent_memory_transformer", {}),
    "gru_uniform": ("memory_caching_mean", {"policy": "uniform"}),
    "gru_uniform_cache": ("memory_caching_mean", {"policy": "uniform"}),
    "uniform_cache": ("memory_caching_mean", {"policy": "uniform"}),
    "gru_reservoir": ("memory_caching_mean", {"policy": "reservoir"}),
    "gru_reservoir_cache": ("memory_caching_mean", {"policy": "reservoir"}),
    "reservoir_cache": ("memory_caching_mean", {"policy": "reservoir"}),
    "memory_caching": ("memory_caching_gated", {"policy": "uniform"}),
    "memory_caching_recurrent": ("memory_caching_gated", {"policy": "uniform"}),
}

REQUIRED_SECTION18_ALIASES = {
    "gru",
    "lstm",
    "transformer",
    "mamba",
    "rmt",
    "gru_uniform",
    "gru_reservoir",
    "memory_caching",
    "budgetmem_r",
}

INPUT_COLUMNS = (
    "input_ids",
    "token_ids",
    "tokens",
    "sequence",
    "inputs",
    "input",
    "x",
    "features",
)
TARGET_COLUMNS = (
    "target_ids",
    "targets",
    "labels",
    "label",
    "target",
    "y",
    "anomaly",
    "is_anomaly",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def canonical_name(value: Any) -> str:
    name = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def first_value(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return default


def resolve_cell(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    task = canonical_name(
        first_value(
            nested(config, "task", "name"),
            config.get("task"),
            nested(config, "experiment", "task"),
            default="selective_copy",
        )
    )
    dataset = canonical_name(
        first_value(
            nested(config, "data", "dataset"),
            nested(config, "task", "dataset"),
            config.get("dataset"),
            default="synthetic",
        )
    )
    model = canonical_name(
        first_value(
            nested(config, "model", "name"),
            config.get("model"),
            default="gru",
        )
    )
    sequence_length = int(
        first_value(
            nested(config, "task", "sequence_length"),
            nested(config, "data", "sequence_length"),
            nested(config, "evaluation", "sequence_length"),
            config.get("sequence_length"),
            default=256,
        )
    )
    budget = int(
        first_value(
            nested(config, "memory", "budget"),
            config.get("memory_budget"),
            default=32,
        )
    )
    retrieval_k = int(
        first_value(
            nested(config, "memory", "retrieval_k"),
            config.get("retrieval_k"),
            default=4,
        )
    )
    seed = int(
        first_value(
            nested(config, "training", "seed"),
            nested(config, "data", "split_seed"),
            config.get("seed"),
            default=2026,
        )
    )
    run_id = str(
        first_value(
            nested(config, "experiment", "run_id"),
            config.get("run_id"),
            default=config_path.stem,
        )
    )
    output_dir = Path(
        str(
            first_value(
                nested(config, "experiment", "output_dir"),
                nested(config, "training", "output_dir"),
                config.get("output_dir"),
                default=f"artifacts/section18/runs/{run_id}",
            )
        )
    )
    if sequence_length <= 0 or budget <= 0 or retrieval_k <= 0:
        raise ValueError("sequence_length, budget, and retrieval_k must be positive")
    if retrieval_k > budget:
        raise ValueError("retrieval_k cannot exceed the memory budget")
    return {
        "task": task,
        "dataset": dataset,
        "model": model,
        "sequence_length": sequence_length,
        "budget": budget,
        "retrieval_k": retrieval_k,
        "seed": seed,
        "run_id": run_id,
        "output_dir": output_dir,
    }


def validate_model_registry() -> dict[str, Any]:
    available = set(CONTROLLED_MODELS) | set(PILOT_MODELS) | {"budgetmem_r"}
    missing = sorted(REQUIRED_SECTION18_ALIASES - available)
    result = {
        "required_aliases": sorted(REQUIRED_SECTION18_ALIASES),
        "available_aliases": sorted(available),
        "missing_aliases": missing,
        "status": "PASS" if not missing else "FAIL",
    }
    if missing:
        raise RuntimeError(f"Section 18 model aliases are missing: {missing}")
    return result


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def build_pilot_config(
    source_config: dict[str, Any],
    cell: dict[str, Any],
    *,
    smoke: bool,
) -> tuple[dict[str, Any], Path]:
    base_path = Path("configs/experiments/pilot.yaml")
    base = read_yaml(base_path)
    model_name = PILOT_MODELS.get(cell["model"])
    if model_name is None:
        raise ValueError(
            f"Model is not supported by the pilot backend: {cell['model']}"
        )

    config = copy.deepcopy(base)
    config["experiment_name"] = f"section18_{cell['run_id']}"
    config["seed"] = int(cell["seed"])
    matrix = config.setdefault("matrix", {})
    matrix["tasks"] = [cell["task"]]
    matrix["evaluation_sequence_lengths"] = [int(cell["sequence_length"])]
    matrix["memory_budgets"] = [int(cell["budget"])]
    matrix["models"] = [model_name]

    training = config.setdefault("training", {})
    training["seed"] = int(cell["seed"])
    if smoke:
        training["train_samples"] = min(int(training.get("train_samples", 64)), 64)
        training["validation_samples"] = min(
            int(training.get("validation_samples", 24)), 24
        )
        training["epochs"] = 1
        training["batch_size"] = min(int(training.get("batch_size", 8)), 4)

    model_cfg = config.setdefault("model", {})
    model_cfg["retrieval_k"] = int(cell["retrieval_k"])

    overrides = source_config.get("pilot_overrides", {})
    if isinstance(overrides, dict):
        _deep_update(config, overrides)

    output_dir = Path(cell["output_dir"])
    artifacts = {
        "output_root": str(output_dir / "outputs"),
        "results_csv": str(output_dir / "results.csv"),
        "summary_json": str(output_dir / "summary.json"),
        "gate_json": str(output_dir / "gate.json"),
        "report_markdown": str(output_dir / "report.md"),
        "checkpoint_root": str(output_dir / "checkpoints"),
    }
    config["artifacts"] = artifacts
    effective_path = output_dir / "effective_pilot_config.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config, effective_path


def run_pilot_backend(
    source_config: dict[str, Any],
    config_path: Path,
    cell: dict[str, Any],
    *,
    smoke: bool,
    resume: bool,
) -> dict[str, Any]:
    config, effective_path = build_pilot_config(source_config, cell, smoke=smoke)
    config_hash = sha256_file(effective_path.resolve())
    pilot_model = PILOT_MODELS[cell["model"]]
    model_seed = (
        int(cell["seed"])
        + stable_int(f"section18:{cell['task']}:{pilot_model}:{cell['run_id']}")
        % 1_000_000
    )
    seed_everything(model_seed)

    started = time.perf_counter()
    model, training_record = train_one_model(
        cfg=config,
        config_sha256=config_hash,
        task=cell["task"],
        model_name=pilot_model,
        seed=model_seed,
        resume=resume,
    )
    print(
        f"TRAINED task={cell['task']} model={cell['model']} "
        f"seed={cell['seed']} backend=pilot",
        flush=True,
    )
    row = evaluate_model(
        cfg=config,
        config_path=effective_path.resolve(),
        config_sha256=config_hash,
        task=cell["task"],
        sequence_length=int(cell["sequence_length"]),
        budget=int(cell["budget"]),
        model_name=pilot_model,
        model=model,
        training_record=training_record,
        seed=int(cell["seed"]),
    )
    print(
        f"EVALUATED task={cell['task']} model={cell['model']} "
        f"sequence_length={cell['sequence_length']} budget={cell['budget']}",
        flush=True,
    )

    output_dir = Path(cell["output_dir"])
    write_csv(output_dir / "results.csv", [row])
    record_payload = (
        asdict(training_record)
        if is_dataclass(training_record)
        else dict(training_record)
    )
    duration = time.perf_counter() - started
    payload = {
        "status": "PASS",
        "backend": "pilot",
        "source_config": str(config_path.resolve()),
        "effective_config": str(effective_path.resolve()),
        "cell": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in cell.items()
        },
        "training_record": record_payload,
        "result": row,
        "duration_seconds": duration,
        "parameter_count": parameter_count(model),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def _parse_cell_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    text = str(value).strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        pass
    return text.split()


def _find_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    mapping = {canonical_name(column): column for column in columns}
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    for normalized, original in mapping.items():
        if any(candidate in normalized for candidate in candidates):
            return original
    return None


def _records_from_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        if isinstance(payload, dict):
            for key in ("records", "data", "examples", "rows"):
                if isinstance(payload.get(key), list):
                    return [dict(row) for row in payload[key]]
        raise ValueError(f"Unsupported JSON dataset structure: {path}")
    if suffix in {".parquet", ".pq"}:
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported dataset file: {path}")


def _candidate_dataset_roots(cell: dict[str, Any], split: str) -> list[Path]:
    dataset = cell["dataset"]
    task = cell["task"]
    roots = [
        Path("data/processed") / dataset / split,
        Path("data/processed") / dataset,
        Path("data/processed/synthetic") / task / split,
        Path("data/processed/synthetic") / task,
        Path("data") / dataset / split,
        Path("data") / dataset,
    ]
    return roots


def _find_dataset_file(
    source_config: dict[str, Any],
    cell: dict[str, Any],
    split: str,
) -> Path:
    explicit = first_value(
        nested(source_config, "data", f"{split}_path"),
        nested(source_config, "data", "processed_dir"),
        nested(source_config, "data", "path"),
    )
    roots: list[Path] = []
    if explicit:
        explicit_path = Path(str(explicit))
        if explicit_path.is_file():
            return explicit_path
        roots.append(explicit_path / split)
        roots.append(explicit_path)
    roots.extend(_candidate_dataset_roots(cell, split))

    filenames = (
        f"{split}.jsonl",
        f"{split}.parquet",
        f"{split}.csv",
        "data.parquet",
        "data.jsonl",
        "data.csv",
    )
    for root in roots:
        if root.is_file():
            return root
        if not root.exists():
            continue
        for filename in filenames:
            candidate = root / filename
            if candidate.is_file():
                return candidate
        for pattern in ("*.parquet", "*.jsonl", "*.csv"):
            candidates = sorted(root.glob(pattern))
            if candidates:
                return candidates[0]
    raise FileNotFoundError(
        f"No prepared {cell['dataset']} {split} dataset was found. "
        f"Checked: {[str(path) for path in roots]}"
    )


class PreparedSequenceDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        sequence_length: int,
        max_examples: int | None = None,
    ) -> None:
        if max_examples is not None:
            rows = rows[:max_examples]
        if not rows:
            raise ValueError("Prepared dataset is empty")
        input_column = _find_column(rows[0].keys(), INPUT_COLUMNS)
        target_column = _find_column(rows[0].keys(), TARGET_COLUMNS)
        if input_column is None or target_column is None:
            raise ValueError(
                f"Dataset columns do not expose inputs and targets. "
                f"Columns={sorted(rows[0].keys())}"
            )

        self.examples: list[tuple[list[int], int | list[int]]] = []
        self.sequence_length = int(sequence_length)
        self.scalar_targets = True
        maximum_token = 0
        maximum_target = 0
        for row in rows:
            parsed_input = _parse_cell_value(row.get(input_column))
            parsed_target = _parse_cell_value(row.get(target_column))
            if not isinstance(parsed_input, (list, tuple)):
                raise ValueError(f"Input column {input_column} is not sequence-like")
            input_ids = [int(float(value)) for value in parsed_input]
            input_ids = input_ids[: self.sequence_length]
            if len(input_ids) < self.sequence_length:
                input_ids.extend([0] * (self.sequence_length - len(input_ids)))
            maximum_token = max(maximum_token, max(input_ids, default=0))

            if isinstance(parsed_target, (list, tuple)):
                self.scalar_targets = False
                target_ids = [int(float(value)) for value in parsed_target]
                target_ids = target_ids[: self.sequence_length]
                if len(target_ids) < self.sequence_length:
                    target_ids.extend([-100] * (self.sequence_length - len(target_ids)))
                non_ignored = [value for value in target_ids if value >= 0]
                maximum_target = max(maximum_target, max(non_ignored, default=0))
                target: int | list[int] = target_ids
            else:
                target = int(float(parsed_target))
                maximum_target = max(maximum_target, int(target))
            self.examples.append((input_ids, target))

        self.vocab_size = max(16, maximum_token + 2)
        self.output_dim = max(2, maximum_target + 1)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        input_ids, target = self.examples[index]
        x = torch.tensor(input_ids, dtype=torch.long)
        if isinstance(target, list):
            y = torch.tensor(target, dtype=torch.long)
        else:
            y = torch.tensor(target, dtype=torch.long)
        return x, y


class EmbeddedControlledModel(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        sequence_length: int,
        budget: int,
        seed: int,
    ) -> None:
        super().__init__()
        canonical = canonical_name(model_name)
        if canonical not in CONTROLLED_MODELS:
            raise KeyError(f"Controlled model alias is unsupported: {model_name}")
        registry_name, fixed_kwargs = CONTROLLED_MODELS[canonical]
        kwargs: dict[str, Any] = {
            "input_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
        }
        kwargs.update(fixed_kwargs)
        if registry_name.startswith("memory_caching"):
            kwargs.update({"budget": min(budget, sequence_length), "seed": seed})
        elif registry_name.startswith("transformer"):
            heads = 4 if hidden_dim % 4 == 0 else 1
            kwargs.update(
                {
                    "num_layers": 1,
                    "num_heads": heads,
                    "window_size": min(256, sequence_length),
                    "max_length": sequence_length,
                }
            )
        elif registry_name == "state_space":
            kwargs.update({"num_layers": 1, "state_dim": max(4, hidden_dim // 4)})
        elif registry_name == "recurrent_memory_transformer":
            heads = 4 if hidden_dim % 4 == 0 else 1
            kwargs.update(
                {
                    "segment_length": min(128, sequence_length),
                    "memory_tokens": max(1, min(8, budget)),
                    "num_layers": 1,
                    "num_heads": heads,
                }
            )
        else:
            kwargs.update({"num_layers": 1})

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.sequence_model = build_baseline(registry_name, **kwargs)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.sequence_model(self.embedding(input_ids))


class EmbeddedBudgetMemR(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        budget: int,
        retrieval_k: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        signature = inspect.signature(BudgetMemR)
        values: dict[str, Any] = {
            "input_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "key_dim": hidden_dim,
            "value_dim": hidden_dim,
            "max_budget": budget,
            "allowed_budgets": (budget,),
            "retrieval_k": min(retrieval_k, budget),
            "num_layers": 1,
            "detach_memory_writes": True,
            "detach_memory": True,
        }
        kwargs = {name: values[name] for name in signature.parameters if name in values}
        self.model = BudgetMemR(**kwargs)
        self.budget = int(budget)

    def forward(self, input_ids: Tensor) -> Tensor:
        embedded = self.embedding(input_ids)
        signature = inspect.signature(self.model.forward)
        kwargs: dict[str, Any] = {}
        batch_size = input_ids.shape[0]
        if "budgets" in signature.parameters:
            kwargs["budgets"] = torch.full(
                (batch_size,), self.budget, dtype=torch.long, device=input_ids.device
            )
        elif "budget" in signature.parameters:
            kwargs["budget"] = self.budget
        if "reset" in signature.parameters:
            kwargs["reset"] = True
        if "reset_memory" in signature.parameters:
            kwargs["reset_memory"] = True
        output = self.model(embedded, **kwargs)
        if isinstance(output, Tensor):
            return output
        for name in ("logits", "output", "predictions"):
            value = getattr(output, name, None)
            if isinstance(value, Tensor):
                return value
        if (
            isinstance(output, (tuple, list))
            and output
            and isinstance(output[0], Tensor)
        ):
            return output[0]
        raise TypeError("BudgetMemR output does not expose a logits tensor")


def build_generic_model(
    cell: dict[str, Any],
    *,
    vocab_size: int,
    output_dim: int,
    source_config: dict[str, Any],
) -> nn.Module:
    model_cfg = source_config.get("model", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    embedding_dim = int(model_cfg.get("embedding_dim", 32))
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    if cell["model"] in {"budgetmem_r", "budgetmem-r"}:
        return EmbeddedBudgetMemR(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            budget=int(cell["budget"]),
            retrieval_k=int(cell["retrieval_k"]),
        )
    return EmbeddedControlledModel(
        model_name=cell["model"],
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        sequence_length=int(cell["sequence_length"]),
        budget=int(cell["budget"]),
        seed=int(cell["seed"]),
    )


def _classification_metrics(
    targets: list[int], predictions: list[int]
) -> dict[str, float]:
    if not targets:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    correct = sum(int(a == b) for a, b in zip(targets, predictions, strict=True))
    tp = sum(int(a == 1 and b == 1) for a, b in zip(targets, predictions, strict=True))
    fp = sum(int(a != 1 and b == 1) for a, b in zip(targets, predictions, strict=True))
    fn = sum(int(a == 1 and b != 1) for a, b in zip(targets, predictions, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": correct / len(targets),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_controlled_backend(
    source_config: dict[str, Any],
    config_path: Path,
    cell: dict[str, Any],
    *,
    smoke: bool,
    resume: bool,
) -> dict[str, Any]:
    output_dir = Path(cell["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = _find_dataset_file(source_config, cell, "train")
    validation_split = "validation"
    try:
        validation_path = _find_dataset_file(source_config, cell, validation_split)
    except FileNotFoundError:
        validation_split = "test"
        validation_path = _find_dataset_file(source_config, cell, validation_split)

    max_examples = 64 if smoke else None
    train_dataset = PreparedSequenceDataset(
        _records_from_file(train_path),
        sequence_length=int(cell["sequence_length"]),
        max_examples=max_examples,
    )
    validation_dataset = PreparedSequenceDataset(
        _records_from_file(validation_path),
        sequence_length=int(cell["sequence_length"]),
        max_examples=32 if smoke else None,
    )
    if train_dataset.scalar_targets != validation_dataset.scalar_targets:
        raise ValueError("Train and validation target structures differ")

    seed_everything(int(cell["seed"]))
    model = build_generic_model(
        cell,
        vocab_size=max(train_dataset.vocab_size, validation_dataset.vocab_size),
        output_dim=max(train_dataset.output_dim, validation_dataset.output_dim),
        source_config=source_config,
    )
    device = torch.device("cpu")
    model.to(device)

    training_cfg = source_config.get("training", {})
    if not isinstance(training_cfg, dict):
        training_cfg = {}
    batch_size = int(training_cfg.get("batch_size", 4 if smoke else 8))
    max_steps = int(training_cfg.get("max_steps", 2 if smoke else 100))
    if smoke:
        max_steps = min(max_steps, 2)
    learning_rate = float(training_cfg.get("learning_rate", 1.0e-3))
    loader_generator = torch.Generator().manual_seed(int(cell["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, batch_size),
        shuffle=True,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    checkpoint_path = output_dir / "checkpoint.pt"
    start_step = 0
    if resume and checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", 0))

    started = time.perf_counter()
    model.train()
    step = start_step
    losses: list[float] = []
    while step < max_steps:
        progressed = False
        for input_ids, targets in train_loader:
            progressed = True
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            if train_dataset.scalar_targets:
                if logits.ndim == 3:
                    logits = logits.mean(dim=1)
                loss = F.cross_entropy(logits, targets)
            else:
                if logits.ndim != 3:
                    raise ValueError("Token-level targets require sequence logits")
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
            if step >= max_steps:
                break
        if not progressed:
            raise RuntimeError("Training loader produced no batches")

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "cell": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in cell.items()
            },
        },
        checkpoint_path,
    )
    print(
        f"TRAINED task={cell['task']} model={cell['model']} "
        f"seed={cell['seed']} backend=controlled steps={step}",
        flush=True,
    )

    model.eval()
    scalar_targets: list[int] = []
    scalar_predictions: list[int] = []
    token_correct = 0
    token_total = 0
    with torch.no_grad():
        for input_ids, targets in validation_loader:
            logits = model(input_ids.to(device))
            if validation_dataset.scalar_targets:
                if logits.ndim == 3:
                    logits = logits.mean(dim=1)
                predictions = logits.argmax(dim=-1).cpu()
                scalar_targets.extend(int(value) for value in targets.tolist())
                scalar_predictions.extend(int(value) for value in predictions.tolist())
            else:
                predictions = logits.argmax(dim=-1).cpu()
                mask = targets.ne(-100)
                token_correct += int((predictions[mask] == targets[mask]).sum())
                token_total += int(mask.sum())

    if validation_dataset.scalar_targets:
        metrics = _classification_metrics(scalar_targets, scalar_predictions)
        primary_metric_name = "f1" if validation_dataset.output_dim == 2 else "accuracy"
    else:
        metrics = {
            "token_accuracy": token_correct / token_total if token_total else 0.0
        }
        primary_metric_name = "token_accuracy"
    duration = time.perf_counter() - started
    metrics.update(
        {
            "primary_metric_name": primary_metric_name,
            "primary_metric_value": float(metrics[primary_metric_name]),
            "train_loss": float(sum(losses) / len(losses)),
            "training_steps": step,
            "parameter_count": parameter_count(model),
            "duration_seconds": duration,
        }
    )
    payload = {
        "status": "PASS",
        "backend": "controlled",
        "source_config": str(config_path.resolve()),
        "train_dataset": str(train_path.resolve()),
        "evaluation_dataset": str(validation_path.resolve()),
        "evaluation_split": validation_split,
        "cell": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in cell.items()
        },
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "effective_config.yaml").write_text(
        yaml.safe_dump(source_config, sort_keys=False), encoding="utf-8"
    )
    print(
        f"EVALUATED task={cell['task']} model={cell['model']} "
        f"sequence_length={cell['sequence_length']} budget={cell['budget']} "
        f"{primary_metric_name}={metrics[primary_metric_name]:.8f}",
        flush=True,
    )
    return payload


def choose_backend(cell: dict[str, Any]) -> str:
    if cell["dataset"] == "synthetic" and cell["model"] in PILOT_MODELS:
        return "pilot"
    if cell["model"] in CONTROLLED_MODELS or cell["model"] in {
        "budgetmem_r",
        "budgetmem-r",
    }:
        return "controlled"
    raise ValueError(f"No Section 18 backend is registered for model={cell['model']}")


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    source_config = read_yaml(config_path)
    if not isinstance(source_config, dict):
        raise TypeError("Section 18 configuration must be a YAML mapping")

    registry = validate_model_registry()
    cell = resolve_cell(source_config, config_path)
    backend = choose_backend(cell)
    validation_payload = {
        "status": "PASS",
        "registry": registry,
        "backend": backend,
        "cell": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in cell.items()
        },
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    if args.validate_only:
        print(json.dumps(validation_payload, indent=2, sort_keys=True))
        return 0

    output_dir = Path(cell["output_dir"])
    if not args.resume and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runner_validation.json").write_text(
        json.dumps(validation_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if backend == "pilot":
        payload = run_pilot_backend(
            source_config,
            config_path,
            cell,
            smoke=args.smoke,
            resume=args.resume,
        )
    else:
        payload = run_controlled_backend(
            source_config,
            config_path,
            cell,
            smoke=args.smoke,
            resume=args.resume,
        )

    print("SECTION18_RESULT_JSON=" + json.dumps(payload, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SECTION18_RUNNER_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
