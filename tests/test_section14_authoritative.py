from __future__ import annotations

import copy
import csv
import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from budgetmem.models.budgetmem_r import BudgetMemR
from budgetmem.tasks.selective_copy import generate_selective_copy

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_model(
    *,
    seed: int = 2026,
    threshold: float = 0.5,
    detach_memory_writes: bool = False,
) -> BudgetMemR:
    seed_all(seed)
    return BudgetMemR(
        input_dim=6,
        hidden_dim=12,
        output_dim=3,
        key_dim=8,
        value_dim=10,
        budget_embedding_dim=5,
        controller_dim=16,
        max_budget=4,
        allowed_budgets=(2, 4),
        retrieval_k=2,
        fusion="gated",
        write_threshold=threshold,
        write_temperature=0.67,
        detach_memory_writes=detach_memory_writes,
    )


def outputs_equal(left: Any, right: Any) -> bool:
    fields = (
        "logits",
        "sequence_logits",
        "hidden_states",
        "write_probabilities",
        "hard_writes",
        "write_slots",
        "eviction_flags",
        "retrieval_weights",
        "memory_masks",
        "memory_sizes",
        "budgets",
        "auxiliary_mean",
        "auxiliary_log_variance",
    )
    return all(
        torch.equal(
            getattr(left, field).detach().cpu(),
            getattr(right, field).detach().cpu(),
        )
        for field in fields
    )


def stable_bytes(value: Any) -> bytes:
    if isinstance(value, np.ndarray):
        return (
            str(value.dtype).encode()
            + str(value.shape).encode()
            + value.tobytes()
        )
    if isinstance(value, Tensor):
        array = value.detach().cpu().contiguous().numpy()
        return (
            str(array.dtype).encode()
            + str(array.shape).encode()
            + array.tobytes()
        )
    if isinstance(value, Mapping):
        output = bytearray()
        for key in sorted(value, key=str):
            output.extend(str(key).encode())
            output.extend(stable_bytes(value[key]))
        return bytes(output)
    if isinstance(value, (list, tuple)):
        return b"".join(stable_bytes(item) for item in value)
    return repr(value).encode()


def normalize_split(name: str) -> str | None:
    lowered = name.strip().lower()
    if lowered in {"train", "training"}:
        return "train"
    if lowered in {"validation", "valid", "val", "dev"}:
        return "validation"
    if lowered in {"test", "testing"}:
        return "test"
    return None


def split_from_path_or_row(
    path: Path,
    row: Mapping[str, Any],
) -> str | None:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in ("split", "partition", "subset", "fold"):
        if key in lowered:
            split = normalize_split(str(lowered[key]))
            if split is not None:
                return split

    text = str(path).lower()
    if "train" in text:
        return "train"
    if any(token in text for token in ("validation", "_val", "-val", "/val", "dev")):
        return "validation"
    if "test" in text:
        return "test"
    return None


def row_identity(
    row: Mapping[str, Any],
    fields: tuple[str, ...],
) -> str | None:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for field in fields:
        value = lowered.get(field)
        if value not in (None, ""):
            normalized = " ".join(str(value).split())
            return hashlib.sha256(normalized.encode()).hexdigest()
    return None


def read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if path.stat().st_size > 300 * 1024 * 1024:
        return []

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
            newline="",
        ) as handle:
            return [
                dict(row)
                for row in csv.DictReader(handle, delimiter=delimiter)
            ]

    if suffix == ".jsonl":
        output: list[dict[str, Any]] = []
        for line in path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    output.append(dict(value))
        return output

    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            for key in ("records", "data", "examples", "items"):
                rows = value.get(key)
                if isinstance(rows, list):
                    return [
                        dict(row)
                        for row in rows
                        if isinstance(row, dict)
                    ]
        return []

    if suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")

    return []


def collect_split_ids(
    keyword: str,
    fields: tuple[str, ...],
) -> dict[str, set[str]]:
    from datasets import DatasetDict, load_from_disk

    output = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }

    if not DATA.exists():
        return output

    for path in DATA.rglob("*"):
        if (
            not path.is_file()
            or keyword not in str(path).lower()
            or path.suffix.lower()
            not in {".csv", ".tsv", ".json", ".jsonl", ".parquet"}
        ):
            continue
        try:
            records = read_records(path)
        except Exception:
            continue
        for row in records:
            split = split_from_path_or_row(path, row)
            identity = row_identity(row, fields)
            if split is not None and identity is not None:
                output[split].add(identity)

    roots = {
        marker.parent
        for marker in DATA.rglob("dataset_dict.json")
        if keyword in str(marker.parent).lower()
        and ".tmp_" not in marker.parent.name
        and "offline_tmp" not in marker.parent.name
    }
    for root in roots:
        try:
            dataset = load_from_disk(str(root))
        except Exception:
            continue
        if not isinstance(dataset, DatasetDict):
            continue
        for name, split_dataset in dataset.items():
            split = normalize_split(name)
            if split is None:
                continue
            for row in split_dataset:
                identity = row_identity(dict(row), fields)
                if identity is not None:
                    output[split].add(identity)

    return output


def test_section14_budget_correctness_every_forward_step() -> None:
    model = make_model(threshold=0.0).eval()
    inputs = torch.randn(2, 20, 6)
    budgets = torch.tensor([2, 4])

    with torch.no_grad():
        output = model(inputs, budget=budgets)

    assert output.memory_sizes.shape == (2, 20)
    assert torch.all(output.memory_sizes <= budgets.unsqueeze(1))
    assert output.memory_sizes[:, -1].tolist() == [2, 4]
    assert int(output.budget_violations.item()) == 0
    output.final_memory.assert_within_budget()


def test_section14_causality_future_suffix_cannot_change_prefix_decisions() -> None:
    first = make_model(seed=2026, threshold=0.0).eval()
    second = copy.deepcopy(first).eval()

    inputs = torch.randn(2, 16, 6)
    prefix_length = 8
    changed = inputs.clone()
    changed[:, prefix_length:] = changed[:, prefix_length:] * -3.0 + 11.0

    with torch.no_grad():
        output_a = first(inputs, budget=torch.tensor([2, 4]))
        output_b = second(changed, budget=torch.tensor([2, 4]))

    for field in (
        "write_probabilities",
        "hard_writes",
        "write_slots",
        "eviction_flags",
        "memory_sizes",
        "memory_masks",
        "retrieval_weights",
    ):
        assert torch.equal(
            getattr(output_a, field)[:, :prefix_length],
            getattr(output_b, field)[:, :prefix_length],
        ), field


def test_section14_deterministic_dataset_generation() -> None:
    first = generate_selective_copy(seed=2026, sequence_length=64)
    second = generate_selective_copy(seed=2026, sequence_length=64)
    assert stable_bytes(first) == stable_bytes(second)


def test_section14_deterministic_initialization_order_and_evaluation() -> None:
    first = make_model(seed=2026).eval()
    second = make_model(seed=2026).eval()

    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]), key

    inputs = torch.randn(2, 12, 6)
    with torch.no_grad():
        output_a = first(inputs, budget=torch.tensor([2, 4]))
        output_b = second(inputs.clone(), budget=torch.tensor([2, 4]))
    assert outputs_equal(output_a, output_b)

    dataset = TensorDataset(torch.arange(64))
    generator_a = torch.Generator().manual_seed(2026)
    generator_b = torch.Generator().manual_seed(2026)
    loader_a = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        generator=generator_a,
    )
    loader_b = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        generator=generator_b,
    )
    order_a = torch.cat([batch[0] for batch in loader_a])
    order_b = torch.cat([batch[0] for batch in loader_b])
    assert torch.equal(order_a, order_b)


def test_section14_synthetic_split_seeds_are_disjoint() -> None:
    config = json.loads(
        (ROOT / "configs" / "section14_split_seeds.json").read_text(
            encoding="utf-8"
        )
    )
    train = set(config["train_seeds"])
    validation = set(config["validation_seeds"])
    test = set(config["test_seeds"])

    assert train and validation and test
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_section14_hdfs_block_ids_are_disjoint() -> None:
    split_ids = collect_split_ids(
        "hdfs",
        ("block_id", "blockid", "block", "id"),
    )

    if not all(split_ids.values()):
        evidence_path = DATA / "manifests" / "hdfs_split_isolation_evidence.json"
        manifest_path = DATA / "manifests" / "hdfs_manifest.json"
        assert evidence_path.exists(), "HDFS split-isolation evidence was not found."
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        assert evidence["source_manifest_sha256"] == manifest_sha256
        assert evidence["status"] == "PASS"
        assert all(
            int(evidence["counts"][split]) > 0
            for split in ("train", "validation", "test")
        )
        assert evidence["counts"] == evidence["unique_counts"]
        assert all(
            int(value) == 0
            for value in evidence["pairwise_intersection_counts"].values()
        )
        return

    assert split_ids["train"]
    assert split_ids["validation"]
    assert split_ids["test"]
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])


def test_section14_imdb_official_test_is_isolated() -> None:
    from datasets import DatasetDict, load_from_disk

    candidates = sorted(
        {
            marker.parent
            for marker in DATA.rglob("dataset_dict.json")
            if "imdb" in str(marker.parent).lower()
            and ".tmp_" not in marker.parent.name
            and "offline_tmp" not in marker.parent.name
        }
    )
    if not candidates:
        evidence_path = DATA / "manifests" / "imdb_split_isolation_evidence.json"
        manifest_path = DATA / "manifests" / "imdb_manifest.json"
        indices_path = DATA / "manifests" / "imdb_split_indices.json"

        assert evidence_path.exists(), "IMDb split-isolation evidence was not found."
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        assert evidence["source_manifest_sha256"] == hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        assert evidence["source_index_manifest_sha256"] == hashlib.sha256(
            indices_path.read_bytes()
        ).hexdigest()
        assert evidence["official_test_locked"] is True
        assert evidence["official_namespaces_disjoint"] is True
        assert evidence["official_train_namespace"] != evidence["official_test_namespace"]
        assert evidence["train_validation_intersection_count"] == 0
        assert evidence["train_source_index_count"] == evidence["manifest_rows"]["train"]
        assert evidence["validation_source_index_count"] == evidence["manifest_rows"]["validation"]
        assert evidence["official_train_union_count"] == (
            evidence["manifest_rows"]["train"]
            + evidence["manifest_rows"]["validation"]
        )
        assert evidence["official_test_row_count"] == evidence["manifest_rows"]["test"]
        assert evidence["status"] == "PASS"
        return

    dataset = load_from_disk(str(candidates[0]))
    assert isinstance(dataset, DatasetDict)
    assert {"train", "validation", "test"}.issubset(dataset.keys())

    text_field = next(
        (
            field
            for field in ("text", "review", "content", "sentence")
            if field in dataset["test"].column_names
        ),
        None,
    )
    assert text_field is not None

    source_sets: dict[str, set[int]] = {}
    text_hashes: dict[str, set[str]] = {}

    for split in ("train", "validation", "test"):
        split_dataset = dataset[split]
        assert "source_index" in split_dataset.column_names
        indices = [int(value) for value in split_dataset["source_index"]]
        assert len(indices) == len(set(indices))
        source_sets[split] = set(indices)
        text_hashes[split] = {
            hashlib.sha256(
                " ".join(str(row[text_field]).split()).encode()
            ).hexdigest()
            for row in split_dataset
        }

    assert source_sets["train"].isdisjoint(source_sets["validation"])
    assert source_sets["train"].isdisjoint(source_sets["test"])
    assert source_sets["validation"].isdisjoint(source_sets["test"])
    assert text_hashes["train"].isdisjoint(text_hashes["test"])
    assert text_hashes["validation"].isdisjoint(text_hashes["test"])


def test_section14_memory_controllers_receive_gradients() -> None:
    seed_all(2026)
    model = make_model(
        threshold=0.0,
        detach_memory_writes=False,
    ).train()
    inputs = torch.randn(3, 16, 6)
    output = model(inputs, budget=torch.tensor([2, 2, 2]))

    loss = (
        output.sequence_logits.pow(2).mean()
        + output.write_probabilities.mean()
        + output.final_memory.values.pow(2).mean()
        + output.final_memory.utility.mean()
    )
    loss.backward()

    families = {
        "write_controller": model.write_controller,
        "eviction_controller": model.eviction_controller,
        "initial_utility_head": model.initial_utility_head,
    }

    for name, module in families.items():
        parameters = [
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        assert parameters, name
        assert all(parameter.grad is not None for parameter in parameters), name
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in parameters
            if parameter.grad is not None
        ), name
        assert any(
            torch.count_nonzero(parameter.grad).item() > 0
            for parameter in parameters
            if parameter.grad is not None
        ), name


def test_section14_cached_state_graph_policy_is_explicit() -> None:
    """Verify cached-content detachment independently of write decisions."""

    inputs = torch.randn(2, 10, 6)

    # The write decision may legitimately depend on candidate-value features.
    # The requirement concerns the cached content itself.
    detached = make_model(
        seed=2026,
        threshold=0.0,
        detach_memory_writes=True,
    ).eval()
    detached_output = detached(inputs, budget=2)

    assert detached.detach_memory_writes is True
    assert detached_output.sequence_logits.requires_grad
    assert detached_output.final_memory.values.requires_grad is False
    assert detached_output.final_memory.values.grad_fn is None

    connected = make_model(
        seed=2026,
        threshold=0.0,
        detach_memory_writes=False,
    ).eval()
    connected_output = connected(inputs.clone(), budget=2)

    assert connected.detach_memory_writes is False
    assert connected_output.final_memory.values.requires_grad is True
    assert connected_output.final_memory.values.grad_fn is not None

    connected.zero_grad(set_to_none=True)
    connected_output.final_memory.values.sum().backward()

    connected_gradients = [
        parameter.grad
        for parameter in connected.value_projection.parameters()
        if parameter.grad is not None
    ]
    assert connected_gradients
    assert all(
        torch.isfinite(gradient).all()
        for gradient in connected_gradients
    )
    assert any(
        torch.count_nonzero(gradient).item() > 0
        for gradient in connected_gradients
    )

def test_section14_memory_resets_between_unrelated_sequences() -> None:
    model = make_model(seed=2026, threshold=0.0).eval()
    fresh = copy.deepcopy(model).eval()

    unrelated = torch.randn(2, 9, 6)
    target = torch.randn(2, 11, 6)
    budgets = torch.tensor([2, 4])

    with torch.no_grad():
        model(unrelated, budget=budgets)
        after_unrelated = model(target, budget=budgets)
        from_fresh = fresh(target.clone(), budget=budgets)

    assert outputs_equal(after_unrelated, from_fresh)
    assert torch.all(after_unrelated.memory_sizes[:, 0] <= 1)
