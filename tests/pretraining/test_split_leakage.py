"""Dataset partition and leakage tests required before training."""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(relative_path: str) -> dict:
    path = REPO_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_synthetic_split_and_task_seeds_do_not_overlap() -> None:
    config = _read_yaml("configs/data/synthetic.yaml")
    split_seeds = {
        str(name): int(value) for name, value in config["split_seeds"].items()
    }
    assert len(split_seeds) == len(set(split_seeds.values()))

    enabled_tasks = {
        name: values
        for name, values in config["tasks"].items()
        if bool(values.get("enabled", False))
    }
    task_seeds = {
        name: int(values["random_seed"]) for name, values in enabled_tasks.items()
    }
    assert len(task_seeds) == len(set(task_seeds.values()))

    combined = {
        (split_name, task_name): split_seed + task_seeds[task_name]
        for split_name, split_seed in split_seeds.items()
        for task_name in enabled_tasks
    }
    assert len(combined) == len(set(combined.values()))


def test_generated_synthetic_sample_ids_do_not_cross_splits() -> None:
    config = _read_yaml("configs/data/synthetic.yaml")
    root = REPO_ROOT / str(config["output_root"])
    enabled = [
        name
        for name, values in config["tasks"].items()
        if bool(values.get("enabled", False))
    ]

    required = [
        root / task / split / "data.parquet"
        for task in enabled
        for split in ("train", "validation", "test")
    ]
    if not all(path.exists() for path in required):
        pytest.skip("Generated synthetic partitions are not all present")

    for task in enabled:
        identifiers: dict[str, set[str]] = {}
        for split in ("train", "validation", "test"):
            frame = pd.read_parquet(
                root / task / split / "data.parquet",
                columns=["sample_id"],
            )
            identifiers[split] = set(frame["sample_id"].astype(str))

        for left, right in combinations(identifiers, 2):
            assert identifiers[left].isdisjoint(identifiers[right])


def test_hdfs_block_ids_do_not_cross_splits() -> None:
    root = REPO_ROOT / "data" / "processed" / "hdfs"
    paths = {
        split: root / split / "data.parquet"
        for split in ("train", "validation", "test")
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("Processed HDFS partitions are not all present")

    identifiers: dict[str, set[str]] = {}
    for split, path in paths.items():
        frame = pd.read_parquet(path, columns=["block_id"])
        assert not frame["block_id"].duplicated().any()
        identifiers[split] = set(frame["block_id"].astype(str))

    for left, right in combinations(identifiers, 2):
        assert identifiers[left].isdisjoint(identifiers[right])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_imdb_official_test_examples_are_not_in_train_or_validation() -> None:
    root = REPO_ROOT / "data" / "processed" / "imdb"
    paths = {
        "train": root / "train" / "data.parquet",
        "validation": root / "validation" / "data.parquet",
        "test": root / "test_locked" / "data.parquet",
    }
    manifest_path = REPO_ROOT / "data" / "manifests" / "imdb_manifest.json"
    indices_path = REPO_ROOT / "data" / "manifests" / "imdb_split_indices.json"

    required = [*paths.values(), manifest_path, indices_path]
    if not all(item.exists() for item in required):
        pytest.skip("IMDb data or provenance manifests are not all present")

    train = pd.read_parquet(paths["train"], columns=["source_index"])
    validation = pd.read_parquet(
        paths["validation"],
        columns=["source_index"],
    )
    test = pd.read_parquet(paths["test"], columns=["source_index"])

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    indices = yaml.safe_load(indices_path.read_text(encoding="utf-8"))

    assert len(train) == 22500
    assert len(validation) == 2500
    assert len(test) == 25000

    train_indices = list(map(int, indices["train_source_indices"]))
    validation_indices = list(map(int, indices["validation_source_indices"]))
    official_test_indices = list(map(int, indices["official_test_source_indices"]))

    assert train["source_index"].astype(int).tolist() == train_indices
    assert validation["source_index"].astype(int).tolist() == validation_indices
    assert test["source_index"].astype(int).tolist() == official_test_indices

    assert set(train_indices).isdisjoint(validation_indices)
    assert set(train_indices) | set(validation_indices) == set(range(25000))
    assert official_test_indices == list(range(25000))

    assert bool(manifest["official_test_locked"])
    assert manifest["train_fingerprint"]
    assert manifest["test_fingerprint"]
    assert manifest["train_fingerprint"] != manifest["test_fingerprint"]

    for split, parquet_path in paths.items():
        manifest_name = "test_locked" if split == "test" else split
        assert _sha256_file(parquet_path) == manifest["files"][manifest_name]["sha256"]

    assert (root / "test_locked" / "DO_NOT_USE_FOR_DEVELOPMENT.txt").exists()
