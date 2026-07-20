from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

PRIMARY_TASKS = [
    "selective_copy",
    "associative_recall",
    "multiple_key_retrieval",
    "delayed_xor",
    "rare_event_recall",
    "distractor_heavy_retrieval",
]
SPLITS = ["train", "validation", "test"]


def validate_synthetic() -> list[str]:
    errors = []
    cfg = yaml.safe_load(Path("configs/data/synthetic.yaml").read_text(encoding="utf-8"))
    seeds = list(cfg["split_seeds"].values())
    if len(seeds) != len(set(seeds)):
        errors.append("Synthetic split seeds are not unique.")
    for task in PRIMARY_TASKS:
        seen_ids = set()
        for split in SPLITS:
            root = Path(cfg["output_root"]) / task / split
            data_path = root / "data.parquet"
            generation_path = root / "generation_config.yaml"
            manifest_path = root / "manifest.json"
            if not all(path.exists() for path in (data_path, generation_path, manifest_path)):
                errors.append(f"Missing synthetic output: {task}/{split}")
                continue
            frame = pd.read_parquet(data_path, columns=["sample_id", "sequence_length"])
            overlap = seen_ids & set(frame["sample_id"])
            if overlap:
                errors.append(f"Duplicate synthetic sample IDs across splits: {task}")
            seen_ids.update(frame["sample_id"])
            expected_length = int(cfg["tasks"][task]["sequence_length"])
            if not (frame["sequence_length"] == expected_length).all():
                errors.append(f"Incorrect sequence length: {task}/{split}")
    return errors


def validate_imdb() -> list[str]:
    errors = []
    root = Path("data/processed/imdb")
    if not root.exists():
        return ["IMDb output is missing."]
    train = pd.read_parquet(root / "train/data.parquet", columns=["source_index", "label"])
    validation = pd.read_parquet(root / "validation/data.parquet", columns=["source_index", "label"])
    test = pd.read_parquet(root / "test_locked/data.parquet", columns=["source_index", "label"])
    if set(train["source_index"]) & set(validation["source_index"]):
        errors.append("IMDb train/validation source-index leakage detected.")
    if len(train) != 22500 or len(validation) != 2500 or len(test) != 25000:
        errors.append("IMDb split sizes are not 22,500/2,500/25,000.")
    if not (root / "test_locked/DO_NOT_USE_FOR_DEVELOPMENT.txt").exists():
        errors.append("IMDb official-test lock marker is missing.")
    return errors


def validate_hdfs() -> list[str]:
    root = Path("data/processed/hdfs")
    if not (root / "train/data.parquet").exists():
        return []
    errors = []
    ids = {}
    for split in SPLITS:
        frame = pd.read_parquet(root / split / "data.parquet", columns=["block_id", "label"])
        ids[split] = set(frame["block_id"])
        if frame["block_id"].duplicated().any():
            errors.append(f"Duplicate HDFS block IDs within {split}.")
    if ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]:
        errors.append("HDFS block leakage across partitions detected.")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-imdb", action="store_true")
    args = parser.parse_args()

    errors = []
    if not args.skip_synthetic:
        errors.extend(validate_synthetic())
    if not args.skip_imdb:
        errors.extend(validate_imdb())
    errors.extend(validate_hdfs())
    report = {"status": "PASS" if not errors else "FAIL", "errors": errors}
    Path("data/manifests/dataset_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
