from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_frame(split, indices: list[int]) -> pd.DataFrame:
    rows = []
    for source_index in indices:
        item = split[int(source_index)]
        text = str(item["text"])
        rows.append(
            {
                "source_index": int(source_index),
                "text": text,
                "label": int(item["label"]),
                "character_length": len(text),
                "byte_length_utf8": len(text.encode("utf-8")),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/imdb.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_root = Path(cfg["output_root"])
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    if cfg.get("revision"):
        kwargs["revision"] = cfg["revision"]
    dataset = load_dataset(cfg["dataset_name"], **kwargs)

    labels = list(map(int, dataset["train"]["label"]))
    all_indices = list(range(len(labels)))
    train_idx, validation_idx = train_test_split(
        all_indices,
        test_size=float(cfg["validation_fraction"]),
        random_state=int(cfg["split_seed"]),
        stratify=labels,
        shuffle=True,
    )
    train_idx = sorted(map(int, train_idx))
    validation_idx = sorted(map(int, validation_idx))
    test_idx = list(range(len(dataset["test"])))

    split_map = {
        "train": (dataset["train"], train_idx),
        "validation": (dataset["train"], validation_idx),
        "test_locked": (dataset["test"], test_idx),
    }
    files = {}
    for split_name, (source_split, indices) in split_map.items():
        split_root = output_root / split_name
        split_root.mkdir(parents=True, exist_ok=True)
        path = split_root / "data.parquet"
        frame = make_frame(source_split, indices)
        frame.to_parquet(path, index=False)
        files[split_name] = {
            "path": str(path),
            "rows": len(frame),
            "sha256": sha256(path),
            "positive": int(frame["label"].sum()),
            "negative": int((frame["label"] == 0).sum()),
        }
        print(f"Wrote {path} ({len(frame)} rows)")

    (output_root / "test_locked" / "DO_NOT_USE_FOR_DEVELOPMENT.txt").write_text(
        "The official IMDb test set is reserved for one final evaluation after all model and hyperparameter decisions are frozen.\n",
        encoding="utf-8",
    )
    split_payload = {
        "dataset_name": cfg["dataset_name"],
        "split_seed": int(cfg["split_seed"]),
        "validation_fraction": float(cfg["validation_fraction"]),
        "train_source_indices": train_idx,
        "validation_source_indices": validation_idx,
        "official_test_source_indices": test_idx,
    }
    Path(cfg["split_indices"]).write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": cfg["dataset_name"],
        "revision_requested": cfg.get("revision"),
        "train_fingerprint": getattr(dataset["train"], "_fingerprint", None),
        "test_fingerprint": getattr(dataset["test"], "_fingerprint", None),
        "sequence_limits": cfg["sequence_limits"],
        "input_mode": cfg["input_mode"],
        "official_test_locked": bool(cfg["lock_official_test"]),
        "files": files,
    }
    Path(cfg["manifest"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {cfg['manifest']}")


if __name__ == "__main__":
    main()
