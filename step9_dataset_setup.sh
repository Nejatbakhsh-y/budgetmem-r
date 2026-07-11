#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .git ]]; then
  echo "ERROR: Run this from the budgetmem-r repository root." >&2
  exit 1
fi

mkdir -p \
  configs/data \
  scripts/data \
  data/raw/hdfs \
  data/raw/bgl \
  data/interim/hdfs \
  data/processed/synthetic \
  data/processed/hdfs \
  data/processed/imdb \
  data/manifests \
  docs

cat > configs/data/synthetic.yaml <<'YAML'
output_root: data/processed/synthetic
schema_version: "1.0"
split_sizes:
  train: 2000
  validation: 500
  test: 500
split_seeds:
  train: 11001
  validation: 22001
  test: 33001

tasks:
  selective_copy:
    enabled: true
    sequence_length: 256
    vocabulary_size: 128
    number_keys: 8
    number_queries: 1
    delay_length: 32
    distractor_percentage: 80
    number_relevant_events: 8
    random_seed: 101

  associative_recall:
    enabled: true
    sequence_length: 256
    vocabulary_size: 128
    number_keys: 12
    number_queries: 1
    delay_length: 32
    distractor_percentage: 65
    number_relevant_events: 12
    random_seed: 102

  multiple_key_retrieval:
    enabled: true
    sequence_length: 384
    vocabulary_size: 192
    number_keys: 16
    number_queries: 4
    delay_length: 64
    distractor_percentage: 70
    number_relevant_events: 16
    random_seed: 103

  delayed_xor:
    enabled: true
    sequence_length: 256
    vocabulary_size: 32
    number_keys: 2
    number_queries: 1
    delay_length: 128
    distractor_percentage: 90
    number_relevant_events: 2
    random_seed: 104

  rare_event_recall:
    enabled: true
    sequence_length: 512
    vocabulary_size: 128
    number_keys: 8
    number_queries: 2
    delay_length: 128
    distractor_percentage: 95
    number_relevant_events: 4
    random_seed: 105

  distractor_heavy_retrieval:
    enabled: true
    sequence_length: 512
    vocabulary_size: 192
    number_keys: 12
    number_queries: 3
    delay_length: 128
    distractor_percentage: 97
    number_relevant_events: 12
    random_seed: 106

  sequence_reversal:
    enabled: false
    sequence_length: 256
    vocabulary_size: 128
    number_keys: 8
    number_queries: 1
    delay_length: 32
    distractor_percentage: 50
    number_relevant_events: 32
    random_seed: 201

  nested_parentheses:
    enabled: false
    sequence_length: 256
    vocabulary_size: 32
    number_keys: 2
    number_queries: 1
    delay_length: 32
    distractor_percentage: 50
    number_relevant_events: 32
    random_seed: 202
YAML

cat > configs/data/hdfs.yaml <<'YAML'
raw_log: data/raw/hdfs/HDFS.log
labels: data/raw/hdfs/anomaly_label.csv
output_root: data/processed/hdfs
interim_db: data/interim/hdfs/hdfs_sequences.sqlite
manifest: data/manifests/hdfs_manifest.json
split_method: stratified_block_hash
split_salt: budgetmem-r-hdfs-v1-2026
split_proportions:
  train: 0.80
  validation: 0.10
  test: 0.10
dev_fraction: 0.02
parser:
  name: deterministic_regex_template_v1
  batch_size: 20000
  parquet_batch_blocks: 5000
YAML

cat > configs/data/imdb.yaml <<'YAML'
dataset_name: stanfordnlp/imdb
revision: null
output_root: data/processed/imdb
manifest: data/manifests/imdb_manifest.json
split_indices: data/manifests/imdb_split_indices.json
validation_fraction: 0.10
split_seed: 42026
sequence_limits: [1024, 2048, 4096]
input_mode: byte_or_character
lock_official_test: true
YAML

cat > configs/data/bgl.yaml <<'YAML'
enabled: false
role: external_validation_only
raw_root: data/raw/bgl
output_root: data/processed/bgl
activate_only_after:
  - hdfs_pipeline_complete
  - imdb_pipeline_complete
YAML

cat > scripts/data/generate_synthetic.py <<'PY'
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

PAD, BOS, EOS, SEP, QUERY, MARK = 0, 1, 2, 3, 4, 5
DATA_START = 16
REQUIRED_PARAMETERS = {
    "sequence_length",
    "vocabulary_size",
    "number_keys",
    "number_queries",
    "delay_length",
    "distractor_percentage",
    "number_relevant_events",
    "random_seed",
}


def stable_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_parameters(task: str, cfg: dict[str, Any]) -> None:
    missing = REQUIRED_PARAMETERS - set(cfg)
    if missing:
        raise ValueError(f"{task}: missing parameters: {sorted(missing)}")
    if cfg["vocabulary_size"] <= DATA_START + cfg["number_keys"] + 4:
        raise ValueError(f"{task}: vocabulary_size is too small")
    if not 0 <= cfg["distractor_percentage"] <= 100:
        raise ValueError(f"{task}: distractor_percentage must be in [0, 100]")


def random_data(rng: np.random.Generator, vocab: int, size: int) -> list[int]:
    if size <= 0:
        return []
    return rng.integers(DATA_START, vocab, size=size).astype(int).tolist()


def fit(prefix: list[int], suffix: list[int], length: int, rng: np.random.Generator, vocab: int) -> list[int]:
    if len(prefix) + len(suffix) > length:
        raise ValueError(f"Payload length {len(prefix) + len(suffix)} exceeds sequence_length={length}")
    filler = random_data(rng, vocab, length - len(prefix) - len(suffix))
    return prefix + filler + suffix


def choose_nonadjacent(rng: np.random.Generator, start: int, stop: int, count: int) -> list[int]:
    candidates = list(range(start, stop))
    rng.shuffle(candidates)
    selected: list[int] = []
    for candidate in candidates:
        if all(abs(candidate - existing) > 1 for existing in selected):
            selected.append(candidate)
            if len(selected) == count:
                return sorted(selected)
    raise ValueError(f"Cannot place {count} non-adjacent events in [{start}, {stop})")


def key_value_space(cfg: dict[str, Any], rng: np.random.Generator) -> tuple[list[int], list[int]]:
    n_keys = int(cfg["number_keys"])
    keys = list(range(DATA_START, DATA_START + n_keys))
    values = rng.integers(DATA_START + n_keys, int(cfg["vocabulary_size"]), size=n_keys).astype(int).tolist()
    return keys, values


def selective_copy(cfg: dict[str, Any], rng: np.random.Generator) -> tuple[list[int], list[int], list[int], list[int], dict[str, Any]]:
    n = int(cfg["number_relevant_events"])
    values = random_data(rng, int(cfg["vocabulary_size"]), n)
    prefix = [BOS]
    relevant_positions: list[int] = []
    for value in values:
        prefix.extend(random_data(rng, int(cfg["vocabulary_size"]), int(rng.integers(0, 4))))
        prefix.extend([MARK, value])
        relevant_positions.append(len(prefix) - 1)
    suffix = [SEP, QUERY, EOS]
    sequence = fit(prefix, suffix, int(cfg["sequence_length"]), rng, int(cfg["vocabulary_size"]))
    query_positions = [len(sequence) - 2]
    return sequence, values, relevant_positions, query_positions, {}


def associative_recall(cfg: dict[str, Any], rng: np.random.Generator) -> tuple[list[int], list[int], list[int], list[int], dict[str, Any]]:
    keys, values = key_value_space(cfg, rng)
    prefix = [BOS]
    relevant_positions: list[int] = []
    for key, value in zip(keys, values):
        prefix.extend([key, value, SEP])
        relevant_positions.append(len(prefix) - 2)
    qn = min(int(cfg["number_queries"]), len(keys))
    selected = rng.choice(len(keys), size=qn, replace=False).astype(int).tolist()
    suffix = [QUERY] + [keys[i] for i in selected] + [EOS]
    sequence = fit(prefix, suffix, int(cfg["sequence_length"]), rng, int(cfg["vocabulary_size"]))
    query_positions = list(range(len(sequence) - qn - 1, len(sequence) - 1))
    return sequence, [values[i] for i in selected], relevant_positions, query_positions, {"queried_keys": [keys[i] for i in selected]}


def multiple_key_retrieval(cfg: dict[str, Any], rng: np.random.Generator) -> tuple[list[int], list[int], list[int], list[int], dict[str, Any]]:
    keys, values = key_value_space(cfg, rng)
    prefix = [BOS]
    relevant_positions: list[int] = []
    gap_max = max(1, int(cfg["distractor_percentage"]) // 12)
    order = rng.permutation(len(keys)).astype(int).tolist()
    for idx in order:
        prefix.extend(random_data(rng, int(cfg["vocabulary_size"]), int(rng.integers(0, gap_max + 1))))
        prefix.extend([MARK, keys[idx], values[idx]])
        relevant_positions.append(len(prefix) - 1)
    qn = min(int(cfg["number_queries"]), len(keys))
    selected = rng.choice(len(keys), size=qn, replace=False).astype(int).tolist()
    suffix = [SEP, QUERY] + [keys[i] for i in selected] + [EOS]
    sequence = fit(prefix, suffix, int(cfg["sequence_length"]), rng, int(cfg["vocabulary_size"]))
    query_positions = list(range(len(sequence) - qn - 1, len(sequence) - 1))
    return sequence, [values[i] for i in selected], relevant_positions, query_positions, {"queried_keys": [keys[i] for i in selected]}


def delayed_xor(cfg: dict[str, Any], rng: np.random.Generator) -> tuple[list[int], list[int], list[int], list[int], dict[str, Any]]:
    length = int(cfg["sequence_length"])
    delay = int(cfg["delay_length"])
    if delay >= length - 4:
        raise ValueError("delayed_xor: delay_length is too large")
    sequence = random_data(rng, int(cfg["vocabulary_size"]), length)
    p1 = int(rng.integers(1, length - delay - 3))
    p2 = p1 + delay
    bit1, bit2 = int(rng.integers(0, 2)), int(rng.integers(0, 2))
    sequence[0] = BOS
    sequence[p1] = 8 + bit1
    sequence[p2] = 8 + bit2
    sequence[-2:] = [QUERY, EOS]
    return sequence, [bit1 ^ bit2], [p1, p2], [length - 2], {"bit_values": [bit1, bit2]}


def rare_event_recall(cfg: dict[str, Any], rng: np.random.Generator) -> tuple[list[int], list[int], list[int], list[int], dict[str, Any]]:
    length = int(cfg["sequence_length"])
    n = int(cfg["number_relevant_events"])
    qn = min(int(cfg["number_queries"]), n)
    sequence = random_data(rng, int(cfg["vocabulary_size"]), length)
    sequence[0] = BOS
    sequence[-2:] = [QUERY, EOS]
    positions = choose_nonadjacent(rng, 2, length - 3, n)
    values = random_data(rng, int(cfg["vocabulary_size"]), n)
    for pos, value in zip(positions, values):
        sequence[pos] = MARK
        sequence[pos + 1] = value
    target = values[-qn:]
    relevant_positions = [p + 1 for p in positions]
    return sequence, target, relevant_positions, [length - 2], {"rare_event_positions": positions}


def distractor_heavy_retrieval(cfg: dict[str, Any], rng: np.random.Generator) -> tuple[list[int], list[int], list[int], list[int], dict[str, Any]]:
    keys, values = key_value_space(cfg, rng)
    length = int(cfg["sequence_length"])
    qn = min(int(cfg["number_queries"]), len(keys))
    sequence = random_data(rng, int(cfg["vocabulary_size"]), length)
    sequence[0] = BOS
    suffix_length = qn + 2
    selected_slots = choose_nonadjacent(rng, 2, length - suffix_length - 1, len(keys))
    relevant_positions: list[int] = []
    for pos, key, value in zip(selected_slots, keys, values):
        sequence[pos] = key
        sequence[pos + 1] = value
        relevant_positions.append(pos + 1)
    queried = rng.choice(len(keys), size=qn, replace=False).astype(int).tolist()
    suffix = [QUERY] + [keys[i] for i in queried] + [EOS]
    sequence[-len(suffix):] = suffix
    query_positions = list(range(length - qn - 1, length - 1))
    return sequence, [values[i] for i in queried], relevant_positions, query_positions, {"queried_keys": [keys[i] for i in queried]}


def sequence_reversal(cfg: dict[str, Any], rng: np.random.Generator) -> tuple[list[int], list[int], list[int], list[int], dict[str, Any]]:
    n = int(cfg["number_relevant_events"])
    values = random_data(rng, int(cfg["vocabulary_size"]), n)
    prefix = [BOS] + values
    suffix = [SEP, QUERY, EOS]
    sequence = fit(prefix, suffix, int(cfg["sequence_length"]), rng, int(cfg["vocabulary_size"]))
    return sequence, list(reversed(values)), list(range(1, n + 1)), [len(sequence) - 2], {}


def nested_parentheses(cfg: dict[str, Any], rng: np.random.Generator) -> tuple[list[int], list[int], list[int], list[int], dict[str, Any]]:
    length = int(cfg["sequence_length"])
    body_len = max(2, min(int(cfg["number_relevant_events"]), length - 3))
    body_len -= body_len % 2
    opens = body_len // 2
    tokens = [10] * opens + [11] * opens
    rng.shuffle(tokens)
    balance = 0
    valid = True
    for token in tokens:
        balance += 1 if token == 10 else -1
        if balance < 0:
            valid = False
    valid = valid and balance == 0
    sequence = fit([BOS] + tokens, [QUERY, EOS], length, rng, int(cfg["vocabulary_size"]))
    return sequence, [int(valid)], list(range(1, body_len + 1)), [len(sequence) - 2], {}


GENERATORS: dict[str, Callable[..., tuple[list[int], list[int], list[int], list[int], dict[str, Any]]]] = {
    "selective_copy": selective_copy,
    "associative_recall": associative_recall,
    "multiple_key_retrieval": multiple_key_retrieval,
    "delayed_xor": delayed_xor,
    "rare_event_recall": rare_event_recall,
    "distractor_heavy_retrieval": distractor_heavy_retrieval,
    "sequence_reversal": sequence_reversal,
    "nested_parentheses": nested_parentheses,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/synthetic.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])
    enabled = {name: cfg for name, cfg in config["tasks"].items() if cfg.get("enabled", False)}
    aggregate_manifest: list[dict[str, Any]] = []
    expected_primary = {
        "selective_copy",
        "associative_recall",
        "multiple_key_retrieval",
        "delayed_xor",
        "rare_event_recall",
        "distractor_heavy_retrieval",
    }
    if set(enabled) != expected_primary:
        raise ValueError(f"Primary phase must enable exactly six tasks. Enabled={sorted(enabled)}")

    for task_name, task_cfg in enabled.items():
        validate_parameters(task_name, task_cfg)
        task_root = output_root / task_name
        if task_root.exists() and args.overwrite:
            shutil.rmtree(task_root)
        task_root.mkdir(parents=True, exist_ok=True)

        for split, count in config["split_sizes"].items():
            split_root = task_root / split
            split_root.mkdir(parents=True, exist_ok=True)
            split_seed = int(config["split_seeds"][split]) + int(task_cfg["random_seed"])
            rows: list[dict[str, Any]] = []
            for index in range(int(count)):
                example_seed = split_seed * 1_000_003 + index
                rng = np.random.default_rng(example_seed)
                input_ids, target_ids, relevant_positions, query_positions, metadata = GENERATORS[task_name](task_cfg, rng)
                rows.append(
                    {
                        "sample_id": f"{task_name}-{split}-{index:08d}",
                        "task": task_name,
                        "input_ids": input_ids,
                        "target_ids": target_ids,
                        "relevant_positions": relevant_positions,
                        "query_positions": query_positions,
                        "sequence_length": len(input_ids),
                        "example_seed": example_seed,
                        "metadata_json": json.dumps(metadata, sort_keys=True),
                    }
                )

            data_path = split_root / "data.parquet"
            pd.DataFrame(rows).to_parquet(data_path, index=False)
            generation_cfg = {
                "schema_version": config["schema_version"],
                "task": task_name,
                "split": split,
                "split_seed": split_seed,
                "sample_count": int(count),
                "parameters": task_cfg,
            }
            (split_root / "generation_config.yaml").write_text(
                yaml.safe_dump(generation_cfg, sort_keys=False), encoding="utf-8"
            )
            manifest = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "data_file": str(data_path),
                "sha256": stable_hash(data_path),
                "row_count": int(count),
                "split_seed": split_seed,
            }
            (split_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            aggregate_manifest.append({"task": task_name, "split": split, "generation": generation_cfg, "artifact": manifest})
            print(f"Wrote {data_path} ({count} rows)")

    manifest_root = Path("data/manifests")
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "synthetic_manifest.json").write_text(
        json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "partitions": aggregate_manifest}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
PY

cat > scripts/data/prepare_imdb.py <<'PY'
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
PY

cat > scripts/data/prepare_hdfs.py <<'PY'
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

BLOCK_RE = re.compile(r"blk_-?\d+")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b")
HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])")
PREFIX_RE = re.compile(r"^\d{6}\s+\d{6}\s+\d+\s+\w+\s+")

SCHEMA = pa.schema(
    [
        ("block_id", pa.string()),
        ("event_ids", pa.list_(pa.string())),
        ("label", pa.int8()),
        ("sequence_length", pa.int32()),
        ("first_line", pa.int64()),
        ("last_line", pa.int64()),
    ]
)


def normalize_template(line: str) -> str:
    text = PREFIX_RE.sub("", line.strip())
    text = BLOCK_RE.sub("<BLOCK>", text)
    text = IP_RE.sub("<IP>", text)
    text = HEX_RE.sub("<HEX>", text)
    text = NUMBER_RE.sub("<NUM>", text)
    return " ".join(text.split())


def event_id(template: str) -> str:
    return "E" + hashlib.sha1(template.encode("utf-8")).hexdigest()[:12]


def load_labels(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path)
    columns = {str(c).strip().lower(): c for c in frame.columns}
    block_col = columns.get("blockid") or columns.get("block_id")
    label_col = columns.get("label")
    if block_col is None or label_col is None:
        raise ValueError(f"Expected BlockId and Label columns in {path}")
    result = {}
    for block, label in zip(frame[block_col], frame[label_col]):
        label_text = str(label).strip().lower()
        if label_text not in {"normal", "anomaly"}:
            raise ValueError(f"Unexpected HDFS label: {label}")
        result[str(block)] = int(label_text == "anomaly")
    return result


def assign_split(block_id: str, label: int, cfg: dict) -> str:
    salt = str(cfg["split_salt"])
    token = f"{salt}|{label}|{block_id}".encode("utf-8")
    value = int(hashlib.sha256(token).hexdigest()[:16], 16) / float(16**16)
    train_p = float(cfg["split_proportions"]["train"])
    val_p = float(cfg["split_proportions"]["validation"])
    if value < train_p:
        return "train"
    if value < train_p + val_p:
        return "validation"
    return "test"


def rows_from_db(connection: sqlite3.Connection, labels: dict[str, int]) -> Iterator[tuple[str, list[str], int, int, int]]:
    cursor = connection.execute("SELECT block_id, line_no, event_id FROM events ORDER BY block_id, line_no")
    current_block = None
    events: list[str] = []
    first_line = last_line = 0
    for block_id, line_no, eid in cursor:
        if current_block is not None and block_id != current_block:
            if current_block in labels:
                yield current_block, events, labels[current_block], first_line, last_line
            events = []
        if block_id != current_block:
            current_block = block_id
            first_line = int(line_no)
        events.append(str(eid))
        last_line = int(line_no)
    if current_block is not None and current_block in labels:
        yield current_block, events, labels[current_block], first_line, last_line


def write_batch(writer: pq.ParquetWriter, rows: list[dict]) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
        rows.clear()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/hdfs.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-lines", type=int, default=None, help="Development-only line cap. Do not use for final paper data.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw_log = Path(cfg["raw_log"])
    label_path = Path(cfg["labels"])
    output_root = Path(cfg["output_root"])
    db_path = Path(cfg["interim_db"])

    if not raw_log.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"Required files are missing. Expected:\n  {raw_log}\n  {label_path}\n"
            "Use HDFS_v1, not HDFS_v2."
        )
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    labels = load_labels(label_path)
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE events (block_id TEXT NOT NULL, line_no INTEGER NOT NULL, event_id TEXT NOT NULL)")

    templates: dict[str, str] = {}
    batch: list[tuple[str, int, str]] = []
    batch_size = int(cfg["parser"]["batch_size"])
    parsed_lines = 0
    with raw_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            blocks = sorted(set(BLOCK_RE.findall(line)))
            if blocks:
                template = normalize_template(line)
                eid = event_id(template)
                templates[eid] = template
                for block in blocks:
                    batch.append((block, line_no, eid))
                if len(batch) >= batch_size:
                    connection.executemany("INSERT INTO events VALUES (?, ?, ?)", batch)
                    connection.commit()
                    batch.clear()
            parsed_lines = line_no
            if args.max_lines and line_no >= args.max_lines:
                break
    if batch:
        connection.executemany("INSERT INTO events VALUES (?, ?, ?)", batch)
        connection.commit()
    connection.execute("CREATE INDEX idx_events_block_line ON events(block_id, line_no)")
    connection.commit()

    writers = {}
    buffers = {split: [] for split in ("train", "validation", "test")}
    dev_buffers = {split: [] for split in ("train", "validation", "test")}
    counters = Counter()
    batch_blocks = int(cfg["parser"]["parquet_batch_blocks"])
    for split in buffers:
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        writers[split] = pq.ParquetWriter(split_dir / "data.parquet", SCHEMA, compression="zstd")

    for block_id, events, label, first_line, last_line in rows_from_db(connection, labels):
        split = assign_split(block_id, label, cfg)
        row = {
            "block_id": block_id,
            "event_ids": events,
            "label": int(label),
            "sequence_length": len(events),
            "first_line": first_line,
            "last_line": last_line,
        }
        buffers[split].append(row)
        counters[f"{split}_blocks"] += 1
        counters[f"{split}_anomalies"] += int(label)
        dev_token = int(hashlib.sha256(f"dev|{block_id}".encode()).hexdigest()[:16], 16) / float(16**16)
        if dev_token < float(cfg["dev_fraction"]):
            dev_buffers[split].append(row)
        if len(buffers[split]) >= batch_blocks:
            write_batch(writers[split], buffers[split])

    for split in buffers:
        write_batch(writers[split], buffers[split])
        writers[split].close()
        dev_dir = output_root / "development_subset" / split
        dev_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dev_buffers[split]).to_parquet(dev_dir / "data.parquet", index=False)

    connection.close()
    templates_path = output_root / "event_templates.csv"
    pd.DataFrame(sorted(templates.items()), columns=["event_id", "template"]).to_csv(templates_path, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "LogHub HDFS_v1",
        "raw_log": str(raw_log),
        "labels": str(label_path),
        "parsed_lines": parsed_lines,
        "development_line_cap": args.max_lines,
        "parser": cfg["parser"],
        "split_method": cfg["split_method"],
        "split_salt": cfg["split_salt"],
        "counts": dict(counters),
        "template_count": len(templates),
        "label_count": len(labels),
    }
    Path(cfg["manifest"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
PY

cat > scripts/data/validate_datasets.py <<'PY'
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
PY

cat > docs/dataset_protocol.md <<'MD'
# Dataset Protocol

## Primary datasets

1. Six deterministic synthetic sequence-memory tasks.
2. LogHub HDFS_v1 block-level anomaly detection.
3. Stanford Large Movie Review Dataset binary sentiment classification.

## Deferred datasets

- Sequence reversal and nested parentheses remain disabled until the six-task pipeline is stable.
- BGL is reserved for external validation after HDFS and IMDb experiments are complete.

## Leakage controls

- Synthetic train, validation, and test partitions use separate split seeds.
- HDFS partitioning occurs only at the block-ID level. A block cannot appear in multiple partitions.
- IMDb uses a fixed stratified 90/10 split of the official training data. The official test set remains locked until final evaluation.

## HDFS metrics

F1, precision, recall, area under the precision-recall curve, area under the ROC curve, false-positive rate, peak memory, latency per block, and event throughput.

## IMDb metrics

Accuracy, macro F1, negative log-likelihood, expected calibration error, latency, throughput, and peak GPU memory.
MD

cat >> .gitignore <<'TXT'

# Step 9 datasets: never commit downloaded or generated data
/data/raw/**
/data/interim/**
/data/processed/**
!/data/raw/.gitkeep
!/data/interim/.gitkeep
!/data/processed/.gitkeep

# Keep reproducibility metadata
!/data/manifests/
!/data/manifests/**
TXT

find data/raw data/interim data/processed -type d -exec touch {}/.gitkeep \;

python -m py_compile \
  scripts/data/generate_synthetic.py \
  scripts/data/prepare_imdb.py \
  scripts/data/prepare_hdfs.py \
  scripts/data/validate_datasets.py
find scripts/data -type d -name __pycache__ -prune -exec rm -rf {} +

echo "Step 9 files created successfully."
echo "Next: run the generation commands shown in the instructions."
