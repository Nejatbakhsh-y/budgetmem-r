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
