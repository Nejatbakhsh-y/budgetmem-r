#!/usr/bin/env bash
# Section 18.2 — BGL loader + dedicated Section 18 runner release gate
#
# Run this file from the budgetmem-r repository root in the VS Code WSL terminal.
#
# This automation:
#   1. Creates a deterministic BGL parser, preparation pipeline, and PyTorch dataset.
#   2. Creates scripts/run_section18.py as the dedicated Section 18 cell runner.
#   3. Connects the runner to the established Section 15 pilot backend and the
#      controlled Section 12 baseline registry.
#   4. Creates focused contract tests.
#   5. Prepares a local BGL fixture and validates split isolation.
#   6. Runs one substantive synthetic training/evaluation cell.
#   7. Re-runs the Section 18 readiness audit when available.
#
# It does NOT launch the full Section 18 matrix.
# It does NOT stage, commit, push, or delete the approximately 3,032 generated files.

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

AUTOMATION_NAME="28_section18_bgl_runner_release_gate.sh"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
RUN_RUFF="${RUN_RUFF:-1}"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$ROOT" ]] || die "Run this automation from inside the budgetmem-r Git repository."
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  die "Python was not found."
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=2026
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_ROOT="reports/evidence/backups/section18_runner_release_${STAMP}"
EVIDENCE_ROOT="reports/evidence/section18"
RUNTIME_ROOT="artifacts/section18/release_gate"
FIXTURE_RAW="$RUNTIME_ROOT/bgl_fixture/BGL.log"
FIXTURE_OUT="$RUNTIME_ROOT/bgl_processed"
SMOKE_CONFIG="configs/experiments/section18/smoke_single_cell.yaml"
SMOKE_RUN_DIR="$RUNTIME_ROOT/smoke_single_cell"
SMOKE_LOG="$EVIDENCE_ROOT/section18_single_cell_smoke_${STAMP}.log"
REPORT="$EVIDENCE_ROOT/section18_runner_release_gate_${STAMP}.txt"
LATEST_REPORT="$EVIDENCE_ROOT/section18_runner_release_gate_latest.txt"

mkdir -p "$BACKUP_ROOT" "$EVIDENCE_ROOT" "$RUNTIME_ROOT" \
  src/budgetmem/data scripts/data scripts tests configs/experiments/section18

required_existing=(
  "configs/experiments/pilot.yaml"
  "scripts/run_pilot.py"
  "src/budgetmem/experiments/pilot.py"
  "src/budgetmem/models/budgetmem_r.py"
  "src/budgetmem/baselines/controlled.py"
)
for path in "${required_existing[@]}"; do
  [[ -f "$path" ]] || die "Required established implementation is missing: $path"
done

backup_if_present() {
  local path="$1"
  if [[ -e "$path" ]]; then
    mkdir -p "$BACKUP_ROOT/$(dirname "$path")"
    cp -a "$path" "$BACKUP_ROOT/$path"
  fi
}

for path in \
  src/budgetmem/data/__init__.py \
  src/budgetmem/data/bgl.py \
  scripts/data/prepare_bgl.py \
  scripts/run_section18.py \
  tests/test_section18_release_gate.py \
  "$SMOKE_CONFIG"; do
  backup_if_present "$path"
done

log "Repository: $ROOT"
log "Python: $PYTHON"
log "Backup directory: $BACKUP_ROOT"
log "Installing the BGL data pipeline and dedicated runner."

cat > src/budgetmem/data/bgl.py <<'PY'
"""Deterministic BGL log preparation and loading for Section 18.

The parser accepts the common whitespace-delimited BGL/Blue Gene/L log shape,
but intentionally preserves the original line and tolerates shorter variants.
Prepared examples are non-overlapping fixed-length event sequences written as
JSON Lines. Parquet mirrors are written when a parquet engine is available.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - preparation can run without torch
    torch = None
    Tensor = Any

    class Dataset:  # type: ignore[no-redef]
        pass


_NORMAL_LABELS = {"-", "0", "normal", "false", "none", "ok"}
_TOKEN_CLEAN = re.compile(r"\b(?:0x[0-9a-f]+|\d+)\b", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class BGLEvent:
    event_id: int
    label_raw: str
    anomaly: int
    timestamp: str
    date: str
    node: str
    time: str
    node_repeat: str
    event_type: str
    component: str
    level: str
    message: str
    raw_line: str


@dataclass(frozen=True)
class BGLPreparationSummary:
    raw_path: str
    output_dir: str
    seed: int
    sequence_length: int
    stride: int
    source_line_count: int
    parsed_event_count: int
    sequence_count: int
    split_counts: dict[str, int]
    anomaly_sequence_counts: dict[str, int]
    malformed_line_count: int
    source_sha256: str
    output_sha256: dict[str, str]
    split_event_intersections: dict[str, int]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bgl_line(line: str, event_id: int) -> BGLEvent | None:
    """Parse one BGL line without discarding unrecognized message content."""
    raw = line.rstrip("\n\r")
    if not raw.strip():
        return None

    parts = raw.split(maxsplit=9)
    if len(parts) < 2:
        return None

    padded = parts + [""] * (10 - len(parts))
    label_raw = padded[0]
    anomaly = 0 if label_raw.strip().lower() in _NORMAL_LABELS else 1
    return BGLEvent(
        event_id=int(event_id),
        label_raw=label_raw,
        anomaly=anomaly,
        timestamp=padded[1],
        date=padded[2],
        node=padded[3] or "UNKNOWN_NODE",
        time=padded[4],
        node_repeat=padded[5],
        event_type=padded[6],
        component=padded[7],
        level=padded[8],
        message=padded[9],
        raw_line=raw,
    )


def normalize_event(event: BGLEvent) -> str:
    text = " ".join(
        part
        for part in (
            event.event_type,
            event.component,
            event.level,
            event.message,
        )
        if part
    ).lower()
    text = _TOKEN_CLEAN.sub("<num>", text)
    return _WHITESPACE.sub(" ", text).strip() or "<empty>"


def event_token_id(event: BGLEvent, vocab_size: int = 32768) -> int:
    if vocab_size < 16:
        raise ValueError("vocab_size must be at least 16")
    payload = normalize_event(event).encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return 2 + value % (vocab_size - 2)


def read_bgl_events(path: str | Path) -> tuple[list[BGLEvent], int, int]:
    raw_path = Path(path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"BGL raw file not found: {raw_path}")

    events: list[BGLEvent] = []
    source_lines = 0
    malformed = 0
    with raw_path.open("r", encoding="utf-8", errors="replace") as handle:
        for source_lines, line in enumerate(handle, start=1):
            event = parse_bgl_line(line, event_id=source_lines - 1)
            if event is None:
                malformed += 1
                continue
            events.append(event)
    return events, source_lines, malformed


def _sequence_records(
    events: Sequence[BGLEvent],
    *,
    sequence_length: int,
    stride: int,
    vocab_size: int,
) -> list[dict[str, Any]]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    records: list[dict[str, Any]] = []
    for start in range(0, len(events), stride):
        chunk = list(events[start : start + sequence_length])
        if not chunk:
            continue
        if len(chunk) < sequence_length:
            break
        event_ids = [event.event_id for event in chunk]
        token_ids = [event_token_id(event, vocab_size=vocab_size) for event in chunk]
        label = int(any(event.anomaly for event in chunk))
        nodes = sorted({event.node for event in chunk})
        key_payload = f"{event_ids[0]}:{event_ids[-1]}:{','.join(nodes)}".encode("utf-8")
        sequence_id = hashlib.sha256(key_payload).hexdigest()[:24]
        records.append(
            {
                "sequence_id": sequence_id,
                "input_ids": token_ids,
                "attention_mask": [1] * len(token_ids),
                "label": label,
                "event_ids": event_ids,
                "nodes": nodes,
                "start_event_id": event_ids[0],
                "end_event_id": event_ids[-1],
                "anomaly_event_count": sum(event.anomaly for event in chunk),
            }
        )
    return records


def _deterministic_split(
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in (0, 1)")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    ordered = sorted(
        records,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['sequence_id']}".encode("utf-8")
        ).hexdigest(),
    )
    count = len(ordered)
    if count == 0:
        raise ValueError("BGL preparation produced zero complete sequences")

    if count >= 3:
        train_count = max(1, int(math.floor(count * train_ratio)))
        validation_count = max(1, int(math.floor(count * validation_ratio)))
        if train_count + validation_count >= count:
            train_count = max(1, count - 2)
            validation_count = 1
    else:
        train_count = max(1, count - 1)
        validation_count = 0

    return {
        "train": ordered[:train_count],
        "validation": ordered[train_count : train_count + validation_count],
        "test": ordered[train_count + validation_count :],
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _write_optional_parquet(path: Path, records: list[dict[str, Any]]) -> bool:
    try:
        import pandas as pd

        frame = pd.DataFrame(records)
        frame.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def _event_set(records: Sequence[dict[str, Any]]) -> set[int]:
    values: set[int] = set()
    for row in records:
        values.update(int(value) for value in row["event_ids"])
    return values


def prepare_bgl(
    raw_path: str | Path,
    output_dir: str | Path,
    *,
    sequence_length: int = 1024,
    stride: int | None = None,
    seed: int = 2026,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    vocab_size: int = 32768,
) -> BGLPreparationSummary:
    """Prepare deterministic non-overlapping BGL sequence splits."""
    raw = Path(raw_path).resolve()
    out = Path(output_dir)
    actual_stride = int(stride if stride is not None else sequence_length)
    if actual_stride < sequence_length:
        raise ValueError(
            "stride must be at least sequence_length so split examples cannot share events"
        )

    events, source_lines, malformed = read_bgl_events(raw)
    records = _sequence_records(
        events,
        sequence_length=int(sequence_length),
        stride=actual_stride,
        vocab_size=int(vocab_size),
    )
    splits = _deterministic_split(
        records,
        seed=int(seed),
        train_ratio=float(train_ratio),
        validation_ratio=float(validation_ratio),
    )

    train_events = _event_set(splits["train"])
    validation_events = _event_set(splits["validation"])
    test_events = _event_set(splits["test"])
    intersections = {
        "train_validation": len(train_events & validation_events),
        "train_test": len(train_events & test_events),
        "validation_test": len(validation_events & test_events),
    }
    if any(intersections.values()):
        raise RuntimeError(f"BGL split event leakage detected: {intersections}")

    out.mkdir(parents=True, exist_ok=True)
    output_sha256: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    anomaly_counts: dict[str, int] = {}
    parquet_written: dict[str, bool] = {}
    for split_name, split_records in splits.items():
        jsonl_path = out / f"{split_name}.jsonl"
        _write_jsonl(jsonl_path, split_records)
        output_sha256[jsonl_path.name] = sha256_path(jsonl_path)
        split_counts[split_name] = len(split_records)
        anomaly_counts[split_name] = sum(int(row["label"]) for row in split_records)
        parquet_written[split_name] = _write_optional_parquet(
            out / f"{split_name}.parquet", split_records
        )

    summary = BGLPreparationSummary(
        raw_path=str(raw),
        output_dir=str(out.resolve()),
        seed=int(seed),
        sequence_length=int(sequence_length),
        stride=actual_stride,
        source_line_count=source_lines,
        parsed_event_count=len(events),
        sequence_count=len(records),
        split_counts=split_counts,
        anomaly_sequence_counts=anomaly_counts,
        malformed_line_count=malformed,
        source_sha256=sha256_path(raw),
        output_sha256=output_sha256,
        split_event_intersections=intersections,
    )
    manifest = asdict(summary)
    manifest["parquet_written"] = parquet_written
    manifest["format"] = "jsonl_authoritative"
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def load_bgl_records(
    processed_dir: str | Path,
    split: str,
) -> list[dict[str, Any]]:
    split_name = str(split).lower()
    if split_name not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    path = Path(processed_dir) / f"{split_name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared BGL split not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("input_ids"), list):
                raise ValueError(f"Invalid input_ids at {path}:{line_number}")
            records.append(row)
    return records


class BGLSequenceDataset(Dataset):
    """PyTorch dataset over prepared BGL JSONL sequences."""

    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        *,
        sequence_length: int | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to construct BGLSequenceDataset")
        self.records = load_bgl_records(processed_dir, split)
        self.sequence_length = int(sequence_length) if sequence_length else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.records[index]
        input_ids = [int(value) for value in row["input_ids"]]
        if self.sequence_length is not None:
            input_ids = input_ids[: self.sequence_length]
            padding = self.sequence_length - len(input_ids)
            if padding > 0:
                input_ids = input_ids + [0] * padding
        attention_mask = [1 if value != 0 else 0 for value in input_ids]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


__all__ = [
    "BGLEvent",
    "BGLPreparationSummary",
    "BGLSequenceDataset",
    "event_token_id",
    "load_bgl_records",
    "normalize_event",
    "parse_bgl_line",
    "prepare_bgl",
    "read_bgl_events",
]
PY

# Preserve existing public APIs and add the BGL API once.
touch src/budgetmem/data/__init__.py
if ! grep -q 'from budgetmem.data.bgl import' src/budgetmem/data/__init__.py; then
  cat >> src/budgetmem/data/__init__.py <<'PY'

# Section 18 BGL public API.
from budgetmem.data.bgl import (
    BGLEvent as BGLEvent,
    BGLPreparationSummary as BGLPreparationSummary,
    BGLSequenceDataset as BGLSequenceDataset,
    load_bgl_records as load_bgl_records,
    parse_bgl_line as parse_bgl_line,
    prepare_bgl as prepare_bgl,
)
PY
fi

cat > scripts/data/prepare_bgl.py <<'PY'
#!/usr/bin/env python3
"""Prepare deterministic BGL splits for Section 18."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from budgetmem.data.bgl import prepare_bgl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Raw BGL log file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/bgl"),
        help="Prepared output directory",
    )
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--vocab-size", type=int, default=32768)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_bgl(
        args.input,
        args.output,
        sequence_length=args.sequence_length,
        stride=args.stride,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        vocab_size=args.vocab_size,
    )
    payload = asdict(summary)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if any(payload["split_event_intersections"].values()):
        return 2
    if payload["sequence_count"] <= 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
chmod +x scripts/data/prepare_bgl.py

cat > scripts/run_section18.py <<'PY'
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
        raise ValueError(f"Model is not supported by the pilot backend: {cell['model']}")

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
        training["validation_samples"] = min(int(training.get("validation_samples", 24)), 24)
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
    model_seed = int(cell["seed"]) + stable_int(
        f"section18:{cell['task']}:{pilot_model}:{cell['run_id']}"
    ) % 1_000_000
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
    record_payload = asdict(training_record) if is_dataclass(training_record) else dict(training_record)
    duration = time.perf_counter() - started
    payload = {
        "status": "PASS",
        "backend": "pilot",
        "source_config": str(config_path.resolve()),
        "effective_config": str(effective_path.resolve()),
        "cell": {key: str(value) if isinstance(value, Path) else value for key, value in cell.items()},
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
        kwargs = {
            name: values[name]
            for name in signature.parameters
            if name in values
        }
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
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
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


def _classification_metrics(targets: list[int], predictions: list[int]) -> dict[str, float]:
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
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
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
            "cell": {key: str(value) if isinstance(value, Path) else value for key, value in cell.items()},
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
        metrics = {"token_accuracy": token_correct / token_total if token_total else 0.0}
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
        "cell": {key: str(value) if isinstance(value, Path) else value for key, value in cell.items()},
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
    if cell["model"] in CONTROLLED_MODELS or cell["model"] in {"budgetmem_r", "budgetmem-r"}:
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
        "cell": {key: str(value) if isinstance(value, Path) else value for key, value in cell.items()},
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
        json.dumps(validation_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
PY
chmod +x scripts/run_section18.py

cat > tests/test_section18_release_gate.py <<'PY'
"""Focused contracts for the Section 18 BGL and runner release gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from budgetmem.data.bgl import load_bgl_records, parse_bgl_line, prepare_bgl


def test_bgl_parser_preserves_normal_and_anomaly_labels() -> None:
    normal = parse_bgl_line(
        "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 12:02:50 R02-M1-N0-C:J12-U11 RAS KERNEL INFO normal message",
        0,
    )
    anomaly = parse_bgl_line(
        "KERNDTLB 1117838571 2005.06.03 R02-M1-N0-C:J12-U11 12:02:51 R02-M1-N0-C:J12-U11 RAS KERNEL FATAL anomalous message",
        1,
    )
    assert normal is not None and normal.anomaly == 0
    assert anomaly is not None and anomaly.anomaly == 1
    assert "anomalous message" in anomaly.message


def test_bgl_preparation_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    raw = tmp_path / "BGL.log"
    lines = []
    for index in range(48):
        label = "KERNDTLB" if index % 11 == 0 else "-"
        level = "FATAL" if label != "-" else "INFO"
        lines.append(
            f"{label} {1117838570 + index} 2005.06.03 NODE{index % 8:02d} "
            f"12:02:{index % 60:02d} NODE{index % 8:02d} RAS KERNEL {level} event {index}"
        )
    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
    first = tmp_path / "first"
    second = tmp_path / "second"
    summary_a = prepare_bgl(raw, first, sequence_length=4, stride=4, seed=2026)
    summary_b = prepare_bgl(raw, second, sequence_length=4, stride=4, seed=2026)
    assert summary_a.output_sha256 == summary_b.output_sha256
    assert not any(summary_a.split_event_intersections.values())
    assert sum(summary_a.split_counts.values()) == summary_a.sequence_count
    for split in ("train", "validation", "test"):
        assert load_bgl_records(first, split) == load_bgl_records(second, split)


def test_dedicated_runner_exposes_all_section18_aliases() -> None:
    path = Path("scripts/run_section18.py")
    spec = importlib.util.spec_from_file_location("section18_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.validate_model_registry()
    assert result["status"] == "PASS"
    assert not result["missing_aliases"]


def test_smoke_config_is_one_cell() -> None:
    import yaml

    payload = yaml.safe_load(
        Path("configs/experiments/section18/smoke_single_cell.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert payload["experiment"]["run_id"] == "section18_release_smoke"
    assert payload["task"]["name"] == "selective_copy"
    assert payload["model"]["name"] == "budgetmem_r"
    assert payload["memory"]["budget"] == 32
PY

cat > "$SMOKE_CONFIG" <<'YAML'
schema_version: "1.0"
section: 18
experiment:
  name: section18_runner_release_gate
  phase: release_smoke
  run_id: section18_release_smoke
  output_dir: artifacts/section18/release_gate/smoke_single_cell

task:
  name: selective_copy
  dataset: synthetic
  sequence_length: 256

data:
  dataset: synthetic
  sequence_length: 256
  split_seed: 2026

model:
  name: budgetmem_r
  embedding_dim: 32
  hidden_dim: 64

memory:
  budget: 32
  retrieval_k: 4
  strict_budget_enforcement: true

training:
  seed: 2026
  deterministic: true
  resume: false

evaluation:
  seed: 2026
  sequence_length: 256
  save_metrics: true
YAML

# Add runtime directories only. Source, tests, configs, and evidence remain visible to Git.
touch .gitignore
if ! grep -q '^artifacts/section18/$' .gitignore; then
  cat >> .gitignore <<'EOF'

# Section 18 runtime outputs
artifacts/section18/
EOF
fi

log "Creating a controlled BGL fixture."
mkdir -p "$(dirname "$FIXTURE_RAW")"
"$PYTHON" - "$FIXTURE_RAW" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = []
for index in range(96):
    abnormal = index % 13 == 0 or index % 29 == 0
    label = "KERNDTLB" if abnormal else "-"
    level = "FATAL" if abnormal else "INFO"
    node = f"R{index % 4:02d}-M{index % 3}-N{index % 8}-C:J{index % 16:02d}-U{index % 12:02d}"
    message = (
        f"uncorrectable translation fault address 0x{index:08x}"
        if abnormal
        else f"kernel heartbeat counter {index} completed"
    )
    lines.append(
        f"{label} {1117838570 + index} 2005.06.03 {node} "
        f"12:{(index // 60) % 60:02d}:{index % 60:02d} {node} RAS KERNEL {level} {message}"
    )
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"WROTE {path} lines={len(lines)}")
PY

log "Compiling generated Python files."
"$PYTHON" -m compileall -q \
  src/budgetmem/data/bgl.py \
  scripts/data/prepare_bgl.py \
  scripts/run_section18.py \
  tests/test_section18_release_gate.py

if [[ "$RUN_RUFF" == "1" ]] && "$PYTHON" -c 'import ruff' >/dev/null 2>&1; then
  log "Formatting and checking controlled files with Ruff."
  "$PYTHON" -m ruff format \
    src/budgetmem/data/bgl.py \
    scripts/data/prepare_bgl.py \
    scripts/run_section18.py \
    tests/test_section18_release_gate.py
  "$PYTHON" -m ruff check \
    src/budgetmem/data/bgl.py \
    scripts/data/prepare_bgl.py \
    scripts/run_section18.py \
    tests/test_section18_release_gate.py
else
  log "Ruff is unavailable or disabled; compilation remains mandatory."
fi

log "Running focused Section 18 release-gate tests."
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q -o addopts='' \
  tests/test_section18_release_gate.py

log "Preparing and validating the controlled BGL fixture."
rm -rf "$FIXTURE_OUT"
"$PYTHON" scripts/data/prepare_bgl.py \
  --input "$FIXTURE_RAW" \
  --output "$FIXTURE_OUT" \
  --sequence-length 4 \
  --stride 4 \
  --seed 2026 \
  > "$EVIDENCE_ROOT/bgl_fixture_preparation_${STAMP}.json"

"$PYTHON" - "$FIXTURE_OUT/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
assert payload["sequence_count"] > 0
assert payload["split_counts"]["train"] > 0
assert payload["split_counts"]["validation"] > 0
assert payload["split_counts"]["test"] > 0
assert not any(payload["split_event_intersections"].values())
for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
    assert (path.parent / name).is_file()
print("BGL fixture preparation: PASS")
print(json.dumps(payload["split_counts"], sort_keys=True))
PY

log "Validating the dedicated runner configuration and registry."
"$PYTHON" scripts/run_section18.py --config "$SMOKE_CONFIG" --validate-only \
  > "$EVIDENCE_ROOT/section18_runner_validation_${STAMP}.json"

log "Running one substantive Section 18 training/evaluation cell."
rm -rf "$SMOKE_RUN_DIR"
set +e
timeout "${TIMEOUT_SECONDS}s" \
  "$PYTHON" scripts/run_section18.py \
  --config "$SMOKE_CONFIG" \
  --smoke \
  2>&1 | tee "$SMOKE_LOG"
SMOKE_EXIT="${PIPESTATUS[0]}"
set -e

if [[ "$SMOKE_EXIT" -eq 124 ]]; then
  die "The single-cell smoke test timed out after ${TIMEOUT_SECONDS} seconds. Review $SMOKE_LOG"
elif [[ "$SMOKE_EXIT" -ne 0 ]]; then
  die "The single-cell smoke test failed with exit code $SMOKE_EXIT. Review $SMOKE_LOG"
fi

"$PYTHON" - "$SMOKE_RUN_DIR/metrics.json" "$SMOKE_RUN_DIR/results.csv" "$SMOKE_LOG" <<'PY'
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

metrics_path, results_path, log_path = map(Path, sys.argv[1:])
assert metrics_path.is_file() and metrics_path.stat().st_size > 0
assert results_path.is_file() and results_path.stat().st_size > 0
assert log_path.is_file() and log_path.stat().st_size > 0
payload = json.loads(metrics_path.read_text(encoding="utf-8"))
assert payload["status"] == "PASS"
assert payload["backend"] == "pilot"
result = payload["result"]
metric_candidates = (
    "token_accuracy",
    "exact_match_accuracy",
    "memory_recall",
    "accuracy",
    "f1",
)
metric_name = next((name for name in metric_candidates if name in result), None)
assert metric_name is not None, sorted(result)
metric_value = float(result[metric_name])
assert math.isfinite(metric_value)
with results_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 1
text = log_path.read_text(encoding="utf-8", errors="replace")
assert re.search(r"^TRAINED task=", text, re.MULTILINE)
assert re.search(r"^EVALUATED task=", text, re.MULTILINE)
assert "Traceback" not in text
print("Section 18 substantive single-cell smoke: PASS")
print(f"Primary observed metric: {metric_name}={metric_value}")
PY

AUDIT_STATUS="NOT_RUN"
AUDIT_REPORT=""
if [[ -f "27_section18_runner_readiness_fast.sh" ]]; then
  log "Re-running the fast Section 18 readiness audit."
  set +e
  bash 27_section18_runner_readiness_fast.sh
  AUDIT_EXIT=$?
  set -e
  AUDIT_REPORT="$EVIDENCE_ROOT/section18_runner_readiness_latest.txt"
  if [[ "$AUDIT_EXIT" -eq 0 ]]; then
    AUDIT_STATUS="PASS"
  else
    AUDIT_STATUS="FAIL"
  fi
elif [[ -f "26_section18_runner_readiness_audit.sh" ]]; then
  log "Re-running the Section 18 readiness audit."
  set +e
  bash 26_section18_runner_readiness_audit.sh
  AUDIT_EXIT=$?
  set -e
  AUDIT_STATUS="$([[ "$AUDIT_EXIT" -eq 0 ]] && printf PASS || printf FAIL)"
else
  AUDIT_STATUS="SKIPPED_NO_AUDIT_SCRIPT"
fi

if [[ "$AUDIT_STATUS" == "FAIL" ]]; then
  die "The implementation and smoke test passed, but the existing readiness audit still failed. Review $AUDIT_REPORT"
fi

log "Writing the release-gate evidence report."
"$PYTHON" - \
  "$REPORT" "$LATEST_REPORT" "$BACKUP_ROOT" "$FIXTURE_OUT/manifest.json" \
  "$SMOKE_RUN_DIR/metrics.json" "$SMOKE_LOG" "$AUDIT_STATUS" "$AUDIT_REPORT" <<'PY'
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

(
    report_path,
    latest_path,
    backup_root,
    bgl_manifest_path,
    smoke_metrics_path,
    smoke_log_path,
    audit_status,
    audit_report,
) = sys.argv[1:]

bgl = json.loads(Path(bgl_manifest_path).read_text(encoding="utf-8"))
smoke = json.loads(Path(smoke_metrics_path).read_text(encoding="utf-8"))
result = smoke.get("result", {})
metric_name = next(
    (
        name
        for name in (
            "token_accuracy",
            "exact_match_accuracy",
            "memory_recall",
            "accuracy",
            "f1",
        )
        if name in result
    ),
    "not_detected",
)
metric_value = result.get(metric_name, "not_detected")

lines = [
    "Section 18 BGL + Dedicated Runner Release Gate",
    "",
    "BGL preparation/loader: PASS",
    f"  Parsed events: {bgl['parsed_event_count']}",
    f"  Prepared sequences: {bgl['sequence_count']}",
    f"  Split counts: {bgl['split_counts']}",
    f"  Split event intersections: {bgl['split_event_intersections']}",
    "Dedicated scripts/run_section18.py: PASS",
    "Section 12/15 model integration: PASS",
    "Substantive single-cell training: PASS",
    f"  Backend: {smoke['backend']}",
    f"  Model: {smoke['cell']['model']}",
    f"  Task: {smoke['cell']['task']}",
    f"  Sequence length: {smoke['cell']['sequence_length']}",
    f"  Memory budget: {smoke['cell']['budget']}",
    f"  Observed metric: {metric_name}={metric_value}",
    f"Existing readiness audit: {audit_status}",
    "",
    "Safety controls",
    "  Full Section 18 matrix launched: NO",
    "  Generated matrix files staged: NO",
    "  Git commit created: NO",
    "  Git push performed: NO",
    f"  Source backup: {backup_root}",
    f"  Smoke log: {smoke_log_path}",
]
if audit_report:
    lines.append(f"  Audit report: {audit_report}")
lines.extend(
    [
        "",
        "FINAL DECISION: SECTION 18 RUNNER RELEASE GATE PASSED",
        "",
        "The two previously identified blockers are resolved. Do not run the",
        "full matrix until the final screen and readiness report have been reviewed.",
    ]
)
Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
shutil.copyfile(report_path, latest_path)
print("\n".join(lines))
PY

log "Checking whitespace only in controlled Section 18 implementation files."
"$PYTHON" - <<'PY'
from pathlib import Path

paths = [
    Path(".gitignore"),
    Path("src/budgetmem/data/__init__.py"),
    Path("src/budgetmem/data/bgl.py"),
    Path("scripts/data/prepare_bgl.py"),
    Path("scripts/run_section18.py"),
    Path("tests/test_section18_release_gate.py"),
    Path("configs/experiments/section18/smoke_single_cell.yaml"),
]

errors = []
for path in paths:
    if not path.is_file():
        errors.append(f"missing controlled file: {path}")
        continue
    text = path.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.endswith((" ", "\t")):
            errors.append(f"{path}:{line_number}: trailing whitespace")
    if text and not text.endswith("\n"):
        errors.append(f"{path}: missing final newline")

if errors:
    raise SystemExit("\n".join(errors))

print("Controlled Section 18 whitespace: PASS")
PY

git status --short -- \
  .gitignore \
  src/budgetmem/data/__init__.py \
  src/budgetmem/data/bgl.py \
  scripts/data/prepare_bgl.py \
  scripts/run_section18.py \
  tests/test_section18_release_gate.py \
  configs/experiments/section18/smoke_single_cell.yaml

cat <<EOF

============================================================
SECTION 18.2 RELEASE GATE COMPLETED
============================================================
Final evidence:
  $LATEST_REPORT

Expected final line:
  FINAL DECISION: SECTION 18 RUNNER RELEASE GATE PASSED

Do not run 21_section18_main_experiment_matrix.sh --execute yet.
Do not stage or commit the approximately 3,032 generated matrix files yet.
Review or provide the final report and terminal screen first.
============================================================
EOF
