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
        key_payload = f"{event_ids[0]}:{event_ids[-1]}:{','.join(nodes)}".encode(
            "utf-8"
        )
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
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
