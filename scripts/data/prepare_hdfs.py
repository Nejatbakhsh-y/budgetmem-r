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
