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
