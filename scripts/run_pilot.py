"""Run the Section 15 pilot experiment from its YAML configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from budgetmem.experiments.pilot import run_pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/pilot.yaml"),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a reduced infrastructure validation, not the research pilot.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume compatible checkpoints when present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate = run_pilot(args.config, smoke=args.smoke, resume=args.resume)
    print("\nSECTION 15 PILOT DECISION")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
