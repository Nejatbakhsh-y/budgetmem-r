"""Run the Section 14 pretraining gates and save a machine-readable report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "reports" / "evidence" / "pretraining_gate_report.json"


def _required_dataset_paths() -> list[Path]:
    synthetic_config_path = REPO_ROOT / "configs" / "data" / "synthetic.yaml"
    synthetic_root = REPO_ROOT / "data" / "processed" / "synthetic"
    synthetic_tasks: tuple[str, ...] = ()

    if synthetic_config_path.exists():
        import yaml

        config = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
        synthetic_root = REPO_ROOT / str(
            config.get("output_root", "data/processed/synthetic")
        )
        synthetic_tasks = tuple(
            str(name)
            for name, values in config.get("tasks", {}).items()
            if bool(values.get("enabled", False))
        )

    paths = [
        synthetic_root / task / split / "data.parquet"
        for task in synthetic_tasks
        for split in ("train", "validation", "test")
    ]
    paths.extend(
        REPO_ROOT / "data" / "processed" / "hdfs" / split / "data.parquet"
        for split in ("train", "validation", "test")
    )
    paths.extend(
        (REPO_ROOT / "data" / "processed" / "imdb" / split / "data.parquet")
        for split in ("train", "validation")
    )
    paths.append(
        REPO_ROOT / "data" / "processed" / "imdb" / "test_locked" / "data.parquet"
    )
    return paths


def _run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Fail unless every synthetic, HDFS, and IMDb partition exists.",
    )
    args = parser.parse_args()

    required_paths = _required_dataset_paths()
    missing_paths = [
        str(path.relative_to(REPO_ROOT)) for path in required_paths if not path.exists()
    ]
    data_complete = not missing_paths

    pytest_result = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/models/test_budgetmem_r.py",
            "tests/pretraining",
            "-q",
        ]
    )

    dataset_result: dict[str, object] | None = None
    if data_complete:
        dataset_result = _run(
            [
                sys.executable,
                "scripts/data/validate_datasets.py",
            ]
        )

    failed = int(pytest_result["returncode"]) != 0
    if dataset_result is not None:
        failed = failed or int(dataset_result["returncode"]) != 0
    if args.require_data and not data_complete:
        failed = True

    if failed:
        status = "FAIL"
    elif not data_complete:
        status = "PENDING_DATA"
    else:
        status = "PASS"

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "cpu_only_validation": True,
        "cuda_note": (
            "The current laptop validation is CPU-only. CUDA reproducibility "
            "must be revalidated separately on the future GPU machine because "
            "some GPU kernels may be nondeterministic across hardware, drivers, "
            "or PyTorch builds."
        ),
        "require_data": bool(args.require_data),
        "all_required_dataset_partitions_present": data_complete,
        "missing_dataset_paths": missing_paths,
        "pytest": pytest_result,
        "dataset_validator": dataset_result,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
