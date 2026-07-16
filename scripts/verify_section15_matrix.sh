#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$PROJECT_ROOT"

SOURCE_PATH="${1:-}"

mkdir -p reports/evidence

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: Python was not found."
    exit 1
fi

"$PYTHON_BIN" - "$PROJECT_ROOT" "$SOURCE_PATH" <<'PY' 
from __future__ import annotations

import csv
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(sys.argv[1]).resolve()
SOURCE_ARGUMENT = sys.argv[2].strip()

EXPECTED_TASKS = (
    "selective_copy",
    "associative_recall",
    "distractor_retrieval",
)

EXPECTED_MODELS = (
    "gru",
    "gru_uniform",
    "gru_reservoir",
    "budgetmem_r",
)

MODEL_DISPLAY = {
    "gru": "GRU",
    "gru_uniform": "GRU + uniform cache",
    "gru_reservoir": "GRU + reservoir cache",
    "budgetmem_r": "BudgetMem-R",
}

EXPECTED_LENGTHS = (256, 512, 1024)
EXPECTED_BUDGETS = (16, 32)
EXPECTED_SEED = 2026

EVIDENCE_DIR = PROJECT_ROOT / "reports" / "evidence"
REPORT_PATH = EVIDENCE_DIR / "section15_matrix_coverage.txt"
CSV_PATH = EVIDENCE_DIR / "section15_matrix_coverage.csv"
JSON_PATH = EVIDENCE_DIR / "section15_matrix_coverage.json"

GENERATED_NAMES = {
    REPORT_PATH.name,
    CSV_PATH.name,
    JSON_PATH.name,
}

TASK_KEYS = (
    "task",
    "task_name",
    "dataset",
    "dataset_name",
    "benchmark",
)

MODEL_KEYS = (
    "model",
    "model_name",
    "method",
    "architecture",
    "policy",
)

LENGTH_KEYS = (
    "sequence_length",
    "seq_length",
    "seq_len",
    "length",
    "context_length",
)

BUDGET_KEYS = (
    "memory_budget",
    "budget",
    "mem_budget",
    "cache_budget",
    "capacity",
)

SEED_KEYS = (
    "seed",
    "random_seed",
    "rng_seed",
)

STATUS_KEYS = (
    "status",
    "run_status",
    "outcome",
    "state",
)

FAILURE_WORDS = {
    "failed",
    "failure",
    "error",
    "crashed",
    "unstable",
    "nan",
    "oom",
    "aborted",
    "incomplete",
}


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def normalize_task(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple)):
        return None

    text = normalize_key(value)
    text = text.removesuffix("_py")

    aliases = {
        "selectivecopy": "selective_copy",
        "selective_copy_task": "selective_copy",
        "associativerecall": "associative_recall",
        "associative_recall_task": "associative_recall",
        "distractorretrieval": "distractor_retrieval",
        "distractor_retrieval_task": "distractor_retrieval",
        "distractorheavyretrieval": "distractor_retrieval",
        "distractor_heavy_retrieval": "distractor_retrieval",
        "distractor_heavy_retrieval_task": "distractor_retrieval",
    }

    text = aliases.get(text, text)

    for task in EXPECTED_TASKS:
        if task in text:
            return task

    return None


def normalize_model(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple)):
        return None

    text = normalize_key(value)

    if "budgetmem" in text:
        return "budgetmem_r"

    if "gru" in text and "reservoir" in text:
        return "gru_reservoir"

    if "gru" in text and "uniform" in text:
        return "gru_uniform"

    if text in {
        "gru",
        "baseline_gru",
        "vanilla_gru",
        "plain_gru",
        "gru_baseline",
    }:
        return "gru"

    return None


def normalize_integer(value: Any) -> int | None:
    if value is None or isinstance(value, (dict, list, tuple)):
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value) if value.is_integer() else None

    text = str(value).replace(",", "").strip()
    match = re.search(r"-?\d+", text)

    return int(match.group()) if match else None


def get_first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    normalized = {normalize_key(key): value for key, value in record.items()}

    for key in keys:
        if key in normalized:
            return normalized[key]

    return None


def record_is_failure(record: dict[str, Any]) -> bool:
    success_value = get_first(record, ("success", "completed", "passed"))

    if isinstance(success_value, bool):
        return not success_value

    status = get_first(record, STATUS_KEYS)

    if status is None:
        return False

    status_text = normalize_key(status)
    return any(word in status_text for word in FAILURE_WORDS)


def extract_record(record: dict[str, Any], source: Path) -> dict[str, Any] | None:
    task = normalize_task(get_first(record, TASK_KEYS))
    model = normalize_model(get_first(record, MODEL_KEYS))
    sequence_length = normalize_integer(get_first(record, LENGTH_KEYS))
    budget = normalize_integer(get_first(record, BUDGET_KEYS))
    seed = normalize_integer(get_first(record, SEED_KEYS))

    if None in (task, model, sequence_length, budget):
        return None

    return {
        "task": task,
        "model": model,
        "sequence_length": sequence_length,
        "budget": budget,
        "seed": seed,
        "failed": record_is_failure(record),
        "source": str(source.relative_to(PROJECT_ROOT)),
    }


def walk_json_object(value: Any, source: Path, output: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        extracted = extract_record(value, source)

        if extracted is not None:
            output.append(extracted)

        for nested_value in value.values():
            walk_json_object(nested_value, source, output)

    elif isinstance(value, list):
        for item in value:
            walk_json_object(item, source, output)


def parse_csv_file(path: Path, output: list[dict[str, Any]]) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)

            for row in reader:
                extracted = extract_record(dict(row), path)

                if extracted is not None:
                    output.append(extracted)
    except (UnicodeDecodeError, csv.Error, OSError):
        return


def parse_json_file(path: Path, output: list[dict[str, Any]]) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)

        walk_json_object(value, path, output)
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return


def parse_jsonl_file(path: Path, output: list[dict[str, Any]]) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()

                if not line:
                    continue

                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue

                walk_json_object(value, path, output)
    except (UnicodeDecodeError, OSError):
        return


def infer_from_text(text: str, source: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    task_patterns = {
        "selective_copy": r"\bselective[_ -]?copy\b",
        "associative_recall": r"\bassociative[_ -]?recall\b",
        "distractor_retrieval": r"\bdistractor[_ -]?retrieval\b",
    }

    model_patterns = (
        ("budgetmem_r", r"\bbudgetmem[_ -]?r\b"),
        (
            "gru_reservoir",
            r"(?:\bgru\b.*\breservoir\b|\breservoir\b.*\bgru\b)",
        ),
        (
            "gru_uniform",
            r"(?:\bgru\b.*\buniform\b|\buniform\b.*\bgru\b)",
        ),
        ("gru", r"\bgru\b"),
    )

    sequence_pattern = re.compile(
        r"(?:sequence[_ -]?length|seq(?:uence)?[_ -]?len(?:gth)?|seq)"
        r"[\s:=_-]*(256|512|1024)\b",
        re.IGNORECASE,
    )

    budget_pattern = re.compile(
        r"(?:memory[_ -]?budget|mem[_ -]?budget|budget)"
        r"[\s:=_-]*(16|32)\b",
        re.IGNORECASE,
    )

    seed_pattern = re.compile(
        r"(?:random[_ -]?seed|rng[_ -]?seed|seed)"
        r"[\s:=_-]*(\d+)\b",
        re.IGNORECASE,
    )

    for line in text.splitlines():
        lower_line = line.lower()

        task = next(
            (
                task_name
                for task_name, pattern in task_patterns.items()
                if re.search(pattern, lower_line, re.IGNORECASE)
            ),
            None,
        )

        model = None

        for model_name, pattern in model_patterns:
            if re.search(pattern, lower_line, re.IGNORECASE):
                model = model_name
                break

        sequence_match = sequence_pattern.search(line)
        budget_match = budget_pattern.search(line)
        seed_match = seed_pattern.search(line)

        if not all((task, model, sequence_match, budget_match)):
            continue

        failed = any(word in normalize_key(line) for word in FAILURE_WORDS)

        records.append(
            {
                "task": task,
                "model": model,
                "sequence_length": int(sequence_match.group(1)),
                "budget": int(budget_match.group(1)),
                "seed": int(seed_match.group(1)) if seed_match else None,
                "failed": failed,
                "source": str(source.relative_to(PROJECT_ROOT)),
            }
        )

    return records


def parse_text_file(path: Path, output: list[dict[str, Any]]) -> None:
    try:
        if path.stat().st_size > 20_000_000:
            return

        text = path.read_text(encoding="utf-8", errors="ignore")
        output.extend(infer_from_text(text, path))
    except OSError:
        return


def get_search_roots() -> list[Path]:
    if SOURCE_ARGUMENT:
        requested = Path(SOURCE_ARGUMENT)

        if not requested.is_absolute():
            requested = PROJECT_ROOT / requested

        if not requested.exists():
            raise SystemExit(f"Requested result path does not exist: {requested}")

        return [requested.resolve()]

    candidates = (
        "reports",
        "results",
        "outputs",
        "artifacts",
        "runs",
        "logs",
    )

    return [
        (PROJECT_ROOT / candidate).resolve()
        for candidate in candidates
        if (PROJECT_ROOT / candidate).exists()
    ]


records: list[dict[str, Any]] = []
scanned_files: set[Path] = set()

for search_root in get_search_roots():
    files = [search_root] if search_root.is_file() else search_root.rglob("*")

    for path in files:
        if not path.is_file():
            continue

        if path.name in GENERATED_NAMES:
            continue

        if path in scanned_files:
            continue

        scanned_files.add(path)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            parse_csv_file(path, records)
        elif suffix == ".json":
            parse_json_file(path, records)
        elif suffix in {".jsonl", ".ndjson"}:
            parse_jsonl_file(path, records)
        elif suffix in {".txt", ".log", ".out"}:
            parse_text_file(path, records)

expected_combinations = list(
    itertools.product(
        EXPECTED_TASKS,
        EXPECTED_MODELS,
        EXPECTED_LENGTHS,
        EXPECTED_BUDGETS,
    )
)

observations: dict[
    tuple[str, str, int, int],
    list[dict[str, Any]],
] = defaultdict(list)

for record in records:
    combination = (
        record["task"],
        record["model"],
        record["sequence_length"],
        record["budget"],
    )

    if combination not in expected_combinations:
        continue

    if record["seed"] not in (None, EXPECTED_SEED):
        continue

    observations[combination].append(record)

successful = {
    combination
    for combination, items in observations.items()
    if any(not item["failed"] for item in items)
}

failed_only = {
    combination
    for combination, items in observations.items()
    if items and all(item["failed"] for item in items)
}

missing = set(expected_combinations) - successful
passed = not missing and not failed_only

distinct_tasks = sorted({item[0] for item in successful})
distinct_models = sorted({item[1] for item in successful})
distinct_lengths = sorted({item[2] for item in successful})
distinct_budgets = sorted({item[3] for item in successful})

with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "task",
        "model",
        "sequence_length",
        "memory_budget",
        "seed",
        "tested_successfully",
        "record_count",
        "source_files",
    ]

    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()

    for combination in expected_combinations:
        task, model, sequence_length, budget = combination
        source_files = sorted(
            {
                item["source"]
                for item in observations.get(combination, [])
                if not item["failed"]
            }
        )

        writer.writerow(
            {
                "task": task,
                "model": MODEL_DISPLAY[model],
                "sequence_length": sequence_length,
                "memory_budget": budget,
                "seed": EXPECTED_SEED,
                "tested_successfully": "YES" if combination in successful else "NO",
                "record_count": len(observations.get(combination, [])),
                "source_files": "; ".join(source_files),
            }
        )

summary = {
    "section": "Section 15 pilot experiment matrix coverage",
    "expected_seed": EXPECTED_SEED,
    "expected_run_count": len(expected_combinations),
    "successful_unique_run_count": len(successful),
    "missing_run_count": len(missing),
    "failed_only_run_count": len(failed_only),
    "pass": passed,
    "tasks_tested": distinct_tasks,
    "models_tested": [MODEL_DISPLAY[model] for model in distinct_models],
    "sequence_lengths_tested": distinct_lengths,
    "memory_budgets_tested": distinct_budgets,
    "scanned_file_count": len(scanned_files),
    "evidence_csv": str(CSV_PATH.relative_to(PROJECT_ROOT)),
}

JSON_PATH.write_text(
    json.dumps(summary, indent=2) + "\n",
    encoding="utf-8",
)

report_lines = [
    "SECTION 15 PILOT MATRIX COVERAGE",
    "=" * 60,
    "",
    "Required tasks:",
    "  - selective_copy",
    "  - associative_recall",
    "  - distractor_retrieval",
    "",
    "Required models:",
    "  - GRU",
    "  - GRU + uniform cache",
    "  - GRU + reservoir cache",
    "  - BudgetMem-R",
    "",
    "Required sequence lengths: 256, 512, 1024",
    "Required memory budgets: 16, 32",
    f"Required random seed: {EXPECTED_SEED}",
    "",
    "Expected matrix calculation:",
    "  3 tasks x 4 models x 3 sequence lengths x 2 budgets x 1 seed",
    f"  Expected total: {len(expected_combinations)} runs",
    "",
    f"Scanned files: {len(scanned_files)}",
    f"Successful unique combinations: {len(successful)}",
    f"Missing combinations: {len(missing)}",
    f"Failed-only combinations: {len(failed_only)}",
    "",
    "Observed dimension coverage:",
    f"  Tasks: {', '.join(distinct_tasks) if distinct_tasks else 'NONE'}",
    (
        "  Models: "
        + (
            ", ".join(MODEL_DISPLAY[model] for model in distinct_models)
            if distinct_models
            else "NONE"
        )
    ),
    (
        "  Sequence lengths: "
        + (
            ", ".join(str(value) for value in distinct_lengths)
            if distinct_lengths
            else "NONE"
        )
    ),
    (
        "  Memory budgets: "
        + (
            ", ".join(str(value) for value in distinct_budgets)
            if distinct_budgets
            else "NONE"
        )
    ),
    "",
]

if missing:
    report_lines.append("MISSING OR UNSUCCESSFUL COMBINATIONS")
    report_lines.append("-" * 60)

    for task, model, sequence_length, budget in sorted(missing):
        report_lines.append(
            f"  task={task}, model={MODEL_DISPLAY[model]}, "
            f"sequence_length={sequence_length}, budget={budget}, "
            f"seed={EXPECTED_SEED}"
        )

    report_lines.append("")

if failed_only:
    report_lines.append("FAILED-ONLY COMBINATIONS")
    report_lines.append("-" * 60)

    for task, model, sequence_length, budget in sorted(failed_only):
        report_lines.append(
            f"  task={task}, model={MODEL_DISPLAY[model]}, "
            f"sequence_length={sequence_length}, budget={budget}, "
            f"seed={EXPECTED_SEED}"
        )

    report_lines.append("")

report_lines.extend(
    [
        "FINAL RESULT",
        "=" * 60,
        (
            "PASS: All required tasks, models, sequence lengths, and "
            "memory budgets were tested successfully."
            if passed
            else
            "FAIL: The required pilot matrix is incomplete. Review the "
            "missing combinations above and rerun only those configurations."
        ),
        "",
        f"Detailed coverage table: {CSV_PATH.relative_to(PROJECT_ROOT)}",
        f"Machine-readable summary: {JSON_PATH.relative_to(PROJECT_ROOT)}",
    ]
)

REPORT_PATH.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

print(REPORT_PATH.read_text(encoding="utf-8"))

if not passed:
    raise SystemExit(2)
PY
