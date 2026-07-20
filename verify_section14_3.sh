#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/mnt/c/Users/nejat/OneDrive/Desktop/UN/Skills/GitHub 2026/budgetmem-r"
REPORT_DIR="$PROJECT_ROOT/reports/evidence"
REPORT_FILE="$REPORT_DIR/section14_3_memory_budget_enforcement.txt"
TEST_LOG="$REPORT_DIR/section14_3_budget_tests.log"

cd "$PROJECT_ROOT" || {
    echo "ERROR: Project directory was not found."
    exit 1
}

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "ERROR: WSL virtual environment was not found at .venv/bin/activate"
    exit 1
fi

source .venv/bin/activate
mkdir -p "$REPORT_DIR"

echo "============================================================"
echo "SECTION 14.3 — MEMORY-BUDGET ENFORCEMENT VERIFICATION"
echo "============================================================"

echo
echo "[1/3] Collecting memory-budget tests..."

mapfile -t BUDGET_TESTS < <(
    python -m pytest --collect-only -q tests 2>/dev/null |
    grep '::' |
    grep -Ei \
        'budget|capacity|memory.*slot|slot.*memory|cache.*size|size.*cache|occupancy'
)

if [[ ${#BUDGET_TESTS[@]} -eq 0 ]]; then
    echo "ERROR: No memory-budget enforcement tests were discovered."
    echo
    echo "Search results:"
    grep -RniE \
        'memory_budget|budget.*violat|capacity|active_slots|memory_size|cache_size' \
        tests src 2>/dev/null | head -100
    exit 1
fi

printf 'Discovered %s relevant test(s).\n' "${#BUDGET_TESTS[@]}"
printf '  %s\n' "${BUDGET_TESTS[@]}"

echo
echo "[2/3] Running memory-budget tests..."

set +e
python -m pytest -vv "${BUDGET_TESTS[@]}" 2>&1 | tee "$TEST_LOG"
PYTEST_EXIT=${PIPESTATUS[0]}
set -e

if [[ "$PYTEST_EXIT" -ne 0 ]]; then
    TEST_STATUS="FAIL"
    echo
    echo "ERROR: One or more memory-budget tests failed."
else
    TEST_STATUS="PASS"
    echo
    echo "All discovered memory-budget tests passed."
fi

echo
echo "[3/3] Checking experiment evidence for budget violations..."

export PROJECT_ROOT
export REPORT_FILE
export TEST_LOG
export TEST_STATUS
export PYTEST_EXIT

set +e
python - <<'PY'
from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

project_root = Path(os.environ["PROJECT_ROOT"])
report_file = Path(os.environ["REPORT_FILE"])
test_log = Path(os.environ["TEST_LOG"])
test_status = os.environ["TEST_STATUS"]
pytest_exit = int(os.environ["PYTEST_EXIT"])

budget_keys = {
    "budget",
    "memory_budget",
    "cache_budget",
    "slot_budget",
    "memory_capacity",
    "cache_capacity",
    "max_memory_slots",
    "maximum_memory_slots",
    "configured_budget",
    "requested_budget",
}

usage_keys = {
    "memory_size",
    "cache_size",
    "active_slots",
    "used_slots",
    "occupied_slots",
    "memory_occupancy",
    "cache_occupancy",
    "num_memory_slots",
    "memory_slots_used",
    "current_memory_size",
    "maximum_memory_size",
    "max_memory_size",
    "max_observed_memory",
    "max_memory_used",
    "peak_memory_slots",
    "peak_cache_size",
}

violation_boolean_keys = {
    "memory_budget_violated",
    "budget_violated",
    "budget_violation",
    "capacity_violated",
    "memory_overflow",
}

violation_count_keys = {
    "budget_violation_count",
    "memory_budget_violation_count",
    "capacity_violation_count",
    "overflow_count",
}


def normalize_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return None

    return None


def as_boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "yes", "1", "failed", "violation"}:
            return True
        if cleaned in {"false", "no", "0", "passed", "none"}:
            return False

    if isinstance(value, (int, float)):
        return bool(value)

    return None


comparisons: list[dict[str, Any]] = []
violations: list[str] = []
files_checked: set[str] = set()


def inspect_record(
    record: dict[str, Any],
    source: str,
    location: str,
    inherited_budget: float | None = None,
) -> None:
    normalized = {normalize_key(key): value for key, value in record.items()}

    local_budget = inherited_budget

    for key in budget_keys:
        if key in normalized:
            candidate = as_number(normalized[key])
            if candidate is not None and candidate >= 0:
                local_budget = candidate
                break

    for key in violation_boolean_keys:
        if key in normalized:
            result = as_boolean(normalized[key])
            if result is True:
                violations.append(
                    f"{source} [{location}]: {key} indicates a violation."
                )

    for key in violation_count_keys:
        if key in normalized:
            count = as_number(normalized[key])
            if count is not None and count > 0:
                violations.append(
                    f"{source} [{location}]: {key}={count:g}."
                )

    if local_budget is not None:
        for key in usage_keys:
            if key not in normalized:
                continue

            usage = as_number(normalized[key])
            if usage is None:
                continue

            passed = usage <= local_budget
            comparisons.append(
                {
                    "source": source,
                    "location": location,
                    "usage_key": key,
                    "usage": usage,
                    "budget": local_budget,
                    "passed": passed,
                }
            )

            if not passed:
                violations.append(
                    f"{source} [{location}]: "
                    f"{key}={usage:g} exceeds budget={local_budget:g}."
                )

    for key, value in record.items():
        child_location = f"{location}.{key}"

        if isinstance(value, dict):
            inspect_record(
                value,
                source,
                child_location,
                inherited_budget=local_budget,
            )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    inspect_record(
                        item,
                        source,
                        f"{child_location}[{index}]",
                        inherited_budget=local_budget,
                    )


def inspect_json_file(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return

    files_checked.add(str(path.relative_to(project_root)))

    if isinstance(data, dict):
        inspect_record(data, str(path.relative_to(project_root)), "root")
    elif isinstance(data, list):
        for index, item in enumerate(data):
            if isinstance(item, dict):
                inspect_record(
                    item,
                    str(path.relative_to(project_root)),
                    f"row[{index}]",
                )


def inspect_jsonl_file(path: Path) -> None:
    found_valid_record = False

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if isinstance(record, dict):
                    found_valid_record = True
                    inspect_record(
                        record,
                        str(path.relative_to(project_root)),
                        f"line[{line_number}]",
                    )
    except (OSError, UnicodeDecodeError):
        return

    if found_valid_record:
        files_checked.add(str(path.relative_to(project_root)))


def inspect_csv_file(path: Path) -> None:
    found_record = False

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                found_record = True
                inspect_record(
                    dict(row),
                    str(path.relative_to(project_root)),
                    f"row[{row_number}]",
                )
    except (OSError, UnicodeDecodeError, csv.Error):
        return

    if found_record:
        files_checked.add(str(path.relative_to(project_root)))


candidate_directories = [
    project_root / "reports" / "evidence",
    project_root / "results",
    project_root / "outputs",
    project_root / "artifacts",
]

for directory in candidate_directories:
    if not directory.exists():
        continue

    for path in directory.rglob("*"):
        if not path.is_file():
            continue

        if path == report_file or path == test_log:
            continue

        try:
            if path.stat().st_size > 20_000_000:
                continue
        except OSError:
            continue

        suffix = path.suffix.lower()

        if suffix == ".json":
            inspect_json_file(path)
        elif suffix in {".jsonl", ".ndjson"}:
            inspect_jsonl_file(path)
        elif suffix in {".csv", ".tsv"}:
            inspect_csv_file(path)

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except (OSError, subprocess.CalledProcessError):
    git_commit = "Unavailable"

valid_comparisons = [
    item for item in comparisons
    if item["budget"] >= 0 and item["usage"] >= 0
]

max_ratio = None
max_record = None

for item in valid_comparisons:
    budget = item["budget"]
    usage = item["usage"]

    if budget == 0:
        ratio = 0.0 if usage == 0 else float("inf")
    else:
        ratio = usage / budget

    if max_ratio is None or ratio > max_ratio:
        max_ratio = ratio
        max_record = item

evidence_status = (
    "PASS"
    if valid_comparisons and not violations
    else "FAIL"
)

overall_status = (
    "PASS"
    if pytest_exit == 0 and evidence_status == "PASS"
    else "FAIL"
)

lines = [
    "SECTION 14.3 — MEMORY-BUDGET ENFORCEMENT",
    "=" * 60,
    f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
    f"Git commit: {git_commit}",
    "",
    f"Budget-test status: {test_status}",
    f"Pytest exit code: {pytest_exit}",
    f"Pytest log: {test_log.relative_to(project_root)}",
    "",
    f"Evidence status: {evidence_status}",
    f"Evidence files checked: {len(files_checked)}",
    f"Budget/usage comparisons: {len(valid_comparisons)}",
    f"Detected violations: {len(violations)}",
]

if max_record is not None:
    ratio_text = (
        "infinite"
        if max_ratio == float("inf")
        else f"{max_ratio:.6f}"
    )

    lines.extend(
        [
            f"Maximum observed usage/budget ratio: {ratio_text}",
            (
                "Maximum observation: "
                f"{max_record['source']} [{max_record['location']}], "
                f"{max_record['usage_key']}={max_record['usage']:g}, "
                f"budget={max_record['budget']:g}"
            ),
        ]
    )

lines.extend(["", "Violations"])

if violations:
    lines.extend(f"- {item}" for item in violations)
else:
    lines.append("- None detected.")

if not valid_comparisons:
    lines.extend(
        [
            "",
            "ERROR:",
            "No evidence record containing both a memory budget and",
            "an observed memory/cache size was found.",
        ]
    )

lines.extend(
    [
        "",
        f"OVERALL RESULT: {overall_status}",
        "",
    ]
)

if overall_status == "PASS":
    lines.append(
        "Conclusion: Memory-budget enforcement passed. "
        "All relevant tests passed, and no observed memory usage "
        "exceeded its configured budget."
    )
else:
    lines.append(
        "Conclusion: Memory-budget enforcement has not yet been "
        "demonstrated successfully."
    )

report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

print()
print("\n".join(lines))

raise SystemExit(0 if overall_status == "PASS" else 1)
PY

VERIFY_EXIT=$?
set -e

echo
echo "============================================================"

if [[ "$VERIFY_EXIT" -eq 0 ]]; then
    echo "SECTION 14.3: PASS"
    echo "Memory-budget enforcement passed."
    echo
    echo "Evidence file:"
    echo "reports/evidence/section14_3_memory_budget_enforcement.txt"
else
    echo "SECTION 14.3: FAIL"
    echo "Review:"
    echo "reports/evidence/section14_3_memory_budget_enforcement.txt"
    echo
    echo "Test log:"
    echo "reports/evidence/section14_3_budget_tests.log"
fi

echo "============================================================"

exit "$VERIFY_EXIT"
