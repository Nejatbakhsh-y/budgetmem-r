#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Install the already-validated temporary IMDb DatasetDict into its canonical
# path without using pathlib.rename(), then run the Section 14 gate.
#
# This avoids OneDrive/WSL rename semantics and does not rerun dataset recovery.

readonly TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

find_repo_root() {
    if root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
        printf '%s\n' "$root"
        return 0
    fi

    local cursor
    cursor="$(pwd)"
    while [[ "$cursor" != "/" ]]; do
        if [[ -f "$cursor/pyproject.toml" && -d "$cursor/src/budgetmem" ]]; then
            printf '%s\n' "$cursor"
            return 0
        fi
        cursor="$(dirname "$cursor")"
    done
    return 1
}

choose_python() {
    local candidate
    for candidate in \
        "$REPO_ROOT/.venv/bin/python" \
        "$REPO_ROOT/venv/bin/python" \
        "$REPO_ROOT/.env/bin/python" \
        python3 \
        python
    do
        if [[ "$candidate" == */* ]]; then
            [[ -x "$candidate" ]] && {
                printf '%s\n' "$candidate"
                return 0
            }
        elif command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

REPO_ROOT="$(find_repo_root)" || die "The budgetmem-r repository root was not found."
cd "$REPO_ROOT"

PYTHON_BIN="$(choose_python)" || die "Python was not found."

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SECTION14_TIMESTAMP="$TIMESTAMP"

BACKUP_DIR="$REPO_ROOT/reports/evidence/backups/section14_imdb_install/$TIMESTAMP"
LOG_FILE="$REPO_ROOT/reports/evidence/logs/section14_imdb_install_${TIMESTAMP}.log"
JUNIT_FILE="$REPO_ROOT/reports/evidence/junit/section14_imdb_install_${TIMESTAMP}.xml"
REPORT_FILE="$REPO_ROOT/reports/evidence/section14_unit_tests_report.txt"
RESULTS_FILE="$REPO_ROOT/reports/tables/section14_unit_test_results.csv"
IMDB_REPORT="$REPO_ROOT/reports/evidence/section14_imdb_install_${TIMESTAMP}.json"

mkdir -p \
    "$BACKUP_DIR" \
    reports/evidence/logs \
    reports/evidence/junit \
    reports/tables

export SECTION14_REPO_ROOT="$REPO_ROOT"
export SECTION14_BACKUP_DIR="$BACKUP_DIR"
export SECTION14_IMDB_REPORT="$IMDB_REPORT"

log "Installing the validated temporary IMDb DatasetDict."

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk

root = Path(os.environ["SECTION14_REPO_ROOT"])
backup_root = Path(os.environ["SECTION14_BACKUP_DIR"])
report_path = Path(os.environ["SECTION14_IMDB_REPORT"])
timestamp = os.environ["SECTION14_TIMESTAMP"]

destination = root / "data" / "processed" / "imdb"


def make_writable(path: str | Path) -> None:
    try:
        os.chmod(
            path,
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IWGRP
            | stat.S_IXGRP,
        )
    except OSError:
        pass


def remove_tree(path: Path) -> None:
    if not path.exists():
        return

    def onerror(function, failing_path, _exc_info):
        make_writable(failing_path)
        function(failing_path)

    last_error: Exception | None = None
    for attempt in range(10):
        try:
            shutil.rmtree(path, onerror=onerror)
            return
        except (PermissionError, OSError) as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))

    raise RuntimeError(f"Could not remove {path}: {last_error}")


def user_text_field(dataset: Dataset) -> str | None:
    for name in ("text", "review", "content", "sentence"):
        if name in dataset.column_names:
            return name
    return None


def identity(row: dict[str, Any], field: str) -> str:
    text = str(row[field]).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


candidate_paths: list[Path] = []
data_root = root / "data"
if data_root.exists():
    for marker in data_root.rglob("dataset_dict.json"):
        parent = marker.parent
        lowered = str(parent).lower()
        if "imdb" not in lowered:
            continue
        if (
            ".offline_tmp_" in parent.name
            or ".tmp_" in parent.name
            or "offline_tmp" in lowered
        ):
            candidate_paths.append(parent)

# Previous recovery runs may have created a validated DatasetDict backup.
backup_search_root = root / "reports" / "evidence" / "backups"
if backup_search_root.exists():
    for marker in backup_search_root.rglob("dataset_dict.json"):
        parent = marker.parent
        lowered = str(parent).lower()
        if "imdb" in lowered:
            candidate_paths.append(parent)

candidates: list[tuple[int, Path, DatasetDict]] = []
errors: list[dict[str, str]] = []

for candidate_path in sorted(set(candidate_paths)):
    try:
        loaded = load_from_disk(str(candidate_path))
        if not isinstance(loaded, DatasetDict):
            continue
        required = {"train", "validation", "test"}
        if not required.issubset(loaded.keys()):
            continue
        total = sum(len(loaded[name]) for name in required)
        candidates.append((total, candidate_path, loaded))
    except Exception as exc:
        errors.append(
            {
                "path": str(candidate_path),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

if not candidates:
    raise RuntimeError(
        "No readable temporary or backup IMDb DatasetDict with "
        "train/validation/test splits was found."
    )

candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
total_rows, source_path, source_dataset = candidates[0]

split_sizes = {
    name: len(source_dataset[name])
    for name in ("train", "validation", "test")
}

minimums = {
    "train": 22000,
    "validation": 2400,
    "test": 24900,
}
for name, minimum in minimums.items():
    if split_sizes[name] < minimum:
        raise RuntimeError(
            f"Temporary IMDb {name} split is incomplete: "
            f"{split_sizes[name]} < {minimum}."
        )

if total_rows < 49800:
    raise RuntimeError(
        f"Temporary IMDb corpus is incomplete: {total_rows} < 49800."
    )

# Validate source_index uniqueness and separation.
source_sets: dict[str, set[int]] = {}
for name in ("train", "validation", "test"):
    split = source_dataset[name]
    if "source_index" not in split.column_names:
        raise RuntimeError(
            f"Temporary IMDb split {name} has no source_index column."
        )
    values = [int(value) for value in split["source_index"]]
    if len(values) != len(set(values)):
        raise RuntimeError(
            f"Temporary IMDb split {name} contains duplicate source_index values."
        )
    source_sets[name] = set(values)

for left_index, left in enumerate(("train", "validation", "test")):
    for right in ("train", "validation", "test")[left_index + 1:]:
        overlap = source_sets[left] & source_sets[right]
        if overlap:
            raise RuntimeError(
                f"Temporary IMDb source_index overlap remains between "
                f"{left} and {right}: {len(overlap)}."
            )

field = user_text_field(source_dataset["test"])
if field is None:
    raise RuntimeError("IMDb text/review field was not found.")

test_hashes = {
    identity(dict(row), field)
    for row in source_dataset["test"]
}
for name in ("train", "validation"):
    hashes = {
        identity(dict(row), field)
        for row in source_dataset[name]
    }
    overlap = hashes & test_hashes
    if overlap:
        raise RuntimeError(
            f"Official IMDb test leakage remains in {name}: {len(overlap)}."
        )

# Preserve the current canonical destination when present.
if destination.exists():
    backup_destination = backup_root / "previous_imdb"
    if backup_destination.exists():
        remove_tree(backup_destination)
    backup_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(destination, backup_destination)

remove_tree(destination)
destination.parent.mkdir(parents=True, exist_ok=True)

last_error: Exception | None = None
for attempt in range(10):
    try:
        shutil.copytree(
            source_path,
            destination,
            copy_function=shutil.copy2,
        )
        break
    except (PermissionError, OSError) as exc:
        last_error = exc
        remove_tree(destination)
        time.sleep(0.5 * (attempt + 1))
else:
    raise RuntimeError(
        f"Could not copy {source_path} to {destination}: {last_error}"
    )

# Validate the installed canonical DatasetDict.
installed = load_from_disk(str(destination))
if not isinstance(installed, DatasetDict):
    raise RuntimeError("Installed IMDb object is not a DatasetDict.")

installed_sizes = {
    name: len(installed[name])
    for name in ("train", "validation", "test")
}
if installed_sizes != split_sizes:
    raise RuntimeError(
        f"Installed split sizes differ: {installed_sizes} != {split_sizes}."
    )

installed_sets = {
    name: set(int(value) for value in installed[name]["source_index"])
    for name in ("train", "validation", "test")
}
for left_index, left in enumerate(("train", "validation", "test")):
    if len(installed_sets[left]) != len(installed[left]):
        raise RuntimeError(
            f"Installed {left} source_index values are not unique."
        )
    for right in ("train", "validation", "test")[left_index + 1:]:
        if installed_sets[left] & installed_sets[right]:
            raise RuntimeError(
                f"Installed source_index overlap: {left}/{right}."
            )

# Best-effort removal of stale temporary copies after successful installation.
removed_temporary_paths: list[str] = []
for candidate_path in sorted(set(candidate_paths)):
    if candidate_path == destination:
        continue
    if ".offline_tmp_" not in candidate_path.name and ".tmp_" not in candidate_path.name:
        continue
    try:
        remove_tree(candidate_path)
        removed_temporary_paths.append(str(candidate_path))
    except Exception:
        pass

report = {
    "generated_utc": timestamp,
    "status": "PASS",
    "network_used": False,
    "source": str(source_path),
    "destination": str(destination.relative_to(root)),
    "split_sizes": installed_sizes,
    "total_rows": sum(installed_sizes.values()),
    "source_index_unique_and_disjoint": True,
    "official_test_isolated": True,
    "temporary_paths_removed": removed_temporary_paths,
    "candidate_errors": errors,
}
report_path.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)

print(json.dumps(report, indent=2))
PY

log "Confirming the restored BudgetMem-R contract."

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import torch

from budgetmem.models.budgetmem_r import BudgetMemR

model = BudgetMemR(
    input_dim=6,
    hidden_dim=12,
    output_dim=3,
    max_budget=16,
    training_budgets=(4, 8, 16),
    top_k=3,
).eval()

inputs = torch.randn(2, 12, 6)
outputs = model(inputs, budget=torch.tensor([4, 8]))

for name in (
    "hidden_states",
    "write_probabilities",
    "hard_writes",
    "write_slots",
    "eviction_flags",
    "memory_masks",
    "memory_sizes",
    "budgets",
    "final_memory",
):
    assert hasattr(outputs, name), name

assert torch.all(outputs.memory_sizes <= outputs.budgets.unsqueeze(1))
print("BudgetMem-R restored-contract smoke test: PASS")
PY

log "Checking Python syntax."

"$PYTHON_BIN" -m py_compile src/budgetmem/models/budgetmem_r.py
[[ -f tests/section14_runtime.py ]] && \
    "$PYTHON_BIN" -m py_compile tests/section14_runtime.py
[[ -f tests/test_section14_required.py ]] && \
    "$PYTHON_BIN" -m py_compile tests/test_section14_required.py

log "Selecting Section 14 tests."

mapfile -t TARGETS < <(
    "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import ast
import re
from pathlib import Path

root = Path.cwd()
targets: list[str] = []

for explicit in (
    root / "tests" / "test_budgetmem_r.py",
    root / "tests" / "test_section14_required.py",
):
    if explicit.exists():
        targets.append(str(explicit.relative_to(root)))

pattern = re.compile(
    r"("
    r"strict_budget|hard_budget|memory.*budget|budget.*violat|"
    r"budget_sampler|budget_sampling|invalid_budget|"
    r"future_tokens|causal|determin|same_seed|training_order|"
    r"synthetic.*seed|seed.*overlap|"
    r"hdfs.*block|block.*overlap|"
    r"imdb.*test|official.*test|split.*leak|"
    r"gradient|controller.*gradient|backpropagat.*controller|"
    r"graph_policy|cached_state|trainable_cache|detached_cache|"
    r"memory.*reset|reset.*memory"
    r")",
    re.IGNORECASE,
)

for path in sorted((root / "tests").rglob("test*.py")):
    relative = str(path.relative_to(root))
    if relative in targets:
        continue

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        continue

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_") and pattern.search(node.name):
                targets.append(f"{relative}::{node.name}")
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name.startswith("test_") and pattern.search(child.name):
                        targets.append(
                            f"{relative}::{node.name}::{child.name}"
                        )

for target in dict.fromkeys(targets):
    print(target)
PY
)

[[ "${#TARGETS[@]}" -gt 0 ]] || die "No Section 14 tests were found."

printf 'Selected targets: %s\n' "${#TARGETS[@]}"
printf '  %s\n' "${TARGETS[@]}"

log "Verifying pytest collection."

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest \
    -q \
    -o addopts='' \
    --collect-only \
    "${TARGETS[@]}" \
    >/dev/null

log "Running the Section 14 gate."

set +e
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest \
    -q \
    -o addopts='' \
    "${TARGETS[@]}" \
    --junitxml="$JUNIT_FILE" \
    2>&1 | tee "$LOG_FILE"
PYTEST_EXIT="${PIPESTATUS[0]}"
set -e

export SECTION14_JUNIT_FILE="$JUNIT_FILE"
export SECTION14_LOG_FILE="$LOG_FILE"
export SECTION14_REPORT_FILE="$REPORT_FILE"
export SECTION14_RESULTS_FILE="$RESULTS_FILE"
export SECTION14_IMDB_REPORT="$IMDB_REPORT"
export SECTION14_PYTEST_EXIT="$PYTEST_EXIT"

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import csv
import os
import xml.etree.ElementTree as ET
from pathlib import Path

junit = Path(os.environ["SECTION14_JUNIT_FILE"])
log = Path(os.environ["SECTION14_LOG_FILE"])
report = Path(os.environ["SECTION14_REPORT_FILE"])
results = Path(os.environ["SECTION14_RESULTS_FILE"])
imdb_report = Path(os.environ["SECTION14_IMDB_REPORT"])
exit_code = int(os.environ["SECTION14_PYTEST_EXIT"])
timestamp = os.environ["SECTION14_TIMESTAMP"]

cases: list[dict[str, str]] = []
if junit.exists():
    root = ET.parse(junit).getroot()
    for case in root.iter("testcase"):
        status = "PASS"
        detail = ""
        for child_name in ("failure", "error", "skipped"):
            child = case.find(child_name)
            if child is not None:
                status = child_name.upper()
                detail = (
                    child.attrib.get("message")
                    or child.text
                    or ""
                ).strip()
                break

        cases.append(
            {
                "classname": case.attrib.get("classname", ""),
                "test_name": case.attrib.get("name", ""),
                "status": status,
                "seconds": case.attrib.get("time", "0"),
                "detail": detail.replace("\n", " ")[:5000],
            }
        )

with results.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=(
            "classname",
            "test_name",
            "status",
            "seconds",
            "detail",
        ),
    )
    writer.writeheader()
    writer.writerows(cases)

required = {
    "Budget correctness": (
        "test_14_01_",
        "test_hard_budget_",
        "test_strict_budget_",
    ),
    "Causality": (
        "test_14_02_",
        "test_future_tokens_",
    ),
    "Determinism": (
        "test_14_03_",
        "test_14_04_",
        "test_14_05_",
        "test_14_06_",
        "test_eval_is_deterministic",
        "test_same_seed_",
    ),
    "Synthetic seed isolation": ("test_14_07_",),
    "HDFS block isolation": ("test_14_08_",),
    "IMDb official-test isolation": (
        "test_14_09_",
        "test_imdb_official_test_",
    ),
    "Gradient flow": (
        "test_14_10_",
        "test_training_loss_backpropagates_",
        "test_memory_controllers_receive_",
        "test_composite_objective_",
    ),
    "Cached-state graph policy": (
        "test_14_11_",
        "test_14_12_",
    ),
    "Memory reset": (
        "test_14_13_",
        "test_memory_is_reset_",
    ),
}

def matching(prefixes: tuple[str, ...]) -> list[dict[str, str]]:
    return [
        case
        for case in cases
        if any(case["test_name"].startswith(prefix) for prefix in prefixes)
    ]

statuses: dict[str, str] = {}
for category, prefixes in required.items():
    matched = matching(prefixes)
    statuses[category] = (
        "PASS"
        if matched and all(case["status"] == "PASS" for case in matched)
        else "FAIL"
    )

all_selected_pass = bool(cases) and all(
    case["status"] == "PASS" for case in cases
)
go = (
    exit_code == 0
    and all_selected_pass
    and all(status == "PASS" for status in statuses.values())
)

lines = [
    "Section 14 — Unit Tests Required Before Training",
    f"Generated UTC: {timestamp}",
    "",
]
for category, status in statuses.items():
    lines.append(f"{category}: {status}")

lines.extend(
    [
        f"All selected Section 14 tests: {'PASS' if all_selected_pass else 'FAIL'}",
        f"Pytest exit code: {exit_code}",
        "",
        f"Final decision: {'GO' if go else 'NO-GO'}",
        f"Section 14: {'COMPLETE' if go else 'INCOMPLETE'}",
        "",
        f"JUnit evidence: {junit}",
        f"Detailed log: {log}",
        f"Result table: {results}",
        f"IMDb installation evidence: {imdb_report}",
    ]
)

failed = [case for case in cases if case["status"] != "PASS"]
if failed:
    lines.extend(["", "Failed or unresolved checks:"])
    for case in failed:
        lines.append(
            f"- {case['test_name']}: {case['status']} — "
            f"{case['detail'] or 'No detail recorded.'}"
        )

report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print()
print(report.read_text(encoding="utf-8"))
PY

if [[ "$PYTEST_EXIT" -eq 0 ]]; then
    printf '\nSECTION 14 RESULT: GO\n'
    printf 'Section 14 is complete. Training may begin.\n'
else
    printf '\nSECTION 14 RESULT: NO-GO\n'
    printf 'Review reports/evidence/section14_unit_tests_report.txt.\n'
    exit "$PYTEST_EXIT"
fi
