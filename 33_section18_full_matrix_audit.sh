#!/usr/bin/env bash
set -Eeuo pipefail

# Section 18 full-matrix audit.
# This script does not rerun experiments and does not stage, commit, or push files.

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

if [[ ! -d .git ]]; then
  echo "ERROR: Run this script from inside the budgetmem-r Git repository." >&2
  exit 2
fi

BRANCH="$(git branch --show-current)"
if [[ "$BRANCH" != "feature/18-main-experiment-matrix" ]]; then
  echo "ERROR: Expected branch feature/18-main-experiment-matrix, found: $BRANCH" >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EVIDENCE_DIR="reports/evidence/section18"
mkdir -p "$EVIDENCE_DIR"
REPORT="$EVIDENCE_DIR/section18_full_matrix_audit_${STAMP}.txt"
LATEST="$EVIDENCE_DIR/section18_full_matrix_audit_latest.txt"
JSON_REPORT="$EVIDENCE_DIR/section18_full_matrix_audit_${STAMP}.json"
MANIFEST="$EVIDENCE_DIR/section18_result_file_manifest_${STAMP}.csv"

exec > >(tee "$REPORT") 2>&1

fail() {
  echo "ERROR: $*" >&2
  cp "$REPORT" "$LATEST" 2>/dev/null || true
  exit 1
}

printf '%s\n' "======================================================================"
printf '%s\n' "SECTION 18 FULL MATRIX AUDIT"
printf '%s\n' "======================================================================"
echo "UTC timestamp: $STAMP"
echo "Repository: $REPO_ROOT"
echo "Branch: $BRANCH"
echo

# Locate authoritative completion evidence from the execution automation.
SUMMARY_FILE="$EVIDENCE_DIR/section18_summary.txt"
LATEST_MATRIX_LOG="$(find "$EVIDENCE_DIR" -maxdepth 1 -type f -name 'section18_full_matrix_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {$1=""; sub(/^ /,""); print; exit}')"

[[ -f "$SUMMARY_FILE" ]] || fail "Missing $SUMMARY_FILE"
[[ -n "$LATEST_MATRIX_LOG" && -f "$LATEST_MATRIX_LOG" ]] || fail "No Section 18 full-matrix execution log was found."

echo "Summary: $SUMMARY_FILE"
echo "Matrix log: $LATEST_MATRIX_LOG"
echo

COMPLETION_EVIDENCE=""
if grep -Fq "FINAL DECISION: SECTION 18 MATRIX EXECUTION COMPLETED" "$LATEST_MATRIX_LOG"; then
  COMPLETION_EVIDENCE="wrapper completion decision recorded in matrix log"
elif grep -Fq "Section 18 automation finished." "$LATEST_MATRIX_LOG" \
  && grep -Fq "Summary:" "$LATEST_MATRIX_LOG"; then
  COMPLETION_EVIDENCE="matrix-runner completion marker and summary reference"
else
  fail "The latest matrix log contains neither the wrapper decision nor the matrix-runner completion evidence."
fi

echo "Execution completion marker: PASS"
echo "Completion evidence: $COMPLETION_EVIDENCE"

# Require the principal Section 18 terminal gates when present in the summary.
SUMMARY_REQUIRED=(
  "memory_budget_violations=PASS"
  "bgl_complete=PASS"
)
for marker in "${SUMMARY_REQUIRED[@]}"; do
  if grep -Fqi "$marker" "$SUMMARY_FILE"; then
    echo "$marker"
  else
    fail "Required summary marker is missing: $marker"
  fi
done

echo

# Reject obvious fatal execution signals, while allowing explicitly reported OOM rows.
if grep -Ein "(^|[^A-Za-z])(traceback|segmentation fault|assertionerror|fatal error|uncaught exception)([^A-Za-z]|$)" "$LATEST_MATRIX_LOG" >/tmp/section18_fatal_hits.txt 2>/dev/null; then
  echo "Fatal signals detected in matrix log:"
  cat /tmp/section18_fatal_hits.txt
  fail "Matrix log contains fatal execution signals."
fi
rm -f /tmp/section18_fatal_hits.txt

echo "Fatal-log scan: PASS"

echo
set +e
python - "$REPO_ROOT" "$SUMMARY_FILE" "$LATEST_MATRIX_LOG" "$JSON_REPORT" "$MANIFEST" <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

root = Path(sys.argv[1]).resolve()


def absolute_from_root(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


summary_path = absolute_from_root(sys.argv[2])
log_path = absolute_from_root(sys.argv[3])
json_report_path = absolute_from_root(sys.argv[4])
manifest_path = absolute_from_root(sys.argv[5])

SEARCH_ROOTS = [
    root / "reports" / "tables" / "section18",
    root / "reports" / "section18",
    root / "outputs" / "section18",
    root / "results" / "section18",
    root / "artifacts" / "section18",
    root / "reports" / "tables",
]

# Files clearly unrelated to Section 18 are excluded even when reports/tables is searched.
EXCLUDE_PARTS = {
    "pilot",
    "section15",
    "section16",
    "section17",
    "backups",
    "fixtures",
    ".git",
    ".venv",
    "node_modules",
}

SUPPORTED = {".csv", ".jsonl", ".ndjson", ".json", ".parquet"}


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def norm_value(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


ALIASES = {
    "task": {"task", "task_name", "dataset_task", "benchmark", "problem"},
    "dataset": {"dataset", "dataset_name", "corpus", "data_source"},
    "model": {"model", "model_name", "method", "architecture", "baseline"},
    "seed": {"seed", "random_seed", "run_seed"},
    "sequence_length": {"sequence_length", "seq_len", "seqlen", "context_length", "length"},
    "memory_budget": {"memory_budget", "budget", "memory_size", "configured_budget", "b"},
    "status": {"status", "run_status", "state", "outcome", "result_status"},
    "budget_pass": {"budget_pass", "memory_budget_pass", "budget_correct", "budget_ok"},
    "max_memory_size": {"max_memory_size", "observed_max_memory", "peak_memory_slots", "max_slots"},
    "oom": {"oom", "out_of_memory", "oom_reported"},
    "metric": {
        "token_accuracy",
        "exact_match_accuracy",
        "accuracy",
        "memory_recall",
        "long_range_recall",
        "successful_long_range_retrievals",
        "f1",
        "loss",
        "perplexity",
        "auc",
    },
}
ALIAS_NORM = {key: {norm_name(v) for v in vals} for key, vals in ALIASES.items()}


def canonical_columns(fieldnames: Iterable[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for original in fieldnames:
        n = norm_name(str(original))
        for canonical, names in ALIAS_NORM.items():
            if n in names and canonical not in output:
                output[canonical] = str(original)
    return output


def looks_section18(path: Path) -> bool:
    rel = path.relative_to(root).as_posix().lower()
    if "section18" in rel or "main_experiment" in rel or "matrix" in path.name.lower():
        return True
    # Permit result-like files under a dedicated Section 18 output root.
    return any(parent.exists() and parent in path.parents for parent in SEARCH_ROOTS[:5])


def excluded(path: Path) -> bool:
    rel_parts = {part.lower() for part in path.relative_to(root).parts}
    return bool(rel_parts & EXCLUDE_PARTS)


files: list[Path] = []
seen: set[Path] = set()
for base in SEARCH_ROOTS:
    if not base.exists():
        continue
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        if excluded(path) or not looks_section18(path):
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            files.append(path)

files.sort()

rows: list[dict[str, Any]] = []
file_records: list[dict[str, Any]] = []
read_errors: list[str] = []


def append_rows(path: Path, raw_rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        mapped = {str(k): v for k, v in raw.items()}
        cols = canonical_columns(mapped.keys())
        canonical: dict[str, Any] = {"_source_file": path.relative_to(root).as_posix()}
        for key, original in cols.items():
            canonical[key] = mapped.get(original)
        # Keep every numeric candidate so finite-value validation can inspect it.
        canonical["_raw"] = mapped
        if any(key in canonical for key in ("task", "dataset", "model", "seed", "sequence_length", "metric")):
            rows.append(canonical)
            count += 1
    return count


for path in files:
    rel = path.relative_to(root).as_posix()
    size = path.stat().st_size
    sha = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                sha.update(chunk)
    except OSError as exc:
        read_errors.append(f"{rel}: hash failed: {exc}")
        continue

    parsed = 0
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                parsed = append_rows(path, reader)
        elif suffix in {".jsonl", ".ndjson"}:
            def jsonl_rows() -> Iterable[dict[str, Any]]:
                with path.open("r", encoding="utf-8-sig") as fh:
                    for line_no, line in enumerate(fh, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError as exc:
                            read_errors.append(f"{rel}:{line_no}: {exc}")
                            continue
                        if isinstance(item, dict):
                            yield item
            parsed = append_rows(path, jsonl_rows())
        elif suffix == ".json":
            with path.open("r", encoding="utf-8-sig") as fh:
                obj = json.load(fh)
            if isinstance(obj, list):
                parsed = append_rows(path, (x for x in obj if isinstance(x, dict)))
            elif isinstance(obj, dict):
                candidate = None
                for key in ("results", "runs", "records", "rows", "cells", "data"):
                    if isinstance(obj.get(key), list):
                        candidate = obj[key]
                        break
                parsed = append_rows(path, candidate if candidate is not None else [obj])
        elif suffix == ".parquet":
            try:
                import pandas as pd  # type: ignore
                frame = pd.read_parquet(path)
                parsed = append_rows(path, frame.to_dict(orient="records"))
            except Exception as exc:  # optional dependency or unreadable file
                read_errors.append(f"{rel}: parquet not parsed: {exc}")
    except Exception as exc:
        read_errors.append(f"{rel}: parse failed: {exc}")

    file_records.append(
        {
            "path": rel,
            "bytes": size,
            "sha256": sha.hexdigest(),
            "parsed_result_rows": parsed,
        }
    )

manifest_path.parent.mkdir(parents=True, exist_ok=True)
with manifest_path.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["path", "bytes", "sha256", "parsed_result_rows"])
    writer.writeheader()
    writer.writerows(file_records)


def distinct(key: str) -> list[str]:
    return sorted({norm_value(row.get(key)) for row in rows if norm_value(row.get(key))})


def is_true(value: Any) -> bool | None:
    text = norm_value(value)
    if text in {"true", "1", "yes", "y", "pass", "passed", "ok"}:
        return True
    if text in {"false", "0", "no", "n", "fail", "failed", "violation"}:
        return False
    return None


def as_number(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        number = float(value)
        return number
    except (TypeError, ValueError):
        return None


def combined_label(row: dict[str, Any]) -> str:
    return " ".join(filter(None, [norm_value(row.get("dataset")), norm_value(row.get("task"))]))


datasets = distinct("dataset")
tasks = distinct("task")
models = distinct("model")
seeds = distinct("seed")
seqs = distinct("sequence_length")
budgets = distinct("memory_budget")

fatal_statuses: list[str] = []
oom_rows = 0
budget_violations: list[str] = []
nonfinite_metrics: list[str] = []

for idx, row in enumerate(rows, 1):
    source = row.get("_source_file", "?")
    status = norm_value(row.get("status"))
    oom_flag = is_true(row.get("oom"))
    if oom_flag is True or "oom" in status or "out of memory" in status:
        oom_rows += 1
    elif status in {"failed", "failure", "error", "crashed", "aborted"}:
        fatal_statuses.append(f"row {idx} in {source}: status={status}")

    budget_pass = is_true(row.get("budget_pass"))
    if budget_pass is False:
        budget_violations.append(f"row {idx} in {source}: budget_pass={row.get('budget_pass')}")
    configured = as_number(row.get("memory_budget"))
    observed = as_number(row.get("max_memory_size"))
    if configured is not None and observed is not None and observed > configured + 1e-9:
        budget_violations.append(
            f"row {idx} in {source}: observed max memory {observed} exceeds budget {configured}"
        )

    raw = row.get("_raw", {})
    if isinstance(raw, dict):
        for key, value in raw.items():
            if norm_name(str(key)) not in ALIAS_NORM["metric"]:
                continue
            number = as_number(value)
            if number is not None and not math.isfinite(number):
                nonfinite_metrics.append(f"row {idx} in {source}: {key}={value}")

# Duplicate detection uses only rows with a meaningful experiment identity.
duplicate_counter: Counter[tuple[str, ...]] = Counter()
for row in rows:
    identity = tuple(
        norm_value(row.get(key))
        for key in ("dataset", "task", "model", "sequence_length", "memory_budget", "seed")
    )
    if sum(bool(x) for x in identity) >= 4:
        duplicate_counter[identity] += 1
duplicates = {"|".join(k): v for k, v in duplicate_counter.items() if v > 1}

# Required coverage checks. These are intentionally alias-tolerant.
all_labels = [combined_label(row) for row in rows]
model_blob = "\n".join(models)
label_blob = "\n".join(all_labels)


def has_any(blob: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, blob, flags=re.I) for pattern in patterns)

coverage = {
    "hdfs": has_any(label_blob, [r"\bhdfs\b"]),
    "imdb": has_any(label_blob, [r"\bimdb\b"]),
    "bgl": has_any(label_blob, [r"\bbgl\b"]),
    "gru": has_any(model_blob, [r"(^|\b)gru($|\b)"]),
    "budgetmem_r": has_any(model_blob, [r"budget[ _-]*mem", r"budgetmem"]),
    "transformer": has_any(model_blob, [r"transformer"]),
    "mamba": has_any(model_blob, [r"mamba"]),
    "rmt": has_any(model_blob, [r"(^|\b)rmt($|\b)", r"recurrent memory transformer"]),
    "lstm": has_any(model_blob, [r"lstm"]),
}

# BGL should contain the strongest three methods. Count models represented in BGL rows.
bgl_models = sorted(
    {
        norm_value(row.get("model"))
        for row in rows
        if "bgl" in combined_label(row) and norm_value(row.get("model"))
    }
)
coverage["bgl_three_methods"] = len(bgl_models) >= 3

# Synthetic matrix fallback checks. Prefer plan-vs-result comparison when a plan file is available.
plan_files = [
    p for p in files
    if "plan" in p.name.lower() or "matrix_manifest" in p.name.lower() or "expected" in p.name.lower()
]
plan_rows: list[tuple[str, ...]] = []
result_rows: list[tuple[str, ...]] = []

for row in rows:
    identity = tuple(
        norm_value(row.get(key))
        for key in ("dataset", "task", "model", "sequence_length", "memory_budget", "seed")
    )
    if sum(bool(x) for x in identity) < 4:
        continue
    source_name = str(row.get("_source_file", "")).lower()
    if any(token in source_name for token in ("plan", "expected", "manifest")):
        plan_rows.append(identity)
    else:
        result_rows.append(identity)

missing_planned: list[str] = []
if plan_rows:
    expected = set(plan_rows)
    observed = set(result_rows)
    missing_planned = ["|".join(x) for x in sorted(expected - observed)]

# If no plan was discoverable, use dimensional coverage as an auditable fallback.
synthetic_rows = [
    row for row in rows
    if not any(name in combined_label(row) for name in ("hdfs", "imdb", "bgl"))
]
synthetic_tasks = sorted({norm_value(r.get("task")) for r in synthetic_rows if norm_value(r.get("task"))})
synthetic_seqs = sorted({norm_value(r.get("sequence_length")) for r in synthetic_rows if norm_value(r.get("sequence_length"))})
synthetic_seeds = sorted({norm_value(r.get("seed")) for r in synthetic_rows if norm_value(r.get("seed"))})
synthetic_budgets = sorted({norm_value(r.get("memory_budget")) for r in synthetic_rows if norm_value(r.get("memory_budget"))})

fallback_dimensions = {
    "synthetic_tasks_at_least_6": len(synthetic_tasks) >= 6,
    "synthetic_sequence_lengths_at_least_7": len(synthetic_seqs) >= 7,
    "synthetic_seeds_at_least_5": len(synthetic_seeds) >= 5,
    "synthetic_budgets_at_least_2": len(synthetic_budgets) >= 2,
}

blocking: list[str] = []
warnings: list[str] = []

if not files:
    blocking.append("No Section 18 result files were discovered.")
if not rows:
    blocking.append("No structured Section 18 result rows could be parsed.")
if read_errors:
    # Parquet dependency omissions are warnings if other structured results are available.
    non_parquet_errors = [x for x in read_errors if "parquet not parsed" not in x]
    if non_parquet_errors:
        blocking.append(f"{len(non_parquet_errors)} result files could not be parsed.")
    else:
        warnings.append(f"{len(read_errors)} Parquet files were not parsed; CSV/JSON evidence was used.")
if fatal_statuses:
    blocking.append(f"{len(fatal_statuses)} non-OOM failed/error result rows were found.")
if budget_violations:
    blocking.append(f"{len(budget_violations)} memory-budget violations were found.")
if nonfinite_metrics:
    blocking.append(f"{len(nonfinite_metrics)} non-finite metric values were found.")

for key in ("hdfs", "imdb", "bgl", "gru", "budgetmem_r", "transformer", "mamba", "rmt", "lstm"):
    if not coverage[key]:
        blocking.append(f"Required coverage missing: {key}")
if not coverage["bgl_three_methods"]:
    blocking.append(f"BGL contains only {len(bgl_models)} distinct model(s); at least 3 are required.")

if plan_rows:
    if missing_planned:
        blocking.append(f"{len(missing_planned)} planned experiment cells are missing from observed results.")
else:
    warnings.append("No machine-readable Section 18 plan manifest was discovered; dimensional fallback checks were used.")
    for key, passed in fallback_dimensions.items():
        if not passed:
            blocking.append(f"Fallback dimensional check failed: {key}")

if duplicates:
    warnings.append(
        f"{len(duplicates)} duplicate experiment identities were found. This may be legitimate for retries, "
        "but the duplicate manifest must be reviewed before committing results."
    )
if oom_rows:
    warnings.append(f"{oom_rows} OOM result row(s) were explicitly reported. They were not treated as silent failures.")

report = {
    "decision": "PASS" if not blocking else "HOLD",
    "result_file_count": len(files),
    "parsed_result_rows": len(rows),
    "datasets": datasets,
    "tasks": tasks,
    "models": models,
    "seeds": seeds,
    "sequence_lengths": seqs,
    "memory_budgets": budgets,
    "bgl_models": bgl_models,
    "coverage": coverage,
    "fallback_dimensions": fallback_dimensions,
    "plan_rows": len(set(plan_rows)),
    "observed_identity_rows": len(set(result_rows)),
    "missing_planned_cells": missing_planned[:200],
    "duplicate_identities": dict(list(sorted(duplicates.items()))[:200]),
    "oom_rows": oom_rows,
    "fatal_statuses": fatal_statuses[:200],
    "budget_violations": budget_violations[:200],
    "nonfinite_metrics": nonfinite_metrics[:200],
    "read_errors": read_errors[:200],
    "warnings": warnings,
    "blocking_reasons": blocking,
    "summary_file": summary_path.relative_to(root).as_posix(),
    "matrix_log": log_path.relative_to(root).as_posix(),
    "manifest": manifest_path.relative_to(root).as_posix(),
}
json_report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

print("Structured result audit")
print(f"  Result files discovered: {len(files)}")
print(f"  Parsed result rows: {len(rows)}")
print(f"  Distinct datasets: {len(datasets)}")
print(f"  Distinct tasks: {len(tasks)}")
print(f"  Distinct models: {len(models)}")
print(f"  Distinct seeds: {len(seeds)}")
print(f"  Distinct sequence lengths: {len(seqs)}")
print(f"  Distinct memory budgets: {len(budgets)}")
print(f"  Explicitly reported OOM rows: {oom_rows}")
print(f"  Duplicate identities: {len(duplicates)}")
print(f"  Result manifest: {manifest_path.relative_to(root)}")
print(f"  JSON audit: {json_report_path.relative_to(root)}")
print()

print("Required coverage")
for key, value in coverage.items():
    print(f"  {key}: {'PASS' if value else 'FAIL'}")
print()

if plan_rows:
    print("Plan reconciliation")
    print(f"  Planned unique cells: {len(set(plan_rows))}")
    print(f"  Observed unique cells: {len(set(result_rows))}")
    print(f"  Missing planned cells: {len(missing_planned)}")
else:
    print("Dimensional fallback")
    for key, value in fallback_dimensions.items():
        print(f"  {key}: {'PASS' if value else 'FAIL'}")
print()

if warnings:
    print("Warnings")
    for item in warnings:
        print(f"  - {item}")
    print()

if blocking:
    print("Blocking reasons")
    for item in blocking:
        print(f"  - {item}")
    print()
    print("PYTHON_AUDIT_DECISION=HOLD")
    raise SystemExit(3)

print("Blocking reasons")
print("  - None")
print()
print("PYTHON_AUDIT_DECISION=PASS")
PY
PY_STATUS=$?
set -e

if [[ $PY_STATUS -ne 0 ]]; then
  echo
  echo "FINAL DECISION: HOLD SECTION 18 RESULT COMMIT"
  echo "Review: $REPORT"
  echo "JSON evidence: $JSON_REPORT"
  cp "$REPORT" "$LATEST"
  exit "$PY_STATUS"
fi

echo
printf '%s\n' "======================================================================"
printf '%s\n' "SECTION 18 FULL MATRIX AUDIT RESULT"
printf '%s\n' "======================================================================"
echo "FINAL DECISION: SECTION 18 FULL MATRIX AUDIT PASSED"
echo "Generated result artifacts staged: NO"
echo "Git commit created by this audit: NO"
echo "Git push performed by this audit: NO"
echo "Audit report: $REPORT"
echo "JSON evidence: $JSON_REPORT"
echo "Result-file manifest: $MANIFEST"
echo
cp "$REPORT" "$LATEST"
echo "Latest audit: $LATEST"
