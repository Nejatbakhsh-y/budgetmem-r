#!/usr/bin/env bash
set -Eeuo pipefail

if git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    cd "$git_root"
else
    echo "ERROR: Run this automation from inside the budgetmem-r repository." >&2
    exit 1
fi

PYTHON=".venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: WSL Python was not found at $PYTHON" >&2
    exit 1
fi

export PYTHONPATH="$PWD/src"
export PYTHONHASHSEED="2026"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"

CONFIG="configs/experiments/pilot.yaml"
RUNNER="scripts/run_pilot.py"
SUMMARY="reports/evidence/pilot_summary.json"
GATE="reports/evidence/pilot_go_no_go.json"
RESULTS="reports/tables/pilot_results.csv"
REPORT_JSON="reports/evidence/section14_4_operational_gate.json"
REPORT_TXT="reports/evidence/section14_4_operational_gate.txt"

required_files=(
    "$CONFIG"
    "$RUNNER"
    "$SUMMARY"
    "$GATE"
    "$RESULTS"
)

echo
echo "============================================================"
echo "SECTION 14.4"
echo "STABILITY, RESOURCE, PROVENANCE, AND RESUMPTION VALIDATION"
echo "============================================================"

for path in "${required_files[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: Required pilot artifact is missing: $path" >&2
        echo "Run the complete Section 15 pilot before this validation." >&2
        exit 1
    fi
done

echo
echo "1. Running focused pilot tests."
"$PYTHON" -m pytest tests/pilot -q

echo
echo "2. Explicitly testing checkpoint resumption."
"$PYTHON" "$RUNNER" \
    --config "$CONFIG" \
    --resume

echo
echo "3. Validating operational evidence."

"$PYTHON" - <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

config_path = Path("configs/experiments/pilot.yaml").resolve()
summary_path = Path("reports/evidence/pilot_summary.json")
gate_path = Path("reports/evidence/pilot_go_no_go.json")
results_path = Path("reports/tables/pilot_results.csv")

report_json_path = Path(
    "reports/evidence/section14_4_operational_gate.json"
)
report_txt_path = Path(
    "reports/evidence/section14_4_operational_gate.txt"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_key(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return None

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "true",
            "pass",
            "passed",
            "yes",
            "success",
            "successful",
            "1",
        }:
            return True

        if normalized in {
            "false",
            "fail",
            "failed",
            "no",
            "0",
        }:
            return False

    if isinstance(value, dict):
        for candidate in ("passed", "pass", "status", "result"):
            if candidate in value:
                converted = as_bool(value[candidate])
                if converted is not None:
                    return converted

    return None


def collect_key_values(
    payload: Any,
    accepted_keys: set[str],
) -> list[Any]:
    collected: list[Any] = []

    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = normalize_key(key)

            if normalized in accepted_keys:
                collected.append(value)

            collected.extend(
                collect_key_values(value, accepted_keys)
            )

    elif isinstance(payload, list):
        for value in payload:
            collected.extend(
                collect_key_values(value, accepted_keys)
            )

    return collected


def all_boolean_values(
    payloads: list[Any],
    accepted_keys: set[str],
) -> tuple[bool, list[bool]]:
    raw_values: list[Any] = []

    for payload in payloads:
        raw_values.extend(
            collect_key_values(payload, accepted_keys)
        )

    converted = [
        value
        for raw in raw_values
        if (value := as_bool(raw)) is not None
    ]

    return bool(converted) and all(converted), converted


def criterion_from_gate(
    criteria: dict[str, Any],
    required_words: tuple[str, ...],
) -> bool | None:
    matches: list[bool] = []

    for name, value in criteria.items():
        normalized = normalize_key(name)

        if any(word in normalized for word in required_words):
            converted = as_bool(value)

            if converted is not None:
                matches.append(converted)

    if not matches:
        return None

    return all(matches)


summary = json.loads(summary_path.read_text(encoding="utf-8"))
gate = json.loads(gate_path.read_text(encoding="utf-8"))
criteria = gate.get("criteria", {})

if not isinstance(criteria, dict):
    criteria = {}

with results_path.open(
    "r",
    newline="",
    encoding="utf-8",
) as handle:
    result_rows = list(csv.DictReader(handle))

if not result_rows:
    raise SystemExit("ERROR: pilot_results.csv contains no rows.")

payloads: list[Any] = [summary, gate, result_rows]

# ---------------------------------------------------------
# 1. Stability
# ---------------------------------------------------------

stability_direct = criterion_from_gate(
    criteria,
    ("stability", "stable"),
)

stability_derived, stability_values = all_boolean_values(
    payloads,
    {
        "stability_pass",
        "stable",
        "training_stability_pass",
        "finite_loss_pass",
    },
)

stability_pass = (
    stability_direct
    if stability_direct is not None
    else stability_derived
)

# ---------------------------------------------------------
# 2. Resource measurement
# ---------------------------------------------------------

resource_direct = criterion_from_gate(
    criteria,
    ("resource", "latency", "memory_measurement"),
)

resource_derived, resource_values = all_boolean_values(
    payloads,
    {
        "resource_measurement_pass",
        "resource_pass",
        "measurement_pass",
    },
)

resource_numeric_columns = [
    column
    for column in result_rows[0]
    if any(
        token in normalize_key(column)
        for token in (
            "latency",
            "throughput",
            "peak_rss",
            "memory_mb",
            "wall_time",
            "elapsed",
        )
    )
]

numeric_resource_values: list[float] = []

for row in result_rows:
    for column in resource_numeric_columns:
        raw = row.get(column)

        if raw in (None, ""):
            continue

        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            continue

        if math.isfinite(numeric):
            numeric_resource_values.append(numeric)

numeric_resource_pass = (
    bool(resource_numeric_columns)
    and bool(numeric_resource_values)
    and all(value >= 0.0 for value in numeric_resource_values)
)

resource_pass = (
    resource_direct
    if resource_direct is not None
    else resource_derived and numeric_resource_pass
)

# ---------------------------------------------------------
# 3. Configuration provenance
# ---------------------------------------------------------

provenance_direct = criterion_from_gate(
    criteria,
    ("provenance", "configuration", "config_hash"),
)

actual_config_sha256 = sha256_file(config_path)

hash_values = [
    str(value).strip()
    for value in collect_key_values(
        payloads,
        {
            "config_sha256",
            "configuration_sha256",
            "config_hash",
            "configuration_hash",
        },
    )
    if str(value).strip()
]

matching_hash_values = [
    value
    for value in hash_values
    if value == actual_config_sha256
]

config_path_values = [
    str(value).strip()
    for value in collect_key_values(
        payloads,
        {
            "config_path",
            "configuration_path",
        },
    )
    if str(value).strip()
]

existing_config_paths: list[str] = []

for raw_path in config_path_values:
    candidate = Path(raw_path)

    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate

    if candidate.exists():
        existing_config_paths.append(str(candidate.resolve()))

effective_config_files = sorted(
    str(path)
    for path in Path("outputs").glob(
        "**/*effective*config*.y*ml"
    )
    if path.is_file()
)

provenance_derived = (
    bool(hash_values)
    and len(matching_hash_values) == len(hash_values)
    and bool(config_path_values)
    and len(existing_config_paths) == len(config_path_values)
)

provenance_pass = (
    provenance_direct
    if provenance_direct is not None
    else provenance_derived
)

# ---------------------------------------------------------
# 4. Checkpoint resumption
# ---------------------------------------------------------

resume_direct = criterion_from_gate(
    criteria,
    ("checkpoint", "resume", "resumption"),
)

resume_derived, resume_values = all_boolean_values(
    payloads,
    {
        "checkpoint_resume_pass",
        "checkpoint_resumption_pass",
        "resume_pass",
        "resumption_pass",
    },
)

checkpoint_files = sorted(
    str(path)
    for path in Path("outputs").glob("**/checkpoints/**/*")
    if path.is_file()
)

checkpoint_resume_pass = (
    resume_direct
    if resume_direct is not None
    else resume_derived and bool(checkpoint_files)
)

checks = {
    "stability_pass": bool(stability_pass),
    "resource_measurement_pass": bool(resource_pass),
    "configuration_provenance_pass": bool(provenance_pass),
    "checkpoint_resumption_pass": bool(
        checkpoint_resume_pass
    ),
}

final_pass = all(checks.values())

report = {
    "section": "14.4",
    "status": "PASS" if final_pass else "FAIL",
    "checks": checks,
    "evidence": {
        "pilot_summary": str(summary_path),
        "pilot_gate": str(gate_path),
        "pilot_results": str(results_path),
        "configuration_file": str(config_path),
        "actual_configuration_sha256": actual_config_sha256,
        "recorded_configuration_hash_count": len(hash_values),
        "matching_configuration_hash_count": len(
            matching_hash_values
        ),
        "recorded_configuration_paths": config_path_values,
        "existing_configuration_paths": existing_config_paths,
        "effective_configuration_files": effective_config_files,
        "checkpoint_file_count": len(checkpoint_files),
        "resource_numeric_columns": resource_numeric_columns,
        "resource_numeric_value_count": len(
            numeric_resource_values
        ),
        "stability_evidence_values": stability_values,
        "resource_evidence_values": resource_values,
        "resume_evidence_values": resume_values,
    },
}

report_json_path.parent.mkdir(parents=True, exist_ok=True)
report_json_path.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)

lines = [
    "SECTION 14.4 OPERATIONAL VALIDATION",
    "===================================",
    "",
]

for name, passed in checks.items():
    label = name.removesuffix("_pass").replace("_", " ")
    lines.append(
        f"{'PASS' if passed else 'FAIL'}: {label}"
    )

lines.extend(
    [
        "",
        f"FINAL STATUS: {report['status']}",
        "",
        f"Configuration SHA-256: {actual_config_sha256}",
        f"Pilot result rows: {len(result_rows)}",
        f"Checkpoint files found: {len(checkpoint_files)}",
        (
            "Resource columns: "
            + ", ".join(resource_numeric_columns)
        ),
        "",
        "Evidence files:",
        f"  {summary_path}",
        f"  {gate_path}",
        f"  {results_path}",
        f"  {report_json_path}",
    ]
)

report_txt_path.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print()
print("\n".join(lines))

if not final_pass:
    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    raise SystemExit(
        "Section 14.4 failed: " + ", ".join(failed)
    )
PY

echo
echo "4. Checking repository whitespace."
git diff --check

echo
echo "============================================================"
echo "SECTION 14.4 COMPLETED SUCCESSFULLY"
echo "============================================================"
echo "Evidence:"
echo "  reports/evidence/section14_4_operational_gate.json"
echo "  reports/evidence/section14_4_operational_gate.txt"
echo
echo "Required final result:"
echo "  PASS: stability"
echo "  PASS: resource measurement"
echo "  PASS: configuration provenance"
echo "  PASS: checkpoint resumption"
echo "  FINAL STATUS: PASS"
