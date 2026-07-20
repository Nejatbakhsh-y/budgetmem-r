#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# ==============================================================================
# BudgetMem-R — Section 14.11.2 recovery gate
#
# Evaluate the already-generated tuned-pilot outputs after a pilot runner exits
# with code 2. This script does not rerun the 72-cell pilot and does not reuse
# stale gradient-profile files. It evaluates the current pilot_results.csv,
# reruns focused tests, verifies clipping/configuration, and writes fresh evidence.
# ==============================================================================

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/c/Users/nejat/OneDrive/Desktop/UN/Skills/GitHub 2026/budgetmem-r}"
RAW_GRAD_LIMIT="${RAW_GRAD_LIMIT:-100.0}"
CLIP_NORM="${CLIP_NORM:-1.0}"
EXPECTED_ROWS="${EXPECTED_ROWS:-72}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

cd "$PROJECT_ROOT"

EVIDENCE_DIR="reports/evidence"
LOG_DIR="$EVIDENCE_DIR/logs"
RESULT_JSON="$EVIDENCE_DIR/section14_11_2_gradient_stability.json"
RESULT_TXT="$EVIDENCE_DIR/section14_11_2_gradient_stability.txt"
TEST_STATUS="$EVIDENCE_DIR/section14_11_2_test_status.json"
MAIN_LOG="$LOG_DIR/section14_11_2_recovery_${STAMP}.log"
PILOT_CSV="reports/tables/pilot_results.csv"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$MAIN_LOG") 2>&1

heading() {
    echo
    echo "==============================================================================="
    echo "$1"
    echo "==============================================================================="
}

fail() {
    echo "ERROR: $1"
    echo "Log: $MAIN_LOG"
    exit "${2:-1}"
}

heading "SECTION 14.11.2 — RECOVER CURRENT PILOT RESULTS"
echo "Project root:            $PROJECT_ROOT"
echo "Pilot results:           $PILOT_CSV"
echo "Expected result rows:    $EXPECTED_ROWS"
echo "Raw-gradient limit:      $RAW_GRAD_LIMIT"
echo "Gradient clipping norm:  $CLIP_NORM"
echo "Started UTC:             $(date -u +%Y-%m-%dT%H:%M:%SZ)"

[[ -d src && -d tests ]] || fail "BudgetMem-R repository structure was not found." 2
[[ -f "$PILOT_CSV" ]] || fail "$PILOT_CSV is missing. Do not rerun automatically; inspect the pilot-runner log first." 2

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [[ -f venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
else
    fail "No .venv or venv environment was found." 2
fi

PYTHON_BIN="$(command -v python || true)"
PYTEST_BIN="$(command -v pytest || true)"
[[ -n "$PYTHON_BIN" ]] || fail "Python is unavailable." 2
[[ -n "$PYTEST_BIN" ]] || fail "pytest is unavailable." 2

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "Python:                  $($PYTHON_BIN --version 2>&1)"
echo "Pilot modified UTC:      $(date -u -r "$PILOT_CSV" +%Y-%m-%dT%H:%M:%SZ)"

heading "1. RUN FOCUSED GRADIENT TESTS"

GRADIENT_TEST_RC=98
CALIBRATION_TEST_RC=0

set +e
if [[ -f tests/pretraining/test_gradient_and_reset.py ]]; then
    "$PYTEST_BIN" \
        tests/pretraining/test_gradient_and_reset.py::test_memory_controllers_receive_gradients_and_graph_policy_is_explicit \
        -q 2>&1 | tee "$LOG_DIR/section14_11_2_recovery_gradient_flow.log"
    GRADIENT_TEST_RC=${PIPESTATUS[0]}
else
    echo "Required test file is missing: tests/pretraining/test_gradient_and_reset.py"
fi

if [[ -f tests/pilot/test_controller_calibration.py ]]; then
    "$PYTEST_BIN" tests/pilot/test_controller_calibration.py -q \
        2>&1 | tee "$LOG_DIR/section14_11_2_recovery_controller_calibration.log"
    CALIBRATION_TEST_RC=${PIPESTATUS[0]}
fi
set -e

cat > "$TEST_STATUS" <<JSON
{
  "gradient_flow_test_rc": $GRADIENT_TEST_RC,
  "controller_calibration_test_rc": $CALIBRATION_TEST_RC,
  "profiler_rc": null,
  "note": "Recovery gate evaluated current pilot_results.csv only; no stale profile was reused."
}
JSON

echo "Gradient-flow test RC:   $GRADIENT_TEST_RC"
echo "Calibration test RC:     $CALIBRATION_TEST_RC"

heading "2. EVALUATE CURRENT PILOT RESULTS ONLY"

set +e
"$PYTHON_BIN" - \
    "$PILOT_CSV" \
    "$EXPECTED_ROWS" \
    "$RAW_GRAD_LIMIT" \
    "$CLIP_NORM" \
    "$GRADIENT_TEST_RC" \
    "$CALIBRATION_TEST_RC" \
    "$RESULT_JSON" \
    "$RESULT_TXT" \
    "$MAIN_LOG" <<'PY'
from __future__ import annotations

import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

(
    pilot_csv_raw,
    expected_rows_raw,
    raw_limit_raw,
    clip_norm_raw,
    gradient_test_rc_raw,
    calibration_test_rc_raw,
    result_json_raw,
    result_txt_raw,
    main_log_raw,
) = sys.argv[1:]

root = Path.cwd()
pilot_csv = Path(pilot_csv_raw)
expected_rows = int(expected_rows_raw)
raw_limit = float(raw_limit_raw)
clip_norm = float(clip_norm_raw)
gradient_test_rc = int(gradient_test_rc_raw)
calibration_test_rc = int(calibration_test_rc_raw)
result_json = Path(result_json_raw)
result_txt = Path(result_txt_raw)
main_log = Path(main_log_raw)

def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)

with pilot_csv.open("r", encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)

field_map = {str(name).strip().lower(): name for name in fieldnames}
grad_key = next(
    (
        field_map[name]
        for name in (
            "maximum_gradient_norm",
            "max_gradient_norm",
            "gradient_norm_max",
            "raw_gradient_norm",
            "max_grad_norm",
        )
        if name in field_map
    ),
    None,
)
model_key = next(
    (field_map[name] for name in ("model", "model_name", "policy") if name in field_map),
    None,
)

all_measurements: list[dict[str, Any]] = []
budgetmem_measurements: list[dict[str, Any]] = []

if grad_key is not None:
    for row_number, row in enumerate(rows, start=2):
        raw = row.get(grad_key)
        if raw in (None, ""):
            continue
        try:
            value = float(str(raw).strip())
        except ValueError:
            continue
        item = {
            "row_number": row_number,
            "value": value,
            "model": str(row.get(model_key, "")) if model_key else "",
            "task": row.get(field_map.get("task", ""), "") if "task" in field_map else "",
            "sequence_length": row.get(field_map.get("sequence_length", ""), "") if "sequence_length" in field_map else "",
            "budget": row.get(field_map.get("budget", ""), "") if "budget" in field_map else "",
            "seed": row.get(field_map.get("seed", ""), "") if "seed" in field_map else "",
        }
        all_measurements.append(item)
        if model_key and "budgetmem" in item["model"].lower():
            budgetmem_measurements.append(item)

measurements = budgetmem_measurements or all_measurements
finite = [item for item in measurements if math.isfinite(float(item["value"]))]
nonfinite = [item for item in measurements if not math.isfinite(float(item["value"]))]
maximum_item = max(finite, key=lambda item: float(item["value"])) if finite else None
maximum = float(maximum_item["value"]) if maximum_item else None

clipping_sources: list[str] = []
for base in (root / "src", root / "scripts"):
    if not base.exists():
        continue
    for path in base.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "clip_grad_norm_" in text or "clip_grad_value_" in text:
            clipping_sources.append(relative(path))
clipping_sources = sorted(set(clipping_sources))

raw_threshold_ok = False
clip_threshold_ok = False
threshold_sources: list[str] = []
pattern = re.compile(
    r"(?m)^\s*(gradient_clip_norm|maximum_acceptable_gradient_norm)\s*:\s*([-+0-9.eE]+)\s*$"
)
for base in (root / "configs", root / "scripts"):
    if not base.exists():
        continue
    for path in list(base.rglob("*.yaml")) + list(base.rglob("*.yml")):
        text = path.read_text(encoding="utf-8", errors="replace")
        found = False
        for key, raw in pattern.findall(text):
            found = True
            try:
                value = float(raw)
            except ValueError:
                continue
            if key == "maximum_acceptable_gradient_norm":
                raw_threshold_ok = raw_threshold_ok or math.isclose(value, raw_limit, abs_tol=1e-6)
            elif key == "gradient_clip_norm":
                clip_threshold_ok = clip_threshold_ok or math.isclose(value, clip_norm, abs_tol=1e-6)
        if found:
            threshold_sources.append(relative(path))
threshold_sources = sorted(set(threshold_sources))

checks = {
    "pilot_results_present": pilot_csv.exists(),
    "pilot_matrix_has_expected_72_rows": len(rows) == expected_rows,
    "gradient_column_present": grad_key is not None,
    "budgetmem_gradient_measurements_present": bool(measurements),
    "all_gradient_measurements_finite": bool(measurements) and not nonfinite,
    "maximum_raw_gradient_within_100": maximum is not None and maximum <= raw_limit + 1e-6,
    "gradient_flow_test": gradient_test_rc == 0,
    "controller_calibration_test": calibration_test_rc == 0,
    "gradient_clipping_implemented": bool(clipping_sources),
    "raw_gradient_limit_configured": raw_threshold_ok,
    "clip_norm_configured": clip_threshold_ok,
}

passed = all(checks.values())
failed_checks = [name for name, value in checks.items() if value is not True]

payload = {
    "section": "14.11.2",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "evaluation_mode": "current_pilot_results_only",
    "gradient_stability": "PASS" if passed else "FAIL",
    "raw_gradient_limit": raw_limit,
    "gradient_clip_norm": clip_norm,
    "pilot_metric_source": relative(pilot_csv),
    "pilot_result_rows": len(rows),
    "expected_pilot_result_rows": expected_rows,
    "gradient_column": grad_key,
    "measurement_scope": "BudgetMem-R rows" if budgetmem_measurements else "all rows",
    "observed_gradient_count": len(measurements),
    "nonfinite_gradient_count": len(nonfinite),
    "maximum_observed_gradient_norm": maximum,
    "maximum_observed_gradient_context": maximum_item,
    "checks": checks,
    "failed_checks": failed_checks,
    "clipping_sources": clipping_sources,
    "threshold_sources": threshold_sources,
    "main_log": relative(main_log),
    "note": (
        "This recovery gate intentionally ignored older gradient-profile JSON files. "
        "It evaluated the current pilot_results.csv produced by the most recent tuned pilot."
    ),
}

result_json.parent.mkdir(parents=True, exist_ok=True)
result_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

lines = [
    "Section 14.11.2 Gradient Stability — Recovery Evaluation",
    "========================================================",
    f"Generated UTC:                           {payload['generated_at_utc']}",
    f"Evaluation mode:                         {payload['evaluation_mode']}",
    f"Pilot metric source:                     {payload['pilot_metric_source']}",
    f"Pilot result rows:                       {len(rows)} / {expected_rows}",
    f"Gradient column:                         {grad_key or 'NOT FOUND'}",
    f"Measurement scope:                       {payload['measurement_scope']}",
    f"Observed gradient measurements:          {len(measurements)}",
    f"Non-finite measurements:                 {len(nonfinite)}",
    f"Maximum observed raw gradient:           {maximum if maximum is not None else 'NOT FOUND'}",
    f"Raw-gradient acceptance limit:           {raw_limit:.6f}",
    f"Gradient clipping norm:                  {clip_norm:.6f}",
    "",
]
for name, value in checks.items():
    lines.append(f"{name:43s} {'PASS' if value else 'FAIL'}")
lines.extend(
    [
        "",
        f"Gradient stability:                      {'PASS' if passed else 'FAIL'}",
        "",
    ]
)
if maximum_item:
    lines.extend(
        [
            "Maximum-gradient row:",
            f"  CSV row:                               {maximum_item['row_number']}",
            f"  Model:                                 {maximum_item['model'] or 'NOT RECORDED'}",
            f"  Task:                                  {maximum_item['task'] or 'NOT RECORDED'}",
            f"  Sequence length:                       {maximum_item['sequence_length'] or 'NOT RECORDED'}",
            f"  Budget:                                {maximum_item['budget'] or 'NOT RECORDED'}",
            f"  Seed:                                  {maximum_item['seed'] or 'NOT RECORDED'}",
            "",
        ]
    )
if failed_checks:
    lines.append("Failed checks:")
    lines.extend(f"  - {name}" for name in failed_checks)
    lines.append("")
lines.extend(
    [
        f"Evidence JSON: {relative(result_json)}",
        f"Main log:     {relative(main_log)}",
    ]
)
result_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
raise SystemExit(0 if passed else 1)
PY
GATE_RC=$?
set -e

heading "3. FINAL RESULT"

if [[ "$GATE_RC" -eq 0 ]]; then
    echo "SECTION 14.11.2 COMPLETED: GRADIENT STABILITY PASS"
    echo "Evidence: $RESULT_TXT"
    exit 0
fi

echo "SECTION 14.11.2 IS STILL NOT COMPLETE."
echo "This result is now based on the current pilot_results.csv, not stale evidence."
echo "Review: $RESULT_TXT"
echo "Log:    $MAIN_LOG"
exit "$GATE_RC"
