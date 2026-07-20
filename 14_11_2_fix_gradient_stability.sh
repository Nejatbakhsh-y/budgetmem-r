#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# ==============================================================================
# BudgetMem-R — Section 14.11.2
# Repair the gradient-stability profiler/gate, rerun the required checks,
# and regenerate auditable evidence without forcing the final GO decision.
#
# Run in the VS Code WSL terminal:
#   bash 14_11_2_fix_gradient_stability.sh
# ==============================================================================

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/c/Users/nejat/OneDrive/Desktop/UN/Skills/GitHub 2026/budgetmem-r}"
RAW_GRAD_LIMIT="${RAW_GRAD_LIMIT:-100.0}"
CLIP_NORM="${CLIP_NORM:-1.0}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

cd "$PROJECT_ROOT"

EVIDENCE_DIR="reports/evidence"
LOG_DIR="$EVIDENCE_DIR/logs"
BACKUP_DIR="$EVIDENCE_DIR/backups/section14_11_2_$STAMP"
MAIN_LOG="$LOG_DIR/section14_11_2_$STAMP.log"
TEST_STATUS="$EVIDENCE_DIR/section14_11_2_test_status.json"
RESULT_JSON="$EVIDENCE_DIR/section14_11_2_gradient_stability.json"
RESULT_TXT="$EVIDENCE_DIR/section14_11_2_gradient_stability.txt"

mkdir -p "$LOG_DIR" "$BACKUP_DIR" scripts
exec > >(tee -a "$MAIN_LOG") 2>&1

fail() {
    local rc="${2:-1}"
    echo "ERROR: $1"
    echo "Log: $MAIN_LOG"
    exit "$rc"
}

heading() {
    echo
    echo "=============================================================================== "
    echo "$1"
    echo "=============================================================================== "
}

heading "SECTION 14.11.2 — GRADIENT STABILITY REPAIR"
echo "Project:             $PROJECT_ROOT"
echo "Raw-gradient limit:  $RAW_GRAD_LIMIT"
echo "Gradient clip norm:  $CLIP_NORM"
echo "Started:             $(date -u +%Y-%m-%dT%H:%M:%SZ)"

[[ -d src && -d tests ]] || fail "BudgetMem-R repository structure was not found." 2

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [[ -f venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
else
    fail "No .venv or venv environment was found." 2
fi

PYTHON_BIN="$(command -v python)"
PYTEST_BIN="$(command -v pytest || true)"
[[ -n "$PYTEST_BIN" ]] || fail "pytest is not installed in the active environment." 2

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "Python:              $($PYTHON_BIN --version 2>&1)"
echo "Branch:              $(git branch --show-current 2>/dev/null || echo unknown)"

heading "1. PATCH THE BudgetMemRAdapter PROFILER STATE"

"$PYTHON_BIN" - "$BACKUP_DIR" <<'PY'
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

backup_root = Path(sys.argv[1])
found = 0
patched = 0


def insertion_index(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Zero-based line insertion point after a method docstring, if present."""
    if not node.body:
        return node.lineno
    first = node.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return int(first.end_lineno or first.lineno)
    return first.lineno - 1


for root in (Path("scripts"), Path("src")):
    if not root.exists():
        continue
    for path in root.rglob("*.py"):
        try:
            original = path.read_text(encoding="utf-8")
            tree = ast.parse(original)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue

        adapter = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "BudgetMemRAdapter"
            ),
            None,
        )
        if adapter is None:
            continue

        found += 1
        lines = original.splitlines(keepends=True)
        additions: list[tuple[int, str]] = []

        init_method = next(
            (
                node
                for node in adapter.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "__init__"
            ),
            None,
        )
        constructor_method = next(
            (
                node
                for node in adapter.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "_constructor_kwargs"
            ),
            None,
        )

        if init_method is not None:
            args = {
                arg.arg
                for arg in (
                    list(init_method.args.posonlyargs)
                    + list(init_method.args.args)
                    + list(init_method.args.kwonlyargs)
                )
            }
            body = "".join(lines[init_method.lineno - 1 : init_method.end_lineno])
            if "model_cfg" in args and "self._model_cfg" not in body:
                indent = " " * (init_method.col_offset + 4)
                additions.append(
                    (insertion_index(init_method), f"{indent}self._model_cfg = model_cfg\n")
                )

        if constructor_method is not None:
            body = "".join(
                lines[constructor_method.lineno - 1 : constructor_method.end_lineno]
            )
            if "model_cfg" in body and "model_cfg = self._model_cfg" not in body:
                indent = " " * (constructor_method.col_offset + 4)
                additions.append(
                    (
                        insertion_index(constructor_method),
                        f"{indent}model_cfg = self._model_cfg\n",
                    )
                )

        if not additions:
            print(f"Already correct: {path}")
            continue

        backup = backup_root / path
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)

        for index, text in sorted(additions, reverse=True):
            lines.insert(index, text)

        updated = "".join(lines)
        ast.parse(updated)
        path.write_text(updated, encoding="utf-8", newline="\n")
        patched += 1
        print(f"Patched: {path}")

if found == 0:
    print("No BudgetMemRAdapter class was found; no adapter patch was needed.")
else:
    print(f"Adapter files found: {found}; files patched: {patched}")
PY

heading "2. RESTORE THE DOCUMENTED GRADIENT THRESHOLDS"

"$PYTHON_BIN" - "$RAW_GRAD_LIMIT" "$CLIP_NORM" "$BACKUP_DIR" <<'PY'
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

raw_limit, clip_norm, backup_root_raw = sys.argv[1:4]
backup_root = Path(backup_root_raw)
changed = 0
found = 0

for root in (Path("configs"), Path("scripts")):
    if not root.exists():
        continue
    for path in list(root.rglob("*.yaml")) + list(root.rglob("*.yml")):
        original = path.read_text(encoding="utf-8", errors="replace")
        if not any(
            token in original
            for token in (
                "section15_pilot",
                "maximum_acceptable_gradient_norm",
                "gradient_clip_norm",
            )
        ):
            continue
        found += 1
        updated = original
        replacements = {
            "maximum_acceptable_gradient_norm": raw_limit,
            "gradient_clip_norm": clip_norm,
        }
        for key, value in replacements.items():
            pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s*:\s*).*$")
            updated = pattern.sub(rf"\g<1>{value}", updated)

        if updated != original:
            backup = backup_root / path
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copy2(path, backup)
            path.write_text(updated, encoding="utf-8", newline="\n")
            changed += 1
            print(f"Updated: {path}")

print(f"Relevant YAML files found: {found}; changed: {changed}")
PY

heading "3. VERIFY SOURCE SYNTAX"
"$PYTHON_BIN" -m compileall -q src scripts
echo "Source syntax: PASS"

heading "4. RUN REQUIRED GRADIENT TESTS"

GRADIENT_TEST_RC=98
CALIBRATION_TEST_RC=0

set +e
if [[ -f tests/pretraining/test_gradient_and_reset.py ]]; then
    "$PYTEST_BIN" \
        tests/pretraining/test_gradient_and_reset.py::test_memory_controllers_receive_gradients_and_graph_policy_is_explicit \
        -q 2>&1 | tee "$LOG_DIR/section14_11_2_gradient_flow_test.log"
    GRADIENT_TEST_RC=${PIPESTATUS[0]}
else
    echo "Missing: tests/pretraining/test_gradient_and_reset.py"
fi

if [[ -f tests/pilot/test_controller_calibration.py ]]; then
    "$PYTEST_BIN" tests/pilot/test_controller_calibration.py -q \
        2>&1 | tee "$LOG_DIR/section14_11_2_controller_calibration_test.log"
    CALIBRATION_TEST_RC=${PIPESTATUS[0]}
fi
set -e

cat > "$TEST_STATUS" <<JSON
{
  "gradient_flow_test_rc": $GRADIENT_TEST_RC,
  "controller_calibration_test_rc": $CALIBRATION_TEST_RC,
  "profiler_rc": null
}
JSON

[[ "$GRADIENT_TEST_RC" -eq 0 ]] || fail "The required gradient-flow test failed." 3
[[ "$CALIBRATION_TEST_RC" -eq 0 ]] || fail "The controller-calibration test failed." 3

echo "Required gradient tests: PASS"

heading "5. RERUN THE EXISTING GRADIENT PROFILER"

PROFILE_SCRIPT=""
PROFILE_RC=0

while IFS= read -r candidate; do
    [[ "$candidate" == "scripts/section14_11_2_gradient_gate.py" ]] && continue
    PROFILE_SCRIPT="$candidate"
    break
done < <(
    find scripts -maxdepth 3 -type f \
        \( -iname '*14*11*gradient*.py' -o -iname '*gradient*stability*.py' -o -iname '*gradient*profile*.py' \) \
        | sort
)

if [[ -n "$PROFILE_SCRIPT" ]]; then
    echo "Profiler: $PROFILE_SCRIPT"
    set +e
    "$PYTHON_BIN" "$PROFILE_SCRIPT" \
        2>&1 | tee "$LOG_DIR/section14_11_2_profiler.log"
    PROFILE_RC=${PIPESTATUS[0]}
    set -e
    echo "Profiler return code: $PROFILE_RC"
else
    echo "No standalone profiler was found. Official pilot metrics will be used."
fi

"$PYTHON_BIN" - "$TEST_STATUS" "$PROFILE_RC" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
data["profiler_rc"] = int(sys.argv[2])
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY

heading "6. GENERATE THE SECTION 14.11.2 EVIDENCE GATE"

cat > scripts/section14_11_2_gradient_gate.py <<'PY'
from __future__ import annotations

import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports" / "evidence"
RESULT_JSON = EVIDENCE / "section14_11_2_gradient_stability.json"
RESULT_TXT = EVIDENCE / "section14_11_2_gradient_stability.txt"
TEST_STATUS = EVIDENCE / "section14_11_2_test_status.json"
RAW_LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
CLIP_NORM = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
TOL = 1e-6


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def collect_from_official_csv() -> tuple[list[float], str | None]:
    preferred = ROOT / "reports" / "tables" / "pilot_results.csv"
    candidates = [preferred] if preferred.exists() else []
    candidates.extend(
        sorted(
            (ROOT / "reports").rglob("*pilot*result*.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = {str(name).strip().lower(): name for name in reader.fieldnames or []}
            grad_key = next(
                (
                    fields[name]
                    for name in (
                        "maximum_gradient_norm",
                        "max_gradient_norm",
                        "gradient_norm_max",
                        "raw_gradient_norm",
                        "max_grad_norm",
                    )
                    if name in fields
                ),
                None,
            )
            if grad_key is None:
                continue
            model_key = next(
                (fields[name] for name in ("model", "model_name", "policy") if name in fields),
                None,
            )
            all_values: list[float] = []
            budgetmem_values: list[float] = []
            for row in reader:
                raw = row.get(grad_key)
                if raw in (None, ""):
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                all_values.append(value)
                model = str(row.get(model_key, "")).lower() if model_key else ""
                if "budgetmem" in model:
                    budgetmem_values.append(value)
            values = budgetmem_values or all_values
            if values:
                return values, str(path.relative_to(ROOT))
    return [], None


def recursively_find_gradient_values(obj: Any) -> list[float]:
    keys = {
        "raw_gradient_norm",
        "maximum_gradient_norm",
        "max_gradient_norm",
        "max_raw_gradient_norm",
    }
    values: list[float] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in keys and isinstance(value, (int, float)):
                values.append(float(value))
            values.extend(recursively_find_gradient_values(value))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(recursively_find_gradient_values(value))
    return values


def collect_from_latest_profile_json() -> tuple[list[float], str | None]:
    candidates = []
    for path in EVIDENCE.rglob("*.json"):
        if path in {RESULT_JSON, TEST_STATUS}:
            continue
        lower = path.name.lower()
        if "gradient" in lower or "profile" in lower or "diagnostic" in lower:
            candidates.append(path)
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        values = recursively_find_gradient_values(load_json(path))
        if values:
            return values, str(path.relative_to(ROOT))
    return [], None


def clipping_sources() -> list[str]:
    matches: list[str] = []
    for base in (ROOT / "src", ROOT / "scripts"):
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if path.name == Path(__file__).name:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if "clip_grad_norm_" in text or "clip_grad_value_" in text:
                matches.append(str(path.relative_to(ROOT)))
    return sorted(set(matches))


def threshold_evidence() -> tuple[bool, bool, list[str]]:
    raw_ok = False
    clip_ok = False
    sources: list[str] = []
    pattern = re.compile(
        r"(?m)^\s*(gradient_clip_norm|maximum_acceptable_gradient_norm)\s*:\s*([-+0-9.eE]+)\s*$"
    )
    for base in (ROOT / "configs", ROOT / "scripts"):
        if not base.exists():
            continue
        for path in list(base.rglob("*.yaml")) + list(base.rglob("*.yml")):
            text = path.read_text(encoding="utf-8", errors="replace")
            matched = False
            for key, raw in pattern.findall(text):
                matched = True
                value = float(raw)
                if key == "maximum_acceptable_gradient_norm":
                    raw_ok = raw_ok or math.isclose(value, RAW_LIMIT, abs_tol=TOL)
                if key == "gradient_clip_norm":
                    clip_ok = clip_ok or math.isclose(value, CLIP_NORM, abs_tol=TOL)
            if matched:
                sources.append(str(path.relative_to(ROOT)))
    return raw_ok, clip_ok, sorted(set(sources))


def newest_performance_gate() -> tuple[bool | None, str | None]:
    candidates = []
    for pattern in ("*go*no*go*.json", "*final*decision*.json", "pilot_go_no_go.json"):
        candidates.extend(EVIDENCE.rglob(pattern))
    for path in sorted(set(candidates), key=lambda item: item.stat().st_mtime, reverse=True):
        data = load_json(path)
        if data is None:
            continue

        def visit(obj: Any, prefix: str = "") -> tuple[bool | None, str | None]:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full = f"{prefix}.{key}" if prefix else str(key)
                    lower = full.lower()
                    if "outperform" in lower and ("two" in lower or "2" in lower):
                        if isinstance(value, bool):
                            return value, f"{path.relative_to(ROOT)}:{full}"
                        if isinstance(value, str):
                            upper = value.upper()
                            if upper in {"PASS", "GO"}:
                                return True, f"{path.relative_to(ROOT)}:{full}"
                            if upper in {"FAIL", "NO_GO", "NO-GO"}:
                                return False, f"{path.relative_to(ROOT)}:{full}"
                    result = visit(value, full)
                    if result[0] is not None:
                        return result
            elif isinstance(obj, list):
                for index, value in enumerate(obj):
                    result = visit(value, f"{prefix}[{index}]")
                    if result[0] is not None:
                        return result
            return None, None

        result = visit(data)
        if result[0] is not None:
            return result
    return None, None


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    tests = load_json(TEST_STATUS) or {}
    csv_values, csv_source = collect_from_official_csv()
    profile_values, profile_source = collect_from_latest_profile_json()
    observed = csv_values + profile_values
    finite = [value for value in observed if math.isfinite(value)]
    nonfinite_count = len(observed) - len(finite)
    maximum = max(finite) if finite else None

    clip_sources = clipping_sources()
    raw_threshold_ok, clip_threshold_ok, threshold_sources = threshold_evidence()

    checks = {
        "gradient_flow_test": tests.get("gradient_flow_test_rc") == 0,
        "controller_calibration_test": tests.get("controller_calibration_test_rc", 0) == 0,
        "gradient_measurements_present": bool(observed),
        "all_gradient_measurements_finite": bool(observed) and nonfinite_count == 0,
        "maximum_raw_gradient_within_100": maximum is not None and maximum <= RAW_LIMIT + TOL,
        "gradient_clipping_implemented": bool(clip_sources),
        "raw_gradient_limit_configured": raw_threshold_ok,
        "clip_norm_configured": clip_threshold_ok,
    }
    gradient_pass = all(checks.values())
    performance_pass, performance_source = newest_performance_gate()
    section_complete = gradient_pass and performance_pass is True

    payload = {
        "section": "14.11.2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gradient_stability": "PASS" if gradient_pass else "FAIL",
        "section_14_11": "COMPLETE" if section_complete else "NOT COMPLETE",
        "raw_gradient_limit": RAW_LIMIT,
        "gradient_clip_norm": CLIP_NORM,
        "maximum_observed_gradient_norm": maximum,
        "observed_gradient_count": len(observed),
        "nonfinite_gradient_count": nonfinite_count,
        "checks": checks,
        "pilot_metric_source": csv_source,
        "profile_metric_source": profile_source,
        "clipping_sources": clip_sources,
        "threshold_sources": threshold_sources,
        "same_budget_outperforms_two_deterministic_policies": performance_pass,
        "performance_gate_source": performance_source,
        "note": (
            "Gradient stability is evaluated against the documented raw threshold of 100.0 "
            "and clipping threshold of 1.0. The final GO decision is not forced."
        ),
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "Section 14.11.2 Gradient Stability",
        "==================================",
        f"Generated UTC:                           {payload['generated_at_utc']}",
        f"Raw-gradient limit:                      {RAW_LIMIT:.6f}",
        f"Gradient clipping norm:                  {CLIP_NORM:.6f}",
        f"Maximum observed gradient norm:          {maximum if maximum is not None else 'NOT FOUND'}",
        f"Observed gradient measurements:          {len(observed)}",
        f"Non-finite measurements:                 {nonfinite_count}",
        "",
    ]
    for name, passed in checks.items():
        lines.append(f"{name:43s} {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            f"Gradient stability:                      {'PASS' if gradient_pass else 'FAIL'}",
            "Same-budget two-policy outperformance:  "
            + (
                "PASS"
                if performance_pass is True
                else "FAIL"
                if performance_pass is False
                else "NOT VERIFIED"
            ),
            f"Section 14.11:                            {'COMPLETE' if section_complete else 'NOT COMPLETE'}",
            "",
            f"Evidence JSON: {RESULT_JSON.relative_to(ROOT)}",
        ]
    )
    RESULT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if gradient_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
PY

chmod +x scripts/section14_11_2_gradient_gate.py

set +e
"$PYTHON_BIN" scripts/section14_11_2_gradient_gate.py "$RAW_GRAD_LIMIT" "$CLIP_NORM"
GATE_RC=$?
set -e

heading "7. REGENERATE AN EXISTING FINAL DECISION, IF PRESENT"

FINAL_SCRIPT=""
while IFS= read -r candidate; do
    [[ "$candidate" == "scripts/section14_11_2_gradient_gate.py" ]] && continue
    FINAL_SCRIPT="$candidate"
    break
done < <(
    find scripts -maxdepth 3 -type f \
        \( -iname '*14*11*final*.py' -o -iname '*final*go*decision*.py' -o -iname '*go*no*go*.py' \) \
        | sort
)

if [[ -n "$FINAL_SCRIPT" ]]; then
    echo "Final-decision builder: $FINAL_SCRIPT"
    set +e
    "$PYTHON_BIN" "$FINAL_SCRIPT" \
        2>&1 | tee "$LOG_DIR/section14_11_2_final_decision.log"
    FINAL_RC=${PIPESTATUS[0]}
    set -e
    echo "Final-decision builder return code: $FINAL_RC"

    set +e
    "$PYTHON_BIN" scripts/section14_11_2_gradient_gate.py "$RAW_GRAD_LIMIT" "$CLIP_NORM"
    GATE_RC=$?
    set -e
else
    echo "No separate final-decision builder was found."
fi

heading "8. FINAL RESULT"
cat "$RESULT_TXT"
echo "Main log:  $MAIN_LOG"
echo "Backup:    $BACKUP_DIR"
echo "Git diff:"
git status --short

if [[ "$GATE_RC" -ne 0 ]]; then
    echo
    echo "SECTION 14.11.2 FAILED. Review the failed check in the evidence report."
    exit "$GATE_RC"
fi

echo
echo "SECTION 14.11.2 COMPLETED: GRADIENT STABILITY PASS"
echo "The final Section 14.11 GO decision remains controlled by the two-policy performance criterion."
