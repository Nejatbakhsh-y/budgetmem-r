#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# ==============================================================================
# BudgetMem-R — Section 14.11.2
# Stabilize the actual training run, rerun the tuned Section 15 pilot, and
# re-evaluate the documented raw-gradient gate without changing its threshold.
#
# Current documented requirements preserved by this automation:
#   maximum raw gradient norm <= 100.0
#   gradient clipping norm     = 1.0
#
# Run from the VS Code WSL terminal:
#   bash 14_11_2_stabilize_gradient_and_rerun.sh
#
# Optional overrides:
#   PROJECT_ROOT="/path/to/repo" bash 14_11_2_stabilize_gradient_and_rerun.sh
#   PILOT_RUNNER="18_run_tuned_pilot_and_final_go_decision.sh" bash ...
#   PILOT_CONFIG="configs/.../pilot.yaml" bash ...
# ==============================================================================

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/c/Users/nejat/OneDrive/Desktop/UN/Skills/GitHub 2026/budgetmem-r}"
RAW_GRAD_LIMIT="${RAW_GRAD_LIMIT:-100.0}"
CLIP_NORM="${CLIP_NORM:-1.0}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
LR_FACTOR="${LR_FACTOR:-0.70}"
MIN_LR="${MIN_LR:-0.00001}"
MIN_WARMUP_STEPS="${MIN_WARMUP_STEPS:-100}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SELF_NAME="$(basename "$0")"

cd "$PROJECT_ROOT"

EVIDENCE_DIR="reports/evidence"
LOG_DIR="$EVIDENCE_DIR/logs"
BACKUP_DIR="$EVIDENCE_DIR/backups/section14_11_2_stabilize_$STAMP"
MAIN_LOG="$LOG_DIR/section14_11_2_stabilize_$STAMP.log"
SUMMARY_JSON="$EVIDENCE_DIR/section14_11_2_stabilization_summary.json"
SUMMARY_TXT="$EVIDENCE_DIR/section14_11_2_stabilization_summary.txt"
PATCH_HISTORY="$EVIDENCE_DIR/section14_11_2_lr_patch_history.jsonl"

mkdir -p "$LOG_DIR" "$BACKUP_DIR/configs" "$BACKUP_DIR/stale_gradient_evidence"
exec > >(tee -a "$MAIN_LOG") 2>&1

fail() {
    local message="$1"
    local code="${2:-1}"
    echo
    echo "ERROR: $message"
    echo "Main log: $MAIN_LOG"
    exit "$code"
}

heading() {
    echo
    echo "==============================================================================="
    echo "$1"
    echo "==============================================================================="
}

heading "SECTION 14.11.2 — STABILIZE AND RERUN"
echo "Project root:            $PROJECT_ROOT"
echo "Raw-gradient limit:      $RAW_GRAD_LIMIT"
echo "Clipping norm:           $CLIP_NORM"
echo "Maximum attempts:        $MAX_ATTEMPTS"
echo "Per-attempt LR factor:   $LR_FACTOR"
echo "Minimum LR:              $MIN_LR"
echo "Minimum warmup steps:    $MIN_WARMUP_STEPS"
echo "Started UTC:             $(date -u +%Y-%m-%dT%H:%M:%SZ)"

[[ -d src && -d configs ]] || fail "The BudgetMem-R repository was not found at the configured path." 2

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
[[ -n "$PYTHON_BIN" ]] || fail "Python is unavailable in the active environment." 2

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

"$PYTHON_BIN" - <<'PY' || exit 2
try:
    import yaml  # noqa: F401
except Exception as exc:
    raise SystemExit(f"PyYAML is required but unavailable: {exc}")
PY

echo "Python:                  $($PYTHON_BIN --version 2>&1)"
echo "Git branch:              $(git branch --show-current 2>/dev/null || echo unknown)"

heading "1. CONFIRM THE CURRENT FAILURE AND IDENTIFY ITS SOURCE"

"$PYTHON_BIN" - "$RAW_GRAD_LIMIT" "$EVIDENCE_DIR" <<'PY'
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

limit = float(sys.argv[1])
evidence = Path(sys.argv[2])
root = Path.cwd()
records: list[dict[str, Any]] = []

csv_path = root / "reports" / "tables" / "pilot_results.csv"
if csv_path.exists():
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
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
        if grad_key:
            for row_number, row in enumerate(reader, start=2):
                try:
                    value = float(row.get(grad_key, ""))
                except (TypeError, ValueError):
                    continue
                records.append(
                    {
                        "source": str(csv_path.relative_to(root)),
                        "location": f"row {row_number}",
                        "value": value,
                        "context": row,
                    }
                )

wanted_keys = {
    "raw_gradient_norm",
    "maximum_gradient_norm",
    "max_gradient_norm",
    "max_raw_gradient_norm",
}


def visit(obj: Any, source: Path, pointer: str = "$") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{pointer}.{key}"
            if str(key).lower() in wanted_keys and isinstance(value, (int, float)):
                records.append(
                    {
                        "source": str(source.relative_to(root)),
                        "location": child,
                        "value": float(value),
                        "context": None,
                    }
                )
            visit(value, source, child)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            visit(value, source, f"{pointer}[{index}]")

for path in sorted((root / "reports" / "evidence").glob("*gradient*.json")):
    if path.name.startswith("section14_11_2_"):
        continue
    try:
        visit(json.loads(path.read_text(encoding="utf-8")), path)
    except Exception:
        pass

finite = [item for item in records if math.isfinite(float(item["value"]))]
maximum = max(finite, key=lambda item: float(item["value"])) if finite else None
output = {
    "raw_gradient_limit": limit,
    "measurement_count": len(records),
    "maximum": maximum,
    "currently_passes": bool(maximum and float(maximum["value"]) <= limit),
}
(evidence / "section14_11_2_failure_source.json").write_text(
    json.dumps(output, indent=2) + "\n", encoding="utf-8"
)

print(f"Measurements found:      {len(records)}")
if maximum:
    print(f"Maximum raw gradient:    {maximum['value']}")
    print(f"Maximum source:          {maximum['source']}")
    print(f"Maximum location:        {maximum['location']}")
    context = maximum.get("context")
    if isinstance(context, dict):
        for key in (
            "task",
            "model",
            "model_name",
            "sequence_length",
            "budget",
            "seed",
            "run_id",
        ):
            if context.get(key) not in (None, ""):
                print(f"{key:24s}{context[key]}")
else:
    print("Maximum raw gradient:    NOT FOUND")
PY

heading "2. LOCATE THE TUNED PILOT RUNNER"

if [[ -n "${PILOT_RUNNER:-}" ]]; then
    RUNNER="$PILOT_RUNNER"
else
    RUNNER="$($PYTHON_BIN - "$SELF_NAME" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

self_name = sys.argv[1]
root = Path.cwd()
exact = [
    root / "18_run_tuned_pilot_and_final_go_decision.sh",
    root / "18_run_tuned_pilot_and_final_go.sh",
    root / "run_tuned_pilot_and_final_go_decision.sh",
    root / "scripts" / "run_tuned_pilot_and_final_go_decision.sh",
    root / "run_section15_pilot.sh",
    root / "scripts" / "run_section15_pilot.sh",
]
for path in exact:
    if path.is_file():
        print(path.relative_to(root))
        raise SystemExit(0)

ranked: list[tuple[int, Path]] = []
for path in root.glob("*.sh"):
    if path.name == self_name or path.name.startswith("14_11_2_"):
        continue
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    name = path.name.lower()
    score = 0
    if "pilot" in name:
        score += 40
    if "tuned" in name or "improved" in name:
        score += 30
    if "run" in name or re.match(r"^18_", name):
        score += 20
    if "pilot_results.csv" in text:
        score += 30
    if "section15" in text or "section 15" in text:
        score += 15
    if "final_go" in name or "go_decision" in name:
        score += 5
    if "diagnos" in name or "debug" in name or "bundle" in name:
        score -= 50
    if score > 0:
        ranked.append((score, path))

if ranked:
    ranked.sort(key=lambda pair: (pair[0], pair[1].stat().st_mtime), reverse=True)
    print(ranked[0][1].relative_to(root))
PY
)"
fi

[[ -n "${RUNNER:-}" && -f "$RUNNER" ]] || fail \
    "No tuned Section 15 pilot runner was found. Set PILOT_RUNNER to its relative path." 3

bash -n "$RUNNER" || fail "The selected pilot runner has invalid Bash syntax: $RUNNER" 3
echo "Selected pilot runner:  $RUNNER"

heading "3. LOCATE EVERY PILOT CONFIG USED BY THE RUNNER"

CONFIG_LIST_FILE="$BACKUP_DIR/config_list.txt"

"$PYTHON_BIN" - "$RUNNER" "${PILOT_CONFIG:-}" > "$CONFIG_LIST_FILE" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

runner = Path(sys.argv[1])
override = sys.argv[2].strip()
root = Path.cwd()
found: list[Path] = []


def add(path: Path) -> None:
    try:
        path = path.resolve()
        path.relative_to(root.resolve())
    except Exception:
        return
    if path.is_file() and path.suffix.lower() in {".yaml", ".yml"} and path not in found:
        found.append(path)

if override:
    add(root / override)

for candidate in (
    root / "configs" / "experiments" / "pilot.yaml",
    root / "configs" / "pilot.yaml",
    root / "configs" / "section15" / "pilot.yaml",
    root / "configs" / "experiments" / "section15_pilot.yaml",
    root / "configs" / "section15_pilot.yaml",
):
    add(candidate)

text = runner.read_text(encoding="utf-8", errors="replace")
for match in re.findall(r"(?:configs/)?[A-Za-z0-9_./-]+\.ya?ml", text):
    candidate = root / match
    if candidate.is_file():
        body = candidate.read_text(encoding="utf-8", errors="replace").lower()
        if "pilot" in candidate.name.lower() or "budgetmem" in body or "section15" in body:
            add(candidate)

if not found:
    candidates = sorted((root / "configs").rglob("*.yaml")) + sorted(
        (root / "configs").rglob("*.yml")
    )
    scored: list[tuple[int, Path]] = []
    for path in candidates:
        body = path.read_text(encoding="utf-8", errors="replace").lower()
        score = 0
        if "pilot" in path.name.lower():
            score += 40
        if "budgetmem" in body:
            score += 30
        if "gradient_clip_norm" in body:
            score += 10
        if "sequence_lengths" in body or "sequence_length" in body:
            score += 10
        if score:
            scored.append((score, path))
    if scored:
        scored.sort(key=lambda pair: pair[0], reverse=True)
        add(scored[0][1])

for path in found:
    print(path.relative_to(root))
PY

[[ -s "$CONFIG_LIST_FILE" ]] || fail "No pilot YAML configuration could be located." 4

echo "Pilot configuration files:"
cat "$CONFIG_LIST_FILE"

while IFS= read -r config; do
    [[ -n "$config" ]] || continue
    mkdir -p "$BACKUP_DIR/configs/$(dirname "$config")"
    cp -p "$config" "$BACKUP_DIR/configs/$config"
done < "$CONFIG_LIST_FILE"

heading "4. ARCHIVE STALE GRADIENT-PROFILE JSON"

# The gate combines official pilot measurements with the newest auxiliary
# gradient-profile JSON. Keeping an old failed trace active would make a fresh
# stable pilot fail for stale evidence. Files are archived, not discarded.
while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    base="$(basename "$path")"
    case "$base" in
        section14_11_2_gradient_stability.json|section14_11_2_test_status.json|section14_11_2_failure_source.json)
            continue
            ;;
    esac
    cp -p "$path" "$BACKUP_DIR/stale_gradient_evidence/$base"
    rm -f "$path"
    echo "Archived stale profile: $path"
done < <(find reports/evidence -maxdepth 1 -type f -iname '*gradient*.json' | sort)

heading "5. ITERATIVELY LOWER BUDGETMEM-R LEARNING RATES AND RERUN"

attempt=0
passed=0
last_max=""
last_gate_rc=99

while (( attempt < MAX_ATTEMPTS )); do
    attempt=$((attempt + 1))
    echo
    echo "------------------------------ ATTEMPT $attempt/$MAX_ATTEMPTS ------------------------------"

    "$PYTHON_BIN" - \
        "$attempt" "$LR_FACTOR" "$MIN_LR" "$MIN_WARMUP_STEPS" \
        "$RAW_GRAD_LIMIT" "$CLIP_NORM" "$CONFIG_LIST_FILE" "$PATCH_HISTORY" <<'PY'
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

attempt = int(sys.argv[1])
factor = float(sys.argv[2])
min_lr = float(sys.argv[3])
min_warmup = int(sys.argv[4])
raw_limit = float(sys.argv[5])
clip_norm = float(sys.argv[6])
config_list = Path(sys.argv[7])
history_path = Path(sys.argv[8])

lr_keys = {
    "learning_rate",
    "lr",
    "controller_learning_rate",
    "controller_lr",
    "write_controller_lr",
    "eviction_controller_lr",
    "memory_controller_lr",
    "memory_lr",
    "policy_lr",
    "gate_lr",
}
warmup_keys = {"warmup_steps", "lr_warmup_steps", "scheduler_warmup_steps"}
identity_keys = {"name", "model", "model_name", "type", "policy", "architecture"}


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def budget_identity(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    for key, value in node.items():
        if str(key).lower() in identity_keys:
            text = str(value).lower().replace("-", "").replace("_", "")
            if "budgetmem" in text:
                return True
    return False


def walk(node: Any, path: tuple[str, ...], inherited_budget: bool, changes: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        path_text = ".".join(path).lower().replace("-", "").replace("_", "")
        local_budget = inherited_budget or "budgetmem" in path_text or budget_identity(node)

        for key in list(node.keys()):
            key_text = str(key).lower()
            value = node[key]
            full_path = ".".join((*path, str(key)))

            if key_text == "maximum_acceptable_gradient_norm":
                old = node[key]
                node[key] = raw_limit
                if old != node[key]:
                    changes.append({"path": full_path, "old": old, "new": node[key]})
                continue

            if key_text == "gradient_clip_norm":
                old = node[key]
                node[key] = clip_norm
                if old != node[key]:
                    changes.append({"path": full_path, "old": old, "new": node[key]})
                continue

            if local_budget and key_text in lr_keys:
                old_num = numeric(value)
                if old_num is not None and old_num > 0:
                    new_num = max(min_lr, old_num * factor)
                    if isinstance(value, int) and not isinstance(value, bool):
                        node[key] = new_num
                    else:
                        node[key] = float(f"{new_num:.12g}")
                    if not math.isclose(old_num, float(node[key]), rel_tol=0.0, abs_tol=1e-15):
                        changes.append({"path": full_path, "old": value, "new": node[key]})
                    continue

            if local_budget and key_text in warmup_keys:
                old_num = numeric(value)
                if old_num is not None:
                    new_value = max(min_warmup, int(old_num * 2))
                    node[key] = new_value
                    if value != new_value:
                        changes.append({"path": full_path, "old": value, "new": new_value})
                    continue

            walk(value, (*path, str(key)), local_budget, changes)

    elif isinstance(node, list):
        for index, value in enumerate(node):
            walk(value, (*path, str(index)), inherited_budget, changes)


for line in config_list.read_text(encoding="utf-8").splitlines():
    path = Path(line.strip())
    if not path.is_file():
        continue
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if data is None:
        continue

    changes: list[dict[str, Any]] = []
    walk(data, (), False, changes)

    # Fallback: a pilot file sometimes defines one optimizer shared by all models.
    # Patch it only when no BudgetMem-R-specific LR key was discoverable.
    if not any(str(item["path"]).lower().endswith(tuple(lr_keys)) for item in changes):
        def patch_single_global_lr(node: Any, path_parts: tuple[str, ...] = ()) -> bool:
            if isinstance(node, dict):
                for key, value in node.items():
                    key_text = str(key).lower()
                    parent_text = ".".join(path_parts).lower()
                    if key_text in {"learning_rate", "lr"} and any(
                        token in parent_text for token in ("training", "optimizer", "pilot")
                    ):
                        old_num = numeric(value)
                        if old_num is not None and old_num > 0:
                            new_num = max(min_lr, old_num * factor)
                            node[key] = float(f"{new_num:.12g}")
                            changes.append(
                                {
                                    "path": ".".join((*path_parts, str(key))),
                                    "old": value,
                                    "new": node[key],
                                    "scope": "shared_optimizer_fallback",
                                }
                            )
                            return True
                    if patch_single_global_lr(value, (*path_parts, str(key))):
                        return True
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    if patch_single_global_lr(value, (*path_parts, str(index))):
                        return True
            return False

        patch_single_global_lr(data)

    if not changes:
        raise SystemExit(
            f"No stabilizing field could be changed in {path}. "
            "Set PILOT_CONFIG to the actual tuned pilot YAML."
        )

    path.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
        newline="\n",
    )
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": attempt,
        "config": str(path),
        "changes": changes,
    }
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")

    print(f"Patched {path}:")
    for item in changes:
        print(f"  {item['path']}: {item['old']} -> {item['new']}")
PY

    before_hash=""
    if [[ -f reports/tables/pilot_results.csv ]]; then
        before_hash="$(sha256sum reports/tables/pilot_results.csv | awk '{print $1}')"
    fi

    ATTEMPT_LOG="$LOG_DIR/section14_11_2_pilot_attempt_${attempt}_$STAMP.log"
    echo "Running: bash $RUNNER"
    set +e
    bash "$RUNNER" 2>&1 | tee "$ATTEMPT_LOG"
    runner_rc=${PIPESTATUS[0]}
    set -e

    if [[ "$runner_rc" -ne 0 ]]; then
        echo "Pilot runner return code: $runner_rc"
        echo "Attempt $attempt did not complete; continuing only if attempts remain."
        continue
    fi

    [[ -f reports/tables/pilot_results.csv ]] || {
        echo "The runner completed but reports/tables/pilot_results.csv was not produced."
        continue
    }

    after_hash="$(sha256sum reports/tables/pilot_results.csv | awk '{print $1}')"
    if [[ -n "$before_hash" && "$before_hash" == "$after_hash" ]]; then
        echo "Warning: pilot_results.csv hash did not change. The runner may have reused cached output."
    else
        echo "pilot_results.csv was regenerated."
    fi

    # Re-run the established 14.11.2 automation so tests, profiling, and the
    # official evidence gate are rebuilt from the new pilot.
    if [[ -f 14_11_2_fix_gradient_stability.sh ]]; then
        echo "Rebuilding official 14.11.2 evidence..."
        set +e
        bash 14_11_2_fix_gradient_stability.sh 2>&1 | tee \
            "$LOG_DIR/section14_11_2_gate_attempt_${attempt}_$STAMP.log"
        last_gate_rc=${PIPESTATUS[0]}
        set -e
    elif [[ -f scripts/section14_11_2_gradient_gate.py ]]; then
        set +e
        "$PYTHON_BIN" scripts/section14_11_2_gradient_gate.py \
            "$RAW_GRAD_LIMIT" "$CLIP_NORM" 2>&1 | tee \
            "$LOG_DIR/section14_11_2_gate_attempt_${attempt}_$STAMP.log"
        last_gate_rc=${PIPESTATUS[0]}
        set -e
    else
        fail "No Section 14.11.2 evidence gate is available after the pilot rerun." 5
    fi

    readarray -t gate_values < <("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

path = Path("reports/evidence/section14_11_2_gradient_stability.json")
if not path.exists():
    print("")
    print("FAIL")
else:
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get("maximum_observed_gradient_norm")
    print("" if value is None else value)
    print(data.get("gradient_stability", "FAIL"))
PY
)

    last_max="${gate_values[0]:-}"
    gate_status="${gate_values[1]:-FAIL}"
    echo "Attempt $attempt maximum raw gradient: ${last_max:-NOT FOUND}"
    echo "Attempt $attempt gradient gate:        $gate_status"

    if [[ "$gate_status" == "PASS" && "$last_gate_rc" -eq 0 ]]; then
        passed=1
        break
    fi

done

heading "6. WRITE FINAL STABILIZATION SUMMARY"

"$PYTHON_BIN" - \
    "$SUMMARY_JSON" "$SUMMARY_TXT" "$passed" "$attempt" "$last_max" \
    "$RAW_GRAD_LIMIT" "$CLIP_NORM" "$RUNNER" "$PATCH_HISTORY" "$MAIN_LOG" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    summary_json,
    summary_txt,
    passed_raw,
    attempts_raw,
    last_max_raw,
    raw_limit_raw,
    clip_norm_raw,
    runner,
    patch_history,
    main_log,
) = sys.argv[1:]

passed = passed_raw == "1"
attempts = int(attempts_raw)
last_max = float(last_max_raw) if last_max_raw.strip() else None
payload = {
    "section": "14.11.2",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "gradient_stability": "PASS" if passed else "FAIL",
    "attempts_used": attempts,
    "maximum_observed_gradient_norm": last_max,
    "raw_gradient_limit": float(raw_limit_raw),
    "gradient_clip_norm": float(clip_norm_raw),
    "pilot_runner": runner,
    "patch_history": patch_history,
    "main_log": main_log,
    "threshold_was_changed": False,
}
Path(summary_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
lines = [
    "Section 14.11.2 Stabilization Summary",
    "======================================",
    f"Generated UTC:                  {payload['generated_at_utc']}",
    f"Pilot runner:                   {runner}",
    f"Attempts used:                  {attempts}",
    f"Raw-gradient limit:             {float(raw_limit_raw):.6f}",
    f"Gradient clipping norm:         {float(clip_norm_raw):.6f}",
    f"Maximum observed gradient:      {last_max if last_max is not None else 'NOT FOUND'}",
    "Threshold changed:              NO",
    f"Gradient stability:             {'PASS' if passed else 'FAIL'}",
    f"Patch history:                  {patch_history}",
    f"Main log:                       {main_log}",
]
Path(summary_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

heading "7. FINAL RESULT"

if [[ "$passed" -eq 1 ]]; then
    cat reports/evidence/section14_11_2_gradient_stability.txt
    echo
    echo "SECTION 14.11.2 COMPLETED: GRADIENT STABILITY PASS"
    echo "The documented 100.0 threshold was preserved."
    echo "Do not infer that the full Section 14.11 is complete unless the separate"
    echo "same-budget two-policy outperformance gate also reports PASS."
    echo "Backup directory: $BACKUP_DIR"
    echo "Main log:        $MAIN_LOG"
    exit 0
fi

echo "SECTION 14.11.2 REMAINS FAILED AFTER $attempt ATTEMPT(S)."
echo "Last maximum raw gradient: ${last_max:-NOT FOUND}"
echo "The automation did not weaken or bypass the 100.0 acceptance threshold."
echo "Review: $SUMMARY_TXT"
echo "Main log: $MAIN_LOG"
exit 1
