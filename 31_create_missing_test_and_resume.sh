#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
TEST_FILE="tests/pilot/test_detached_memory_adapter.py"
RESUME_SCRIPT="29_fix_adapter_scope_and_resume.sh"
CONFIG_FILE="configs/experiments/pilot_assoc_detached_memory.yaml"

echo "============================================================"
echo " Create Missing Adapter Test and Resume"
echo "============================================================"
echo "Repository: $ROOT"
echo

for required in \
    "$PYTHON" \
    "$RESUME_SCRIPT" \
    "$CONFIG_FILE" \
    "src/budgetmem/experiments/pilot.py"; do

    if [[ ! -e "$required" ]]; then
        echo "ERROR: Missing required path:"
        echo "  $required"
        exit 1
    fi
done

export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p tests/pilot

cat > "$TEST_FILE" <<'PY'
"""Regression test for detached memory writes in the pilot adapter."""

from __future__ import annotations

from pathlib import Path

import yaml

from budgetmem.experiments.pilot import BudgetMemRAdapter


def test_pilot_adapter_passes_detach_memory_writes() -> None:
    config_path = Path(
        "configs/experiments/"
        "pilot_assoc_detached_memory.yaml"
    )

    assert config_path.exists()

    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    adapter = BudgetMemRAdapter(config)

    assert hasattr(adapter, "_model_cfg")
    assert adapter.core.detach_memory_writes is True
PY

echo "Created:"
echo "  $TEST_FILE"
echo

echo "Checking Python syntax."

"$PYTHON" -m py_compile \
    "$TEST_FILE" \
    src/budgetmem/experiments/pilot.py

echo
echo "Running the missing regression test."

"$PYTHON" -m pytest \
    "$TEST_FILE" \
    -q

echo
echo "Regression test: PASS"

echo
echo "Verifying the constructor directly."

"$PYTHON" - <<'PY'
from pathlib import Path

import yaml

from budgetmem.experiments.pilot import BudgetMemRAdapter


config_path = Path(
    "configs/experiments/"
    "pilot_assoc_detached_memory.yaml"
)

config = yaml.safe_load(
    config_path.read_text(encoding="utf-8")
)

adapter = BudgetMemRAdapter(config)

print(
    "Adapter _model_cfg exists: "
    f"{hasattr(adapter, '_model_cfg')}"
)

print(
    "Core detach_memory_writes: "
    f"{adapter.core.detach_memory_writes}"
)

if not hasattr(adapter, "_model_cfg"):
    raise SystemExit(
        "ERROR: _model_cfg is still missing."
    )

if adapter.core.detach_memory_writes is not True:
    raise SystemExit(
        "ERROR: detach_memory_writes is not enabled."
    )

print("Constructor verification: PASS")
PY

echo
echo "============================================================"
echo " Resuming the detached-memory targeted screen"
echo "============================================================"
echo
echo "Do not interrupt after training begins."
echo

bash "$RESUME_SCRIPT"
