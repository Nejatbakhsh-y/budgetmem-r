#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
PILOT_FILE="src/budgetmem/experiments/pilot.py"
RESUME_SCRIPT="29_fix_adapter_scope_and_resume.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP=".section15_backup/model_cfg_fix_${STAMP}"

echo "============================================================"
echo " Repair BudgetMemRAdapter _model_cfg and Resume"
echo "============================================================"

for required in \
    "$PYTHON" \
    "$PILOT_FILE" \
    "$RESUME_SCRIPT" \
    "configs/experiments/pilot_assoc_detached_memory.yaml"; do

    if [[ ! -e "$required" ]]; then
        echo "ERROR: Missing required path:"
        echo "  $required"
        exit 1
    fi
done

mkdir -p "$BACKUP"
cp -f "$PILOT_FILE" "$BACKUP/pilot.py"

echo "Backup:"
echo "  $BACKUP/pilot.py"
echo

"$PYTHON" - <<'PY'
from pathlib import Path


path = Path("src/budgetmem/experiments/pilot.py")
text = path.read_text(encoding="utf-8")

class_marker = "class BudgetMemRAdapter(nn.Module):"
constructor_marker = "    def __init__(self, cfg: Mapping[str, Any]) -> None:"
kwargs_marker = (
    "    def _constructor_kwargs("
    "self, budgetmem_class: type[nn.Module]"
    ") -> dict[str, Any]:"
)
next_method_marker = "    @staticmethod"

class_start = text.find(class_marker)

if class_start == -1:
    raise SystemExit(
        "ERROR: BudgetMemRAdapter class was not found."
    )

constructor_start = text.find(
    constructor_marker,
    class_start,
)

kwargs_start = text.find(
    kwargs_marker,
    constructor_start,
)

if constructor_start == -1 or kwargs_start == -1:
    raise SystemExit(
        "ERROR: BudgetMemRAdapter methods were not found."
    )

constructor_segment = text[
    constructor_start:kwargs_start
]

model_cfg_line = '        model_cfg = cfg["model"]\n'
stored_cfg_line = "        self._model_cfg = model_cfg\n"

if model_cfg_line not in constructor_segment:
    raise SystemExit(
        "ERROR: model_cfg initialization was not found "
        "inside BudgetMemRAdapter.__init__()."
    )

if stored_cfg_line not in constructor_segment:
    absolute_position = (
        constructor_start
        + constructor_segment.index(model_cfg_line)
        + len(model_cfg_line)
    )

    text = (
        text[:absolute_position]
        + stored_cfg_line
        + text[absolute_position:]
    )

    print(
        "Inserted self._model_cfg inside "
        "BudgetMemRAdapter.__init__()."
    )
else:
    print(
        "self._model_cfg is already stored inside "
        "BudgetMemRAdapter.__init__()."
    )


# Recalculate method locations after the insertion.
kwargs_start = text.find(
    kwargs_marker,
    class_start,
)

next_method_start = text.find(
    next_method_marker,
    kwargs_start,
)

if next_method_start == -1:
    raise SystemExit(
        "ERROR: Could not determine the end of "
        "_constructor_kwargs()."
    )

kwargs_segment = text[
    kwargs_start:next_method_start
]

scope_line = "        model_cfg = self._model_cfg\n"

if scope_line not in kwargs_segment:
    method_header_end = text.find(
        "\n",
        kwargs_start,
    ) + 1

    text = (
        text[:method_header_end]
        + scope_line
        + text[method_header_end:]
    )

    print(
        "Inserted model_cfg = self._model_cfg inside "
        "_constructor_kwargs()."
    )
else:
    print(
        "_constructor_kwargs() already reads "
        "self._model_cfg."
    )


path.write_text(
    text,
    encoding="utf-8",
)

print(f"Repaired: {path}")
PY

echo
echo "Checking Python syntax."

"$PYTHON" -m py_compile "$PILOT_FILE"

echo
echo "Showing the repaired adapter section."

grep -n -A30 -B3 \
    "class BudgetMemRAdapter" \
    "$PILOT_FILE" \
    | head -40

echo
echo "Verifying constructor behavior."

export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" - <<'PY'
from pathlib import Path

import yaml

from budgetmem.experiments.pilot import BudgetMemRAdapter


config_path = Path(
    "configs/experiments/"
    "pilot_assoc_detached_memory.yaml"
)

cfg = yaml.safe_load(
    config_path.read_text(encoding="utf-8")
)

adapter = BudgetMemRAdapter(cfg)

if not hasattr(adapter, "_model_cfg"):
    raise SystemExit(
        "ERROR: Adapter still has no _model_cfg attribute."
    )

if adapter._model_cfg is not cfg["model"]:
    print(
        "Note: _model_cfg is an equivalent stored mapping."
    )

actual = bool(
    adapter.core.detach_memory_writes
)

print("Adapter _model_cfg exists: True")
print(
    "Core detach_memory_writes: "
    f"{actual}"
)

if actual is not True:
    raise SystemExit(
        "ERROR: detach_memory_writes did not reach "
        "the BudgetMem-R core."
    )

print("Constructor verification: PASS")
PY

echo
echo "Running focused adapter regression test."

"$PYTHON" -m pytest \
    tests/pilot/test_detached_memory_adapter.py \
    -q

echo
echo "============================================================"
echo " Scope repair passed"
echo " Resuming the detached-memory targeted screen"
echo "============================================================"
echo

bash "$RESUME_SCRIPT"
