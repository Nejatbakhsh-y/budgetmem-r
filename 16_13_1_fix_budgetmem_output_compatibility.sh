#!/usr/bin/env bash
# Repair Section 13 BudgetMem-R output compatibility.
# Run from the budgetmem-r repository root in the VS Code WSL terminal.

set -Eeuo pipefail
IFS=$'\n\t'

readonly TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly MODEL_FILE="src/budgetmem/models/budgetmem_r.py"
readonly AUTOMATION_FILE="16_13_1_implement_budgetmem_r.sh"
readonly LOG_DIR="reports/evidence/logs"
readonly BACKUP_DIR="reports/evidence/backups/section13_output_fix_${TIMESTAMP}"
readonly LOG_FILE="${LOG_DIR}/section13_output_fix_${TIMESTAMP}.log"
readonly EVIDENCE_FILE="reports/evidence/section13_budgetmem_output_compatibility.txt"

if git rev-parse --show-toplevel >/dev/null 2>&1; then
    ROOT_DIR="$(git rev-parse --show-toplevel)"
else
    ROOT_DIR="$(pwd)"
fi
cd "$ROOT_DIR"

mkdir -p "$LOG_DIR" "$BACKUP_DIR" "$(dirname "$EVIDENCE_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
    local exit_code=$?
    local line_number=${1:-unknown}
    echo
    echo "ERROR: Section 13 compatibility repair failed at line ${line_number} with exit code ${exit_code}."
    echo "Review: ${LOG_FILE}"
    exit "$exit_code"
}
trap 'on_error $LINENO' ERR

printf '%s\n' "============================================================"
printf '%s\n' "16.13.1 — Repair BudgetMemROutput Compatibility"
printf '%s\n' "Repository: $ROOT_DIR"
printf '%s\n' "UTC run:    $TIMESTAMP"
printf '%s\n' "============================================================"

if [[ ! -f "$MODEL_FILE" ]]; then
    echo "ERROR: $MODEL_FILE was not found."
    exit 2
fi

PYTHON_BIN=""
for candidate in ".venv/bin/python" "venv/bin/python" "python3" "python"; do
    if [[ -x "$candidate" ]] || command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done
if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: Python was not found."
    exit 2
fi

echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

# Prevent CPU thread oversubscription during the verification models.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

backup_file() {
    local source="$1"
    [[ -f "$source" ]] || return 0
    mkdir -p "$BACKUP_DIR/$(dirname "$source")"
    cp -p "$source" "$BACKUP_DIR/$source"
    echo "BACKUP     $source -> $BACKUP_DIR/$source"
}

backup_file "$MODEL_FILE"
backup_file "$AUTOMATION_FILE"
backup_file "src/budgetmem/models/__init__.py"

patch_python_source() {
    local target="$1"
    "$PYTHON_BIN" - "$target" <<'PY'
from __future__ import annotations

import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
original = text

class_block = '''\n\nclass BudgetMemROutput(dict[str, object]):\n    """Dictionary-compatible output with attribute access for project adapters."""\n\n    def __init__(self, **values: object) -> None:\n        super().__init__(values)\n\n    def __getattr__(self, name: str) -> object:\n        try:\n            return self[name]\n        except KeyError as exc:\n            raise AttributeError(name) from exc\n'''

if "class BudgetMemROutput(" not in text:
    marker = "\n\nclass BudgetMemR(nn.Module):"
    if marker not in text:
        raise SystemExit(f"Could not locate BudgetMemR class marker in {path}")
    text = text.replace(marker, class_block + marker, 1)

text = text.replace(
    ") -> dict[str, Tensor | BudgetMemoryState]:\n        if inputs.ndim != 3:",
    ") -> BudgetMemROutput:\n        if inputs.ndim != 3:",
    1,
)

old_return = '''        return {\n            "logits": stacked_logits[:, -1],\n            "sequence_logits": stacked_logits,\n            "auxiliary_predictions": torch.stack(auxiliary_predictions, dim=1),\n            "write_probabilities": torch.stack(write_probabilities, dim=1),\n            "write_gates": torch.stack(write_gates, dim=1),\n            "memory_sizes": stacked_sizes,\n            "budgets": budgets,\n            "budget_violations": budget_violations,\n            "retrieval_weights": torch.stack(retrieval_weights, dim=1),\n            "final_memory": state,\n        }'''

new_return = '''        stacked_write_probabilities = torch.stack(write_probabilities, dim=1)\n        stacked_write_gates = torch.stack(write_gates, dim=1)\n        return BudgetMemROutput(\n            logits=stacked_logits[:, -1],\n            sequence_logits=stacked_logits,\n            auxiliary_predictions=torch.stack(auxiliary_predictions, dim=1),\n            write_probabilities=stacked_write_probabilities,\n            controller_probabilities=stacked_write_probabilities,\n            write_gates=stacked_write_gates,\n            hard_writes=stacked_write_gates >= self.write_threshold,\n            memory_sizes=stacked_sizes,\n            memory_trace=stacked_sizes,\n            budgets=budgets,\n            budget_violations=budget_violations,\n            retrieval_weights=torch.stack(retrieval_weights, dim=1),\n            final_state=hidden,\n            final_memory=state,\n            memory_mask=state.valid,\n        )'''

if old_return in text:
    text = text.replace(old_return, new_return, 1)
elif "return BudgetMemROutput(" not in text:
    raise SystemExit(f"Could not locate the BudgetMem-R return block in {path}")

if text != original:
    path.write_text(text, encoding="utf-8")
    print(f"PATCHED    {path}")
else:
    print(f"UNCHANGED  {path}")
PY
}

patch_python_source "$MODEL_FILE"

# Patch the generator as well, so rerunning the original Section 13 automation
# does not recreate the incompatible output implementation.
if [[ -f "$AUTOMATION_FILE" ]]; then
    patch_python_source "$AUTOMATION_FILE"
fi

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import pathlib
import re

path = pathlib.Path("src/budgetmem/models/__init__.py")
text = path.read_text(encoding="utf-8") if path.exists() else '"""Public model exports."""\n'
export_line = (
    "from budgetmem.models.budgetmem_r import "
    "BudgetMemR as BudgetMemR, BudgetMemROutput as BudgetMemROutput"
)
pattern = re.compile(
    r"^from budgetmem\.models\.budgetmem_r import [^\n]*$", re.MULTILINE
)
if pattern.search(text):
    text = pattern.sub(export_line, text, count=1)
else:
    if text and not text.endswith("\n"):
        text += "\n"
    text += "\n" + export_line + "\n"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(text, encoding="utf-8")
print(f"PATCHED    {path}")
PY

echo
echo "Compiling repaired Python modules..."
"$PYTHON_BIN" -m compileall -q src/budgetmem

echo
echo "Running import and adapter compatibility smoke test..."
PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import torch

from budgetmem.models import BudgetMemR, BudgetMemROutput

model = BudgetMemR(
    input_dim=6,
    hidden_dim=12,
    output_dim=3,
    key_dim=8,
    value_dim=10,
    budget_embedding_dim=5,
    controller_dim=16,
    max_budget=16,
    training_budgets=(4, 8, 16),
    top_k=3,
).eval()

with torch.no_grad():
    output = model(torch.randn(2, 10, 6), budget=torch.tensor([4, 8]))

assert isinstance(output, BudgetMemROutput)
assert output["logits"] is output.logits
assert output.memory_sizes.shape == (2, 10)
assert output.hard_writes.shape == (2, 10)
assert output.final_memory.valid.shape[0] == 2
assert torch.all(output.memory_sizes <= torch.tensor([4, 8]).unsqueeze(1))
print("BudgetMemROutput import: PASS")
print("Attribute access: PASS")
print("Dictionary access: PASS")
print("Required adapter fields: PASS")
print("Hard memory budget: PASS")
PY

echo
echo "Running Section 13 unit tests..."
PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m pytest -q tests/test_budgetmem_r.py

if [[ -f "scripts/verify_section13.py" ]]; then
    echo
    echo "Running Section 13 verifier..."
    PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" scripts/verify_section13.py
fi

cat > "$EVIDENCE_FILE" <<EOF
Section 13 BudgetMem-R Output Compatibility Repair
UTC timestamp: $TIMESTAMP

Root cause:
The public model package imported BudgetMemROutput, but the generated
src/budgetmem/models/budgetmem_r.py did not define it and returned a plain dict.

Repair status:
BudgetMemROutput public class: PASS
Package import compatibility: PASS
Dictionary access compatibility: PASS
Attribute access compatibility: PASS
Required adapter fields: PASS
Hard memory-budget enforcement: PASS
Section 13 unit tests: PASS
Final decision: PASS

Log: $LOG_FILE
Backup: $BACKUP_DIR
EOF

echo
echo "============================================================"
echo "BudgetMemROutput public class: PASS"
echo "Package import compatibility: PASS"
echo "Required adapter fields: PASS"
echo "Hard memory-budget enforcement: PASS"
echo "Section 13 unit tests: PASS"
echo "SECTION 13 COMPATIBILITY REPAIR: PASS"
echo "Evidence: $EVIDENCE_FILE"
echo "Log:      $LOG_FILE"
echo "============================================================"
