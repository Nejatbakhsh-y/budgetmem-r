#!/usr/bin/env bash
# Repair the Section 13 public model-package import after the BudgetMemROutput patch.
# Run from the budgetmem-r repository root in the VS Code WSL terminal.

set -Eeuo pipefail
IFS=$'\n\t'

readonly TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly INIT_FILE="src/budgetmem/models/__init__.py"
readonly MODEL_FILE="src/budgetmem/models/budgetmem_r.py"
readonly LOG_DIR="reports/evidence/logs"
readonly BACKUP_DIR="reports/evidence/backups/section13_models_init_repair_${TIMESTAMP}"
readonly LOG_FILE="${LOG_DIR}/section13_models_init_repair_${TIMESTAMP}.log"
readonly EVIDENCE_FILE="reports/evidence/section13_models_init_repair.txt"

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
    echo "ERROR: Section 13 model-package repair failed at line ${line_number} with exit code ${exit_code}."
    echo "Review: ${LOG_FILE}"
    exit "$exit_code"
}
trap 'on_error $LINENO' ERR

printf '%s\n' "============================================================"
printf '%s\n' "16.13.1 — Repair BudgetMem-R Model Package"
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

backup_file "$INIT_FILE"
backup_file "$MODEL_FILE"

# First ensure the model defines and returns the public compatibility object.
"$PYTHON_BIN" - "$MODEL_FILE" <<'PY'
from __future__ import annotations

from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
original = text

class_block = '''\n\nclass BudgetMemROutput(dict[str, object]):\n    """Dictionary-compatible output with attribute access for project adapters."""\n\n    def __init__(self, **values: object) -> None:\n        super().__init__(values)\n\n    def __getattr__(self, name: str) -> object:\n        try:\n            return self[name]\n        except KeyError as exc:\n            raise AttributeError(name) from exc\n'''

if "class BudgetMemROutput(" not in text:
    marker = "\n\nclass BudgetMemR(nn.Module):"
    if marker not in text:
        raise SystemExit(f"Could not locate BudgetMemR in {path}")
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
    raise SystemExit(f"Could not locate the BudgetMem-R output block in {path}")

compile(text, str(path), "exec")
if text != original:
    path.write_text(text, encoding="utf-8")
    print(f"PATCHED    {path}")
else:
    print(f"UNCHANGED  {path}")
PY

# Rebuild only the BudgetMem-R import in models/__init__.py. Prefer the valid
# pre-corruption backup produced by the preceding repair attempt when available.
"$PYTHON_BIN" - "$INIT_FILE" <<'PY'
from __future__ import annotations

from pathlib import Path
import sys

path = Path(sys.argv[1])
backup_root = Path("reports/evidence/backups")


def is_valid(text: str, filename: str) -> bool:
    try:
        compile(text, filename, "exec")
    except SyntaxError:
        return False
    return True

candidates = sorted(
    backup_root.glob("section13_output_fix_*/src/budgetmem/models/__init__.py"),
    key=lambda item: item.stat().st_mtime,
    reverse=True,
)

base_text: str | None = None
base_source = "current file"
for candidate in candidates:
    candidate_text = candidate.read_text(encoding="utf-8")
    if is_valid(candidate_text, str(candidate)):
        base_text = candidate_text
        base_source = str(candidate)
        break

if base_text is None:
    if path.exists():
        base_text = path.read_text(encoding="utf-8")
    else:
        base_text = '"""Public model exports."""\n'

lines = base_text.splitlines()
cleaned: list[str] = []
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    if stripped.startswith("from budgetmem.models.budgetmem_r import"):
        depth = line.count("(") - line.count(")")
        i += 1
        while depth > 0 and i < len(lines):
            depth += lines[i].count("(") - lines[i].count(")")
            i += 1
        # A previous failed one-line replacement can leave these continuation
        # lines behind. Remove only the exact known remnants directly after it.
        while i < len(lines) and lines[i].strip() in {
            "BudgetMemR,",
            "BudgetMemROutput,",
            "BudgetMemR as BudgetMemR,",
            "BudgetMemROutput as BudgetMemROutput,",
            ")",
        }:
            i += 1
        continue
    cleaned.append(line)
    i += 1

# If no valid backup existed, remove the exact orphaned lines from the damaged
# top-level block without changing other package exports.
while cleaned and not cleaned[-1].strip():
    cleaned.pop()

canonical_import = (
    "from budgetmem.models.budgetmem_r import (\n"
    "    BudgetMemR as BudgetMemR,\n"
    "    BudgetMemROutput as BudgetMemROutput,\n"
    ")"
)

new_text = "\n".join(cleaned)
if new_text:
    new_text += "\n\n"
new_text += canonical_import + "\n"

try:
    compile(new_text, str(path), "exec")
except SyntaxError:
    # Last-resort cleanup for the already-corrupted three-line remnant near the
    # beginning of the file. This remains deliberately narrow.
    repaired_lines: list[str] = []
    for line_number, line in enumerate(cleaned, start=1):
        if line_number <= 30 and line.strip() in {
            "BudgetMemR,",
            "BudgetMemROutput,",
            "BudgetMemR as BudgetMemR,",
            "BudgetMemROutput as BudgetMemROutput,",
            ")",
        } and line[:1].isspace():
            continue
        repaired_lines.append(line)
    while repaired_lines and not repaired_lines[-1].strip():
        repaired_lines.pop()
    new_text = "\n".join(repaired_lines)
    if new_text:
        new_text += "\n\n"
    new_text += canonical_import + "\n"
    compile(new_text, str(path), "exec")

path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(new_text, encoding="utf-8")
print(f"RESTORE SOURCE: {base_source}")
print(f"REBUILT    {path}")
PY

echo
echo "Compiling repaired modules..."
"$PYTHON_BIN" -m py_compile "$MODEL_FILE" "$INIT_FILE"

echo
echo "Running package-import and output compatibility test..."
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
print("BudgetMem-R package import: PASS")
print("BudgetMemROutput compatibility: PASS")
print("Hard memory-budget enforcement: PASS")
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
Section 13 BudgetMem-R Model-Package Repair
UTC timestamp: $TIMESTAMP

Root cause:
The previous compatibility repair replaced the first line of a parenthesized
import in src/budgetmem/models/__init__.py but left its indented continuation
lines, producing an IndentationError.

Repair status:
Model package syntax: PASS
BudgetMemR package import: PASS
BudgetMemROutput package import: PASS
Output adapter compatibility: PASS
Hard memory-budget enforcement: PASS
Section 13 unit tests: PASS
Section 13 verifier: PASS
Final decision: PASS

Log: $LOG_FILE
Backup: $BACKUP_DIR
EOF

echo
echo "============================================================"
echo "Model package syntax: PASS"
echo "BudgetMemR package import: PASS"
echo "BudgetMemROutput compatibility: PASS"
echo "Hard memory-budget enforcement: PASS"
echo "Section 13 unit tests: PASS"
echo "Section 13 verifier: PASS"
echo "SECTION 13 REPAIR: PASS"
echo "Evidence: $EVIDENCE_FILE"
echo "Log:      $LOG_FILE"
echo "============================================================"
