#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
MODEL="src/budgetmem/models/budgetmem_r.py"
TEST="tests/pilot/test_controller_calibration.py"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP="reports/evidence/backups/write_gate_${TIMESTAMP}"

mkdir -p "$BACKUP" reports/evidence/logs

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python environment not found at $PYTHON"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Missing $MODEL"
    exit 1
fi

cp "$MODEL" "$BACKUP/budgetmem_r.py"

"$PYTHON" - "$MODEL" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
original = text

method = '''    def _write_gate(self, logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Calculate probabilities and a deterministic hard-write decision."""
        probabilities = torch.sigmoid(logits)
        hard = (probabilities >= self.write_threshold).to(logits.dtype)

        if self.training:
            relaxed = torch.sigmoid(logits / self.write_temperature)
            gate = hard.detach() - relaxed.detach() + relaxed
        else:
            gate = hard

        return probabilities, hard, gate

'''

if "def _write_gate(self, logits:" not in text:
    markers = [
        "    def _choose_write_slots(",
        "    def _apply_writes(",
        "    def forward(",
    ]

    for marker in markers:
        location = text.find(marker)
        if location >= 0:
            text = text[:location] + method + text[location:]
            break
    else:
        raise SystemExit(
            "ERROR: Could not find a safe location for _write_gate."
        )

old = '''            write_gate = self.write_controller.differentiable_gate(
                write_probability,
                training=self.training,
                threshold=self.write_threshold,
                temperature=self.write_temperature,
            )
            hard_write = write_gate.detach() >= 0.5
'''

new = '''            epsilon = torch.finfo(write_probability.dtype).eps
            write_logits = torch.logit(
                write_probability.clamp(
                    min=epsilon,
                    max=1.0 - epsilon,
                )
            )
            write_probability, hard_write_value, write_gate = self._write_gate(
                write_logits
            )
            hard_write = hard_write_value.to(torch.bool)
'''

if old in text:
    text = text.replace(old, new, 1)

compile(text, str(path), "exec")
path.write_text(text, encoding="utf-8", newline="\n")

if text == original:
    print("The _write_gate repair was already present.")
else:
    print("BudgetMemR._write_gate added successfully.")
PY

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=2026

"$PYTHON" -m py_compile "$MODEL"

echo
echo "Running the previously failed test..."

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest \
    -q -o addopts='' \
    "${TEST}::test_training_and_evaluation_use_the_same_hard_write_decision"

echo
echo "Running all Section 15 pilot tests..."

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest \
    -q -o addopts='' \
    tests/pilot

cat > reports/evidence/section15_write_gate_fix.txt <<EOF
Section 15 write-gate repair: PASS
Timestamp UTC: $TIMESTAMP
Model: $MODEL
Exact failed test: PASS
Pilot tests: PASS
Backup: $BACKUP
EOF

echo
echo "WRITE-GATE REPAIR: PASS"

if [[ ! -f "16_15_1_section15_pilot.sh" ]]; then
    echo "ERROR: 16_15_1_section15_pilot.sh is missing."
    exit 1
fi

chmod +x 16_15_1_section15_pilot.sh

echo
echo "Starting the full Section 15 pilot..."
exec ./16_15_1_section15_pilot.sh full
