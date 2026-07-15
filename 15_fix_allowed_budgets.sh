#!/usr/bin/env bash
set -Eeuo pipefail

# Section 15 compatibility repair for BudgetMemR.allowed_budgets.
# Run from the budgetmem-r repository in the VS Code WSL terminal.

if git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    cd "$git_root"
else
    printf 'ERROR: Run this file from inside the budgetmem-r Git repository.\n' >&2
    exit 1
fi

if [[ ! -f "15_pilot_experiment.sh" ]]; then
    printf 'ERROR: 15_pilot_experiment.sh was not found in %s\n' "$PWD" >&2
    exit 1
fi

if [[ ! -f "src/budgetmem/experiments/pilot.py" ]]; then
    printf 'ERROR: src/budgetmem/experiments/pilot.py was not found.\n' >&2
    exit 1
fi

PYTHON=".venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    printf 'ERROR: %s was not found. Activate or create the project virtual environment first.\n' "$PYTHON" >&2
    exit 1
fi

"$PYTHON" - <<'PY'
from __future__ import annotations

import base64
import re
import shutil
from datetime import datetime
from pathlib import Path

pilot_path = Path("src/budgetmem/experiments/pilot.py")
automation_path = Path("15_pilot_experiment.sh")
backup_root = Path(".section15_backup") / datetime.now().strftime("allowed_budgets_%Y%m%d_%H%M%S")
backup_root.mkdir(parents=True, exist_ok=True)
shutil.copy2(pilot_path, backup_root / "pilot.py")
shutil.copy2(automation_path, backup_root / "15_pilot_experiment.sh")

text = pilot_path.read_text(encoding="utf-8")
class_marker = "class BudgetMemRAdapter(nn.Module):"
if class_marker not in text:
    raise SystemExit("ERROR: BudgetMemRAdapter was not found in pilot.py")

prefix, adapter = text.split(class_marker, 1)

old_budget_init = '        self.max_budget = max(map(int, matrix_cfg["memory_budgets"]))'
new_budget_init = (
    '        self.allowed_budgets = tuple('
    'map(int, matrix_cfg["memory_budgets"]))\n'
    '        self.max_budget = max(self.allowed_budgets)'
)

if "self.allowed_budgets = tuple(" not in adapter:
    if old_budget_init not in adapter:
        raise SystemExit("ERROR: BudgetMemRAdapter max-budget initialization was not found")
    adapter = adapter.replace(old_budget_init, new_budget_init, 1)

old_mapping = '            "max_budget": self.max_budget,\n'
new_mapping = (
    '            "max_budget": self.max_budget,\n'
    '            "allowed_budgets": self.allowed_budgets,\n'
)
if '"allowed_budgets": self.allowed_budgets' not in adapter:
    if old_mapping not in adapter:
        raise SystemExit("ERROR: BudgetMemR constructor mapping was not found")
    adapter = adapter.replace(old_mapping, new_mapping, 1)

patched_text = prefix + class_marker + adapter
pilot_path.write_text(patched_text, encoding="utf-8")

# Persist the repair inside the generated-file payload of the main Bash automation.
automation = automation_path.read_text(encoding="utf-8")
encoded = base64.b64encode(patched_text.encode("utf-8")).decode("ascii")
pattern = re.compile(
    r'(\["src/budgetmem/experiments/pilot\.py"\]=")([A-Za-z0-9+/=]+)(")'
)
updated, count = pattern.subn(lambda m: m.group(1) + encoded + m.group(3), automation, count=1)
if count != 1:
    raise SystemExit("ERROR: Could not update the embedded pilot.py payload")
automation_path.write_text(updated, encoding="utf-8")

print("PATCHED src/budgetmem/experiments/pilot.py")
print("PATCHED 15_pilot_experiment.sh embedded payload")
print(f"BACKUP  {backup_root}")
PY

printf '\nValidating the repair.\n'
"$PYTHON" -m compileall -q src/budgetmem/experiments/pilot.py
"$PYTHON" -m pytest tests/pilot/test_pilot.py -q

grep -n -A2 -B2 'allowed_budgets' src/budgetmem/experiments/pilot.py

printf '\nRepair completed. Resuming the Section 15 pilot now.\n'
bash 15_pilot_experiment.sh resume
