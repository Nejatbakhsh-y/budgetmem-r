#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-full}"
case "$MODE" in
  full|calibrate|patch-only) ;;
  *)
    printf 'ERROR: mode must be full, calibrate, or patch-only.\n' >&2
    exit 2
    ;;
esac

DEFAULT_REPO_ROOT="/mnt/c/Users/nejat/OneDrive/Desktop/UN/Skills/GitHub 2026/budgetmem-r"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/.git" ]]; then
  REPO_ROOT="$SCRIPT_DIR"
else
  REPO_ROOT="${BUDGETMEM_REPO_ROOT:-$DEFAULT_REPO_ROOT}"
fi

if [[ ! -d "$REPO_ROOT/.git" ]]; then
  printf 'ERROR: budgetmem-r Git repository not found: %s\n' "$REPO_ROOT" >&2
  exit 1
fi

cd "$REPO_ROOT"
PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  printf 'ERROR: virtual-environment Python not found: %s\n' "$PYTHON" >&2
  exit 1
fi

required=(
  "src/budgetmem/models/budgetmem_r.py"
  "src/budgetmem/experiments/pilot.py"
  "configs/experiments/pilot.yaml"
  "scripts/run_pilot.py"
  "15_pilot_experiment.sh"
)
for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    printf 'ERROR: required file is missing: %s\n' "$path" >&2
    exit 1
  fi
done

export PYTHONPATH="$REPO_ROOT/src"
export PYTHONHASHSEED=2026
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

timestamp="$(date +%Y%m%d_%H%M%S)"
backup_root="$REPO_ROOT/.section15_backup/controller_fix_$timestamp"
mkdir -p "$backup_root"

printf '\n============================================================\n'
printf 'SECTION 15 - BUDGETMEM-R CONTROLLER-COLLAPSE REPAIR\n'
printf 'MODE: %s\n' "$MODE"
printf '============================================================\n\n'

# Preserve the valid NO_GO evidence before changing code or configuration.
mkdir -p "$backup_root/reports/evidence" "$backup_root/reports/tables"
for path in \
  reports/evidence/pilot_go_no_go.json \
  reports/evidence/pilot_summary.json \
  reports/evidence/section15_controller_diagnostic.txt \
  reports/tables/pilot_results.csv \
  reports/pilot_report.md; do
  if [[ -f "$path" ]]; then
    mkdir -p "$backup_root/$(dirname "$path")"
    cp -f "$path" "$backup_root/$path"
  fi
done

for path in \
  src/budgetmem/models/budgetmem_r.py \
  src/budgetmem/experiments/pilot.py \
  configs/experiments/pilot.yaml \
  15_pilot_experiment.sh; do
  mkdir -p "$backup_root/$(dirname "$path")"
  cp -f "$path" "$backup_root/$path"
done

printf 'Preserved baseline and source backup at:\n%s\n\n' "$backup_root"

"$PYTHON" - <<'PY'
from __future__ import annotations

import base64
import re
from pathlib import Path

import yaml


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"Cannot patch {label}: expected exactly one original block, found {count}."
        )
    return text.replace(old, new, 1)


core_path = Path("src/budgetmem/models/budgetmem_r.py")
core = core_path.read_text(encoding="utf-8")
old_gate = '''    def _write_gate(self, logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        probabilities = torch.sigmoid(logits)
        if self.training:
            uniform = torch.rand_like(logits).clamp_(1.0e-6, 1.0 - 1.0e-6)
            logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
            relaxed = torch.sigmoid((logits + logistic_noise) / self.write_temperature)
            hard = (relaxed >= self.write_threshold).to(logits.dtype)
            straight_through = hard.detach() - relaxed.detach() + relaxed
            return probabilities, hard, straight_through
        hard = (probabilities >= self.write_threshold).to(logits.dtype)
        return probabilities, hard, hard
'''
new_gate = '''    def _write_gate(self, logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Training and evaluation must use the same deterministic hard decision.
        # The relaxed value is retained only as the straight-through gradient path.
        probabilities = torch.sigmoid(logits)
        hard = (probabilities >= self.write_threshold).to(logits.dtype)
        if self.training:
            relaxed = torch.sigmoid(logits / self.write_temperature)
            straight_through = hard.detach() - relaxed.detach() + relaxed
            return probabilities, hard, straight_through
        return probabilities, hard, hard
'''
core = replace_once(core, old_gate, new_gate, label="BudgetMemR._write_gate")
core_path.write_text(core, encoding="utf-8")

pilot_path = Path("src/budgetmem/experiments/pilot.py")
pilot = pilot_path.read_text(encoding="utf-8")

old_cfg_init = '''        model_cfg = cfg["model"]
        matrix_cfg = cfg["matrix"]
        self.vocab_size = int(model_cfg["vocabulary_size"])
'''
new_cfg_init = '''        model_cfg = cfg["model"]
        matrix_cfg = cfg["matrix"]
        training_cfg = cfg["training"]
        self.write_threshold = float(training_cfg.get("write_threshold", 0.5))
        self.write_temperature = float(training_cfg.get("write_temperature", 0.67))
        self.vocab_size = int(model_cfg["vocabulary_size"])
'''
pilot = replace_once(pilot, old_cfg_init, new_cfg_init, label="adapter configuration")

old_constructor = '''            "write_threshold": 0.5,
            "temperature": 1.0,
'''
new_constructor = '''            "write_threshold": self.write_threshold,
            "write_temperature": self.write_temperature,
            "temperature": self.write_temperature,
'''
pilot = replace_once(pilot, old_constructor, new_constructor, label="adapter controller parameters")

new_loss = '''    if output.write_probabilities is not None:
        probabilities = output.write_probabilities.float()
        threshold = float(training_cfg.get("write_threshold", 0.5))
        hard = (probabilities >= threshold).to(probabilities.dtype)
        straight_through = hard.detach() - probabilities.detach() + probabilities
        per_sample_write_rate = straight_through.mean(dim=1)
        target_rate = float(training_cfg["write_rate_target"])
        loss = loss + float(training_cfg["write_rate_penalty"]) * (
            per_sample_write_rate - target_rate
        ).square().mean()
        loss = loss + float(training_cfg.get("write_binarization_penalty", 0.0)) * (
            probabilities * (1.0 - probabilities)
        ).mean()
'''

def patch_write_rate_loss(text: str, *, label: str) -> str:
    if new_loss in text:
        return text
    pattern = re.compile(
        r"    if output\.write_probabilities is not None:\n"
        r".*?"
        r"(?=    if output\.memory_sizes is not None:)",
        flags=re.DOTALL,
    )
    updated, count = pattern.subn(new_loss, text, count=1)
    if count != 1:
        raise SystemExit(
            f"Cannot patch {label}: expected one write-probability loss block, found {count}."
        )
    return updated

pilot = patch_write_rate_loss(pilot, label="write-rate loss")
pilot_path.write_text(pilot, encoding="utf-8")

config_path = Path("configs/experiments/pilot.yaml")
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
training = config["training"]
training["write_threshold"] = 0.5
training["write_temperature"] = 0.67
training["write_rate_penalty"] = 5.0
training["write_binarization_penalty"] = 0.05
config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

# Keep the one-file Section 15 installer synchronized so a later rerun cannot
# silently restore the defective pilot implementation or old configuration.
automation_path = Path("15_pilot_experiment.sh")
automation = automation_path.read_text(encoding="utf-8")


def replace_payload(script: str, relative_path: str, transform) -> str:
    pattern = re.compile(
        r'(\["' + re.escape(relative_path) + r'"\]="?)([A-Za-z0-9+/=]+)("?)'
    )
    match = pattern.search(script)
    if not match:
        raise SystemExit(f"Cannot locate embedded payload for {relative_path}")
    decoded = base64.b64decode(match.group(2)).decode("utf-8")
    updated = transform(decoded)
    encoded = base64.b64encode(updated.encode("utf-8")).decode("ascii")
    return script[: match.start(2)] + encoded + script[match.end(2) :]


def patch_embedded_pilot(text: str) -> str:
    text = replace_once(text, old_cfg_init, new_cfg_init, label="embedded adapter configuration")
    text = replace_once(text, old_constructor, new_constructor, label="embedded controller parameters")
    text = patch_write_rate_loss(text, label="embedded write-rate loss")
    return text


def patch_embedded_config(text: str) -> str:
    data = yaml.safe_load(text)
    section = data["training"]
    section["write_threshold"] = 0.5
    section["write_temperature"] = 0.67
    section["write_rate_penalty"] = 5.0
    section["write_binarization_penalty"] = 0.05
    return yaml.safe_dump(data, sort_keys=False)


automation = replace_payload(
    automation, "src/budgetmem/experiments/pilot.py", patch_embedded_pilot
)
automation = replace_payload(
    automation, "configs/experiments/pilot.yaml", patch_embedded_config
)
automation_path.write_text(automation, encoding="utf-8")

# Add focused regression tests for the exact failure mode.
test_path = Path("tests/pilot/test_controller_calibration.py")
test_path.parent.mkdir(parents=True, exist_ok=True)
test_path.write_text(
    '''"""Regression tests for deterministic BudgetMem-R write calibration."""

from __future__ import annotations

import torch

from budgetmem.experiments.pilot import PilotOutput, total_training_loss
from budgetmem.models.budgetmem_r import BudgetMemR


def test_training_and_evaluation_use_the_same_hard_write_decision() -> None:
    model = BudgetMemR(
        input_dim=4,
        hidden_dim=8,
        output_dim=3,
        max_budget=4,
        allowed_budgets=(2, 4),
        write_threshold=0.5,
    )
    logits = torch.tensor([-1.0, 1.0])
    model.train()
    _, training_hard, _ = model._write_gate(logits)
    model.eval()
    _, evaluation_hard, _ = model._write_gate(logits)
    assert torch.equal(training_hard, evaluation_hard)


def test_write_rate_loss_pushes_collapsed_probabilities_upward() -> None:
    probabilities = torch.full((2, 8), 0.10, requires_grad=True)
    logits = torch.zeros(2, 12, 192, requires_grad=True)
    target = torch.zeros(2, 12, dtype=torch.long)
    output = PilotOutput(logits=logits, write_probabilities=probabilities)
    cfg = {
        "training": {
            "write_threshold": 0.5,
            "write_rate_target": 0.15,
            "write_rate_penalty": 5.0,
            "write_binarization_penalty": 0.05,
            "budget_violation_penalty": 10.0,
        }
    }
    loss = total_training_loss(
        output,
        target,
        model_name="budgetmem_r",
        budget=16,
        cfg=cfg,
    )
    loss.backward()
    assert probabilities.grad is not None
    assert float(probabilities.grad.mean()) < 0.0
''',
    encoding="utf-8",
)

print("Patched deterministic write gating, hard-rate calibration loss, configuration, installer, and tests.")
PY

printf '\nCompiling and checking the repair.\n'
"$PYTHON" -m compileall -q \
  src/budgetmem/models/budgetmem_r.py \
  src/budgetmem/experiments/pilot.py \
  tests/pilot/test_controller_calibration.py

"$PYTHON" -m ruff format \
  src/budgetmem/models/budgetmem_r.py \
  src/budgetmem/experiments/pilot.py \
  tests/pilot/test_controller_calibration.py

"$PYTHON" -m ruff check \
  src/budgetmem/models/budgetmem_r.py \
  src/budgetmem/experiments/pilot.py \
  tests/pilot/test_controller_calibration.py

"$PYTHON" -m pytest tests/pilot -q

if [[ "$MODE" == "patch-only" ]]; then
  printf '\nPATCH-ONLY MODE COMPLETE.\n'
  git status --short
  exit 0
fi

printf '\nCreating an isolated controller-calibration smoke configuration.\n'
calibration_config="configs/experiments/pilot_controller_calibration.yaml"
"$PYTHON" - <<'PY'
from pathlib import Path
import yaml

source = Path("configs/experiments/pilot.yaml")
target = Path("configs/experiments/pilot_controller_calibration.yaml")
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config["matrix"]["tasks"] = ["selective_copy"]
config["matrix"]["evaluation_sequence_lengths"] = [256]
config["matrix"]["memory_budgets"] = [16]
config["matrix"]["models"] = ["budgetmem_r"]
config["training"]["train_samples"] = 64
config["training"]["validation_samples"] = 24
config["training"]["epochs"] = 4
config["artifacts"] = {
    "output_root": "outputs/pilot_controller_calibration",
    "results_csv": "reports/tables/pilot_controller_calibration_results.csv",
    "summary_json": "reports/evidence/pilot_controller_calibration_summary.json",
    "gate_json": "reports/evidence/pilot_controller_calibration_gate.json",
    "report_markdown": "reports/pilot_controller_calibration_report.md",
    "checkpoint_root": "outputs/pilot_controller_calibration/checkpoints",
}
target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

rm -rf outputs/pilot_controller_calibration
rm -f \
  reports/tables/pilot_controller_calibration_results.csv \
  reports/evidence/pilot_controller_calibration_summary.json \
  reports/evidence/pilot_controller_calibration_gate.json \
  reports/pilot_controller_calibration_report.md

printf '\nRunning isolated BudgetMem-R controller calibration.\n'
"$PYTHON" - <<'CALIBRATIONPY'
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from budgetmem.experiments.pilot import (
    evaluate_model,
    read_yaml,
    seed_everything,
    sha256_file,
    stable_int,
    train_one_model,
    write_csv,
)

config_path = Path("configs/experiments/pilot_controller_calibration.yaml").resolve()
cfg = read_yaml(config_path)
seed = int(cfg["seed"])
seed_everything(seed)
config_sha256 = sha256_file(config_path)
task = "selective_copy"
model_name = "budgetmem_r"
model_seed = seed + stable_int(f"{task}:{model_name}") % 1_000_000
model, record = train_one_model(
    cfg=cfg,
    config_sha256=config_sha256,
    task=task,
    model_name=model_name,
    seed=model_seed,
    resume=False,
)
row = evaluate_model(
    cfg=cfg,
    config_path=config_path,
    config_sha256=config_sha256,
    task=task,
    sequence_length=256,
    budget=16,
    model_name=model_name,
    model=model,
    training_record=record,
    seed=seed,
)
results_path = Path(cfg["artifacts"]["results_csv"])
write_csv(results_path, [row])
summary_path = Path(cfg["artifacts"]["summary_json"])
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(
    json.dumps({"training_record": asdict(record), "result": row}, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(row, indent=2))
CALIBRATIONPY

printf '\nChecking whether BudgetMem-R now writes nontrivially.\n'
"$PYTHON" - <<'PY'
import json
from pathlib import Path

import pandas as pd

path = Path("reports/tables/pilot_controller_calibration_results.csv")
df = pd.read_csv(path)
rows = df[df["model"].eq("budgetmem_r")]
if rows.empty:
    raise SystemExit("Controller calibration failed: no BudgetMem-R result row.")
write_frequency = float(rows["write_frequency"].mean())
maximum_size = int(rows["max_memory_size"].max())
budget_pass = bool(rows["budget_pass"].all())
passed = 0.01 <= write_frequency <= 0.95 and budget_pass
summary = {
    "status": "PASS" if passed else "FAIL",
    "mean_budgetmem_write_frequency": write_frequency,
    "maximum_memory_size": maximum_size,
    "budget_pass": budget_pass,
    "required_write_frequency_interval": [0.01, 0.95],
}
Path("reports/evidence/section15_controller_fix_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
if not passed:
    raise SystemExit(
        "Controller calibration still failed. The full pilot was not rerun."
    )
PY

if [[ "$MODE" == "calibrate" ]]; then
  printf '\nCALIBRATION MODE COMPLETE. The full pilot was not rerun.\n'
  git status --short
  exit 0
fi

printf '\nController calibration passed. Archiving the previous NO_GO operational output.\n'
if [[ -d outputs/pilot ]]; then
  archive_dir="outputs/section15_no_go_$timestamp"
  mv outputs/pilot "$archive_dir"
  printf 'Archived old checkpoints and effective configuration at: %s\n' "$archive_dir"
fi

printf '\nRunning a fresh full Section 15 pilot with the repaired controller.\n'
"$PYTHON" scripts/run_pilot.py --config configs/experiments/pilot.yaml

printf '\nFINAL SECTION 15 DECISION\n'
cat reports/evidence/pilot_go_no_go.json

printf '\nRepository status:\n'
git status --short

printf '\n============================================================\n'
printf 'SECTION 15 CONTROLLER REPAIR AUTOMATION COMPLETED\n'
printf 'Baseline backup: %s\n' "$backup_root"
printf 'Controller-fix summary: reports/evidence/section15_controller_fix_summary.json\n'
printf 'Final decision: reports/evidence/pilot_go_no_go.json\n'
printf '============================================================\n'
