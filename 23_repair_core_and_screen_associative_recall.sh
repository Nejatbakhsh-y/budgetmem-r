#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"

MODEL_FILE="src/budgetmem/models/budgetmem_r.py"
GENERATOR_FILE="scripts/data/generate_synthetic.py"
BASE_CONFIG="configs/experiments/pilot_tuned.yaml"

SCREEN_CONFIG="configs/experiments/pilot_assoc_core_repair_screen.yaml"
SCREEN_RUNNER="scripts/run_assoc_core_repair_screen.py"

RESULTS="reports/tables/assoc_core_repair_screen_results.csv"
DECISION_JSON="reports/evidence/assoc_core_repair_screen_decision.json"
DECISION_TXT="reports/evidence/assoc_core_repair_screen_decision.txt"
LOG_FILE="reports/logs/assoc_core_repair_screen.log"

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP=".section15_backup/core_repair_${STAMP}"

echo "============================================================"
echo " Section 15 Core Repair and Associative-Recall Screen"
echo "============================================================"
echo "Repository: $ROOT"
echo

for required in \
    "$PYTHON" \
    "$MODEL_FILE" \
    "$GENERATOR_FILE" \
    "$BASE_CONFIG"; do

    if [[ ! -e "$required" ]]; then
        echo "ERROR: Missing required path:"
        echo "  $required"
        exit 1
    fi
done

mkdir -p \
    "$BACKUP/src/budgetmem/models" \
    "$BACKUP/scripts/data" \
    "$BACKUP/tests/models" \
    "$BACKUP/tests/tasks" \
    reports/tables \
    reports/evidence \
    reports/logs \
    configs/experiments

cp -f "$MODEL_FILE" \
    "$BACKUP/src/budgetmem/models/budgetmem_r.py"

cp -f "$GENERATOR_FILE" \
    "$BACKUP/scripts/data/generate_synthetic.py"

git status --short > "$BACKUP/git_status_before.txt"
git diff > "$BACKUP/git_diff_before.patch"

echo "Backup created:"
echo "  $BACKUP"
echo

# ============================================================
# PATCH THE PILOT ORACLE AND BUDGETMEM-R WRITE GRADIENT
# ============================================================

"$PYTHON" - <<'PY'
from __future__ import annotations

from pathlib import Path


def replace_once(
    text: str,
    old: str,
    new: str,
    *,
    description: str,
) -> str:
    count = text.count(old)

    if count != 1:
        raise SystemExit(
            f"ERROR: Expected one occurrence for {description}, "
            f"found {count}."
        )

    return text.replace(old, new, 1)


# ------------------------------------------------------------
# Repair the actual synthetic generator used by the pilot.
# ------------------------------------------------------------

generator_path = Path(
    "scripts/data/generate_synthetic.py"
)

generator_text = generator_path.read_text(
    encoding="utf-8"
)

association_marker = (
    "# SECTION15_ORACLE_REPAIR: retain only queried values"
)

if association_marker not in generator_text:
    old = """    selected = rng.choice(len(keys), size=qn, replace=False).astype(int).tolist()
    suffix = [QUERY] + [keys[i] for i in selected] + [EOS]
"""

    new = """    selected = rng.choice(len(keys), size=qn, replace=False).astype(int).tolist()

    # SECTION15_ORACLE_REPAIR: retain only queried values.
    # The answer depends only on values belonging to the queried keys.
    relevant_positions = [
        relevant_positions[index]
        for index in selected
    ]

    suffix = [QUERY] + [keys[i] for i in selected] + [EOS]
"""

    generator_text = replace_once(
        generator_text,
        old,
        new,
        description="associative-recall oracle repair",
    )


multiple_marker = (
    "# SECTION15_MULTIKEY_ORACLE_REPAIR: retain only queried values"
)

if multiple_marker not in generator_text:
    old = """    selected = rng.choice(len(keys), size=qn, replace=False).astype(int).tolist()
    suffix = [SEP, QUERY] + [keys[i] for i in selected] + [EOS]
"""

    new = """    selected = rng.choice(len(keys), size=qn, replace=False).astype(int).tolist()

    # SECTION15_MULTIKEY_ORACLE_REPAIR: retain only queried values.
    relevant_positions = [
        relevant_positions[index]
        for index in selected
    ]

    suffix = [SEP, QUERY] + [keys[i] for i in selected] + [EOS]
"""

    generator_text = replace_once(
        generator_text,
        old,
        new,
        description="multiple-key oracle repair",
    )


distractor_marker = (
    "# SECTION15_DISTRACTOR_ORACLE_REPAIR: retain only queried values"
)

if distractor_marker not in generator_text:
    old = """    queried = rng.choice(len(keys), size=qn, replace=False).astype(int).tolist()
    suffix = [QUERY] + [keys[i] for i in queried] + [EOS]
"""

    new = """    queried = rng.choice(len(keys), size=qn, replace=False).astype(int).tolist()

    # SECTION15_DISTRACTOR_ORACLE_REPAIR: retain only queried values.
    relevant_positions = [
        relevant_positions[index]
        for index in queried
    ]

    suffix = [QUERY] + [keys[i] for i in queried] + [EOS]
"""

    generator_text = replace_once(
        generator_text,
        old,
        new,
        description="distractor-retrieval oracle repair",
    )


generator_path.write_text(
    generator_text,
    encoding="utf-8",
)


# ------------------------------------------------------------
# Repair the BudgetMem-R write-gradient path.
# ------------------------------------------------------------

model_path = Path(
    "src/budgetmem/models/budgetmem_r.py"
)

model_text = model_path.read_text(
    encoding="utf-8"
)

selection_marker = (
    "# SECTION15_STRAIGHT_THROUGH_REPAIR:"
)

if selection_marker not in model_text:
    old = """        selection = selection * hard_write.unsqueeze(1)
        selected = selection.argmax(dim=1)
"""

    new = """        # SECTION15_STRAIGHT_THROUGH_REPAIR:
        # Keep the selected slot available to the relaxed backward
        # path even when the hard forward decision is no-write.
        # Forward execution remains exactly hard because the
        # straight-through gate has the hard value in the forward pass.
        selected = selection.argmax(dim=1)
"""

    model_text = replace_once(
        model_text,
        old,
        new,
        description="straight-through slot-selection repair",
    )


fill_marker = (
    "# SECTION15_INITIAL_FILL_REPAIR:"
)

if fill_marker not in model_text:
    old = """            write_probability, hard_write, straight_through_gate = self._write_gate(
                write_logits
            )
            utility_scores = self.utility_controller(
"""

    new = """            write_probability, hard_write, straight_through_gate = self._write_gate(
                write_logits
            )

            # SECTION15_INITIAL_FILL_REPAIR:
            # Empty logical slots are filled before learned replacement
            # decisions begin. This prevents an initially conservative
            # controller from creating permanently unreachable memory.
            must_fill = memory.sizes() < budgets

            hard_write = torch.where(
                must_fill,
                torch.ones_like(hard_write),
                hard_write,
            )

            straight_through_gate = torch.where(
                must_fill,
                torch.ones_like(straight_through_gate),
                straight_through_gate,
            )

            utility_scores = self.utility_controller(
"""

    model_text = replace_once(
        model_text,
        old,
        new,
        description="initial-memory-fill repair",
    )


model_path.write_text(
    model_text,
    encoding="utf-8",
)

print("Patched:")
print(f"- {generator_path}")
print(f"- {model_path}")
PY

# ============================================================
# ADD REGRESSION TESTS
# ============================================================

cat > tests/models/test_budgetmem_r_straight_through_repair.py <<'PY'
"""Regression tests for the Section 15 write-gradient repair."""

from __future__ import annotations

import torch

from budgetmem.models.budgetmem_r import BudgetMemR


def _model() -> BudgetMemR:
    return BudgetMemR(
        input_dim=6,
        hidden_dim=12,
        output_dim=4,
        max_budget=4,
        allowed_budgets=(2, 4),
        key_dim=8,
        value_dim=10,
        retrieval_k=2,
        write_threshold=1.0,
    )


def test_empty_memory_is_filled_before_replacement_control() -> None:
    torch.manual_seed(2026)

    model = _model().eval()
    inputs = torch.randn(3, 9, 6)

    with torch.no_grad():
        output = model(inputs, budget=2)

    assert torch.equal(
        output.final_memory.sizes(),
        torch.full((3,), 2, dtype=torch.long),
    )

    assert torch.all(
        output.memory_sizes <= 2
    )

    assert torch.all(
        output.hard_writes[:, :2] == 1
    )


def test_task_path_reaches_write_controller_after_memory_is_full() -> None:
    torch.manual_seed(2026)

    model = _model().train()
    inputs = torch.randn(3, 10, 6)

    output = model(inputs, budget=2)

    loss = output.logits.square().mean()
    loss.backward()

    gradient_sum = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.write_controller.parameters()
        if parameter.grad is not None
    )

    assert gradient_sum > 0.0
PY

cat > tests/tasks/test_pilot_oracle_alignment.py <<'PY'
"""Tests for the synthetic generator actually used by Section 15."""

from __future__ import annotations

import numpy as np

from scripts.data.generate_synthetic import associative_recall


def test_pilot_associative_oracle_contains_only_queried_values() -> None:
    cfg = {
        "sequence_length": 128,
        "vocabulary_size": 192,
        "number_keys": 12,
        "number_queries": 1,
        "delay_length": 32,
        "distractor_percentage": 65,
        "number_relevant_events": 12,
        "random_seed": 102,
    }

    rng = np.random.default_rng(2026)

    (
        sequence,
        target,
        relevant_positions,
        query_positions,
        metadata,
    ) = associative_recall(
        cfg,
        rng,
    )

    assert len(target) == 1
    assert len(relevant_positions) == 1
    assert len(query_positions) == 1
    assert len(metadata["queried_keys"]) == 1

    relevant_position = relevant_positions[0]

    assert sequence[relevant_position] == target[0]
    assert relevant_position < query_positions[0]
PY

export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo
echo "Running focused regression tests."

"$PYTHON" -m pytest \
    tests/models/test_budgetmem_r.py \
    tests/models/test_budgetmem_r_straight_through_repair.py \
    tests/tasks/test_associative_recall.py \
    tests/tasks/test_pilot_oracle_alignment.py \
    tests/pilot/test_controller_calibration.py \
    tests/pilot/test_pilot.py \
    -q

echo
echo "Regression tests: PASS"

# ============================================================
# CREATE AN ISOLATED TARGETED CONFIGURATION
# ============================================================

"$PYTHON" - <<'PY'
from __future__ import annotations

from pathlib import Path

import yaml


source = Path(
    "configs/experiments/pilot_tuned.yaml"
)

destination = Path(
    "configs/experiments/"
    "pilot_assoc_core_repair_screen.yaml"
)

cfg = yaml.safe_load(
    source.read_text(encoding="utf-8")
)

cfg["experiment_name"] = (
    "section15_associative_core_repair_screen"
)

# The official matrix remains in the stored configuration.
# The targeted Python runner narrows an in-memory copy only.
cfg["artifacts"] = {
    "output_root": (
        "outputs/assoc_core_repair_screen"
    ),
    "results_csv": (
        "reports/tables/"
        "assoc_core_repair_screen_results.csv"
    ),
    "summary_json": (
        "reports/evidence/"
        "assoc_core_repair_screen_summary.json"
    ),
    "gate_json": (
        "reports/evidence/"
        "assoc_core_repair_screen_decision.json"
    ),
    "report_markdown": (
        "reports/"
        "assoc_core_repair_screen_report.md"
    ),
    "checkpoint_root": (
        "outputs/assoc_core_repair_screen/checkpoints"
    ),
}

destination.write_text(
    yaml.safe_dump(
        cfg,
        sort_keys=False,
    ),
    encoding="utf-8",
)

print(f"Created: {destination}")
PY

# ============================================================
# CREATE THE ISOLATED TARGETED RUNNER
# ============================================================

cat > "$SCREEN_RUNNER" <<'PY'
from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from budgetmem.experiments.pilot import (
    evaluate_model,
    read_yaml,
    seed_everything,
    sha256_file,
    stable_int,
    train_one_model,
    validate_config,
    validate_pretraining_gate,
    write_csv,
)


ROOT = Path.cwd()

CONFIG_PATH = ROOT / (
    "configs/experiments/"
    "pilot_assoc_core_repair_screen.yaml"
)

RESULTS_PATH = ROOT / (
    "reports/tables/"
    "assoc_core_repair_screen_results.csv"
)

SUMMARY_PATH = ROOT / (
    "reports/evidence/"
    "assoc_core_repair_screen_summary.json"
)

DECISION_JSON = ROOT / (
    "reports/evidence/"
    "assoc_core_repair_screen_decision.json"
)

DECISION_TXT = ROOT / (
    "reports/evidence/"
    "assoc_core_repair_screen_decision.txt"
)

MINIMUM_GAIN = 0.02
TARGET_TASK = "associative_recall"
TARGET_LENGTH = 1024
TARGET_BUDGETS = [16, 32]

TARGET_MODELS = [
    "gru_uniform_cache",
    "gru_reservoir_cache",
    "budgetmem_r",
]


base_cfg = read_yaml(CONFIG_PATH)

# Validate the complete stored Section 15 configuration before
# creating the isolated diagnostic copy.
validate_config(base_cfg)
validate_pretraining_gate(base_cfg)

cfg = copy.deepcopy(base_cfg)

cfg["experiment_name"] = (
    "section15_associative_core_repair_screen"
)

cfg["matrix"]["tasks"] = [
    TARGET_TASK,
]

cfg["matrix"]["evaluation_sequence_lengths"] = [
    TARGET_LENGTH,
]

cfg["matrix"]["memory_budgets"] = (
    TARGET_BUDGETS
)

cfg["matrix"]["models"] = (
    TARGET_MODELS
)

effective_path = (
    ROOT
    / "outputs/assoc_core_repair_screen/"
    "effective_screen_config.yaml"
)

effective_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

effective_path.write_text(
    yaml.safe_dump(
        cfg,
        sort_keys=False,
    ),
    encoding="utf-8",
)

config_sha256 = sha256_file(
    effective_path
)

seed = int(cfg["seed"])
seed_everything(seed)

trained = {}
training_records = []

for model_name in TARGET_MODELS:
    model_seed = (
        seed
        + stable_int(
            f"{TARGET_TASK}:{model_name}"
        )
        % 1_000_000
    )

    model, record = train_one_model(
        cfg=cfg,
        config_sha256=config_sha256,
        task=TARGET_TASK,
        model_name=model_name,
        seed=model_seed,
        resume=False,
    )

    trained[model_name] = model
    training_records.append(record)

    print(
        "TRAINED "
        f"task={TARGET_TASK} "
        f"model={model_name} "
        f"loss={record.final_loss:.6f} "
        f"stable={record.stability_pass}",
        flush=True,
    )


record_lookup = {
    record.model: record
    for record in training_records
}

rows = []

for model_name in TARGET_MODELS:
    for budget in TARGET_BUDGETS:
        row = evaluate_model(
            cfg=cfg,
            config_path=effective_path,
            config_sha256=config_sha256,
            task=TARGET_TASK,
            sequence_length=TARGET_LENGTH,
            budget=budget,
            model_name=model_name,
            model=trained[model_name],
            training_record=record_lookup[
                model_name
            ],
            seed=seed,
        )

        rows.append(row)

        print(
            "EVALUATED "
            f"model={model_name} "
            f"budget={budget} "
            f"memory_recall="
            f"{row['memory_recall']:.6f} "
            f"token_accuracy="
            f"{row['token_accuracy']:.6f}",
            flush=True,
        )


write_csv(
    RESULTS_PATH,
    rows,
)

frame = pd.DataFrame(rows)

comparisons = []

for policy in (
    "gru_uniform_cache",
    "gru_reservoir_cache",
):
    for budget in TARGET_BUDGETS:
        budgetmem_row = frame[
            (frame["model"] == "budgetmem_r")
            & (
                frame["memory_budget"]
                == budget
            )
        ].iloc[0]

        policy_row = frame[
            (frame["model"] == policy)
            & (
                frame["memory_budget"]
                == budget
            )
        ].iloc[0]

        gain = float(
            budgetmem_row["memory_recall"]
            - policy_row["memory_recall"]
        )

        comparisons.append(
            {
                "policy": policy,
                "budget": budget,
                "budgetmem_memory_recall": float(
                    budgetmem_row[
                        "memory_recall"
                    ]
                ),
                "policy_memory_recall": float(
                    policy_row[
                        "memory_recall"
                    ]
                ),
                "absolute_gain": gain,
                "clear_win": (
                    gain >= MINIMUM_GAIN
                ),
            }
        )


comparison_frame = pd.DataFrame(
    comparisons
)

policy_summary = (
    comparison_frame
    .groupby(
        "policy",
        as_index=False,
    )
    .agg(
        matched_budgets=(
            "budget",
            "size",
        ),
        mean_gain=(
            "absolute_gain",
            "mean",
        ),
        minimum_gain=(
            "absolute_gain",
            "min",
        ),
        clear_win_rate=(
            "clear_win",
            "mean",
        ),
    )
)

policy_summary[
    "qualifies"
] = (
    (
        policy_summary[
            "minimum_gain"
        ]
        >= MINIMUM_GAIN
    )
    & (
        policy_summary[
            "clear_win_rate"
        ]
        == 1.0
    )
)

budgetmem_rows = frame[
    frame["model"] == "budgetmem_r"
]

stability_pass = bool(
    budgetmem_rows[
        "stability_pass"
    ].astype(bool).all()
)

budget_pass = bool(
    budgetmem_rows[
        "budget_pass"
    ].astype(bool).all()
)

resource_pass = bool(
    budgetmem_rows[
        "resource_measurement_pass"
    ].astype(bool).all()
)

write_frequency = float(
    budgetmem_rows[
        "write_frequency"
    ].mean()
)

write_pass = bool(
    0.01
    <= write_frequency
    <= 0.95
)

qualified_policy_count = int(
    policy_summary[
        "qualifies"
    ].sum()
)

decision = (
    "TARGETED_GO"
    if (
        qualified_policy_count >= 2
        and stability_pass
        and budget_pass
        and resource_pass
        and write_pass
    )
    else "TARGETED_NO_GO"
)

summary_payload = {
    "schema_version": "1.0",
    "config_path": str(
        CONFIG_PATH.relative_to(ROOT)
    ),
    "effective_config_path": str(
        effective_path.relative_to(ROOT)
    ),
    "config_sha256": config_sha256,
    "training_records": [
        asdict(record)
        for record in training_records
    ],
    "result_count": len(rows),
    "results_csv": str(
        RESULTS_PATH.relative_to(ROOT)
    ),
}

SUMMARY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_PATH.write_text(
    json.dumps(
        summary_payload,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

decision_payload = {
    "decision": decision,
    "target_task": TARGET_TASK,
    "target_sequence_length": (
        TARGET_LENGTH
    ),
    "target_budgets": (
        TARGET_BUDGETS
    ),
    "minimum_required_gain": (
        MINIMUM_GAIN
    ),
    "qualified_policy_count": (
        qualified_policy_count
    ),
    "criteria": {
        "stability_pass": (
            stability_pass
        ),
        "budget_pass": (
            budget_pass
        ),
        "resource_pass": (
            resource_pass
        ),
        "write_frequency_pass": (
            write_pass
        ),
        "write_frequency": (
            write_frequency
        ),
        "outperforms_both_policies": (
            qualified_policy_count >= 2
        ),
    },
    "policy_summary": (
        policy_summary.to_dict(
            orient="records"
        )
    ),
    "matched_comparisons": (
        comparison_frame.to_dict(
            orient="records"
        )
    ),
}

DECISION_JSON.write_text(
    json.dumps(
        decision_payload,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

lines = [
    "SECTION 15 ASSOCIATIVE-RECALL CORE-REPAIR SCREEN",
    "================================================",
    f"Decision: {decision}",
    "",
    f"Task: {TARGET_TASK}",
    f"Sequence length: {TARGET_LENGTH}",
    f"Budgets: {TARGET_BUDGETS}",
    (
        "Required recall gain over each policy "
        f"at each budget: {MINIMUM_GAIN:.4f}"
    ),
    "",
    "Criteria:",
    f"- Stability: {stability_pass}",
    f"- Strict budget: {budget_pass}",
    f"- Resource measurements: {resource_pass}",
    (
        "- Nontrivial write frequency: "
        f"{write_pass} "
        f"({write_frequency:.6f})"
    ),
    (
        "- Policies clearly outperformed: "
        f"{qualified_policy_count}/2"
    ),
    "",
    "Matched comparisons:",
    comparison_frame.to_string(
        index=False
    ),
    "",
    "Policy summary:",
    policy_summary.to_string(
        index=False
    ),
    "",
]

if decision == "TARGETED_GO":
    lines.extend(
        [
            "Interpretation:",
            (
                "The repaired implementation passed the "
                "associative-recall screen."
            ),
            (
                "The complete Section 15 pilot may now "
                "be rerun with isolated artifacts."
            ),
        ]
    )
else:
    lines.extend(
        [
            "Interpretation:",
            (
                "The implementation repair did not yet "
                "satisfy the targeted gate."
            ),
            (
                "Do not start another complete 72-cell "
                "pilot."
            ),
        ]
    )

DECISION_TXT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print()
print(
    DECISION_TXT.read_text(
        encoding="utf-8"
    )
)

raise SystemExit(
    0
    if decision == "TARGETED_GO"
    else 2
)
PY

# ============================================================
# RUN THE TARGETED SCREEN
# ============================================================

rm -rf outputs/assoc_core_repair_screen

rm -f \
    "$RESULTS" \
    "$DECISION_JSON" \
    "$DECISION_TXT" \
    reports/evidence/assoc_core_repair_screen_summary.json \
    reports/assoc_core_repair_screen_report.md \
    "$LOG_FILE"

echo
echo "============================================================"
echo " Running Isolated Associative-Recall Screen"
echo "============================================================"
echo
echo "This run does not overwrite the completed pilot artifacts."
echo

set +e

"$PYTHON" "$SCREEN_RUNNER" \
    2>&1 | tee "$LOG_FILE"

SCREEN_STATUS="${PIPESTATUS[0]}"

set -e

echo

if [[ "$SCREEN_STATUS" -eq 0 ]]; then
    echo "============================================================"
    echo " TARGETED DECISION: GO"
    echo "============================================================"
    echo
    echo "The core repair passed the associative-recall screen."
    echo
    echo "Do not commit yet."
    echo "Do not run Scripts 20 or 21."
    echo
    echo "Review:"
    echo "  $DECISION_TXT"
    echo "  $RESULTS"
    echo
    echo "The next operation is one isolated full Section 15 rerun."
    exit 0
fi

if [[ "$SCREEN_STATUS" -eq 2 ]]; then
    echo "============================================================"
    echo " TARGETED DECISION: NO-GO"
    echo "============================================================"
    echo
    echo "The repair was tested, but the targeted gate did not pass."
    echo "The complete pilot was not started."
    echo
    echo "Review:"
    echo "  $DECISION_TXT"
    echo "  $RESULTS"
    echo "  $LOG_FILE"
    echo
    echo "Section 14.11 remains incomplete."
    exit 2
fi

echo "ERROR: The targeted screen failed with status:"
echo "  $SCREEN_STATUS"
echo
echo "Inspect:"
echo "  $LOG_FILE"

exit "$SCREEN_STATUS"
