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
