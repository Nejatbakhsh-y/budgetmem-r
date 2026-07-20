#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ============================================================
# SECTION 14.11 CONFIGURATION
# ============================================================

# Sequence length 1024 is the long-range pilot condition.
LONG_SEQ_MIN="${LONG_SEQ_MIN:-1024}"

# BudgetMem-R must exceed a comparator by at least 0.02
# absolute recall to count as a clear win.
MIN_ABS_GAIN="${MIN_ABS_GAIN:-0.02}"

# BudgetMem-R must clearly win at least 67% of matched cells
# for a comparator to count as outperformed.
MIN_WIN_RATE="${MIN_WIN_RATE:-0.67}"

# The pilot has three tasks and two budgets, normally giving
# six matched cells per policy. Require at least four.
MIN_MATCHED_CELLS="${MIN_MATCHED_CELLS:-4}"

# Section 14.11 requires at least two policies.
REQUIRED_POLICY_WINS="${REQUIRED_POLICY_WINS:-2}"

# Optional exact result file.
#
# Example:
# RESULT_FILE="reports/tables/section15_pilot_summary.csv" \
# bash 17_section15_final_go_decision.sh
RESULT_FILE="${RESULT_FILE:-}"

# Git controls.
AUTO_COMMIT="${AUTO_COMMIT:-1}"
AUTO_PUSH="${AUTO_PUSH:-1}"

EVIDENCE_DIR="reports/evidence"
TABLE_DIR="reports/tables"

EVIDENCE_FILE="$EVIDENCE_DIR/section15_final_go_decision.txt"
JSON_FILE="$EVIDENCE_DIR/section15_final_go_decision.json"
SUMMARY_FILE="$TABLE_DIR/section15_final_go_comparison.csv"
DETAIL_FILE="$TABLE_DIR/section15_final_go_matched_cells.csv"

mkdir -p "$EVIDENCE_DIR" "$TABLE_DIR"

echo "============================================================"
echo " Section 15 Final GO/NO-GO Decision"
echo "============================================================"
echo "Repository:                         $ROOT"
echo "Long-range minimum sequence length: $LONG_SEQ_MIN"
echo "Minimum absolute recall gain:       $MIN_ABS_GAIN"
echo "Minimum policy win rate:            $MIN_WIN_RATE"
echo "Minimum matched cells per policy:   $MIN_MATCHED_CELLS"
echo "Required policies outperformed:     $REQUIRED_POLICY_WINS"
echo

set +e

LONG_SEQ_MIN="$LONG_SEQ_MIN" \
MIN_ABS_GAIN="$MIN_ABS_GAIN" \
MIN_WIN_RATE="$MIN_WIN_RATE" \
MIN_MATCHED_CELLS="$MIN_MATCHED_CELLS" \
REQUIRED_POLICY_WINS="$REQUIRED_POLICY_WINS" \
RESULT_FILE="$RESULT_FILE" \
EVIDENCE_FILE="$EVIDENCE_FILE" \
JSON_FILE="$JSON_FILE" \
SUMMARY_FILE="$SUMMARY_FILE" \
DETAIL_FILE="$DETAIL_FILE" \
python - <<'PY'
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ImportError as exc:
    print(
        "ERROR: pandas is required. Activate the project virtual "
        "environment and install the project dependencies."
    )
    raise SystemExit(3) from exc


ROOT = Path.cwd()

LONG_SEQ_MIN = int(os.environ["LONG_SEQ_MIN"])
MIN_ABS_GAIN = float(os.environ["MIN_ABS_GAIN"])
MIN_WIN_RATE = float(os.environ["MIN_WIN_RATE"])
MIN_MATCHED_CELLS = int(os.environ["MIN_MATCHED_CELLS"])
REQUIRED_POLICY_WINS = int(os.environ["REQUIRED_POLICY_WINS"])
RESULT_FILE = os.environ.get("RESULT_FILE", "").strip()

EVIDENCE_FILE = ROOT / os.environ["EVIDENCE_FILE"]
JSON_FILE = ROOT / os.environ["JSON_FILE"]
SUMMARY_FILE = ROOT / os.environ["SUMMARY_FILE"]
DETAIL_FILE = ROOT / os.environ["DETAIL_FILE"]

GENERATED_FILES = {
    EVIDENCE_FILE.resolve(),
    JSON_FILE.resolve(),
    SUMMARY_FILE.resolve(),
    DETAIL_FILE.resolve(),
}


def normalize(value: object) -> str:
    """Normalize names for reliable matching."""
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def find_column(
    columns: Iterable[str],
    aliases: Iterable[str],
) -> str | None:
    lookup = {
        normalize(column): column
        for column in columns
    }

    for alias in aliases:
        if alias in lookup:
            return lookup[alias]

    return None


def numeric_series(series: pd.Series) -> pd.Series:
    """Convert numeric values and strings such as B16 or budget_32."""
    converted = pd.to_numeric(series, errors="coerce")
    missing = converted.isna()

    if missing.any():
        extracted = (
            series.astype(str)
            .str.extract(
                r"([-+]?[0-9]*\.?[0-9]+)",
                expand=False,
            )
        )

        converted.loc[missing] = pd.to_numeric(
            extracted.loc[missing],
            errors="coerce",
        )

    return converted


def load_table(path: Path) -> pd.DataFrame | None:
    """Load supported result formats."""
    try:
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(path)

        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)

        if suffix == ".json":
            payload = json.loads(
                path.read_text(encoding="utf-8")
            )

            if isinstance(payload, list):
                return pd.DataFrame(payload)

            if isinstance(payload, dict):
                for key in (
                    "results",
                    "records",
                    "rows",
                    "data",
                    "metrics",
                    "summary",
                ):
                    value = payload.get(key)

                    if isinstance(value, list):
                        return pd.DataFrame(value)

                return pd.json_normalize(payload)

        if suffix == ".parquet":
            return pd.read_parquet(path)

    except Exception:
        return None

    return None


def identify_model(value: object) -> str | None:
    """Map project model labels to standard comparison names."""
    text = normalize(value)

    if "budgetmem" in text or "budget_mem" in text:
        return "budgetmem_r"

    if "uniform" in text:
        return "uniform_cache"

    if "reservoir" in text:
        return "reservoir_cache"

    if "fifo" in text:
        return "fifo_cache"

    if "lru" in text:
        return "lru_cache"

    if (
        "most_recent" in text
        or "recent_state" in text
        or "recency" in text
    ):
        return "most_recent_cache"

    if "novelty" in text:
        return "novelty_policy"

    if "surprise" in text:
        return "surprise_policy"

    if (
        "random_replacement" in text
        or "random_policy" in text
        or text == "random"
    ):
        return "random_policy"

    return None


MODEL_ALIASES = [
    "model",
    "model_name",
    "method",
    "architecture",
    "system",
    "baseline",
]

POLICY_ALIASES = [
    "memory_policy",
    "policy",
    "cache_policy",
    "replacement_policy",
    "memory_type",
]

TASK_ALIASES = [
    "task",
    "task_name",
    "dataset",
    "benchmark_task",
]

SEQUENCE_ALIASES = [
    "sequence_length",
    "seq_length",
    "seq_len",
    "context_length",
    "length",
]

BUDGET_ALIASES = [
    "memory_budget",
    "budget",
    "cache_size",
    "memory_size",
    "num_slots",
    "slots",
]

SEED_ALIASES = [
    "seed",
    "random_seed",
]

METRIC_ALIASES = [
    "long_range_recall",
    "long_range_retrieval_recall",
    "long_range_accuracy",
    "retrieval_recall",
    "memory_recall",
    "relevant_state_retention_rate",
    "successful_long_range_retrieval_rate",
    "recall",
]


def standardize_long(
    dataframe: pd.DataFrame,
    source: str,
) -> pd.DataFrame | None:
    """Convert a long-format result table to the required schema."""
    model_column = find_column(
        dataframe.columns,
        MODEL_ALIASES,
    )

    policy_column = find_column(
        dataframe.columns,
        POLICY_ALIASES,
    )

    task_column = find_column(
        dataframe.columns,
        TASK_ALIASES,
    )

    sequence_column = find_column(
        dataframe.columns,
        SEQUENCE_ALIASES,
    )

    budget_column = find_column(
        dataframe.columns,
        BUDGET_ALIASES,
    )

    seed_column = find_column(
        dataframe.columns,
        SEED_ALIASES,
    )

    metric_column = find_column(
        dataframe.columns,
        METRIC_ALIASES,
    )

    if not all(
        (
            model_column,
            task_column,
            sequence_column,
            budget_column,
            metric_column,
        )
    ):
        return None

    model_text = dataframe[model_column].astype(str)

    if (
        policy_column
        and policy_column != model_column
    ):
        model_text = (
            model_text
            + " "
            + dataframe[policy_column].astype(str)
        )

    result = pd.DataFrame(
        {
            "model_raw": model_text,
            "task": dataframe[task_column].astype(str),
            "sequence_length": numeric_series(
                dataframe[sequence_column]
            ),
            "budget": numeric_series(
                dataframe[budget_column]
            ),
            "seed": (
                dataframe[seed_column].astype(str)
                if seed_column
                else "unspecified"
            ),
            "recall": numeric_series(
                dataframe[metric_column]
            ),
            "metric": metric_column,
            "source_file": source,
        }
    )

    result["model"] = result[
        "model_raw"
    ].map(identify_model)

    return result.dropna(
        subset=[
            "model",
            "sequence_length",
            "budget",
            "recall",
        ]
    )


def standardize_wide(
    dataframe: pd.DataFrame,
    source: str,
) -> pd.DataFrame | None:
    """Convert a wide-format result table to the required schema."""
    task_column = find_column(
        dataframe.columns,
        TASK_ALIASES,
    )

    sequence_column = find_column(
        dataframe.columns,
        SEQUENCE_ALIASES,
    )

    budget_column = find_column(
        dataframe.columns,
        BUDGET_ALIASES,
    )

    seed_column = find_column(
        dataframe.columns,
        SEED_ALIASES,
    )

    if not all(
        (
            task_column,
            sequence_column,
            budget_column,
        )
    ):
        return None

    model_columns: list[tuple[str, str]] = []

    for column in dataframe.columns:
        model = identify_model(column)

        if model is not None:
            model_columns.append(
                (column, model)
            )

    if not model_columns:
        return None

    pieces: list[pd.DataFrame] = []

    for column, model in model_columns:
        piece = pd.DataFrame(
            {
                "model_raw": column,
                "model": model,
                "task": dataframe[task_column].astype(str),
                "sequence_length": numeric_series(
                    dataframe[sequence_column]
                ),
                "budget": numeric_series(
                    dataframe[budget_column]
                ),
                "seed": (
                    dataframe[seed_column].astype(str)
                    if seed_column
                    else "unspecified"
                ),
                "recall": numeric_series(
                    dataframe[column]
                ),
                "metric": column,
                "source_file": source,
            }
        )

        pieces.append(piece)

    result = pd.concat(
        pieces,
        ignore_index=True,
    )

    return result.dropna(
        subset=[
            "model",
            "sequence_length",
            "budget",
            "recall",
        ]
    )


def usable_models(
    frame: pd.DataFrame,
) -> set[str]:
    return set(
        frame["model"]
        .dropna()
        .astype(str)
    )


# ============================================================
# FIND PILOT RESULTS
# ============================================================

if RESULT_FILE:
    explicit = Path(
        RESULT_FILE
    ).expanduser()

    if not explicit.is_absolute():
        explicit = ROOT / explicit

    candidate_files = [explicit]

else:
    candidate_files: list[Path] = []

    for directory_name in (
        "reports",
        "results",
        "outputs",
        "artifacts",
    ):
        directory = ROOT / directory_name

        if not directory.exists():
            continue

        for suffix in (
            "*.csv",
            "*.json",
            "*.jsonl",
            "*.parquet",
        ):
            candidate_files.extend(
                directory.rglob(suffix)
            )

    candidate_files = sorted(
        {
            path.resolve()
            for path in candidate_files
            if path.resolve() not in GENERATED_FILES
        },
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


if not candidate_files:
    print(
        "ERROR: No CSV, JSON, JSONL, or Parquet "
        "pilot result files were found."
    )
    raise SystemExit(3)


usable_by_file: list[
    tuple[Path, pd.DataFrame]
] = []


for path in candidate_files:
    if (
        not path.exists()
        or path.resolve() in GENERATED_FILES
    ):
        continue

    dataframe = load_table(path)

    if dataframe is None or dataframe.empty:
        continue

    source = (
        str(path.relative_to(ROOT))
        if path.is_relative_to(ROOT)
        else str(path)
    )

    standardized = standardize_long(
        dataframe,
        source,
    )

    if (
        standardized is None
        or standardized.empty
    ):
        standardized = standardize_wide(
            dataframe,
            source,
        )

    if (
        standardized is None
        or standardized.empty
    ):
        continue

    models = usable_models(
        standardized
    )

    if (
        "budgetmem_r" in models
        and len(
            models - {"budgetmem_r"}
        ) >= 1
    ):
        usable_by_file.append(
            (path, standardized)
        )


if not usable_by_file:
    print(
        "ERROR: No usable pilot result table was found."
    )
    print()
    print(
        "Required information:"
    )
    print(
        "- model or policy name"
    )
    print(
        "- task"
    )
    print(
        "- sequence length"
    )
    print(
        "- memory budget"
    )
    print(
        "- long-range recall"
    )
    print()
    print(
        "Specify the exact result file when necessary:"
    )
    print()
    print(
        'RESULT_FILE="reports/tables/'
        'section15_pilot_summary.csv" '
        'bash 17_section15_final_go_decision.sh'
    )

    raise SystemExit(3)


# Prefer the newest single result table containing BudgetMem-R
# and at least two comparison policies.
selected_frames: list[
    pd.DataFrame
] = []


for path, frame in usable_by_file:
    comparator_count = len(
        usable_models(frame)
        - {"budgetmem_r"}
    )

    if comparator_count >= REQUIRED_POLICY_WINS:
        selected_frames = [frame]
        break


# If results are distributed across several files, combine them.
if not selected_frames:
    selected_frames = [
        frame
        for _, frame in usable_by_file
    ]


all_results = pd.concat(
    selected_frames,
    ignore_index=True,
)


# ============================================================
# FILTER LONG-RANGE RESULTS
# ============================================================

all_results = all_results[
    all_results["sequence_length"]
    >= LONG_SEQ_MIN
].copy()


if all_results.empty:
    print(
        "ERROR: No rows were found with "
        f"sequence_length >= {LONG_SEQ_MIN}."
    )
    raise SystemExit(3)


all_results["task"] = all_results[
    "task"
].map(normalize)

all_results["sequence_length"] = (
    all_results["sequence_length"]
    .astype(int)
)

all_results["budget"] = (
    all_results["budget"]
    .astype(int)
)


# Convert percentages such as 87.5 into proportions such as 0.875.
if (
    all_results["recall"].median() > 1.5
    and all_results["recall"].max() <= 100
):
    all_results["recall"] = (
        all_results["recall"] / 100.0
    )


# Avoid counting copied or duplicated result rows more than once.
all_results = all_results.drop_duplicates(
    subset=[
        "model",
        "task",
        "sequence_length",
        "budget",
        "seed",
        "recall",
    ]
)


# Average repeated observations within exactly matched cells.
matching_keys = [
    "task",
    "sequence_length",
    "budget",
    "seed",
]

aggregated = (
    all_results
    .groupby(
        matching_keys + ["model"],
        as_index=False,
    )["recall"]
    .mean()
)


budgetmem = (
    aggregated[
        aggregated["model"]
        == "budgetmem_r"
    ]
    .drop(columns="model")
    .rename(
        columns={
            "recall": "budgetmem_recall",
        }
    )
)


policy_order = [
    "uniform_cache",
    "reservoir_cache",
    "fifo_cache",
    "lru_cache",
    "most_recent_cache",
    "novelty_policy",
    "surprise_policy",
    "random_policy",
]


summary_rows: list[
    dict[str, object]
] = []

detail_frames: list[
    pd.DataFrame
] = []


# ============================================================
# MATCH BUDGETMEM-R AGAINST EACH POLICY
# ============================================================

for policy in policy_order:
    policy_frame = (
        aggregated[
            aggregated["model"]
            == policy
        ]
        .drop(columns="model")
        .rename(
            columns={
                "recall": "policy_recall",
            }
        )
    )

    if policy_frame.empty:
        continue

    matched = budgetmem.merge(
        policy_frame,
        on=matching_keys,
        how="inner",
    )

    if matched.empty:
        continue

    matched["policy"] = policy

    matched["absolute_gain"] = (
        matched["budgetmem_recall"]
        - matched["policy_recall"]
    )

    matched["clear_win"] = (
        matched["absolute_gain"]
        >= MIN_ABS_GAIN
    )

    detail_frames.append(matched)

    matched_count = int(
        len(matched)
    )

    clear_wins = int(
        matched["clear_win"].sum()
    )

    win_rate = (
        clear_wins / matched_count
    )

    mean_gain = float(
        matched["absolute_gain"].mean()
    )

    qualifies = bool(
        matched_count >= MIN_MATCHED_CELLS
        and mean_gain >= MIN_ABS_GAIN
        and win_rate >= MIN_WIN_RATE
    )

    summary_rows.append(
        {
            "policy": policy,
            "matched_same_budget_cells": matched_count,
            "budgetmem_mean_recall": float(
                matched[
                    "budgetmem_recall"
                ].mean()
            ),
            "policy_mean_recall": float(
                matched[
                    "policy_recall"
                ].mean()
            ),
            "mean_absolute_gain": mean_gain,
            "minimum_absolute_gain": float(
                matched[
                    "absolute_gain"
                ].min()
            ),
            "clear_wins": clear_wins,
            "clear_win_rate": win_rate,
            "qualifies_as_outperformed": qualifies,
        }
    )


summary = pd.DataFrame(
    summary_rows
)


if summary.empty:
    print(
        "ERROR: No matched same-task, same-sequence, "
        "same-budget, same-seed comparisons were found."
    )
    raise SystemExit(3)


summary = summary.sort_values(
    [
        "qualifies_as_outperformed",
        "mean_absolute_gain",
    ],
    ascending=[
        False,
        False,
    ],
)


qualified = summary[
    summary[
        "qualifies_as_outperformed"
    ]
]


qualified_count = int(
    len(qualified)
)


decision = (
    "GO"
    if qualified_count
    >= REQUIRED_POLICY_WINS
    else "NO-GO"
)


# ============================================================
# SAVE TABLES
# ============================================================

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


if detail_frames:
    details = pd.concat(
        detail_frames,
        ignore_index=True,
    )

    details = details[
        [
            "policy",
            "task",
            "sequence_length",
            "budget",
            "seed",
            "budgetmem_recall",
            "policy_recall",
            "absolute_gain",
            "clear_win",
        ]
    ].sort_values(
        [
            "policy",
            "task",
            "sequence_length",
            "budget",
            "seed",
        ]
    )

    details.to_csv(
        DETAIL_FILE,
        index=False,
    )


source_files = sorted(
    set(
        all_results[
            "source_file"
        ].astype(str)
    )
)


qualified_names = (
    qualified["policy"]
    .astype(str)
    .tolist()
)


# ============================================================
# WRITE FINAL HUMAN-READABLE EVIDENCE
# ============================================================

lines = [
    "SECTION 15 FINAL GO/NO-GO DECISION",
    "==================================",
    f"Decision: {decision}",
    "",
    "Operational gate:",
    (
        "- Long-range sequence length: "
        f">= {LONG_SEQ_MIN}"
    ),
    (
        "- Comparisons matched by task, sequence length, "
        "memory budget, and seed"
    ),
    (
        "- Minimum absolute recall gain per clear win: "
        f"{MIN_ABS_GAIN:.4f}"
    ),
    (
        "- Minimum policy-level clear-win rate: "
        f"{MIN_WIN_RATE:.2%}"
    ),
    (
        "- Minimum matched cells per policy: "
        f"{MIN_MATCHED_CELLS}"
    ),
    (
        "- Required outperformed fixed-policy baselines: "
        f"{REQUIRED_POLICY_WINS}"
    ),
    (
        "- Policies satisfying the gate: "
        f"{qualified_count}"
    ),
    "",
    "Policy comparisons:",
]


for row in summary.to_dict(
    orient="records"
):
    lines.append(
        (
            "- {policy}: "
            "BudgetMem-R={bm:.4f}, "
            "policy={base:.4f}, "
            "mean gain={gain:+.4f}, "
            "clear wins={wins}/{cells} "
            "({rate:.1%}), "
            "qualifies={qualifies}"
        ).format(
            policy=row["policy"],
            bm=row[
                "budgetmem_mean_recall"
            ],
            base=row[
                "policy_mean_recall"
            ],
            gain=row[
                "mean_absolute_gain"
            ],
            wins=row[
                "clear_wins"
            ],
            cells=row[
                "matched_same_budget_cells"
            ],
            rate=row[
                "clear_win_rate"
            ],
            qualifies=(
                "YES"
                if row[
                    "qualifies_as_outperformed"
                ]
                else "NO"
            ),
        )
    )


lines.extend(
    [
        "",
        "Qualified policies:",
        (
            "- "
            + (
                ", ".join(
                    qualified_names
                )
                if qualified_names
                else "None"
            )
        ),
        "",
        "Interpretation:",
        (
            (
                "GO: BudgetMem-R clearly outperformed "
                f"at least {REQUIRED_POLICY_WINS} "
                "fixed-policy memory baselines on "
                "long-range recall at matched memory budgets."
            )
            if decision == "GO"
            else
            (
                "NO-GO: BudgetMem-R did not clearly "
                f"outperform at least "
                f"{REQUIRED_POLICY_WINS} fixed-policy "
                "memory baselines. Return to Sections "
                "14.9 and 14.10, tune the model, rerun "
                "the pilot, and rerun this gate."
            )
        ),
        "",
        "Source result files:",
    ]
)


lines.extend(
    f"- {source}"
    for source in source_files
)


EVIDENCE_FILE.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)


# ============================================================
# WRITE MACHINE-READABLE EVIDENCE
# ============================================================

payload = {
    "decision": decision,
    "criteria": {
        "long_sequence_minimum": LONG_SEQ_MIN,
        "minimum_absolute_recall_gain": MIN_ABS_GAIN,
        "minimum_policy_clear_win_rate": MIN_WIN_RATE,
        "minimum_matched_cells_per_policy": MIN_MATCHED_CELLS,
        "required_outperformed_policies": REQUIRED_POLICY_WINS,
    },
    "qualified_policy_count": qualified_count,
    "qualified_policies": qualified_names,
    "comparisons": summary.to_dict(
        orient="records"
    ),
    "source_files": source_files,
}


JSON_FILE.write_text(
    json.dumps(
        payload,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    EVIDENCE_FILE.read_text(
        encoding="utf-8"
    )
)

print(
    "Generated files:"
)

print(
    f"- {EVIDENCE_FILE.relative_to(ROOT)}"
)

print(
    f"- {JSON_FILE.relative_to(ROOT)}"
)

print(
    f"- {SUMMARY_FILE.relative_to(ROOT)}"
)

print(
    f"- {DETAIL_FILE.relative_to(ROOT)}"
)


# 0 = GO
# 2 = valid analysis but NO-GO
# 3 = missing or incompatible data
raise SystemExit(
    0
    if decision == "GO"
    else 2
)
PY

PY_STATUS=$?

set -e


# ============================================================
# HANDLE RESULT
# ============================================================

if [[ "$PY_STATUS" -eq 3 ]]; then
    echo
    echo "The decision could not be calculated because a compatible"
    echo "pilot result table was not found."
    echo
    echo "Locate possible result files with:"
    echo
    echo 'find reports results outputs artifacts -type f \'
    echo '  \( -name "*.csv" -o -name "*.json" -o \'
    echo '     -name "*.jsonl" -o -name "*.parquet" \) \'
    echo '  2>/dev/null | sort'
    echo
    echo "Then rerun with the exact file, for example:"
    echo
    echo 'RESULT_FILE="reports/tables/section15_pilot_summary.csv" \'
    echo 'bash 17_section15_final_go_decision.sh'
    echo
    exit 3
fi


if [[ "$PY_STATUS" -eq 2 ]]; then
    echo
    echo "============================================================"
    echo " Final decision: NO-GO"
    echo "============================================================"
    echo
    echo "The evidence files were saved."
    echo
    echo "Section 14.11 is not complete because BudgetMem-R did not"
    echo "clearly outperform at least two comparison policies."
    echo
    echo "Return to Sections 14.9 and 14.10, tune BudgetMem-R,"
    echo "rerun the pilot, and rerun this same automation."
    echo
    echo "Evidence:"
    echo "$EVIDENCE_FILE"
    echo
    exit 2
fi


echo
echo "============================================================"
echo " Final decision: GO"
echo "============================================================"


# ============================================================
# COMMIT AND PUSH ONLY AFTER GO
# ============================================================

if [[ "$AUTO_COMMIT" == "1" ]]; then
    echo
    echo "Saving the final GO evidence to Git..."

    git add \
        "$SCRIPT_PATH" \
        "$EVIDENCE_FILE" \
        "$JSON_FILE" \
        "$SUMMARY_FILE" \
        "$DETAIL_FILE"

    if git diff --cached --quiet; then
        echo "No new files required a Git commit."
    else
        git commit -m \
            "docs: record Section 15 final GO decision"
    fi

    if [[ "$AUTO_PUSH" == "1" ]]; then
        BRANCH="$(git branch --show-current)"

        if (
            [[ -n "$BRANCH" ]]
            && git remote get-url origin >/dev/null 2>&1
        ); then
            git push -u origin "$BRANCH"
            echo "Pushed branch: $BRANCH"
        else
            echo
            echo "Git push was skipped because the current branch"
            echo "or the origin remote was unavailable."
        fi
    fi
fi


echo
echo "============================================================"
echo " Section 14.11 is complete"
echo "============================================================"
echo
echo "Final evidence:"
echo "$EVIDENCE_FILE"
echo
echo "Comparison summary:"
echo "$SUMMARY_FILE"
echo
echo "Git status:"
git status --short
echo
