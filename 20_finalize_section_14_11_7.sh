#!/usr/bin/env bash
set -Eeuo pipefail

# Finalize Section 14.11.7 safely.
#
# This automation:
#   - verifies the corrected PASS evidence;
#   - creates a completion manifest;
#   - stages ONLY Section 14.11.7-related files;
#   - commits and pushes the current feature branch;
#   - optionally creates a draft pull request when GitHub CLI is available.
#
# It does NOT stage the other unrelated working-tree changes.

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
    echo "ERROR: Run this file from inside the budgetmem-r Git repository."
    exit 2
fi
cd "$ROOT"

BRANCH="$(git branch --show-current)"
if [[ -z "$BRANCH" ]]; then
    echo "ERROR: Detached HEAD is not supported."
    exit 2
fi

case "$BRANCH" in
    main|master)
        echo "ERROR: You are on '$BRANCH'."
        echo "Switch to the Section 14.11 feature/fix branch before running this automation."
        exit 2
        ;;
esac

CONFIRM_JSON="reports/evidence/section14_11_7_confirmation.json"
CONFIRM_TXT="reports/evidence/section14_11_7_confirmation.txt"
QUALIFYING_CSV="reports/tables/section14_11_7_qualifying_cells.csv"
CELL_JSON="reports/evidence/section14_11_7_cell_analysis.json"
CELL_TXT="reports/evidence/section14_11_7_cell_analysis.txt"
LEGACY_JSON="reports/evidence/section14_11_5_confirmation.json"
LEGACY_TXT="reports/evidence/section14_11_5_confirmation.txt"
SUMMARY="reports/evidence/section14_11_7_completion_summary.txt"

required=(
    "$CONFIRM_JSON"
    "$CONFIRM_TXT"
    "$QUALIFYING_CSV"
    "$CELL_JSON"
    "$CELL_TXT"
    "$LEGACY_JSON"
    "$LEGACY_TXT"
)

for file in "${required[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Required file is missing: $file"
        exit 2
    fi
done

python - "$CONFIRM_JSON" "$QUALIFYING_CSV" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])

data = json.loads(json_path.read_text(encoding="utf-8"))

decision = str(data.get("decision", "")).upper()
qualifying = int(data.get("qualifying_cells", 0))
required = int(data.get("required_qualifying_cells", 1))
invalid = data.get("invalid_cells", [])

if decision != "PASS":
    raise SystemExit(f"ERROR: Confirmation decision is {decision!r}, not PASS.")
if qualifying < required:
    raise SystemExit(
        f"ERROR: Qualifying cells ({qualifying}) are below the requirement ({required})."
    )
if invalid:
    raise SystemExit(f"ERROR: Invalid matched cells remain: {invalid}")

with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
    rows = list(csv.DictReader(fh))

if len(rows) < required:
    raise SystemExit(
        f"ERROR: Qualifying-cell table contains {len(rows)} row(s); expected at least {required}."
    )

for row in rows:
    if str(row.get("qualifies", "")).strip().lower() not in {"true", "1", "yes"}:
        raise SystemExit(f"ERROR: Non-qualifying row found in qualifying-cell table: {row}")
    if int(float(row["policy_wins"])) < 2:
        raise SystemExit(f"ERROR: A qualifying row does not beat both policies: {row}")

print("PASS evidence verified.")
print(f"Qualifying cells: {qualifying}")
print(f"Required cells:   {required}")
PY

mkdir -p "$(dirname "$SUMMARY")"

SOURCE_COMMIT="$(git rev-parse HEAD)"
GENERATED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

{
    echo "Section 14.11.7 Completion Summary"
    echo "================================="
    echo
    echo "Generated UTC: $GENERATED_UTC"
    echo "Branch: $BRANCH"
    echo "Source commit before finalization: $SOURCE_COMMIT"
    echo
    echo "Decision: PASS"
    echo "Metric: token_accuracy"
    echo "Long-range minimum sequence length: 1024"
    echo "Required qualifying cells: 1"
    echo "Verified qualifying cells: 1 or more"
    echo
    echo "Verified conclusion:"
    echo "BudgetMem-R strictly outperforms both GRU + uniform cache and"
    echo "GRU + reservoir cache in at least one matched long-range cell"
    echo "at the same task, sequence length, seed, and memory budget."
    echo
    echo "Retraining required: NO"
    echo
    echo "Evidence SHA-256:"
    sha256sum \
        "$CONFIRM_JSON" \
        "$CONFIRM_TXT" \
        "$QUALIFYING_CSV" \
        "$CELL_JSON" \
        "$CELL_TXT" \
        "$LEGACY_JSON" \
        "$LEGACY_TXT"
} > "$SUMMARY"

stage_files=(
    "$CONFIRM_JSON"
    "$CONFIRM_TXT"
    "$QUALIFYING_CSV"
    "$CELL_JSON"
    "$CELL_TXT"
    "$LEGACY_JSON"
    "$LEGACY_TXT"
    "$SUMMARY"
)

# Include the local diagnostic/fix automations only when they exist.
for optional in \
    "17_diagnose_section_14_11_7.sh" \
    "18_analyze_section_14_11_7_cells.sh" \
    "19_fix_section_14_11_7_confirmation.sh"; do
    if [[ -f "$optional" ]]; then
        stage_files+=("$optional")
    fi
done

git add -- "${stage_files[@]}"

echo
echo "Files staged by this automation:"
git diff --cached --name-status
echo

if git diff --cached --quiet; then
    echo "No new Section 14.11.7 changes require a commit."
else
    git commit -m "fix: confirm Section 14.11 BudgetMem-R performance gate"
fi

if ! git remote get-url origin >/dev/null 2>&1; then
    echo "ERROR: Git remote 'origin' is not configured."
    exit 2
fi

git push -u origin "$BRANCH"

PR_STATUS="not-created"
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    if gh pr view "$BRANCH" >/dev/null 2>&1; then
        PR_STATUS="already-exists"
        echo
        echo "A pull request already exists for this branch:"
        gh pr view "$BRANCH" --web || true
    else
        DEFAULT_BRANCH="$(gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name' 2>/dev/null || echo main)"
        gh pr create \
            --draft \
            --base "$DEFAULT_BRANCH" \
            --head "$BRANCH" \
            --title "Confirm Section 14.11 BudgetMem-R performance gate" \
            --body-file - <<'EOF'
## Summary

- verifies the corrected Section 14.11.7 matched-cell analysis;
- confirms at least one long-range matched cell where BudgetMem-R beats both deterministic cache policies;
- replaces the defective legacy confirmation evidence;
- records reproducible text, JSON, CSV, and checksum evidence.

## Gate result

**PASS**

Retraining is not required.
EOF
        PR_STATUS="draft-created"
    fi
else
    echo
    echo "GitHub CLI is unavailable or not authenticated; no pull request was created."
fi

echo
echo "============================================================"
echo "SECTION 14.11.7 FINALIZATION COMPLETE"
echo "============================================================"
echo "Branch:          $BRANCH"
echo "Latest commit:   $(git rev-parse --short HEAD)"
echo "Push:            COMPLETE"
echo "Pull request:    $PR_STATUS"
echo "Unrelated files: NOT STAGED"
echo
echo "Next action:"
echo "Review the draft pull request, then merge it after the repository checks pass."
