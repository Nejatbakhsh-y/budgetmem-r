#!/usr/bin/env bash
set -Eeuo pipefail

# Section 18 controlled release:
#   1. Verify the release gate and readiness audit.
#   2. Run focused quality checks.
#   3. Stage only the approved Section 18 implementation files.
#   4. Commit and push the feature branch.
#   5. Launch the full Section 18 matrix only with --execute.
#
# This script intentionally leaves reports, outputs, data products, logs,
# checkpoints, and unrelated generated files unstaged.

EXPECTED_BRANCH="feature/18-main-experiment-matrix"
MATRIX_SCRIPT="21_section18_main_experiment_matrix.sh"
RELEASE_REPORT="reports/evidence/section18/section18_runner_release_gate_latest.txt"
READINESS_REPORT="reports/evidence/section18/section18_runner_readiness_latest.txt"
COMMIT_MESSAGE="Implement Section 18 BGL loader and dedicated runner"
EXECUTE_MATRIX=0

usage() {
  cat <<'USAGE'
Usage:
  ./31_section18_selective_commit_and_execute.sh
      Verify, test, selectively commit, and push Section 18 implementation.

  ./31_section18_selective_commit_and_execute.sh --execute
      Perform the controlled commit/push and then launch the full matrix.
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE_MATRIX=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown argument: $arg" >&2; usage; exit 2 ;;
  esac
done

trap 'echo; echo "ERROR: Section 18 controlled release failed at line $LINENO." >&2' ERR

header() {
  printf '\n%s\n' "======================================================================"
  printf '%s\n' "$1"
  printf '%s\n' "======================================================================"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: Required file is missing: $path" >&2
    exit 1
  fi
}

# Resolve repository root from any subdirectory.
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  echo "ERROR: Run this automation from inside the budgetmem-r Git repository." >&2
  exit 1
fi
cd "$REPO_ROOT"

header "SECTION 18 CONTROLLED RELEASE"
echo "Repository: $REPO_ROOT"

BRANCH="$(git branch --show-current)"
echo "Current branch: $BRANCH"
if [[ "$BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "ERROR: Expected branch '$EXPECTED_BRANCH', but current branch is '$BRANCH'." >&2
  exit 1
fi

# Never mix this controlled commit with files that were already staged manually.
if ! git diff --cached --quiet; then
  echo "ERROR: The Git index already contains staged changes." >&2
  echo "Unstage them before running this automation; no files were changed by this check." >&2
  exit 1
fi

require_file "$RELEASE_REPORT"
require_file "$READINESS_REPORT"
require_file "$MATRIX_SCRIPT"
require_file "src/budgetmem/data/bgl.py"
require_file "src/budgetmem/data/__init__.py"
require_file "scripts/data/prepare_bgl.py"
require_file "scripts/run_section18.py"
require_file "tests/test_section18_release_gate.py"

header "VERIFY FINAL RELEASE EVIDENCE"
if ! grep -Fq "FINAL DECISION: SECTION 18 RUNNER RELEASE GATE PASSED" "$RELEASE_REPORT"; then
  echo "ERROR: Release-gate PASS decision was not found in $RELEASE_REPORT" >&2
  exit 1
fi
if ! grep -Fq "FINAL DECISION: READY FOR SECTION 18 EXECUTION" "$READINESS_REPORT"; then
  echo "ERROR: Readiness decision was not found in $READINESS_REPORT" >&2
  exit 1
fi
echo "Release gate: PASS"
echo "Readiness audit: READY FOR SECTION 18 EXECUTION"

header "RUN FOCUSED QUALITY CHECKS"
python -m py_compile \
  src/budgetmem/data/bgl.py \
  scripts/data/prepare_bgl.py \
  scripts/run_section18.py \
  tests/test_section18_release_gate.py

if command -v ruff >/dev/null 2>&1; then
  ruff check \
    src/budgetmem/data/bgl.py \
    src/budgetmem/data/__init__.py \
    scripts/data/prepare_bgl.py \
    scripts/run_section18.py \
    tests/test_section18_release_gate.py
else
  python -m ruff check \
    src/budgetmem/data/bgl.py \
    src/budgetmem/data/__init__.py \
    scripts/data/prepare_bgl.py \
    scripts/run_section18.py \
    tests/test_section18_release_gate.py
fi

python -m pytest -q tests/test_section18_release_gate.py

echo "Focused compilation, Ruff, and pytest checks: PASS"

header "STAGE ONLY APPROVED SECTION 18 FILES"

# Production and reproducibility files only. Recovery-only scripts 29 and 30,
# runtime evidence, generated datasets, checkpoints, and matrix outputs are excluded.
APPROVED_FILES=(
  "src/budgetmem/data/bgl.py"
  "src/budgetmem/data/__init__.py"
  "scripts/data/prepare_bgl.py"
  "scripts/run_section18.py"
  "tests/test_section18_release_gate.py"
  "28_section18_bgl_runner_release_gate.sh"
  "31_section18_selective_commit_and_execute.sh"
)

FILES_TO_STAGE=()
for path in "${APPROVED_FILES[@]}"; do
  if [[ -e "$path" ]]; then
    FILES_TO_STAGE+=("$path")
  fi
done

git add -- "${FILES_TO_STAGE[@]}"

mapfile -t STAGED_FILES < <(git diff --cached --name-only)
if (( ${#STAGED_FILES[@]} == 0 )); then
  echo "No new approved changes require a commit. Required files may already be committed."
else
  printf 'Staged files (%d):\n' "${#STAGED_FILES[@]}"
  printf '  %s\n' "${STAGED_FILES[@]}"

  # Hard safety boundary against generated/runtime content.
  for path in "${STAGED_FILES[@]}"; do
    case "$path" in
      reports/*|outputs/*|data/*|logs/*|checkpoints/*|.venv/*)
        echo "ERROR: Prohibited generated/runtime path was staged: $path" >&2
        git restore --staged -- "${STAGED_FILES[@]}"
        exit 1
        ;;
    esac
  done

  # Ensure the staged patch itself has no whitespace defects.
  git diff --cached --check

  git commit -m "$COMMIT_MESSAGE"
fi

header "PUSH FEATURE BRANCH"
git push -u origin "$BRANCH"
echo "Feature branch pushed: $BRANCH"

UNCOMMITTED_COUNT="$(git status --porcelain=v1 --untracked-files=all | wc -l | tr -d ' ')"
echo "Uncommitted/untracked paths intentionally left outside the commit: $UNCOMMITTED_COUNT"

if (( EXECUTE_MATRIX == 0 )); then
  header "SECTION 18 IMPLEMENTATION RELEASED"
  echo "Commit/push phase: COMPLETE"
  echo "Full matrix launched: NO"
  echo
  echo "To launch the full matrix after this controlled release, run:"
  echo "  ./31_section18_selective_commit_and_execute.sh --execute"
  exit 0
fi

header "LAUNCH FULL SECTION 18 MATRIX"
chmod +x "$MATRIX_SCRIPT"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MATRIX_LOG="reports/evidence/section18/section18_full_matrix_${TIMESTAMP}.log"
mkdir -p "$(dirname "$MATRIX_LOG")"

echo "Matrix command: ./$MATRIX_SCRIPT --execute"
echo "Live log: $MATRIX_LOG"
echo "Generated outputs will remain unstaged for post-run validation."

set +e
bash "$MATRIX_SCRIPT" --execute 2>&1 | tee "$MATRIX_LOG"
MATRIX_STATUS=${PIPESTATUS[0]}
set -e

header "SECTION 18 MATRIX EXECUTION RESULT"
if (( MATRIX_STATUS != 0 )); then
  echo "FINAL DECISION: SECTION 18 MATRIX FAILED OR STOPPED"
  echo "Exit status: $MATRIX_STATUS"
  echo "Review log: $MATRIX_LOG"
  exit "$MATRIX_STATUS"
fi

echo "FINAL DECISION: SECTION 18 MATRIX EXECUTION COMPLETED"
echo "Matrix log: $MATRIX_LOG"
echo "Git staging performed for generated outputs: NO"
echo "Next required action: audit the complete matrix before committing any result artifacts."
