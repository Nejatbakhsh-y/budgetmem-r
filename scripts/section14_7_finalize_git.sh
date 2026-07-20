#!/usr/bin/env bash
set -Eeuo pipefail

echo "============================================================"
echo "SECTION 14.7 — COMMIT, PUSH, AND RUNTIME-OUTPUT EXCLUSION"
echo "============================================================"

# ------------------------------------------------------------
# 1. Confirm repository
# ------------------------------------------------------------
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: This command must be run inside the budgetmem-r repository."
    exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "Repository: $REPO_ROOT"

if ! git remote get-url origin >/dev/null 2>&1; then
    echo "ERROR: Git remote 'origin' is not configured."
    exit 1
fi

# ------------------------------------------------------------
# 2. Ensure a feature branch is being used
# ------------------------------------------------------------
BRANCH="$(git branch --show-current)"

if [[ -z "$BRANCH" ]]; then
    BRANCH="feature/section14-finalization"
    git switch -c "$BRANCH"
elif [[ "$BRANCH" == "main" || "$BRANCH" == "master" ]]; then
    BRANCH="feature/section14-finalization"

    if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
        git switch "$BRANCH"
    else
        git switch -c "$BRANCH"
    fi
fi

echo "Feature branch: $BRANCH"

# ------------------------------------------------------------
# 3. Add runtime-output exclusions
# ------------------------------------------------------------
IGNORE_BEGIN="# BEGIN SECTION 14 RUNTIME OUTPUTS"
IGNORE_END="# END SECTION 14 RUNTIME OUTPUTS"

if ! grep -Fq "$IGNORE_BEGIN" .gitignore 2>/dev/null; then
    cat >> .gitignore <<'EOF'

# BEGIN SECTION 14 RUNTIME OUTPUTS
# Experiment execution directories
.runtime/
runtime/
runs/
checkpoints/
wandb/
mlruns/
tensorboard/
lightning_logs/

# Temporary runtime reports
reports/runtime/
reports/tmp/
artifacts/runtime/

# Python and test caches
**/__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.mypy_cache/
.ipynb_checkpoints/

# Coverage and profiling outputs
.coverage
coverage.xml
htmlcov/
.prof/
*.prof

# Runtime logs and temporary files
*.log
*.tmp
*.temp
*.pid
# END SECTION 14 RUNTIME OUTPUTS
EOF
    echo "Runtime-output exclusions added to .gitignore."
else
    echo "Runtime-output exclusion block already exists."
fi

# ------------------------------------------------------------
# 4. Protect required evidence
# ------------------------------------------------------------
mkdir -p reports/evidence

if git check-ignore -q --no-index \
    reports/evidence/section14_7_git_verification.txt; then
    echo "ERROR: reports/evidence is ignored by the current .gitignore."
    echo "Remove the broad reports/ ignore rule before continuing."
    git check-ignore -v --no-index \
        reports/evidence/section14_7_git_verification.txt || true
    exit 1
fi

# ------------------------------------------------------------
# 5. Stop tracking files now covered by .gitignore
#    Local files are preserved.
# ------------------------------------------------------------
mapfile -d '' TRACKED_IGNORED < <(
    git ls-files -ci --exclude-standard -z || true
)

if (( ${#TRACKED_IGNORED[@]} > 0 )); then
    echo
    echo "Removing ignored runtime files from the Git index:"
    printf '  %s\n' "${TRACKED_IGNORED[@]}"

    printf '%s\0' "${TRACKED_IGNORED[@]}" |
        xargs -0 git rm --cached --ignore-unmatch --
else
    echo "No tracked runtime-output files require removal."
fi

# ------------------------------------------------------------
# 6. Stage and validate all Section 14 changes
# ------------------------------------------------------------
git add -A

REMAINING_TRACKED_IGNORED="$(
    git ls-files -ci --exclude-standard | wc -l | tr -d ' '
)"

if [[ "$REMAINING_TRACKED_IGNORED" != "0" ]]; then
    echo "ERROR: Some ignored runtime files remain tracked:"
    git ls-files -ci --exclude-standard
    exit 1
fi

echo
echo "Staged changes:"
git status --short

# ------------------------------------------------------------
# 7. Commit and push project changes
# ------------------------------------------------------------
if git diff --cached --quiet; then
    echo "No uncommitted project changes were found."
else
    git commit -m \
        "Finalize Section 14 validation and exclude runtime outputs"
fi

git push --set-upstream origin "$BRANCH"

git fetch origin "$BRANCH"

LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git rev-parse "origin/$BRANCH")"

if [[ "$LOCAL_SHA" != "$REMOTE_SHA" ]]; then
    echo "ERROR: Local and remote feature branches do not match."
    echo "Local:  $LOCAL_SHA"
    echo "Remote: $REMOTE_SHA"
    exit 1
fi

# ------------------------------------------------------------
# 8. Create verification evidence
# ------------------------------------------------------------
EVIDENCE_FILE="reports/evidence/section14_7_git_verification.txt"

{
    echo "Section 14.7 Git Verification"
    echo "============================="
    echo
    echo "Repository: $REPO_ROOT"
    echo "Feature branch: $BRANCH"
    echo "Validated commit: $LOCAL_SHA"
    echo "Remote: $(git remote get-url origin)"
    echo "Verification timestamp UTC: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo
    echo "Checks"
    echo "------"
    echo "[PASS] Work was committed on a feature branch."
    echo "[PASS] Feature branch was pushed to origin."
    echo "[PASS] Local and remote commit references matched."
    echo "[PASS] Runtime-output exclusions were added to .gitignore."
    echo "[PASS] Previously tracked ignored files were removed from the index."
    echo "[PASS] Runtime files were preserved locally."
    echo "[PASS] reports/evidence remains eligible for Git tracking."
    echo
    echo "Runtime-output ignore rules"
    echo "---------------------------"
    sed -n \
        "/^${IGNORE_BEGIN//\//\\/}$/,/^${IGNORE_END//\//\\/}$/p" \
        .gitignore
    echo
    echo "Final tracked-but-ignored file count: 0"
} > "$EVIDENCE_FILE"

git add .gitignore scripts/section14_7_finalize_git.sh "$EVIDENCE_FILE"

if ! git diff --cached --quiet; then
    git commit -m "Record Section 14.7 Git verification evidence"
fi

git push origin "$BRANCH"
git fetch origin "$BRANCH"

FINAL_LOCAL_SHA="$(git rev-parse HEAD)"
FINAL_REMOTE_SHA="$(git rev-parse "origin/$BRANCH")"

if [[ "$FINAL_LOCAL_SHA" != "$FINAL_REMOTE_SHA" ]]; then
    echo "ERROR: Final evidence commit was not synchronized."
    exit 1
fi

echo
echo "============================================================"
echo "SECTION 14.7 COMPLETED"
echo "============================================================"
echo "Feature branch: $BRANCH"
echo "Final commit:   $FINAL_LOCAL_SHA"
echo "Evidence:       $EVIDENCE_FILE"
echo
echo "Runtime outputs remain on the computer but are excluded"
echo "from Git tracking."
echo
git status --short --branch
