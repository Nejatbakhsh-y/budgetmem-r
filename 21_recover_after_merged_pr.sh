#!/usr/bin/env bash
set -Eeuo pipefail

# Safely recover local work after PR #1 was merged remotely while the local
# checkout could not switch branches because of uncommitted changes.
#
# Run from the budgetmem-r repository root in the VS Code WSL terminal.
#
# This script does NOT delete local work and does NOT drop the safety stash.

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
    echo "ERROR: Run this script from inside the budgetmem-r Git repository."
    exit 2
fi
cd "$ROOT"

if [[ -d .git/rebase-merge || -d .git/rebase-apply || -f .git/MERGE_HEAD ]]; then
    echo "ERROR: A merge or rebase is already in progress."
    echo "Finish or abort it before running this recovery automation."
    exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CURRENT_BRANCH="$(git branch --show-current)"
CURRENT_HEAD="$(git rev-parse HEAD)"
BACKUP_DIR="$ROOT/.git/local-recovery/$STAMP"
STASH_MESSAGE="post-merge local recovery $STAMP"
RECOVERY_BRANCH="recovery/post-merge-local-work-${STAMP,,}"

mkdir -p "$BACKUP_DIR"

echo "============================================================"
echo "POST-MERGE LOCAL-WORK RECOVERY"
echo "============================================================"
echo "Current branch: $CURRENT_BRANCH"
echo "Current commit: $CURRENT_HEAD"
echo "Backup folder:  $BACKUP_DIR"
echo

# Record the exact pre-recovery state outside the tracked working tree.
git status --short --branch > "$BACKUP_DIR/status-before.txt"
git diff --binary > "$BACKUP_DIR/unstaged.patch" || true
git diff --cached --binary > "$BACKUP_DIR/staged.patch" || true
git ls-files --others --exclude-standard -z > "$BACKUP_DIR/untracked-files.zlist"

if [[ -s "$BACKUP_DIR/untracked-files.zlist" ]]; then
    tar --null -T "$BACKUP_DIR/untracked-files.zlist" \
        -czf "$BACKUP_DIR/untracked-files.tar.gz"
fi

printf '%s\n' \
    "Original branch: $CURRENT_BRANCH" \
    "Original commit: $CURRENT_HEAD" \
    "Created UTC: $STAMP" \
    > "$BACKUP_DIR/recovery-metadata.txt"

# Collect the failed GitHub Actions logs for the commit when possible.
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    gh run list \
        --commit "$CURRENT_HEAD" \
        --limit 20 \
        --json databaseId,name,workflowName,conclusion,status,url \
        > "$BACKUP_DIR/github-runs.json" 2>/dev/null || true

    while IFS= read -r run_id; do
        [[ -z "$run_id" ]] && continue
        gh run view "$run_id" --log-failed \
            > "$BACKUP_DIR/github-run-${run_id}-failed.log" 2>&1 || true
    done < <(
        gh run list \
            --commit "$CURRENT_HEAD" \
            --limit 20 \
            --json databaseId,conclusion \
            --jq '.[] | select(.conclusion == "failure") | .databaseId' \
            2>/dev/null || true
    )

    gh pr view 1 \
        --json number,state,isDraft,mergedAt,mergeCommit,url \
        > "$BACKUP_DIR/pr-1-status.json" 2>/dev/null || true
fi

# Preserve all tracked, staged, and untracked work.
if [[ -n "$(git status --porcelain=v1 -uall)" ]]; then
    git stash push --include-untracked -m "$STASH_MESSAGE"
    STASH_REF="$(git stash list --format='%gd%x09%s' |
        awk -F '\t' -v msg="$STASH_MESSAGE" '$2 == msg {print $1; exit}')"

    if [[ -z "$STASH_REF" ]]; then
        echo "ERROR: The safety stash could not be identified."
        echo "Your patch backups remain in: $BACKUP_DIR"
        exit 2
    fi
else
    STASH_REF=""
    echo "Working tree was already clean; no stash was required."
fi

if [[ -n "$(git status --porcelain=v1 -uall)" ]]; then
    echo "ERROR: Working tree is not clean after stashing."
    echo "Stop and inspect: git status"
    exit 2
fi

git fetch --prune origin

DEFAULT_BRANCH=""
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    DEFAULT_BRANCH="$(gh repo view --json defaultBranchRef \
        --jq '.defaultBranchRef.name' 2>/dev/null || true)"
fi

if [[ -z "$DEFAULT_BRANCH" ]]; then
    DEFAULT_BRANCH="$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD \
        2>/dev/null | sed 's#^origin/##' || true)"
fi

if [[ -z "$DEFAULT_BRANCH" ]]; then
    if git show-ref --verify --quiet refs/remotes/origin/main; then
        DEFAULT_BRANCH="main"
    elif git show-ref --verify --quiet refs/remotes/origin/master; then
        DEFAULT_BRANCH="master"
    else
        echo "ERROR: Could not determine the repository default branch."
        exit 2
    fi
fi

if git show-ref --verify --quiet "refs/heads/$DEFAULT_BRANCH"; then
    git switch "$DEFAULT_BRANCH"
else
    git switch --track -c "$DEFAULT_BRANCH" "origin/$DEFAULT_BRANCH"
fi

git pull --ff-only origin "$DEFAULT_BRANCH"

if git show-ref --verify --quiet "refs/heads/$RECOVERY_BRANCH"; then
    echo "ERROR: Recovery branch already exists: $RECOVERY_BRANCH"
    exit 2
fi

git switch -c "$RECOVERY_BRANCH"

APPLY_RESULT="not-required"
if [[ -n "$STASH_REF" ]]; then
    set +e
    git stash apply --index "$STASH_REF"
    APPLY_CODE=$?
    set -e

    if [[ "$APPLY_CODE" -eq 0 ]]; then
        APPLY_RESULT="applied-cleanly"
    else
        APPLY_RESULT="conflicts-require-review"
    fi
fi

git status --short --branch > "$BACKUP_DIR/status-after.txt"

echo
echo "============================================================"
echo "RECOVERY RESULT"
echo "============================================================"
echo "Updated base branch:   $DEFAULT_BRANCH"
echo "New recovery branch:   $RECOVERY_BRANCH"
echo "Safety stash:          ${STASH_REF:-not required}"
echo "Stash application:     $APPLY_RESULT"
echo "Backup folder:         $BACKUP_DIR"
echo

if [[ "$APPLY_RESULT" == "conflicts-require-review" ]]; then
    echo "ACTION REQUIRED:"
    echo "The local work was preserved, but Git found conflicts while restoring it."
    echo "Do not run git stash drop."
    echo "Resolve the files marked by: git status"
    echo "Then run your tests and commit the repaired work on:"
    echo "  $RECOVERY_BRANCH"
    exit 1
fi

echo "Local work is restored on the new recovery branch."
echo "The safety stash was intentionally retained."
echo
echo "NEXT COMMANDS:"
echo "  git status --short"
echo "  ls -1 \"$BACKUP_DIR\""
echo
echo "Do not merge another pull request yet."
echo "Use the collected failed-action logs to repair the two failing test checks,"
echo "run the local test suite, and open a new pull request from the recovery branch."
