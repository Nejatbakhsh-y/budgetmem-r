#!/usr/bin/env bash
set -Eeuo pipefail

COMMIT_MESSAGE="${COMMIT_MESSAGE:-Implement controlled Section 12 baselines}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "Run this script from inside the budgetmem-r Git repository."

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

BRANCH="$(git branch --show-current)"
[[ -n "$BRANCH" ]] || die "Detached HEAD detected. Switch to a branch before committing."

printf 'Repository: %s\n' "$ROOT"
printf 'Branch:     %s\n\n' "$BRANCH"

# Clear only the staging area. This does not delete or revert working-tree files.
git reset --quiet

add_if_exists() {
    local path="$1"
    if [[ -e "$path" ]]; then
        git add -- "$path"
    fi
}

# Automation and configuration.
add_if_exists "15_12_1_implement_section12_baselines.sh"
add_if_exists "configs/baselines/section12_baselines.yaml"

# Controlled baseline package and registries.
add_if_exists "src/budgetmem/baselines"
add_if_exists "src/budgetmem/models/__init__.py"
add_if_exists "src/budgetmem/memory/__init__.py"

# Stage 1: recurrent models.
for path in \
    src/budgetmem/models/rnn.py \
    src/budgetmem/models/gru.py \
    src/budgetmem/models/lstm.py
do
    add_if_exists "$path"
done

# Stage 2: deterministic memory policies.
for path in \
    src/budgetmem/memory/uniform.py \
    src/budgetmem/memory/most_recent.py \
    src/budgetmem/memory/fifo.py \
    src/budgetmem/memory/lru.py \
    src/budgetmem/memory/random_policy.py \
    src/budgetmem/memory/reservoir.py \
    src/budgetmem/memory/novelty.py \
    src/budgetmem/memory/surprise.py
do
    add_if_exists "$path"
done

# Stages 3 through 6.
for path in \
    src/budgetmem/models/attention.py \
    src/budgetmem/models/state_space.py \
    src/budgetmem/models/recurrent_memory.py \
    src/budgetmem/models/memory_caching.py
do
    add_if_exists "$path"
done

# Verification, calibration, tests, and auditable evidence.
while IFS= read -r -d '' path; do
    git add -- "$path"
done < <(find scripts -maxdepth 1 -type f -iname '*section12*' -print0 2>/dev/null || true)

add_if_exists "tests/test_section12_baselines.py"

while IFS= read -r -d '' path; do
    git add -- "$path"
done < <(find reports/evidence -maxdepth 1 -type f -name 'section12_*' -print0 2>/dev/null || true)

while IFS= read -r -d '' path; do
    git add -- "$path"
done < <(find reports/tables -maxdepth 1 -type f -name 'section12_*' -print0 2>/dev/null || true)

mapfile -t STAGED < <(git diff --cached --name-only)

if (( ${#STAGED[@]} == 0 )); then
    printf '\nNo Section 12 changes were staged.\n'
    printf 'Nothing was committed or pushed.\n'
    exit 2
fi

printf '\nStaged Section 12 files:\n'
printf '  %s\n' "${STAGED[@]}"

# Defensive check: reject any staged file outside the controlled Section 12 allowlist.
is_allowed() {
    local path="$1"
    case "$path" in
        15_12_1_implement_section12_baselines.sh) return 0 ;;
        configs/baselines/section12_baselines.yaml) return 0 ;;
        src/budgetmem/baselines/*) return 0 ;;
        src/budgetmem/models/__init__.py) return 0 ;;
        src/budgetmem/memory/__init__.py) return 0 ;;
        src/budgetmem/models/rnn.py) return 0 ;;
        src/budgetmem/models/gru.py) return 0 ;;
        src/budgetmem/models/lstm.py) return 0 ;;
        src/budgetmem/models/attention.py) return 0 ;;
        src/budgetmem/models/state_space.py) return 0 ;;
        src/budgetmem/models/recurrent_memory.py) return 0 ;;
        src/budgetmem/models/memory_caching.py) return 0 ;;
        src/budgetmem/memory/uniform.py) return 0 ;;
        src/budgetmem/memory/most_recent.py) return 0 ;;
        src/budgetmem/memory/fifo.py) return 0 ;;
        src/budgetmem/memory/lru.py) return 0 ;;
        src/budgetmem/memory/random_policy.py) return 0 ;;
        src/budgetmem/memory/reservoir.py) return 0 ;;
        src/budgetmem/memory/novelty.py) return 0 ;;
        src/budgetmem/memory/surprise.py) return 0 ;;
        scripts/*section12*) return 0 ;;
        tests/test_section12_baselines.py) return 0 ;;
        reports/evidence/section12_*) return 0 ;;
        reports/tables/section12_*) return 0 ;;
        *) return 1 ;;
    esac
}

UNEXPECTED=()
for path in "${STAGED[@]}"; do
    if ! is_allowed "$path"; then
        UNEXPECTED+=("$path")
    fi
done

if (( ${#UNEXPECTED[@]} > 0 )); then
    printf '\nUnexpected staged files detected:\n' >&2
    printf '  %s\n' "${UNEXPECTED[@]}" >&2
    git reset --quiet
    die "Commit cancelled. The staging area was cleared; working files were preserved."
fi

printf '\nStaged summary:\n'
git diff --cached --stat

git commit -m "$COMMIT_MESSAGE"

printf '\nPushing branch %s...\n' "$BRANCH"
git push -u origin "$BRANCH"

printf '\nSection 12 commit and push: COMPLETE\n'
printf 'Commit: %s\n' "$(git rev-parse --short HEAD)"
printf 'Branch: %s\n' "$BRANCH"
printf '\nRemaining unstaged project changes were not included:\n'
git status --short
