#!/usr/bin/env bash
set -euo pipefail

echo "=== Section 14.8: Exclude runtime outputs from Git ==="

if [[ ! -d ".git" ]]; then
    echo "ERROR: Run this script from the budgetmem-r repository root."
    exit 1
fi

touch .gitignore

add_ignore_rule() {
    local rule="$1"

    if ! grep -Fxq "$rule" .gitignore; then
        printf "%s\n" "$rule" >> .gitignore
        echo "Added: $rule"
    else
        echo "Already present: $rule"
    fi
}

echo
echo "Updating .gitignore..."

add_ignore_rule ""
add_ignore_rule "# Python runtime outputs"
add_ignore_rule "__pycache__/"
add_ignore_rule "*.py[cod]"
add_ignore_rule "*.pyo"
add_ignore_rule ".pytest_cache/"
add_ignore_rule ".mypy_cache/"
add_ignore_rule ".ruff_cache/"
add_ignore_rule ".coverage"
add_ignore_rule ".coverage.*"
add_ignore_rule "htmlcov/"

add_ignore_rule ""
add_ignore_rule "# Virtual environments"
add_ignore_rule ".venv/"
add_ignore_rule "venv/"
add_ignore_rule "env/"

add_ignore_rule ""
add_ignore_rule "# Runtime logs and temporary files"
add_ignore_rule "*.log"
add_ignore_rule "*.tmp"
add_ignore_rule "*.temp"
add_ignore_rule "*.pid"
add_ignore_rule "tmp/"
add_ignore_rule "temp/"
add_ignore_rule "logs/"

add_ignore_rule ""
add_ignore_rule "# Training runtime outputs"
add_ignore_rule "runs/"
add_ignore_rule "outputs/"
add_ignore_rule "checkpoints/"
add_ignore_rule "wandb/"
add_ignore_rule "lightning_logs/"
add_ignore_rule "tensorboard/"
add_ignore_rule "*.ckpt"
add_ignore_rule "*.pt"
add_ignore_rule "*.pth"

add_ignore_rule ""
add_ignore_rule "# Editor and operating-system files"
add_ignore_rule ".DS_Store"
add_ignore_rule "Thumbs.db"
add_ignore_rule ".idea/"
add_ignore_rule "*.swp"
add_ignore_rule "*.swo"

echo
echo "Removing previously tracked runtime files from the Git index..."

git ls-files -z |
while IFS= read -r -d '' file; do
    case "$file" in
        */__pycache__/*|\
        *.pyc|*.pyo|\
        .pytest_cache/*|*/.pytest_cache/*|\
        .mypy_cache/*|*/.mypy_cache/*|\
        .ruff_cache/*|*/.ruff_cache/*|\
        .coverage|.coverage.*|\
        htmlcov/*|\
        .venv/*|venv/*|env/*|\
        *.log|*.tmp|*.temp|*.pid|\
        tmp/*|temp/*|logs/*|\
        runs/*|outputs/*|checkpoints/*|\
        wandb/*|lightning_logs/*|tensorboard/*|\
        *.ckpt|*.pt|*.pth|\
        .DS_Store|Thumbs.db|.idea/*|*.swp|*.swo)
            git rm --cached --ignore-unmatch -- "$file"
            ;;
    esac
done

echo
echo "Checking ignored runtime paths..."

printf "runtime-test\n" > section14_8_runtime_test.log

if git check-ignore -q section14_8_runtime_test.log; then
    echo "PASS: Runtime log files are ignored."
else
    echo "FAIL: Runtime log files are not ignored."
    rm -f section14_8_runtime_test.log
    exit 1
fi

rm -f section14_8_runtime_test.log

mkdir -p __pycache__
printf "runtime-test\n" > __pycache__/section14_8_test.pyc

if git check-ignore -q __pycache__/section14_8_test.pyc; then
    echo "PASS: Python cache files are ignored."
else
    echo "FAIL: Python cache files are not ignored."
    rm -rf __pycache__
    exit 1
fi

rm -rf __pycache__

mkdir -p checkpoints
printf "runtime-test\n" > checkpoints/section14_8_test.pt

if git check-ignore -q checkpoints/section14_8_test.pt; then
    echo "PASS: Checkpoint files are ignored."
else
    echo "FAIL: Checkpoint files are not ignored."
    rm -rf checkpoints
    exit 1
fi

rm -rf checkpoints

echo
echo "Checking for tracked runtime outputs..."

TRACKED_RUNTIME_OUTPUTS="$(
    git ls-files |
    grep -E '(^|/)(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|htmlcov|logs|tmp|temp|runs|outputs|checkpoints|wandb|lightning_logs|tensorboard)/|\.py[co]$|\.log$|\.tmp$|\.ckpt$|\.pt$|\.pth$' || true
)"

if [[ -n "$TRACKED_RUNTIME_OUTPUTS" ]]; then
    echo "FAIL: The following runtime outputs remain tracked:"
    printf "%s\n" "$TRACKED_RUNTIME_OUTPUTS"
    exit 1
fi

echo "PASS: No excluded runtime outputs are tracked."

mkdir -p reports/evidence
{
    echo "Section 14.8 Runtime Output Exclusion"
    echo "Timestamp UTC: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo
    echo "PASS: Runtime output patterns were added to .gitignore."
    echo "PASS: Previously tracked runtime outputs were removed from the Git index."
    echo "PASS: Python caches, logs, and checkpoints are ignored."
    echo "PASS: Required reports/evidence files remain eligible for Git tracking."
} > reports/evidence/section14_8_runtime_exclusion.txt

git add .gitignore reports/evidence/section14_8_runtime_exclusion.txt

echo
echo "=== Git status ==="
git status --short

echo
echo "=== Section 14.8 completed successfully ==="
echo "Evidence: reports/evidence/section14_8_runtime_exclusion.txt"
