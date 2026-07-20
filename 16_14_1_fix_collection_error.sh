#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Repairs the Section 14 pytest collection failure, verifies test collection,
# and reruns the original Section 14 automation.
#
# Run from the budgetmem-r repository in the VS Code WSL/Bash terminal.

readonly TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly ORIGINAL_SCRIPT="16_14_1_section14_unit_tests.sh"

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

find_repo_root() {
    if root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
        printf '%s\n' "$root"
        return 0
    fi

    local cursor
    cursor="$(pwd)"
    while [[ "$cursor" != "/" ]]; do
        if [[ -f "$cursor/pyproject.toml" && -d "$cursor/src/budgetmem" ]]; then
            printf '%s\n' "$cursor"
            return 0
        fi
        cursor="$(dirname "$cursor")"
    done
    return 1
}

choose_python() {
    local candidate
    for candidate in \
        "$REPO_ROOT/.venv/bin/python" \
        "$REPO_ROOT/venv/bin/python" \
        "$REPO_ROOT/.env/bin/python" \
        python3 \
        python
    do
        if [[ "$candidate" == */* ]]; then
            [[ -x "$candidate" ]] && {
                printf '%s\n' "$candidate"
                return 0
            }
        elif command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

REPO_ROOT="$(find_repo_root)" || die "The budgetmem-r repository root could not be found."
cd "$REPO_ROOT"

PYTHON_BIN="$(choose_python)" || die "Python could not be found."
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

[[ -f "$ORIGINAL_SCRIPT" ]] || die "Missing $ORIGINAL_SCRIPT in $REPO_ROOT."
[[ -f tests/test_section14_required.py ]] || die "Missing tests/test_section14_required.py."
[[ -f tests/section14_runtime.py ]] || die "Missing tests/section14_runtime.py."

mkdir -p reports/evidence/logs reports/evidence/backups/section14_collection_fix

BACKUP_DIR="$REPO_ROOT/reports/evidence/backups/section14_collection_fix/$TIMESTAMP"
LOG_FILE="$REPO_ROOT/reports/evidence/logs/section14_collection_fix_${TIMESTAMP}.log"
mkdir -p "$BACKUP_DIR"

cp "$ORIGINAL_SCRIPT" "$BACKUP_DIR/$ORIGINAL_SCRIPT"
cp tests/test_section14_required.py "$BACKUP_DIR/test_section14_required.py"
cp tests/section14_runtime.py "$BACKUP_DIR/section14_runtime.py"

log "Backups saved under ${BACKUP_DIR#$REPO_ROOT/}."
log "Patching the current test and the embedded test template."

export SECTION14_REPO_ROOT="$REPO_ROOT"
export SECTION14_ORIGINAL_SCRIPT="$REPO_ROOT/$ORIGINAL_SCRIPT"

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

root = Path(os.environ["SECTION14_REPO_ROOT"])
original_script = Path(os.environ["SECTION14_ORIGINAL_SCRIPT"])
current_test = root / "tests" / "test_section14_required.py"

old_header = """from __future__ import annotations

import copy
"""

new_header = """from __future__ import annotations

import sys
from pathlib import Path

# Pytest can use importlib collection mode, in which the tests directory is not
# automatically importable as a top-level module. Add it explicitly so the
# generated runtime adapter can always be imported.
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import copy
"""

def patch_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")

    if "TESTS_DIR = Path(__file__).resolve().parent" in text:
        return "already patched"

    if old_header not in text:
        raise RuntimeError(
            f"Expected generated header was not found in {path}. "
            "The file may have been edited independently."
        )

    path.write_text(text.replace(old_header, new_header, 1), encoding="utf-8", newline="\n")
    return "patched"

print(f"{current_test}: {patch_file(current_test)}")
print(f"{original_script}: {patch_file(original_script)}")
PY

chmod +x "$ORIGINAL_SCRIPT"

log "Checking Bash syntax."
bash -n "$ORIGINAL_SCRIPT"

log "Checking generated Python syntax."
"$PYTHON_BIN" -m py_compile \
    tests/section14_runtime.py \
    tests/test_section14_required.py

log "Running pytest collection verification."
set +e
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest \
    -vv \
    -o addopts='' \
    --collect-only \
    tests/test_section14_required.py \
    2>&1 | tee "$LOG_FILE"
COLLECT_EXIT="${PIPESTATUS[0]}"
set -e

if [[ "$COLLECT_EXIT" -ne 0 ]]; then
    printf '\nSECTION 14 COLLECTION REPAIR: FAIL\n'
    printf 'The import-path repair was applied, but another collection error remains.\n'
    printf 'Full traceback: %s\n' "${LOG_FILE#$REPO_ROOT/}"
    exit "$COLLECT_EXIT"
fi

COLLECTED_COUNT="$(
    grep -Eo 'collected [0-9]+ items?|[0-9]+ tests? collected' "$LOG_FILE" \
        | grep -Eo '[0-9]+' \
        | tail -n 1 \
        || true
)"

if [[ -z "$COLLECTED_COUNT" || "$COLLECTED_COUNT" -lt 13 ]]; then
    printf '\nSECTION 14 COLLECTION REPAIR: FAIL\n'
    printf 'Expected at least 13 generated Section 14 tests, but collection reported %s.\n' "${COLLECTED_COUNT:-0}"
    printf 'Review: %s\n' "${LOG_FILE#$REPO_ROOT/}"
    exit 3
fi

printf '\nSECTION 14 COLLECTION REPAIR: PASS\n'
printf 'Collected tests: %s\n' "$COLLECTED_COUNT"
printf 'Rerunning the complete Section 14 automation now.\n\n'

exec "./$ORIGINAL_SCRIPT"
