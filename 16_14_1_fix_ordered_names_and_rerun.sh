#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Fix the undefined ordered_names variable introduced into the existing
# offline IMDb recovery script, validate the script, and resume Section 14.

readonly TARGET_SCRIPT="16_14_1_offline_imdb_recovery_and_rerun.sh"
readonly TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

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

REPO_ROOT="$(find_repo_root)" || die "The budgetmem-r repository root was not found."
cd "$REPO_ROOT"

PYTHON_BIN="$(choose_python)" || die "Python was not found."
[[ -f "$TARGET_SCRIPT" ]] || die "Missing $TARGET_SCRIPT in $REPO_ROOT."

BACKUP_DIR="$REPO_ROOT/reports/evidence/backups/section14_ordered_names_fix/$TIMESTAMP"
mkdir -p "$BACKUP_DIR"
cp "$TARGET_SCRIPT" "$BACKUP_DIR/$TARGET_SCRIPT"

export SECTION14_TARGET_SCRIPT="$REPO_ROOT/$TARGET_SCRIPT"

log "Removing the undefined ordered_names dependency."

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import os
import re
from pathlib import Path

path = Path(os.environ["SECTION14_TARGET_SCRIPT"])
text = path.read_text(encoding="utf-8")

explicit_total = (
    'clean_total = (\n'
    '    len(rebuilt["train"])\n'
    '    + len(rebuilt["validation"])\n'
    '    + len(rebuilt["test"])\n'
    ')'
)

patterns = (
    r'clean_total\s*=\s*sum\(\s*len\(rebuilt\[name\]\)\s*'
    r'for\s+name\s+in\s+ordered_names\s*\)',
    r'clean_total\s*=\s*sum\(\s*len\(rebuilt\[name\]\)\s*'
    r'for\s+name\s+in\s*\(\s*["\']train["\']\s*,\s*'
    r'["\']validation["\']\s*,\s*["\']test["\']\s*\)\s*\)',
)

updated = text
replacement_count = 0
for pattern in patterns:
    updated, count = re.subn(
        pattern,
        explicit_total,
        updated,
        count=1,
        flags=re.MULTILINE,
    )
    replacement_count += count
    if count:
        break

if replacement_count == 0:
    if explicit_total in updated:
        print("Explicit aggregate total was already installed.")
    elif "clean_total" in updated and "ordered_names" in updated:
        # Formatting-independent line-level fallback.
        lines = updated.splitlines()
        repaired = []
        changed = False
        for line in lines:
            if (
                not changed
                and "clean_total" in line
                and "ordered_names" in line
            ):
                indentation = line[: len(line) - len(line.lstrip())]
                repaired.extend(
                    [
                        indentation + "clean_total = (",
                        indentation + '    len(rebuilt["train"])',
                        indentation + '    + len(rebuilt["validation"])',
                        indentation + '    + len(rebuilt["test"])',
                        indentation + ")",
                    ]
                )
                changed = True
            else:
                repaired.append(line)

        if not changed:
            raise RuntimeError(
                "The undefined ordered_names expression could not be located."
            )

        updated = "\n".join(repaired) + (
            "\n" if text.endswith("\n") else ""
        )
        replacement_count = 1
    else:
        raise RuntimeError(
            "The recovery script does not contain the expected clean_total "
            "aggregate check."
        )

# As an additional safeguard, define ordered_names before any remaining use
# when the local script uses it elsewhere without an assignment.
uses_ordered_names = bool(re.search(r'\bordered_names\b', updated))
defines_ordered_names = bool(
    re.search(
        r'(?m)^\s*ordered_names\s*=',
        updated,
    )
)

if uses_ordered_names and not defines_ordered_names:
    insertion_marker = "clean_total = ("
    assignment = (
        'ordered_names = ("train", "validation", "test")\n\n'
    )
    if insertion_marker in updated:
        updated = updated.replace(
            insertion_marker,
            assignment + insertion_marker,
            1,
        )
        print("Added ordered_names definition for remaining references.")

path.write_text(updated, encoding="utf-8", newline="\n")
print("Replaced the aggregate total with explicit split lengths.")
print(f"Patched: {path}")
PY

chmod +x "$TARGET_SCRIPT"

log "Checking the patched recovery automation."
bash -n "$TARGET_SCRIPT"

printf '\nIMDb ORDERED-NAMES REPAIR: PASS\n'
printf 'Aggregate total now uses explicit train, validation, and test lengths.\n\n'

log "Resuming offline IMDb recovery and Section 14 verification."
exec "./$TARGET_SCRIPT"
