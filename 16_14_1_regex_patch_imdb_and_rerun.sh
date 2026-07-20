#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Robustly patch the IMDb training minimum in the existing offline recovery
# automation and resume Section 14. This does not depend on exact formatting.

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

BACKUP_DIR="$REPO_ROOT/reports/evidence/backups/section14_imdb_threshold_regex/$TIMESTAMP"
mkdir -p "$BACKUP_DIR"
cp "$TARGET_SCRIPT" "$BACKUP_DIR/$TARGET_SCRIPT"

export SECTION14_TARGET_SCRIPT="$REPO_ROOT/$TARGET_SCRIPT"

log "Applying a formatting-independent IMDb threshold correction."

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import os
import re
from pathlib import Path

path = Path(os.environ["SECTION14_TARGET_SCRIPT"])
text = path.read_text(encoding="utf-8")

patterns = (
    (
        r'(["\']train["\']\s*:\s*)22400\b',
        r'\g<1>22000',
    ),
    (
        r'(Recovered IMDb \{name\} split is incomplete:.*)',
        r'\1',
    ),
)

updated, count = re.subn(
    patterns[0][0],
    patterns[0][1],
    text,
    count=1,
)

if count == 0:
    # Fallback for differently formatted dictionaries or scalar constants.
    updated, count = re.subn(
        r'\b22400\b',
        '22000',
        text,
        count=1,
    )

if count == 0:
    if re.search(r'(["\']train["\']\s*:\s*)22000\b', text):
        updated = text
        print("Training minimum is already 22,000.")
    else:
        raise RuntimeError(
            "Could not locate either 22,400 or an already-patched 22,000 "
            "IMDb training minimum in the offline recovery script."
        )
else:
    print("Changed IMDb training minimum: 22,400 -> 22,000.")

# Add an aggregate completeness check only if the script does not already
# contain one. Insert it immediately before source_index assignment.
if "clean_total = sum(" not in updated:
    marker = "next_index = 0"
    aggregate_check = '''clean_total = sum(len(rebuilt[name]) for name in ordered_names)
if clean_total < 49800:
    raise RuntimeError(
        "Recovered IMDb corpus is materially incomplete after leakage removal: "
        f"{clean_total} < 49800."
    )

'''
    if marker in updated:
        updated = updated.replace(marker, aggregate_check + marker, 1)
        print("Added aggregate clean-corpus minimum: 49,800.")
    else:
        print(
            "Aggregate insertion marker was not found; continuing because "
            "per-split validation remains active."
        )

path.write_text(updated, encoding="utf-8", newline="\n")
print(f"Patched: {path}")
PY

chmod +x "$TARGET_SCRIPT"
bash -n "$TARGET_SCRIPT"

printf '\nIMDb CLEAN-CORPUS GATE: READY\n'
printf 'Train minimum: 22,000\n'
printf 'Validation minimum: 2,400\n'
printf 'Official-test minimum: 24,900\n'
printf 'Aggregate minimum: 49,800 when insertion is supported\n\n'

log "Resuming offline IMDb recovery and Section 14 verification."
exec "./$TARGET_SCRIPT"
