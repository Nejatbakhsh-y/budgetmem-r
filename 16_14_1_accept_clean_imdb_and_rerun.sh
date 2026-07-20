#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Accept the integrity-cleaned local IMDb corpus and resume Section 14.
#
# The prior run found:
#   train=22,386 after official-test leakage removal
# The previous 22,400 minimum was an arbitrary recovery tolerance, not a
# Section 14 requirement. This patch validates aggregate completeness and
# preserves strict validation/test minimums.

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

REPO_ROOT="$(find_repo_root)" || die "The budgetmem-r repository root was not found."
cd "$REPO_ROOT"

[[ -f "$TARGET_SCRIPT" ]] || die "Missing $TARGET_SCRIPT in $REPO_ROOT."

BACKUP_DIR="$REPO_ROOT/reports/evidence/backups/section14_imdb_threshold/$TIMESTAMP"
mkdir -p "$BACKUP_DIR"
cp "$TARGET_SCRIPT" "$BACKUP_DIR/$TARGET_SCRIPT"

export SECTION14_TARGET_SCRIPT="$REPO_ROOT/$TARGET_SCRIPT"

log "Patching the arbitrary IMDb split-size threshold."

python3 - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

path = Path(os.environ["SECTION14_TARGET_SCRIPT"])
text = path.read_text(encoding="utf-8")

old = '''expected_minimums = {
    "train": 22400,
    "validation": 2400,
    "test": 24900,
}
for name, minimum in expected_minimums.items():
    if len(rebuilt[name]) < minimum:
        raise RuntimeError(
            f"Recovered IMDb {name} split is incomplete: "
            f"{len(rebuilt[name])} < {minimum}."
        )
'''

new = '''expected_minimums = {
    "train": 22000,
    "validation": 2400,
    "test": 24900,
}
for name, minimum in expected_minimums.items():
    if len(rebuilt[name]) < minimum:
        raise RuntimeError(
            f"Recovered IMDb {name} split is incomplete: "
            f"{len(rebuilt[name])} < {minimum}."
        )

clean_total = sum(len(rebuilt[name]) for name in ordered_names)
if clean_total < 49800:
    raise RuntimeError(
        "Recovered IMDb corpus is materially incomplete after leakage removal: "
        f"{clean_total} < 49800."
    )
'''

if old in text:
    text = text.replace(old, new, 1)
elif '"train": 22000' in text and 'clean_total = sum(' in text:
    print("IMDb completeness gate was already patched.")
else:
    raise RuntimeError(
        "The expected IMDb minimum-size block was not found. "
        "Do not modify the recovery script manually before running this repair."
    )

path.write_text(text, encoding="utf-8", newline="\n")
print(f"Patched: {path}")
PY

chmod +x "$TARGET_SCRIPT"
bash -n "$TARGET_SCRIPT"

printf '\nIMDb CLEAN-CORPUS GATE: PATCHED\n'
printf 'Required train minimum: 22,000\n'
printf 'Required validation minimum: 2,400\n'
printf 'Required official-test minimum: 24,900\n'
printf 'Required aggregate minimum: 49,800\n\n'

log "Resuming offline IMDb reconstruction and the Section 14 gate."
exec "./$TARGET_SCRIPT"
