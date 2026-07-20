#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Repair OneDrive/WSL PermissionError during IMDb DatasetDict replacement.
#
# The offline recovery already creates and validates the rebuilt DatasetDict.
# Windows OneDrive may reject pathlib.Path.rename() across the mounted NTFS
# directory even when source and destination are siblings. This patch replaces
# rename() with a retrying copytree/rmtree operation and resumes Section 14.

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

BACKUP_DIR="$REPO_ROOT/reports/evidence/backups/section14_onedrive_replace/$TIMESTAMP"
mkdir -p "$BACKUP_DIR"
cp "$TARGET_SCRIPT" "$BACKUP_DIR/$TARGET_SCRIPT"

export SECTION14_TARGET_SCRIPT="$REPO_ROOT/$TARGET_SCRIPT"

log "Replacing pathlib rename with a OneDrive-safe directory copy."

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import os
import re
from pathlib import Path

path = Path(os.environ["SECTION14_TARGET_SCRIPT"])
text = path.read_text(encoding="utf-8")

old = '''if destination.exists():
    shutil.rmtree(destination)
temporary.rename(destination)
'''

new = '''def replace_directory_on_mounted_windows(
    source: Path,
    destination: Path,
) -> None:
    # Replace a directory without relying on NTFS/OneDrive rename semantics.
    import stat
    import time

    def make_writable(target: str) -> None:
        try:
            os.chmod(
                target,
                stat.S_IRUSR
                | stat.S_IWUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IWGRP
                | stat.S_IXGRP,
            )
        except OSError:
            pass

    def remove_tree(target: Path) -> None:
        if not target.exists():
            return

        def onerror(function, failing_path, _exc_info):
            make_writable(failing_path)
            function(failing_path)

        last_error: Exception | None = None
        for attempt in range(8):
            try:
                shutil.rmtree(target, onerror=onerror)
                return
            except (PermissionError, OSError) as exc:
                last_error = exc
                time.sleep(0.5 * (attempt + 1))

        raise RuntimeError(
            f"Could not remove existing DatasetDict directory {target}: "
            f"{last_error}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    remove_tree(destination)

    last_error: Exception | None = None
    for attempt in range(8):
        try:
            shutil.copytree(
                source,
                destination,
                copy_function=shutil.copy2,
                dirs_exist_ok=True,
            )
            source_files = {
                item.relative_to(source)
                for item in source.rglob("*")
                if item.is_file()
            }
            destination_files = {
                item.relative_to(destination)
                for item in destination.rglob("*")
                if item.is_file()
            }
            if not source_files.issubset(destination_files):
                missing = sorted(source_files - destination_files)
                raise RuntimeError(
                    "DatasetDict copy is incomplete; missing files: "
                    + ", ".join(str(item) for item in missing[:10])
                )

            remove_tree(source)
            return
        except (PermissionError, OSError, RuntimeError) as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))

    raise RuntimeError(
        f"Could not copy rebuilt DatasetDict from {source} to "
        f"{destination}: {last_error}"
    )


replace_directory_on_mounted_windows(temporary, destination)
'''

if old in text:
    updated = text.replace(old, new, 1)
elif "replace_directory_on_mounted_windows(temporary, destination)" in text:
    updated = text
    print("OneDrive-safe directory replacement is already installed.")
else:
    pattern = (
        r'if\s+destination\.exists\(\)\s*:\s*\n'
        r'\s*shutil\.rmtree\(destination\)\s*\n'
        r'\s*temporary\.rename\(destination\)'
    )
    updated, count = re.subn(pattern, new.rstrip(), text, count=1)
    if count == 0:
        raise RuntimeError(
            "Could not locate the temporary.rename(destination) replacement "
            "block in the offline IMDb recovery script."
        )

path.write_text(updated, encoding="utf-8", newline="\n")
print(f"Patched: {path}")
PY

chmod +x "$TARGET_SCRIPT"

log "Checking the patched recovery automation."
bash -n "$TARGET_SCRIPT"

printf '\nONEDRIVE DATASETDICT REPLACEMENT: PATCHED\n'
printf 'Validated temporary DatasetDicts will now be copied with retries.\n\n'

log "Resuming offline IMDb recovery and Section 14 verification."
exec "./$TARGET_SCRIPT"
