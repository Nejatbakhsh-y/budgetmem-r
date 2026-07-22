#!/usr/bin/env bash
set -Eeuo pipefail

# Repair the exact Section 18 release so audit-approved files located under an
# ignored artifacts/section18 directory are force-added individually. This
# automation never force-adds a directory and never broadens the audited file
# boundary.

EXPECTED_BRANCH="feature/18-main-experiment-matrix"
TARGET="39_section18_exact_audited_results_commit.sh"
EVIDENCE_DIR="reports/evidence/section18"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="$EVIDENCE_DIR/backups/section18_ignored_artifact_repair_${STAMP}"
RUN_LOG="$EVIDENCE_DIR/section18_ignored_artifact_repair_${STAMP}.log"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

header() {
  printf '\n%s\n' "======================================================================"
  printf '%s\n' "$1"
  printf '%s\n' "======================================================================"
}

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$REPO_ROOT" ]] || fail "Run this automation from inside the budgetmem-r Git repository."
cd "$REPO_ROOT"

BRANCH="$(git branch --show-current)"
[[ "$BRANCH" == "$EXPECTED_BRANCH" ]] || \
  fail "Expected branch '$EXPECTED_BRANCH', found '$BRANCH'."

[[ -f "$TARGET" ]] || fail "Missing required automation: $TARGET"
mkdir -p "$BACKUP_DIR" "$EVIDENCE_DIR"
cp -p "$TARGET" "$BACKUP_DIR/$TARGET"

header "CHECK AND CLEAN FAILED SECTION 18 STAGING"
mapfile -d '' -t PRESTAGED < <(git diff --cached --name-only -z)
if (( ${#PRESTAGED[@]} > 0 )); then
  echo "Staged paths left by the failed exact-release attempt: ${#PRESTAGED[@]}"
  printf '  %s\n' "${PRESTAGED[@]}"

  python - "${PRESTAGED[@]}" <<'PY'
from __future__ import annotations

import sys
from pathlib import PurePosixPath

allowed_exact = {
    "21_section18_main_experiment_matrix.sh",
    "33_section18_full_matrix_audit.sh",
    "39_section18_exact_audited_results_commit.sh",
    "40_section18_ignored_artifact_repair_and_commit.sh",
    "scripts/run_section18.py",
    "scripts/data/prepare_bgl.py",
    "src/budgetmem/data/bgl.py",
    "src/budgetmem/data/__init__.py",
    "tests/test_section18_release_gate.py",
}
allowed_prefixes = (
    "artifacts/section18/",
    "outputs/section18/",
    "results/section18/",
    "reports/section18/",
    "reports/tables/section18/",
    "reports/evidence/section18/",
)

unexpected: list[str] = []
for raw in sys.argv[1:]:
    path = PurePosixPath(raw).as_posix()
    if path in allowed_exact or path.startswith(allowed_prefixes):
        continue
    unexpected.append(path)

if unexpected:
    print("ERROR: Refusing to unstage paths outside the controlled Section 18 boundary:", file=sys.stderr)
    for path in unexpected:
        print(f"  - {path}", file=sys.stderr)
    raise SystemExit(1)
PY

  git restore --staged -- "${PRESTAGED[@]}"
  git diff --cached --quiet || fail "Could not restore the Git index to an empty state."
  echo "Controlled failed-attempt staging removed: PASS"
else
  echo "Git index already clean: PASS"
fi

header "PATCH EXACT-PATH STAGING FOR IGNORED AUDITED FILES"
python - "$TARGET" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

old_stage = 'xargs -0 git add -- < "$CANDIDATE_NUL"'
new_stage = '''while IFS= read -r path; do
  [[ -n "$path" ]] || continue
  if git check-ignore -q -- "$path"; then
    echo "Force-adding exact audit-approved ignored file: $path"
    git add -f -- "$path"
  else
    git add -- "$path"
  fi
done < "$CANDIDATE_TEXT"'''

if old_stage in text:
    text = text.replace(old_stage, new_stage, 1)
elif new_stage not in text:
    raise SystemExit("ERROR: Could not locate the exact staging command in file 39.")

anchor = '    Path("39_section18_exact_audited_results_commit.sh"),\n'
addition = anchor + '    Path("40_section18_ignored_artifact_repair_and_commit.sh"),\n'
if 'Path("40_section18_ignored_artifact_repair_and_commit.sh")' not in text:
    if anchor not in text:
        raise SystemExit("ERROR: Could not locate file 39 in the explicit durable-file list.")
    text = text.replace(anchor, addition, 1)

path.write_text(text, encoding="utf-8")
PY

# Safety assertions: force-add must be file-by-file, sourced only from the exact
# candidate text generated and hash-verified by file 39.
grep -Fq 'git add -f -- "$path"' "$TARGET" || \
  fail "Exact ignored-file staging patch was not installed."
grep -Fq 'done < "$CANDIDATE_TEXT"' "$TARGET" || \
  fail "Patched staging is not bounded by the exact candidate list."
if grep -Eq 'git add -f --[[:space:]]+(artifacts|reports|outputs|results)(/|$)' "$TARGET"; then
  fail "Unsafe directory-level force-add detected."
fi

bash -n "$TARGET"
echo "Exact ignored-file staging patch: PASS"
echo "Directory-level force-add: NO"
echo "Recursive bundle expansion: NO"
echo "Backup: $BACKUP_DIR/$TARGET"

header "RESUME EXACT AUDITED SECTION 18 COMMIT"
set +e
bash "$TARGET" 2>&1 | tee "$RUN_LOG"
STATUS=${PIPESTATUS[0]}
set -e

if (( STATUS != 0 )); then
  fail "Exact audited release still failed. Review: $RUN_LOG"
fi

grep -Fq "FINAL DECISION: SECTION 18 EXACT AUDITED RESULTS COMMITTED AND PUSHED" "$RUN_LOG" || \
  fail "File 39 exited without its authoritative success decision. Review: $RUN_LOG"

header "SECTION 18 IGNORED-ARTIFACT REPAIR COMPLETED"
echo "FINAL DECISION: SECTION 18 EXACT AUDITED RESULTS COMMITTED AND PUSHED"
echo "Ignored audit-approved files force-added individually: YES"
echo "Ignored directories force-added recursively: NO"
echo "Full matrix rerun: NO"
echo "Repair log: $RUN_LOG"
echo "Backup: $BACKUP_DIR/$TARGET"
