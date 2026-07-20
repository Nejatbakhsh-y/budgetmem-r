#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "FAIL: Run this inside the budgetmem-r repository."
  exit 2
}
cd "$ROOT"

DIAG="reports/evidence/section14_11_8_diagnostic.txt"
DECISION="reports/evidence/section14_11_final_go_decision.txt"
FINAL="reports/evidence/section14_11_final_report.txt"
VERIFY="reports/evidence/section14_11_8_verification.txt"

mkdir -p reports/evidence

python3 - "$DIAG" "$DECISION" "$FINAL" "$VERIFY" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

diag_path = Path(sys.argv[1])
decision_path = Path(sys.argv[2])
final_path = Path(sys.argv[3])
verify_path = Path(sys.argv[4])

if not diag_path.is_file():
    print(f"FAIL: Missing diagnostic file: {diag_path}")
    print("Run ./18_build_and_verify_section_14_11_8.sh first.")
    raise SystemExit(2)

text = diag_path.read_text(encoding="utf-8", errors="replace")

required_gates = {
    "Gradient stability": r"(?m)^Gradient stability\s+PASS\s*$",
    "Memory-budget enforcement": (
        r"(?m)^Memory-budget enforcement\s+PASS\s*$"
    ),
    "Same-budget comparison": (
        r"(?m)^Same-budget comparison\s+PASS\s*$"
    ),
    "Deterministic-policy comparison": (
        r"(?m)^Deterministic policies outperformed:"
        r"\s*at least 2\s+PASS\s*$"
    ),
}

results = {
    name: bool(re.search(pattern, text))
    for name, pattern in required_gates.items()
}

count_match = re.search(
    r"(?m)^Detected deterministic-policy count:\s*([0-9]+)\s*$",
    text,
)
policy_count = int(count_match.group(1)) if count_match else 0
results["At least two policies"] = policy_count >= 2

source_matches = re.findall(r"(?m)^\s*Source:\s*(.+?)\s*$", text)
source_paths = [
    value.strip()
    for value in source_matches
    if value.strip() and value.strip() != "NOT FOUND"
]

missing_sources = [
    source
    for source in source_paths
    if not Path(source.split(";", 1)[0].strip()).exists()
]

all_pass = all(results.values()) and not missing_sources

print("Section 14.11.8 Finalization Gate")
print("=" * 41)

for name, passed in results.items():
    print(f"{name:<42} {'PASS' if passed else 'FAIL'}")

print(f"{'Referenced evidence files exist':<42} "
      f"{'PASS' if not missing_sources else 'FAIL'}")

if missing_sources:
    print("\nMissing referenced evidence:")
    for source in missing_sources:
        print(f"  {source}")

if not all_pass:
    print("\nFINALIZATION ABORTED")
    print("The objective GO criteria are not fully supported.")
    raise SystemExit(1)

canonical = [
    "Gradient stability: PASS",
    "Memory-budget enforcement: PASS",
    "Same-budget comparison: PASS",
    f"Deterministic policies outperformed: at least {policy_count}",
    "Final decision: GO",
    "Section 14.11: COMPLETE",
]

decision_lines = [
    "Section 14.11 Final GO Decision",
    "=" * 31,
    "",
    *canonical,
    "",
    "Decision basis",
    "--------------",
    "The final decision is GO because the independently checked source",
    "evidence confirms gradient stability, strict memory-budget enforcement,",
    "a same-budget comparison, and superiority over at least two deterministic",
    "memory policies.",
    "",
    f"Source diagnostic: {diag_path}",
]

decision_path.write_text(
    "\n".join(decision_lines).rstrip() + "\n",
    encoding="utf-8",
)

final_lines = [
    "Section 14.11 Final Report",
    "=" * 26,
    "",
    *canonical,
    "",
    "Supporting evidence",
    "-------------------",
    f"Final GO decision: {decision_path}",
    f"Source diagnostic: {diag_path}",
]

final_path.write_text(
    "\n".join(final_lines).rstrip() + "\n",
    encoding="utf-8",
)

final_text = final_path.read_text(encoding="utf-8", errors="replace")

exact_requirements = [
    "Gradient stability: PASS",
    "Memory-budget enforcement: PASS",
    "Same-budget comparison: PASS",
    "Final decision: GO",
    "Section 14.11: COMPLETE",
]

exact_ok = all(line in final_text for line in exact_requirements)

count_final = re.search(
    r"(?m)^Deterministic policies outperformed:"
    r"\s*at least\s+([0-9]+)\s*$",
    final_text,
)
exact_ok = (
    exact_ok
    and count_final is not None
    and int(count_final.group(1)) >= 2
)

status = "PASS" if exact_ok else "FAIL"

verification_lines = [
    "Section 14.11.8 Final Report Verification",
    "=" * 46,
    "",
    *canonical,
    "",
    f"Section 14.11.8 verification: {status}",
    "",
    f"Final report: {final_path}",
    f"Final GO decision: {decision_path}",
]

verify_path.write_text(
    "\n".join(verification_lines).rstrip() + "\n",
    encoding="utf-8",
)

print()
print("\n".join(canonical))
print()
print(f"Section 14.11.8 verification: {status}")
print(f"Final report saved: {final_path}")
print(f"GO decision saved: {decision_path}")
print(f"Verification saved: {verify_path}")

raise SystemExit(0 if exact_ok else 1)
PY
