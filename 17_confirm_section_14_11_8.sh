#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "FAIL: Run this inside the budgetmem-r repository."
    exit 2
}

cd "$ROOT"
mkdir -p reports/evidence

python3 - <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

root = Path.cwd()
reports = root / "reports"
output = reports / "evidence" / "section14_11_8_verification.txt"

if not reports.is_dir():
    print("FAIL: reports/ directory was not found.")
    sys.exit(2)

preferred = [
    reports / "evidence" / "section14_11_final_report.txt",
    reports / "evidence" / "section14_11_completion_report.txt",
    reports / "evidence" / "section14_11_final_go_decision.txt",
    reports / "evidence" / "section15_final_go_decision.txt",
    reports / "pilot_tuned_report.md",
    reports / "pilot_report.md",
]

extensions = {".txt", ".md", ".log", ".csv", ".rst"}

candidates = []
seen = set()

for path in preferred + sorted(reports.rglob("*")):
    if (
        path.is_file()
        and path.suffix.lower() in extensions
        and path.resolve() != output.resolve()
        and path.resolve() not in seen
    ):
        seen.add(path.resolve())
        candidates.append(path)

if not candidates:
    print("FAIL: No readable report files were found under reports/.")
    sys.exit(2)

checks = [
    (
        "Gradient stability: PASS",
        re.compile(
            r"(?im)^\s*(?:[-*|>]\s*)?"
            r"gradient\s+stability\s*:\s*pass\b"
        ),
    ),
    (
        "Memory-budget enforcement: PASS",
        re.compile(
            r"(?im)^\s*(?:[-*|>]\s*)?"
            r"memory[-\s]+budget\s+enforcement\s*:\s*pass\b"
        ),
    ),
    (
        "Same-budget comparison: PASS",
        re.compile(
            r"(?im)^\s*(?:[-*|>]\s*)?"
            r"same[-\s]+budget\s+comparison\s*:\s*pass\b"
        ),
    ),
    (
        "Deterministic policies outperformed: at least 2",
        re.compile(
            r"(?im)^\s*(?:[-*|>]\s*)?"
            r"deterministic\s+policies\s+outperformed\s*:\s*"
            r"at\s+least\s+([0-9]+)\b"
        ),
    ),
    (
        "Final decision: GO",
        re.compile(
            r"(?im)^\s*(?:[-*|>]\s*)?"
            r"final\s+decision\s*:\s*go\b"
        ),
    ),
    (
        "Section 14.11: COMPLETE",
        re.compile(
            r"(?im)^\s*(?:[-*|>]\s*)?"
            r"section\s+14\.11\s*:\s*complete\b"
        ),
    ),
]


def normalize(text: str) -> str:
    for character in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014"):
        text = text.replace(character, "-")
    return text.replace("`", "").replace("**", "").replace("__", "")


def inspect(path: Path):
    try:
        text = normalize(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return -1, []

    results = []
    score = 0

    for index, (label, pattern) in enumerate(checks):
        match = pattern.search(text)
        passed = bool(match)
        detail = ""

        if index == 3 and match:
            count = int(match.group(1))
            passed = count >= 2
            detail = f"reported count={count}"

        if passed:
            score += 1

        results.append((label, passed, detail))

    return score, results


ranked = []

for order, path in enumerate(candidates):
    score, results = inspect(path)
    ranked.append((score, -order, path, results))

ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)

best_score, _, selected, results = ranked[0]
passed = best_score == len(checks)
status = "PASS" if passed else "FAIL"

lines = [
    "Section 14.11.8 Final Report Verification",
    "=" * 46,
    f"Repository: {root}",
    f"Report checked: {selected.relative_to(root)}",
    "",
]

for label, result, detail in results:
    suffix = f" ({detail})" if detail else ""
    lines.append(
        f"{label:<58} {'PASS' if result else 'FAIL'}{suffix}"
    )

lines.extend(
    [
        "",
        f"Section 14.11.8 verification: {status}",
    ]
)

if not passed:
    lines.extend(
        [
            "",
            "The final report does not yet contain every required statement.",
            "",
            "Highest-scoring report candidates:",
        ]
    )

    for score, _, path, _ in ranked[:10]:
        lines.append(f"  {score}/6  {path.relative_to(root)}")

text = "\n".join(lines) + "\n"
output.write_text(text, encoding="utf-8")

print(text, end="")
print(f"Evidence saved: {output.relative_to(root)}")

sys.exit(0 if passed else 1)
PY
