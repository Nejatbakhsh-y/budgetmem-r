#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

readonly TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly TEST_FILE="tests/test_section14_authoritative.py"

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

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SECTION14_TIMESTAMP="$TIMESTAMP"
export SECTION14_REPO_ROOT="$REPO_ROOT"
export SECTION14_FUNCTION_B64="ZGVmIHRlc3Rfc2VjdGlvbjE0X2NhY2hlZF9zdGF0ZV9ncmFwaF9wb2xpY3lfaXNfZXhwbGljaXQoKSAtPiBOb25lOgogICAgIiIiVmVyaWZ5IGNhY2hlZC1jb250ZW50IGRldGFjaG1lbnQgaW5kZXBlbmRlbnRseSBvZiB3cml0ZSBkZWNpc2lvbnMuIiIiCgogICAgaW5wdXRzID0gdG9yY2gucmFuZG4oMiwgMTAsIDYpCgogICAgIyBUaGUgd3JpdGUgZGVjaXNpb24gbWF5IGxlZ2l0aW1hdGVseSBkZXBlbmQgb24gY2FuZGlkYXRlLXZhbHVlIGZlYXR1cmVzLgogICAgIyBUaGUgcmVxdWlyZW1lbnQgY29uY2VybnMgdGhlIGNhY2hlZCBjb250ZW50IGl0c2VsZi4KICAgIGRldGFjaGVkID0gbWFrZV9tb2RlbCgKICAgICAgICBzZWVkPTIwMjYsCiAgICAgICAgdGhyZXNob2xkPTAuMCwKICAgICAgICBkZXRhY2hfbWVtb3J5X3dyaXRlcz1UcnVlLAogICAgKS5ldmFsKCkKICAgIGRldGFjaGVkX291dHB1dCA9IGRldGFjaGVkKGlucHV0cywgYnVkZ2V0PTIpCgogICAgYXNzZXJ0IGRldGFjaGVkLmRldGFjaF9tZW1vcnlfd3JpdGVzIGlzIFRydWUKICAgIGFzc2VydCBkZXRhY2hlZF9vdXRwdXQuc2VxdWVuY2VfbG9naXRzLnJlcXVpcmVzX2dyYWQKICAgIGFzc2VydCBkZXRhY2hlZF9vdXRwdXQuZmluYWxfbWVtb3J5LnZhbHVlcy5yZXF1aXJlc19ncmFkIGlzIEZhbHNlCiAgICBhc3NlcnQgZGV0YWNoZWRfb3V0cHV0LmZpbmFsX21lbW9yeS52YWx1ZXMuZ3JhZF9mbiBpcyBOb25lCgogICAgY29ubmVjdGVkID0gbWFrZV9tb2RlbCgKICAgICAgICBzZWVkPTIwMjYsCiAgICAgICAgdGhyZXNob2xkPTAuMCwKICAgICAgICBkZXRhY2hfbWVtb3J5X3dyaXRlcz1GYWxzZSwKICAgICkuZXZhbCgpCiAgICBjb25uZWN0ZWRfb3V0cHV0ID0gY29ubmVjdGVkKGlucHV0cy5jbG9uZSgpLCBidWRnZXQ9MikKCiAgICBhc3NlcnQgY29ubmVjdGVkLmRldGFjaF9tZW1vcnlfd3JpdGVzIGlzIEZhbHNlCiAgICBhc3NlcnQgY29ubmVjdGVkX291dHB1dC5maW5hbF9tZW1vcnkudmFsdWVzLnJlcXVpcmVzX2dyYWQgaXMgVHJ1ZQogICAgYXNzZXJ0IGNvbm5lY3RlZF9vdXRwdXQuZmluYWxfbWVtb3J5LnZhbHVlcy5ncmFkX2ZuIGlzIG5vdCBOb25lCgogICAgY29ubmVjdGVkLnplcm9fZ3JhZChzZXRfdG9fbm9uZT1UcnVlKQogICAgY29ubmVjdGVkX291dHB1dC5maW5hbF9tZW1vcnkudmFsdWVzLnN1bSgpLmJhY2t3YXJkKCkKCiAgICBjb25uZWN0ZWRfZ3JhZGllbnRzID0gWwogICAgICAgIHBhcmFtZXRlci5ncmFkCiAgICAgICAgZm9yIHBhcmFtZXRlciBpbiBjb25uZWN0ZWQudmFsdWVfcHJvamVjdGlvbi5wYXJhbWV0ZXJzKCkKICAgICAgICBpZiBwYXJhbWV0ZXIuZ3JhZCBpcyBub3QgTm9uZQogICAgXQogICAgYXNzZXJ0IGNvbm5lY3RlZF9ncmFkaWVudHMKICAgIGFzc2VydCBhbGwoCiAgICAgICAgdG9yY2guaXNmaW5pdGUoZ3JhZGllbnQpLmFsbCgpCiAgICAgICAgZm9yIGdyYWRpZW50IGluIGNvbm5lY3RlZF9ncmFkaWVudHMKICAgICkKICAgIGFzc2VydCBhbnkoCiAgICAgICAgdG9yY2guY291bnRfbm9uemVybyhncmFkaWVudCkuaXRlbSgpID4gMAogICAgICAgIGZvciBncmFkaWVudCBpbiBjb25uZWN0ZWRfZ3JhZGllbnRzCiAgICApCg=="

[[ -f "$TEST_FILE" ]] || die "Missing $TEST_FILE."
[[ -f tests/test_budgetmem_r.py ]] || die "Missing tests/test_budgetmem_r.py."

BACKUP_DIR="$REPO_ROOT/reports/evidence/backups/section14_graph_policy_fix/$TIMESTAMP"
LOG_FILE="$REPO_ROOT/reports/evidence/logs/section14_graph_policy_${TIMESTAMP}.log"
JUNIT_FILE="$REPO_ROOT/reports/evidence/junit/section14_graph_policy_${TIMESTAMP}.xml"
REPORT_FILE="$REPO_ROOT/reports/evidence/section14_unit_tests_report.txt"
RESULTS_FILE="$REPO_ROOT/reports/tables/section14_unit_test_results.csv"
MANIFEST_FILE="$REPO_ROOT/reports/evidence/section14_authoritative_gate_manifest.json"
GRAPH_EVIDENCE_FILE="$REPO_ROOT/reports/evidence/section14_cached_state_graph_policy_${TIMESTAMP}.json"

mkdir -p \
    "$BACKUP_DIR" \
    reports/evidence/logs \
    reports/evidence/junit \
    reports/tables

cp "$TEST_FILE" "$BACKUP_DIR/test_section14_authoritative.py"
[[ -f "$REPORT_FILE" ]] && cp "$REPORT_FILE" "$BACKUP_DIR/section14_unit_tests_report.txt"
[[ -f "$RESULTS_FILE" ]] && cp "$RESULTS_FILE" "$BACKUP_DIR/section14_unit_test_results.csv"
[[ -f "$MANIFEST_FILE" ]] && cp "$MANIFEST_FILE" "$BACKUP_DIR/section14_authoritative_gate_manifest.json"

export SECTION14_GRAPH_EVIDENCE_FILE="$GRAPH_EVIDENCE_FILE"

log "Replacing the cached-state graph-policy test."

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import base64
import os
from pathlib import Path

root = Path(os.environ["SECTION14_REPO_ROOT"])
path = root / "tests" / "test_section14_authoritative.py"
text = path.read_text(encoding="utf-8")

start_marker = "def test_section14_cached_state_graph_policy_is_explicit() -> None:"
end_marker = "\ndef test_section14_memory_resets_between_unrelated_sequences() -> None:"

start = text.find(start_marker)
end = text.find(end_marker, start)
if start < 0 or end < 0:
    raise RuntimeError(
        "The cached-state graph-policy test block could not be located."
    )

replacement = base64.b64decode(
    os.environ["SECTION14_FUNCTION_B64"]
).decode("utf-8")

updated = text[:start] + replacement.rstrip() + "\n" + text[end:]
path.write_text(updated, encoding="utf-8", newline="\n")
print(f"Patched: {path.relative_to(root)}")
PY

log "Recording direct cached-content graph evidence."

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from budgetmem.models.budgetmem_r import BudgetMemR

evidence_path = Path(os.environ["SECTION14_GRAPH_EVIDENCE_FILE"])
timestamp = os.environ["SECTION14_TIMESTAMP"]


def build(detach: bool) -> BudgetMemR:
    torch.manual_seed(2026)
    return BudgetMemR(
        input_dim=6,
        hidden_dim=12,
        output_dim=3,
        key_dim=8,
        value_dim=10,
        budget_embedding_dim=5,
        controller_dim=16,
        max_budget=4,
        allowed_budgets=(2, 4),
        retrieval_k=2,
        fusion="gated",
        write_threshold=0.0,
        write_temperature=0.67,
        detach_memory_writes=detach,
    ).eval()


inputs = torch.randn(2, 10, 6)

detached_model = build(True)
detached_output = detached_model(inputs, budget=2)

connected_model = build(False)
connected_output = connected_model(inputs.clone(), budget=2)
connected_model.zero_grad(set_to_none=True)
connected_output.final_memory.values.sum().backward()

gradient_norms = {
    name: float(parameter.grad.norm().item())
    for name, parameter in connected_model.value_projection.named_parameters()
    if parameter.grad is not None
}

payload = {
    "schema_version": "1.0",
    "generated_utc": timestamp,
    "detached_cached_values": {
        "detach_memory_writes": True,
        "requires_grad": bool(
            detached_output.final_memory.values.requires_grad
        ),
        "grad_fn": (
            type(detached_output.final_memory.values.grad_fn).__name__
            if detached_output.final_memory.values.grad_fn is not None
            else None
        ),
    },
    "trainable_cached_values": {
        "detach_memory_writes": False,
        "requires_grad": bool(
            connected_output.final_memory.values.requires_grad
        ),
        "grad_fn": (
            type(connected_output.final_memory.values.grad_fn).__name__
            if connected_output.final_memory.values.grad_fn is not None
            else None
        ),
        "value_projection_gradient_norms": gradient_norms,
    },
}

assert payload["detached_cached_values"]["requires_grad"] is False
assert payload["detached_cached_values"]["grad_fn"] is None
assert payload["trainable_cached_values"]["requires_grad"] is True
assert gradient_norms
assert any(value > 0.0 for value in gradient_norms.values())

evidence_path.write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2))
PY

log "Checking test syntax."
"$PYTHON_BIN" -m py_compile "$TEST_FILE"

TARGETS=(
    "tests/test_budgetmem_r.py"
    "tests/test_section14_authoritative.py"
)

log "Verifying collection."
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest \
    -q \
    -o addopts='' \
    --collect-only \
    "${TARGETS[@]}" \
    >/dev/null

log "Running the complete authoritative Section 14 gate."

set +e
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest \
    -q \
    -o addopts='' \
    "${TARGETS[@]}" \
    --junitxml="$JUNIT_FILE" \
    2>&1 | tee "$LOG_FILE"
PYTEST_EXIT="${PIPESTATUS[0]}"
set -e

export SECTION14_JUNIT_FILE="$JUNIT_FILE"
export SECTION14_LOG_FILE="$LOG_FILE"
export SECTION14_REPORT_FILE="$REPORT_FILE"
export SECTION14_RESULTS_FILE="$RESULTS_FILE"
export SECTION14_MANIFEST_FILE="$MANIFEST_FILE"
export SECTION14_PYTEST_EXIT="$PYTEST_EXIT"

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import csv
import json
import os
import platform
import xml.etree.ElementTree as ET
from pathlib import Path

junit = Path(os.environ["SECTION14_JUNIT_FILE"])
log = Path(os.environ["SECTION14_LOG_FILE"])
report = Path(os.environ["SECTION14_REPORT_FILE"])
results = Path(os.environ["SECTION14_RESULTS_FILE"])
manifest = Path(os.environ["SECTION14_MANIFEST_FILE"])
graph_evidence = Path(os.environ["SECTION14_GRAPH_EVIDENCE_FILE"])
exit_code = int(os.environ["SECTION14_PYTEST_EXIT"])
timestamp = os.environ["SECTION14_TIMESTAMP"]

cases: list[dict[str, str]] = []
if junit.exists():
    root = ET.parse(junit).getroot()
    for case in root.iter("testcase"):
        status = "PASS"
        detail = ""
        for child_name in ("failure", "error", "skipped"):
            child = case.find(child_name)
            if child is not None:
                status = child_name.upper()
                detail = (
                    child.attrib.get("message")
                    or child.text
                    or ""
                ).strip()
                break
        cases.append(
            {
                "classname": case.attrib.get("classname", ""),
                "test_name": case.attrib.get("name", ""),
                "status": status,
                "seconds": case.attrib.get("time", "0"),
                "detail": detail.replace("\n", " ")[:5000],
            }
        )

with results.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=(
            "classname",
            "test_name",
            "status",
            "seconds",
            "detail",
        ),
    )
    writer.writeheader()
    writer.writerows(cases)

required = {
    "Budget correctness": (
        "test_section14_budget_correctness_every_forward_step",
    ),
    "Causality": (
        "test_section14_causality_future_suffix_cannot_change_prefix_decisions",
    ),
    "Determinism": (
        "test_section14_deterministic_dataset_generation",
        "test_section14_deterministic_initialization_order_and_evaluation",
    ),
    "Synthetic seed isolation": (
        "test_section14_synthetic_split_seeds_are_disjoint",
    ),
    "HDFS block isolation": (
        "test_section14_hdfs_block_ids_are_disjoint",
    ),
    "IMDb official-test isolation": (
        "test_section14_imdb_official_test_is_isolated",
    ),
    "Gradient flow": (
        "test_section14_memory_controllers_receive_gradients",
    ),
    "Cached-state graph policy": (
        "test_section14_cached_state_graph_policy_is_explicit",
    ),
    "Memory reset": (
        "test_section14_memory_resets_between_unrelated_sequences",
    ),
}

statuses: dict[str, str] = {}
for category, names in required.items():
    matched = [
        case for case in cases
        if case["test_name"] in names
    ]
    statuses[category] = (
        "PASS"
        if len(matched) == len(names)
        and all(case["status"] == "PASS" for case in matched)
        else "FAIL"
    )

production_cases = [
    case for case in cases
    if "test_budgetmem_r" in case["classname"]
]
production_pass = bool(production_cases) and all(
    case["status"] == "PASS"
    for case in production_cases
)
all_selected_pass = bool(cases) and all(
    case["status"] == "PASS"
    for case in cases
)
go = (
    exit_code == 0
    and production_pass
    and all_selected_pass
    and all(status == "PASS" for status in statuses.values())
)

lines = [
    "Section 14 — Authoritative Unit Tests Required Before Training",
    f"Generated UTC: {timestamp}",
    "",
]
for category, status in statuses.items():
    lines.append(f"{category}: {status}")

lines.extend(
    [
        f"Production BudgetMem-R tests: "
        f"{'PASS' if production_pass else 'FAIL'}",
        f"All selected authoritative tests: "
        f"{'PASS' if all_selected_pass else 'FAIL'}",
        f"Pytest exit code: {exit_code}",
        "",
        f"Final decision: {'GO' if go else 'NO-GO'}",
        f"Section 14: {'COMPLETE' if go else 'INCOMPLETE'}",
        "",
        "Determinism scope: exact CPU execution. CUDA limitations remain "
        "outside this CPU pre-training gate and must be documented separately.",
        f"Cached-state graph evidence: {graph_evidence}",
        f"JUnit evidence: {junit}",
        f"Detailed log: {log}",
        f"Result table: {results}",
        f"Gate manifest: {manifest}",
    ]
)

failed = [case for case in cases if case["status"] != "PASS"]
if failed:
    lines.extend(["", "Failed or unresolved checks:"])
    for case in failed:
        lines.append(
            f"- {case['test_name']}: {case['status']} — "
            f"{case['detail'] or 'No detail recorded.'}"
        )

report.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

manifest.write_text(
    json.dumps(
        {
            "schema_version": "1.1",
            "generated_utc": timestamp,
            "section": "14",
            "gate_type": "authoritative_pretraining",
            "selected_targets": [
                "tests/test_budgetmem_r.py",
                "tests/test_section14_authoritative.py",
            ],
            "cached_state_graph_policy": {
                "method": (
                    "Inspect cached-content autograd connectivity directly; "
                    "do not infer content detachment from write-decision "
                    "controller gradients."
                ),
                "evidence": str(graph_evidence),
            },
            "category_status": statuses,
            "production_tests": (
                "PASS" if production_pass else "FAIL"
            ),
            "pytest_exit_code": exit_code,
            "decision": "GO" if go else "NO-GO",
            "platform": platform.platform(),
            "evidence": {
                "junit": str(junit),
                "log": str(log),
                "results_csv": str(results),
                "report": str(report),
            },
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print()
print(report.read_text(encoding="utf-8"))
PY

if [[ "$PYTEST_EXIT" -eq 0 ]]; then
    printf '\nSECTION 14 RESULT: GO\n'
    printf 'Section 14 is complete. Training may begin.\n'
else
    printf '\nSECTION 14 RESULT: NO-GO\n'
    printf 'Review reports/evidence/section14_unit_tests_report.txt.\n'
    exit "$PYTEST_EXIT"
fi
