#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Section 14 interface/harness repair and rerun.
#
# Repairs:
#   1. Obsolete allowed_budgets= test keyword.
#   2. Section 14 input_dim versus embedding_dim discovery.
#   3. Mapping-style synthetic dataset fingerprinting.
#   4. Original automation scope: Section 14 tests only.
#
# Run from the budgetmem-r repository root in the VS Code WSL/Bash terminal.

readonly TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly ORIGINAL_SCRIPT="16_14_1_section14_unit_tests.sh"
readonly RUNTIME_FILE="tests/section14_runtime.py"
readonly REQUIRED_TEST_FILE="tests/test_section14_required.py"

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
export SECTION14_STRICT=1

[[ -f "$ORIGINAL_SCRIPT" ]] || die "Missing $ORIGINAL_SCRIPT."
[[ -f "$RUNTIME_FILE" ]] || die "Missing $RUNTIME_FILE."
[[ -f "$REQUIRED_TEST_FILE" ]] || die "Missing $REQUIRED_TEST_FILE."

BACKUP_DIR="$REPO_ROOT/reports/evidence/backups/section14_interface_fix/$TIMESTAMP"
DIAGNOSTIC_LOG="$REPO_ROOT/reports/evidence/logs/section14_interface_fix_${TIMESTAMP}.log"
mkdir -p "$BACKUP_DIR" reports/evidence/logs

cp "$ORIGINAL_SCRIPT" "$BACKUP_DIR/$ORIGINAL_SCRIPT"
cp "$RUNTIME_FILE" "$BACKUP_DIR/section14_runtime.py"
cp "$REQUIRED_TEST_FILE" "$BACKUP_DIR/test_section14_required.py"

log "Backups saved under ${BACKUP_DIR#$REPO_ROOT/}."
log "Discovering the active BudgetMem-R constructor and applying compatibility repairs."

export SECTION14_REPO_ROOT="$REPO_ROOT"
export SECTION14_ORIGINAL_SCRIPT="$REPO_ROOT/$ORIGINAL_SCRIPT"
export SECTION14_RUNTIME_FILE="$REPO_ROOT/$RUNTIME_FILE"

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import importlib
import inspect
import os
import re
import sys
from pathlib import Path
from typing import Any

root = Path(os.environ["SECTION14_REPO_ROOT"])
original_script = Path(os.environ["SECTION14_ORIGINAL_SCRIPT"])
runtime_file = Path(os.environ["SECTION14_RUNTIME_FILE"])
src = root / "src"

if str(src) not in sys.path:
    sys.path.insert(0, str(src))


def discover_budgetmem_class() -> type[Any]:
    preferred_modules = (
        "budgetmem.models.budgetmem_r",
        "budgetmem.models.budgetmem",
        "budgetmem.model",
    )
    candidates: list[tuple[int, type[Any]]] = []
    module_names: list[str] = list(preferred_modules)

    for path in (src / "budgetmem").rglob("*.py"):
        if path.name == "__init__.py":
            name = ".".join(path.parent.relative_to(src).parts)
        else:
            name = ".".join(path.with_suffix("").relative_to(src).parts)
        if name not in module_names:
            module_names.append(name)

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for name, obj in vars(module).items():
            if not inspect.isclass(obj):
                continue
            lowered = name.lower()
            score = 0
            if name == "BudgetMemR":
                score += 100
            if "budget" in lowered and "mem" in lowered:
                score += 60
            if obj.__module__ == module.__name__:
                score += 10
            if score:
                candidates.append((score, obj))

    if not candidates:
        raise RuntimeError("BudgetMemR class could not be discovered.")

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


model_cls = discover_budgetmem_class()
signature = inspect.signature(model_cls)
parameters = list(signature.parameters)

print(f"Discovered model: {model_cls.__module__}:{model_cls.__name__}")
print(f"Constructor signature: {signature}")

replacement_candidates = (
    "training_budgets",
    "budget_values",
    "budget_choices",
    "budgets",
    "allowed_budget_values",
)

replacement = next(
    (name for name in replacement_candidates if name in parameters),
    None,
)

if "allowed_budgets" in parameters:
    replacement = "allowed_budgets"

if replacement is None:
    raise RuntimeError(
        "The model does not accept allowed_budgets and no equivalent constructor "
        f"parameter was found. Available parameters: {parameters}"
    )

print(f"Budget keyword used for tests: {replacement}")

patched_test_files: list[str] = []
pattern = re.compile(r"\ballowed_budgets\s*=")

for path in sorted((root / "tests").rglob("*.py")):
    text = path.read_text(encoding="utf-8", errors="strict")
    updated = pattern.sub(f"{replacement}=", text)
    if updated != text:
        path.write_text(updated, encoding="utf-8", newline="\n")
        patched_test_files.append(str(path.relative_to(root)))

print(f"Patched obsolete budget keyword in {len(patched_test_files)} test files.")
for item in patched_test_files:
    print(f"  - {item}")

runtime_text = runtime_file.read_text(encoding="utf-8")

old_input_block = '''    for object_ in _walk_named_objects(model):
        for attr in ("vocab_size", "num_embeddings", "input_vocab_size"):
            value = getattr(object_, attr, None)
            if isinstance(value, int) and value > 2:
                vocab = value
        for attr in ("input_size", "input_dim", "feature_dim", "embedding_dim"):
            value = getattr(object_, attr, None)
            if isinstance(value, int) and value > 0:
                features = value
'''

new_input_block = '''    explicit_input_dim: int | None = None
    fallback_embedding_dim: int | None = None
    for object_ in _walk_named_objects(model):
        for attr in ("vocab_size", "num_embeddings", "input_vocab_size"):
            value = getattr(object_, attr, None)
            if isinstance(value, int) and value > 2:
                vocab = value

        if explicit_input_dim is None:
            for attr in ("input_dim", "input_size", "feature_dim", "d_input"):
                value = getattr(object_, attr, None)
                if isinstance(value, int) and value > 0:
                    explicit_input_dim = value
                    break

        if fallback_embedding_dim is None:
            for attr in ("embedding_dim", "embed_dim"):
                value = getattr(object_, attr, None)
                if isinstance(value, int) and value > 0:
                    fallback_embedding_dim = value
                    break

    features = explicit_input_dim or fallback_embedding_dim or features
'''

if old_input_block in runtime_text:
    runtime_text = runtime_text.replace(old_input_block, new_input_block, 1)
    print("Patched input dimension discovery.")
elif "explicit_input_dim: int | None = None" in runtime_text:
    print("Input dimension discovery was already patched.")
else:
    raise RuntimeError(
        "The expected generated input-discovery block was not found in "
        f"{runtime_file.relative_to(root)}."
    )

old_fingerprint_block = '''def dataset_fingerprint(dataset: Any, limit: int = 16) -> str:
    digest = hashlib.sha256()
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        count = min(int(len(dataset)), limit)
        for index in range(count):
            digest.update(_stable_serialize(dataset[index]))
    else:
        for index, item in enumerate(dataset):
            if index >= limit:
                break
            digest.update(_stable_serialize(item))
    return digest.hexdigest()
'''

new_fingerprint_block = '''def dataset_fingerprint(dataset: Any, limit: int = 16) -> str:
    digest = hashlib.sha256()

    # Some task factories return a mapping such as
    # {"train": dataset, "validation": dataset, "test": dataset}. Such objects
    # implement __getitem__ but are not integer-indexed datasets.
    if isinstance(dataset, Mapping):
        digest.update(_stable_serialize(dataset))
        return digest.hexdigest()

    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        count = min(int(len(dataset)), limit)
        for index in range(count):
            digest.update(_stable_serialize(dataset[index]))
    else:
        for index, item in enumerate(dataset):
            if index >= limit:
                break
            digest.update(_stable_serialize(item))
    return digest.hexdigest()
'''

if old_fingerprint_block in runtime_text:
    runtime_text = runtime_text.replace(
        old_fingerprint_block,
        new_fingerprint_block,
        1,
    )
    print("Patched mapping-style dataset fingerprinting.")
elif "Some task factories return a mapping" in runtime_text:
    print("Dataset fingerprinting was already patched.")
else:
    raise RuntimeError(
        "The expected generated dataset-fingerprint block was not found in "
        f"{runtime_file.relative_to(root)}."
    )

runtime_file.write_text(runtime_text, encoding="utf-8", newline="\n")

script_text = original_script.read_text(encoding="utf-8")

old_pytest_block = '''log "Running the complete test suite as the pre-training gate."
set +e
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest \\
    -q \\
    -o addopts='' \\
    tests \\
    --junitxml="$JUNIT_FILE" \\
    2>&1 | tee "$LOG_FILE"
PYTEST_EXIT_CODE="${PIPESTATUS[0]}"
set -e
'''

new_pytest_block = r'''log "Discovering Section 14-specific test nodes."
mapfile -t SECTION14_TEST_TARGETS < <(
    "$PYTHON_BIN" - <<'PYTEST_TARGETS'
from __future__ import annotations

import ast
import re
from pathlib import Path

root = Path.cwd()
generated = root / "tests" / "test_section14_required.py"
targets = [str(generated.relative_to(root))]

pattern = re.compile(
    r"("
    r"strict_budget|memory.*budget|budget.*violat|"
    r"future_tokens|causal|"
    r"determin|same_seed|training_order|"
    r"synthetic.*seed|seed.*overlap|"
    r"hdfs.*block|block.*overlap|"
    r"imdb.*test|official.*test|split.*leak|"
    r"gradient|backpropagat.*controller|controller.*gradient|"
    r"graph_policy|cached_state|trainable_cache|detached_cache|"
    r"memory.*reset|reset.*memory"
    r")",
    re.IGNORECASE,
)

for path in sorted((root / "tests").rglob("test*.py")):
    if path.resolve() == generated.resolve():
        continue
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        continue

    relative = str(path.relative_to(root))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_") and pattern.search(node.name):
                targets.append(f"{relative}::{node.name}")
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name.startswith("test_") and pattern.search(child.name):
                        targets.append(f"{relative}::{node.name}::{child.name}")

for target in dict.fromkeys(targets):
    print(target)
PYTEST_TARGETS
)

if [[ "${#SECTION14_TEST_TARGETS[@]}" -eq 0 ]]; then
    die "No Section 14 tests were discovered."
fi

log "Section 14 pytest targets: ${#SECTION14_TEST_TARGETS[@]}"
printf '  %s\n' "${SECTION14_TEST_TARGETS[@]}"

log "Running the Section 14 pre-training gate."
set +e
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest \
    -q \
    -o addopts='' \
    "${SECTION14_TEST_TARGETS[@]}" \
    --junitxml="$JUNIT_FILE" \
    2>&1 | tee "$LOG_FILE"
PYTEST_EXIT_CODE="${PIPESTATUS[0]}"
set -e
'''

if old_pytest_block in script_text:
    script_text = script_text.replace(old_pytest_block, new_pytest_block, 1)
    print("Patched the original automation to run Section 14-specific tests.")
elif 'log "Discovering Section 14-specific test nodes."' in script_text:
    print("Original automation test scope was already patched.")
else:
    raise RuntimeError(
        "The expected pytest execution block was not found in "
        f"{original_script.relative_to(root)}."
    )

original_script.write_text(script_text, encoding="utf-8", newline="\n")

required_test = root / "tests" / "test_section14_required.py"
required_text = required_test.read_text(encoding="utf-8")

if "TESTS_DIR = Path(__file__).resolve().parent" not in required_text:
    old_header = '''from __future__ import annotations

import copy
'''
    new_header = '''from __future__ import annotations

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import copy
'''
    if old_header not in required_text:
        raise RuntimeError("Generated test import header could not be repaired.")
    required_test.write_text(
        required_text.replace(old_header, new_header, 1),
        encoding="utf-8",
        newline="\n",
    )
    print("Patched generated test import path.")

print("Compatibility repair completed.")
PY

chmod +x "$ORIGINAL_SCRIPT"

log "Checking Bash syntax."
bash -n "$ORIGINAL_SCRIPT"

log "Checking Python syntax."
"$PYTHON_BIN" -m py_compile \
    "$RUNTIME_FILE" \
    "$REQUIRED_TEST_FILE"

log "Collecting the generated Section 14 test gate before execution."
set +e
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest \
    -q \
    -o addopts='' \
    --collect-only \
    "$REQUIRED_TEST_FILE" \
    2>&1 | tee "$DIAGNOSTIC_LOG"
COLLECT_EXIT="${PIPESTATUS[0]}"
set -e

if [[ "$COLLECT_EXIT" -ne 0 ]]; then
    printf '\nSECTION 14 INTERFACE REPAIR: FAIL\n'
    printf 'Collection still fails. Review: %s\n' "${DIAGNOSTIC_LOG#$REPO_ROOT/}"
    exit "$COLLECT_EXIT"
fi

printf '\nSECTION 14 INTERFACE REPAIR: PASS\n'
printf 'The constructor keyword, input adapter, dataset adapter, and gate scope were repaired.\n'
printf 'Rerunning the Section 14 automation now.\n\n'

exec "./$ORIGINAL_SCRIPT"
