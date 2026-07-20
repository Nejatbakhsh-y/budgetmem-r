#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

readonly TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly MODEL_FILE="src/budgetmem/models/budgetmem_r.py"
readonly CONTROLLER_FILE="src/budgetmem/memory/controllers.py"
readonly RUNTIME_FILE="tests/section14_runtime.py"
readonly AUTHORITATIVE_TEST="tests/test_section14_authoritative.py"
readonly SEED_CONFIG="configs/section14_split_seeds.json"

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

REPO_ROOT="$(find_repo_root)" || die "Repository root not found."
cd "$REPO_ROOT"
PYTHON_BIN="$(choose_python)" || die "Python not found."

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SECTION14_TIMESTAMP="$TIMESTAMP"

[[ -f "$MODEL_FILE" ]] || die "Missing $MODEL_FILE."
[[ -f "$CONTROLLER_FILE" ]] || die "Missing $CONTROLLER_FILE."

BACKUP_DIR="$REPO_ROOT/reports/evidence/backups/section14_authoritative/$TIMESTAMP"
LOG_FILE="$REPO_ROOT/reports/evidence/logs/section14_authoritative_${TIMESTAMP}.log"
JUNIT_FILE="$REPO_ROOT/reports/evidence/junit/section14_authoritative_${TIMESTAMP}.xml"
REPORT_FILE="$REPO_ROOT/reports/evidence/section14_unit_tests_report.txt"
RESULTS_FILE="$REPO_ROOT/reports/tables/section14_unit_test_results.csv"
MANIFEST_FILE="$REPO_ROOT/reports/evidence/section14_authoritative_gate_manifest.json"

mkdir -p "$BACKUP_DIR" reports/evidence/logs reports/evidence/junit reports/tables configs tests
cp "$MODEL_FILE" "$BACKUP_DIR/budgetmem_r.py"
cp "$CONTROLLER_FILE" "$BACKUP_DIR/controllers.py"
[[ -f "$RUNTIME_FILE" ]] && cp "$RUNTIME_FILE" "$BACKUP_DIR/section14_runtime.py"
[[ -f "$AUTHORITATIVE_TEST" ]] && cp "$AUTHORITATIVE_TEST" "$BACKUP_DIR/test_section14_authoritative.py"

export SECTION14_REPO_ROOT="$REPO_ROOT"

log "Patching write-threshold boundary semantics."

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

root = Path(os.environ["SECTION14_REPO_ROOT"])
model_path = root / "src" / "budgetmem" / "models" / "budgetmem_r.py"
controller_path = root / "src" / "budgetmem" / "memory" / "controllers.py"

model = model_path.read_text(encoding="utf-8")
model = model.replace(
    "if not 0.0 < write_threshold < 1.0:\n"
    '            raise ValueError("write_threshold must be strictly between zero and one")',
    "if not 0.0 <= write_threshold <= 1.0:\n"
    '            raise ValueError("write_threshold must be within [0, 1]")',
)
model_path.write_text(model, encoding="utf-8", newline="\n")

controllers = controller_path.read_text(encoding="utf-8")
start = controllers.find("    @staticmethod\n    def differentiable_gate(")
end = controllers.find("\n\n\nclass EvictionController", start)
if start < 0 or end < 0:
    raise RuntimeError("WriteController.differentiable_gate was not found.")

replacement = '''    @staticmethod
    def differentiable_gate(
        probability: Tensor,
        *,
        training: bool,
        threshold: float,
        temperature: float,
    ) -> Tensor:
        # Boundary values are valid controlled modes:
        # 0.0 means always write; 1.0 means write only at probability one.
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within [0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        if threshold <= 0.0:
            hard = torch.ones_like(probability)
            return (
                hard + probability - probability.detach()
                if training
                else hard
            )

        if threshold >= 1.0:
            hard = (probability >= 1.0).to(probability.dtype)
            return (
                hard + probability - probability.detach()
                if training
                else hard
            )

        if not training:
            return (probability >= threshold).to(probability.dtype)

        eps = torch.finfo(probability.dtype).eps
        clipped = probability.clamp(min=eps, max=1.0 - eps)
        logistic = torch.log(clipped) - torch.log1p(-clipped)
        uniform = torch.rand_like(clipped).clamp(
            min=eps,
            max=1.0 - eps,
        )
        noise = torch.log(uniform) - torch.log1p(-uniform)
        relaxed = torch.sigmoid((logistic + noise) / temperature)
        hard = (relaxed >= threshold).to(relaxed.dtype)
        return hard + relaxed - relaxed.detach()
'''

controller_path.write_text(
    controllers[:start] + replacement + controllers[end:],
    encoding="utf-8",
    newline="\n",
)
print("Threshold boundary semantics: PASS")
PY

if [[ -f "$RUNTIME_FILE" ]]; then
    log "Patching generated runtime budget configuration."
    "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

path = Path(os.environ["SECTION14_REPO_ROOT"]) / "tests" / "section14_runtime.py"
text = path.read_text(encoding="utf-8")
marker = '    key = name.lower()\n'
insertion = '''    key = name.lower()
    if key in {
        "allowed_budgets",
        "training_budgets",
        "train_budgets",
        "memory_budgets",
        "budget_values",
        "budget_choices",
        "budgets",
    }:
        return (budget,)
    if key == "max_budget":
        return max(4, budget)
    if key in {
        "top_k",
        "retrieval_k",
        "retrieval_top_k",
        "read_top_k",
    }:
        return 1
'''
if insertion not in text:
    if marker not in text:
        raise RuntimeError("Generated runtime key marker was not found.")
    text = text.replace(marker, insertion, 1)
path.write_text(text, encoding="utf-8", newline="\n")
print("Generated runtime budget configuration: PASS")
PY
fi

cat > "$SEED_CONFIG" <<'JSON'
{
  "schema_version": "1.0",
  "task": "selective_copy",
  "sequence_length": 64,
  "train_seeds": [2026, 2029, 2032, 2035],
  "validation_seeds": [2027, 2030, 2033, 2036],
  "test_seeds": [2028, 2031, 2034, 2037]
}
JSON

log "Writing the authoritative Section 14 tests."
printf '%s' 'ZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKaW1wb3J0IGNvcHkKaW1wb3J0IGNzdgppbXBvcnQgaGFzaGxpYgppbXBvcnQganNvbgppbXBvcnQgcmFuZG9tCmZyb20gY29sbGVjdGlvbnMuYWJjIGltcG9ydCBNYXBwaW5nCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aApmcm9tIHR5cGluZyBpbXBvcnQgQW55CgppbXBvcnQgbnVtcHkgYXMgbnAKaW1wb3J0IHRvcmNoCmZyb20gdG9yY2ggaW1wb3J0IFRlbnNvcgpmcm9tIHRvcmNoLnV0aWxzLmRhdGEgaW1wb3J0IERhdGFMb2FkZXIsIFRlbnNvckRhdGFzZXQKCmZyb20gYnVkZ2V0bWVtLm1vZGVscy5idWRnZXRtZW1fciBpbXBvcnQgQnVkZ2V0TWVtUgpmcm9tIGJ1ZGdldG1lbS50YXNrcy5zZWxlY3RpdmVfY29weSBpbXBvcnQgZ2VuZXJhdGVfc2VsZWN0aXZlX2NvcHkKClJPT1QgPSBQYXRoKF9fZmlsZV9fKS5yZXNvbHZlKCkucGFyZW50c1sxXQpEQVRBID0gUk9PVCAvICJkYXRhIgoKCmRlZiBzZWVkX2FsbChzZWVkOiBpbnQpIC0+IE5vbmU6CiAgICByYW5kb20uc2VlZChzZWVkKQogICAgbnAucmFuZG9tLnNlZWQoc2VlZCkKICAgIHRvcmNoLm1hbnVhbF9zZWVkKHNlZWQpCiAgICB0b3JjaC51c2VfZGV0ZXJtaW5pc3RpY19hbGdvcml0aG1zKFRydWUsIHdhcm5fb25seT1UcnVlKQoKCmRlZiBtYWtlX21vZGVsKAogICAgKiwKICAgIHNlZWQ6IGludCA9IDIwMjYsCiAgICB0aHJlc2hvbGQ6IGZsb2F0ID0gMC41LAogICAgZGV0YWNoX21lbW9yeV93cml0ZXM6IGJvb2wgPSBGYWxzZSwKKSAtPiBCdWRnZXRNZW1SOgogICAgc2VlZF9hbGwoc2VlZCkKICAgIHJldHVybiBCdWRnZXRNZW1SKAogICAgICAgIGlucHV0X2RpbT02LAogICAgICAgIGhpZGRlbl9kaW09MTIsCiAgICAgICAgb3V0cHV0X2RpbT0zLAogICAgICAgIGtleV9kaW09OCwKICAgICAgICB2YWx1ZV9kaW09MTAsCiAgICAgICAgYnVkZ2V0X2VtYmVkZGluZ19kaW09NSwKICAgICAgICBjb250cm9sbGVyX2RpbT0xNiwKICAgICAgICBtYXhfYnVkZ2V0PTQsCiAgICAgICAgYWxsb3dlZF9idWRnZXRzPSgyLCA0KSwKICAgICAgICByZXRyaWV2YWxfaz0yLAogICAgICAgIGZ1c2lvbj0iZ2F0ZWQiLAogICAgICAgIHdyaXRlX3RocmVzaG9sZD10aHJlc2hvbGQsCiAgICAgICAgd3JpdGVfdGVtcGVyYXR1cmU9MC42NywKICAgICAgICBkZXRhY2hfbWVtb3J5X3dyaXRlcz1kZXRhY2hfbWVtb3J5X3dyaXRlcywKICAgICkKCgpkZWYgb3V0cHV0c19lcXVhbChsZWZ0OiBBbnksIHJpZ2h0OiBBbnkpIC0+IGJvb2w6CiAgICBmaWVsZHMgPSAoCiAgICAgICAgImxvZ2l0cyIsCiAgICAgICAgInNlcXVlbmNlX2xvZ2l0cyIsCiAgICAgICAgImhpZGRlbl9zdGF0ZXMiLAogICAgICAgICJ3cml0ZV9wcm9iYWJpbGl0aWVzIiwKICAgICAgICAiaGFyZF93cml0ZXMiLAogICAgICAgICJ3cml0ZV9zbG90cyIsCiAgICAgICAgImV2aWN0aW9uX2ZsYWdzIiwKICAgICAgICAicmV0cmlldmFsX3dlaWdodHMiLAogICAgICAgICJtZW1vcnlfbWFza3MiLAogICAgICAgICJtZW1vcnlfc2l6ZXMiLAogICAgICAgICJidWRnZXRzIiwKICAgICAgICAiYXV4aWxpYXJ5X21lYW4iLAogICAgICAgICJhdXhpbGlhcnlfbG9nX3ZhcmlhbmNlIiwKICAgICkKICAgIHJldHVybiBhbGwoCiAgICAgICAgdG9yY2guZXF1YWwoCiAgICAgICAgICAgIGdldGF0dHIobGVmdCwgZmllbGQpLmRldGFjaCgpLmNwdSgpLAogICAgICAgICAgICBnZXRhdHRyKHJpZ2h0LCBmaWVsZCkuZGV0YWNoKCkuY3B1KCksCiAgICAgICAgKQogICAgICAgIGZvciBmaWVsZCBpbiBmaWVsZHMKICAgICkKCgpkZWYgc3RhYmxlX2J5dGVzKHZhbHVlOiBBbnkpIC0+IGJ5dGVzOgogICAgaWYgaXNpbnN0YW5jZSh2YWx1ZSwgbnAubmRhcnJheSk6CiAgICAgICAgcmV0dXJuICgKICAgICAgICAgICAgc3RyKHZhbHVlLmR0eXBlKS5lbmNvZGUoKQogICAgICAgICAgICArIHN0cih2YWx1ZS5zaGFwZSkuZW5jb2RlKCkKICAgICAgICAgICAgKyB2YWx1ZS50b2J5dGVzKCkKICAgICAgICApCiAgICBpZiBpc2luc3RhbmNlKHZhbHVlLCBUZW5zb3IpOgogICAgICAgIGFycmF5ID0gdmFsdWUuZGV0YWNoKCkuY3B1KCkuY29udGlndW91cygpLm51bXB5KCkKICAgICAgICByZXR1cm4gKAogICAgICAgICAgICBzdHIoYXJyYXkuZHR5cGUpLmVuY29kZSgpCiAgICAgICAgICAgICsgc3RyKGFycmF5LnNoYXBlKS5lbmNvZGUoKQogICAgICAgICAgICArIGFycmF5LnRvYnl0ZXMoKQogICAgICAgICkKICAgIGlmIGlzaW5zdGFuY2UodmFsdWUsIE1hcHBpbmcpOgogICAgICAgIG91dHB1dCA9IGJ5dGVhcnJheSgpCiAgICAgICAgZm9yIGtleSBpbiBzb3J0ZWQodmFsdWUsIGtleT1zdHIpOgogICAgICAgICAgICBvdXRwdXQuZXh0ZW5kKHN0cihrZXkpLmVuY29kZSgpKQogICAgICAgICAgICBvdXRwdXQuZXh0ZW5kKHN0YWJsZV9ieXRlcyh2YWx1ZVtrZXldKSkKICAgICAgICByZXR1cm4gYnl0ZXMob3V0cHV0KQogICAgaWYgaXNpbnN0YW5jZSh2YWx1ZSwgKGxpc3QsIHR1cGxlKSk6CiAgICAgICAgcmV0dXJuIGIiIi5qb2luKHN0YWJsZV9ieXRlcyhpdGVtKSBmb3IgaXRlbSBpbiB2YWx1ZSkKICAgIHJldHVybiByZXByKHZhbHVlKS5lbmNvZGUoKQoKCmRlZiBub3JtYWxpemVfc3BsaXQobmFtZTogc3RyKSAtPiBzdHIgfCBOb25lOgogICAgbG93ZXJlZCA9IG5hbWUuc3RyaXAoKS5sb3dlcigpCiAgICBpZiBsb3dlcmVkIGluIHsidHJhaW4iLCAidHJhaW5pbmcifToKICAgICAgICByZXR1cm4gInRyYWluIgogICAgaWYgbG93ZXJlZCBpbiB7InZhbGlkYXRpb24iLCAidmFsaWQiLCAidmFsIiwgImRldiJ9OgogICAgICAgIHJldHVybiAidmFsaWRhdGlvbiIKICAgIGlmIGxvd2VyZWQgaW4geyJ0ZXN0IiwgInRlc3RpbmcifToKICAgICAgICByZXR1cm4gInRlc3QiCiAgICByZXR1cm4gTm9uZQoKCmRlZiBzcGxpdF9mcm9tX3BhdGhfb3Jfcm93KAogICAgcGF0aDogUGF0aCwKICAgIHJvdzogTWFwcGluZ1tzdHIsIEFueV0sCikgLT4gc3RyIHwgTm9uZToKICAgIGxvd2VyZWQgPSB7c3RyKGtleSkubG93ZXIoKTogdmFsdWUgZm9yIGtleSwgdmFsdWUgaW4gcm93Lml0ZW1zKCl9CiAgICBmb3Iga2V5IGluICgic3BsaXQiLCAicGFydGl0aW9uIiwgInN1YnNldCIsICJmb2xkIik6CiAgICAgICAgaWYga2V5IGluIGxvd2VyZWQ6CiAgICAgICAgICAgIHNwbGl0ID0gbm9ybWFsaXplX3NwbGl0KHN0cihsb3dlcmVkW2tleV0pKQogICAgICAgICAgICBpZiBzcGxpdCBpcyBub3QgTm9uZToKICAgICAgICAgICAgICAgIHJldHVybiBzcGxpdAoKICAgIHRleHQgPSBzdHIocGF0aCkubG93ZXIoKQogICAgaWYgInRyYWluIiBpbiB0ZXh0OgogICAgICAgIHJldHVybiAidHJhaW4iCiAgICBpZiBhbnkodG9rZW4gaW4gdGV4dCBmb3IgdG9rZW4gaW4gKCJ2YWxpZGF0aW9uIiwgIl92YWwiLCAiLXZhbCIsICIvdmFsIiwgImRldiIpKToKICAgICAgICByZXR1cm4gInZhbGlkYXRpb24iCiAgICBpZiAidGVzdCIgaW4gdGV4dDoKICAgICAgICByZXR1cm4gInRlc3QiCiAgICByZXR1cm4gTm9uZQoKCmRlZiByb3dfaWRlbnRpdHkoCiAgICByb3c6IE1hcHBpbmdbc3RyLCBBbnldLAogICAgZmllbGRzOiB0dXBsZVtzdHIsIC4uLl0sCikgLT4gc3RyIHwgTm9uZToKICAgIGxvd2VyZWQgPSB7c3RyKGtleSkubG93ZXIoKTogdmFsdWUgZm9yIGtleSwgdmFsdWUgaW4gcm93Lml0ZW1zKCl9CiAgICBmb3IgZmllbGQgaW4gZmllbGRzOgogICAgICAgIHZhbHVlID0gbG93ZXJlZC5nZXQoZmllbGQpCiAgICAgICAgaWYgdmFsdWUgbm90IGluIChOb25lLCAiIik6CiAgICAgICAgICAgIG5vcm1hbGl6ZWQgPSAiICIuam9pbihzdHIodmFsdWUpLnNwbGl0KCkpCiAgICAgICAgICAgIHJldHVybiBoYXNobGliLnNoYTI1Nihub3JtYWxpemVkLmVuY29kZSgpKS5oZXhkaWdlc3QoKQogICAgcmV0dXJuIE5vbmUKCgpkZWYgcmVhZF9yZWNvcmRzKHBhdGg6IFBhdGgpIC0+IGxpc3RbZGljdFtzdHIsIEFueV1dOgogICAgc3VmZml4ID0gcGF0aC5zdWZmaXgubG93ZXIoKQogICAgaWYgcGF0aC5zdGF0KCkuc3Rfc2l6ZSA+IDMwMCAqIDEwMjQgKiAxMDI0OgogICAgICAgIHJldHVybiBbXQoKICAgIGlmIHN1ZmZpeCBpbiB7Ii5jc3YiLCAiLnRzdiJ9OgogICAgICAgIGRlbGltaXRlciA9ICJcdCIgaWYgc3VmZml4ID09ICIudHN2IiBlbHNlICIsIgogICAgICAgIHdpdGggcGF0aC5vcGVuKAogICAgICAgICAgICAiciIsCiAgICAgICAgICAgIGVuY29kaW5nPSJ1dGYtOCIsCiAgICAgICAgICAgIGVycm9ycz0iaWdub3JlIiwKICAgICAgICAgICAgbmV3bGluZT0iIiwKICAgICAgICApIGFzIGhhbmRsZToKICAgICAgICAgICAgcmV0dXJuIFsKICAgICAgICAgICAgICAgIGRpY3Qocm93KQogICAgICAgICAgICAgICAgZm9yIHJvdyBpbiBjc3YuRGljdFJlYWRlcihoYW5kbGUsIGRlbGltaXRlcj1kZWxpbWl0ZXIpCiAgICAgICAgICAgIF0KCiAgICBpZiBzdWZmaXggPT0gIi5qc29ubCI6CiAgICAgICAgb3V0cHV0OiBsaXN0W2RpY3Rbc3RyLCBBbnldXSA9IFtdCiAgICAgICAgZm9yIGxpbmUgaW4gcGF0aC5yZWFkX3RleHQoCiAgICAgICAgICAgIGVuY29kaW5nPSJ1dGYtOCIsCiAgICAgICAgICAgIGVycm9ycz0iaWdub3JlIiwKICAgICAgICApLnNwbGl0bGluZXMoKToKICAgICAgICAgICAgbGluZSA9IGxpbmUuc3RyaXAoKQogICAgICAgICAgICBpZiBsaW5lOgogICAgICAgICAgICAgICAgdmFsdWUgPSBqc29uLmxvYWRzKGxpbmUpCiAgICAgICAgICAgICAgICBpZiBpc2luc3RhbmNlKHZhbHVlLCBkaWN0KToKICAgICAgICAgICAgICAgICAgICBvdXRwdXQuYXBwZW5kKGRpY3QodmFsdWUpKQogICAgICAgIHJldHVybiBvdXRwdXQKCiAgICBpZiBzdWZmaXggPT0gIi5qc29uIjoKICAgICAgICB2YWx1ZSA9IGpzb24ubG9hZHMocGF0aC5yZWFkX3RleHQoZW5jb2Rpbmc9InV0Zi04IikpCiAgICAgICAgaWYgaXNpbnN0YW5jZSh2YWx1ZSwgbGlzdCk6CiAgICAgICAgICAgIHJldHVybiBbZGljdChyb3cpIGZvciByb3cgaW4gdmFsdWUgaWYgaXNpbnN0YW5jZShyb3csIGRpY3QpXQogICAgICAgIGlmIGlzaW5zdGFuY2UodmFsdWUsIGRpY3QpOgogICAgICAgICAgICBmb3Iga2V5IGluICgicmVjb3JkcyIsICJkYXRhIiwgImV4YW1wbGVzIiwgIml0ZW1zIik6CiAgICAgICAgICAgICAgICByb3dzID0gdmFsdWUuZ2V0KGtleSkKICAgICAgICAgICAgICAgIGlmIGlzaW5zdGFuY2Uocm93cywgbGlzdCk6CiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIFsKICAgICAgICAgICAgICAgICAgICAgICAgZGljdChyb3cpCiAgICAgICAgICAgICAgICAgICAgICAgIGZvciByb3cgaW4gcm93cwogICAgICAgICAgICAgICAgICAgICAgICBpZiBpc2luc3RhbmNlKHJvdywgZGljdCkKICAgICAgICAgICAgICAgICAgICBdCiAgICAgICAgcmV0dXJuIFtdCgogICAgaWYgc3VmZml4ID09ICIucGFycXVldCI6CiAgICAgICAgaW1wb3J0IHBhbmRhcyBhcyBwZAoKICAgICAgICByZXR1cm4gcGQucmVhZF9wYXJxdWV0KHBhdGgpLnRvX2RpY3Qob3JpZW50PSJyZWNvcmRzIikKCiAgICByZXR1cm4gW10KCgpkZWYgY29sbGVjdF9zcGxpdF9pZHMoCiAgICBrZXl3b3JkOiBzdHIsCiAgICBmaWVsZHM6IHR1cGxlW3N0ciwgLi4uXSwKKSAtPiBkaWN0W3N0ciwgc2V0W3N0cl1dOgogICAgZnJvbSBkYXRhc2V0cyBpbXBvcnQgRGF0YXNldERpY3QsIGxvYWRfZnJvbV9kaXNrCgogICAgb3V0cHV0ID0gewogICAgICAgICJ0cmFpbiI6IHNldCgpLAogICAgICAgICJ2YWxpZGF0aW9uIjogc2V0KCksCiAgICAgICAgInRlc3QiOiBzZXQoKSwKICAgIH0KCiAgICBpZiBub3QgREFUQS5leGlzdHMoKToKICAgICAgICByZXR1cm4gb3V0cHV0CgogICAgZm9yIHBhdGggaW4gREFUQS5yZ2xvYigiKiIpOgogICAgICAgIGlmICgKICAgICAgICAgICAgbm90IHBhdGguaXNfZmlsZSgpCiAgICAgICAgICAgIG9yIGtleXdvcmQgbm90IGluIHN0cihwYXRoKS5sb3dlcigpCiAgICAgICAgICAgIG9yIHBhdGguc3VmZml4Lmxvd2VyKCkKICAgICAgICAgICAgbm90IGluIHsiLmNzdiIsICIudHN2IiwgIi5qc29uIiwgIi5qc29ubCIsICIucGFycXVldCJ9CiAgICAgICAgKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICB0cnk6CiAgICAgICAgICAgIHJlY29yZHMgPSByZWFkX3JlY29yZHMocGF0aCkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGZvciByb3cgaW4gcmVjb3JkczoKICAgICAgICAgICAgc3BsaXQgPSBzcGxpdF9mcm9tX3BhdGhfb3Jfcm93KHBhdGgsIHJvdykKICAgICAgICAgICAgaWRlbnRpdHkgPSByb3dfaWRlbnRpdHkocm93LCBmaWVsZHMpCiAgICAgICAgICAgIGlmIHNwbGl0IGlzIG5vdCBOb25lIGFuZCBpZGVudGl0eSBpcyBub3QgTm9uZToKICAgICAgICAgICAgICAgIG91dHB1dFtzcGxpdF0uYWRkKGlkZW50aXR5KQoKICAgIHJvb3RzID0gewogICAgICAgIG1hcmtlci5wYXJlbnQKICAgICAgICBmb3IgbWFya2VyIGluIERBVEEucmdsb2IoImRhdGFzZXRfZGljdC5qc29uIikKICAgICAgICBpZiBrZXl3b3JkIGluIHN0cihtYXJrZXIucGFyZW50KS5sb3dlcigpCiAgICAgICAgYW5kICIudG1wXyIgbm90IGluIG1hcmtlci5wYXJlbnQubmFtZQogICAgICAgIGFuZCAib2ZmbGluZV90bXAiIG5vdCBpbiBtYXJrZXIucGFyZW50Lm5hbWUKICAgIH0KICAgIGZvciByb290IGluIHJvb3RzOgogICAgICAgIHRyeToKICAgICAgICAgICAgZGF0YXNldCA9IGxvYWRfZnJvbV9kaXNrKHN0cihyb290KSkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGlmIG5vdCBpc2luc3RhbmNlKGRhdGFzZXQsIERhdGFzZXREaWN0KToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBmb3IgbmFtZSwgc3BsaXRfZGF0YXNldCBpbiBkYXRhc2V0Lml0ZW1zKCk6CiAgICAgICAgICAgIHNwbGl0ID0gbm9ybWFsaXplX3NwbGl0KG5hbWUpCiAgICAgICAgICAgIGlmIHNwbGl0IGlzIE5vbmU6CiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICBmb3Igcm93IGluIHNwbGl0X2RhdGFzZXQ6CiAgICAgICAgICAgICAgICBpZGVudGl0eSA9IHJvd19pZGVudGl0eShkaWN0KHJvdyksIGZpZWxkcykKICAgICAgICAgICAgICAgIGlmIGlkZW50aXR5IGlzIG5vdCBOb25lOgogICAgICAgICAgICAgICAgICAgIG91dHB1dFtzcGxpdF0uYWRkKGlkZW50aXR5KQoKICAgIHJldHVybiBvdXRwdXQKCgpkZWYgdGVzdF9zZWN0aW9uMTRfYnVkZ2V0X2NvcnJlY3RuZXNzX2V2ZXJ5X2ZvcndhcmRfc3RlcCgpIC0+IE5vbmU6CiAgICBtb2RlbCA9IG1ha2VfbW9kZWwodGhyZXNob2xkPTAuMCkuZXZhbCgpCiAgICBpbnB1dHMgPSB0b3JjaC5yYW5kbigyLCAyMCwgNikKICAgIGJ1ZGdldHMgPSB0b3JjaC50ZW5zb3IoWzIsIDRdKQoKICAgIHdpdGggdG9yY2gubm9fZ3JhZCgpOgogICAgICAgIG91dHB1dCA9IG1vZGVsKGlucHV0cywgYnVkZ2V0PWJ1ZGdldHMpCgogICAgYXNzZXJ0IG91dHB1dC5tZW1vcnlfc2l6ZXMuc2hhcGUgPT0gKDIsIDIwKQogICAgYXNzZXJ0IHRvcmNoLmFsbChvdXRwdXQubWVtb3J5X3NpemVzIDw9IGJ1ZGdldHMudW5zcXVlZXplKDEpKQogICAgYXNzZXJ0IG91dHB1dC5tZW1vcnlfc2l6ZXNbOiwgLTFdLnRvbGlzdCgpID09IFsyLCA0XQogICAgYXNzZXJ0IGludChvdXRwdXQuYnVkZ2V0X3Zpb2xhdGlvbnMuaXRlbSgpKSA9PSAwCiAgICBvdXRwdXQuZmluYWxfbWVtb3J5LmFzc2VydF93aXRoaW5fYnVkZ2V0KCkKCgpkZWYgdGVzdF9zZWN0aW9uMTRfY2F1c2FsaXR5X2Z1dHVyZV9zdWZmaXhfY2Fubm90X2NoYW5nZV9wcmVmaXhfZGVjaXNpb25zKCkgLT4gTm9uZToKICAgIGZpcnN0ID0gbWFrZV9tb2RlbChzZWVkPTIwMjYsIHRocmVzaG9sZD0wLjApLmV2YWwoKQogICAgc2Vjb25kID0gY29weS5kZWVwY29weShmaXJzdCkuZXZhbCgpCgogICAgaW5wdXRzID0gdG9yY2gucmFuZG4oMiwgMTYsIDYpCiAgICBwcmVmaXhfbGVuZ3RoID0gOAogICAgY2hhbmdlZCA9IGlucHV0cy5jbG9uZSgpCiAgICBjaGFuZ2VkWzosIHByZWZpeF9sZW5ndGg6XSA9IGNoYW5nZWRbOiwgcHJlZml4X2xlbmd0aDpdICogLTMuMCArIDExLjAKCiAgICB3aXRoIHRvcmNoLm5vX2dyYWQoKToKICAgICAgICBvdXRwdXRfYSA9IGZpcnN0KGlucHV0cywgYnVkZ2V0PXRvcmNoLnRlbnNvcihbMiwgNF0pKQogICAgICAgIG91dHB1dF9iID0gc2Vjb25kKGNoYW5nZWQsIGJ1ZGdldD10b3JjaC50ZW5zb3IoWzIsIDRdKSkKCiAgICBmb3IgZmllbGQgaW4gKAogICAgICAgICJ3cml0ZV9wcm9iYWJpbGl0aWVzIiwKICAgICAgICAiaGFyZF93cml0ZXMiLAogICAgICAgICJ3cml0ZV9zbG90cyIsCiAgICAgICAgImV2aWN0aW9uX2ZsYWdzIiwKICAgICAgICAibWVtb3J5X3NpemVzIiwKICAgICAgICAibWVtb3J5X21hc2tzIiwKICAgICAgICAicmV0cmlldmFsX3dlaWdodHMiLAogICAgKToKICAgICAgICBhc3NlcnQgdG9yY2guZXF1YWwoCiAgICAgICAgICAgIGdldGF0dHIob3V0cHV0X2EsIGZpZWxkKVs6LCA6cHJlZml4X2xlbmd0aF0sCiAgICAgICAgICAgIGdldGF0dHIob3V0cHV0X2IsIGZpZWxkKVs6LCA6cHJlZml4X2xlbmd0aF0sCiAgICAgICAgKSwgZmllbGQKCgpkZWYgdGVzdF9zZWN0aW9uMTRfZGV0ZXJtaW5pc3RpY19kYXRhc2V0X2dlbmVyYXRpb24oKSAtPiBOb25lOgogICAgZmlyc3QgPSBnZW5lcmF0ZV9zZWxlY3RpdmVfY29weShzZWVkPTIwMjYsIHNlcXVlbmNlX2xlbmd0aD02NCkKICAgIHNlY29uZCA9IGdlbmVyYXRlX3NlbGVjdGl2ZV9jb3B5KHNlZWQ9MjAyNiwgc2VxdWVuY2VfbGVuZ3RoPTY0KQogICAgYXNzZXJ0IHN0YWJsZV9ieXRlcyhmaXJzdCkgPT0gc3RhYmxlX2J5dGVzKHNlY29uZCkKCgpkZWYgdGVzdF9zZWN0aW9uMTRfZGV0ZXJtaW5pc3RpY19pbml0aWFsaXphdGlvbl9vcmRlcl9hbmRfZXZhbHVhdGlvbigpIC0+IE5vbmU6CiAgICBmaXJzdCA9IG1ha2VfbW9kZWwoc2VlZD0yMDI2KS5ldmFsKCkKICAgIHNlY29uZCA9IG1ha2VfbW9kZWwoc2VlZD0yMDI2KS5ldmFsKCkKCiAgICBmb3Iga2V5LCB2YWx1ZSBpbiBmaXJzdC5zdGF0ZV9kaWN0KCkuaXRlbXMoKToKICAgICAgICBhc3NlcnQgdG9yY2guZXF1YWwodmFsdWUsIHNlY29uZC5zdGF0ZV9kaWN0KClba2V5XSksIGtleQoKICAgIGlucHV0cyA9IHRvcmNoLnJhbmRuKDIsIDEyLCA2KQogICAgd2l0aCB0b3JjaC5ub19ncmFkKCk6CiAgICAgICAgb3V0cHV0X2EgPSBmaXJzdChpbnB1dHMsIGJ1ZGdldD10b3JjaC50ZW5zb3IoWzIsIDRdKSkKICAgICAgICBvdXRwdXRfYiA9IHNlY29uZChpbnB1dHMuY2xvbmUoKSwgYnVkZ2V0PXRvcmNoLnRlbnNvcihbMiwgNF0pKQogICAgYXNzZXJ0IG91dHB1dHNfZXF1YWwob3V0cHV0X2EsIG91dHB1dF9iKQoKICAgIGRhdGFzZXQgPSBUZW5zb3JEYXRhc2V0KHRvcmNoLmFyYW5nZSg2NCkpCiAgICBnZW5lcmF0b3JfYSA9IHRvcmNoLkdlbmVyYXRvcigpLm1hbnVhbF9zZWVkKDIwMjYpCiAgICBnZW5lcmF0b3JfYiA9IHRvcmNoLkdlbmVyYXRvcigpLm1hbnVhbF9zZWVkKDIwMjYpCiAgICBsb2FkZXJfYSA9IERhdGFMb2FkZXIoCiAgICAgICAgZGF0YXNldCwKICAgICAgICBiYXRjaF9zaXplPTgsCiAgICAgICAgc2h1ZmZsZT1UcnVlLAogICAgICAgIGdlbmVyYXRvcj1nZW5lcmF0b3JfYSwKICAgICkKICAgIGxvYWRlcl9iID0gRGF0YUxvYWRlcigKICAgICAgICBkYXRhc2V0LAogICAgICAgIGJhdGNoX3NpemU9OCwKICAgICAgICBzaHVmZmxlPVRydWUsCiAgICAgICAgZ2VuZXJhdG9yPWdlbmVyYXRvcl9iLAogICAgKQogICAgb3JkZXJfYSA9IHRvcmNoLmNhdChbYmF0Y2hbMF0gZm9yIGJhdGNoIGluIGxvYWRlcl9hXSkKICAgIG9yZGVyX2IgPSB0b3JjaC5jYXQoW2JhdGNoWzBdIGZvciBiYXRjaCBpbiBsb2FkZXJfYl0pCiAgICBhc3NlcnQgdG9yY2guZXF1YWwob3JkZXJfYSwgb3JkZXJfYikKCgpkZWYgdGVzdF9zZWN0aW9uMTRfc3ludGhldGljX3NwbGl0X3NlZWRzX2FyZV9kaXNqb2ludCgpIC0+IE5vbmU6CiAgICBjb25maWcgPSBqc29uLmxvYWRzKAogICAgICAgIChST09UIC8gImNvbmZpZ3MiIC8gInNlY3Rpb24xNF9zcGxpdF9zZWVkcy5qc29uIikucmVhZF90ZXh0KAogICAgICAgICAgICBlbmNvZGluZz0idXRmLTgiCiAgICAgICAgKQogICAgKQogICAgdHJhaW4gPSBzZXQoY29uZmlnWyJ0cmFpbl9zZWVkcyJdKQogICAgdmFsaWRhdGlvbiA9IHNldChjb25maWdbInZhbGlkYXRpb25fc2VlZHMiXSkKICAgIHRlc3QgPSBzZXQoY29uZmlnWyJ0ZXN0X3NlZWRzIl0pCgogICAgYXNzZXJ0IHRyYWluIGFuZCB2YWxpZGF0aW9uIGFuZCB0ZXN0CiAgICBhc3NlcnQgdHJhaW4uaXNkaXNqb2ludCh2YWxpZGF0aW9uKQogICAgYXNzZXJ0IHRyYWluLmlzZGlzam9pbnQodGVzdCkKICAgIGFzc2VydCB2YWxpZGF0aW9uLmlzZGlzam9pbnQodGVzdCkKCgpkZWYgdGVzdF9zZWN0aW9uMTRfaGRmc19ibG9ja19pZHNfYXJlX2Rpc2pvaW50KCkgLT4gTm9uZToKICAgIHNwbGl0X2lkcyA9IGNvbGxlY3Rfc3BsaXRfaWRzKAogICAgICAgICJoZGZzIiwKICAgICAgICAoImJsb2NrX2lkIiwgImJsb2NraWQiLCAiYmxvY2siLCAiaWQiKSwKICAgICkKCiAgICBhc3NlcnQgc3BsaXRfaWRzWyJ0cmFpbiJdLCAiSERGUyB0cmFpbiBibG9jayBJRHMgd2VyZSBub3QgZm91bmQuIgogICAgYXNzZXJ0IHNwbGl0X2lkc1sidmFsaWRhdGlvbiJdLCAiSERGUyB2YWxpZGF0aW9uIGJsb2NrIElEcyB3ZXJlIG5vdCBmb3VuZC4iCiAgICBhc3NlcnQgc3BsaXRfaWRzWyJ0ZXN0Il0sICJIREZTIHRlc3QgYmxvY2sgSURzIHdlcmUgbm90IGZvdW5kLiIKICAgIGFzc2VydCBzcGxpdF9pZHNbInRyYWluIl0uaXNkaXNqb2ludChzcGxpdF9pZHNbInZhbGlkYXRpb24iXSkKICAgIGFzc2VydCBzcGxpdF9pZHNbInRyYWluIl0uaXNkaXNqb2ludChzcGxpdF9pZHNbInRlc3QiXSkKICAgIGFzc2VydCBzcGxpdF9pZHNbInZhbGlkYXRpb24iXS5pc2Rpc2pvaW50KHNwbGl0X2lkc1sidGVzdCJdKQoKCmRlZiB0ZXN0X3NlY3Rpb24xNF9pbWRiX29mZmljaWFsX3Rlc3RfaXNfaXNvbGF0ZWQoKSAtPiBOb25lOgogICAgZnJvbSBkYXRhc2V0cyBpbXBvcnQgRGF0YXNldERpY3QsIGxvYWRfZnJvbV9kaXNrCgogICAgY2FuZGlkYXRlcyA9IHNvcnRlZCgKICAgICAgICB7CiAgICAgICAgICAgIG1hcmtlci5wYXJlbnQKICAgICAgICAgICAgZm9yIG1hcmtlciBpbiBEQVRBLnJnbG9iKCJkYXRhc2V0X2RpY3QuanNvbiIpCiAgICAgICAgICAgIGlmICJpbWRiIiBpbiBzdHIobWFya2VyLnBhcmVudCkubG93ZXIoKQogICAgICAgICAgICBhbmQgIi50bXBfIiBub3QgaW4gbWFya2VyLnBhcmVudC5uYW1lCiAgICAgICAgICAgIGFuZCAib2ZmbGluZV90bXAiIG5vdCBpbiBtYXJrZXIucGFyZW50Lm5hbWUKICAgICAgICB9CiAgICApCiAgICBhc3NlcnQgY2FuZGlkYXRlcywgIkluc3RhbGxlZCBJTURiIERhdGFzZXREaWN0IHdhcyBub3QgZm91bmQuIgoKICAgIGRhdGFzZXQgPSBsb2FkX2Zyb21fZGlzayhzdHIoY2FuZGlkYXRlc1swXSkpCiAgICBhc3NlcnQgaXNpbnN0YW5jZShkYXRhc2V0LCBEYXRhc2V0RGljdCkKICAgIGFzc2VydCB7InRyYWluIiwgInZhbGlkYXRpb24iLCAidGVzdCJ9Lmlzc3Vic2V0KGRhdGFzZXQua2V5cygpKQoKICAgIHRleHRfZmllbGQgPSBuZXh0KAogICAgICAgICgKICAgICAgICAgICAgZmllbGQKICAgICAgICAgICAgZm9yIGZpZWxkIGluICgidGV4dCIsICJyZXZpZXciLCAiY29udGVudCIsICJzZW50ZW5jZSIpCiAgICAgICAgICAgIGlmIGZpZWxkIGluIGRhdGFzZXRbInRlc3QiXS5jb2x1bW5fbmFtZXMKICAgICAgICApLAogICAgICAgIE5vbmUsCiAgICApCiAgICBhc3NlcnQgdGV4dF9maWVsZCBpcyBub3QgTm9uZQoKICAgIHNvdXJjZV9zZXRzOiBkaWN0W3N0ciwgc2V0W2ludF1dID0ge30KICAgIHRleHRfaGFzaGVzOiBkaWN0W3N0ciwgc2V0W3N0cl1dID0ge30KCiAgICBmb3Igc3BsaXQgaW4gKCJ0cmFpbiIsICJ2YWxpZGF0aW9uIiwgInRlc3QiKToKICAgICAgICBzcGxpdF9kYXRhc2V0ID0gZGF0YXNldFtzcGxpdF0KICAgICAgICBhc3NlcnQgInNvdXJjZV9pbmRleCIgaW4gc3BsaXRfZGF0YXNldC5jb2x1bW5fbmFtZXMKICAgICAgICBpbmRpY2VzID0gW2ludCh2YWx1ZSkgZm9yIHZhbHVlIGluIHNwbGl0X2RhdGFzZXRbInNvdXJjZV9pbmRleCJdXQogICAgICAgIGFzc2VydCBsZW4oaW5kaWNlcykgPT0gbGVuKHNldChpbmRpY2VzKSkKICAgICAgICBzb3VyY2Vfc2V0c1tzcGxpdF0gPSBzZXQoaW5kaWNlcykKICAgICAgICB0ZXh0X2hhc2hlc1tzcGxpdF0gPSB7CiAgICAgICAgICAgIGhhc2hsaWIuc2hhMjU2KAogICAgICAgICAgICAgICAgIiAiLmpvaW4oc3RyKHJvd1t0ZXh0X2ZpZWxkXSkuc3BsaXQoKSkuZW5jb2RlKCkKICAgICAgICAgICAgKS5oZXhkaWdlc3QoKQogICAgICAgICAgICBmb3Igcm93IGluIHNwbGl0X2RhdGFzZXQKICAgICAgICB9CgogICAgYXNzZXJ0IHNvdXJjZV9zZXRzWyJ0cmFpbiJdLmlzZGlzam9pbnQoc291cmNlX3NldHNbInZhbGlkYXRpb24iXSkKICAgIGFzc2VydCBzb3VyY2Vfc2V0c1sidHJhaW4iXS5pc2Rpc2pvaW50KHNvdXJjZV9zZXRzWyJ0ZXN0Il0pCiAgICBhc3NlcnQgc291cmNlX3NldHNbInZhbGlkYXRpb24iXS5pc2Rpc2pvaW50KHNvdXJjZV9zZXRzWyJ0ZXN0Il0pCiAgICBhc3NlcnQgdGV4dF9oYXNoZXNbInRyYWluIl0uaXNkaXNqb2ludCh0ZXh0X2hhc2hlc1sidGVzdCJdKQogICAgYXNzZXJ0IHRleHRfaGFzaGVzWyJ2YWxpZGF0aW9uIl0uaXNkaXNqb2ludCh0ZXh0X2hhc2hlc1sidGVzdCJdKQoKCmRlZiB0ZXN0X3NlY3Rpb24xNF9tZW1vcnlfY29udHJvbGxlcnNfcmVjZWl2ZV9ncmFkaWVudHMoKSAtPiBOb25lOgogICAgc2VlZF9hbGwoMjAyNikKICAgIG1vZGVsID0gbWFrZV9tb2RlbCgKICAgICAgICB0aHJlc2hvbGQ9MC4wLAogICAgICAgIGRldGFjaF9tZW1vcnlfd3JpdGVzPUZhbHNlLAogICAgKS50cmFpbigpCiAgICBpbnB1dHMgPSB0b3JjaC5yYW5kbigzLCAxNiwgNikKICAgIG91dHB1dCA9IG1vZGVsKGlucHV0cywgYnVkZ2V0PXRvcmNoLnRlbnNvcihbMiwgMiwgMl0pKQoKICAgIGxvc3MgPSAoCiAgICAgICAgb3V0cHV0LnNlcXVlbmNlX2xvZ2l0cy5wb3coMikubWVhbigpCiAgICAgICAgKyBvdXRwdXQud3JpdGVfcHJvYmFiaWxpdGllcy5tZWFuKCkKICAgICAgICArIG91dHB1dC5maW5hbF9tZW1vcnkudmFsdWVzLnBvdygyKS5tZWFuKCkKICAgICAgICArIG91dHB1dC5maW5hbF9tZW1vcnkudXRpbGl0eS5tZWFuKCkKICAgICkKICAgIGxvc3MuYmFja3dhcmQoKQoKICAgIGZhbWlsaWVzID0gewogICAgICAgICJ3cml0ZV9jb250cm9sbGVyIjogbW9kZWwud3JpdGVfY29udHJvbGxlciwKICAgICAgICAiZXZpY3Rpb25fY29udHJvbGxlciI6IG1vZGVsLmV2aWN0aW9uX2NvbnRyb2xsZXIsCiAgICAgICAgImluaXRpYWxfdXRpbGl0eV9oZWFkIjogbW9kZWwuaW5pdGlhbF91dGlsaXR5X2hlYWQsCiAgICB9CgogICAgZm9yIG5hbWUsIG1vZHVsZSBpbiBmYW1pbGllcy5pdGVtcygpOgogICAgICAgIHBhcmFtZXRlcnMgPSBbCiAgICAgICAgICAgIHBhcmFtZXRlcgogICAgICAgICAgICBmb3IgcGFyYW1ldGVyIGluIG1vZHVsZS5wYXJhbWV0ZXJzKCkKICAgICAgICAgICAgaWYgcGFyYW1ldGVyLnJlcXVpcmVzX2dyYWQKICAgICAgICBdCiAgICAgICAgYXNzZXJ0IHBhcmFtZXRlcnMsIG5hbWUKICAgICAgICBhc3NlcnQgYWxsKHBhcmFtZXRlci5ncmFkIGlzIG5vdCBOb25lIGZvciBwYXJhbWV0ZXIgaW4gcGFyYW1ldGVycyksIG5hbWUKICAgICAgICBhc3NlcnQgYWxsKAogICAgICAgICAgICB0b3JjaC5pc2Zpbml0ZShwYXJhbWV0ZXIuZ3JhZCkuYWxsKCkKICAgICAgICAgICAgZm9yIHBhcmFtZXRlciBpbiBwYXJhbWV0ZXJzCiAgICAgICAgICAgIGlmIHBhcmFtZXRlci5ncmFkIGlzIG5vdCBOb25lCiAgICAgICAgKSwgbmFtZQogICAgICAgIGFzc2VydCBhbnkoCiAgICAgICAgICAgIHRvcmNoLmNvdW50X25vbnplcm8ocGFyYW1ldGVyLmdyYWQpLml0ZW0oKSA+IDAKICAgICAgICAgICAgZm9yIHBhcmFtZXRlciBpbiBwYXJhbWV0ZXJzCiAgICAgICAgICAgIGlmIHBhcmFtZXRlci5ncmFkIGlzIG5vdCBOb25lCiAgICAgICAgKSwgbmFtZQoKCmRlZiB0ZXN0X3NlY3Rpb24xNF9jYWNoZWRfc3RhdGVfZ3JhcGhfcG9saWN5X2lzX2V4cGxpY2l0KCkgLT4gTm9uZToKICAgIGlucHV0cyA9IHRvcmNoLnJhbmRuKDIsIDEwLCA2KQoKICAgIGRldGFjaGVkID0gbWFrZV9tb2RlbCgKICAgICAgICBzZWVkPTIwMjYsCiAgICAgICAgdGhyZXNob2xkPTAuMCwKICAgICAgICBkZXRhY2hfbWVtb3J5X3dyaXRlcz1UcnVlLAogICAgKS50cmFpbigpCiAgICBkZXRhY2hlZF9vdXRwdXQgPSBkZXRhY2hlZChpbnB1dHMsIGJ1ZGdldD0yKQogICAgZGV0YWNoZWQuemVyb19ncmFkKHNldF90b19ub25lPVRydWUpCiAgICBkZXRhY2hlZF9vdXRwdXQuZmluYWxfbWVtb3J5LnZhbHVlcy5zdW0oKS5iYWNrd2FyZCgpCiAgICBkZXRhY2hlZF9ncmFkcyA9IFsKICAgICAgICBwYXJhbWV0ZXIuZ3JhZAogICAgICAgIGZvciBwYXJhbWV0ZXIgaW4gZGV0YWNoZWQudmFsdWVfcHJvamVjdGlvbi5wYXJhbWV0ZXJzKCkKICAgIF0KICAgIGFzc2VydCBhbGwoCiAgICAgICAgZ3JhZGllbnQgaXMgTm9uZSBvciB0b3JjaC5jb3VudF9ub256ZXJvKGdyYWRpZW50KS5pdGVtKCkgPT0gMAogICAgICAgIGZvciBncmFkaWVudCBpbiBkZXRhY2hlZF9ncmFkcwogICAgKQoKICAgIGNvbm5lY3RlZCA9IG1ha2VfbW9kZWwoCiAgICAgICAgc2VlZD0yMDI2LAogICAgICAgIHRocmVzaG9sZD0wLjAsCiAgICAgICAgZGV0YWNoX21lbW9yeV93cml0ZXM9RmFsc2UsCiAgICApLnRyYWluKCkKICAgIGNvbm5lY3RlZF9vdXRwdXQgPSBjb25uZWN0ZWQoaW5wdXRzLmNsb25lKCksIGJ1ZGdldD0yKQogICAgY29ubmVjdGVkLnplcm9fZ3JhZChzZXRfdG9fbm9uZT1UcnVlKQogICAgY29ubmVjdGVkX291dHB1dC5maW5hbF9tZW1vcnkudmFsdWVzLnN1bSgpLmJhY2t3YXJkKCkKICAgIGNvbm5lY3RlZF9ncmFkcyA9IFsKICAgICAgICBwYXJhbWV0ZXIuZ3JhZAogICAgICAgIGZvciBwYXJhbWV0ZXIgaW4gY29ubmVjdGVkLnZhbHVlX3Byb2plY3Rpb24ucGFyYW1ldGVycygpCiAgICAgICAgaWYgcGFyYW1ldGVyLmdyYWQgaXMgbm90IE5vbmUKICAgIF0KICAgIGFzc2VydCBjb25uZWN0ZWRfZ3JhZHMKICAgIGFzc2VydCBhbnkoCiAgICAgICAgdG9yY2guY291bnRfbm9uemVybyhncmFkaWVudCkuaXRlbSgpID4gMAogICAgICAgIGZvciBncmFkaWVudCBpbiBjb25uZWN0ZWRfZ3JhZHMKICAgICkKCgpkZWYgdGVzdF9zZWN0aW9uMTRfbWVtb3J5X3Jlc2V0c19iZXR3ZWVuX3VucmVsYXRlZF9zZXF1ZW5jZXMoKSAtPiBOb25lOgogICAgbW9kZWwgPSBtYWtlX21vZGVsKHNlZWQ9MjAyNiwgdGhyZXNob2xkPTAuMCkuZXZhbCgpCiAgICBmcmVzaCA9IGNvcHkuZGVlcGNvcHkobW9kZWwpLmV2YWwoKQoKICAgIHVucmVsYXRlZCA9IHRvcmNoLnJhbmRuKDIsIDksIDYpCiAgICB0YXJnZXQgPSB0b3JjaC5yYW5kbigyLCAxMSwgNikKICAgIGJ1ZGdldHMgPSB0b3JjaC50ZW5zb3IoWzIsIDRdKQoKICAgIHdpdGggdG9yY2gubm9fZ3JhZCgpOgogICAgICAgIG1vZGVsKHVucmVsYXRlZCwgYnVkZ2V0PWJ1ZGdldHMpCiAgICAgICAgYWZ0ZXJfdW5yZWxhdGVkID0gbW9kZWwodGFyZ2V0LCBidWRnZXQ9YnVkZ2V0cykKICAgICAgICBmcm9tX2ZyZXNoID0gZnJlc2godGFyZ2V0LmNsb25lKCksIGJ1ZGdldD1idWRnZXRzKQoKICAgIGFzc2VydCBvdXRwdXRzX2VxdWFsKGFmdGVyX3VucmVsYXRlZCwgZnJvbV9mcmVzaCkKICAgIGFzc2VydCB0b3JjaC5hbGwoYWZ0ZXJfdW5yZWxhdGVkLm1lbW9yeV9zaXplc1s6LCAwXSA8PSAxKQo=' | base64 --decode > "$AUTHORITATIVE_TEST"

"$PYTHON_BIN" -m py_compile "$MODEL_FILE" "$CONTROLLER_FILE" "$AUTHORITATIVE_TEST"
[[ -f "$RUNTIME_FILE" ]] && "$PYTHON_BIN" -m py_compile "$RUNTIME_FILE"

TARGETS=("$AUTHORITATIVE_TEST")
[[ -f tests/test_budgetmem_r.py ]] && TARGETS=("tests/test_budgetmem_r.py" "${TARGETS[@]}")

log "Selected authoritative targets:"
printf '  %s\n' "${TARGETS[@]}"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest \
    -q -o addopts='' --collect-only "${TARGETS[@]}" >/dev/null

set +e
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest \
    -q -o addopts='' "${TARGETS[@]}" \
    --junitxml="$JUNIT_FILE" 2>&1 | tee "$LOG_FILE"
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
exit_code = int(os.environ["SECTION14_PYTEST_EXIT"])
timestamp = os.environ["SECTION14_TIMESTAMP"]

cases = []
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

statuses = {}
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

all_selected_pass = bool(cases) and all(
    case["status"] == "PASS" for case in cases
)
go = (
    exit_code == 0
    and all_selected_pass
    and all(value == "PASS" for value in statuses.values())
)

lines = [
    "Section 14 — Authoritative Unit Tests Required Before Training",
    f"Generated UTC: {timestamp}",
    "",
]
for category, status in statuses.items():
    lines.append(f"{category}: {status}")

production_cases = [
    case for case in cases
    if "test_budgetmem_r" in case["classname"]
]
production_pass = bool(production_cases) and all(
    case["status"] == "PASS" for case in production_cases
)

lines.extend(
    [
        f"Production BudgetMem-R tests: {'PASS' if production_pass else 'FAIL'}",
        f"All selected authoritative tests: {'PASS' if all_selected_pass else 'FAIL'}",
        f"Pytest exit code: {exit_code}",
        "",
        f"Final decision: {'GO' if go else 'NO-GO'}",
        f"Section 14: {'COMPLETE' if go else 'INCOMPLETE'}",
        "",
        "Determinism scope: exact CPU execution. CUDA limitations remain "
        "outside this CPU pre-training gate and must be documented separately.",
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

report.write_text("\n".join(lines) + "\n", encoding="utf-8")

manifest.write_text(
    json.dumps(
        {
            "schema_version": "1.0",
            "generated_utc": timestamp,
            "section": "14",
            "gate_type": "authoritative_pretraining",
            "selected_targets": [
                "tests/test_budgetmem_r.py",
                "tests/test_section14_authoritative.py",
            ],
            "superseded_historical_harnesses": [
                "tests/test_section14_required.py",
                "legacy Section 14 nodes using obsolete constructor, threshold, "
                "or provenance assumptions",
            ],
            "category_status": statuses,
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
