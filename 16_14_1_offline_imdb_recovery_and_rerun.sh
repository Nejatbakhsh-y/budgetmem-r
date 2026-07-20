#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Offline IMDb recovery and Section 14 rerun.
# No Hugging Face network request is made.

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    :
else
    REPO_ROOT="$(pwd)"
fi
cd "$REPO_ROOT"

[[ -f pyproject.toml && -d src/budgetmem ]] || die "Run this from the budgetmem-r repository root."

for candidate in .venv/bin/python venv/bin/python python3 python; do
    if [[ "$candidate" == */* && -x "$candidate" ]]; then PYTHON_BIN="$candidate"; break; fi
    if [[ "$candidate" != */* ]] && command -v "$candidate" >/dev/null 2>&1; then PYTHON_BIN="$(command -v "$candidate")"; break; fi
done
[[ -n "${PYTHON_BIN:-}" ]] || die "Python was not found."

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SECTION14_REPO_ROOT="$REPO_ROOT"
export SECTION14_TIMESTAMP="$TIMESTAMP"

BACKUP_DIR="$REPO_ROOT/reports/evidence/backups/section14_offline_imdb/$TIMESTAMP"
LOG_FILE="$REPO_ROOT/reports/evidence/logs/section14_offline_imdb_${TIMESTAMP}.log"
JUNIT_FILE="$REPO_ROOT/reports/evidence/junit/section14_offline_imdb_${TIMESTAMP}.xml"
REPORT_FILE="$REPO_ROOT/reports/evidence/section14_unit_tests_report.txt"
RESULTS_FILE="$REPO_ROOT/reports/tables/section14_unit_test_results.csv"
IMDB_REPORT="$REPO_ROOT/reports/evidence/section14_imdb_offline_recovery_${TIMESTAMP}.json"
mkdir -p "$BACKUP_DIR" reports/evidence/logs reports/evidence/junit reports/tables
export SECTION14_BACKUP_DIR="$BACKUP_DIR"
export SECTION14_IMDB_REPORT="$IMDB_REPORT"

log "Confirming the restored BudgetMem-R contract."
"$PYTHON_BIN" - <<'PY'
import torch
from budgetmem.models.budgetmem_r import BudgetMemR
m = BudgetMemR(input_dim=6, hidden_dim=12, output_dim=3, max_budget=16,
               training_budgets=(4, 8, 16), top_k=3).eval()
x = torch.randn(2, 12, 6)
o = m(x, budget=torch.tensor([4, 8]))
for name in ("hidden_states", "write_slots", "eviction_flags", "memory_masks",
             "memory_sizes", "budgets", "final_memory"):
    assert hasattr(o, name), name
assert torch.all(o.memory_sizes <= o.budgets.unsqueeze(1))
print("BudgetMem-R restored-contract smoke test: PASS")
PY

log "Recovering IMDb from local DatasetDicts and Arrow files only."
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations
import hashlib, json, os, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

root = Path(os.environ["SECTION14_REPO_ROOT"])
backup_dir = Path(os.environ["SECTION14_BACKUP_DIR"]) / "imdb"
report_path = Path(os.environ["SECTION14_IMDB_REPORT"])
timestamp = os.environ["SECTION14_TIMESTAMP"]

@dataclass
class Candidate:
    source: str
    splits: dict[str, Dataset]
    total: int
    score: tuple[int, int, int, str]

def norm_split(name: str) -> str | None:
    n = name.lower()
    if n in {"train", "training"}: return "train"
    if n in {"validation", "valid", "val", "dev"}: return "validation"
    if n in {"test", "testing"}: return "test"
    return None

def infer_split(path: Path) -> str | None:
    s = "/" + "/".join(p.lower() for p in path.parts) + "/"
    name = path.name.lower()
    if "/validation/" in s or "-validation." in name or "_validation." in name: return "validation"
    if "/test/" in s or "-test." in name or "_test." in name or "test-" in name: return "test"
    if "/train/" in s or "-train." in name or "_train." in name or "train-" in name: return "train"
    if "/val/" in s or "-val." in name or "_val." in name: return "validation"
    return None

def clean(ds: Dataset) -> Dataset:
    bad = [c for c in ds.column_names if c.startswith("__") or c in {
        "source_index", "_fragment_index", "_batch_index", "_last_in_fragment", "_filename"}]
    return ds.remove_columns(bad) if bad else ds

def read_arrows(paths: list[Path]) -> Dataset:
    parts = []
    for p in sorted(paths):
        try: d = clean(Dataset.from_file(str(p)))
        except Exception: continue
        if len(d): parts.append(d)
    if not parts: raise RuntimeError("No readable Arrow fragments")
    common = set(parts[0].column_names)
    for d in parts[1:]: common &= set(d.column_names)
    cols = [c for c in parts[0].column_names if c in common]
    if not cols: raise RuntimeError("Arrow fragments have no common columns")
    parts = [d.select_columns(cols) for d in parts]
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)

def score(source: str, splits: dict[str, Dataset]) -> Candidate | None:
    if "train" not in splits or "test" not in splits: return None
    total = sum(len(v) for v in splits.values())
    return Candidate(source, splits, total,
                     (int(total == 50000), int(len(splits["test"]) >= 25000), -abs(total-50000), source))

def from_dict(root_path: Path) -> Candidate | None:
    try: obj = load_from_disk(str(root_path))
    except Exception: return None
    if not isinstance(obj, DatasetDict): return None
    splits = {n: clean(d) for k, d in obj.items() if (n := norm_split(k))}
    return score(f"DatasetDict:{root_path}", splits)

def from_arrow_root(root_path: Path) -> Candidate | None:
    groups = {"train": [], "validation": [], "test": []}
    for p in root_path.rglob("*.arrow"):
        n = infer_split(p)
        if n: groups[n].append(p)
    if not groups["train"] or not groups["test"]: return None
    try:
        splits = {n: read_arrows(ps) for n, ps in groups.items() if ps}
    except Exception:
        return None
    return score(f"ArrowRoot:{root_path}", splits)

dict_roots: set[Path] = set()
for base in (root/"data", root/"reports"/"evidence"/"backups"):
    if base.exists():
        for marker in base.rglob("dataset_dict.json"):
            if "imdb" in str(marker).lower(): dict_roots.add(marker.parent)

arrow_roots: set[Path] = set(dict_roots)
cache = Path.home()/".cache"/"huggingface"/"datasets"
if cache.exists():
    for p in cache.rglob("*.arrow"):
        if "imdb" not in str(p).lower(): continue
        cur = p.parent
        for _ in range(5):
            if "imdb" in str(cur).lower(): arrow_roots.add(cur)
            if cur.parent == cur: break
            cur = cur.parent

candidates: list[Candidate] = []
for p in sorted(dict_roots):
    c = from_dict(p)
    if c: candidates.append(c)
for p in sorted(arrow_roots):
    c = from_arrow_root(p)
    if c: candidates.append(c)

if not candidates:
    raise RuntimeError("No local IMDb DatasetDict or train/test Arrow source was found; no network request was attempted.")

candidates.sort(key=lambda c: c.score, reverse=True)
selected = candidates[0]
if selected.total < 49000:
    report_path.write_text(json.dumps({
        "status": "FAIL", "reason": "No complete local source", "candidates": [
            {"source": c.source, "total": c.total, "splits": {k: len(v) for k,v in c.splits.items()}}
            for c in candidates[:20]]}, indent=2)+"\n")
    raise RuntimeError(f"Best local IMDb source has only {selected.total} examples; see {report_path}")

splits = dict(selected.splits)
if "validation" not in splits:
    s = splits["train"].train_test_split(test_size=0.10, seed=2026, shuffle=True)
    splits["train"], splits["validation"] = s["train"], s["test"]

field = next((f for f in ("text","review","content","sentence") if f in splits["test"].column_names), None)
if field is None: raise RuntimeError("IMDb text field not found")
def h(row: dict[str, Any]) -> str:
    return hashlib.sha256(str(row[field]).strip().encode()).hexdigest()

test_hashes = {h(dict(row)) for row in splits["test"]}
rebuilt: dict[str, Dataset] = {}
removed = 0
for name in ("train","validation","test"):
    ds = clean(splits[name])
    keep = []
    for i, row in enumerate(ds):
        if name != "test" and h(dict(row)) in test_hashes:
            removed += 1
            continue
        keep.append(i)
    rebuilt[name] = ds.select(keep)

mins = {"train":22000, "validation":2400, "test":24900}
for name, minimum in mins.items():
    if len(rebuilt[name]) < minimum:
        raise RuntimeError(f"Recovered {name} split incomplete: {len(rebuilt[name])} < {minimum}")

clean_total = (
    len(rebuilt["train"])
    + len(rebuilt["validation"])
    + len(rebuilt["test"])
)
if clean_total < 49800:
    raise RuntimeError(
        "Recovered IMDb corpus is materially incomplete after leakage removal: "
        f"{clean_total} < 49800."
    )

next_index = 0
for name in ("train","validation","test"):
    ds = rebuilt[name]
    if "source_index" in ds.column_names: ds = ds.remove_columns(["source_index"])
    idx = list(range(next_index, next_index+len(ds)))
    next_index += len(ds)
    rebuilt[name] = ds.add_column("source_index", idx)

out = DatasetDict({n: rebuilt[n] for n in ("train","validation","test")})
current_roots = sorted({m.parent for m in (root/"data").rglob("dataset_dict.json") if "imdb" in str(m).lower()}) if (root/"data").exists() else []
destination = current_roots[0] if current_roots else root/"data"/"processed"/"imdb"
if destination.exists():
    b = backup_dir/destination.relative_to(root); b.parent.mkdir(parents=True, exist_ok=True)
    if b.exists(): shutil.rmtree(b)
    shutil.copytree(destination, b)

tmp = destination.with_name(destination.name+f".offline_tmp_{timestamp}")
if tmp.exists(): shutil.rmtree(tmp)
tmp.parent.mkdir(parents=True, exist_ok=True)
out.save_to_disk(str(tmp))
valid = load_from_disk(str(tmp)); assert isinstance(valid, DatasetDict)
sets = {n:set(valid[n]["source_index"]) for n in valid}
for i, a in enumerate(("train","validation","test")):
    assert len(sets[a]) == len(valid[a])
    for b in ("train","validation","test")[i+1:]: assert not (sets[a] & sets[b])
test2 = {h(dict(row)) for row in valid["test"]}
for n in ("train","validation"):
    assert not ({h(dict(row)) for row in valid[n]} & test2)
if destination.exists(): shutil.rmtree(destination)
tmp.rename(destination)
for extra in current_roots[1:]:
    if extra.exists(): shutil.rmtree(extra)
    shutil.copytree(destination, extra)

report = {
    "generated_utc": timestamp, "status":"PASS", "network_used":False,
    "selected_source":selected.source, "selected_source_rows":selected.total,
    "destination":str(destination.relative_to(root)),
    "splits":{n:len(out[n]) for n in out},
    "official_test_leakage_removed":removed,
    "source_index_unique_and_disjoint":True,
    "top_candidates":[{"source":c.source,"total":c.total,"splits":{k:len(v) for k,v in c.splits.items()}} for c in candidates[:10]]}
report_path.write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
PY

log "Checking syntax and collecting Section 14 tests."
"$PYTHON_BIN" -m py_compile src/budgetmem/models/budgetmem_r.py
[[ -f tests/section14_runtime.py ]] && "$PYTHON_BIN" -m py_compile tests/section14_runtime.py
[[ -f tests/test_section14_required.py ]] && "$PYTHON_BIN" -m py_compile tests/test_section14_required.py

TARGETS=(tests/test_budgetmem_r.py)
[[ -f tests/test_section14_required.py ]] && TARGETS+=(tests/test_section14_required.py)

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest -q -o addopts='' --collect-only "${TARGETS[@]}" >/dev/null

log "Running the Section 14 gate."
set +e
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest -q -o addopts='' "${TARGETS[@]}" --junitxml="$JUNIT_FILE" 2>&1 | tee "$LOG_FILE"
PYTEST_EXIT="${PIPESTATUS[0]}"
set -e

export SECTION14_JUNIT_FILE="$JUNIT_FILE"
export SECTION14_LOG_FILE="$LOG_FILE"
export SECTION14_REPORT_FILE="$REPORT_FILE"
export SECTION14_RESULTS_FILE="$RESULTS_FILE"
export SECTION14_IMDB_REPORT="$IMDB_REPORT"
export SECTION14_PYTEST_EXIT="$PYTEST_EXIT"

"$PYTHON_BIN" - <<'PY'
import csv, os, xml.etree.ElementTree as ET
from pathlib import Path
junit=Path(os.environ["SECTION14_JUNIT_FILE"]); report=Path(os.environ["SECTION14_REPORT_FILE"])
results=Path(os.environ["SECTION14_RESULTS_FILE"]); log=Path(os.environ["SECTION14_LOG_FILE"])
imdb=Path(os.environ["SECTION14_IMDB_REPORT"]); code=int(os.environ["SECTION14_PYTEST_EXIT"])
cases=[]
if junit.exists():
    for case in ET.parse(junit).getroot().iter("testcase"):
        status="PASS"; detail=""
        for tag in ("failure","error","skipped"):
            child=case.find(tag)
            if child is not None:
                status=tag.upper(); detail=(child.attrib.get("message") or child.text or "").strip(); break
        cases.append({"classname":case.attrib.get("classname",""),"test_name":case.attrib.get("name",""),"status":status,"seconds":case.attrib.get("time","0"),"detail":detail.replace("\n"," ")[:5000]})
with results.open("w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=("classname","test_name","status","seconds","detail")); w.writeheader(); w.writerows(cases)
all_pass=bool(cases) and all(c["status"]=="PASS" for c in cases)
go=code==0 and all_pass
lines=["Section 14 — Unit Tests Required Before Training",f"Generated UTC: {os.environ['SECTION14_TIMESTAMP']}","",f"All selected Section 14 tests: {'PASS' if all_pass else 'FAIL'}",f"Pytest exit code: {code}","",f"Final decision: {'GO' if go else 'NO-GO'}",f"Section 14: {'COMPLETE' if go else 'INCOMPLETE'}","",f"JUnit evidence: {junit}",f"Detailed log: {log}",f"Result table: {results}",f"IMDb offline-recovery evidence: {imdb}"]
failed=[c for c in cases if c["status"]!="PASS"]
if failed:
    lines += ["","Failed or unresolved checks:"] + [f"- {c['test_name']}: {c['status']} — {c['detail'] or 'No detail recorded.'}" for c in failed]
report.write_text("\n".join(lines)+"\n",encoding="utf-8")
print("\n"+report.read_text(encoding="utf-8"))
PY

if [[ "$PYTEST_EXIT" -eq 0 ]]; then
    printf '\nSECTION 14 RESULT: GO\nSection 14 is complete. Training may begin.\n'
else
    printf '\nSECTION 14 RESULT: NO-GO\nReview reports/evidence/section14_unit_tests_report.txt.\n'
    exit "$PYTEST_EXIT"
fi
