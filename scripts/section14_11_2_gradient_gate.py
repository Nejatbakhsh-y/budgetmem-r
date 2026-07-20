from __future__ import annotations

import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports" / "evidence"
RESULT_JSON = EVIDENCE / "section14_11_2_gradient_stability.json"
RESULT_TXT = EVIDENCE / "section14_11_2_gradient_stability.txt"
TEST_STATUS = EVIDENCE / "section14_11_2_test_status.json"
RAW_LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
CLIP_NORM = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
TOL = 1e-6


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def collect_from_official_csv() -> tuple[list[float], str | None]:
    preferred = ROOT / "reports" / "tables" / "pilot_results.csv"
    candidates = [preferred] if preferred.exists() else []
    candidates.extend(
        sorted(
            (ROOT / "reports").rglob("*pilot*result*.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = {str(name).strip().lower(): name for name in reader.fieldnames or []}
            grad_key = next(
                (
                    fields[name]
                    for name in (
                        "maximum_gradient_norm",
                        "max_gradient_norm",
                        "gradient_norm_max",
                        "raw_gradient_norm",
                        "max_grad_norm",
                    )
                    if name in fields
                ),
                None,
            )
            if grad_key is None:
                continue
            model_key = next(
                (fields[name] for name in ("model", "model_name", "policy") if name in fields),
                None,
            )
            all_values: list[float] = []
            budgetmem_values: list[float] = []
            for row in reader:
                raw = row.get(grad_key)
                if raw in (None, ""):
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                all_values.append(value)
                model = str(row.get(model_key, "")).lower() if model_key else ""
                if "budgetmem" in model:
                    budgetmem_values.append(value)
            values = budgetmem_values or all_values
            if values:
                return values, str(path.relative_to(ROOT))
    return [], None


def recursively_find_gradient_values(obj: Any) -> list[float]:
    keys = {
        "raw_gradient_norm",
        "maximum_gradient_norm",
        "max_gradient_norm",
        "max_raw_gradient_norm",
    }
    values: list[float] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in keys and isinstance(value, (int, float)):
                values.append(float(value))
            values.extend(recursively_find_gradient_values(value))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(recursively_find_gradient_values(value))
    return values


def collect_from_latest_profile_json() -> tuple[list[float], str | None]:
    candidates = []
    for path in EVIDENCE.rglob("*.json"):
        if path in {RESULT_JSON, TEST_STATUS}:
            continue
        lower = path.name.lower()
        if "gradient" in lower or "profile" in lower or "diagnostic" in lower:
            candidates.append(path)
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        values = recursively_find_gradient_values(load_json(path))
        if values:
            return values, str(path.relative_to(ROOT))
    return [], None


def clipping_sources() -> list[str]:
    matches: list[str] = []
    for base in (ROOT / "src", ROOT / "scripts"):
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if path.name == Path(__file__).name:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if "clip_grad_norm_" in text or "clip_grad_value_" in text:
                matches.append(str(path.relative_to(ROOT)))
    return sorted(set(matches))


def threshold_evidence() -> tuple[bool, bool, list[str]]:
    raw_ok = False
    clip_ok = False
    sources: list[str] = []
    pattern = re.compile(
        r"(?m)^\s*(gradient_clip_norm|maximum_acceptable_gradient_norm)\s*:\s*([-+0-9.eE]+)\s*$"
    )
    for base in (ROOT / "configs", ROOT / "scripts"):
        if not base.exists():
            continue
        for path in list(base.rglob("*.yaml")) + list(base.rglob("*.yml")):
            text = path.read_text(encoding="utf-8", errors="replace")
            matched = False
            for key, raw in pattern.findall(text):
                matched = True
                value = float(raw)
                if key == "maximum_acceptable_gradient_norm":
                    raw_ok = raw_ok or math.isclose(value, RAW_LIMIT, abs_tol=TOL)
                if key == "gradient_clip_norm":
                    clip_ok = clip_ok or math.isclose(value, CLIP_NORM, abs_tol=TOL)
            if matched:
                sources.append(str(path.relative_to(ROOT)))
    return raw_ok, clip_ok, sorted(set(sources))


def newest_performance_gate() -> tuple[bool | None, str | None]:
    candidates = []
    for pattern in ("*go*no*go*.json", "*final*decision*.json", "pilot_go_no_go.json"):
        candidates.extend(EVIDENCE.rglob(pattern))
    for path in sorted(set(candidates), key=lambda item: item.stat().st_mtime, reverse=True):
        data = load_json(path)
        if data is None:
            continue

        def visit(obj: Any, prefix: str = "") -> tuple[bool | None, str | None]:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full = f"{prefix}.{key}" if prefix else str(key)
                    lower = full.lower()
                    if "outperform" in lower and ("two" in lower or "2" in lower):
                        if isinstance(value, bool):
                            return value, f"{path.relative_to(ROOT)}:{full}"
                        if isinstance(value, str):
                            upper = value.upper()
                            if upper in {"PASS", "GO"}:
                                return True, f"{path.relative_to(ROOT)}:{full}"
                            if upper in {"FAIL", "NO_GO", "NO-GO"}:
                                return False, f"{path.relative_to(ROOT)}:{full}"
                    result = visit(value, full)
                    if result[0] is not None:
                        return result
            elif isinstance(obj, list):
                for index, value in enumerate(obj):
                    result = visit(value, f"{prefix}[{index}]")
                    if result[0] is not None:
                        return result
            return None, None

        result = visit(data)
        if result[0] is not None:
            return result
    return None, None


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    tests = load_json(TEST_STATUS) or {}
    csv_values, csv_source = collect_from_official_csv()
    profile_values, profile_source = collect_from_latest_profile_json()
    observed = csv_values + profile_values
    finite = [value for value in observed if math.isfinite(value)]
    nonfinite_count = len(observed) - len(finite)
    maximum = max(finite) if finite else None

    clip_sources = clipping_sources()
    raw_threshold_ok, clip_threshold_ok, threshold_sources = threshold_evidence()

    checks = {
        "gradient_flow_test": tests.get("gradient_flow_test_rc") == 0,
        "controller_calibration_test": tests.get("controller_calibration_test_rc", 0) == 0,
        "gradient_measurements_present": bool(observed),
        "all_gradient_measurements_finite": bool(observed) and nonfinite_count == 0,
        "maximum_raw_gradient_within_100": maximum is not None and maximum <= RAW_LIMIT + TOL,
        "gradient_clipping_implemented": bool(clip_sources),
        "raw_gradient_limit_configured": raw_threshold_ok,
        "clip_norm_configured": clip_threshold_ok,
    }
    gradient_pass = all(checks.values())
    performance_pass, performance_source = newest_performance_gate()
    section_complete = gradient_pass and performance_pass is True

    payload = {
        "section": "14.11.2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gradient_stability": "PASS" if gradient_pass else "FAIL",
        "section_14_11": "COMPLETE" if section_complete else "NOT COMPLETE",
        "raw_gradient_limit": RAW_LIMIT,
        "gradient_clip_norm": CLIP_NORM,
        "maximum_observed_gradient_norm": maximum,
        "observed_gradient_count": len(observed),
        "nonfinite_gradient_count": nonfinite_count,
        "checks": checks,
        "pilot_metric_source": csv_source,
        "profile_metric_source": profile_source,
        "clipping_sources": clip_sources,
        "threshold_sources": threshold_sources,
        "same_budget_outperforms_two_deterministic_policies": performance_pass,
        "performance_gate_source": performance_source,
        "note": (
            "Gradient stability is evaluated against the documented raw threshold of 100.0 "
            "and clipping threshold of 1.0. The final GO decision is not forced."
        ),
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "Section 14.11.2 Gradient Stability",
        "==================================",
        f"Generated UTC:                           {payload['generated_at_utc']}",
        f"Raw-gradient limit:                      {RAW_LIMIT:.6f}",
        f"Gradient clipping norm:                  {CLIP_NORM:.6f}",
        f"Maximum observed gradient norm:          {maximum if maximum is not None else 'NOT FOUND'}",
        f"Observed gradient measurements:          {len(observed)}",
        f"Non-finite measurements:                 {nonfinite_count}",
        "",
    ]
    for name, passed in checks.items():
        lines.append(f"{name:43s} {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            f"Gradient stability:                      {'PASS' if gradient_pass else 'FAIL'}",
            "Same-budget two-policy outperformance:  "
            + (
                "PASS"
                if performance_pass is True
                else "FAIL"
                if performance_pass is False
                else "NOT VERIFIED"
            ),
            f"Section 14.11:                            {'COMPLETE' if section_complete else 'NOT COMPLETE'}",
            "",
            f"Evidence JSON: {RESULT_JSON.relative_to(ROOT)}",
        ]
    )
    RESULT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if gradient_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
