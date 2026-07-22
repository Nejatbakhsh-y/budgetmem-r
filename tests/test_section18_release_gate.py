"""Focused contracts for the Section 18 BGL and runner release gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from budgetmem.data.bgl import load_bgl_records, parse_bgl_line, prepare_bgl


def test_bgl_parser_preserves_normal_and_anomaly_labels() -> None:
    normal = parse_bgl_line(
        "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 12:02:50 R02-M1-N0-C:J12-U11 RAS KERNEL INFO normal message",
        0,
    )
    anomaly = parse_bgl_line(
        "KERNDTLB 1117838571 2005.06.03 R02-M1-N0-C:J12-U11 12:02:51 R02-M1-N0-C:J12-U11 RAS KERNEL FATAL anomalous message",
        1,
    )
    assert normal is not None and normal.anomaly == 0
    assert anomaly is not None and anomaly.anomaly == 1
    assert "anomalous message" in anomaly.message


def test_bgl_preparation_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    raw = tmp_path / "BGL.log"
    lines = []
    for index in range(48):
        label = "KERNDTLB" if index % 11 == 0 else "-"
        level = "FATAL" if label != "-" else "INFO"
        lines.append(
            f"{label} {1117838570 + index} 2005.06.03 NODE{index % 8:02d} "
            f"12:02:{index % 60:02d} NODE{index % 8:02d} RAS KERNEL {level} event {index}"
        )
    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
    first = tmp_path / "first"
    second = tmp_path / "second"
    summary_a = prepare_bgl(raw, first, sequence_length=4, stride=4, seed=2026)
    summary_b = prepare_bgl(raw, second, sequence_length=4, stride=4, seed=2026)
    assert summary_a.output_sha256 == summary_b.output_sha256
    assert not any(summary_a.split_event_intersections.values())
    assert sum(summary_a.split_counts.values()) == summary_a.sequence_count
    for split in ("train", "validation", "test"):
        assert load_bgl_records(first, split) == load_bgl_records(second, split)


def test_dedicated_runner_exposes_all_section18_aliases() -> None:
    path = Path("scripts/run_section18.py")
    spec = importlib.util.spec_from_file_location("section18_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.validate_model_registry()
    assert result["status"] == "PASS"
    assert not result["missing_aliases"]


def test_smoke_config_is_one_cell() -> None:
    import yaml

    payload = yaml.safe_load(
        Path("configs/experiments/section18/smoke_single_cell.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert payload["experiment"]["run_id"] == "section18_release_smoke"
    assert payload["task"]["name"] == "selective_copy"
    assert payload["model"]["name"] == "budgetmem_r"
    assert payload["memory"]["budget"] == 32
