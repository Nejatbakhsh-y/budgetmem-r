"""Regression test for detached memory writes in the pilot adapter."""

from pathlib import Path

import yaml

from budgetmem.experiments.pilot import BudgetMemRAdapter


def test_pilot_adapter_passes_detach_memory_writes() -> None:
    path = Path(
        "configs/experiments/"
        "pilot_assoc_detached_memory.yaml"
    )

    cfg = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    adapter = BudgetMemRAdapter(cfg)

    assert adapter.core.detach_memory_writes is True
