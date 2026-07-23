from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_section19_hyperparameter_search.py"
SPEC = importlib.util.spec_from_file_location("section19_search", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def search_space() -> dict:
    config = yaml.safe_load(
        (ROOT / "configs/hyperparameter_search/section19_search.yaml").read_text(
            encoding="utf-8"
        )
    )
    return config["search_space"]


def test_sampling_is_deterministic() -> None:
    first = MODULE.sample_hyperparameters(
        "budgetmem_r", 3, search_space(), 2026, 32
    )
    second = MODULE.sample_hyperparameters(
        "budgetmem_r", 3, search_space(), 2026, 32
    )
    assert first == second


def test_non_controller_families_do_not_fake_controller_parameters() -> None:
    values = MODULE.sample_hyperparameters("gru", 0, search_space(), 2026, 32)
    assert values["memory_controller_temperature"] is None
    assert values["auxiliary_loss_coefficient"] is None
    assert values["budget_penalty"] is None
    assert values["retrieval_top_k"] is None
    assert values["write_threshold"] is None


def test_budgetmem_retrieval_top_k_never_exceeds_budget() -> None:
    for trial in range(50):
        values = MODULE.sample_hyperparameters(
            "budgetmem_r", trial, search_space(), 2026, 4
        )
        assert 1 <= values["retrieval_top_k"] <= 4


def test_validation_metric_extraction(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps({"metrics": {"primary_metric_value": 0.75}}),
        encoding="utf-8",
    )
    name, value, direction = MODULE.extract_validation_objective(path)
    assert name == "primary_metric_value"
    assert value == 0.75
    assert direction == "maximize"


def test_plan_has_equal_trials_for_all_families() -> None:
    families = list(MODULE.DEFAULT_FAMILIES)
    trials = 20
    counts = {family: trials for family in families}
    assert len(set(counts.values())) == 1
    assert sum(counts.values()) == len(families) * trials
