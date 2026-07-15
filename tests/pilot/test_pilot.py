"""Focused tests for the Section 15 pilot infrastructure."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from budgetmem.experiments.pilot import (
    CachedGRUPilotModel,
    summarize_go_no_go,
    validate_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "configs" / "experiments" / "pilot.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_pilot_matrix_is_exact() -> None:
    cfg = _config()
    validate_config(cfg)
    assert cfg["matrix"]["evaluation_sequence_lengths"] == [256, 512, 1024]
    assert cfg["matrix"]["memory_budgets"] == [16, 32]
    assert cfg["device"] == "cpu"


def test_uniform_and_reservoir_never_exceed_budget() -> None:
    cfg = _config()
    inputs = torch.randint(0, 64, (3, 40), generator=torch.Generator().manual_seed(4))
    for policy in ("uniform", "reservoir"):
        model = CachedGRUPilotModel(cfg, policy=policy, seed=2026).eval()
        with torch.no_grad():
            output = model(inputs, budget=16)
        assert output.memory_sizes is not None
        assert int(output.memory_sizes.max()) <= 16
        assert all(len(row) <= 16 for row in output.retained_positions)


def test_go_no_go_requires_both_memory_policy_wins() -> None:
    cfg = _config()
    common = {
        "sequence_length": 1024,
        "stability_pass": True,
        "budget_pass": True,
        "resource_measurement_pass": True,
        "checkpoint_resume_pass": True,
        "config_sha256": "abc",
        "write_frequency": 0.15,
        "recent_state_overlap": 0.20,
        "relevant_state_retention_rate": 0.20,
        "expected_random_retention": 0.03125,
    }
    rows = [
        {**common, "model": "budgetmem_r", "token_accuracy": 0.70},
        {**common, "model": "gru_uniform_cache", "token_accuracy": 0.60},
        {**common, "model": "gru_reservoir_cache", "token_accuracy": 0.61},
        {**common, "model": "gru", "token_accuracy": 0.55, "budget_pass": True},
    ]
    gate = summarize_go_no_go(rows, cfg, smoke=False)
    assert gate["criteria"]["outperforms_two_memory_policies"] is True
    assert gate["status"] == "GO"


def test_go_no_go_fails_when_one_policy_is_not_clearly_beaten() -> None:
    cfg = _config()
    common = {
        "sequence_length": 1024,
        "stability_pass": True,
        "budget_pass": True,
        "resource_measurement_pass": True,
        "checkpoint_resume_pass": True,
        "config_sha256": "abc",
        "write_frequency": 0.15,
        "recent_state_overlap": 0.20,
        "relevant_state_retention_rate": 0.20,
        "expected_random_retention": 0.03125,
    }
    rows = [
        {**common, "model": "budgetmem_r", "token_accuracy": 0.70},
        {**common, "model": "gru_uniform_cache", "token_accuracy": 0.60},
        {**common, "model": "gru_reservoir_cache", "token_accuracy": 0.69},
        {**common, "model": "gru", "token_accuracy": 0.55, "budget_pass": True},
    ]
    gate = summarize_go_no_go(rows, cfg, smoke=False)
    assert gate["criteria"]["outperforms_two_memory_policies"] is False
    assert gate["status"] == "NO_GO"
