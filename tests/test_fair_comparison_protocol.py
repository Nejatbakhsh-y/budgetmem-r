from __future__ import annotations

from pathlib import Path

import pytest

from budgetmem.protocols.fair_comparison import (
    expected_batch_size,
    load_protocol,
    parameter_regimes,
    tokens_per_step,
    validate_parameter_count,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "fair_comparison.yaml"


@pytest.fixture(scope="module")
def protocol():
    return load_protocol(PROTOCOL_PATH)


def test_parameter_regimes_are_exactly_small_and_medium(protocol):
    regimes = parameter_regimes(protocol)
    assert set(regimes) == {"small", "medium"}
    assert regimes["small"].target == 1_000_000
    assert regimes["small"].minimum == 950_000
    assert regimes["small"].maximum == 1_050_000
    assert regimes["medium"].target == 5_000_000
    assert regimes["medium"].minimum == 4_750_000
    assert regimes["medium"].maximum == 5_250_000


@pytest.mark.parametrize(
    ("regime", "count"),
    [
        ("small", 950_000),
        ("small", 1_000_000),
        ("small", 1_050_000),
        ("medium", 4_750_000),
        ("medium", 5_000_000),
        ("medium", 5_250_000),
    ],
)
def test_parameter_counts_inside_tolerance_pass(protocol, regime, count):
    assert validate_parameter_count(protocol, regime, count) == []


def test_out_of_tolerance_requires_architecture_exception(protocol):
    errors = validate_parameter_count(protocol, "small", 1_200_000)
    assert errors

    errors = validate_parameter_count(
        protocol,
        "small",
        1_200_000,
        architecture_permits_matching=False,
        exception_reason="Discrete head width prevents a closer configuration.",
    )
    assert errors == []


@pytest.mark.parametrize("sequence_length", [256, 512, 1024])
def test_training_tokens_per_step_are_constant(protocol, sequence_length):
    batch_size = expected_batch_size(protocol, sequence_length)
    assert (
        tokens_per_step(batch_size, sequence_length)
        == protocol["training_budget"]["tokens_per_step"]
    )


def test_latency_batch_sizes_are_fixed(protocol):
    assert protocol["latency_testing"]["fixed_batch_sizes"] == [1, 8]


def test_memory_budgets_and_primary_budgets(protocol):
    assert protocol["memory_budget"]["allowed"] == [8, 16, 32, 64, 128]
    assert protocol["memory_budget"]["primary"] == [32, 64]


def test_retrieval_budgets_and_primary_k(protocol):
    assert protocol["retrieval_budget"]["allowed"] == [1, 4, 8]
    assert protocol["retrieval_budget"]["primary"] == 4


def test_sequence_lengths(protocol):
    assert protocol["sequence_lengths"]["training"] == [256, 512, 1024]
    assert protocol["sequence_lengths"]["testing"] == [
        256,
        512,
        1024,
        2048,
        4096,
        8192,
    ]
    assert protocol["sequence_lengths"][
        "length_8192_requires_reliability_qualification"
    ]


def test_oom_and_timeout_reporting_is_mandatory(protocol):
    reporting = protocol["failure_reporting"]
    assert reporting["oom_must_be_reported"]
    assert reporting["timeout_must_be_reported"]
    assert reporting["failure_reason_required"]
    assert reporting["elapsed_time_required"]
