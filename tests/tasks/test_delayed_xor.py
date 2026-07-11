"""Tests for the deterministic delayed-XOR synthetic task."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from budgetmem.tasks.delayed_xor import (
    DISTRACTOR_MARKER,
    FIRST_OPERAND_MARKER,
    MINIMUM_SEQUENCE_LENGTH,
    QUERY_MARKER,
    SECOND_OPERAND_MARKER,
    generate_delayed_xor,
)


def _assert_nested_equal(left: Any, right: Any) -> None:
    """Recursively compare generator outputs containing NumPy arrays."""

    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
        return

    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()

        for key in left:
            _assert_nested_equal(left[key], right[key])

        return

    if isinstance(left, list):
        assert isinstance(right, list)
        assert len(left) == len(right)

        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)

        return

    assert left == right


def test_same_seed_produces_identical_output() -> None:
    """The same configuration must reproduce the complete task instance."""

    first = generate_delayed_xor(seed=101, sequence_length=32)
    second = generate_delayed_xor(seed=101, sequence_length=32)

    _assert_nested_equal(first, second)


def test_different_seeds_can_produce_different_instances() -> None:
    """Multiple seeds should not all collapse to one generated instance."""

    signatures: set[tuple[object, ...]] = set()

    for seed in range(10):
        sample = generate_delayed_xor(
            seed=seed,
            sequence_length=32,
        )

        signatures.add(
            (
                sample["input"].tobytes(),
                sample["target"].tobytes(),
                tuple(sample["relevant_positions"]),
            )
        )

    assert len(signatures) > 1


def test_return_schema_is_complete() -> None:
    """The generator must return the common synthetic-task fields."""

    sample = generate_delayed_xor(seed=7, sequence_length=24)

    assert set(sample) == {
        "input",
        "target",
        "relevant_positions",
        "distractor_positions",
        "oracle",
    }


def test_input_and_target_shapes_and_dtypes() -> None:
    """Input and target arrays must have stable shapes and integer types."""

    sequence_length = 40
    sample = generate_delayed_xor(
        seed=13,
        sequence_length=sequence_length,
    )

    input_sequence = sample["input"]
    target = sample["target"]

    assert input_sequence.shape == (sequence_length, 2)
    assert target.shape == (1,)

    assert np.issubdtype(input_sequence.dtype, np.integer)
    assert np.issubdtype(target.dtype, np.integer)

    assert int(target[0]) in {0, 1}


def test_operand_positions_are_delayed_and_query_is_last() -> None:
    """Operands must be separated and the query must be at the end."""

    sequence_length = 36
    sample = generate_delayed_xor(
        seed=29,
        sequence_length=sequence_length,
    )

    oracle = sample["oracle"]

    first_position = oracle["operand_positions"]["first"]
    second_position = oracle["operand_positions"]["second"]
    query_position = oracle["query_positions"][0]

    assert 0 <= first_position < sequence_length // 3
    assert sequence_length // 2 <= second_position < sequence_length - 1
    assert first_position < second_position
    assert query_position == sequence_length - 1


def test_relevant_distractor_and_query_positions_are_valid() -> None:
    """Position categories must be valid, disjoint, and exhaustive."""

    sequence_length = 31
    sample = generate_delayed_xor(
        seed=17,
        sequence_length=sequence_length,
    )

    relevant = set(sample["relevant_positions"])
    distractors = set(sample["distractor_positions"])
    queries = set(sample["oracle"]["query_positions"])

    all_positions = relevant | distractors | queries

    assert len(relevant) == 2
    assert len(queries) == 1

    assert relevant.isdisjoint(distractors)
    assert relevant.isdisjoint(queries)
    assert distractors.isdisjoint(queries)

    assert all_positions == set(range(sequence_length))

    assert all(0 <= position < sequence_length for position in all_positions)


def test_role_markers_match_position_categories() -> None:
    """Role markers must identify operands, distractors, and query."""

    sample = generate_delayed_xor(
        seed=41,
        sequence_length=30,
    )

    input_sequence = sample["input"]
    oracle = sample["oracle"]

    first_position = oracle["operand_positions"]["first"]
    second_position = oracle["operand_positions"]["second"]
    query_position = oracle["query_positions"][0]

    assert int(input_sequence[first_position, 1]) == FIRST_OPERAND_MARKER
    assert int(input_sequence[second_position, 1]) == SECOND_OPERAND_MARKER
    assert int(input_sequence[query_position, 1]) == QUERY_MARKER

    for position in sample["distractor_positions"]:
        assert int(input_sequence[position, 1]) == DISTRACTOR_MARKER


def test_target_is_xor_of_relevant_operand_bits() -> None:
    """The target must equal the XOR of the two relevant input bits."""

    sample = generate_delayed_xor(
        seed=53,
        sequence_length=48,
    )

    input_sequence = sample["input"]
    oracle = sample["oracle"]

    first_position = oracle["operand_positions"]["first"]
    second_position = oracle["operand_positions"]["second"]

    first_bit = int(input_sequence[first_position, 0])
    second_bit = int(input_sequence[second_position, 0])

    expected_target = first_bit ^ second_bit

    assert first_bit == oracle["operand_bits"]["first"]
    assert second_bit == oracle["operand_bits"]["second"]
    assert expected_target == int(sample["target"][0])
    assert expected_target == oracle["target_values"][0]


def test_oracle_matches_public_position_fields() -> None:
    """Oracle position records must match the public return fields."""

    sample = generate_delayed_xor(
        seed=61,
        sequence_length=28,
    )

    oracle = sample["oracle"]

    assert oracle["task"] == "delayed_xor"
    assert oracle["seed"] == 61
    assert oracle["sequence_length"] == 28
    assert oracle["operation"] == "xor"
    assert oracle["answer_position_space"] == "input_operands"

    assert oracle["required_positions"] == sample["relevant_positions"]
    assert oracle["relevant_positions"] == sample["relevant_positions"]
    assert oracle["answer_positions"] == sample["relevant_positions"]
    assert oracle["distractor_positions"] == sample["distractor_positions"]


@pytest.mark.parametrize(
    ("seed", "sequence_length", "exception_type"),
    [
        ("7", 16, TypeError),
        (7.5, 16, TypeError),
        (True, 16, TypeError),
        (-1, 16, ValueError),
        (7, "16", TypeError),
        (7, 16.5, TypeError),
        (7, True, TypeError),
        (7, MINIMUM_SEQUENCE_LENGTH - 1, ValueError),
    ],
)
def test_invalid_arguments_raise_clear_errors(
    seed: object,
    sequence_length: object,
    exception_type: type[Exception],
) -> None:
    """Invalid seeds and sequence lengths must be rejected."""

    with pytest.raises(exception_type):
        generate_delayed_xor(
            seed=seed,  # type: ignore[arg-type]
            sequence_length=sequence_length,  # type: ignore[arg-type]
        )


def test_minimum_sequence_length_is_supported() -> None:
    """The documented minimum sequence length must generate successfully."""

    sample = generate_delayed_xor(
        seed=73,
        sequence_length=MINIMUM_SEQUENCE_LENGTH,
    )

    assert sample["input"].shape == (
        MINIMUM_SEQUENCE_LENGTH,
        2,
    )
    assert sample["oracle"]["query_positions"] == [MINIMUM_SEQUENCE_LENGTH - 1]
