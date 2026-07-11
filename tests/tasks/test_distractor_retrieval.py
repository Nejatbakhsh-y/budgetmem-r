"""Tests for the distractor-retrieval synthetic task."""

from __future__ import annotations

import numpy as np
import pytest

from budgetmem.tasks.distractor_retrieval import (
    DISTRACTOR_MARKER,
    MIN_SEQUENCE_LENGTH,
    QUERY_MARKER,
    QUERY_TOKEN,
    TARGET_MARKER,
    generate_distractor_retrieval,
)


def test_output_schema_shapes_and_dtypes() -> None:
    """The generator must return the standard synthetic-task schema."""
    sequence_length = 64

    example = generate_distractor_retrieval(
        seed=17,
        sequence_length=sequence_length,
    )

    assert set(example) == {
        "input",
        "target",
        "relevant_positions",
        "distractor_positions",
        "oracle",
    }

    input_array = example["input"]
    target = example["target"]

    assert isinstance(input_array, np.ndarray)
    assert isinstance(target, np.ndarray)

    assert input_array.shape == (sequence_length, 2)
    assert target.shape == (1,)

    assert input_array.dtype == np.int64
    assert target.dtype == np.int64

    assert isinstance(example["relevant_positions"], list)
    assert isinstance(example["distractor_positions"], list)
    assert isinstance(example["oracle"], dict)


@pytest.mark.parametrize(
    ("seed", "sequence_length"),
    [
        (0, 8),
        (19, 31),
        (2026, 128),
    ],
)
def test_same_configuration_is_deterministic(
    seed: int,
    sequence_length: int,
) -> None:
    """The same seed and sequence length must produce identical output."""
    first = generate_distractor_retrieval(
        seed=seed,
        sequence_length=sequence_length,
    )

    second = generate_distractor_retrieval(
        seed=seed,
        sequence_length=sequence_length,
    )

    np.testing.assert_array_equal(
        first["input"],
        second["input"],
    )

    np.testing.assert_array_equal(
        first["target"],
        second["target"],
    )

    assert first["relevant_positions"] == second["relevant_positions"]
    assert first["distractor_positions"] == second["distractor_positions"]
    assert first["oracle"] == second["oracle"]


def test_different_seeds_change_the_generated_example() -> None:
    """Different seeds should produce different generated sequences."""
    first = generate_distractor_retrieval(
        seed=100,
        sequence_length=64,
    )

    second = generate_distractor_retrieval(
        seed=101,
        sequence_length=64,
    )

    assert not np.array_equal(
        first["input"],
        second["input"],
    )


@pytest.mark.parametrize(
    "sequence_length",
    [
        8,
        37,
        128,
    ],
)
def test_position_sets_are_valid_and_complete(
    sequence_length: int,
) -> None:
    """Relevant and distractor positions must form a complete partition."""
    example = generate_distractor_retrieval(
        seed=7,
        sequence_length=sequence_length,
    )

    relevant_positions = example["relevant_positions"]
    distractor_positions = example["distractor_positions"]

    assert len(relevant_positions) == 2
    assert len(set(relevant_positions)) == 2

    assert set(relevant_positions).isdisjoint(distractor_positions)

    combined_positions = set(relevant_positions) | set(distractor_positions)

    assert combined_positions == set(range(sequence_length))

    assert all(0 <= position < sequence_length for position in relevant_positions)

    assert all(0 <= position < sequence_length for position in distractor_positions)


@pytest.mark.parametrize(
    ("seed", "sequence_length"),
    [
        (3, 8),
        (41, 64),
        (999, 256),
    ],
)
def test_target_query_distractors_and_oracle_are_correct(
    seed: int,
    sequence_length: int,
) -> None:
    """Markers, target value, distractors, delay, and oracle must agree."""
    example = generate_distractor_retrieval(
        seed=seed,
        sequence_length=sequence_length,
    )

    input_array = example["input"]
    target = example["target"]
    relevant_positions = example["relevant_positions"]
    distractor_positions = example["distractor_positions"]
    oracle = example["oracle"]

    target_position = relevant_positions[0]
    query_position = relevant_positions[1]
    target_value = int(target[0])

    assert query_position == sequence_length - 1

    assert input_array[target_position, 0] == target_value
    assert input_array[target_position, 1] == TARGET_MARKER

    assert input_array[query_position, 0] == QUERY_TOKEN
    assert input_array[query_position, 1] == QUERY_MARKER

    assert np.all(input_array[distractor_positions, 1] == DISTRACTOR_MARKER)

    # The target value must not appear at an irrelevant position.
    assert np.all(input_array[distractor_positions, 0] != target_value)

    # The target is intentionally placed in the first quarter.
    assert target_position < max(1, sequence_length // 4)

    # The query must occur after a substantial distractor interval.
    assert query_position > target_position
    assert query_position - target_position > sequence_length // 2

    assert set(oracle) == {
        "task",
        "seed",
        "sequence_length",
        "target_position",
        "query_position",
        "relevant_positions",
        "distractor_positions",
        "target_value",
        "target_values",
        "retrieval_delay",
        "target_window_end",
        "markers",
    }

    assert oracle["task"] == "distractor_retrieval"
    assert oracle["seed"] == seed
    assert oracle["sequence_length"] == sequence_length
    assert oracle["target_position"] == target_position
    assert oracle["query_position"] == query_position
    assert oracle["relevant_positions"] == relevant_positions
    assert oracle["distractor_positions"] == distractor_positions
    assert oracle["target_value"] == target_value
    assert oracle["target_values"] == target.tolist()

    assert oracle["retrieval_delay"] == (query_position - target_position)

    assert oracle["markers"] == {
        "distractor": DISTRACTOR_MARKER,
        "target": TARGET_MARKER,
        "query": QUERY_MARKER,
    }


@pytest.mark.parametrize(
    "invalid_seed",
    [
        True,
        1.5,
        "7",
        None,
    ],
)
def test_invalid_seed_types_raise_type_error(
    invalid_seed: object,
) -> None:
    """Non-integer seeds, including booleans, must be rejected."""
    with pytest.raises(
        TypeError,
        match="seed must be an integer",
    ):
        generate_distractor_retrieval(
            seed=invalid_seed,  # type: ignore[arg-type]
            sequence_length=32,
        )


@pytest.mark.parametrize(
    "invalid_length",
    [
        True,
        8.5,
        "32",
        None,
    ],
)
def test_invalid_sequence_length_types_raise_type_error(
    invalid_length: object,
) -> None:
    """Non-integer sequence lengths must be rejected."""
    with pytest.raises(
        TypeError,
        match="sequence_length must be an integer",
    ):
        generate_distractor_retrieval(
            seed=0,
            sequence_length=invalid_length,  # type: ignore[arg-type]
        )


def test_negative_seed_raises_value_error() -> None:
    """The PCG64 seed must be non-negative."""
    with pytest.raises(
        ValueError,
        match="seed must be non-negative",
    ):
        generate_distractor_retrieval(
            seed=-1,
            sequence_length=32,
        )


@pytest.mark.parametrize(
    "sequence_length",
    [
        -1,
        0,
        MIN_SEQUENCE_LENGTH - 1,
    ],
)
def test_short_sequence_length_raises_value_error(
    sequence_length: int,
) -> None:
    """Sequences shorter than the documented minimum must be rejected."""
    with pytest.raises(
        ValueError,
        match=(f"sequence_length must be at least " f"{MIN_SEQUENCE_LENGTH}"),
    ):
        generate_distractor_retrieval(
            seed=0,
            sequence_length=sequence_length,
        )
