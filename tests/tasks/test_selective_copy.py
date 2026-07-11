"""Unit tests for the selective-copy task generator."""

from __future__ import annotations

import numpy as np

from budgetmem.tasks.selective_copy import generate_selective_copy


def test_same_configuration_is_identical() -> None:
    """The same seed and sequence length must reproduce the sample."""
    first = generate_selective_copy(
        seed=17,
        sequence_length=32,
    )

    second = generate_selective_copy(
        seed=17,
        sequence_length=32,
    )

    np.testing.assert_array_equal(
        first["input"],
        second["input"],
    )

    np.testing.assert_array_equal(
        first["target"],
        second["target"],
    )

    assert (
        first["relevant_positions"]
        == second["relevant_positions"]
    )

    assert (
        first["distractor_positions"]
        == second["distractor_positions"]
    )

    assert first["oracle"] == second["oracle"]


def test_different_seeds_normally_differ() -> None:
    """Different seeds should normally produce different samples."""
    first = generate_selective_copy(
        seed=17,
        sequence_length=32,
    )

    second = generate_selective_copy(
        seed=18,
        sequence_length=32,
    )

    assert not np.array_equal(
        first["input"],
        second["input"],
    )


def test_output_sequence_length_is_correct() -> None:
    """The input must contain the requested number of positions."""
    sequence_length = 41

    sample = generate_selective_copy(
        seed=3,
        sequence_length=sequence_length,
    )

    assert sample["input"].shape[0] == sequence_length


def test_relevant_positions_are_valid_indices() -> None:
    """Every relevant position must be a valid pre-query index."""
    sequence_length = 24

    sample = generate_selective_copy(
        seed=8,
        sequence_length=sequence_length,
    )

    assert sample["relevant_positions"]

    assert all(
        0 <= position < sequence_length - 1
        for position in sample["relevant_positions"]
    )


def test_distractor_positions_are_valid_indices() -> None:
    """Every distractor position must be a valid pre-query index."""
    sequence_length = 24

    sample = generate_selective_copy(
        seed=8,
        sequence_length=sequence_length,
    )

    assert all(
        0 <= position < sequence_length - 1
        for position in sample["distractor_positions"]
    )


def test_relevant_and_distractor_positions_do_not_overlap() -> None:
    """Relevant, distractor, and query positions must be disjoint."""
    sequence_length = 24

    sample = generate_selective_copy(
        seed=8,
        sequence_length=sequence_length,
    )

    relevant = set(sample["relevant_positions"])
    distractors = set(sample["distractor_positions"])
    query_positions = set(
        sample["oracle"]["query_positions"]
    )

    assert relevant.isdisjoint(distractors)
    assert relevant.isdisjoint(query_positions)
    assert distractors.isdisjoint(query_positions)

    # Together, these categories must cover the complete sequence.
    assert (
        relevant | distractors | query_positions
        == set(range(sequence_length))
    )


def test_oracle_required_positions_define_the_target() -> None:
    """Oracle-required positions must contain the target values."""
    sample = generate_selective_copy(
        seed=29,
        sequence_length=36,
    )

    required = sample["oracle"]["required_positions"]

    assert required == sample["relevant_positions"]

    np.testing.assert_array_equal(
        sample["target"],
        sample["input"][required, 0],
    )

    assert (
        sample["oracle"]["target_values"]
        == sample["target"].tolist()
    )


def test_input_and_target_shapes_are_correct() -> None:
    """Input and target arrays must follow the documented shapes."""
    sequence_length = 28

    sample = generate_selective_copy(
        seed=5,
        sequence_length=sequence_length,
    )

    assert sample["input"].shape == (
        sequence_length,
        2,
    )

    assert sample["target"].ndim == 1

    assert sample["target"].shape == (
        len(sample["relevant_positions"]),
    )
