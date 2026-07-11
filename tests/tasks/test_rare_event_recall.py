from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from budgetmem.tasks.rare_event_recall import (
    DISTRACTOR_MARKER,
    MINIMUM_SEQUENCE_LENGTH,
    QUERY_MARKER,
    RARE_EVENT_MARKER,
    generate_rare_event_recall,
)


def _assert_samples_equal(
    first: dict[str, Any],
    second: dict[str, Any],
) -> None:
    np.testing.assert_array_equal(first["input"], second["input"])
    np.testing.assert_array_equal(first["target"], second["target"])

    assert first["relevant_positions"] == second["relevant_positions"]
    assert first["distractor_positions"] == second["distractor_positions"]
    assert first["oracle"] == second["oracle"]


def test_same_configuration_is_identical() -> None:
    first = generate_rare_event_recall(
        seed=105,
        sequence_length=64,
    )

    second = generate_rare_event_recall(
        seed=105,
        sequence_length=64,
    )

    _assert_samples_equal(first, second)


def test_different_seeds_normally_differ() -> None:
    first = generate_rare_event_recall(
        seed=105,
        sequence_length=64,
    )

    second = generate_rare_event_recall(
        seed=106,
        sequence_length=64,
    )

    assert not (
        np.array_equal(first["input"], second["input"])
        and np.array_equal(first["target"], second["target"])
    )


def test_output_shapes_and_dtypes_are_correct() -> None:
    sample = generate_rare_event_recall(
        seed=105,
        sequence_length=80,
    )

    assert sample["input"].shape == (80, 2)
    assert sample["input"].dtype == np.int64

    assert sample["target"].shape == (2,)
    assert sample["target"].dtype == np.int64


def test_relevant_positions_are_valid_unique_indices() -> None:
    sample = generate_rare_event_recall(
        seed=105,
        sequence_length=64,
    )

    relevant_positions = sample["relevant_positions"]

    assert len(relevant_positions) == 2
    assert len(set(relevant_positions)) == len(relevant_positions)
    assert all(0 <= position < 62 for position in relevant_positions)


def test_distractor_positions_are_valid_indices() -> None:
    sample = generate_rare_event_recall(
        seed=105,
        sequence_length=64,
    )

    distractor_positions = sample["distractor_positions"]

    assert len(distractor_positions) == 60
    assert len(set(distractor_positions)) == len(distractor_positions)
    assert all(0 <= position < 64 for position in distractor_positions)


def test_position_categories_are_disjoint_and_complete() -> None:
    sample = generate_rare_event_recall(
        seed=105,
        sequence_length=64,
    )

    relevant = set(sample["relevant_positions"])
    distractors = set(sample["distractor_positions"])
    queries = set(sample["oracle"]["query_positions"])

    assert relevant.isdisjoint(distractors)
    assert relevant.isdisjoint(queries)
    assert distractors.isdisjoint(queries)

    assert relevant | distractors | queries == set(range(64))


def test_role_markers_identify_events_queries_and_plain_distractors() -> None:
    sample = generate_rare_event_recall(
        seed=105,
        sequence_length=64,
    )

    markers = sample["input"][:, 1]

    rare_events = sample["oracle"]["rare_event_positions"]
    query_positions = sample["oracle"]["query_positions"]

    plain_distractors = (
        set(range(64))
        - set(rare_events)
        - set(query_positions)
    )

    assert np.all(
        markers[rare_events] == RARE_EVENT_MARKER
    )

    assert np.all(
        markers[query_positions] == QUERY_MARKER
    )

    assert np.all(
        markers[list(plain_distractors)] == DISTRACTOR_MARKER
    )


def test_query_tokens_select_the_target_events() -> None:
    sample = generate_rare_event_recall(
        seed=105,
        sequence_length=64,
    )

    oracle = sample["oracle"]

    query_tokens = sample["input"][
        oracle["query_positions"],
        0,
    ]

    np.testing.assert_array_equal(
        query_tokens,
        oracle["queried_event_indices"],
    )

    expected_positions = [
        oracle["rare_event_positions"][event_index]
        for event_index in oracle["queried_event_indices"]
    ]

    assert expected_positions == sample["relevant_positions"]


def test_oracle_required_positions_define_target() -> None:
    sample = generate_rare_event_recall(
        seed=105,
        sequence_length=64,
    )

    required_positions = sample["oracle"]["required_positions"]

    required_values = sample["input"][
        required_positions,
        0,
    ]

    np.testing.assert_array_equal(
        required_values,
        sample["target"],
    )

    assert (
        sample["oracle"]["target_values"]
        == sample["target"].tolist()
    )

    assert sample["oracle"]["retrieval_order"] == "query_order"


def test_unqueried_rare_events_are_task_distractors() -> None:
    sample = generate_rare_event_recall(
        seed=105,
        sequence_length=64,
    )

    unqueried_events = set(
        sample["oracle"]["unqueried_event_positions"]
    )

    assert len(unqueried_events) == 2

    assert unqueried_events <= set(
        sample["distractor_positions"]
    )


@pytest.mark.parametrize(
    "seed",
    [-1, 1.5, "105", True],
)
def test_invalid_seed_is_rejected(seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        generate_rare_event_recall(  # type: ignore[arg-type]
            seed=seed,
            sequence_length=64,
        )


@pytest.mark.parametrize(
    "sequence_length",
    [
        MINIMUM_SEQUENCE_LENGTH - 1,
        64.0,
        "64",
        True,
    ],
)
def test_invalid_sequence_length_is_rejected(
    sequence_length: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        generate_rare_event_recall(  # type: ignore[arg-type]
            seed=105,
            sequence_length=sequence_length,
        )
