from __future__ import annotations

from typing import Any

import numpy as np

DISTRACTOR_MARKER = 0
RARE_EVENT_MARKER = 1
QUERY_MARKER = 2

MINIMUM_SEQUENCE_LENGTH = 12
_NUMBER_OF_RARE_EVENTS = 4
_NUMBER_OF_QUERIES = 2
_TOKEN_LOW = 0
_TOKEN_HIGH = 10


def _validate_inputs(seed: int, sequence_length: int) -> None:
    """Validate public generator arguments."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be a non-negative integer")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")

    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
        raise TypeError("sequence_length must be an integer")
    if sequence_length < MINIMUM_SEQUENCE_LENGTH:
        raise ValueError(
            f"sequence_length must be at least {MINIMUM_SEQUENCE_LENGTH}"
        )


def _choose_rare_event_positions(
    rng: np.random.Generator,
    source_length: int,
) -> list[int]:
    """Choose ordered rare-event positions spread across the source region."""
    candidates = np.arange(source_length, dtype=np.int64)
    segments = np.array_split(candidates, _NUMBER_OF_RARE_EVENTS)

    return [int(rng.choice(segment)) for segment in segments]


def generate_rare_event_recall(
    seed: int,
    sequence_length: int,
) -> dict[str, Any]:
    """Generate a deterministic rare-event recall example.

    Four rare events are embedded among ordinary distractors. The final two
    positions query two of those events by chronological event index. The
    target contains the requested event values in query order.
    """
    _validate_inputs(seed, sequence_length)

    rng = np.random.Generator(np.random.PCG64(seed))

    source_length = sequence_length - _NUMBER_OF_QUERIES

    rare_event_positions = _choose_rare_event_positions(
        rng,
        source_length,
    )

    queried_event_indices = rng.choice(
        _NUMBER_OF_RARE_EVENTS,
        size=_NUMBER_OF_QUERIES,
        replace=False,
    ).astype(np.int64)

    token_values = rng.integers(
        _TOKEN_LOW,
        _TOKEN_HIGH,
        size=sequence_length,
        dtype=np.int64,
    )

    role_markers = np.full(
        sequence_length,
        DISTRACTOR_MARKER,
        dtype=np.int64,
    )

    role_markers[rare_event_positions] = RARE_EVENT_MARKER

    query_positions = list(range(source_length, sequence_length))
    token_values[query_positions] = queried_event_indices
    role_markers[query_positions] = QUERY_MARKER

    relevant_positions = [
        rare_event_positions[int(event_index)]
        for event_index in queried_event_indices
    ]

    target = token_values[relevant_positions].copy()

    input_array = np.column_stack(
        (token_values, role_markers)
    ).astype(
        np.int64,
        copy=False,
    )

    excluded_positions = set(relevant_positions) | set(query_positions)

    distractor_positions = [
        position
        for position in range(sequence_length)
        if position not in excluded_positions
    ]

    queried_event_index_set = set(queried_event_indices.tolist())

    unqueried_event_positions = [
        position
        for event_index, position in enumerate(rare_event_positions)
        if event_index not in queried_event_index_set
    ]

    oracle = {
        "task": "rare_event_recall",
        "seed": seed,
        "sequence_length": sequence_length,
        "required_positions": list(relevant_positions),
        "relevant_positions": list(relevant_positions),
        "query_positions": list(query_positions),
        "answer_positions": list(relevant_positions),
        "distractor_positions": list(distractor_positions),
        "answer_position_space": "input",
        "rare_event_positions": list(rare_event_positions),
        "rare_event_values": token_values[rare_event_positions].tolist(),
        "queried_event_indices": queried_event_indices.tolist(),
        "queried_event_positions": list(relevant_positions),
        "unqueried_event_positions": unqueried_event_positions,
        "number_of_rare_events": _NUMBER_OF_RARE_EVENTS,
        "number_of_queries": _NUMBER_OF_QUERIES,
        "retrieval_order": "query_order",
        "target_values": target.tolist(),
    }

    return {
        "input": input_array,
        "target": target,
        "relevant_positions": relevant_positions,
        "distractor_positions": distractor_positions,
        "oracle": oracle,
    }
