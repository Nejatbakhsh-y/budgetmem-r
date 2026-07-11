"""Deterministic selective-copy synthetic task generator."""

from __future__ import annotations

from typing import Any

import numpy as np


DISTRACTOR_MARKER = 0
RELEVANT_MARKER = 1
QUERY_MARKER = 2


def generate_selective_copy(
    seed: int,
    sequence_length: int,
) -> dict[str, Any]:
    """Generate one deterministic selective-copy sample.

    The input has shape ``(sequence_length, 2)``.

    Column 0 contains token values.

    Column 1 contains marker values:

    - 0: distractor token
    - 1: relevant token that must be copied
    - 2: query token requesting the copied values

    The final sequence position is always the query position.

    The target is a one-dimensional array containing the relevant
    token values in chronological order.

    Args:
        seed: Integer random seed.
        sequence_length: Total number of sequence positions, including
            the final query position.

    Returns:
        A dictionary containing input, target, relevant positions,
        distractor positions, and oracle metadata.

    Raises:
        TypeError: If seed or sequence_length is not an integer.
        ValueError: If sequence_length is smaller than 6.
    """
    _validate_arguments(
        seed=seed,
        sequence_length=sequence_length,
    )

    # PCG64 is named explicitly so that the random-number algorithm
    # is not dependent on an implicit NumPy default.
    rng = np.random.Generator(np.random.PCG64(seed))

    query_position = sequence_length - 1

    candidate_positions = np.arange(
        query_position,
        dtype=np.int64,
    )

    # Select between one and four relevant values, depending on length.
    num_relevant = max(
        1,
        min(4, query_position // 4),
    )

    relevant_positions_array = np.sort(
        rng.choice(
            candidate_positions,
            size=num_relevant,
            replace=False,
        )
    )

    relevant_positions = relevant_positions_array.tolist()
    relevant_position_set = set(relevant_positions)

    # Values 1 through 9 are ordinary task tokens.
    # Value 0 is reserved for the final query token.
    token_values = rng.integers(
        low=1,
        high=10,
        size=sequence_length,
        dtype=np.int64,
    )

    token_values[query_position] = 0

    markers = np.full(
        sequence_length,
        DISTRACTOR_MARKER,
        dtype=np.int64,
    )

    markers[relevant_positions_array] = RELEVANT_MARKER
    markers[query_position] = QUERY_MARKER

    input_array = np.column_stack(
        (
            token_values,
            markers,
        )
    ).astype(
        np.int64,
        copy=False,
    )

    # The target contains the relevant values in chronological order.
    target = token_values[relevant_positions_array].copy()

    distractor_positions = [
        position
        for position in range(query_position)
        if position not in relevant_position_set
    ]

    oracle = {
        "required_positions": relevant_positions.copy(),
        "query_positions": [query_position],
        "answer_positions": list(range(num_relevant)),
        "distractor_positions": distractor_positions.copy(),
        "answer_position_space": "target",
        "target_values": target.tolist(),
    }

    return {
        "input": input_array,
        "target": target,
        "relevant_positions": relevant_positions,
        "distractor_positions": distractor_positions,
        "oracle": oracle,
    }


def _validate_arguments(
    seed: int,
    sequence_length: int,
) -> None:
    """Validate the public generator arguments."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    if isinstance(sequence_length, bool) or not isinstance(
        sequence_length,
        int,
    ):
        raise TypeError("sequence_length must be an integer")

    if sequence_length < 6:
        raise ValueError(
            "sequence_length must be at least 6"
        )
