"""Deterministic multi-key retrieval synthetic task generator."""

from __future__ import annotations

from typing import Any

import numpy as np

DISTRACTOR_MARKER = 0
KEY_MARKER = 1
VALUE_MARKER = 2
QUERY_MARKER = 3

_MINIMUM_SEQUENCE_LENGTH = 18
_NUMBER_OF_QUERIES = 4
_NUMBER_OF_PAIRS = 6


def _choose_pair_starts(
    rng: np.random.Generator,
    source_length: int,
    number_of_pairs: int,
) -> list[int]:
    """Choose non-overlapping starts for adjacent key-value rows."""
    available_starts = np.arange(0, source_length - 1, 2, dtype=np.int64)
    selected = rng.choice(
        available_starts,
        size=number_of_pairs,
        replace=False,
    )
    return sorted(selected.astype(int).tolist())


def generate_multi_key_retrieval(
    seed: int,
    sequence_length: int,
) -> dict[str, Any]:
    """Generate one deterministic multi-key retrieval example.

    The source region contains six key-value records. Four distinct keys are
    queried at the end of the sequence, and the target contains the associated
    values in query order.

    Args:
        seed: Integer seed for NumPy's PCG64 random-number generator.
        sequence_length: Number of rows in the generated input sequence.

    Returns:
        A dictionary containing input, target, relevant_positions,
        distractor_positions, and oracle.

    Raises:
        TypeError: If either argument is not an integer.
        ValueError: If the requested sequence is too short.
    """
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    if not isinstance(sequence_length, int):
        raise TypeError("sequence_length must be an integer")

    if sequence_length < _MINIMUM_SEQUENCE_LENGTH:
        raise ValueError(
            "sequence_length must be at least "
            f"{_MINIMUM_SEQUENCE_LENGTH} for multi-key retrieval"
        )

    rng = np.random.Generator(np.random.PCG64(seed))

    source_length = sequence_length - _NUMBER_OF_QUERIES
    input_array = np.zeros((sequence_length, 2), dtype=np.int64)

    keys = rng.choice(
        np.arange(100, 10_000),
        size=_NUMBER_OF_PAIRS,
        replace=False,
    )

    values = rng.choice(
        np.arange(20_000, 40_000),
        size=_NUMBER_OF_PAIRS,
        replace=False,
    )

    pair_starts = _choose_pair_starts(
        rng=rng,
        source_length=source_length,
        number_of_pairs=_NUMBER_OF_PAIRS,
    )

    record_order = rng.permutation(_NUMBER_OF_PAIRS).astype(int).tolist()

    pair_positions: list[list[int]] = [[-1, -1] for _ in range(_NUMBER_OF_PAIRS)]

    for start, pair_index in zip(
        pair_starts,
        record_order,
        strict=True,
    ):
        key_position = start
        value_position = start + 1

        input_array[key_position] = [
            KEY_MARKER,
            int(keys[pair_index]),
        ]

        input_array[value_position] = [
            VALUE_MARKER,
            int(values[pair_index]),
        ]

        pair_positions[pair_index] = [
            key_position,
            value_position,
        ]

    queried_pair_indices = (
        rng.choice(
            _NUMBER_OF_PAIRS,
            size=_NUMBER_OF_QUERIES,
            replace=False,
        )
        .astype(int)
        .tolist()
    )

    query_positions = list(range(source_length, sequence_length))

    queried_keys: list[int] = []
    target_values: list[int] = []
    answer_positions: list[int] = []
    required_positions: list[int] = []

    for query_position, pair_index in zip(
        query_positions,
        queried_pair_indices,
        strict=True,
    ):
        key = int(keys[pair_index])
        value = int(values[pair_index])

        key_position, value_position = pair_positions[pair_index]

        input_array[query_position] = [
            QUERY_MARKER,
            key,
        ]

        queried_keys.append(key)
        target_values.append(value)
        answer_positions.append(value_position)

        required_positions.extend(
            [
                key_position,
                value_position,
            ]
        )

    required_positions = sorted(set(required_positions))

    reserved_positions = set(required_positions) | set(query_positions)

    distractor_positions = [
        position
        for position in range(sequence_length)
        if position not in reserved_positions
    ]

    oracle: dict[str, Any] = {
        "required_positions": required_positions,
        "query_positions": query_positions,
        "answer_positions": answer_positions,
        "distractor_positions": distractor_positions,
        "answer_position_space": "input",
        "queried_pair_indices": queried_pair_indices,
        "queried_keys": queried_keys,
        "target_values": target_values,
        "pair_positions": pair_positions,
    }

    return {
        "input": input_array.tolist(),
        "target": target_values,
        "relevant_positions": required_positions,
        "distractor_positions": distractor_positions,
        "oracle": oracle,
    }
