"""Deterministic associative-recall synthetic task generator."""

from __future__ import annotations

from typing import Any

import numpy as np

DISTRACTOR_MARKER = 0
KEY_MARKER = 1
VALUE_MARKER = 2
QUERY_MARKER = 3


def generate_associative_recall(
    seed: int,
    sequence_length: int,
) -> dict[str, Any]:
    """Generate one deterministic associative-recall sample.

    The input has shape ``(sequence_length, 2)``.

    Column 0 contains token values.

    Column 1 contains role markers:

    - 0: filler distractor
    - 1: association key
    - 2: association value
    - 3: final query key

    Each key is immediately followed by its associated value. The final
    position repeats one earlier key as a query. The target contains the
    value associated with that key.

    Args:
        seed: Integer random seed.
        sequence_length: Total sequence length, including the final query.

    Returns:
        A dictionary containing input, target, relevant positions,
        distractor positions, and oracle metadata.

    Raises:
        TypeError: If seed or sequence_length is not an integer.
        ValueError: If sequence_length is smaller than 8.
    """
    _validate_arguments(
        seed=seed,
        sequence_length=sequence_length,
    )

    rng = np.random.Generator(np.random.PCG64(seed))

    query_position = sequence_length - 1
    pre_query_length = query_position

    # Create between two and five key-value associations.
    num_pairs = max(
        2,
        min(5, pre_query_length // 3),
    )

    filler_count = pre_query_length - (2 * num_pairs)

    # Randomly distribute filler distractors before, between,
    # and after the key-value pairs.
    gap_sizes = rng.multinomial(
        filler_count,
        np.full(
            num_pairs + 1,
            1.0 / (num_pairs + 1),
        ),
    )

    # Keys and values use separate token ranges.
    keys = rng.choice(
        np.arange(
            1,
            100,
            dtype=np.int64,
        ),
        size=num_pairs,
        replace=False,
    )

    values = rng.choice(
        np.arange(
            100,
            200,
            dtype=np.int64,
        ),
        size=num_pairs,
        replace=False,
    )

    input_array = np.empty(
        (sequence_length, 2),
        dtype=np.int64,
    )

    pair_positions: list[dict[str, int]] = []

    cursor = 0

    for pair_index in range(num_pairs):
        gap_size = int(gap_sizes[pair_index])

        if gap_size:
            input_array[
                cursor : cursor + gap_size,
                0,
            ] = rng.integers(
                low=200,
                high=1000,
                size=gap_size,
                dtype=np.int64,
            )

            input_array[
                cursor : cursor + gap_size,
                1,
            ] = DISTRACTOR_MARKER

            cursor += gap_size

        key_position = cursor
        value_position = cursor + 1

        input_array[key_position] = (
            keys[pair_index],
            KEY_MARKER,
        )

        input_array[value_position] = (
            values[pair_index],
            VALUE_MARKER,
        )

        pair_positions.append(
            {
                "key_position": key_position,
                "value_position": value_position,
            }
        )

        cursor += 2

    final_gap_size = int(gap_sizes[-1])

    if final_gap_size:
        input_array[
            cursor : cursor + final_gap_size,
            0,
        ] = rng.integers(
            low=200,
            high=1000,
            size=final_gap_size,
            dtype=np.int64,
        )

        input_array[
            cursor : cursor + final_gap_size,
            1,
        ] = DISTRACTOR_MARKER

        cursor += final_gap_size

    if cursor != query_position:
        raise RuntimeError("internal sequence construction error")

    # Select one earlier association for the final query.
    queried_pair_index = int(
        rng.integers(
            low=0,
            high=num_pairs,
        )
    )

    queried_key = int(keys[queried_pair_index])
    target_value = int(values[queried_pair_index])

    input_array[query_position] = (
        queried_key,
        QUERY_MARKER,
    )

    selected_pair = pair_positions[queried_pair_index]

    relevant_positions = [
        selected_pair["key_position"],
        selected_pair["value_position"],
    ]

    relevant_position_set = set(relevant_positions)

    distractor_positions = [
        position
        for position in range(query_position)
        if position not in relevant_position_set
    ]

    target = np.asarray(
        [target_value],
        dtype=np.int64,
    )

    oracle = {
        "required_positions": relevant_positions.copy(),
        "query_positions": [query_position],
        "answer_positions": [0],
        "distractor_positions": distractor_positions.copy(),
        "answer_position_space": "target",
        "queried_pair_index": queried_pair_index,
        "queried_key": queried_key,
        "target_values": target.tolist(),
        "pair_positions": pair_positions,
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

    if sequence_length < 8:
        raise ValueError("sequence_length must be at least 8")
