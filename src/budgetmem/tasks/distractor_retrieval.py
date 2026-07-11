"""Synthetic distractor-retrieval task.

The sequence contains:

1. One relevant target token placed early in the sequence.
2. Many irrelevant distractor tokens.
3. One retrieval query at the final sequence position.

The model must retain the relevant target token while ignoring the
intervening distractors.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DISTRACTOR_MARKER = 0
TARGET_MARKER = 1
QUERY_MARKER = 2

QUERY_TOKEN = 0
TOKEN_MIN = 1
TOKEN_MAX = 256

MIN_SEQUENCE_LENGTH = 8

TaskExample = dict[str, Any]


def _validate_integer(name: str, value: object) -> int:
    """Validate and normalize an integer argument."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")

    return int(value)


def generate_distractor_retrieval(
    seed: int,
    sequence_length: int,
) -> TaskExample:
    """Generate one deterministic distractor-retrieval example.

    Parameters
    ----------
    seed:
        Non-negative random seed used by NumPy PCG64.
    sequence_length:
        Total number of sequence positions. It must be at least 8.

    Returns
    -------
    dict
        Dictionary containing:

        - ``input``: integer array with shape ``(sequence_length, 2)``.
          Column 0 contains token values and column 1 contains role markers.
        - ``target``: one-element integer array containing the token that
          must be retrieved.
        - ``relevant_positions``: target and query positions.
        - ``distractor_positions``: all irrelevant sequence positions.
        - ``oracle``: complete record of the task construction.

    Notes
    -----
    Role markers are:

    - 0: distractor
    - 1: relevant target token
    - 2: retrieval query
    """
    seed = _validate_integer("seed", seed)
    sequence_length = _validate_integer(
        "sequence_length",
        sequence_length,
    )

    if seed < 0:
        raise ValueError("seed must be non-negative.")

    if sequence_length < MIN_SEQUENCE_LENGTH:
        raise ValueError(f"sequence_length must be at least {MIN_SEQUENCE_LENGTH}.")

    rng = np.random.Generator(np.random.PCG64(seed))

    query_position = sequence_length - 1

    # Place the target in the first quarter of the sequence so that the
    # query requires long-range retention through many distractors.
    target_window_end = max(1, sequence_length // 4)
    target_position = int(rng.integers(0, target_window_end))

    target_value = int(
        rng.integers(
            TOKEN_MIN,
            TOKEN_MAX,
        )
    )

    # Generate distractors while excluding the target value. This prevents
    # accidental duplicate occurrences from making the oracle ambiguous.
    token_values = rng.integers(
        TOKEN_MIN,
        TOKEN_MAX,
        size=sequence_length,
        dtype=np.int64,
    )

    collision_mask = token_values == target_value

    while np.any(collision_mask):
        token_values[collision_mask] = rng.integers(
            TOKEN_MIN,
            TOKEN_MAX,
            size=int(np.sum(collision_mask)),
            dtype=np.int64,
        )
        collision_mask = token_values == target_value

    input_array = np.empty(
        (sequence_length, 2),
        dtype=np.int64,
    )

    input_array[:, 0] = token_values
    input_array[:, 1] = DISTRACTOR_MARKER

    input_array[target_position, 0] = target_value
    input_array[target_position, 1] = TARGET_MARKER

    input_array[query_position, 0] = QUERY_TOKEN
    input_array[query_position, 1] = QUERY_MARKER

    target = np.asarray(
        [target_value],
        dtype=np.int64,
    )

    relevant_positions = [
        target_position,
        query_position,
    ]

    relevant_position_set = set(relevant_positions)

    distractor_positions = [
        position
        for position in range(sequence_length)
        if position not in relevant_position_set
    ]

    oracle = {
        "task": "distractor_retrieval",
        "seed": seed,
        "sequence_length": sequence_length,
        "target_position": target_position,
        "query_position": query_position,
        "relevant_positions": list(relevant_positions),
        "distractor_positions": list(distractor_positions),
        "target_value": target_value,
        "target_values": target.tolist(),
        "retrieval_delay": query_position - target_position,
        "target_window_end": target_window_end,
        "markers": {
            "distractor": DISTRACTOR_MARKER,
            "target": TARGET_MARKER,
            "query": QUERY_MARKER,
        },
    }

    return {
        "input": input_array,
        "target": target,
        "relevant_positions": relevant_positions,
        "distractor_positions": distractor_positions,
        "oracle": oracle,
    }
