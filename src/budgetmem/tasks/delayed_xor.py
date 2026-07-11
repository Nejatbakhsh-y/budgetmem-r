"""Deterministic delayed-XOR synthetic sequence task.

The task contains two relevant binary operands separated by distractors.
A query token appears at the final sequence position. The target is the XOR
of the two relevant operand bits.

Input representation
--------------------
Each input row has two integer columns:

    input[position] = [token_value, role_marker]

Role markers:

    0 = distractor
    1 = first XOR operand
    2 = second XOR operand
    3 = query

The two operand positions are the relevant positions. All nonoperand,
nonquery positions are distractors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

DISTRACTOR_MARKER = 0
FIRST_OPERAND_MARKER = 1
SECOND_OPERAND_MARKER = 2
QUERY_MARKER = 3

MINIMUM_SEQUENCE_LENGTH = 8


def _validate_inputs(seed: int, sequence_length: int) -> None:
    """Validate delayed-XOR generator arguments."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    if seed < 0:
        raise ValueError("seed must be nonnegative")

    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
        raise TypeError("sequence_length must be an integer")

    if sequence_length < MINIMUM_SEQUENCE_LENGTH:
        raise ValueError(
            "sequence_length must be at least "
            f"{MINIMUM_SEQUENCE_LENGTH} for delayed XOR"
        )


def generate_delayed_xor(
    seed: int,
    sequence_length: int,
) -> dict[str, Any]:
    """Generate one deterministic delayed-XOR task instance.

    Parameters
    ----------
    seed:
        Nonnegative random seed used to initialize NumPy PCG64.
    sequence_length:
        Total number of input positions. The final position is reserved
        for the query.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        - ``input``: integer array with shape ``(sequence_length, 2)``
        - ``target``: one-element integer array containing the XOR result
        - ``relevant_positions``: positions of the two XOR operands
        - ``distractor_positions``: nonoperand and nonquery positions
        - ``oracle``: complete record of the task-generating information
    """

    _validate_inputs(seed=seed, sequence_length=sequence_length)

    rng = np.random.Generator(np.random.PCG64(seed))

    query_position = sequence_length - 1

    # Force the first operand into the first third of the sequence.
    first_operand_pool = np.arange(
        0,
        sequence_length // 3,
        dtype=np.int64,
    )

    # Force the second operand into the latter half, before the query.
    second_operand_pool = np.arange(
        sequence_length // 2,
        query_position,
        dtype=np.int64,
    )

    first_operand_position = int(rng.choice(first_operand_pool))
    second_operand_position = int(rng.choice(second_operand_pool))

    first_operand_bit = int(rng.integers(0, 2))
    second_operand_bit = int(rng.integers(0, 2))
    xor_result = first_operand_bit ^ second_operand_bit

    # Distractor tokens are also binary so that token value alone does not
    # reveal which positions are relevant.
    token_values = rng.integers(
        low=0,
        high=2,
        size=sequence_length,
        dtype=np.int64,
    )

    role_markers = np.full(
        shape=sequence_length,
        fill_value=DISTRACTOR_MARKER,
        dtype=np.int64,
    )

    token_values[first_operand_position] = first_operand_bit
    role_markers[first_operand_position] = FIRST_OPERAND_MARKER

    token_values[second_operand_position] = second_operand_bit
    role_markers[second_operand_position] = SECOND_OPERAND_MARKER

    # The query carries no operand value. Its role marker identifies it.
    token_values[query_position] = 0
    role_markers[query_position] = QUERY_MARKER

    input_sequence: NDArray[np.int64] = np.column_stack(
        (token_values, role_markers)
    ).astype(np.int64, copy=False)

    target: NDArray[np.int64] = np.asarray(
        [xor_result],
        dtype=np.int64,
    )

    relevant_positions = [
        first_operand_position,
        second_operand_position,
    ]

    excluded_positions = {
        first_operand_position,
        second_operand_position,
        query_position,
    }

    distractor_positions = [
        position
        for position in range(sequence_length)
        if position not in excluded_positions
    ]

    oracle = {
        "task": "delayed_xor",
        "seed": seed,
        "sequence_length": sequence_length,
        "required_positions": relevant_positions.copy(),
        "relevant_positions": relevant_positions.copy(),
        "query_positions": [query_position],
        "answer_positions": relevant_positions.copy(),
        "distractor_positions": distractor_positions.copy(),
        "answer_position_space": "input_operands",
        "operand_positions": {
            "first": first_operand_position,
            "second": second_operand_position,
        },
        "operand_bits": {
            "first": first_operand_bit,
            "second": second_operand_bit,
        },
        "operation": "xor",
        "target_values": [xor_result],
    }

    return {
        "input": input_sequence,
        "target": target,
        "relevant_positions": relevant_positions,
        "distractor_positions": distractor_positions,
        "oracle": oracle,
    }
