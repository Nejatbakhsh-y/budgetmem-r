"""Tests for the deterministic multi-key retrieval generator."""

from budgetmem.tasks.multi_key_retrieval import (
    KEY_MARKER,
    QUERY_MARKER,
    VALUE_MARKER,
    generate_multi_key_retrieval,
)


def test_same_configuration_is_identical() -> None:
    first = generate_multi_key_retrieval(
        seed=123,
        sequence_length=32,
    )

    second = generate_multi_key_retrieval(
        seed=123,
        sequence_length=32,
    )

    assert first == second


def test_different_seeds_normally_differ() -> None:
    first = generate_multi_key_retrieval(
        seed=123,
        sequence_length=32,
    )

    second = generate_multi_key_retrieval(
        seed=124,
        sequence_length=32,
    )

    assert first != second


def test_output_sequence_length_is_correct() -> None:
    example = generate_multi_key_retrieval(
        seed=5,
        sequence_length=40,
    )

    assert len(example["input"]) == 40


def test_multiple_queries_and_targets_are_returned() -> None:
    example = generate_multi_key_retrieval(
        seed=17,
        sequence_length=32,
    )

    oracle = example["oracle"]

    assert len(oracle["query_positions"]) == 4
    assert len(oracle["queried_keys"]) == 4
    assert len(example["target"]) == 4


def test_relevant_positions_are_valid_indices() -> None:
    sequence_length = 32

    example = generate_multi_key_retrieval(
        seed=9,
        sequence_length=sequence_length,
    )

    assert example["relevant_positions"]

    assert all(
        0 <= position < sequence_length for position in example["relevant_positions"]
    )


def test_distractor_positions_are_valid_indices() -> None:
    sequence_length = 32

    example = generate_multi_key_retrieval(
        seed=9,
        sequence_length=sequence_length,
    )

    assert example["distractor_positions"]

    assert all(
        0 <= position < sequence_length for position in example["distractor_positions"]
    )


def test_position_categories_are_disjoint_and_complete() -> None:
    sequence_length = 32

    example = generate_multi_key_retrieval(
        seed=31,
        sequence_length=sequence_length,
    )

    relevant = set(example["relevant_positions"])
    distractors = set(example["distractor_positions"])
    queries = set(example["oracle"]["query_positions"])

    assert relevant.isdisjoint(distractors)
    assert relevant.isdisjoint(queries)
    assert distractors.isdisjoint(queries)

    assert relevant | distractors | queries == set(range(sequence_length))


def test_oracle_required_positions_define_the_target() -> None:
    example = generate_multi_key_retrieval(
        seed=81,
        sequence_length=32,
    )

    input_sequence = example["input"]
    target = example["target"]
    oracle = example["oracle"]

    assert oracle["required_positions"] == example["relevant_positions"]

    assert oracle["distractor_positions"] == example["distractor_positions"]

    assert oracle["target_values"] == target

    for output_index, pair_index in enumerate(oracle["queried_pair_indices"]):
        query_position = oracle["query_positions"][output_index]

        key_position, value_position = oracle["pair_positions"][pair_index]

        queried_key = oracle["queried_keys"][output_index]
        target_value = target[output_index]

        assert input_sequence[query_position] == [
            QUERY_MARKER,
            queried_key,
        ]

        assert input_sequence[key_position] == [
            KEY_MARKER,
            queried_key,
        ]

        assert input_sequence[value_position] == [
            VALUE_MARKER,
            target_value,
        ]

        assert key_position in oracle["required_positions"]
        assert value_position in oracle["required_positions"]

        assert value_position == oracle["answer_positions"][output_index]


def test_input_and_target_shapes_are_correct() -> None:
    example = generate_multi_key_retrieval(
        seed=55,
        sequence_length=32,
    )

    assert len(example["input"]) == 32
    assert all(len(row) == 2 for row in example["input"])

    assert len(example["target"]) == 4

    assert all(isinstance(value, int) for value in example["target"])
