"""Tests for the synthetic generator actually used by Section 15."""

from __future__ import annotations

import numpy as np

from scripts.data.generate_synthetic import associative_recall


def test_pilot_associative_oracle_contains_only_queried_values() -> None:
    cfg = {
        "sequence_length": 128,
        "vocabulary_size": 192,
        "number_keys": 12,
        "number_queries": 1,
        "delay_length": 32,
        "distractor_percentage": 65,
        "number_relevant_events": 12,
        "random_seed": 102,
    }

    rng = np.random.default_rng(2026)

    (
        sequence,
        target,
        relevant_positions,
        query_positions,
        metadata,
    ) = associative_recall(
        cfg,
        rng,
    )

    assert len(target) == 1
    assert len(relevant_positions) == 1
    assert len(query_positions) == 1
    assert len(metadata["queried_keys"]) == 1

    relevant_position = relevant_positions[0]

    assert sequence[relevant_position] == target[0]
    assert relevant_position < query_positions[0]
