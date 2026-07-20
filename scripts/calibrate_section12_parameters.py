"""Calibrate hidden dimensions against the GRU parameter budget."""

from __future__ import annotations

import json
from pathlib import Path

from budgetmem.baselines.controlled import (
    DiagonalSSMBaseline,
    GRUBaseline,
    LSTMBaseline,
    MemoryCachingBaseline,
    RecurrentMemoryTransformer,
    TransformerBaseline,
    VanillaRNNBaseline,
    parameter_count,
)

INPUT_DIM = 32
OUTPUT_DIM = 16
REFERENCE_HIDDEN = 64
DEFAULT_CANDIDATES = list(range(8, 137, 4))
STATE_SPACE_CANDIDATES = list(range(8, 513, 4))


def closest(factory, target: int, candidates: list[int] | None = None) -> dict[str, int | float]:
    best: tuple[float, int, int] | None = None
    for hidden in (candidates or DEFAULT_CANDIDATES):
        try:
            model = factory(hidden)
        except (ValueError, RuntimeError):
            continue
        count = parameter_count(model)
        error = abs(count - target) / target
        if best is None or error < best[0]:
            best = (error, hidden, count)
    if best is None:
        raise RuntimeError("no valid hidden dimension found")
    return {
        "hidden_dim": best[1],
        "parameters": best[2],
        "relative_error": round(best[0], 6),
    }


def main() -> None:
    target = parameter_count(
        GRUBaseline(INPUT_DIM, REFERENCE_HIDDEN, OUTPUT_DIM)
    )
    factories = {
        "vanilla_rnn": lambda h: VanillaRNNBaseline(INPUT_DIM, h, OUTPUT_DIM),
        "gru": lambda h: GRUBaseline(INPUT_DIM, h, OUTPUT_DIM),
        "lstm": lambda h: LSTMBaseline(INPUT_DIM, h, OUTPUT_DIM),
        "transformer_full": lambda h: TransformerBaseline(
            INPUT_DIM,
            h,
            OUTPUT_DIM,
            num_layers=1,
            num_heads=4,
            feedforward_dim=2 * h,
            attention_mode="full",
            max_length=256,
        ),
        "transformer_sliding": lambda h: TransformerBaseline(
            INPUT_DIM,
            h,
            OUTPUT_DIM,
            num_layers=1,
            num_heads=4,
            feedforward_dim=2 * h,
            attention_mode="sliding",
            window_size=32,
            max_length=256,
        ),
        "state_space": lambda h: DiagonalSSMBaseline(
            INPUT_DIM,
            h,
            OUTPUT_DIM,
            num_layers=1,
            state_dim=4,
        ),
        "recurrent_memory_transformer": lambda h: RecurrentMemoryTransformer(
            INPUT_DIM,
            h,
            OUTPUT_DIM,
            segment_length=32,
            memory_tokens=4,
            num_layers=1,
            num_heads=4,
            feedforward_dim=2 * h,
        ),
        "memory_caching_mean": lambda h: MemoryCachingBaseline(
            INPUT_DIM, h, OUTPUT_DIM, budget=8, variant="mean"
        ),
        "memory_caching_gated": lambda h: MemoryCachingBaseline(
            INPUT_DIM, h, OUTPUT_DIM, budget=8, variant="gated"
        ),
        "memory_caching_sparse_selective": lambda h: MemoryCachingBaseline(
            INPUT_DIM, h, OUTPUT_DIM, budget=8, variant="sparse_selective"
        ),
    }
    results = {
        "reference_model": "gru",
        "reference_hidden_dim": REFERENCE_HIDDEN,
        "target_parameters": target,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
                "models": {
            name: closest(
                factory,
                target,
                STATE_SPACE_CANDIDATES if name == "state_space" else DEFAULT_CANDIDATES,
            )
            for name, factory in factories.items()
        },
    }
    output = Path("reports/tables/section12_parameter_calibration.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
