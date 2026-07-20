"""Verify Section 12 baseline completeness and write auditable evidence."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from budgetmem.baselines.controlled import (
    BASELINE_REGISTRY,
    POLICY_REGISTRY,
    DiagonalSSMBaseline,
    GRUBaseline,
    LSTMBaseline,
    MemoryCachingBaseline,
    OfficialMambaOrSSMBaseline,
    RecurrentMemoryTransformer,
    TransformerBaseline,
    VanillaRNNBaseline,
    parameter_count,
)

REQUIRED_FILES = [
    "src/budgetmem/models/rnn.py",
    "src/budgetmem/models/gru.py",
    "src/budgetmem/models/lstm.py",
    "src/budgetmem/memory/uniform.py",
    "src/budgetmem/memory/most_recent.py",
    "src/budgetmem/memory/fifo.py",
    "src/budgetmem/memory/lru.py",
    "src/budgetmem/memory/random_policy.py",
    "src/budgetmem/memory/reservoir.py",
    "src/budgetmem/memory/novelty.py",
    "src/budgetmem/memory/surprise.py",
    "src/budgetmem/models/attention.py",
    "src/budgetmem/models/state_space.py",
    "src/budgetmem/models/recurrent_memory.py",
    "src/budgetmem/models/memory_caching.py",
    "configs/baselines/section12_baselines.yaml",
]

REQUIRED_POLICIES = {
    "uniform",
    "most_recent",
    "fifo",
    "lru",
    "random",
    "reservoir",
    "novelty",
    "surprise",
}
REQUIRED_BASELINES = {
    "vanilla_rnn",
    "gru",
    "lstm",
    "transformer_full",
    "transformer_sliding",
    "state_space",
    "recurrent_memory_transformer",
    "memory_caching_mean",
    "memory_caching_gated",
    "memory_caching_sparse_selective",
}


def check_forward_and_gradient(model: torch.nn.Module, x: torch.Tensor) -> bool:
    model.zero_grad(set_to_none=True)
    output = model(x)
    if output.shape[:2] != x.shape[:2]:
        return False
    loss = output.square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    return bool(gradients) and all(
        grad is None or torch.isfinite(grad).all().item() for grad in gradients
    ) and any(grad is not None and grad.abs().sum().item() > 0 for grad in gradients)


def main() -> None:
    torch.manual_seed(2026)
    x = torch.randn(2, 16, 8)
    checks: dict[str, bool] = {}

    checks["required_files"] = all(Path(path).is_file() for path in REQUIRED_FILES)
    checks["policy_registry"] = REQUIRED_POLICIES.issubset(POLICY_REGISTRY)
    checks["baseline_registry"] = REQUIRED_BASELINES.issubset(BASELINE_REGISTRY)

    stage1_models = [
        VanillaRNNBaseline(8, 16, 4),
        GRUBaseline(8, 16, 4),
        LSTMBaseline(8, 16, 4),
    ]
    checks["stage_1_simple_recurrent"] = all(
        check_forward_and_gradient(model, x) for model in stage1_models
    )

    states = torch.randn(2, 16, 12)
    policy_pass = True
    policy_indices: dict[str, list[list[int]]] = {}
    for name, factory in POLICY_REGISTRY.items():
        kwargs = {"seed": 2026} if name in {"random", "reservoir"} else {}
        policy = factory(**kwargs)
        indices = policy.select_indices(states, 4)
        policy_pass &= indices.shape == (2, 4)
        policy_pass &= bool(((indices >= 0) & (indices < 16)).all().item())
        policy_pass &= all(len(set(row.tolist())) == 4 for row in indices)
        policy_indices[name] = indices.tolist()
    checks["stage_2_deterministic_caching"] = policy_pass

    attention_models = [
        TransformerBaseline(
            8, 16, 4, num_layers=1, num_heads=4, attention_mode="full", max_length=64
        ),
        TransformerBaseline(
            8,
            16,
            4,
            num_layers=1,
            num_heads=4,
            attention_mode="sliding",
            window_size=4,
            max_length=64,
        ),
    ]
    checks["stage_3_attention"] = all(
        check_forward_and_gradient(model, x) for model in attention_models
    )

    state_space = OfficialMambaOrSSMBaseline(
        8, 12, 4, num_layers=1, state_dim=4, backend="auto"
    )
    checks["stage_4_modern_sequence"] = check_forward_and_gradient(state_space, x)

    rmt = RecurrentMemoryTransformer(
        8,
        16,
        4,
        segment_length=6,
        memory_tokens=2,
        num_layers=1,
        num_heads=4,
    )
    checks["stage_5_recurrent_memory"] = check_forward_and_gradient(rmt, x)

    caching_models = [
        MemoryCachingBaseline(8, 12, 4, budget=4, variant=variant)
        for variant in ("mean", "gated", "sparse_selective")
    ]
    checks["stage_6_memory_caching"] = all(
        check_forward_and_gradient(model, x) for model in caching_models
    )

    calibration_path = Path("reports/tables/section12_parameter_calibration.json")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    max_error = max(
        float(item["relative_error"]) for item in calibration["models"].values()
    )
    checks["parameter_budget_calibration"] = max_error <= 0.20
    checks["training_token_budget_contract"] = (
        "target_training_tokens: 1000000"
        in Path("configs/baselines/section12_baselines.yaml").read_text(encoding="utf-8")
    )

    complete = all(checks.values())
    report = {
        "checks": checks,
        "complete": complete,
        "state_space_backend": state_space.backend,
        "maximum_parameter_relative_error": max_error,
        "policy_indices_smoke_test": policy_indices,
        "parameter_counts_smoke_test": {
            "rnn": parameter_count(stage1_models[0]),
            "gru": parameter_count(stage1_models[1]),
            "lstm": parameter_count(stage1_models[2]),
            "s4d_reference": parameter_count(DiagonalSSMBaseline(8, 12, 4, num_layers=1, state_dim=4)),
        },
    }

    json_path = Path("reports/evidence/section12_baselines_verification.json")
    text_path = Path("reports/evidence/section12_baselines_report.txt")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    status = lambda value: "PASS" if value else "FAIL"
    lines = [
        "Section 12 — Controlled Baseline Verification",
        "=" * 44,
        f"Required implementation files: {status(checks['required_files'])}",
        f"Stage 1 — Simple recurrent baselines: {status(checks['stage_1_simple_recurrent'])}",
        f"Stage 2 — Deterministic caching baselines: {status(checks['stage_2_deterministic_caching'])}",
        f"Stage 3 — Attention baselines: {status(checks['stage_3_attention'])}",
        f"Stage 4 — Modern sequence baseline: {status(checks['stage_4_modern_sequence'])}",
        f"Stage 4 backend used: {state_space.backend}",
        f"Stage 5 — Recurrent-memory baseline: {status(checks['stage_5_recurrent_memory'])}",
        f"Stage 6 — Memory Caching baseline: {status(checks['stage_6_memory_caching'])}",
        f"Baseline registry: {status(checks['baseline_registry'])}",
        f"Policy registry: {status(checks['policy_registry'])}",
        f"Parameter-budget calibration: {status(checks['parameter_budget_calibration'])}",
        f"Maximum parameter relative error: {max_error:.4f}",
        f"Equal training-token contract: {status(checks['training_token_budget_contract'])}",
        f"Section 12: {'COMPLETE' if complete else 'NOT COMPLETE'}",
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
