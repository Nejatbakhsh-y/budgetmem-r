"""CPU-only smoke validation for the Section 13 BudgetMem-R implementation."""

from __future__ import annotations

import torch

from budgetmem.models.budgetmem_r import BudgetMemR
from budgetmem.training.budgetmem_loss import compute_budgetmem_r_loss


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    model = BudgetMemR(
        input_dim=8,
        hidden_dim=32,
        output_dim=4,
        max_budget=128,
        allowed_budgets=(8, 16, 32, 64, 128),
        key_dim=24,
        value_dim=32,
        retrieval_k=4,
        backbone="gru",
        fusion="gated",
    )
    model.train()
    inputs = torch.randn(3, 20, 8)
    targets = torch.tensor([0, 1, 3])
    budgets = torch.tensor([8, 16, 32])
    output = model(inputs, budget=budgets)
    losses = compute_budgetmem_r_loss(output, targets)
    losses.total.backward()

    assert torch.all(output.memory_sizes <= budgets.unsqueeze(1))
    print("BUDGETMEM-R VALIDATION PASSED")
    print("device=cpu")
    print(f"final_memory_sizes={output.final_memory.sizes().tolist()}")
    print(f"total_loss={losses.total.item():.6f}")
    print(f"budget_penalty={losses.budget.item():.6f}")


if __name__ == "__main__":
    main()
