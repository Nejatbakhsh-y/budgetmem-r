# BudgetMem-R implementation contract

## Memory representation

Each batch item receives a fixed physical tensor of `max_budget` slots and a
logical per-sample budget `B`. Every slot stores a key, value, predicted utility,
age, retrieval count, and validity flag. The implementation checks
`valid_count <= B` after every write.

## Budget conditioning

The normalized scalar `B / B_max` is encoded by a small neural network and fed
to the write and utility controllers. During training, omitting the `budget`
argument samples from `8, 16, 32, 64, 128` by default.

## Write policy and target-leakage control

The write controller uses the recurrent hidden state, novelty, auxiliary
next-input surprise, auxiliary uncertainty, memory occupancy, time since the
last write, retrieved-memory agreement, and the budget embedding. The model
forward method has no target or label argument. Final task labels enter only
the separate composite-loss function.

Training uses a straight-through stochastic sigmoid gate. Evaluation uses a
hard deterministic threshold.

## Eviction and retrieval

When no free logical slot remains, the model evicts the slot with minimum
predicted future utility. Retrieval applies masked top-k dot-product attention
over valid keys only. Retrieval counts are retained for diagnostics and future
policy comparisons.

## Fusion and objective

The implementation supports concatenation, residual, gated, and attention
fusion. The composite objective includes task, budget, excessive-write,
self-supervised auxiliary, and memory-diversity terms. The hard budget remains
a runtime invariant rather than merely a soft penalty.
