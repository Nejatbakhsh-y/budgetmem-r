# Section 14 — Unit Tests Required Before Training

## Purpose

These tests are a mandatory pretraining gate for BudgetMem-R. They verify the
strict memory-budget invariant, online causality, deterministic CPU execution,
partition isolation, gradient connectivity, and sequence-level memory reset.

## Test inventory

### Budget correctness

For every allowed budget and every sequence step:

- `memory_sizes <= requested budget`
- the number of valid memory-mask entries equals the recorded memory size
- the final external-memory state satisfies its invariant

### Causality

Two sequences share an identical prefix and differ only in future tokens.
All controller decisions, recurrent states, retrieval weights, and memory
states before the perturbation point must be identical.

### Determinism

The same seed and configuration must reproduce:

- a synthetic-task example
- model initialization
- shuffled training order
- CPU evaluation output

The current laptop is CPU-only. CUDA reproducibility must be revalidated on
the later GPU machine and documented with the exact GPU, driver, CUDA, and
PyTorch versions.

### No split leakage

The tests verify:

- distinct synthetic partition seeds
- distinct enabled-task seeds
- no synthetic sample identifier crossing partitions when generated data exist
- no HDFS block identifier crossing partitions when processed data exist
- no official IMDb test review appearing in training or validation when
  processed data exist
- presence of the IMDb test-lock marker

### Gradient flow

The test forces sufficient writes and evictions, backpropagates a composite
loss, and verifies nonzero gradients in both the write controller and utility
controller.

Trainable cached keys, values, and utilities must remain attached to the
autograd graph. Diagnostic metadata such as validity masks, slot indices,
budgets, write steps, and detached retrieval counters must remain detached.

### Memory reset

A forward pass on an unrelated sequence must not affect a later sequence.
The later output and final memory must match a fresh identical model.

## Evidence

The automated gate writes:

`reports/evidence/pretraining_gate_report.json`

Training must not begin unless the report status is `PASS`.

- `PASS`: every model test and every required dataset leakage test passed.
- `PENDING_DATA`: model tests passed, but one or more required processed
  datasets are absent, so training remains blocked.
- `FAIL`: at least one executed gate failed.

When all real and synthetic processed datasets exist, the automation also runs
the complete dataset validator and requires its success.

## IMDb provenance interpretation

The IMDb leakage gate validates dataset-record provenance using:

- the deterministic split-index manifest,
- the official train and test dataset fingerprints,
- exact processed-file hashes,
- exact partition sizes, and
- the locked-test marker.

Identical review text can occur in separate official dataset records. Text
equality alone is not treated as proof that the same dataset example crossed
partitions.
