# Section 15 Pilot Experiment Protocol

## Purpose

The pilot is a controlled gate before the complete experiment matrix. It uses selective copy, associative recall, and distractor-heavy retrieval; evaluates sequence lengths 256, 512, and 1,024; uses strict memory budgets 16 and 32; and fixes one random seed.

## Models

- GRU
- GRU with uniform checkpoint cache
- GRU with reservoir cache
- BudgetMem-R from `src/budgetmem/models/budgetmem_r.py`

The heuristic cached GRU implementations are isolated in the pilot module so that their write schedules, retrieval, strict budget enforcement, and policy traces are directly auditable. BudgetMem-R is imported from the Section 13 implementation and adapted only at the token embedding and output-decoding boundaries.

## Training and transfer design

Each model is trained at sequence length 256. The same checkpoint is then evaluated at lengths 256, 512, and 1,024. Cached models and BudgetMem-R alternate budgets 16 and 32 during training. Evaluation uses the same checkpoint under both requested budgets.

This design establishes:

- checkpoint resumption;
- cross-length evaluation;
- runtime budget conditioning;
- strict memory-capacity enforcement;
- controller write and retention behavior;
- relevant-state retention against the random-retention expectation; and
- measured CPU time, wall time, peak RSS, throughput, and parameter count.

## Go/No-Go rule

The decision is `GO` only when every infrastructure and behavioral criterion passes and BudgetMem-R exceeds both uniform-cache and reservoir-cache token accuracy by at least the configured absolute margin at sequence length 1,024 under matched budgets.

A `NO_GO` result is a valid pilot result. It means the full matrix must not begin until the failed criteria are repaired and the pilot is rerun from configuration.

## Artifacts

- `reports/tables/pilot_results.csv`
- `reports/evidence/pilot_summary.json`
- `reports/evidence/pilot_go_no_go.json`
- `reports/pilot_report.md`
- `outputs/pilot/checkpoints/`

Checkpoints are operational artifacts and should not be committed. The configuration, source code, result tables, evidence JSON, and report should be reviewed before committing.
