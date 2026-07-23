# Section 19 Hyperparameter Search Report

**Mode:** FULL

**Selection split:** validation only

**Equal search budget:** 20 trials per architecture family

## Fairness Audit

| Architecture family | Attempted | Complete | Failed | Budget status |
|---|---:|---:|---:|---|
| gru | 20 | 20 | 0 | PASS |
| gru_uniform | 20 | 20 | 0 | PASS |
| gru_reservoir | 20 | 20 | 0 | PASS |
| memory_caching | 20 | 20 | 0 | PASS |
| budgetmem_r | 20 | 20 | 0 | PASS |
| lstm | 20 | 20 | 0 | PASS |
| transformer | 20 | 20 | 0 | PASS |
| mamba | 20 | 20 | 0 | PASS |
| rmt | 20 | 20 | 0 | PASS |

## Validation-Selected Hyperparameters

| Architecture family | Trial | Validation metric | Value |
|---|---:|---|---:|
| gru | 2 | token_accuracy | 0.02083333 |
| gru_uniform | 6 | token_accuracy | 0.04166667 |
| gru_reservoir | 2 | token_accuracy | 0.02083333 |
| memory_caching | 0 | primary_metric_value | 0.00000000 |
| budgetmem_r | 6 | token_accuracy | 0.02083333 |
| lstm | 0 | primary_metric_value | 0.00000000 |
| transformer | 0 | primary_metric_value | 0.00000000 |
| mamba | 15 | primary_metric_value | 0.02083333 |
| rmt | 13 | primary_metric_value | 0.02083333 |

Controller-specific parameters were searched only for BudgetMem-R because they are not defined for non-controller baselines. This does not change the trial budget: every architecture family receives exactly the same number of validation trials.

No test-set metric was used to rank or select any trial.
