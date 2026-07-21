# BudgetMem-R Main Experimental Protocol

**Protocol version:** protocol-v1.0  
**Status:** Frozen before final test evaluation  
**Freeze date (UTC):** 2026-07-21  
**Repository branch:** `main`

## 1. Protocol-lock rule

This document freezes the confirmatory experimental design before final test results are examined.

The primary research questions, hypotheses, datasets, splits, models, parameter budgets, sequence lengths, memory budgets, seeds, metrics, statistical tests, hyperparameter-search budget, exclusion criteria, OOM policy, early-stopping rule, and primary comparisons must not be changed after inspecting final test results.

A necessary pre-test amendment requires a new protocol document, commit, and version tag. Any analysis added after final test inspection must be labeled exploratory and must not replace the primary analysis.

## 2. Research questions

1. Does BudgetMem-R improve long-range recall over deterministic memory policies under the same task, sequence length, random seed, recurrent backbone, and memory-slot budget?
2. Does BudgetMem-R preserve useful historical states more effectively than uniform checkpointing, reservoir sampling, recency-based caching, and other non-learned policies?
3. Are any gains larger under longer sequences or tighter memory budgets?
4. Can BudgetMem-R enforce a strict memory budget while maintaining stable optimization, causal memory decisions, valid gradient flow, deterministic evaluation, and correct memory resets?
5. Do the results generalize from controlled synthetic tasks to HDFS and IMDb sequence tasks?

## 3. Primary hypotheses

### H1: Same-budget long-range recall

At sequence length 1,024 and memory budgets 16 and 32, BudgetMem-R will outperform at least two deterministic memory policies on the primary long-range recall metric under matched task and seed conditions.

### H2: Practical improvement

For each deterministic policy counted toward H1, BudgetMem-R must achieve a mean paired improvement of at least 2.0 percentage points.

### H3: Statistical evidence

For each deterministic policy counted toward H1, the paired 95% confidence interval for the BudgetMem-R improvement must exclude zero and the Holm-adjusted significance test must satisfy alpha = 0.05.

### H4: Budget correctness

For every forward step:

```text
memory.size <= configured_budget
```

The permitted number of memory-budget violations is zero.

### H5: Nontrivial learned memory behavior

BudgetMem-R must write nontrivially, must not reduce to retaining every most-recent state, and must retain task-relevant states more often than random selection.

## 4. Datasets

### 4.1 Confirmatory synthetic datasets

- Selective copy
- Associative recall
- Distractor retrieval

### 4.2 Secondary synthetic dataset

- Rare-event recall

### 4.3 External-validity datasets

- HDFS block-sequence dataset
- IMDb sentiment dataset

The three confirmatory synthetic tasks determine the primary same-budget long-range recall conclusion. Rare-event recall, HDFS, and IMDb are secondary robustness and external-validity analyses.

## 5. Splits and leakage controls

### Synthetic datasets

- Training, validation, and test examples must use disjoint generation seeds.
- Split manifests must record task, split, seed, example count, sequence length, and content hash.
- Test-generation seeds must not be used during training, validation, early stopping, or hyperparameter selection.

### HDFS

- HDFS block identifiers must be disjoint across training, validation, and test sets.
- All events associated with a block identifier must remain in one split.
- Split assignments must be frozen before training.

### IMDb

- Official IMDb test examples must remain test-only.
- The official training set may be divided deterministically into training and validation subsets.
- Exact duplicates and normalized-text duplicates must not cross split boundaries.

### General controls

- Preprocessing statistics and vocabularies must be fitted using training data only.
- Test labels and test metrics must not influence model or hyperparameter selection.
- Future-token modifications must not alter memory decisions made at earlier steps.
- Memory must be reset between unrelated sequences.
- Dataset manifests and hashes must be retained with experiment evidence.

## 6. Models

### Recurrent baselines

- Vanilla RNN
- GRU
- LSTM

### Deterministic or non-learned memory policies

- Uniform checkpointing
- Most-recent-state cache
- FIFO
- LRU
- Random replacement
- Reservoir sampling
- Novelty-only selection
- Surprise-only selection

### Additional baseline

- Budget-controlled attention baseline, when its effective retained-state budget is directly auditable

### Proposed model

- BudgetMem-R

## 7. Parameter budgets

The primary comparison uses the same recurrent backbone, embedding dimension, hidden dimension, output head, optimizer family, and training schedule wherever technically applicable.

Required fairness controls:

1. Record the exact trainable parameter count for every model.
2. Treat same-backbone comparisons as primary.
3. Perform a parameter-matched sensitivity analysis when total trainable parameters differ by more than 5%.
4. Select any parameter-matched width using validation data only.
5. Report memory-controller parameters separately.
6. Do not convert unused memory slots into additional hidden width.
7. Do not change model width after inspecting final test results.

## 8. Sequence lengths

The frozen sequence lengths are:

- 256
- 512
- 1,024

Sequence length 1,024 is the primary long-range condition. Lengths 256 and 512 support scaling analysis.

## 9. Memory budgets

The frozen strict external-memory budgets are:

- 16 slots
- 32 slots

All matched comparisons must use the same configured slot budget. Silent overflow and unbounded memory are prohibited.

## 10. Seeds

The frozen final experiment seeds are:

- 2026
- 2027
- 2028
- 2029
- 2030

The same seeds must be used for matched models. Each seed controls dataset generation or sampling, initialization, training order, and evaluation order.

Deterministic algorithms must be enabled where supported. Any unavoidable CUDA nondeterminism must be documented with the exact environment and limitation.

## 11. Metrics

### Primary confirmatory metric

For selective copy, associative recall, and distractor retrieval:

- `exact_match_accuracy`

The primary condition is sequence length 1,024, reported separately for budgets 16 and 32.

### Secondary synthetic-task metrics

- `token_accuracy`
- `memory_recall`
- `successful_long_range_retrievals`
- relevant-state retention
- controller write rate
- recency concentration
- memory-budget violation count
- training and validation loss
- gradient norms
- wall-clock runtime
- evaluation latency
- peak host memory
- peak accelerator memory, when applicable

### External-validity metrics

For HDFS:

- Primary: anomalous-block F1
- Secondary: precision, recall, AUROC, and AUPRC

For IMDb:

- Primary: classification accuracy
- Secondary: macro-F1, AUROC, and calibration error

The primary metric must not be replaced because another metric produces a more favorable final result.

## 12. Statistical tests

The matched experimental unit is:

```text
task × sequence length × memory budget × seed
```

For each primary deterministic comparator:

1. Compute paired BudgetMem-R minus comparator differences.
2. Report the paired mean difference in percentage points.
3. Construct a paired 95% bootstrap confidence interval using 10,000 replicates.
4. Apply a paired Wilcoxon signed-rank test when sufficient nonzero pairs are available.
5. Apply Holm correction across the primary comparator tests at alpha = 0.05.
6. Report paired rank-biserial correlation as the nonparametric effect size.

The primary superiority criterion requires all of the following against at least two deterministic policies:

- Identical task, sequence length, budget, and seed
- Mean improvement of at least 2.0 percentage points
- Paired 95% confidence-interval lower bound above zero
- Holm-adjusted p-value below 0.05
- Zero memory-budget violations

Length trends, budget interactions, controller diagnostics, parameter-matched analyses, HDFS, and IMDb are secondary analyses.

## 13. Hyperparameter-search budget

The frozen hyperparameter-search budget is:

- Maximum 20 completed trials per model-dataset family

Rules:

- Search uses training and validation data only.
- Directly compared model families receive equal search budgets unless a predefined search space is exhausted.
- Pilot results may define search ranges but may not serve as final test evidence.
- Failed trials count toward the budget unless failure is verified as an external infrastructure interruption.
- Test performance must not be used to expand, narrow, repeat, or terminate the search.
- The complete trial ledger and selected configuration must be retained.

## 14. Exclusion criteria

A run may be excluded from inferential aggregation only for a predefined technical reason:

- Verified software or infrastructure failure
- NaN or infinite loss, gradient, parameter, prediction, or metric
- Memory-budget violation
- Dataset-hash or split-manifest mismatch
- Split leakage
- Missing or unreadable checkpoint
- Missing required metric fields
- Causality-test failure
- Unexplained determinism-test failure
- Unintended gradient disconnection
- Out-of-memory termination

Poor performance is never an exclusion reason.

Every excluded run must remain in the experiment ledger with its model, task, sequence length, memory budget, seed, reason, logs, and disposition.

## 15. OOM-reporting policy

Every out-of-memory event must be retained and reported as an OOM result.

The record must include:

- Model
- Task or dataset
- Sequence length
- Memory budget
- Seed
- Batch size
- Numerical precision
- Device
- Parameter count
- Available peak-memory measurements

Silent batch-size reduction is prohibited.

A rescue rerun is permitted only under a predefined smaller batch size while preserving effective batch size through gradient accumulation. The original OOM record must remain visible, and the rescue result must be labeled separately.

## 16. Early-stopping rule

- Maximum epochs: 100
- Validation frequency: once per epoch
- Patience: 10 validation checks
- Minimum improvement: 0.0001
- Optimization direction: maximize the predefined validation analogue of the primary metric
- Tie-breaker 1: lower validation loss
- Tie-breaker 2: earlier epoch
- Final evaluation checkpoint: best validation checkpoint
- Final test evaluations: one per selected configuration and seed

The early-stopping rule must not be modified after final test inspection.

## 17. Primary comparisons

### Confirmatory comparisons

BudgetMem-R versus:

1. GRU plus uniform checkpointing
2. GRU plus reservoir sampling

These comparisons use:

- Selective copy
- Associative recall
- Distractor retrieval
- Sequence length 1,024
- Memory budgets 16 and 32
- Seeds 2026, 2027, 2028, 2029, and 2030

### Prespecified secondary comparisons

BudgetMem-R versus:

- GRU without external memory
- GRU plus most-recent-state cache
- GRU plus FIFO
- GRU plus LRU
- GRU plus random replacement
- GRU plus novelty-only selection
- GRU plus surprise-only selection
- Vanilla RNN
- LSTM
- Eligible budget-controlled attention baseline

Secondary comparisons must not replace the confirmatory comparisons.

## 18. Required gates before final experiments

Final experiments may start only after all of the following pass:

- Budget correctness
- Causality
- Determinism
- No split leakage
- Gradient flow to memory-controller parameters
- Intentional cached-state detachment behavior
- Memory reset between unrelated sequences
- Configuration provenance
- Resource-measurement correctness
- Checkpoint resumption
- Pilot go/no-go criterion

## 19. Required result provenance

Every result record must identify:

- Git commit
- Protocol tag
- Configuration file and hash
- Dataset manifest and hash
- Model
- Task or dataset
- Split
- Sequence length
- Memory budget
- Seed
- Parameter count
- Selected epoch
- Primary and secondary metrics
- Exclusion or OOM status
- Runtime and resource measurements
- Hardware and software environment

All planned cells, including failed, excluded, and OOM cells, must appear in the final result ledger.

## 20. Frozen declaration

This protocol is frozen before final test-result inspection.

Primary hypotheses, primary metrics, statistical tests, exclusion criteria, early-stopping rules, and primary comparisons will not be changed in response to final test outcomes.

Any later unplanned analysis will be labeled exploratory.
