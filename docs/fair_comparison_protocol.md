# Section 17 — Fair Comparison Protocol

Generated: 2026-07-21T23:39:10Z

This protocol freezes the controls required for fair comparison before the final experiment matrix is executed.

## 1. Parameter matching

Two trainable-parameter regimes are required:

| Regime | Target | Permitted interval |
|---|---:|---:|
| Small | 1,000,000 | 950,000–1,050,000 |
| Medium | 5,000,000 | 4,750,000–5,250,000 |

Every model configuration must report its exact trainable-parameter count. A model outside the permitted interval is acceptable only when its architecture cannot match more closely and a written exception is recorded.

Use the generic counter as follows:

```bash
.venv/bin/python scripts/count_model_parameters.py \
  --model MODEL_NAME \
  --regime small \
  --factory package.module:factory_function \
  --kwargs-json '{"argument": "value"}'
```

The factory must return the instantiated PyTorch model.

## 2. Frozen training budget

The following values must remain identical across compared models:

| Control | Frozen value |
|---|---|
| Training tokens | 10485760 |
| Maximum optimization steps | 10000 |
| Gradient accumulation | 1 |
| Precision | fp32 |
| Hardware | Linux | 6.18.33.2-microsoft-standard-WSL2 | x86_64 | x86_64 | CPU |
| Hyperparameter-search trials | 10 |
| Evaluation frequency | Every 500 steps |
| Batch control | Fixed tokens per step |
| Tokens per step | 8192 |
| Timeout threshold | 3600 seconds |

Training batch sizes are derived from:

```text
tokens per step = batch size × sequence length
```

| Training sequence length | Required batch size | Tokens per step |
|---:|---:|---:|
| 256 | 32 | 8192 |
| 512 | 16 | 8192 |
| 1,024 | 8 | 8192 |

Gradient accumulation is frozen independently and must not be varied between compared runs.

## 3. Latency testing

Latency measurements must use fixed batch sizes:

- Batch size 1
- Batch size 8

The protocol records ten warm-up iterations and one hundred measured iterations. Median and p95 latency must be reported.

## 4. Memory and retrieval budgets

Allowed memory budgets:

```text
B = 8, 16, 32, 64, 128
```

Primary memory-budget comparisons:

```text
B = 32 and B = 64
```

Allowed retrieval budgets:

```text
k = 1, 4, 8
```

Primary retrieval configuration:

```text
k = 4
```

## 5. Sequence lengths

Training:

```text
256, 512, 1,024
```

Testing:

```text
256, 512, 1,024, 2,048, 4,096, 8,192
```

Sequence length 8,192 may be run only after the task and architecture are explicitly marked reliable at that length.

## 6. OOM and timeout reporting

Every attempted run must have one of these statuses:

```text
completed, oom, timeout, failed
```

OOM, timeout, and failed rows must record:

- Explicit failure reason
- Elapsed time
- Configuration path
- Git commit
- Task, model, parameter regime, sequence length, memory budget, retrieval budget, and seed

Do not silently remove failed runs from the manifest.

## 7. Primary comparison matching keys

A primary comparison is valid only when both rows match on:

- Task
- Sequence length
- Random seed
- Parameter regime
- Training-token budget
- Maximum optimization steps
- Gradient accumulation
- Precision
- Hardware
- Hyperparameter-search trials
- Evaluation frequency
- Batch-control rule
- Memory budget, restricted to 32 or 64
- Retrieval budget, fixed at 4

## 8. Required evidence files

The automation creates:

- `configs/fair_comparison.yaml`
- `reports/tables/model_parameter_counts.csv`
- `reports/tables/fair_comparison_run_manifest.csv`
- `reports/evidence/section17_fair_comparison_report.txt`

Run the audit after parameter registration and after every experiment batch:

```bash
.venv/bin/python scripts/audit_fair_comparison.py
```

The initial audit may show exact parameter counts and experiment runs as `NOT_RUN`. Those statuses become `PASS` only after the corresponding CSV tables contain valid records.
