# Section 13 — BudgetMem-R

## Architecture

BudgetMem-R uses a GRU recurrent backbone. Each recurrent hidden state is projected into a memory key and value. The external memory is tensorized per batch item and enforces a strict deployment budget by maintaining an explicit valid-slot mask and checking the invariant after every write or eviction.

The requested budget is normalized by the configured maximum and encoded by a learned budget conditioner. The resulting embedding is supplied to the write controller, future-utility eviction controller, and retrieval query projection. During training, a budget is sampled independently for each batch item from `8, 16, 32, 64, 128` unless an explicit budget is supplied.

## Write policy

The write controller uses only causal and label-free signals:

- recurrent hidden state;
- hidden-state novelty;
- self-supervised next-input surprise;
- prediction uncertainty;
- current memory occupancy;
- time since the previous write;
- retrieved-memory agreement; and
- requested-budget embedding.

Training uses a straight-through relaxed Bernoulli gate. Evaluation uses a hard threshold.

## Leakage control

The model `forward` method does not accept task labels. Classification or anomaly labels can therefore affect only the primary task loss after the forward pass. Controller surprise is computed from the previous step's auxiliary next-input prediction and the newly observed input. This is a causal self-supervised signal and does not expose the final sentiment, anomaly, or class label.

## Eviction and retrieval

When a sample's memory is full and a hard write is selected, the future-utility controller scores resident slots and the lowest-scoring valid slot is replaced. Retrieval constructs a budget-conditioned query, selects valid top-k keys by scaled dot product, applies a softmax over selected scores, and returns the weighted value sum.

Supported fusion comparisons are concatenation, residual addition, gated fusion, and attention fusion.

## Objective

The training objective contains:

- primary task loss;
- budget-overflow penalty;
- excessive-write penalty;
- auxiliary next-input prediction loss; and
- memory diversity loss.

The budget penalty remains useful as an auditable safety term, while the implementation separately guarantees that resident memory never exceeds the requested budget.

## Verification

Run:

```bash
PYTHONPATH=src python -m pytest -q tests/test_budgetmem_r.py
PYTHONPATH=src python scripts/verify_section13.py
```

The verification report is written to `reports/evidence/section13_budgetmem_r_implementation.txt`.
