# Novelty Matrix: Budget-Conditioned Recurrent Memory

**Review date:** 2026-07-11  
**Project:** Adaptive Memory Sequence Benchmark / BudgetMem-R  
**Decision purpose:** Determine whether the proposed method is sufficiently distinct to justify full implementation and the complete experimental campaign.

## 1. Scope and Interpretation Rules

This review uses conservative classifications.

- **Yes:** The capability is an explicit component of the method.
- **Partial:** The method has a related mechanism, but it does not satisfy the full definition used in this project.
- **No:** The capability is absent or structurally different.
- **N/A:** The item is a benchmark or category for which the column does not apply.
- **Unclear:** The reviewed paper does not establish the capability clearly enough.

Column definitions:

- **Recurrent states:** The method stores or manipulates recurrent hidden states or an equivalent recurrent memory state.
- **Fixed hard budget:** Runtime memory capacity is explicitly capped, rather than merely reduced on average.
- **Learned writing:** A learned controller decides what or how information is written into memory.
- **Learned eviction:** A learned policy decides which retained memory item is removed, expired, or overwritten.
- **Learned retrieval:** A learned mechanism selects or weights stored memory according to the current query or state.
- **Variable-budget model:** One trained model or controller is explicitly conditioned to operate across multiple memory budgets without retraining a separate model for every budget.
- **Length extrapolation:** The method is evaluated beyond its nominal training context or demonstrates transfer to substantially longer sequences.
- **Real-world validation:** The method is evaluated on non-synthetic public datasets or downstream tasks. This does **not** imply production deployment.

## 2. Comparison Matrix

| Method | Recurrent states | Fixed hard budget | Learned writing | Learned eviction | Learned retrieval | Variable-budget model | Length extrapolation | Real-world validation |
|---|---|---|---|---|---|---|---|---|
| **Proposed BudgetMem-R** | **Required** | **Required** | **Required** | **Required** | **Required** | **Required** | **Required** | **Required** |
| Memory Caching | Yes | No | Partial | No | Yes | No | Yes | Yes |
| ATLAS | Yes | Yes | Yes | Partial | Yes | No | Yes | Yes |
| Titans | Yes | Yes | Yes | Partial | Yes | No | Yes | Yes |
| CogScale | N/A: benchmark | N/A | N/A | N/A | N/A | N/A | Yes: evaluates scale | No: synthetic benchmark |
| Recurrent Memory Transformer | Yes | Yes | Yes | No | Yes | No | Yes | Partial |
| Compressive Transformer | Partial: recurrent segment memory | Yes | Partial: learned compression | No: scheduled compression | Yes | No | Partial | Yes |
| Expire-Span | No: retained Transformer states | No: no exact global capacity guarantee | No: states enter memory normally | Yes | Yes | No | Partial | Yes |
| IndexMem | No: KV cache plus latent memory | Yes | Yes: latent-memory update | Yes | Yes | Unclear | Partial | Yes |
| ForesightKV | No: KV cache | Yes | No: standard KV insertion | Yes | No: standard attention read | No | Partial | Yes |
| Learned global KV retention (LKV) | No: KV cache | Yes | No: standard KV insertion | Yes | No: standard attention read | Partial: evaluated across ratios, but not established as a single budget-conditioned controller | Yes | Yes |
| Tensor Cache | Partial: recurrent associative memory, not recurrent hidden-state caching | Yes | Partial: learned write rate, fixed eviction trigger | No: sliding-window eviction | Partial: associative read plus learned fusion gate | No | Yes | Yes |
| Standard memory-augmented neural networks | Yes | Yes | Yes | Yes or Partial, depending on architecture | Yes | No | Partial | Partial |

## 3. Method-by-Method Novelty Findings

### 3.1 Memory Caching

Memory Caching is the most important architectural comparison because it explicitly caches recurrent memory checkpoints and allows later tokens to access selected cached states. Its Sparse Selective Caching variant uses a router to select relevant cached memories.

The principal distinction is resource control. Memory Caching intentionally allows effective memory to grow with sequence length. It does not impose a strict fixed-capacity pool with learned eviction. Therefore, BudgetMem-R must not be described merely as “caching recurrent states.” Its defensible novelty must be:

1. a strict hard memory budget;
2. explicit learned eviction among recurrent states;
3. one budget-conditioned controller that works across multiple capacities; and
4. joint write, eviction, and retrieval decisions under online causality.

### 3.2 ATLAS and Titans

ATLAS and Titans use recurrent neural memory modules with learned test-time memory updates and learned retrieval. Their memory is fixed-size in the sense that information is compressed into a bounded neural state. They also include forgetting, decay, gating, or context-pruning mechanisms.

The proposed method remains distinct only if it maintains an explicit set of retained recurrent hidden states and learns discrete or constrained write–evict–retrieve actions under a hard item or byte budget. A generic learned forgetting gate is not enough to establish novelty over ATLAS or Titans.

### 3.3 Recurrent Memory Transformer

Recurrent Memory Transformer uses a fixed number of special memory tokens passed recurrently between segments. The model learns memory operations through standard training.

The proposed method must distinguish itself through explicit capacity-conditioned retention and eviction. Merely passing a fixed recurrent memory between segments would reproduce the core RMT idea.

### 3.4 Compressive Transformer and Expire-Span

Compressive Transformer bounds memory by moving older activations into a compressed store according to a predetermined schedule. Expire-Span learns how long each memory remains available, but it does not provide the same strict global capacity constraint required here.

BudgetMem-R should emphasize exact capacity satisfaction and competition among candidate recurrent states, not only learned forgetting or age-based compression.

### 3.5 IndexMem, ForesightKV, LKV, and Tensor Cache

These methods are strong comparisons for learned cache management, but they operate primarily on Transformer key-value caches or associative memories rather than cached recurrent hidden states.

Important distinctions:

- **IndexMem** learns KV importance and compresses evicted tokens into an online latent memory.
- **ForesightKV** learns eviction from targets produced using future attention information. This creates a direct comparison for the proposed strict-causality claim.
- **LKV** jointly learns global KV-budget allocation and token selection under an exact retention constraint.
- **Tensor Cache** uses a fixed local KV cache and a bounded associative second-level memory for evicted information.

BudgetMem-R cannot claim novelty from “learned eviction under a fixed budget” alone. That concept already exists in KV-cache research. The claim must be specifically tied to recurrent hidden states, multi-budget conditioning, causal action learning, transfer, and measured systems behavior.

### 3.6 Standard Memory-Augmented Neural Networks

Neural Turing Machines, Differentiable Neural Computers, and related memory-augmented neural networks already support differentiable reading, writing, allocation, and overwrite operations over bounded external memory.

Therefore, the proposed work must not claim that joint memory control itself is new. The defensible contribution is its formulation and validation for modern recurrent sequence models under explicit resource budgets, cross-length and cross-task transfer, strict online causality, and systems-level accounting.

## 4. Novelty Threats

### Threat 1: The method becomes bounded Memory Caching

If the design only caches recurrent checkpoints and applies a heuristic top-k rule, it will appear to be a hard-capped variant of Memory Caching.

**Required response:** Learn the write, eviction, and retrieval policies jointly, condition them on the requested budget, and evaluate transfer across budgets.

### Threat 2: The method becomes another recurrent memory-token architecture

If memory consists only of a fixed set of continuously updated tokens, the method will be difficult to distinguish from RMT and related recurrent-memory Transformers.

**Required response:** Define explicit retained-state candidates, admission decisions, eviction decisions, and query-dependent retrieval.

### Threat 3: The method becomes KV-cache eviction applied to an RNN

If the main contribution is only a learned score that retains the top-k states, reviewers may view it as a direct transplant of ForesightKV, LKV, or IndexMem.

**Required response:** Demonstrate that recurrent hidden-state retention creates a distinct control problem and that a joint policy outperforms independent scoring, recency, reservoir sampling, and oracle-inspired baselines.

### Threat 4: The controller uses privileged future information

A controller trained from future labels, future attention, or full-sequence utility targets may violate the proposed online-causality claim.

**Required response:** At inference, every action at time \(t\) must depend only on the prefix \(x_{\leq t}\), current recurrent state, retained memory, current budget, and past controller actions. Any offline oracle may be used only as an explicitly labeled upper bound, not as information available to the deployed controller.

### Threat 5: Multiple budgets require separately trained models

Training one model for every memory size would invalidate the variable-budget contribution.

**Required response:** Encode the budget as an input to the controller and evaluate one checkpoint across all designated budgets.

## 5. Recommended Contribution Set

The project should commit to the following contributions:

1. **Budget-conditioned recurrent memory:** One trained model supports multiple hard memory budgets through an explicit budget signal.
2. **Joint write–evict–retrieve control:** A unified controller manages admission, removal, and query-dependent access to recurrent hidden states.
3. **Strict online causality:** No action uses future labels, future attention, future tokens, or noncausal oracle information.
4. **Cross-length and cross-task transfer:** The same controller is evaluated on sequence lengths and tasks not used during controller training.
5. **Hard resource accounting:** Report exact retained-state counts or bytes, peak process memory, inference latency, throughput, and controller overhead.
6. **Policy interpretability:** Record retention scores, eviction reasons, retrieval weights, state ages, state reuse, and budget-dependent policy changes.

## 6. Minimum Novelty Gate

The method must demonstrate at least three **clearly distinct and implemented** contributions before the complete experimental campaign begins.

### Mandatory core contributions

The following three form the minimum defensible core:

- **C1:** One budget-conditioned model supports multiple hard memory budgets.
- **C2:** Joint learned write–evict–retrieve control is applied to recurrent hidden states.
- **C3:** The deployed controller is strictly online and causal.

### Strong supporting contributions

At least one of the following should also be implemented:

- **C4:** Cross-length and cross-task controller transfer.
- **C5:** Hard memory and latency accounting using measured runtime data.
- **C6:** Interpretable policy traces explaining retention and eviction behavior.

## 7. Gate Decision

**Current decision: CONDITIONAL PASS FOR METHOD PROTOTYPING.**

The proposal contains at least three conceptually distinct contributions, but they are not yet established by implementation or evidence. Therefore:

- Proceed to formal method specification, controller pseudocode, and a small proof-of-concept implementation.
- Do **not** begin the complete experimental campaign yet.
- Upgrade the gate to **PASS FOR FULL EXPERIMENTS** only after C1, C2, and C3 are implemented and verified by targeted tests.
- Preferably verify at least one of C4, C5, or C6 before launching large sweeps.

## 8. Required Pre-Campaign Verification Tests

The following tests must pass:

1. **Budget invariance test:** For every requested budget \(B\), retained memory never exceeds \(B\) items or the specified byte limit.
2. **Single-checkpoint test:** The same trained checkpoint runs at all target budgets.
3. **Causality test:** Perturbing future tokens does not change controller actions before the perturbation point.
4. **Joint-control ablation:** Joint control outperforms separate or heuristic write, eviction, and retrieval policies under matched budgets.
5. **Closest-prior-work baselines:** Compare directly with bounded Memory Caching, RMT-style memory, recency, random eviction, reservoir sampling, learned top-k retention, and a standard fixed-state recurrent model.
6. **Resource measurement test:** Report actual memory and latency rather than theoretical complexity alone.
7. **Trace audit:** Save per-step actions and verify that every retention or eviction decision can be reconstructed from logged causal inputs.

## 9. Claim Language to Use

Use the following formulation:

> We study budget-conditioned control of an explicit cache of recurrent hidden states. Unlike growing recurrent-state caching, fixed recurrent memory tokens, and learned Transformer KV-cache eviction, the proposed controller jointly learns causal write, eviction, and retrieval actions under an exact runtime memory budget, while one checkpoint operates across multiple budgets.

Do not claim:

- the first learned memory controller;
- the first learned eviction method;
- the first bounded external memory;
- the first recurrent memory architecture;
- the first method to cache recurrent states; or
- the first differentiable read/write memory system.

## 10. Primary References

1. Behrouz et al., **Memory Caching: RNNs with Growing Memory**, arXiv:2602.24281.
2. Behrouz et al., **ATLAS: Learning to Optimally Memorize the Context at Test Time**, arXiv:2505.23735.
3. Behrouz et al., **Titans: Learning to Memorize at Test Time**, arXiv:2501.00663.
4. Bendi-Ouis et al., **CogScale: Scalable Benchmark for Sequence Processing**, arXiv:2605.19758.
5. Bulatov et al., **Recurrent Memory Transformer**, arXiv:2207.06881.
6. Rae et al., **Compressive Transformers for Long-Range Sequence Modelling**, arXiv:1911.05507.
7. Sukhbaatar et al., **Not All Memories Are Created Equal: Learning to Forget by Expiring**, arXiv:2105.06548.
8. Yang et al., **IndexMem: Learned KV-Cache Eviction with Latent Memory for Long-Context LLM Inference**, arXiv:2605.25475.
9. Dong et al., **ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution**, arXiv:2602.03203.
10. Zhou et al., **LKV: End-to-End Learning of Head-wise Budgets and Token Selection for LLM KV Cache Eviction**, arXiv:2605.06676.
11. Swain et al., **Tensor Cache: Eviction-conditioned Associative Memory for Transformers**, arXiv:2605.22884.
12. Graves et al., **Neural Turing Machines**, arXiv:1410.5401.
