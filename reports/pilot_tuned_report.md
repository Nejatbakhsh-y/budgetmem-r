# Section 15 Pilot Experiment Report

**Decision:** `NO_GO`

Do not begin the full experiment matrix.

## Go/No-Go Criteria

| Criterion | Result |
|---|---:|
| training_stability | FAIL |
| strict_memory_budget | PASS |
| nontrivial_controller_writes | PASS |
| not_recent_state_copying | PASS |
| retention_exceeds_random | PASS |
| resource_measurements_valid | PASS |
| checkpoint_resumption | PASS |
| configuration_provenance | PASS |
| outperforms_two_memory_policies | FAIL |

## Long-Range Accuracy

| Model | Token accuracy at length 1,024 |
|---|---:|
| budgetmem_r | 0.012731 |
| gru_uniform_cache | 0.006366 |
| gru_reservoir_cache | 0.005208 |

## Training Records

| Task | Model | First loss | Final loss | Stable | Resume |
|---|---|---:|---:|---:|---:|
| selective_copy | gru | 5.265632 | 4.957169 | True | True |
| selective_copy | gru_uniform_cache | 5.257771 | 4.943180 | True | True |
| selective_copy | gru_reservoir_cache | 5.262793 | 4.950853 | True | True |
| selective_copy | budgetmem_r | 5.404407 | 4.897789 | False | True |
| associative_recall | gru | 5.270842 | 4.880953 | True | True |
| associative_recall | gru_uniform_cache | 5.258583 | 4.870448 | True | True |
| associative_recall | gru_reservoir_cache | 5.250955 | 4.837621 | True | True |
| associative_recall | budgetmem_r | 5.390627 | 4.912584 | False | True |
| distractor_heavy_retrieval | gru | 5.263446 | 4.878850 | True | True |
| distractor_heavy_retrieval | gru_uniform_cache | 5.260176 | 4.904074 | True | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 5.265151 | 4.880176 | True | True |
| distractor_heavy_retrieval | budgetmem_r | 5.432482 | 4.867406 | False | True |

## Evaluation Matrix

| Task | Model | Length | Budget | Token accuracy | Memory recall | Write frequency | Budget pass |
|---|---|---:|---:|---:|---:|---:|---:|
| selective_copy | gru | 256 | 16 | 0.006510 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 256 | 32 | 0.007812 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 512 | 16 | 0.003906 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 512 | 32 | 0.005208 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 1024 | 16 | 0.006510 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 1024 | 32 | 0.002604 | 0.000000 | 0.000000 | True |
| selective_copy | gru_uniform_cache | 256 | 16 | 0.009115 | 0.042969 | 0.062500 | True |
| selective_copy | gru_uniform_cache | 256 | 32 | 0.009115 | 0.110677 | 0.125000 | True |
| selective_copy | gru_uniform_cache | 512 | 16 | 0.003906 | 0.002604 | 0.031250 | True |
| selective_copy | gru_uniform_cache | 512 | 32 | 0.009115 | 0.035156 | 0.062500 | True |
| selective_copy | gru_uniform_cache | 1024 | 16 | 0.002604 | 0.000000 | 0.015625 | True |
| selective_copy | gru_uniform_cache | 1024 | 32 | 0.007812 | 0.009115 | 0.031250 | True |
| selective_copy | gru_reservoir_cache | 256 | 16 | 0.006510 | 0.148438 | 0.231445 | True |
| selective_copy | gru_reservoir_cache | 256 | 32 | 0.007812 | 0.255208 | 0.372070 | True |
| selective_copy | gru_reservoir_cache | 512 | 16 | 0.011719 | 0.084635 | 0.140869 | True |
| selective_copy | gru_reservoir_cache | 512 | 32 | 0.006510 | 0.148438 | 0.232422 | True |
| selective_copy | gru_reservoir_cache | 1024 | 16 | 0.006510 | 0.049479 | 0.081787 | True |
| selective_copy | gru_reservoir_cache | 1024 | 32 | 0.003906 | 0.108073 | 0.138794 | True |
| selective_copy | budgetmem_r | 256 | 16 | 0.003906 | 0.464844 | 0.141724 | True |
| selective_copy | budgetmem_r | 256 | 32 | 0.006510 | 0.882812 | 0.186401 | True |
| selective_copy | budgetmem_r | 512 | 16 | 0.009115 | 0.472656 | 0.126912 | True |
| selective_copy | budgetmem_r | 512 | 32 | 0.001302 | 0.876302 | 0.110779 | True |
| selective_copy | budgetmem_r | 1024 | 16 | 0.003906 | 0.455729 | 0.074473 | True |
| selective_copy | budgetmem_r | 1024 | 32 | 0.006510 | 0.868490 | 0.069631 | True |
| associative_recall | gru | 256 | 16 | 0.000000 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 256 | 32 | 0.000000 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 512 | 16 | 0.031250 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 512 | 32 | 0.000000 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 1024 | 16 | 0.010417 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 1024 | 32 | 0.000000 | 0.000000 | 0.000000 | True |
| associative_recall | gru_uniform_cache | 256 | 16 | 0.010417 | 0.041667 | 0.062500 | True |
| associative_recall | gru_uniform_cache | 256 | 32 | 0.010417 | 0.135417 | 0.125000 | True |
| associative_recall | gru_uniform_cache | 512 | 16 | 0.000000 | 0.000000 | 0.031250 | True |
| associative_recall | gru_uniform_cache | 512 | 32 | 0.000000 | 0.093750 | 0.062500 | True |
| associative_recall | gru_uniform_cache | 1024 | 16 | 0.000000 | 0.000000 | 0.015625 | True |
| associative_recall | gru_uniform_cache | 1024 | 32 | 0.010417 | 0.000000 | 0.031250 | True |
| associative_recall | gru_reservoir_cache | 256 | 16 | 0.000000 | 0.062500 | 0.238770 | True |
| associative_recall | gru_reservoir_cache | 256 | 32 | 0.000000 | 0.218750 | 0.393555 | True |
| associative_recall | gru_reservoir_cache | 512 | 16 | 0.000000 | 0.114583 | 0.140137 | True |
| associative_recall | gru_reservoir_cache | 512 | 32 | 0.000000 | 0.145833 | 0.242188 | True |
| associative_recall | gru_reservoir_cache | 1024 | 16 | 0.010417 | 0.052083 | 0.080200 | True |
| associative_recall | gru_reservoir_cache | 1024 | 32 | 0.000000 | 0.062500 | 0.142090 | True |
| associative_recall | budgetmem_r | 256 | 16 | 0.010417 | 0.281250 | 0.468506 | True |
| associative_recall | budgetmem_r | 256 | 32 | 0.000000 | 0.729167 | 0.457072 | True |
| associative_recall | budgetmem_r | 512 | 16 | 0.000000 | 0.364583 | 0.416260 | True |
| associative_recall | budgetmem_r | 512 | 32 | 0.000000 | 0.666667 | 0.355306 | True |
| associative_recall | budgetmem_r | 1024 | 16 | 0.010417 | 0.291667 | 0.314351 | True |
| associative_recall | budgetmem_r | 1024 | 32 | 0.031250 | 0.541667 | 0.241709 | True |
| distractor_heavy_retrieval | gru | 256 | 16 | 0.013889 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 256 | 32 | 0.003472 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 512 | 16 | 0.003472 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 512 | 32 | 0.010417 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 1024 | 16 | 0.003472 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 1024 | 32 | 0.010417 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 256 | 16 | 0.013889 | 0.038194 | 0.062500 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 256 | 32 | 0.010417 | 0.138889 | 0.125000 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 512 | 16 | 0.000000 | 0.020833 | 0.031250 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 512 | 32 | 0.006944 | 0.090278 | 0.062500 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 1024 | 16 | 0.010417 | 0.027778 | 0.015625 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 1024 | 32 | 0.006944 | 0.031250 | 0.031250 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 256 | 16 | 0.006944 | 0.072917 | 0.220215 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 256 | 32 | 0.003472 | 0.111111 | 0.377930 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 512 | 16 | 0.010417 | 0.024306 | 0.129150 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 512 | 32 | 0.003472 | 0.076389 | 0.228027 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 1024 | 16 | 0.003472 | 0.010417 | 0.076294 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 1024 | 32 | 0.006944 | 0.045139 | 0.134766 | True |
| distractor_heavy_retrieval | budgetmem_r | 256 | 16 | 0.003472 | 0.055556 | 0.115967 | True |
| distractor_heavy_retrieval | budgetmem_r | 256 | 32 | 0.000000 | 0.145833 | 0.185099 | True |
| distractor_heavy_retrieval | budgetmem_r | 512 | 16 | 0.020833 | 0.010417 | 0.055847 | True |
| distractor_heavy_retrieval | budgetmem_r | 512 | 32 | 0.006944 | 0.076389 | 0.088298 | True |
| distractor_heavy_retrieval | budgetmem_r | 1024 | 16 | 0.017361 | 0.006944 | 0.027476 | True |
| distractor_heavy_retrieval | budgetmem_r | 1024 | 32 | 0.006944 | 0.034722 | 0.053497 | True |
