# Section 15 Pilot Experiment Report

**Decision:** `NO_GO`

Do not begin the full experiment matrix.

## Go/No-Go Criteria

| Criterion | Result |
|---|---:|
| training_stability | PASS |
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
| budgetmem_r | 0.005208 |
| gru_uniform_cache | 0.007812 |
| gru_reservoir_cache | 0.005932 |

## Training Records

| Task | Model | First loss | Final loss | Stable | Resume |
|---|---|---:|---:|---:|---:|
| selective_copy | gru | 5.268460 | 5.070259 | True | True |
| selective_copy | gru_uniform_cache | 5.259773 | 5.057557 | True | True |
| selective_copy | gru_reservoir_cache | 5.266052 | 5.103273 | True | True |
| selective_copy | budgetmem_r | 6.871038 | 5.138987 | True | True |
| associative_recall | gru | 5.256029 | 5.073089 | True | True |
| associative_recall | gru_uniform_cache | 5.255232 | 5.101212 | True | True |
| associative_recall | gru_reservoir_cache | 5.241677 | 5.062188 | True | True |
| associative_recall | budgetmem_r | 7.508906 | 5.141234 | True | True |
| distractor_heavy_retrieval | gru | 5.265067 | 5.097738 | True | True |
| distractor_heavy_retrieval | gru_uniform_cache | 5.265892 | 5.064460 | True | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 5.269871 | 5.104188 | True | True |
| distractor_heavy_retrieval | budgetmem_r | 5.549648 | 5.163944 | True | True |

## Evaluation Matrix

| Task | Model | Length | Budget | Token accuracy | Memory recall | Write frequency | Budget pass |
|---|---|---:|---:|---:|---:|---:|---:|
| selective_copy | gru | 256 | 16 | 0.005208 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 256 | 32 | 0.000000 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 512 | 16 | 0.000000 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 512 | 32 | 0.013021 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 1024 | 16 | 0.005208 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 1024 | 32 | 0.007812 | 0.000000 | 0.000000 | True |
| selective_copy | gru_uniform_cache | 256 | 16 | 0.007812 | 0.049479 | 0.062500 | True |
| selective_copy | gru_uniform_cache | 256 | 32 | 0.013021 | 0.109375 | 0.125000 | True |
| selective_copy | gru_uniform_cache | 512 | 16 | 0.005208 | 0.002604 | 0.031250 | True |
| selective_copy | gru_uniform_cache | 512 | 32 | 0.005208 | 0.039062 | 0.062500 | True |
| selective_copy | gru_uniform_cache | 1024 | 16 | 0.005208 | 0.000000 | 0.015625 | True |
| selective_copy | gru_uniform_cache | 1024 | 32 | 0.000000 | 0.013021 | 0.031250 | True |
| selective_copy | gru_reservoir_cache | 256 | 16 | 0.013021 | 0.132812 | 0.231445 | True |
| selective_copy | gru_reservoir_cache | 256 | 32 | 0.007812 | 0.257812 | 0.372070 | True |
| selective_copy | gru_reservoir_cache | 512 | 16 | 0.007812 | 0.085938 | 0.140869 | True |
| selective_copy | gru_reservoir_cache | 512 | 32 | 0.005208 | 0.140625 | 0.232422 | True |
| selective_copy | gru_reservoir_cache | 1024 | 16 | 0.005208 | 0.049479 | 0.081787 | True |
| selective_copy | gru_reservoir_cache | 1024 | 32 | 0.002604 | 0.096354 | 0.138794 | True |
| selective_copy | budgetmem_r | 256 | 16 | 0.000000 | 0.289062 | 0.103841 | True |
| selective_copy | budgetmem_r | 256 | 32 | 0.000000 | 0.364583 | 0.139079 | True |
| selective_copy | budgetmem_r | 512 | 16 | 0.015625 | 0.231771 | 0.072917 | True |
| selective_copy | budgetmem_r | 512 | 32 | 0.005208 | 0.348958 | 0.088542 | True |
| selective_copy | budgetmem_r | 1024 | 16 | 0.005208 | 0.161458 | 0.061239 | True |
| selective_copy | budgetmem_r | 1024 | 32 | 0.005208 | 0.302083 | 0.060689 | True |
| associative_recall | gru | 256 | 16 | 0.020833 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 256 | 32 | 0.000000 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 512 | 16 | 0.000000 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 512 | 32 | 0.000000 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 1024 | 16 | 0.000000 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 1024 | 32 | 0.000000 | 0.000000 | 0.000000 | True |
| associative_recall | gru_uniform_cache | 256 | 16 | 0.000000 | 0.041667 | 0.062500 | True |
| associative_recall | gru_uniform_cache | 256 | 32 | 0.000000 | 0.145833 | 0.125000 | True |
| associative_recall | gru_uniform_cache | 512 | 16 | 0.000000 | 0.000000 | 0.031250 | True |
| associative_recall | gru_uniform_cache | 512 | 32 | 0.020833 | 0.104167 | 0.062500 | True |
| associative_recall | gru_uniform_cache | 1024 | 16 | 0.020833 | 0.000000 | 0.015625 | True |
| associative_recall | gru_uniform_cache | 1024 | 32 | 0.020833 | 0.000000 | 0.031250 | True |
| associative_recall | gru_reservoir_cache | 256 | 16 | 0.020833 | 0.104167 | 0.238770 | True |
| associative_recall | gru_reservoir_cache | 256 | 32 | 0.000000 | 0.208333 | 0.393555 | True |
| associative_recall | gru_reservoir_cache | 512 | 16 | 0.000000 | 0.083333 | 0.140137 | True |
| associative_recall | gru_reservoir_cache | 512 | 32 | 0.000000 | 0.208333 | 0.242188 | True |
| associative_recall | gru_reservoir_cache | 1024 | 16 | 0.000000 | 0.062500 | 0.080200 | True |
| associative_recall | gru_reservoir_cache | 1024 | 32 | 0.000000 | 0.041667 | 0.142090 | True |
| associative_recall | budgetmem_r | 256 | 16 | 0.000000 | 0.083333 | 0.131592 | True |
| associative_recall | budgetmem_r | 256 | 32 | 0.020833 | 0.375000 | 0.185547 | True |
| associative_recall | budgetmem_r | 512 | 16 | 0.000000 | 0.125000 | 0.112508 | True |
| associative_recall | budgetmem_r | 512 | 32 | 0.020833 | 0.250000 | 0.148600 | True |
| associative_recall | budgetmem_r | 1024 | 16 | 0.000000 | 0.041667 | 0.101969 | True |
| associative_recall | budgetmem_r | 1024 | 32 | 0.020833 | 0.083333 | 0.129720 | True |
| distractor_heavy_retrieval | gru | 256 | 16 | 0.006944 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 256 | 32 | 0.000000 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 512 | 16 | 0.000000 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 512 | 32 | 0.020833 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 1024 | 16 | 0.006944 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 1024 | 32 | 0.006944 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 256 | 16 | 0.000000 | 0.055556 | 0.062500 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 256 | 32 | 0.000000 | 0.125000 | 0.125000 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 512 | 16 | 0.000000 | 0.027778 | 0.031250 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 512 | 32 | 0.000000 | 0.083333 | 0.062500 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 1024 | 16 | 0.000000 | 0.027778 | 0.015625 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 1024 | 32 | 0.000000 | 0.006944 | 0.031250 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 256 | 16 | 0.006944 | 0.083333 | 0.220215 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 256 | 32 | 0.000000 | 0.111111 | 0.377930 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 512 | 16 | 0.000000 | 0.020833 | 0.129150 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 512 | 32 | 0.006944 | 0.069444 | 0.228027 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 1024 | 16 | 0.006944 | 0.020833 | 0.076294 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 1024 | 32 | 0.020833 | 0.048611 | 0.134766 | True |
| distractor_heavy_retrieval | budgetmem_r | 256 | 16 | 0.000000 | 0.027778 | 0.138346 | True |
| distractor_heavy_retrieval | budgetmem_r | 256 | 32 | 0.013889 | 0.180556 | 0.125244 | True |
| distractor_heavy_retrieval | budgetmem_r | 512 | 16 | 0.013889 | 0.013889 | 0.141113 | True |
| distractor_heavy_retrieval | budgetmem_r | 512 | 32 | 0.006944 | 0.062500 | 0.115316 | True |
| distractor_heavy_retrieval | budgetmem_r | 1024 | 16 | 0.000000 | 0.000000 | 0.149658 | True |
| distractor_heavy_retrieval | budgetmem_r | 1024 | 32 | 0.000000 | 0.034722 | 0.118835 | True |
