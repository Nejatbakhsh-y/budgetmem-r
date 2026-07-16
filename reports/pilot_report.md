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
| budgetmem_r | 0.004051 |
| gru_uniform_cache | 0.006221 |
| gru_reservoir_cache | 0.004774 |

## Training Records

| Task | Model | First loss | Final loss | Stable | Resume |
|---|---|---:|---:|---:|---:|
| selective_copy | gru | 5.267989 | 4.640260 | True | True |
| selective_copy | gru_uniform_cache | 5.260620 | 4.622276 | True | True |
| selective_copy | gru_reservoir_cache | 5.265009 | 4.629843 | True | True |
| selective_copy | budgetmem_r | 5.502837 | 4.748610 | False | True |
| associative_recall | gru | 5.256309 | 4.674231 | True | True |
| associative_recall | gru_uniform_cache | 5.254156 | 4.563963 | True | True |
| associative_recall | gru_reservoir_cache | 5.237401 | 4.517407 | True | True |
| associative_recall | budgetmem_r | 5.594399 | 4.721569 | False | True |
| distractor_heavy_retrieval | gru | 5.265388 | 4.665494 | True | True |
| distractor_heavy_retrieval | gru_uniform_cache | 5.263244 | 4.563617 | True | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 5.267767 | 4.549723 | True | True |
| distractor_heavy_retrieval | budgetmem_r | 5.575896 | 4.652171 | True | True |

## Evaluation Matrix

| Task | Model | Length | Budget | Token accuracy | Memory recall | Write frequency | Budget pass |
|---|---|---:|---:|---:|---:|---:|---:|
| selective_copy | gru | 256 | 16 | 0.000000 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 256 | 32 | 0.002604 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 512 | 16 | 0.002604 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 512 | 32 | 0.005208 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 1024 | 16 | 0.007812 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 1024 | 32 | 0.010417 | 0.000000 | 0.000000 | True |
| selective_copy | gru_uniform_cache | 256 | 16 | 0.002604 | 0.049479 | 0.062500 | True |
| selective_copy | gru_uniform_cache | 256 | 32 | 0.002604 | 0.109375 | 0.125000 | True |
| selective_copy | gru_uniform_cache | 512 | 16 | 0.002604 | 0.002604 | 0.031250 | True |
| selective_copy | gru_uniform_cache | 512 | 32 | 0.013021 | 0.039062 | 0.062500 | True |
| selective_copy | gru_uniform_cache | 1024 | 16 | 0.000000 | 0.000000 | 0.015625 | True |
| selective_copy | gru_uniform_cache | 1024 | 32 | 0.002604 | 0.013021 | 0.031250 | True |
| selective_copy | gru_reservoir_cache | 256 | 16 | 0.002604 | 0.132812 | 0.231445 | True |
| selective_copy | gru_reservoir_cache | 256 | 32 | 0.005208 | 0.257812 | 0.372070 | True |
| selective_copy | gru_reservoir_cache | 512 | 16 | 0.007812 | 0.085938 | 0.140869 | True |
| selective_copy | gru_reservoir_cache | 512 | 32 | 0.000000 | 0.140625 | 0.232422 | True |
| selective_copy | gru_reservoir_cache | 1024 | 16 | 0.002604 | 0.049479 | 0.081787 | True |
| selective_copy | gru_reservoir_cache | 1024 | 32 | 0.005208 | 0.096354 | 0.138794 | True |
| selective_copy | budgetmem_r | 256 | 16 | 0.005208 | 0.171875 | 0.192301 | True |
| selective_copy | budgetmem_r | 256 | 32 | 0.005208 | 0.442708 | 0.182129 | True |
| selective_copy | budgetmem_r | 512 | 16 | 0.002604 | 0.153646 | 0.111410 | True |
| selective_copy | budgetmem_r | 512 | 32 | 0.007812 | 0.335938 | 0.107544 | True |
| selective_copy | budgetmem_r | 1024 | 16 | 0.002604 | 0.093750 | 0.078044 | True |
| selective_copy | budgetmem_r | 1024 | 32 | 0.007812 | 0.268229 | 0.087077 | True |
| associative_recall | gru | 256 | 16 | 0.000000 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 256 | 32 | 0.020833 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 512 | 16 | 0.000000 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 512 | 32 | 0.000000 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 1024 | 16 | 0.020833 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 1024 | 32 | 0.000000 | 0.000000 | 0.000000 | True |
| associative_recall | gru_uniform_cache | 256 | 16 | 0.000000 | 0.083333 | 0.062500 | True |
| associative_recall | gru_uniform_cache | 256 | 32 | 0.000000 | 0.166667 | 0.125000 | True |
| associative_recall | gru_uniform_cache | 512 | 16 | 0.000000 | 0.000000 | 0.031250 | True |
| associative_recall | gru_uniform_cache | 512 | 32 | 0.000000 | 0.083333 | 0.062500 | True |
| associative_recall | gru_uniform_cache | 1024 | 16 | 0.020833 | 0.000000 | 0.015625 | True |
| associative_recall | gru_uniform_cache | 1024 | 32 | 0.000000 | 0.000000 | 0.031250 | True |
| associative_recall | gru_reservoir_cache | 256 | 16 | 0.000000 | 0.114583 | 0.238770 | True |
| associative_recall | gru_reservoir_cache | 256 | 32 | 0.000000 | 0.177083 | 0.393555 | True |
| associative_recall | gru_reservoir_cache | 512 | 16 | 0.000000 | 0.072917 | 0.140137 | True |
| associative_recall | gru_reservoir_cache | 512 | 32 | 0.000000 | 0.125000 | 0.242188 | True |
| associative_recall | gru_reservoir_cache | 1024 | 16 | 0.000000 | 0.062500 | 0.080200 | True |
| associative_recall | gru_reservoir_cache | 1024 | 32 | 0.000000 | 0.104167 | 0.142090 | True |
| associative_recall | budgetmem_r | 256 | 16 | 0.000000 | 0.168403 | 0.098063 | True |
| associative_recall | budgetmem_r | 256 | 32 | 0.000000 | 0.300347 | 0.142497 | True |
| associative_recall | budgetmem_r | 512 | 16 | 0.000000 | 0.156250 | 0.052083 | True |
| associative_recall | budgetmem_r | 512 | 32 | 0.000000 | 0.227431 | 0.076904 | True |
| associative_recall | budgetmem_r | 1024 | 16 | 0.000000 | 0.137153 | 0.030701 | True |
| associative_recall | budgetmem_r | 1024 | 32 | 0.000000 | 0.178819 | 0.041585 | True |
| distractor_heavy_retrieval | gru | 256 | 16 | 0.020833 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 256 | 32 | 0.000000 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 512 | 16 | 0.006944 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 512 | 32 | 0.013889 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 1024 | 16 | 0.000000 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 1024 | 32 | 0.013889 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 256 | 16 | 0.000000 | 0.046875 | 0.062500 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 256 | 32 | 0.000000 | 0.125000 | 0.125000 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 512 | 16 | 0.013889 | 0.019097 | 0.031250 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 512 | 32 | 0.000000 | 0.064236 | 0.062500 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 1024 | 16 | 0.006944 | 0.019097 | 0.015625 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 1024 | 32 | 0.006944 | 0.027778 | 0.031250 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 256 | 16 | 0.000000 | 0.086806 | 0.220215 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 256 | 32 | 0.000000 | 0.112847 | 0.377930 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 512 | 16 | 0.013889 | 0.022569 | 0.129150 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 512 | 32 | 0.013889 | 0.062500 | 0.228027 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 1024 | 16 | 0.006944 | 0.008681 | 0.076294 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 1024 | 32 | 0.013889 | 0.034722 | 0.134766 | True |
| distractor_heavy_retrieval | budgetmem_r | 256 | 16 | 0.000000 | 0.041667 | 0.214030 | True |
| distractor_heavy_retrieval | budgetmem_r | 256 | 32 | 0.006944 | 0.104167 | 0.181641 | True |
| distractor_heavy_retrieval | budgetmem_r | 512 | 16 | 0.000000 | 0.026042 | 0.172770 | True |
| distractor_heavy_retrieval | budgetmem_r | 512 | 32 | 0.000000 | 0.039931 | 0.146973 | True |
| distractor_heavy_retrieval | budgetmem_r | 1024 | 16 | 0.000000 | 0.012153 | 0.127909 | True |
| distractor_heavy_retrieval | budgetmem_r | 1024 | 32 | 0.013889 | 0.032986 | 0.134054 | True |
