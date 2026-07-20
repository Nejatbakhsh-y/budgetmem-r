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
| budgetmem_r | 0.007017 |
| gru_uniform_cache | 0.004702 |
| gru_reservoir_cache | 0.003328 |

## Training Records

| Task | Model | First loss | Final loss | Stable | Resume |
|---|---|---:|---:|---:|---:|
| selective_copy | gru | 5.262817 | 4.827681 | True | True |
| selective_copy | gru_uniform_cache | 5.256142 | 4.896593 | True | True |
| selective_copy | gru_reservoir_cache | 5.260373 | 4.882675 | True | True |
| selective_copy | budgetmem_r | 5.403127 | 4.408298 | False | True |
| associative_recall | gru | 5.265120 | 4.529042 | True | True |
| associative_recall | gru_uniform_cache | 5.251778 | 4.646859 | True | True |
| associative_recall | gru_reservoir_cache | 5.245639 | 4.584754 | True | True |
| associative_recall | budgetmem_r | 5.395580 | 4.800890 | False | True |
| distractor_heavy_retrieval | gru | 5.259140 | 4.379528 | True | True |
| distractor_heavy_retrieval | gru_uniform_cache | 5.256606 | 4.609639 | True | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 5.259079 | 4.632347 | True | True |
| distractor_heavy_retrieval | budgetmem_r | 5.424952 | 4.341696 | False | True |

## Evaluation Matrix

| Task | Model | Length | Budget | Token accuracy | Memory recall | Write frequency | Budget pass |
|---|---|---:|---:|---:|---:|---:|---:|
| selective_copy | gru | 256 | 16 | 0.006510 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 256 | 32 | 0.002604 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 512 | 16 | 0.006510 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 512 | 32 | 0.009115 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 1024 | 16 | 0.006510 | 0.000000 | 0.000000 | True |
| selective_copy | gru | 1024 | 32 | 0.002604 | 0.000000 | 0.000000 | True |
| selective_copy | gru_uniform_cache | 256 | 16 | 0.006510 | 0.042969 | 0.062500 | True |
| selective_copy | gru_uniform_cache | 256 | 32 | 0.011719 | 0.110677 | 0.125000 | True |
| selective_copy | gru_uniform_cache | 512 | 16 | 0.002604 | 0.002604 | 0.031250 | True |
| selective_copy | gru_uniform_cache | 512 | 32 | 0.005208 | 0.035156 | 0.062500 | True |
| selective_copy | gru_uniform_cache | 1024 | 16 | 0.005208 | 0.000000 | 0.015625 | True |
| selective_copy | gru_uniform_cache | 1024 | 32 | 0.009115 | 0.009115 | 0.031250 | True |
| selective_copy | gru_reservoir_cache | 256 | 16 | 0.002604 | 0.148438 | 0.231445 | True |
| selective_copy | gru_reservoir_cache | 256 | 32 | 0.007812 | 0.255208 | 0.372070 | True |
| selective_copy | gru_reservoir_cache | 512 | 16 | 0.007812 | 0.084635 | 0.140869 | True |
| selective_copy | gru_reservoir_cache | 512 | 32 | 0.001302 | 0.148438 | 0.232422 | True |
| selective_copy | gru_reservoir_cache | 1024 | 16 | 0.003906 | 0.049479 | 0.081787 | True |
| selective_copy | gru_reservoir_cache | 1024 | 32 | 0.009115 | 0.108073 | 0.138794 | True |
| selective_copy | budgetmem_r | 256 | 16 | 0.005208 | 0.417969 | 0.099325 | True |
| selective_copy | budgetmem_r | 256 | 32 | 0.003906 | 0.527344 | 0.106608 | True |
| selective_copy | budgetmem_r | 512 | 16 | 0.011719 | 0.437500 | 0.061096 | True |
| selective_copy | budgetmem_r | 512 | 32 | 0.009115 | 0.472656 | 0.061361 | True |
| selective_copy | budgetmem_r | 1024 | 16 | 0.009115 | 0.398438 | 0.046031 | True |
| selective_copy | budgetmem_r | 1024 | 32 | 0.005208 | 0.503906 | 0.051483 | True |
| associative_recall | gru | 256 | 16 | 0.010417 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 256 | 32 | 0.000000 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 512 | 16 | 0.020833 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 512 | 32 | 0.000000 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 1024 | 16 | 0.010417 | 0.000000 | 0.000000 | True |
| associative_recall | gru | 1024 | 32 | 0.010417 | 0.000000 | 0.000000 | True |
| associative_recall | gru_uniform_cache | 256 | 16 | 0.010417 | 0.083333 | 0.062500 | True |
| associative_recall | gru_uniform_cache | 256 | 32 | 0.020833 | 0.166667 | 0.125000 | True |
| associative_recall | gru_uniform_cache | 512 | 16 | 0.010417 | 0.000000 | 0.031250 | True |
| associative_recall | gru_uniform_cache | 512 | 32 | 0.000000 | 0.083333 | 0.062500 | True |
| associative_recall | gru_uniform_cache | 1024 | 16 | 0.010417 | 0.000000 | 0.015625 | True |
| associative_recall | gru_uniform_cache | 1024 | 32 | 0.000000 | 0.000000 | 0.031250 | True |
| associative_recall | gru_reservoir_cache | 256 | 16 | 0.000000 | 0.114583 | 0.238770 | True |
| associative_recall | gru_reservoir_cache | 256 | 32 | 0.000000 | 0.177083 | 0.393555 | True |
| associative_recall | gru_reservoir_cache | 512 | 16 | 0.000000 | 0.072917 | 0.140137 | True |
| associative_recall | gru_reservoir_cache | 512 | 32 | 0.000000 | 0.125000 | 0.242188 | True |
| associative_recall | gru_reservoir_cache | 1024 | 16 | 0.000000 | 0.062500 | 0.080200 | True |
| associative_recall | gru_reservoir_cache | 1024 | 32 | 0.000000 | 0.104167 | 0.142090 | True |
| associative_recall | budgetmem_r | 256 | 16 | 0.010417 | 0.000000 | 0.000244 | True |
| associative_recall | budgetmem_r | 256 | 32 | 0.000000 | 0.000000 | 0.000163 | True |
| associative_recall | budgetmem_r | 512 | 16 | 0.000000 | 0.000000 | 0.000224 | True |
| associative_recall | budgetmem_r | 512 | 32 | 0.000000 | 0.000000 | 0.000142 | True |
| associative_recall | budgetmem_r | 1024 | 16 | 0.010417 | 0.000000 | 0.000254 | True |
| associative_recall | budgetmem_r | 1024 | 32 | 0.010417 | 0.000000 | 0.000183 | True |
| distractor_heavy_retrieval | gru | 256 | 16 | 0.000000 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 256 | 32 | 0.003472 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 512 | 16 | 0.003472 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 512 | 32 | 0.006944 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 1024 | 16 | 0.003472 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru | 1024 | 32 | 0.006944 | 0.000000 | 0.000000 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 256 | 16 | 0.003472 | 0.046007 | 0.062500 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 256 | 32 | 0.020833 | 0.136285 | 0.125000 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 512 | 16 | 0.006944 | 0.031250 | 0.031250 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 512 | 32 | 0.006944 | 0.071181 | 0.062500 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 1024 | 16 | 0.000000 | 0.019965 | 0.015625 | True |
| distractor_heavy_retrieval | gru_uniform_cache | 1024 | 32 | 0.003472 | 0.032986 | 0.031250 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 256 | 16 | 0.010417 | 0.082465 | 0.220215 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 256 | 32 | 0.003472 | 0.118924 | 0.377930 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 512 | 16 | 0.006944 | 0.030382 | 0.129150 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 512 | 32 | 0.013889 | 0.060764 | 0.228027 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 1024 | 16 | 0.003472 | 0.011285 | 0.076294 | True |
| distractor_heavy_retrieval | gru_reservoir_cache | 1024 | 32 | 0.003472 | 0.037326 | 0.134766 | True |
| distractor_heavy_retrieval | budgetmem_r | 256 | 16 | 0.010417 | 0.060764 | 0.123454 | True |
| distractor_heavy_retrieval | budgetmem_r | 256 | 32 | 0.006944 | 0.105903 | 0.146484 | True |
| distractor_heavy_retrieval | budgetmem_r | 512 | 16 | 0.006944 | 0.030382 | 0.096334 | True |
| distractor_heavy_retrieval | budgetmem_r | 512 | 32 | 0.010417 | 0.055556 | 0.107076 | True |
| distractor_heavy_retrieval | budgetmem_r | 1024 | 16 | 0.003472 | 0.018229 | 0.088949 | True |
| distractor_heavy_retrieval | budgetmem_r | 1024 | 32 | 0.003472 | 0.019965 | 0.072032 | True |
