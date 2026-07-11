# Dataset Protocol

## Primary datasets

1. Six deterministic synthetic sequence-memory tasks.
2. LogHub HDFS_v1 block-level anomaly detection.
3. Stanford Large Movie Review Dataset binary sentiment classification.

## Deferred datasets

- Sequence reversal and nested parentheses remain disabled until the six-task pipeline is stable.
- BGL is reserved for external validation after HDFS and IMDb experiments are complete.

## Leakage controls

- Synthetic train, validation, and test partitions use separate split seeds.
- HDFS partitioning occurs only at the block-ID level. A block cannot appear in multiple partitions.
- IMDb uses a fixed stratified 90/10 split of the official training data. The official test set remains locked until final evaluation.

## HDFS metrics

F1, precision, recall, area under the precision-recall curve, area under the ROC curve, false-positive rate, peak memory, latency per block, and event throughput.

## IMDb metrics

Accuracy, macro F1, negative log-likelihood, expected calibration error, latency, throughput, and peak GPU memory.
