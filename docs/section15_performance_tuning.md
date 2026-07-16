# Section 15 BudgetMem-R Performance Tuning

The initial pilot and controller-repair pilot are retained as valid NO_GO
evidence. This tuning stage does not overwrite those artifacts.

The tuner screens three CPU-conscious candidates:

1. Increased optimization exposure.
2. Longer training with a smaller learning rate.
3. Increased recurrent and retrieval capacity.

Every candidate is trained only for BudgetMem-R and evaluated at sequence
length 1024 under budgets 16 and 32. Candidate results are compared with the
already-completed uniform-cache and reservoir-cache long-range baselines.

A candidate must pass stability, checkpoint-resumption, strict-budget,
resource-measurement, write-frequency, and retention-over-random checks. It
must also obtain at least a 0.01 mean accuracy gain over both deterministic
policies before the automation spends CPU time on a complete four-model
pilot. The final Section 15 GO rule remains unchanged at a 0.02 clear gain.
