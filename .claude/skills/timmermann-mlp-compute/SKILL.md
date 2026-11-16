---
name: timmermann-mlp-compute
description: Plan dense MLP training compute using the Timmermann–Vulicevic workflow. Use for model-size epoch schedules, fixed FLOP budgets, wall-clock estimates, or validation curves that test whether more training helps.
---

# Timmermann–Vulicevic MLP compute

Use the [paper](https://ssrn.com/abstract=6105327) for research context and the
[authors’ implementation](https://github.com/LukaVuli/Fin_Econ_Scaling_Laws) for
scaling-law methods. The schedule below is the recipe adopted in this repository.
Check the paper version and experiment settings before claiming exact replication.

## Calculate the budget

- Count the instantiated model’s trainable scalar parameters, including biases.
  For fully connected widths `[inputs, ...hidden, outputs]`, use
  `P = sum((in_width + 1) * out_width)` over biased layers. Use the actual
  trainable count for architectures with other components.
- Count eligible examples actually processed. Let `D` be examples per complete
  epoch, after filtering, batching and any dropped last batch. In a panel, count
  asset-date rows that reach the optimizer. Use each fold and arm’s own count.
- **Model-size schedule.** Use `E = max(floor(0.2 * P**0.8), 1) + 10`.
  Apply any requested multiplier after rounding and adding 10. For a fixed-epoch
  experiment run all scheduled epochs with early stopping disabled. This gives a
  reference schedule. Evidence for a useful high-compute regime comes from
  validation behavior across budgets.
- **Fixed FLOP budget.** For a dense MLP estimate `C ≈ 6 * P * examples_seen`,
  including repeats. One petaFLOP-day is `8.64e19` FLOPs. For a budget of
  `B` petaFLOP-days, use `floor(8.64e19 * B / (6 * P * D))` full epochs.
  Count actual batch examples and check the next update against the remaining
  budget when epoch sizes vary or partial epochs are allowed. If the budget
  cannot fund one full epoch, report zero full epochs. Use architecture-appropriate
  profiling for other model families or materially different compute patterns.

Example for the repository’s 29 → 64 → 32 → 1 MLP, using 39,000 examples per
epoch for illustration. `P = 4,033`, the reference is `163` epochs, and 10× is
`1,630` epochs (about `1.54e12` FLOPs). A cap of `0.001` petaFLOP-days gives
`91,552,385` full epochs under the same approximation.

## Estimate time and evaluate

- Benchmark steady-state seconds per epoch on the intended device, software,
  batch size, architecture, and representative fold size. Synchronize GPU
  timing. Multiply by epochs and divide by 3,600 for hours. Separate per-fit
  time from the full cost of search, folds, seeds, and refits. Distinguish
  summed compute from elapsed time under actual concurrency.
- For scaling experiments, record cumulative examples and estimated FLOPs at
  log-spaced checkpoints. Compare held-out loss or validation R² on consistent
  chronological splits at common FLOP budgets. Prespecify checkpoint selection.
  For the supplied recipe’s selection rule, choose the best five-epoch
  median-smoothed validation R² after the full run. Keep the outer test split
  for evaluation and charge all training used to select the checkpoint.
- If asked for a limit to predictability, fit loss against a range of observed
  compute levels. For example, use `loss(C) = L_inf + A * C**(-alpha)` with
  positive `A` and `alpha`. Report the uncertainty and sensitivity to the fit
  range. A single training run or a weakly identified asymptote supports only
  a description of the observed compute range.
