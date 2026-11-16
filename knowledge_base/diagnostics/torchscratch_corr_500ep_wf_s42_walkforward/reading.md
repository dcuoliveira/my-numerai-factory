## Reading

**The recipe.** This is the from-scratch torch MLP of E104/E105: identity 64-32-16, sklearn's initial weights, batches of 3 whole days, Adam at a constant 1e-3. It has no regularisation: alpha 0, no early stopping (500 epochs, `--no-stop`), no dropout (the network has none) and no learning-rate annealing.

**The loss** is minus the mean per-day Pearson correlation between prediction and target over the batch's days. It is the differentiable stand-in for CORR: Numerai's CORR ranks the predictions first, and that step has no gradient.

It ran on `crypto/v2.0-fenccausal` (44 features) on every refit of the full walk-forward (12 blocks, 2020-07 → 2026-07, 1,531 held-out days), with seed 42 plus seed 43 as the churn floor. All 24 runs finished.

**In one line: the network matches the ridge's ranking within a few epochs and edges past it in 11 of 12 folds. It never overfits, and two seeds give the same function.**

- **P1: no peak to find.** Every row sits within about ±0.02 of the ridge from the first few epochs to epoch 500, with no darkening late. The two most recent folds are above the ridge throughout.
- **P2: the best epoch is noise, because the curve is flat.**
  - Best epochs have a median of 12 [IQR 2, 30], but six folds stay within 0.005 of their best over spans like 4–363, 4–464 or 2–500 epochs.
  - Seed 43 disagrees freely (fold 2: 185 vs 477; fold 7: 0 vs 120).
  - So the stop rule (epochs 29–116) is harmless: the median gap is +0.0043 at the stop rule against +0.0039 at epoch 500.
- **P3: above the ridge in 11 of 12 folds.**
  - At epoch 500 the gap is median +0.0039 [IQR +0.0017, +0.0053]. The one exception is fold 5 (−0.0024).
  - Raw held-out CORR at epoch 500 is median 0.114, against the MSE network's 0.043 on the same folds.
- **P4: nothing to screen.** The probe-batch gradient norm drifts down slowly in one band, and no fold is flagged.
- **P5: the refit churn is all data, with no instability.**
  - Churn between consecutive refits at epoch 500 is median 0.471 [0.344, 0.509], the same as the ridge's own refit churn (0.463).
  - Two seeds of one refit churn 0.005 [0.003, 0.008]: their within-day rankings correlate at about 0.995.
  - Cross-fold churn sits around 0.4–0.47 from epoch 1 to epoch 500, so training longer adds no turnover.
  - Compare the MSE network: 0.89 across refits and 0.67 between seeds.

## The three fold panels

The tool picked r00 (the first fold, gap +0.002), r08 (the median, +0.003) and r05 (the worst, −0.002). All three are the plain *learning* row of the pairing table, converged early. None shows a failure mode.

- **1a / 2, loss and metric.** The train loss falls to −0.15 to −0.20 within about 5 epochs and creeps down slowly after. Held-out reaches its level just as fast, then moves sideways for 495 epochs with no trend. Epoch-to-epoch noise is about ±0.01 (±0.02 in the short first fold).
- **Train CORR** stays at 0.15–0.20. A linear map cannot memorise a 130-day window the way the relu MSE network did (train CORR 0.80).
- **1b, surrogate vs paid metric.** Held-out Pearson sits a steady 0.008 above Numerai CORR (median; IQR 0.005–0.010). The gap between the optimised surrogate and what pays is small and does not grow.
- **3b, gradient by layer.** The first layer's gradient holds at 0.15–0.2 while the deeper layers' gradients decay by two orders of magnitude.
- **5, weights.** Each layer's norm barely moves, yet the single linear map the stack composes to grows from 1.5 to about 54 (median at epoch 500) as the layers align. The loss cannot see scale, so nothing holds this back. It changes no ranking (late-run movement 0.007 per epoch, all 300 stocks distinct), but it is the one thing a penalty would have stopped.
- **4a, where the CORR comes from.** At epoch 500 the target-bin profile is 0.34 / 0.44 / 0.54 / 0.53 / 0.44, against the ridge's 0.34 / 0.44 / 0.54 / 0.53 / 0.45. The network has found the ridge's ranking, bottom bin included, which is exactly what the MSE network never separated (0.41–0.43).

## Scores

- **All 12 blocks, epoch 500** (n = 1,531 days): CORR 0.1145 against the ridge's 0.1109, a paired difference of +0.0036 (Newey-West t +2.9, clearing the house |t| ≥ 2 bar). The stop-rule epoch gives the same 0.1143.
- **The 547 days the meta model covers**, at epoch 500:
  - CORR 0.145 against 0.139 (t +2.5)
  - MMC 0.070 against 0.069
  - payout 0.085 against 0.083 (dpay +0.002, t +1.0: **not** significant)
  - calmar 0.056 against 0.046, with dd(pay) 1.51 against 1.80

## What it means

With a per-day correlation loss, this network behaves like a ridge that optimises the right quantity: within-day correlation rather than pooled squared error. It gets there in a few epochs, stays there for 500 with no regularisation at all, gives the same answer for either seed, and moves at a refit only as much as the data makes a ridge move.

The CORR edge over the ridge is small but consistent: 11 of 12 folds, t +2.9. One possible reason, untested here: the loss centres and weights each day separately, which a pooled ridge does not. The payout edge is not established.

**Caveats.**

- **Why it doesn't overfit:** the activations are identity, so the network is linear and cannot memorise. A relu version would test whether the loss alone keeps the procedure stable.
- **Seeds:** two are enough to set P5's floor but not to rank recipes.
- **Per-fold noise:** a single fold's gap rests on 130 days of an overlapping 20-day target, so read the median and the count.
