## Reading

**The recipe.** This is the from-scratch torch MLP of E104/E105: identity 64-32-16, sklearn's initial weights, batches of 3 whole days, Adam at a constant 1e-3. It has no regularisation: alpha 0, no early stopping (500 epochs, `--no-stop`), no dropout and no annealing. It is trained on the **old loss**, 0.5 × MSE with each day weighted equally.

The layout and seeds are the same as the first report and the −CORR report: crypto/v2.0-fenccausal, every refit of the full walk-forward (12 blocks, 2020-07 → 2026-07, 1,531 held-out days), seed 42 plus seed 43 as the churn floor. It starts from exactly the same initial weights as the −CORR run. All 24 runs finished.

(A first launch of this run died at epochs 252–397 on 2026-10-17 when the /home disk quota filled. This is a clean rerun from epoch 0.)

**In one line: on the old loss this linear network ends level with the ridge, and with the −CORR network. The first report's failure came from the relu recipe, not from MSE.**

- **P1: slow to settle, then level with the ridge.**
  - Every row is well below the ridge for the first 10–30 epochs.
  - Rows turn neutral around epoch 100 and sit level with the ridge or slightly above it to epoch 500.
  - There is no late darkening, so no overfitting.
- **P2: the peak comes late, and the stop rule fires before it.**
  - Best epochs have a median of 52 [IQR 29, 89], several are flat-topped, and seed 43 disagrees on most folds.
  - The stop rule fires at epochs 16–39, before the best epoch in 11 of 12 folds.
  - That is the reverse of the relu run, where it fired 60–100 epochs too late.
- **P3: above the ridge in 8 of 12 folds.**
  - At epoch 500 the gap is median +0.0046 [IQR −0.0011, +0.0070].
  - At the stop-rule epoch it would have been −0.020: here early stopping costs CORR.
- **P4: nothing to screen.** The gradient falls by about two orders of magnitude over training and no fold is flagged.
- **P5: refit churn is all data.**
  - Churn between consecutive refits at epoch 500 is median 0.447, the same as the ridge's 0.463.
  - Two seeds churn 0.016; their CORR at epoch 500 differs by a median 0.003.

## The fold panels

The tool picked r00 (the first fold, which also has the worst gap, −0.018) and r08 (the median). Both are the *learning* row of the pairing table. It converges slowly, without overfitting.

- **1a / 1b: the loss is done long before the ranking is.**
  - Loss drops onto the "predict the mean" line within a few epochs. The sklearn initial weights are too large for this target, so the first job is shrinking the network's overall linear map, from 1.5 to about 0.14 by epoch ~150.
  - After that the fit beyond the mean is only +2–3% on train and about 0 to +2% held-out, too small for the stop rule's tolerance. So it fires around epoch 22 while the ranking is still forming.
  - This is the "MSE is near-blind to rank" trap again, seen from the other side: there the loss kept improving after the ranking had stopped; here the ranking keeps improving after the loss has stopped.
- **2 / 4c: the ranking churns early, then settles.**
  - Held-out CORR is noisy for the first ~100 epochs while the ranking reshuffles 40–90% per epoch.
  - The ranking only settles, with movement below 0.1 per epoch, from epoch ~190 (median; range 139–310).
  - Train CORR ends at 0.19, so the network does not memorise.
- **3b, 5:** gradients fall steadily in every layer, and the second and fourth layers shrink slowly. The net map ends near 0.14, against about 54 under the scale-blind −CORR loss.
- **4a:** at epoch 500 the target-bin profile is 0.34 / 0.44 / 0.54 / 0.53 / 0.44, the ridge's ranking again, bottom bin included.

## Scores

**1,531 days, CORR (paired t against this MSE network):**

| model | CORR | dCORR | t |
|---|---|---|---|
| from-scratch MSE (this run) | 0.1134 | — | — |
| from-scratch −CORR | 0.1145 | +0.0011 | +1.1 |
| ridge, same windows | 0.1109 | −0.0025 | −1.6 |
| relu MSE (`torch_mlp`, first report) | 0.0460 | −0.0674 | −8.0 |

**The 547 days with the meta model:**

| model | MMC | payout | dd(pay) | calmar | dpay | t |
|---|---|---|---|---|---|---|
| from-scratch MSE (this run) | 0.070 | 0.084 | 1.33 | 0.063 | — | — |
| from-scratch −CORR | 0.070 | 0.085 | 1.51 | 0.056 | +0.0004 | +0.3 |
| ridge | 0.069 | 0.083 | 1.80 | 0.046 | −0.0016 | −0.6 |
| relu MSE | 0.043 | 0.050 | 1.88 | 0.026 | −0.0345 | −2.5 |

## What it means

- **The first report's collapse came from the `torch_mlp` recipe, not from MSE.** That recipe is relu 64-32 on shuffled row batches, and relu lets it memorise a 130-day window (train CORR 0.79 against 0.19 here). Several settings differ between the two recipes (activation, depth, batching, initialisation), but relu is the one that gives the network the capacity to memorise.
- **For this linear network the loss barely matters.**
  - Both losses end at the ridge's ranking and level with each other on CORR, MMC and payout.
  - MSE draws down less on the meta window: calmar 0.063 against 0.056.
  - The −CORR loss gets there in about 5 epochs instead of about 190, and is not misled by the stop rule.
  - Under MSE the stop rule, which reads the loss, fires about 30 epochs in and would cost 0.014 CORR. Under −CORR it is harmless.
- **Neither network clears the ridge decisively.** The −CORR network's +0.0036 over the ridge is the only gap past |t| ≥ 2 (t +2.9, in its own report). This run's +0.0025 is t +1.6.

**Caveats.** Two seeds set P5's floor but do not rank recipes. A per-fold gap rests on 130 days of an overlapping 20-day target. The relu-vs-identity comparison changes several settings at once.
