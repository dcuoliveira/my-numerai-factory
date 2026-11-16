## Reading

**The recipe.** This is TorchMLPRegressor's default network: relu 64-32, Adam at 1e-3, batches of 1,024 rows. It was trained on `crypto/v2.0-fenccausal` (44 features) with no regularisation: alpha 0, no early stopping (500 epochs, `--no-stop`) and no dropout (the network has none). It ran on every refit of the full walk-forward (12 blocks, 2020-07 → 2026-07, 1,531 held-out days) with seed 42, plus seed 43 as the churn floor, on deepfive's GPUs. All 24 runs finished and nothing is left out.

**In one line: every refit overfits the same way and early, and the procedure never beats a ridge fitted on the same window.**

- **P1: a peak location exists, and it is early.** Every row comes closest to the ridge between roughly epochs 5 and 30, then darkens steadily. No fold improves late and none sign-flips: held-out CORR is still positive at epoch 500 in all 12 folds (0.0004 to 0.121).
- **P2: the best epoch is a rule within one seed, but a coin flip across seeds.**
  - Seed 42's best epochs have a median of 19 [IQR 12, 26], and 10 of 12 fall between epochs 6 and 30.
  - Seed 43 agrees on some folds and not others: fold 3 peaks at 10 vs 107, fold 7 at 2 vs 37, fold 8 at 15 vs 165.
  - Three folds are flat-topped: they stay within 0.005 of their best over epochs 10–197, 13–383 and 9–30.
  - The stop rule fires at epochs 84–116, after the best epoch in every fold, and gives up a median 0.027 CORR against that best.
  - Held-out MSE turns even before the ranking does: its minimum falls at epochs 6–13.
- **P3: below the ridge in 12 of 12 folds.**
  - At the kept epoch (500) the gap is median −0.057 [IQR −0.087, −0.052]. At the stop-rule epoch it is −0.062 [−0.073, −0.046].
  - Even at its best epoch the network beats the ridge in only 2 of 12 folds (r03 and r09) with seed 42, and 1 of 12 with seed 43. The best epoch is picked on the scored block, so this is an upper bound.
  - The ridge here is near-OLS (alpha 1). Alpha 100 would score about 0.004 higher on average, which would widen the gap, not close it.
- **P4: nothing to screen.** The probe-batch gradient norm stays in one band across all 12 folds and no fold is flagged.
- **P5: the refit swap nearly redraws the ranking.**
  - At epoch 500, churn between consecutive refits on the same days is median 0.887 [0.880, 0.913]: within a day, their rankings correlate at about 0.11.
  - Two seeds of the same refit churn 0.672 [0.633, 0.689]. The ridge's own refit churn is 0.463 [0.372, 0.563].
  - So reseeding alone moves this network's ranking more than a ridge refit does, and a refit moves it further still.
  - All refits share seed 42's initial weights, so cross-fold churn starts near 0 and grows with training: 0.63 at epoch 10, 0.82 at 50, 0.89 at 500. It crosses the seed floor at about epoch 10.
  - The seed floor bottoms at 0.56–0.57 around epochs 20–50 and rises again as the network overfits. Training longer buys turnover, not skill.

## The three fold panels

The tool picked r00 (the first fold), r06 (the median gap) and r10 (the worst gap, −0.109). All three fall in the same row of the pairing table, **overfitting**:

- **1a / 1b, loss.** Train loss falls throughout, and its fit beyond the mean reaches +65–70%. Held-out loss bottoms at epochs 7–13 and climbs for the rest of the run, ending 80–150% worse than predicting the mean.
- **2, metric.** Train CORR reaches 0.80–0.82, so the network memorises its 130-day window. Held-out CORR peaks inside the first ~30 epochs and then decays.
- **3a / 3b, gradients.** There is no penalty part (alpha 0). The data gradient is moderate (0.05–0.2), noisy, and drifts up slowly with no trend. These are healthy gradients during overfitting, so there is nothing to fix in the optimiser.
- **5, rate and weights.** The rate is constant. The first two layers' norms grow from about 7 to about 10 over 500 epochs, which is the growth a penalty would have held back.
- **4b / 4c, prediction dynamics.** There is no collapse. The within-day spread is about 0.2, roughly 15 orders of magnitude above float64 spacing, and every stock gets a distinct value. At epoch 500 the ranking still moves 1–3% per epoch without improving.
- **4a, where the lost CORR goes.** The network ranks the bottom target bin low and the middle bins high, and the top bin sits below the middle in 12 of 12 folds at epoch 500.
  - That hump comes from this target and feature set, not from the network. The ridge on the same blocks has it too, more sharply: its median profile is 0.34 / 0.44 / 0.54 / 0.53 / 0.45, against the network's 0.41 / 0.46 / 0.52 / 0.52 / 0.48 at its best epoch.
  - What the network lacks is the ridge's separation of the bottom bin, and overfitting erodes what little it has. The bottom bin's percentile rises from 0.41 at the best epoch to 0.43 at epoch 500 (median over folds).

## Scores

- **All 12 blocks, epoch 500** (n = 1,531 days): CORR is 0.046 against the ridge's 0.111, a paired difference of −0.065 (Newey-West t −7.6).
- **All 12 blocks, stop-rule epoch:** CORR is 0.055 (t −7.0 against the ridge).
- **The 547 days the meta model covers** (it starts on 2024-05-07: the last 37 days of fold 7 and all of folds 8–11), at epoch 500. That is 35.7% of the 1,531 held-out days and 164,100 of 459,300 rows; every covered day is covered for all its stocks.
  - payout 0.050 against the ridge's 0.083 (dpay −0.033, t −2.2)
  - MMC 0.043 against 0.069
  - calmar 0.026 against 0.046, on the same window

## What it means

Removing every regulariser costs more than a few wasted epochs. The procedure overfits in every refit, and the stop rule cannot see it because it reads the training loss. The function shipped at epoch 500 changes almost completely at every refit, and as a ranking it stays well below a linear model fitted on the same 130 days. Three things would show a fix working:

- P3's gap closing
- P5's cross-fold churn falling toward the ridge's 0.46
- 4a's bottom bin moving back down toward the ridge's 0.34

**Caveats.** Two seeds are enough to set P5's floor but not to rank recipes. A single fold's gap rests on 130 days of an overlapping 20-day target, so read the median and the count rather than one fold. Nothing here was selected on the held-out blocks except the "best epoch" columns, which are labelled as such.
