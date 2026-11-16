## Reading

**The recipe.** This is the from-scratch torch MLP of E104/E105: identity 64-32-16, sklearn's initial weights, batches of 3 whole days, Adam at a constant 1e-3. It has no regularisation: alpha 0, no early stopping (500 epochs, `--no-stop`), no dropout and no learning-rate annealing.

**The loss** is minus the mean per-day contribution over the meta model. That is Numerai's own MMC formula, with one change: the prediction is z-scored within the day instead of rank-gaussianised, because ranking has no gradient. The meta model and the target are treated exactly as the scorer does. Day by day it tracks the repo's MMC scorer at r > 0.95.

**The data: meta-model days only.** The loss needs the meta model on every training row, and it exists from 2024-05-07, so only those 547 days are used (`--meta-only`). They are cut into 12 equal folds (`--scheme even`). Each trains on the 130 preceding meta-model days and predicts the next 33–34, after the usual 20-day embargo. That gives 397 held-out days, 2024-12-31 → 2026-07-08. It ran with seed 42, plus seed 43 as the churn floor. All 24 runs finished.

**In one line: it learns MMC on its training window but keeps little of it out of sample, and by epoch 500 it is below a ridge fitted on the same folds.**

- **P1: the network loses MMC as it trains.** In the first few epochs most rows are level with the ridge or above it. From about epoch 20–50 on, folds 0, 2, 3, 5 and 6 turn and stay red. Folds 1, 8 and 10 stay blue.
- **P2: the peak is in the first few epochs.**
  - The best held-out MMC comes at epoch 3 or earlier in 6 of 12 folds. In three of them it is at epoch 0, so the untrained network scored best.
  - Two folds have late, flat tops (folds 4 and 5).
  - The median fold's held-out MMC averages 0.038 over epochs 1–10 and falls to 0.017 by epoch 500.
- **P3: below the ridge in 8 of 12 folds.**
  - At epoch 500 the gap is median −0.0067 [IQR −0.0167, +0.0051].
  - At the stop-rule epoch (32–103) it is −0.0011 [−0.0085, +0.0100], about level.
  - Unlike the corr-loss network, here the stop rule matters.
- **P4: three folds are screened, and none is an optimiser failure.**
  - **r01:** the first layer's gradient doubles over 150 epochs as the layers align, then levels off. It is also the best fold: held-out MMC is about 0.11 throughout.
  - **r06:** the gradient sits about 3× the median, with no trend.
  - **r09:** the gradient sits about 0.4× the median.
  - None climbs while the training loss worsens, so all three are differences of level, not divergence.
- **P5: the network moves about twice as much as the ridge at a refit, and the seed is irrelevant.**
  - Refit churn at epoch 500 is median 0.22 [0.14, 0.29], against the ridge's 0.12 [0.09, 0.14].
  - Two seeds churn 0.004, so here too the function is pinned down by the data.
  - These folds sit only 33 days apart and their 130-day training windows overlap by about three quarters, so churn here is lower than in the 130-day layouts and is not comparable with them.

## The fold panels

The tool picked r00 (the first fold and also the worst gap, −0.055) and r11 (the median), plus the three folds P4 flagged (r01, r06, r09).

- **r00, r11: overfitting, without memorisation.**
  - The training contribution rises to 0.11–0.17 within about 10 epochs and holds there.
  - Held-out MMC peaks inside those same 10 epochs (r00: about 0.05–0.08), then decays to about zero by epoch 100.
  - Train CORR stays near 0.15–0.22, so this is not memorisation like the relu MSE network's. What the loss fits is the part of the target the meta model does not explain, and on a 130-day window much of that is specific to the window.
- **r06: never generalises.** Held-out MMC is negative from the first epoch and settles near −0.08, while the training contribution reaches 0.17. Whatever was MMC in March–September 2025 reversed in October–November.
- **r09: overfits slowly.** Held-out MMC starts near 0 and drifts to −0.03 over the run.
- **r01: the case where it works.** Held-out MMC is about 0.12 by epoch 5 and erodes only slightly, to 0.105.
- **Across all of them, 3b and 5:** as in the corr-loss run, the deeper layers' gradients decay by two orders of magnitude, and the composed linear map grows from about 1.5 to about 50 while each layer's own norm barely moves.
- **Train MMC at epoch 500:** median 0.16 (0.11–0.19), against a held-out median of 0.017.

## Scores, same 397 days

| | CORR | MMC | payout | dd(pay) | calmar |
|---|---|---|---|---|---|
| MMC-loss network, epoch 500 | 0.067 | 0.023 | 0.030 | 2.85 | 0.011 |
| MMC-loss network, stop-rule epoch | 0.081 | 0.032 | 0.040 | 2.74 | 0.015 |
| ridge, same meta-only folds | 0.107 | 0.033 | 0.044 | 1.47 | 0.030 |
| corr-loss network, epoch 500 (full-layout folds) | 0.156 | 0.072 | 0.087 | 1.28 | 0.068 |
| ridge, full-layout folds | 0.149 | 0.069 | 0.084 | 1.25 | 0.067 |

Against its own ridge at epoch 500 the network gives up 0.010 MMC and 0.014 payout (t −2.3). The stop rule recovers it to the ridge's level (dpay −0.004, t −0.7).

**The layout costs more than the loss gains.** On these same days the ridge itself halves its MMC on the meta-only folds: 0.033, against 0.069 on the full-layout folds. Where the two layouts' training windows nearly coincide, the two ridges agree (within-day rank correlation 0.96 for meta fold 1 against full fold 9). Elsewhere, the meta-only ridge, which refits every ~33 days on the freshest 130 days, simply scored worse on these 397 days.

So the corr-loss network's 3× higher MMC is mostly the layout, not the loss. The like-for-like reading is each network against the ridge of its own folds: the MMC loss gives up 0.010 MMC, while the corr loss gains 0.003.

## What it means

Training on the MMC formula directly does not buy MMC out of sample here.

- The loss fits the meta-model-orthogonal part of the target within a few epochs, train MMC 0.16, and most of that fails to carry to the next month.
- Without regularisation it keeps fitting for 500 epochs and ends below a linear ridge.
- With the stop rule it only draws level with that ridge.
- On the same days, the corr-loss network trained on the full history earns three times the MMC. That is mostly because its layout trains on longer, older history; under like-for-like ridges the corr loss still edges the MMC loss.

**Caveats.**

- **Small blocks:** the folds are 33 days, so per-fold numbers are noisy. Read the medians and the stitched 397-day table.
- **Two seeds** set P5's floor but do not rank recipes.
- **The comparison with the corr-loss network crosses two layouts.** A corr-loss run on these same meta-only folds would separate the loss from the layout cleanly.
