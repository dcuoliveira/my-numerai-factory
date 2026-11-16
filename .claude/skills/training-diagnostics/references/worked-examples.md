# Worked examples: the panel on the networks diagnosed so far

Numbers that are normal for this data, and what each panel turned out to say. The five
single-run examples are refit 0 of the shared protocol — crypto/v2.0-fenccausal (44 features), train
2023-11-30→2024-05-29 (130 days, 39,000 rows), held-out 2024-07-25→2025-01-22 (130
days), target_binned_return_20 — trained past the stop rule to 120 epochs so the
panel shows both sides of it. Figures: `results/diag/crypto/<tag>/diagnostics.png`;
each took ~8 minutes with CORR and MMC scored every epoch.

## Reference points

| Quantity | Value |
|---|---|
| Loss of predicting the training mean (`0.5·var(y)`) | 0.0250, train and held-out |
| Held-out CORR of an UNTRAINED network (sklearn init, identity 64-32-16) | mean +0.001, sd 0.074 over seeds 42–141 (seed 42 +0.056, seed 47 −0.093) |
| Loss at initialisation, α=100 | 6.9 (seed 42) to 9.3 (seed 47); penalty 84% / 62% of it |
| Loss at initialisation, relu α=0.01, torch init | 0.084 train / 0.097 held-out; penalty 0.7% |
| Gradient norm at initialisation | 7.9 (seed 42) and 22.9 (seed 47), identity sklearn init; 0.26 relu torch init |
| float64 spacing at the predictions' level (~0.5) | 5.6e-17 |
| Held-out CORR / MMC of a working run on this block | 0.13–0.16 / 0.07–0.09 |
| Held-out fit beyond the mean (1b) of every α=100 run | within ±1.3% of zero from epoch 20 on (from epoch 10 at a constant rate) |

The last line is the first thing to know about this data. At α=100 the held-out
data loss never beats predicting the mean by more than about 1%, yet the same models rank
at CORR 0.13–0.16. MSE is close to blind to what pays here: never judge one of these
networks on its loss, or on 1b, alone.

The untrained network is not neutral either. A random linear map already ranks at
CORR ±0.07 on this block, so plot 2's epoch-0 point carries information (see the
seed-47 example).

## identity 64-32-16, α=100, constant 1e-3, 3-day batches, seed 42 (`ref_e105_const_s42`)

The E105 baseline on a seed where it "works".

| Epoch | Train loss | Penalty share | Held-out 1b | Held-out CORR | Held-out MMC | Gradient data / penalty | Net map \|W\| | Spread | Distinct of 300 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.855 | 84% | −4450% | +0.056 | +0.039 | 7.81 / 1.06 | 1.5 | 1.3 | 300 |
| 1 | 4.221 | 99% | −26% | +0.063 | +0.039 | 0.31 / 0.91 | 0.59 | 0.12 | 300 |
| 10 | 0.1262 | 80% | −0.28% | +0.156 | +0.089 | 0.057 / 0.140 | 6.5e-3 | 1.1e-2 | 300 |
| 20 | 0.0365 | 31% | −0.41% | +0.151 | +0.088 | 0.028 / 0.047 | 2.5e-6 | 7.1e-6 | 300 |
| 30 | 0.0280 | 11% | −0.13% | +0.153 | +0.089 | 0.017 / 0.024 | 8.4e-11 | 2.2e-10 | 300 |
| 40 | 0.0261 | 4% | −0.05% | +0.161 | +0.096 | 0.013 / 0.015 | 2.5e-14 | 7.9e-14 | 288 |
| **51 (stop)** | 0.0255 | 2% | −0.02% | **+0.1605** | +0.094 | 0.009 / 0.010 | 7.3e-17 | 6.5e-16 | **23** |
| 60 | 0.0253 | 1% | −0.01% | 0 | 0 | 0.007 / 0.008 | 1.1e-18 | 0 | 1 |

**Row of the pairing table: collapse into the regulariser — with a metric that gives
no warning.** The penalty is 99% of the loss after one epoch and its gradient exceeds
the data gradient at every epoch from 1 to 120. The loss falls onto the
predict-the-mean line because the penalty falls, and 1b never leaves zero.

- *The metric looks excellent throughout.* Held-out CORR is 0.15–0.16 from epoch 10 to
  51. A metric-only curve would show a healthy run.
- *What is actually happening (plot 5, 4b, 4c).* The network's linear map shrinks by
  two to five orders of magnitude every ten epochs, and the within-day spread with
  it. At the stop the spread is 12 float64 spacings and a day's 300 predictions hold 23
  distinct values: the ranking is made of rounding.
- *The stop rule saves it by nine epochs.* It fires at 51 because the penalty stopped
  falling fast enough, not because of anything in the data. Trained on, the
  predictions are one constant by epoch 60 and CORR drops to 0 in a single step.
- *Not overfitting, not the optimiser.* Train and held-out loss are identical to four
  decimals at every epoch, and the gradients fall smoothly.
- What would show a fix worked: plot 5's net map stops falling, 4c stays at 300.

## The same recipe, seed 47 (`ref_e105_const_s47`)

| Epoch | Train loss | Penalty share | Held-out 1b | Train CORR | Held-out CORR | Held-out MMC | Net map \|W\| | Spread | Distinct |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.342 | 62% | −14586% | −0.037 | −0.093 | −0.063 | 2.7 | 2.6 | 300 |
| 10 | 0.538 | 95% | +0.58% | +0.119 | +0.113 | +0.061 | 0.13 | 3.2e-2 | 300 |
| 20 | 0.0899 | 72% | −0.05% | +0.137 | **+0.136** | +0.078 | 5.2e-3 | 5.9e-3 | 300 |
| 30 | 0.0413 | 39% | −0.52% | +0.125 | +0.119 | +0.067 | 1.1e-4 | 1.1e-4 | 300 |
| 40 | 0.0311 | 19% | −0.29% | +0.043 | −0.016 | −0.021 | 1.2e-6 | 6.8e-7 | 300 |
| 50 | 0.0274 | 9% | −0.13% | −0.019 | −0.097 | −0.065 | 8.0e-9 | 6.6e-9 | 300 |
| **71 (stop)** | 0.0253 | 1% | −0.02% | −0.037 | **−0.118** | −0.073 | 2.4e-14 | 2.7e-14 | 274 |
| 90 | 0.0250 | 0.1% | 0.00% | 0 | 0 | 0 | 1.2e-21 | 0 | 1 |

**Row of the pairing table: sign flip, inside a collapse.** The same collapse as seed
42, but the ranking turns over on the way down.

- *The flip happens mid-run, not at initialisation, and not from rounding.* The
  network learns a positive ranking (held-out CORR peaks at +0.136 at epoch 21), then turns
  negative between epochs 30 and 50 while predictions still have full resolution
  (spread 1e-4 to 7e-9, all 300 distinct). As the penalty shrinks the four layers,
  what survives in their product is no longer the direction the data put there. Plot
  4a shows it directly: the profile rises with the target at epochs 10–30 and falls
  with it at the stop.
- *Train CORR turns too* (+0.137 → −0.037), so this is not a held-out effect.
- *The init's sign is a risk factor.* This seed's untrained network scores −0.093.
  Across seeds 42–141 the final sign under a constant rate agrees with the
  untrained network's sign on 49 of 74 non-constant seeds (correlation +0.47): a seed
  that starts negative ends negative 43% of the time, one that starts positive 18%.
  An observation, not yet a tested cause.

## identity 64-32-16, α=100, `OneCycleLR`, seed 42 and seed 47 (`ref_e105_onecycle_s42`, `_s47`)

| Seed | Epoch | Learning rate | Train loss | Penalty share | Held-out 1b | Held-out CORR | Held-out MMC | Net map \|W\| | Distinct |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0 | 4.0e-5 | 6.855 | 84% | −4450% | +0.056 | +0.039 | 1.5 | 300 |
| 42 | 20 | 9.9e-4 | 0.236 | 89% | −0.48% | +0.130 | +0.074 | 2.2e-2 | 300 |
| 42 | 40 | 4.6e-4 | 0.0474 | 47% | −0.74% | +0.150 | +0.089 | 8.6e-5 | 300 |
| 42 | 60–120 | 4.0e-9 | 0.0408 | 38% | −0.56% | **+0.1507** | +0.089 | 1.2e-5 | 300 |
| 47 | 0 | 4.0e-5 | 9.342 | 62% | −14586% | −0.093 | −0.063 | 2.7 | 300 |
| 47 | 5 | 2.1e-4 | 5.456 | 98% | −269% | +0.040 | +0.035 | 2.1 | 300 |
| 47 | 30 | 8.1e-4 | 0.340 | 93% | +0.86% | +0.120 | +0.064 | 7.0e-2 | 300 |
| 47 | 60–120 | 4.0e-9 | 0.1245 | 80% | +0.36% | **+0.1285** | +0.073 | 1.2e-2 | 300 |

**Row of the pairing table: none of the failure rows — weights frozen by the schedule
before the regulariser wins.** The stop rule fires at 65 (seed 42) and 69 (seed 47),
after the rate has reached 4e-9 and nothing moves any more; every panel is flat from
epoch 60 to 120.

- *It does not converge; it stops.* The penalty is still 38% and 80% of the loss when
  the weights freeze, so the regulariser's optimum (the zero map) is never reached.
  The net map is frozen at 1.2e-5 and 1.2e-2 — eleven and twelve orders of
  magnitude above the constant-rate run of the same seed at its stop.
- *Seed 47 is rescued, not just preserved.* Its untrained network scores −0.093; the
  low starting rate (4e-5) lets the data gradient turn the ranking positive by epoch
  5, and the anneal freezes it there. Across 100 seeds `OneCycleLR` turns 35 of 43
  negative-starting seeds positive; 8 of its 9 remaining negatives started negative.
- *1b settles within ±1.3% here too* once the warm-up is over, at CORR 0.13–0.15: the
  level-vs-rank gap again.
- What it costs: nothing visible on this block. What would show it failing: 2 turning
  after the freeze is impossible, so watch 2 before the anneal ends.

## relu 64-32-16, α=0.01, constant 1e-3, 1024-row batches, torch init, seed 42 (`ref_torchmlp_relu_a001_s42`)

The production `torch_mlp` recipe at an ordinary α (`--batch rows --l2 batch --init
torch`; bit-identical to `TorchMLPRegressor.fit`).

| Epoch | Train loss | Held-out loss | Train 1b | Held-out 1b | Train CORR | Held-out CORR | Held-out MMC | Gradient (data) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0840 | 0.0970 | −234% | −286% | +0.008 | +0.003 | +0.005 | 0.26 |
| 1 | 0.0277 | 0.0304 | −8% | −19% | +0.108 | +0.041 | +0.032 | 0.031 |
| 5 | 0.0227 | 0.0272 | +11% | −7% | +0.336 | **+0.113** (peak) | +0.060 | 0.027 |
| 6 | — | **0.0270** (min) | — | −5.7% | — | — | — | — |
| 20 | 0.0176 | 0.0309 | +32% | −21% | +0.559 | +0.075 | +0.036 | 0.029 |
| 50 | 0.0139 | 0.0362 | +47% | −42% | +0.682 | +0.071 | +0.042 | 0.065 |
| **90 (stop)** | 0.0119 | 0.0414 | +55% | −63% | +0.739 | **+0.058** | +0.030 | 0.052 |
| 120 | 0.0114 | 0.0433 | +57% | −70% | +0.754 | +0.050 | +0.024 | 0.070 |

**Row of the pairing table: overfitting, from epoch 5.** The train loss falls without
interruption while the held-out loss bottoms at epoch 6 and climbs for the rest of
the run.

- *The stop rule runs 85 epochs past the held-out best.* It reads the train loss,
  which keeps improving, so it stops at 90 with held-out CORR 0.058 where epoch 5 had
  0.113. Train CORR reaches 0.74: the network memorises the window.
- *Even the best held-out loss is worse than predicting the mean* (1b −5.7% at epoch
  6). The held-out ranking peaks while its MSE is already behind the mean — gap 1
  again.
- *Not the gradients.* The data part is 0.02–0.13, noisy but with no trend, and the
  penalty part is 1e-4: healthy gradients during overfitting, so do not "fix" the
  optimiser.
- *What the predictions do.* The within-day spread grows from 0.08 to 0.19 and the
  ranking moves by 0.03–0.05 every epoch without improving: the model keeps rewriting
  its held-out ranking while getting worse at it.
- *A detail of 4a that is normal here.* At the stop the top bin (target 1.0) ranks
  below the 0.75 bin (0.486 against 0.523). Every positive run in these panels does
  the same — seed 42 under a constant rate 0.494 against 0.552, under OneCycle 0.504
  against 0.549 — so a dip at the top bin is a property of this target, not a
  pathology.

## Procedure level: relu 64-32, α=0, 500 epochs, 12 folds × 2 seeds (`torchmlp_noreg_500ep_wf_s42` / `_s43`)

TorchMLPRegressor's default network with no regularisation, run on every refit of the
full layout with `--save-preds --predict-next`. Report:
`results/reports/crypto/torchmlp_noreg_500ep_wf_s42_walkforward.html`; reading in
`results/diag/crypto/torchmlp_noreg_500ep_wf_s42_walkforward/reading.md`. Median [IQR]
over the 12 folds:

| Quantity | Value |
|---|---|
| Best epoch, seed 42 | 19 [12, 26]; seed 43 disagrees on 4 folds (10 vs 107, 2 vs 37, 15 vs 165, 28 vs 54) |
| Held-out data-loss minimum | epochs 6–13, before the CORR peak |
| Stop rule | epochs 84–116, after the best in every fold; gives up a median 0.027 CORR |
| CORR at epoch 500 / ridge on the same window | 0.043 / 0.112; gap −0.057 [−0.087, −0.052], ≤ 0 in 12 of 12 |
| Gap at the best epoch (an upper bound) | > 0 in 2 of 12 folds (seed 42), 1 of 12 (seed 43) |
| Churn at the refit boundary, epoch 500 | network 0.887 [0.880, 0.913] · seed floor 0.672 · ridge 0.463 |
| Target-bin profile (0 … 1), best epoch | network 0.41 0.46 0.52 0.52 0.48 · ridge 0.34 0.44 0.54 0.53 0.45 |

- *Every fold is the overfitting row.* Train CORR reaches 0.80, held-out loss climbs
  from epoch ~10, gradients stay healthy, nothing collapses, and P4 flags nothing.
- *A peak location exists within a seed, not across seeds.* Three folds are flat-topped:
  they stay within 0.005 of their best over epochs 10–197, 13–383 and 9–30.
- *The top-bin dip is the target's, not the network's.* The ridge on the same blocks
  ranks the 1.0 bin below the middle in 12 of 12 folds, more sharply than the network.
  What the network lacks is the ridge's separation of the BOTTOM bin, and overfitting
  erodes that further (0.41 at the best epoch, 0.43 at 500).
- *Read P5 against both references.* Reseeding alone (0.67) already moves this network
  more than a ridge refit does (0.46). Cross-fold churn starts near 0 only because every
  refit shares the seed's initial weights. It crosses the seed floor at about epoch 10
  and keeps rising as the network overfits.

## What to take from these

When an MLP here "works", check plot 5 and 4c before believing it: a held-out CORR of
0.16 from a network whose linear map is 1e-16 is a ranking of rounding error that the
stop rule happened to catch. When one "fails", plot 2's epoch-0 point and the shape of
the turn in 2 tell a sign flip from a collapse from overfitting in one run, where the
metric distributions of E104/E105 needed a hundred seeds to show that the three exist.
