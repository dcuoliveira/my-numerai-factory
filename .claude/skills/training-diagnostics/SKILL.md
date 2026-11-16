---
name: training-diagnostics
description: >-
  The diagnostic panel for training neural networks in this repo — (1) loss on the
  optimised quantity, data part and L2-penalty part, train and held-out from epoch 0,
  plus the fit beyond predicting the mean, (2) per-day CORR and MMC on the same two
  sets, (3) gradient norms on a fixed batch, split into data and penalty parts, in
  total and per layer, (4) held-out prediction dynamics — per-target-bin ranking,
  within-day spread against float64 spacing, resolution and movement, (5) learning
  rate and weight norms — and how to read their pairings. For a walk-forward fit it
  adds the procedure level across all K refits, then full panels on three folds. Use
  it whenever you train,
  retrain, lengthen, widen, regularise or compare a torch network here (torch_mlp,
  torch_mlp_loss, the from-scratch E104/E105 recipes, the LSTM in src/prediction/nn or
  a new architecture), whenever you plot or report a training curve, and whenever you
  have to explain training behaviour: a held-out score that rises then falls,
  suspected overfitting, a loss that stalls or climbs, a model that predicts a
  constant, a sign-flipped or seed-dependent result, noisy epochs, a bigger model that
  does not fit better, vanishing or exploding gradients, a scheduler that does or does
  not help. Use it before proposing a change to the learning rate, schedule, capacity,
  alpha, loss or number of epochs, and before launching another 100-seed sweep to "see
  what happens" — even if the user only asks "why did it get worse?" or "plot the
  training curve". A curve of the metric alone is not a diagnosis.
---

# Training diagnostics: the panel before any theory

## Why this exists

The optimiser never sees what we are paid on. It minimises `0.5 · MSE + 0.5 · α · Σ‖W‖² /
batch` pooled over every row of a 130-day window. We are paid on CORR — a rank
correlation computed *within each day* — and MMC, its contribution beyond the meta
model. A chart of held-out CORR cannot tell these apart:

- the model stopped generalising (overfitting),
- the model stopped fitting at all (the optimiser is failing),
- the model kept improving its loss while the metric went elsewhere (misaligned),
- **the model is decaying into its regulariser** — the loss falls because the penalty
  falls, and whatever ranking is left lives in the last bits of a float64.

Each calls for a different fix, and here the fourth was guessed at from metric
distributions for three studies. E33's identity MLP at α=100 topped the MLP calmar
board at 0.859; E104 and E105 then spent about 1,500 runs establishing that its
outcome is a lottery across seeds (constant, sign-flipped or 0.13) and that annealing
the learning rate rescues it. One panel of one run shows the mechanism in minutes: the
penalty part owns the loss and the gradient from the first epochs, the network's
linear map decays geometrically toward 1e-16, and the held-out predictions hold a few
dozen distinct values a day. See `references/worked-examples.md`.

So a training run that is being reported, compared or debugged gets the panel, and
the reading of the panel comes before the explanation and before the fix.

## The panel

All measured at the same moments — before training, then after every epoch — on data
that never changes during the run: the whole training window (130 days, 39,000 rows),
the whole held-out block the refit predicts (130 days), and one fixed training batch
of the kind the run trains on (1,024 rows, or 3 whole days) for the gradients.

**1a · Loss — the optimised quantity.** Train and held-out on one log axis from epoch
0: data loss + penalty, in eval mode at fixed weights. The penalty part is drawn on
its own (it is the same on both sets), and so is the level of predicting the training
mean everywhere (`0.5·var(y)`, ≈ 0.025 here). The loop's own running loss is a thin
line: measured while the weights move under it, never compared with held-out.

**1b · Fit beyond predicting the mean.** `1 − data loss ÷ mean-only loss`, as a %, for
both sets — an R² against the training mean. MSE pays for the intercept; a per-day
ranking cannot use it. This is the part of the fit the metric could see.

**2 · Metric — what we are paid on.** Mean per-day CORR on train and held-out, both on
all rows (like for like), and held-out MMC against the real meta model. Train MMC only
when the meta model covers most of the window. A day whose predictions are all equal
scores 0, it is not dropped.

**3a · Gradient on a fixed training batch** (log). The L2 norm of the loss gradient,
and of its two parts — data and penalty — measured with `torch.autograd.grad`, no
optimiser step. A clip line when the loop clips (the LSTM's `max_grad_norm`).
**3b · Per layer** (log): each Linear layer's total; per gate or block for a recurrent
or convolutional net (give it groups in `training_diagnostics.layer_groups`).

**4a · How held-out predictions rank each target bin.** The target has five levels, so
the scatter becomes a profile: the mean within-day percentile of the prediction for
each target value, at a handful of epochs. Rising left to right is skill; flat at 0.5
is none; *falling* is a sign flip. **4b · Within-day spread** (log), against the float64
spacing at the predictions' level — a spread within a factor ~100 of it is a ranking
made of rounding. **4c · Resolution and movement** (0–1): the share of a day's stocks
with distinct predictions, and `1 −` the within-day rank correlation with the previous
epoch's predictions.

**5 · Learning rate and weights** (log). The rate each epoch actually used, each
layer's Frobenius norm and — for a stack of Linear layers alone — the norm of the
single linear map it composes to. Adam normalises its steps, so a small gradient does
not mean small steps: this panel shows what the optimiser did to the weights.

## Producing the panel

The driver trains one walk-forward refit of a torch MLP with the exact loop of the run
it describes, and snapshots every epoch:

```bash
python -m src.prediction.diagnose --tag e105_const_s47 --seed 47 --refit 0 \
    --activation identity --alpha 100 --batch days --days 3 --l2 pinned --init sklearn \
    --schedule const
```

```bash
python -m src.analysis.plot_training_diagnostics e105_const_s47    # or no tag: every run
```

| run being diagnosed | flags |
|---|---|
| `torch_mlp` (`TorchMLPRegressor`, mse) | `--batch rows --l2 batch --init torch` + its `--activation/--alpha/--lr/--hidden` |
| E104 `row` / `rowfix` / `row900` | `--batch rows --l2 batch\|pinned [--batch-size 900] --init sklearn` |
| E104 `day3` / `day1` (+`lit`) | `--batch days --days 3\|1 --l2 pinned\|batch --init sklearn` |
| E105 arms | `--batch days --days 3 --l2 pinned --init sklearn --schedule const\|step\|exp\|cosine\|warmcos\|restarts\|onecycle\|plateau` |
| any of the above trained on −CORR | add `--loss corr`: minus the mean per-day Pearson over the batch's days (use `--batch days`, so each day is whole) |
| … trained on −MMC | add `--loss mmc --meta-only --scheme even --n-refits 12`: Numerai's contribution per day with the prediction z-scored instead of rank-gaussianised. It needs the meta model on every row, which exists from 2024-05-07 only (547 days), so the folds are re-cut evenly over those days. Report with `walkforward_diagnostics --metric mmc` |

With `--loss corr` the panel changes in two places, because the loss has no level or
scale. 1a is drawn on a linear axis, with 0 as a constant prediction. 1b becomes the
optimised per-day Pearson against the Numerai CORR that pays: once level, intercept and
pooling are out of the loss, Pearson-on-values against the ranked CORR is the gap left
between them.

Verified, not assumed: the driver's final held-out predictions equal `TorchMLPRegressor.fit`
bit for bit, and `e105_const` / `e105_onecycle` to scorer noise (9.7e-17) for the same
seed and refit. Output in `results/diag/crypto/<tag>/`: `log.json` (every number,
rewritten each epoch, so the plotter works mid-run), `preds.npz` (held-out predictions
at milestone epochs and the last), `config.json`, `diagnostics.png`.

Practical notes:

- **Cost.** Training a refit takes ~10 s; scoring CORR and MMC takes ~2–4 s per
  snapshot and dominates. `--metric-every 5` thins the metric (the last epoch is always
  scored); losses, gradients and predictions stay per epoch.
- **`--no-stop --max-epochs N`** trains past sklearn's stop rule; the panel still marks
  where it would have stopped. Use it to see what the stop rule is protecting you from.
- **Bit-reproduction needs one thread** (`--threads 1`, the default) and the target in
  the dtype the pipeline scores it in: the driver scores against the parquet's float32
  target; scoring a float64 copy moves CORR by ~1e-9.
- **No replay.** Runs here keep no checkpoints; a refit retrains in seconds, so rerun
  with the same flags and seed instead.
- **sklearn's `mlp` has no hooks.** Diagnose its torch twin with `--init sklearn` (the
  exact sklearn initial weights). For a well-conditioned fit the two agree; at α=100
  they track each other (per-period r 0.95) but not bit for bit — say so.
- **Other loops call the Snapshotter directly.** `Snapshotter(out_dir, Objective(α,
  denominator, day_balanced), train, held, probe)` then `.snapshot(epoch, net, lr=...,
  running_loss=...)` at epoch 0 and after each epoch, and `.finish(net)`. That is how
  the LSTM's `run_epoch` / `train_until_early_stop` or a new driver should log, rather
  than growing its own curves — so every panel stays comparable. Snapshots never step
  the optimiser, write `.grad` or draw random numbers (`src/tests/test_training_diagnostics.py`).

## Reading the panel

Read 1a, 1b and 2 together first. The combination is the diagnosis; any one alone is
ambiguous.

| 1a loss, train · held | 1b fit beyond mean | 2 held-out metric | Reading | Then look at |
|---|---|---|---|---|
| falling · falling | rising on both | rising | learning; nothing to fix yet | — |
| falling · rising | train rising, held falling | falling after a peak | **overfitting** — the optimiser is fine | where held-out turns; α, capacity, early stopping — the stop rule reads the train loss and will not stop it |
| falling · falling, onto the predict-the-mean line | ≈ 0 on both | steady — often good — then a cliff to 0, with no warning in 2 itself | **collapse into the regulariser** | 3a penalty part above the data part; 5 net map falling geometrically; 4b spread → float spacing; 4c distinct share falling |
| falling · falling | ≈ 0 | turns negative after a positive peak; train CORR turns too | **sign flip** — what survives the shrinkage points the wrong way | 4a profile falling at the end, rising earlier; epoch 0 of 2 (the untrained sign); the same recipe at other seeds |
| falling · falling | rising | flat or falling | **objective misaligned** with the metric | the four gaps below |
| flat from the start | 0 | flat | not learning | 3a near zero or pinned at a clip; 4b a flat band |
| rising, or falling then rising | any | any | **the optimiser is failing** | 3a trend; 5 weights growing; lr |
| noisy epoch to epoch | noisy | noisy | step too large for the noise in the batches | 4c movement; 3a range |

If you catch yourself explaining a curve without having named its row in this table,
stop and name it.

**The four gaps between loss and metric here.** When 1a and 2 disagree, it is one of
these, and the panel separates them:

1. *Level versus rank.* MSE pays for getting the intercept and each day's level right;
   CORR ranks within a day and ignores both. 1b removes the intercept; what is left in
   1b is what a ranking could use.
2. *Pooled versus per-day.* The loss pools 130 days; the metric correlates within
   each. The binned target has the same distribution every day, so between-day
   structure can lower the loss without ranking anything. Compare 1b with 2.
3. *Data versus penalty.* The optimised quantity includes the penalty. At α=100 the
   loss curve *is* the penalty curve, and it falls for a reason unrelated to the data.
4. *All rows versus the meta-model subset.* MMC exists only where the meta model is
   published; check its coverage before reading train MMC.

**Plot 3 — gradients.**

| What you see | Reading |
|---|---|
| penalty part above the data part, and staying there | the regulariser, not the data, is doing the training; expect 5 to show the weights shrinking |
| data part falls by orders of magnitude while the penalty part falls in step | the network is being pulled to the origin, with the data gradient vanishing because the weights do |
| total pinned at a clip line for most epochs (LSTM) | clipping sets the step size, not the learning rate |
| total climbs while the train loss worsens | divergence: lower or decay the learning rate |
| one layer, gate or block far below the rest after epoch 1 | that part has stopped learning; if it is what a larger model added, the extra capacity is not used — check after one epoch, before paying for the rest |
| flat, moderate total while held-out loss rises | healthy gradients; this is overfitting, do not "fix" the optimiser |

**Plot 4 — predictions.**

| What you see | Reading |
|---|---|
| 4b spread falling toward the float64 spacing, 4c distinct share falling | predictions are being made of rounding; the metric now ranks noise |
| 4b spread exactly 0, distinct share 1/300 | constant predictor — every day scores 0 |
| 4a profile rising, steepening; 4c movement decaying | genuine sharpening |
| 4a profile falling | sign flip — the model ranks the bins backwards |
| 4a profile rising to the 0.75 bin, dipping at 1.0 | normal for this target — every positive reference run does it |
| 4a profile flattening toward 0.5 while spread holds | held-out ranking decorrelating — overfitting seen from here |
| 4c movement high while 2 is flat | the model reshuffles its ranking every epoch without improving |

**Plot 5 — learning rate and weights.**

| What you see | Reading |
|---|---|
| weights (and the net map) falling geometrically at a constant rate | Adam walking to the penalty's minimum, the zero map; more epochs make it worse |
| weights freezing as the rate anneals | the schedule is stopping the walk — the E105 mechanism; it is early stopping in learning-rate time, not a better fit |
| weights growing while 3a climbs | divergence |

## Across the walk-forward: K refits, then three panels

The pipeline refits every 130 days (4 blocks in the standard layout, 12 with `--scheme
full`), so a recipe has K training trajectories, not one. What is under test is the
*procedure*: does refitting give consistent results? No single-run panel can answer
that. Run the driver once per fold with identical flags and seed (`--refit 0..K-1`, tags
ending `_rNN`, plus `--save-preds --no-train-preds --predict-next`), and do the same for a second seed to
get P5's floor. Then draw the procedure level and read two levels:

```bash
python -m src.analysis.walkforward_diagnostics --base <base> --floor-base <base_seed2> --html
```

It writes P1–P5, the per-fold table and the three folds to panel into
`results/diag/crypto/<base>_walkforward/`. `--html` builds a report from those outputs
and the fold panels, and `--notes <md>` puts your reading at its top.

**Procedure level, all K folds.** These are numbered P1–P5 to keep them apart from the
panel's 1–5.

- **P1 · Heatmap, folds × epochs.** Colour each cell by held-out CORR minus the fold's
  baseline (P3), diverging at 0, and mark each row's stop-rule epoch. It replaces the
  single held-out curve. One curve shows where one run peaked; the heatmap shows
  whether a peak location exists at all across refits. A vertical band means it does,
  scattered peaks mean it does not, and a row that turns negative late is a fold that
  sign-flipped.
- **P2 · Strip plot of the best epoch per fold**, with each fold's stop-rule epoch
  beside it. In one run the best epoch is a scalar; across folds it is a distribution.
  Its spread decides whether the stopping rule is a rule or a coin flip. Draw a
  flat-topped fold as the range of epochs within 0.005 of its best, not as a point,
  because its argmax is noise. The best epoch is picked on the scored block, so it
  judges the stop rule. Never deploy it as a setting.
- **P3 · Score minus the per-fold baseline.** Period difficulty varies across folds
  more than model quality does, so raw per-fold scores mix the two and only the gap is
  comparable. Use a ridge fitted on the same fold's training window and scored on the
  same block as the baseline; it takes seconds on the driver's own scaled matrices. Do
  not use the meta model, which covers only recent history, so most of the 12 full-layout
  folds have no MMC. Do not use the untrained network either: its CORR has an sd of
  0.074 across seeds.
- **P4 · Gradient-norm envelope across folds.** Plot the median of 3a's total across
  folds per epoch with the min–max band, on a log axis. It compresses K copies of 3a
  into one, and the question changes from "were the gradients healthy" to "did any
  fold misbehave". It only screens: a fold that leaves the envelope gets its full panel
  and is not diagnosed here.
- **P5 · Cross-fold churn.** Churn is `1 −` the mean within-day rank correlation
  between refit k's final model and refit k+1's on the same out-of-sample rows (block
  k+1). It measures how far the fitted function moves at each refit, and it reads
  directly as the turnover cost of that swap. The floor is seed churn: two seeds of one
  fold, compared on that fold's block. Churn near the floor means the refit moved the
  function no more than reseeding would (instability). Churn well above it means the new
  window changed the function. If a fold's score drops while churn stays at the floor,
  the period got harder; the model did not change. Refits that share a seed start from
  the same weights, so early-epoch cross-fold churn is near 0 by construction; read it
  at the kept epoch, beside the ridge's own refit churn, which the tool also reports.
  P5 needs `--predict-next`: the
  driver keeps no weights, so without that flag refit k never predicts block k+1. Fold
  k+1's predictions on the 110 days of block k inside its training window are in-sample
  and cannot stand in.

**Run level: three folds, first, median and worst.** These get the full panel, 1a–5,
unchanged and unthinned. Only the sampling changes. You no longer look at every run, so
the choice of fold carries the weight:

- *First* (refit 0) and *worst* (the lowest P3 gap at the stop-rule epoch, which is
  what the pipeline keeps) are the folds that catch bugs.
- *Median* (the median gap) is the reference the other two are read against.
- Add any fold that P1 or P4 flagged.

Keep the panels unthinned because aggregation hides mechanism. The heatmap says fold 7
is bad; only fold 7's panel says which row of the pairing table it falls in.

**Table: one row per fold.** The rows are folds, not checkpoints. Columns:

- held-out dates
- stop-rule epoch and best epoch
- held-out CORR at each of those two epochs
- baseline CORR and the gap
- MMC, where the meta model covers the block
- churn into the next fold

The summary row is the median with its IQR, never the mean, because a mean across
folds is exactly the reduction that hides a dead fold. Also count the folds whose gap
is ≤ 0. With the standard 4 folds, list all four instead, because an IQR of four numbers
is not a distribution.

## Traps specific to this project

- **One seed explains a run; it does not rank recipes.** At α=100 the same recipe gives
  a working, constant or sign-flipped model depending on the seed (E104: 14–20 of 100
  constant, ~22 negative), and E104's seed 42 alone gave a false t +4.18. Rank recipes
  with 100 seeds and the paired tests of `crypto-experiment-reporting`; use the panel
  to explain why one run behaves as it does. K folds of one seed do not rank recipes
  either. They show whether one procedure is consistent.
- **The held-out block is the walk-forward's scored block.** It is clean in a
  diagnostic run because nothing is selected on it. Never pick an epoch on it — the
  stop rule exists for that, and it reads the training loss.
- **The stop rule reads the loss including the penalty.** sklearn's rule (running
  best, tol 1e-4, patience 10) fires when the *total* stops improving; at strong α that
  is when the penalty stops falling, whatever the data part is doing. torch's
  `ReduceLROnPlateau` compares with the last accepted best instead, so at the same
  tolerance it never fires (E105).
- **Adam normalises.** A gradient of 1e-6 still moves each weight by roughly the
  learning rate per step. Read step sizes off plot 5, not plot 3.
- **Epoch 0 is not neutral.** An untrained network already ranks at CORR ±0.07 (sd
  over 100 seeds) on this data. Read plot 2 from epoch 0: under a constant rate a
  negative start ended negative 43% of the time, a positive one 18%.
- **MSE is nearly blind to what pays.** At α=100 the held-out data loss stays within
  ~1% of predicting the mean while CORR reaches 0.13–0.16, and an overfit relu net
  hits its best held-out CORR while its MSE is already behind the mean. Never judge
  these networks on 1a or 1b alone.
- **Train CORR is in-sample** and the two sets differ in regime; held-out above train
  early on is not a leak.
- **Day-balanced and row-mean losses coincide** when every day holds the same number
  of stocks — all four standard training windows hold exactly 300.

`references/worked-examples.md` holds the panels of the networks already diagnosed
here, with the numbers that are normal for this data. Read it when you need to judge
whether a value — a penalty share of 99%, a spread of 1e-15, a train CORR of 0.7 — is
healthy.

## Reporting a run

Lead with the row of the pairing table the run falls in, in one sentence, and the
epoch where it turns. Then show `diagnostics.png`. Then the scores table, following
`crypto-experiment-reporting` (CORR, MMC, payout, calmar with `n`). If a fix is
proposed, say which panel would show that it worked.

For a walk-forward recipe, which is every production fit, lead with the procedure
instead. Give one sentence each for P1–P3: whether a peak exists, how wide the
best-epoch spread is, and the gap's median and IQR. Then name any fold P4 flagged and
give P5 if it was produced. Next come the three fold panels, each with its row of the
pairing table, then the per-fold table, and the scores table last.
