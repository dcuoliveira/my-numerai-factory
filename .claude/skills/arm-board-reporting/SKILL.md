---
name: arm-board-reporting
description: Turn a finished set of runs in results/ into a defensible comparison — a ranked table, paired Newey-West deltas, a prediction-correlation matrix, and screened ensembles — for ANY model class in src/prediction/models.py (ridge, ridge_glmnet, lgb, xgb, catboost, the torch LSTM). Use when asked to compare arms, rank runs, decide whether an arm beat its baseline, check whether two models are redundant, suggest ensembles or blends, or publish a results board. Also use before reporting any result, to check the comparison is paired and the baseline is matched.
---

# Reporting a set of arms

This is the reporting half of `crypto-experiment-reporting`, generalised past
ridge. That skill covers running studies; this one covers turning their output
into a claim you can defend. Both read the same `results/` tree.

## The five checks, in order

Run them in this order. Each one has caught a wrong conclusion in this repo.

### 1. Is the baseline matched?

An arm must be compared to a control that differs by ONE thing. Two failures
have happened here:

- **Crossing a hyperparameter.** A LightGBM arm at `min_child_samples 50` vs a
  baseline at 313 moves the columns AND the regularisation. Every leaf-50 arm
  needs a leaf-50 `raw` control.
- **Crossing search vs pinned.** A `--fixed-params` arm vs a 50-draw searched
  arm differs twice over.

If the arm claims to reproduce a deployed slot, check it reproduces the
registry's recorded numbers first. `c02_raw` matched CURUPIRA_02 to six decimals,
and that is what made everything downstream trustworthy.

### 2. Is the comparison paired, Newey-West at lag 20?

Arms share periods, folds and seed, so the DIFFERENCE series has a far smaller
standard error than either arm alone. The same data gave `t = 0.75` unpaired and
`t = 3.14` paired. And a 20-day target on daily dates leaves consecutive periods
~95% overlapping: 510 periods are ~26 effective independent windows, so treating
them as independent inflates every t by roughly `sqrt(20)`.

```bash
python3 .claude/skills/crypto-experiment-reporting/scripts/paired_delta.py \
    --model lgb --baseline fe_lgb_reg_raw --arms fe_lgb_reg_interact
```

### 3. Is the arm a measurement at all?

An arm that produces the same predictions as another is not evidence about its
feature block. This is not hypothetical:

- `fe_lgb_reg_onehot` came back BYTE-IDENTICAL to `fe_lgb_reg_raw`
  (`max|diff| = 0.00e+00`). A ticker holds at most 130 rows in a 6-month block,
  so at `min_child_samples 313` no split can use a one-hot column and LightGBM
  discards all 740 before training.
- The arm NAME will not tell you. `allfeat`, `allfeat++` and `allfeat+++` all
  carry the same 740 columns and none of them says so.

So derive containment from the data, never the name:

```bash
python3 .claude/skills/arm-board-reporting/scripts/check_arms.py --model lgb --runs RUN_A RUN_B
```

It reports identical prediction pairs, per-arm feature counts, and which blocks
each dataset actually contains.

### 4. Level AND risk, never level alone

Payout is `0.1*corr + mmc`. Report `corr`, `mmc`, `payout`, `corr_sharpe` and
`corr_max_drawdown` together — arms routinely move them in opposite directions.
The largest and most repeatable effect found in the LightGBM study was invisible
in the mean: leaf 50 on a wide feature set roughly HALVES max drawdown
(0.18-0.35 against 0.42-1.17) at indistinguishable payout.

### 5. Would a second arm actually add anything?

Payout cannot answer this; prediction correlation can. Rank predictions within
each period, correlate, average over periods — `within_period_rank_corr` in
`src/analysis/corr_heatmap.py`.

Reference values from this repo, for calibration:

```
ridge variants against each other      0.883 - 0.926   effectively one model
lgb against the ridge book             0.705 - 0.751
lgb classification against ridge       0.253 - 0.293
the torch LSTM against everything      0.061 - 0.183   genuinely orthogonal
```

An arm at 0.9 with a better mean is a worse slot than an arm at 0.2 with a
similar one.

## Ensembles

`src/run_ensemble.py` rank-averages FINISHED forecasts, so screening costs
seconds, not model fits. Screen on the payout series.

**Compare an ensemble to its own best member, not to a global baseline.** The
question is "does combining beat just holding the best of these", and
equal-weight averaging dilutes the mean by construction.

What that screen returned here, over 3,276 combinations: **zero** beat their best
member on payout at |t| > 2, and every point estimate was negative. The case for
blending was entirely risk — the best pairing cut max drawdown from 1.00 to 0.23
and lifted sharpe from 1.47 to 1.72, on members correlating rho 0.03-0.08. Report
blends that way round.

```bash
python3 .claude/skills/arm-board-reporting/scripts/screen_ensembles.py \
    --model lgb --runs RUN_A RUN_B --with-registry --top 8
```

## Writing the result

- **A null is the finding.** On ~26 effective windows |t| < 2 is expected. Write
  "these columns add nothing detectable, and here is the interval", not
  "inconclusive, needs more arms".
- **Say what is missing.** A blank row reads as "tested and null" when it means
  "never run". Name skipped, capped and truncated arms explicitly.
- **Footnote arms that are not candidates, do not delete them.** `fencglobal`
  scored best in the study and is lookahead — a whole-sample feature mean encodes
  values you would not have live. Deleting it means the next reader re-discovers
  and deploys it.
- **Report a result's conditions.** `interact` cleared t = +2.99 at leaf 313 and
  t = +0.36 at leaf 50. It is one result among 38 arms; a Bonferroni-style
  adjustment puts the threshold near |t| = 2.9. Say so next to the number.

## Publishing a board

`scripts/build_board.py` renders any set of runs as an HTML artifact: ranked
table, correlation heatmap, ensemble candidates. It is model-agnostic — pass any
`results/metrics/<tournament>/<model>/` runs.

```bash
python3 .claude/skills/arm-board-reporting/scripts/build_board.py \
    --model lgb --runs 'fe_lgb_reg_*' --baseline fe_lgb_reg_raw --out board.html
```

Group by whatever confound separates the arms (leaf size, cadence, search type)
rather than pooling them into one ranking, and put the summary before the detail.
