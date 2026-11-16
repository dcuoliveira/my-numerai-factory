---
name: crypto-experiment-reporting
description: "Run and report arm-based experiments in the current src/ Numerai Crypto pipeline — src.run_forecast, experiments/*.sh, results/{forecast,per_period,hyperparams,metrics}/, and the CURUPIRA slots in src/live/registry.py. Use this whenever comparing feature sets (the judgement protocol for those lives in feature-set-evaluation) or model variants on crypto, reading a finished run's numbers, deciding whether an arm actually beat its baseline, reproducing a deployed CURUPIRA slot, building a new experiments/ runner, or emailing a long job's result — even if the user just says 'does X help', 'compare these runs', 'add a feature block and see', or 'report the result'."
---

# Crypto experiment reporting

## Orientation

The pipeline is `src/`, and it is shaped like this:

```
src/run_forecast.py        one invocation = one fully-specified run
src/run_ensemble.py        combine finished runs by name (per-period rank average)
src/data_mani/             the dataset layer; builders emit feature_* columns
src/prediction/            models.py (sklearn wrappers), functions.py, metrics.py
src/analysis/              ablation_report.py, cross_target.py, weighting_report.py
src/live/registry.py       the deployed CURUPIRA slots and their fitted configs
experiments/*.sh           one shell script per study
results/                   four parallel output trees, gitignored
```

There is no `experiment.md` and no config files — configuration is argparse plus
shell env vars, and the resolved config is written into every metrics json so a
run is reproducible from its own artifact.

## The reporting contract

This is the part that decides whether a conclusion is right, so start here.

### Compare paired per-period deltas, never two means

Arms share their periods, folds and seed and differ by one block of columns.
Their per-period score series therefore move together, and the DIFFERENCE series
has a far smaller standard error than either series alone. The same data gave
`t = 0.75` unpaired and `t = 3.14` paired (`src/analysis/ablation_report.py`).
Ranking arms on unpaired means finds nothing and concludes, wrongly, that nothing
matters.

### Use Newey-West at lag 20

A 20-day forward target on daily dates leaves consecutive periods ~95%
overlapping. Treating them as independent inflates every t by roughly `sqrt(20)`.
`newey_west(series, lag=20)` lives in `src/analysis/ablation_report.py` and
returns `(mean, se, t)`; `weighting_report.py` and `cross_target.py` carry the
same implementation.

Both of the above are already done for you:

```bash
python3 .claude/skills/crypto-experiment-reporting/scripts/paired_delta.py \
    --model ridge --baseline c02_raw --arms c02_raw_ae
```

It reads `results/per_period/`, joins on the period index, adds the payout
column, and prints baseline / arm / delta / NW se / t per metric. Members in
another model tree are addressed as `model/run`.

### A null result is a finding, not a failure

510 periods at ~95% overlap is roughly **26 effective independent windows**. On a
saturated feature set `|t| < 2` is the expected outcome. Report it as the answer —
"the codes add nothing detectable, and here is the interval" — rather than as an
inconclusive run needing more arms. Eleven feature blocks have been tried on this
panel; none beat the raw 22 by a margin the sample supports. Say so plainly.

### Report level and risk

Payout is `0.1 * corr + mmc`. Always give corr, mmc and payout — but read
`corr_sharpe` and `corr_max_drawdown` from the metrics json too. Arms routinely
move level and risk in opposite directions, and a mean-only table hides it. In the
CURUPIRA_02 autoencoder study the arm lost 0.005 corr while gaining 0.06 sharpe
and cutting max drawdown 24%; reporting only the mean would have been true and
misleading.

**Always carry the prop `calmar` in a comparison table, next to payout.** It is
the per-period mean payout over the max peak-to-trough drop of the CUMULATIVE
payout (`summarize(0.1*corr + mmc)["mean"] / ["max_drawdown"]`, drawdown positive),
formed from the per-period series — NOT rebuilt from stored means. It catches the
case a mean-only table hides: an arm that trails on payout but earns it far more
smoothly (e.g. the crbmc interaction arms sit below raw on payout yet post higher
calmar — lower drawdown per unit payout). Caveat built in: it is comparable only
across runs of equal period count, so print `n` beside it. The shared block runner
`experiments/mommask_core.sh` emits it by default in its level table
(`CORR MMC payout c.sharpe calmar n` per arm); keep that column when you build a
new runner or render a board.

Report downside as both DEPTH and FREQUENCY, because they disagree. `dd(pay)` is
the deepest single fall; `burn(d)` (from `results_table.burn_days`, and a column in
`results_table` / `row_from_frame`) is how many periods came out negative. Two runs
at the same drawdown can burn very different day counts. It is a raw COUNT of periods,
so like `calmar` it is comparable only across rows on the same window: slice to a
common window and print `n` beside it (or use the rate `burn(d)/n`). State the caveat
when you report it — a 20-day target overlaps ~95% period to period, so this is time
under water, not that many independent losing bets.

### Verify a baseline before trusting a comparison

When reproducing a deployed slot, check the rerun reproduces the registry's
recorded numbers before comparing anything to it. `c02_raw` matched CURUPIRA_02's
`corr 0.153169` and `mmc 0.070446` to six decimals, which is what made the
comparison trustworthy. If it does not match, the config is wrong — find out why
before running the treatment arm.

## The results tree

`src/run_forecast.py:write_outputs` writes four files per run:

```
results/forecast/<tournament>/<model>/<run>.csv       per-row OOS predictions
results/per_period/<tournament>/<model>/<run>.csv     date,corr,bmc,mmc,pnl
results/hyperparams/<tournament>/<model>/<run>.csv    one row per refit
results/metrics/<tournament>/<model>/<run>.json       summaries + the full config
```

The metrics json embeds `config` (data_version, n_features, feature_include /
feature_exclude, refit_every, embargo, cv_scheme, train_window, seed, …) plus
`runtime_minutes`. Read a past run's config from there rather than guessing.

**Run names are hand-set and the same name overwrites.** There is no timestamp
suffix. Pick names that encode the arm (`w6r6_glmnet_reg_allfeatpp`,
`c02_raw_ae`), and never reuse a name you still need.

## Running an arm study

One shell script per study in `experiments/`, with a header stating what it runs,
what it is measured against, and what would count as a result. Conventions that
exist because each one has bitten:

- **Prefix every env var** (`AE_`, `AFP_`, `GRP_`, `ABL_`). GNU screen exports
  `$WINDOW=0` into every session it starts, so a bare `WINDOW="${WINDOW:-6month}"`
  silently becomes `0` and every training block comes out empty.
- **Parallel arrays, never pipe-delimited records.** Feature regexes contain `|`
  as alternation, so packing `name|include|exclude` and splitting on `IFS='|'`
  tears them apart and quietly fits on the wrong columns.
- **Gate on feature count before running.** A silent count drift attaches a
  plausible number to the wrong experiment, which is worse than a crash. Read the
  parquet schema with pyarrow and refuse to run on a mismatch.
- **A failed arm must not cost you the others.** Check `PIPESTATUS[0]` after the
  `tee`, record the failure, continue.
- `DRY_RUN=1` prints commands and exits; `ONLY=`/`SKIP=` select arms;
  `SHARD=`/`OF=` round-robin across machines. Tee each arm to `logs/`.

`SHARD=`/`OF=` splits the arms; it does not decide WHICH machines take them or
how much of each box you may have. That is `cluster-job-placement` — the 30%
share, the single box with a working GPU, and the slot sizing that
`--search-jobs` x `--n-jobs` has to match (an `rf_ktz` arm wants 16 cores, which
does not fit a 32-core box's share). Read it before launching anything that
leaves this machine.

For arms whose features are produced by a fitted builder (autoencoder codes,
target encodings, meta proxies), **verify the leakage boundary rather than
asserting it**: compute where the scored block starts, read the builder's recorded
fit boundary, and refuse to run unless the block starts strictly after it. Note
that a builder's boundary is a function of the `--val-frac` it was BUILT with
while the scored block follows the `--val-frac` the run USES — build at 0.3, run
at 0.5, and the block moves back across the boundary with no error.

## Reproducing a deployed CURUPIRA slot

`src/live/registry.py` holds each slot's `wrapper`, `feature_set`, `n_features`,
`backtest_run`, and a `fit` block merged from a shared `FIT` dict plus any
`fit_override`. Copy that block field-for-field into the run_forecast flags:

```bash
python3 -u -m src.run_forecast crypto ridge \
    --task regression --metric corr --bmc-reference benchmark \
    --cv-scheme rolling --train-window 130 --refit-every 130 --embargo 20 \
    --n-splits 5 --n-iter 50 --seed 42 --val-frac 0.3 \
    --data-version crypto/v2.0 --run-name c02_raw
```

Read each slot's `backtest_run` and pull that run's `config` from
`results/metrics/` for the exact `data_version` and any
`feature_include`/`feature_exclude` — the registry names the feature SET, not the
dataset version, and the two are not always the obvious pairing (CURUPIRA_03's
`onehot_roll` is `crypto/v2.0-onehot-rollsel`, not the `tenc-` prefixed one).

Only CURUPIRA_01/02/03/04 are single models. The rest are blends.

## Blends cost seconds, not model runs

`CURUPIRA_05` and `05_60` are equal-weight per-period rank averages of five
feature-block member runs; `06` and `07` are rank averages of other SLOTS. So once
the member runs exist, every blend is a `run_ensemble` call:

```bash
python3 -m src.run_ensemble crypto ridge_glmnet \
    w6r6_alone_onehot w6r6_alone_tenc w6r6_alone_interact \
    ridge/w6r6_grp_size_fixed w6r6_grp_price_glmnet \
    --weights 0.2,0.2,0.2,0.2,0.2 --run-name my_blend
```

A member may live in another model tree (`ridge/…`). A blend of runs trained on a
different label needs `--target`, because a 60-day run's forecast file carries
only `target_binned_return_60`. Plan an experiment around this: derive every blend
from member runs instead of budgeting model fits for it.

## Email a long job's result

`src/live/notify.py` is already configured from `.env`
(`SMTP_HOST/PORT/USER/PASSWORD`, `NOTIFY_TO`):

```python
from src.live.notify import send_text
send_text("study SUCCESS", open("logs/report.txt").read())
```

Mail on success **and** failure — a job that only mails when it breaks is
indistinguishable from a job that silently stopped running. And never let SMTP
break the run: wrap the call, print the mail failure, exit on the experiment's own
status. Give the runner a `NO_MAIL=1` escape hatch.

## Runtime: price a study before you launch it

Across 141 full-resolution ridge arms in `results/metrics/`, wall clock
correlates with the **number of refits** (r = 0.48) and essentially not with
feature count (r = −0.12). The walk-forward and per-period scoring dominate; a
4-feature arm can outlast an 812-feature one. Budget in refits and CV scheme,
never in columns:

| cadence | measured | n |
|---|---|---|
| `rolling`, refit 130, window 130 — **the CURUPIRA cadence** | median 28 min, p90 53, all under 60 | 92 |
| `rolling`, refit 65 | median 117 min, up to 438 | 5 |
| `expanding`, refit 130 | 339 min | 1 |
| `expanding`, refit 65 | 581–647 min | 2 |

Two things blow the budget up, and neither is the feature set. Halving
`--refit-every` doubles the refits. And `--cv-scheme expanding` grows the
training set at every refit, so it costs an order of magnitude more than
`rolling` at the same cadence — 339 min against a 28-min median. Read both off
the arm's config before quoting a time.

`lgb` runs 97–180 min at that same cadence (687 for a Bayes search), and the
LSTM path reached 665.
`run_ensemble` blends are **< 1 min** — they only combine finished forecasts.

**Scout with `--era-step`.** Keeping every Nth period turns an arm into seconds
(`--era-step 4` runs land at 0.1–0.4 min), which is how you shake out a runner's
wiring, a feature-count gate or a new dataset before committing real hours. It is
a scout lever only — the split boundary moves, so never report a downsampled arm
against a full-resolution baseline.

Price a study out loud before starting. At the CURUPIRA cadence that is
`arms × ~30 min`; a 14-config, 2-arm study is ~14 hours. Run it in the background
and say the number up front — "this will take minutes" is the easiest thing to
get wrong here, and the correction lands after the user has already waited.

## Before you report, check

- The baseline reproduces its registry / prior numbers, if it claims to be a
  reproduction.
- The comparison is paired, Newey-West at lag 20, on an inner join of periods.
- Both level (corr, mmc, payout) and risk (sharpe, max_dd) are reported.
- Any fitted feature block's boundary is verified against the scored block, with
  the dates stated.
- A null result is written as the finding, with its interval — not as a failure.
- Anything skipped, capped or truncated is named. Silent truncation reads as
  "covered everything" when it did not.
- The write-up is always delivered as a **PDF** whose name states the question and
  verdict (e.g. `E131_does_litmom_help_REJECT.pdf`), never the HTML alone or a bare
  `report.md`. Author it as HTML, then `python3 -m src.analysis.html_pdf <file>.html`;
  `report_pdf` produces a board's PDF directly.
- For a feature-set / dataset-variant question, follow the
  `feature-set-evaluation` skill (factorial, placebo + tenure controls, untouched
  test block, equal-budget retune, verdict vocabulary).
