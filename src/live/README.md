# Live submission pipeline — Numerai Crypto

Three model slots, refit on the same protocol the backtest measured, submitted
every round with a fallback ladder and an email report.

## Before the first live round

**1. Create the three model slots** at [numer.ai](https://numer.ai) under the
Crypto tournament, named exactly:

```
CURUPIRA_01
CURUPIRA_02
CURUPIRA_03
```

Slots are created in the web UI — there is no API for it. Until they exist the
pipeline writes valid CSVs and reports `no such model slot on the account`.

**2. Set the email password** in `.env` (`SMTP_PASSWORD`). For Gmail this is an
App Password from *Google Account → Security → App passwords*, not the account
password. Leave it blank and the report prints to stdout instead of sending.

**3. Warm the model cache** — takes about three minutes:

```bash
python3 -m src.live.refit --all --force
```

## Every round

```bash
python3 -m src.live.run_round --deadline-minutes 60
```

Rehearse without uploading:

```bash
python3 -m src.live.run_round --dry-run
```

If the window is genuinely tight, `--safety-first` submits every slot from cache
first and only then refits and resubmits.

## Scheduling — installed

> **The submission deadline is `closeStakingTime`, not `closeTime`.**
> A round exposes four timestamps and only one is the cutoff:
> `openTime` 12:00 UTC → **`closeStakingTime` ~13:06 UTC (~66 min later)** →
> `closeTime` 12:00 UTC *the next day*, when the round stops being current →
> `scoreTime` ~6 days out. Reading `closeTime` as the deadline suggests a
> 24-hour window that does not exist, and cost two Late submissions
> (rounds 1332 and 1333). Long rounds exist — round 1334 opened Sat 12:00Z with
> a Sun 16:00Z cutoff — so the margin is read per round rather than assumed.

> **The published round table lags the submission service.** On 2026-08-17 round
> 1335 opened at 12:00Z and accepted all six slots at 12:05Z; at 12:49Z
> `rounds(tournament:12)` still topped out at 1334, `get_current_round()` still
> said 1334, and `roundDetails(1335)` answered HTTP 500. The pipeline believed
> the table, concluded the window had shut on the Sunday, and mailed
> `LATE -- submissions had already closed` about six submissions Numerai had
> accepted on time.
>
> So the round now comes from `model.latestSubmissions`, which is the submission
> service's own view and sees a round the moment it opens, with the round table
> only as a fallback. Highest round number wins — a source can lag, never lead.
> And the *verdict* is read back off the API per slot (`confirm_statuses`)
> rather than computed here: `on-time`, `queued` or `late` is Numerai's call.
> `LATE` reaches the subject line only when Numerai says so.

Rounds open at **12:00 UTC** (09:00 America/São_Paulo). Most close staking
~13:06Z the same day; some run long — the Saturday round has been closing Sunday
16:00Z, and round 1335 opened on a Monday with a 16:00Z cutoff. The cadence is
Numerai's to change, so nothing here encodes a weekday: the cron runs daily, the
window is read per round, and a run that finds nothing open exits 0 with
`no open round -- nothing to submit` rather than uploading into a closed round.

Installed in `crontab -l`:

```
MAILTO=""
5 9 * * *  /bin/bash .../src/live/cron_round.sh
25 9 * * * SKIP_IF_SUBMITTED=1 /bin/bash .../src/live/cron_round.sh
45 9 * * * SKIP_IF_SUBMITTED=1 /bin/bash .../src/live/cron_round.sh
```

| local | UTC | margin | role |
|---|---|---|---|
| 09:05 | 12:05 | ~60 min | primary |
| 09:25 | 12:25 | ~40 min | retry |
| 09:45 | 12:45 | ~20 min | last chance |

Three passes because a normal round takes 25 seconds and the window is 66
minutes — there is room for two failures. `--skip-if-submitted` makes passes 2
and 3 a single API call when pass 1 landed; if it did not, they run only the
outstanding slots and merge their results into the same round report.

## Failure handling

- **Dataset downloads retry** 4× with backoff. They come from an S3 accelerate
  host that fails independently of the GraphQL API — a DNS blip there killed
  round 1333 while the API itself was fine. `live.parquet` downloads to
  `.partial` and is moved into place only when complete, so a mid-download
  failure cannot leave the directory empty.
- **A crash before the slot loop still emails.** Anything escaping `run()`
  sends a `CRASHED -- NOTHING SUBMITTED` report with the rerun command. A silent
  miss is the worst outcome available: you find out days later from a Late flag.
- **A closed window stops the run**, but only on the submission service's word.
  Nothing is uploaded and the subject reads `no open round -- nothing to
  submit`; `--allow-late` overrides it for debugging.
- **Every upload's status is read back** from Numerai and printed per slot
  (`numerai: on-time` / `queued` / `late`). The round report carries it too. No
  local clock decides whether a round was made.
- **The round queries retry** 3× before giving up. They decide whether anything
  uploads at all, and a bare `KeyError('data')` out of numerapi took the
  2026-08-17 09:25 pass with it. If they stay unreadable the run proceeds anyway
  on `get_current_round()`: a broken API is likelier than a closed window, and a
  missed round cannot be undone.

`cron_round.sh` supplies what cron does not: the absolute conda interpreter
(`python3` on cron's PATH has none of the dependencies), the working directory,
and `flock -n -E 99` so the two runs can never overlap. Exit 99 means "another
round is still running"; exit 1 means a slot actually failed.

Log: `logs/live_round.log`. Rehearse the exact cron path with

```bash
env -i HOME=/home/danielco PATH=/usr/bin:/bin DRY_RUN=1 /bin/bash src/live/cron_round.sh
```

## What the slots are

| slot | model | feat | backtest run | payout | p.sharpe | p.maxdd |
|---|---|---|---|---|---|---|
| CURUPIRA_01 | `ridge_glmnet` | 23 | `w6r6_glmnet_reg_tenc` | 0.084132 | **1.2960** | **1.3732** |
| CURUPIRA_02 | `ridge` | 22 | `w6r6_fixed_reg` | 0.085763 | 1.0818 | 1.6557 |
| CURUPIRA_03 | `ridge_glmnet` | 786 | `w6r6_glmnet_reg_onehot_rollsel` | 0.086721 | 1.0554 | 1.8786 |
| CURUPIRA_04 | `ridge_glmnet` | 812 | `w6r6_glmnet_reg_allfeatpp` | **0.087196** | 1.0782 | 1.7246 |

> **CURUPIRA_04 is nearly the same model as CURUPIRA_03** — payout-series
> correlation **0.998**, paired difference +0.000474 at t = 0.72. Running both
> is close to running one slot twice, and the backtest says adding it *hurts*:
>
> | book | payout | p.sharpe | p.maxdd |
> |---|---|---|---|
> | 01+02+03 | 0.085391 | **1.2413** | 1.4217 |
> | 01+02+03+04 (current) | 0.085775 | 1.2157 | 1.4463 |
> | 01+02+04 (swap) | 0.085539 | **1.2481** | **1.4148** |
>
> To swap rather than add, drop `"CURUPIRA_03"` from `ORDER` in `registry.py`
> and re-solve the stake weights with `python3 -m src.live.stake --solve`.
> All three differences are well inside the noise.

All three: regression, rolling 130-period window, 130-period refit, purged CV
with a 20-period embargo, 50-draw randomised search, seed 42, over 510 scored
periods. `models/live/registry.json` is the machine-readable copy.

They were not picked as the three highest means. No arm in the study beats the
baseline by a meaningful margin — the largest paired |t| on payout is 1.30 and it
is negative — so the slots are the best risk-adjusted model plus the two least
correlated with it. Per-period score correlation between them: 01↔02 0.833,
01↔03 0.808, 02↔03 0.952.

## What is staked

Submitting and staking are different questions and `src/live/stake.py` keeps
them apart: eight slots upload every round, three carry money.

```bash
python3 -m src.live.stake            # dry plan
python3 -m src.live.stake --execute  # place it
```

| slot | weight | risk share | why it is in |
|---|---|---|---|
| CURUPIRA_01 | 0.3184 | 30% | best risk-adjusted single model; fits directly, so a member-cache failure cannot take the whole book |
| CURUPIRA_05 | 0.3432 | 33% | least correlated with the rest (0.569 with `_01`), so it buys the most diversification per unit of weight |
| CURUPIRA_08 | 0.3384 | 36% | highest mean MMC, and the only staked route to `CURUPIRA_05_60` — the one series that is *negatively* correlated with the 20-day book |

Weights come from **risk budgeting**: each slot's share of portfolio variance is
proportional to its mean MMC (`--scheme mmc`, the default). MMC carries weight
1.0 in the crypto payout against CORR's 0.1, so the tilt sizes on what pays,
while budgeting in variance rather than in weight is what makes it read the
correlations at the same time. `--scheme erc` drops the tilt; `--scheme equal`
drops both. Re-solve with `--solve` after any member run is re-scored — a test
in `test_live.py` fails if the stored weights drift from the solver.

`02`, `03`, `04` and `07` submit unstaked on purpose: `02/03/04` correlate
0.938–0.998 with each other, and `07` correlates 0.978 with `08` and has no
scored run to weight it with. On the 470 shared dates the staked book scores
payout 0.0917, sharpe **1.527**, max drawdown **0.922**, against 0.0874 / 1.209
/ 1.446 for the old `01+02+03+04` allocation — sharpe +0.30 with a bootstrap CI
of +0.02 to +0.63, while the payout gain is *not* significant (P = 0.73). It is
a risk trade, not a return trade.

> Staking a blend is not a third position. `_08` is a rank average of `_04`,
> `_05` and `_05_60`, so the three staked slots are really **46% `_05`, 32%
> `_01`, 11% `_04`, 11% `_05_60`** — the plan prints that expansion every run.

Sizing: the book's cumulative-payout drawdown is 0.92 **in units of stake**, and
essentially all of it arrives inside one 20-day window. Stake only capital you
would accept losing in full. (Those series carry no payout factor and no
per-round clipping, so 0.92 is an upper bound on the real NMR path rather than a
forecast; the sharpe ranking is unaffected.)

## Fallback ladder

Per slot, first success wins, recorded in the email:

1. **own model** — cached estimator on its own feature set
2. **sibling** — `CURUPIRA_02` on the raw 22 features; the only slot with no
   build step, so if the features loaded at all this runs
3. **previous round** — the last CSV this slot submitted (consecutive dates
   overlap 95% on a 20-day target, so a one-round-stale signal is still a signal)
4. **example preds** — Numerai's own; scores ~0, but keeps the slot alive

Uploads retry 5 times with exponential backoff (~75s worst case).

## Timing

Measured on deepsix, 39,000 training rows:

| step | cost |
|---|---|
| load panel + build features | ~25 s, shared |
| refit CURUPIRA_02 / 01 | ~55 s each |
| refit CURUPIRA_03 (786 feat) | ~75 s |
| predict + write + upload | ~2 s per slot |

Worst case, all three refitting: about 4 minutes. Refits only happen every 130
periods, so most rounds cost seconds. A refit starts only if the remaining
budget covers its estimated cost ×1.5 plus 90 s reserve per outstanding slot;
otherwise the cached fit is used and the report says so.

## Known caveats

- **CURUPIRA_03 cold start.** 73 of 300 live symbols (24%) fall outside the
  one-hot vocabulary and are predicted from raw features alone. This matches the
  21–26% the backtest measured per refit.
- **train/live gap.** `train.parquet` is downloaded every round, so the rolling
  window actually rolls. It still lags live -- Numerai cannot publish a row until
  its 20-day target resolves, which is ~30 days -- and each round's live rows are
  appended to `data/crypto/live/live_history.parquet` so rolling windows close
  the rest of the gap over time.

  It was NOT downloaded at all until 2026-08-23, and that was the single largest
  protocol divergence in the pipeline: every cached fit shared one training block
  ending 2026-07-08 whatever day it was fitted, and the live embargo had drifted
  to 32 periods against a backtested 20, widening by one every round. Refreshing
  it cut the gap to 22 periods at once. Note the gap is ~30 CALENDAR days but 22
  PERIODS -- the grid is weekdays only, and the two are not interchangeable.
- **MMC coverage.** All backtest MMC figures rest on the 32.9% of rows the
  crypto meta model covers.

## Files

| file | role |
|---|---|
| `registry.py` | slot → configuration, the single source of truth |
| `features.py` | live feature builds, sharing code with the offline builders |
| `refit.py` | protocol-faithful refit + model cache |
| `submit.py` | rank-normalise, validate, upload with retry, fallback ladder |
| `notify.py` | email report |
| `run_round.py` | deadline-aware orchestrator |
| `env.py` | `.env` loader |

Tests: `python3 -m pytest src/tests/test_live.py -q` (38 tests, ~35 s).
