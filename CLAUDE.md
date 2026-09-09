# CLAUDE.md

Conventions for working in this repo. The pipeline is `src/`.

## What is running right now

`results/experiments.db` -- a SQLite index of every numbered study. **Read it
before answering "what is running", "did E0x finish" or "what did we try".**

```bash
python3 -m src.analysis.experiment_db                 # state of every study + its idea
python3 -m src.analysis.experiment_db --arms E09      # per-arm detail for one study
python3 -m src.analysis.experiment_db --sql "SELECT ..."
```

It refreshes before every query, so the answer is never stale. Status is
**derived, never recorded**: arm counts come from `results/metrics/`, and
"running" comes from a live runner process in `/proc`. There is deliberately no
"mark this done" verb -- a hand-maintained status field is wrong the first time
someone forgets, and then it is worse than nothing because it still looks
authoritative.

The intent -- id, title, the idea in a sentence, the runner, the prefixes --
lives in `STUDIES` at the top of `src/analysis/experiment_db.py`. That is the
tracked source of truth and the only part edited by hand; **add an entry there
before a study launches**, alongside its row in `experiments/EXPERIMENTS.md`.
The .db itself is under a gitignored directory and is disposable.

Which of the two to read depends on the question. `EXPERIMENTS.md` is the
durable record and holds the FINDINGS, written once a study is over. The
database is the operational view and holds where things GOT TO. Neither replaces
the other.

One trap: the `payout` column there is `0.1*corr + mmc` rebuilt from the two
stored means. That is fine for ranking a list and wrong for a claim -- see the
next section.

## Reporting results

Reporting a set of results means producing **a document**, not just a table.
Two renderers, same numbers, same flags:

```bash
python3 -m src.analysis.report_html --model lgb --runs <run> [<run> ...]
```

```bash
python3 -m src.analysis.report_pdf --model lgb --runs <run> [<run> ...]
```

Both write into `results/reports/crypto/` and **print the same table to stdout,
from the same rows**. Both carry the ranked table, both correlation matrices,
screened ensembles with the commands to price them, and cumulative backtested
payout of the proposed runs against the best single submitted model. Quote the
table in chat; attach the file.

**Reach for the HTML by default** -- it is one self-contained file with no
network dependency, and it is the one you can interrogate: sort any column, hide
columns, filter rows, narrow the date range, toggle series in the chart. The PDF
is the archival form, for mailing and for reading in five years.

The HTML embeds one matrix -- every arm's per-period `corr` and `mmc` on the
block's common dates -- and recomputes every figure in the browser, so narrowing
the range re-ranks the table, re-screens the ensembles and re-bases the curves
together. Every section is also written out server-side, so the numbers are
there before any script runs; the script replaces them to make them sortable.
Renders are coalesced, and the ensemble screen is capped at `--ens-max-arms`
(12) because combinations grow as the cube of the arm count -- 37 arms is 8,436
of them to display 32. The one exception is the PREDICTION correlation matrix, which needs
the row-level forecasts; it is computed over the full window and says so. Since
that puts the same arithmetic in Python and JavaScript,
`src/tests/test_report_html.py` runs the embedded script under node and asserts
it reproduces `report_data` to 1e-9.

Every deployed slot is included by default, so a proposed run is always ranked
against the book rather than against its own study. The tool splits runs into one
block per scored target, puts every row in a block on one common window, and
names anything it left out.

`src/analysis/report_data.py` holds the arithmetic both renderers read; neither
computes anything of its own. For a table and nothing else, `results_table` is
still the right tool and produces the identical rows:

```bash
python3 -m src.analysis.results_table --model lstm --runs <run> [<run> ...] --sort-by payout
```

Runs in another model tree are addressed `model/run` (e.g. `ridge/w6r6_fixed_reg`),
the same convention `paired_delta.py` uses.

Columns, in this order:

| column | meaning |
|---|---|
| `CORR` | mean per-period `numerai_corr` |
| `MMC` | mean per-period contribution over the real meta model |
| `c.sharpe` | CORR mean / std (ddof=0) |
| `dd(corr)` | max drawdown of the cumulative CORR series, POSITIVE (bigger is worse) |
| `payout` | mean per-period `0.1 * corr + mmc` |
| `p.sharpe` | payout mean / std (ddof=0) |
| `dd(pay)` | max drawdown of the cumulative payout series |
| `calmar` | payout mean / `dd(pay)` |
| `burn(d)` | count of periods whose payout was negative — downside FREQUENCY, POSITIVE (bigger is worse) |
| `tgt` | the target the run was SCORED on, read from the forecast header |
| `n` | periods behind the row (and `n(own)` in the PDF, its native length) |

The PDF adds `kind` (`prop` proposed / `sub` submitted single / `blend` /
`slotblend`, `$` = staked), `dpay`/`t` (paired Newey-West delta against the
block's baseline) and `live` (what the slot actually submitted last round).

### Three things that are easy to get wrong here

**Payout is rebuilt PER PERIOD, never from the stored summaries.** The metrics json
summarises the CORR series and the MMC series separately. A sum's Sharpe is not any
function of its parts' Sharpes -- corr and mmc are correlated -- so combining the two
summaries produces a plausible-looking wrong number. Form the payout series first,
one value per period, then summarise once through `prediction.metrics.summarize` so
ddof and drawdown-sign conventions match everything else in the repo.

**`calmar` here is not the textbook Calmar.** It is the per-period mean payout over
the max peak-to-trough fall of the CUMULATIVE payout, in score units rather than
percent. The numerator is a per-period mean while the denominator grows with series
length, so **it is only comparable across runs with the same period count**. Runs in
`results/` have 510, 470, 253, 128 and 64 periods. Print `n` alongside it whenever
rows of different lengths appear in one table.

**`burn(d)` and `calmar` split downside into frequency and depth, and both are
window-relative.** `dd(pay)` is the deepest single fall; `burn(d)` is how MANY periods
came out red. Two runs at the same `dd(pay)` can burn very different day counts. Because
it is a raw COUNT of periods, it is comparable only across rows on the same window —
exactly the `calmar`/`n` caveat — so slice to a common window first and read it with `n`
(or the rate `burn(d)/n` when windows differ). One caveat to state: the target is a
20-day forward return, so consecutive periods overlap ~95% — the count measures time
spent under water, not that many independent losing bets. It comes out of
`results_table.burn_days`, so a table and a report cannot disagree on it.

**Never mix targets in one table.** CORR on `target_binned_return_60` is structurally
different from the 20-day label, so a mixed table ranks the LABEL as much as the
model. Split by target or carry a `tgt` column. Read the target from the FORECAST
header, not from `config["target"]`: `run_ensemble` writes no target into a blend's
config, so a config lookup silently reports every 60-day blend as 20-day. That is
what `results_table.target_of` is for, and 59 runs in `results/` are affected.

Also: name anything excluded. A table that silently drops an arm reads as "this is
everything" when it is not.

## The submitted models

18 slots exist on the Numerai account (query with `CryptoAPI.get_models()` --
do not trust this list, it changes):

```
curupira_01  curupira_02  curupira_03  curupira_04  curupira_05
curupira_06  curupira_07  curupira_08  curupira_iara_01
curupira_saci_01  curupira_saci_02  curupira_saci_03  curupira_saci_04
saci_01  saci_02  saci_03  saci_04  saci_05
```

`src/live/registry.py` defines **20** slots, and they are not the same set:

- `CURUPIRA_05_60` and `IA_01` are in `registry.NOT_SUBMITTED` and have no account
  slot. They are the two true components -- fitted so other slots can rank-average
  them, never uploaded alone. `CURUPIRA_05_60` feeds `CURUPIRA_07`, `_08`,
  `CURUPIRA_SACI_01` and `_03`; `IA_01` feeds `CU_SA_IA_BO_01`.
- `CURUPIRA_IARA_01` is on the account and in no `ORDER`, so it receives nothing.
  `run_round` now warns about that; it did not until 2026-08-23, which is why it
  went unnoticed the day it was created.
- `CU_SA_IA_BO_01` is the newest slot and has NOT yet appeared in a round report:
  round 1340's lists 18 entries and does not include it. It IS in `ORDER`, so the
  next round should pick it up -- confirm against
  `results/live/<round>/round_report.json` rather than assuming it went out.

**`CU_SA_IA_BO_01` is the first slot to mix the tabular book with a neural model.**
A 50/50 rank average of `CURUPIRA_SACI_03` and `IA_01`, where `IA_01` is a plain
feedforward MLP on the 22 raw features plus their 22 CAUSAL per-symbol means
(`fenccausal`, 44 columns) -- an expanding mean of each symbol's own past, so a row
never enters its own encoding.

Against `CURUPIRA_SACI_03` alone, paired Newey-West at lag 20 over 470 periods:
CORR +0.010429 (t +2.83), MMC +0.005178 (t +1.58), payout +0.006221 (t +1.72).
**Only the CORR gain clears the house |t| >= 2 bar.** Deploy it for the correlation
and the MMC, not on the expectation that the payout edge is established. It buys
level rather than stability -- dd(pay) 1.16 against `CURUPIRA_SACI_03`'s 1.03, and
calmar 0.093 against 0.098.

Two traps. The member choice is MEASURED, not assumed: the same blend built with
the raw-feature MLP instead scores payout 0.09975, i.e. worse than
`CURUPIRA_SACI_03` alone (t -0.39), so the causal encoding is what makes the slot
work -- do not substitute a different MLP into it. And it inherits
`CURUPIRA_SACI_03`, which has been falling through to Numerai's EXAMPLE
predictions in recent rounds; if that is still true this slot inherits the same
fallback, so check the round report's source before trusting a live score.

`IA_01` is also the reason the torch LSTM is absent from this registry: the MLP
beats it outright on identical data, CORR 0.15812 against 0.10495 on the raw 22.
Caveat both numbers -- the `mlp` family has four runs in a book of 500+, and its
0.16172 CORR is one configuration never replicated across seeds. See
`results/reports/crypto/lstm_investigation.pdf` and the next section.

**`SACI_03`, `SACI_04` and `SACI_05` are dual-role.** They are submitted models
in their own right AND members of `CURUPIRA_SACI_02`, `_03` and `_04`
respectively. That is not a contradiction, and it has a consequence: a bad fit on
one of them does not stay put, it takes a blend down with it. The round email
ranks a dual-role slot's failure above a leaf slot's for that reason.

**`SACI_03` fits the 20-day label but was backtested on the 60-day one.** Its
`fixed_params` come from a `target_binned_return_60` study and are kept, because
they say which shape of booster suits the feature set; but Numerai pays on
`target_binned_return_20`, so that is what it fits. Ask
`registry.backtest_target(name)` -- never `fit["target"]` -- when you want to know
which label a slot's *numbers* came from. Its 0.093018 does not describe the live
slot, and no re-score exists for it.

Only 3 of the account's slots carry stake, so a comparison aimed at deployment
should say which side of that line a model falls on. Read the names from
`src.live.stake.STAKED` rather than repeating amounts here -- the NMR figures
move whenever stake does, and a hardcoded list goes stale without failing.

### Live state is dynamic -- read it, do not memorise it

Which slots are on the account and which fell through to a fallback changes from
round to round, so both come from the newest `results/live/<round>/round_report.json`
via `src.analysis.slots.live_state()`. The report's `live` column prints it:
`ok`, `prev!` (reused last round's predictions), `example!` (fell all the way
through to Numerai's own), `failed!`, `not-sub`. Check the column, not this
paragraph.

Rounds 1338 and 1339 had `CURUPIRA_SACI_03` and `_04` on the benchmark. Both
causes are fixed: `_03` because `load_slot_blend` could not assemble a slot blend
nested inside another one (`CURUPIRA_08`), and `_04` because `allfeat_pp` does not
contain `cumlag3`'s lag columns, which its member `SACI_05` needs. Note the round
report blamed all three of `_03`'s members; only `CURUPIRA_08` ever failed, and
the note now lists the real ones.

`models/live/registry.json` is generated from `src/live/registry.py` and can go
stale; it has held 8 entries against the source's 18. Edit the `.py`.

### Two slots whose recorded figure is not their series

`src/analysis/slots.py` resolves every slot to a scored series and self-checks
against the registry: **16 of 18 reproduce `backtest.payout` to 1e-6.** The two
that do not are deliberate, and the report footnotes both:

- `CURUPIRA_05_60` -- registry 0.074914 is its predictions RE-SCORED on the payable
  20-day label, out of `results/cache/cross_target_payout.csv`. Its stored series
  (`t60_c05_blend`) is on the 60-day label and pays 0.092201. Same slot, two scales.
- `CURUPIRA_07` -- its `backtest_run` is the string `"(screened, not run as a single
  arm)"`, and 0.084782 is a payout-SERIES average. The prediction-level blend exists
  as `ridge_glmnet/xae_c07_base` and pays 0.098594.

`SACI_03` and `CURUPIRA_05_60` are the only 60-day-scored slots; every other slot
is on the 20-day label, including the 470-period blends that hold a 60-day member.

## The LSTM cannot run live

`src/prediction/nn/` (torch LSTM) is deliberately absent from
`src.prediction.models.MODELS`, which is the sklearn-wrapper contract `src/live/`
consumes via `build_wrapper` + `joblib.dump(search.best_estimator_)`. Two things
must change before any LSTM or LSTM blend can be submitted:

1. A live fit/predict path for torch, assembling per-symbol sequences.
2. `data/crypto/live/live_history.parquet` must reach `seq_length` dates. It grows
   one date per round via `append_live_history`, and held 7 as of 2026-08-20.

Backtest artifacts under `results/forecast/` are out-of-sample predictions for
historical dates and must never be uploaded to a live round.
