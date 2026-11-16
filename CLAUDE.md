# CLAUDE.md

Conventions for working in this repo. The pipeline is `src/`.

## Keep Claude Code and Codex in sync

When the user asks to add repository context or skills, make equivalent
additions for both agents in the same change. Put context in both
`CLAUDE.md` and `AGENTS.md`, and add matching skills under
`.claude/skills/` and `.agents/skills/`. Use each agent's own paths and
invocation syntax, then verify that both versions give the same guidance.
If the user explicitly limits the request to one agent, follow that scope.

## Commit and push repository changes

After verifying each requested repository change, commit it and push it to the
branch's configured remote. Stage only files that belong to the requested
work. If the user explicitly asks to keep a change local, follow that scope.
Report any commit or push failure.

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

## The machines

Fourteen hosts are declared in `~/.ssh/config`, every one of them
`<name>.vision.ime.usp.br` as `danielco` with `~/.ssh/id_vision` and
`ControlPersist 10m`, so the second ssh to a box costs nothing. **A session
usually runs ON `deepfive`, but check `hostname` instead of assuming** -- which
box a session sits on changes, and so does the answer to "is there a GPU here".

`/home` is a single NFS export (`homefs:/mnt/pool01-data/home`) mounted by all
of them, so there is nothing to copy and nothing to sync: one checkout, one
`/home/danielco/miniconda3` (3.12, the interpreter every runner uses), one
`results/`. Every box is x86_64, so that one conda install is valid on all of
them. It also means a job on one box writes the `results/` another box reads,
and that all of them share a single disk quota.

Reachable on 2026-10-23:

| host | cores | RAM | GPU |
|---|---|---|---|
| `hamilton` | 64 | 251 G | -- |
| `hopper` | 64 | 251 G | -- |
| `lovelace` | 64 | 251 G | -- |
| `deepfive` | 32 | 251 G | **2x RTX 2070, 8 G each** |
| `deepfour` | 32 | 251 G | cards present, driver unreachable |
| `deepsix` | 32 | 251 G | -- |
| `tolstoi` | 32 | 251 G | -- |
| `dostoievski` | 32 | 251 G | -- |

`hamilton`, `hopper` and `lovelace` are the widest at 64 cores -- the place for
a 100-seed sweep or a wide `n_jobs` search. **`deepfive` is the only box with a
working GPU**, so every torch and TabPFN fit goes there. `deepfour` has the
hardware but `nvidia-smi` cannot reach the driver, which makes it a CPU box in
practice; do not send a GPU job there expecting it to fall back gracefully.

Not reachable, same date: `puchkin`, `deepzero`, `deepone`, `deeptwo` and
`deepthree` give "no route to host". `curie` fails differently -- it answers on
22 and rejects the key ("Permission denied (publickey)"), so it is up but not
ours. Note the spellings: the hosts are `puchkin` and `dostoievski`, not
*puchki* or *dostoievsky*. Recount rather than trusting the table:

```bash
for h in hamilton hopper lovelace curie puchkin dostoievski tolstoi \
         deepzero deepone deeptwo deepthree deepfour deepfive deepsix; do
  printf '%-12s ' "$h"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" 'printf "%s cores  " "$(nproc)";
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, - \
    || echo "no gpu"' 2>/dev/null || echo unreachable
done
```

### Three traps that follow from the shared home

**`experiment_db` cannot see the other boxes.** `running_studies()` reads the
LOCAL `/proc` -- deliberately, so status stays derived and needs no shell -- so
a study running anywhere else reads as `partial`, which is indistinguishable
from one that died half-finished. On 2026-10-23, E116 was 33 minutes into a run
on `deepsix` while the local process table on `deepfive` was empty. Before
concluding a study stalled, or reclaiming its forecasts, ask every box:

```bash
for h in hamilton hopper lovelace tolstoi dostoievski deepfour deepfive deepsix; do
  printf '=== %s\n' "$h"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
    "ps -u danielco -o etime,cmd | grep -E 'run_forecast|run_nn|experiments/e[0-9]' \
     | grep -v grep" 2>/dev/null
done
```

**`df` lies about free space.** `df -h /home` reports the pool -- 31 T, three
quarters empty -- while a per-user quota around 100 G is what actually stops a
write. Nothing in `df` surfaces it, so a job dies with ENOSPC on a filesystem
that looks idle, and 24 GPU runs died silently that way on 2026-10-17. To get
the real headroom, write incompressible bytes and find where it fails
(`dd if=/dev/urandom of=probe bs=1M count=N`, doubling N) -- the pool is
compressed, so a text file's apparent size is not what it occupies, and `du`
on logs can overstate the reclaimable bytes by more than 10x.

**Never `pkill -f <pattern>` over ssh when the pattern matches your own command
line.** The ssh invocation carries the pattern, so it matches itself and the
connection dies mid-kill, with no way to tell which targets were reached.
Launch long batches with `systemd-run --user` so they survive the session, and
kill them by unit or by PID read from `ps` first.

### Spreading a batch over them

`src/cluster` places jobs on the free boxes and takes at most 30% of any one:

```bash
python3 -m src.cluster probe                              # what is free now
python3 -m src.cluster plan --jobs experiments/cluster_jobs.example
python3 -m src.cluster run  --jobs <file>                 # launch
python3 -m src.cluster status                             # our units, everywhere
python3 -m src.cluster stop --all
```

The cap is held where it CAN be held: each job is pinned to its own cpu set
with `taskset`, so it sees exactly its slot -- `nproc` inside a 8-cpu slot on a
64-core box reports 8, so an `n_jobs=-1` cannot escape it -- and the
OMP/MKL/OPENBLAS/NUMEXPR thread counts are set to the slot size so a search
does not open 64 threads onto 8 cpus. RAM is admission control only. **Do not
hand resource limits to systemd on these hosts**: they run cgroup v1
(`/sys/fs/cgroup` is tmpfs, systemd 245), where a `--user` unit's `MemoryMax`
and `CPUQuota` are recorded and then ignored. Measured, not assumed -- 4 busy
loops under `CPUQuota=100%` took 391% between them, four times the cap, while
systemd reported the quota as applied; 8 loops pinned to 2 cpus with taskset
measured 194%.

A host whose load already exceeds its cores offers nothing, so a box someone
else is hammering is skipped rather than made worse: on 2026-10-23 deepsix
(load 44) and deepfour (load 32) correctly took no work while 84 cores sat free
elsewhere. A job NOT launched through the manager counts as somebody else's
load, which is deliberate -- conservative is the safe direction when the
alternative is oversubscribing a machine in use.

GPU work needs `--gpu-slots 1`. floor(0.30 x 2 cards) is 0, so the strict share
forbids it outright, and one of two cards is 50% -- an exception taken on
purpose rather than a 30% promise rounded up. Jobs run as `nmr-<name>` units so
`stop` can address them by name, and the command itself goes in
`logs/cluster/<name>.cmd.sh` rather than on the unit's command line, because
systemd expands `$` there and collapses `$$` to `$` before bash sees it.

There is no queue and no dependency graph: `run` launches what fits now and
prints what did not.

## Numerai Crypto missing values

Before creating, extending, rebuilding, testing or evaluating a Numerai Crypto
dataset with potentially missing candidate features, read and follow
[the missing-value implementation skill](.claude/skills/numerai-crypto-missing-values/SKILL.md).
Before accepting or interpreting the experiment, apply
[the missing-value audit skill](.claude/skills/numerai-crypto-missing-values-audit/SKILL.md).

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

**Always deliver the PDF.** The HTML is a working view you can interrogate -- sort,
hide columns, filter rows, narrow the date range -- but the report you send is the
PDF. A study write-up authored as HTML is printed with
`python3 -m src.analysis.html_pdf <file>.html`, which writes the PDF beside it.

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

## Diagnosing neural-network training

A curve of the metric alone is not a diagnosis. Any torch network that is trained,
compared or debugged here gets the panel of the `training-diagnostics` skill first --
loss split into its data and L2-penalty parts, fit beyond predicting the mean,
train and held-out CORR/MMC, gradients (data vs penalty, per layer), held-out
prediction dynamics, learning rate and weight norms -- and the reading comes before
the explanation and the fix:

```bash
python -m src.prediction.diagnose --tag <tag> --seed 42 --refit 0 <recipe flags>
python -m src.analysis.plot_training_diagnostics <tag>
```

The driver reproduces `TorchMLPRegressor` bit for bit and the E105 runs to scorer
noise; other loops call `src.prediction.training_diagnostics.Snapshotter` directly.
Output goes to `results/diag/crypto/<tag>/`. A panel explains one run; ranking
recipes still needs the multi-seed paired comparison (E104/E105 used 100 seeds).

A walk-forward fit has one trajectory per refit, so it is reported at the procedure
level first. Run every `--refit` with `--save-preds --no-train-preds --predict-next`, run a second seed
for the churn floor, then call
`python -m src.analysis.walkforward_diagnostics --base <base> --floor-base <base2> --html`.

## The submitted models

Four sets in `src/live/registry.py` decide what a slot is, and they are not the
same set. The counts below were read from the code and the account on 2026-10-22
(round 1382). Slots get added and renamed every few weeks, so recount instead of
trusting a number here:

```bash
python3 -c "from src.live import registry as R; print(len(R.MODELS), len(R.ORDER), len(R.NOT_SUBMITTED), sorted(set(R.ORDER) - R.NOT_SUBMITTED))"
```

| set | what it is | 2026-10-22 |
|---|---|---|
| `registry.MODELS` | every named slot, with its spec and backtest | 40 |
| `registry.ORDER` | fitted every round, in this order (cheapest first, slot blends after their members) | 37 |
| `ORDER` minus `NOT_SUBMITTED` | uploaded every round | 25 |
| `ORDER` and `NOT_SUBMITTED` | components: fitted and cached so a blend can reuse them, never uploaded | 12 |
| `registry.BACKTEST_ONLY` | backtested, no live path, in no `ORDER` (also in `NOT_SUBMITTED`) | 3 |

The account holds exactly the 25 uploaded slots. On 2026-10-22,
`CryptoAPI.get_models()` returned these and nothing else:

```
curupira_01  curupira_04  curupira_05  curupira_06  curupira_07  curupira_08
curupira_boitata_01  curupira_caipora_01
curupira_saci_01  curupira_saci_02  curupira_saci_03  curupira_saci_04
cu_ca_ia_01  cu_r_ia_ma_60_01  cu_sa_ia_bo_01  cu_sa_ia_bo_02
cu_sa_ia_bo_ma_01  cu_sa_ia_bo_ma_r_01  iu_ia_01
saci_01  saci_02  saci_03  saci_04  saci_05  saci_06
```

`run_round` checks both directions every round, so drift shows up in the round
report's warnings before it shows up here. A slot in `ORDER` with no account slot
"cannot be submitted". An account slot in no `ORDER` "receive[s] no predictions at
all". The second check exists because `CURUPIRA_IARA_01` sat on the account from
2026-08-22 receiving nothing, unnoticed. That idle slot was later renamed `iu_ia_01`
and now carries `IU_IA_01`.

The 12 components, and the slots that list each one in `slot_members` (direct
members only, while nesting reaches further: `CURUPIRA_05_60` is inside
`CURUPIRA_08`, which is inside `CURUPIRA_SACI_03`, which is inside
`CU_SA_IA_BO_01`):

| component | model, features | rank-averaged by |
|---|---|---|
| `CURUPIRA_02` | ridge, raw | none -- it is `submit.SIBLING`, the fallback rung for every other slot |
| `CURUPIRA_03` | ridge_glmnet, onehot_roll | `CURUPIRA_SACI_01` |
| `CURUPIRA_05_60` | ridge_glmnet 5-block blend, allfeat_pp, FITS the 60-day label | `CURUPIRA_07`, `_08`, `CURUPIRA_SACI_01`, `CU_SA_IA_BO_02`, `CU_CA_IA_01`, `CU_R_IA_MA_60_01` |
| `IA_01` | mlp, fenccausal | `CU_SA_IA_BO_01`, `CU_R_IA_MA_60_01` |
| `IA_02` | mlp, fenccausal, config pinned | `CU_SA_IA_BO_02`, `CU_CA_IA_01` |
| `IA_03`, `IA_04` | mlp, tenc / interact, config pinned | `IU_IA_01` |
| `IAUARETE_01` | catboost_ktz, fenccausal | `CURUPIRA_CAIPORA_01`, `CU_CA_IA_01`, `IU_IA_01` |
| `IAUARETE_02` | catboost_ktz, allfeat | `IU_IA_01` |
| `MA_FENC` | rf_ktz, fenccausal | `MAPINGUARI_01`, `CU_R_IA_MA_60_01` |
| `MA_ALLFEAT` | rf_ktz, allfeat | `MAPINGUARI_01` |
| `MAPINGUARI_01` | rank average of `MA_FENC` + `MA_ALLFEAT` | `CU_SA_IA_BO_MA_01` |

The three `BACKTEST_ONLY` slots are the TabPFN ones: `TUPA_01`, `TUPA_02` and
`TU_CU_SA_IA_BO_CA_01`. They are named so reports and `slots.slot_series`
resolve them. None goes into `ORDER` until a GPU tabpfn fit / predict / joblib
round trip has been run through `src/live/refit.py`.

On 2026-10-21 the account slot `curupira_02` was renamed `cu_sa_ia_bo_ma_r_01` and
`curupira_03` was renamed `cu_r_ia_ma_60_01`. These are the two E107
meta-residualised slot blends. `CURUPIRA_02` and `CURUPIRA_03` are now in
`registry.NOT_SUBMITTED` but stay in `ORDER`, because both are still needed:
`CURUPIRA_02` is `submit.SIBLING`, the fallback rung for every other slot, and
`CURUPIRA_03` is a member of `CURUPIRA_SACI_01`. A `slot_members` entry carrying
`"resid": 0.5` enters its blend residualised on the meta model lagged 5 days
(`src.prediction.meta_resid`, wrapped live by `blend.MetaResidual`). `run_round`
downloads `historical_meta_models.parquet` every round. Without a snapshot no
older than 14 days, those legs pass through un-residualised and the round report
warns; check `meta_resid` in `round_report.json` and each slot's `resid` list.

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
work -- do not substitute a different MLP into it. And it nests
`CURUPIRA_SACI_03` -> `CURUPIRA_08`, so a fallback anywhere down that chain
reaches it too. Check each one's `source` in the round report before trusting a
live score.

`IA_01` is also the reason the torch LSTM is absent from this registry: the MLP
beats it outright on identical data, CORR 0.15812 against 0.10495 on the raw 22.
Caveat both numbers -- the `mlp` family has four runs in a book of 500+, and its
0.16172 CORR is one configuration never replicated across seeds. See
`results/reports/crypto/lstm_investigation.pdf` and the next section.

**Almost half the uploaded slots are dual-role**: submitted in their own right
AND rank-averaged by another slot. `sorted(registry.reused_slots() -
registry.NOT_SUBMITTED)` lists them, 12 of the 25 on 2026-10-22. The `SACI_`
leaves feed the `CURUPIRA_SACI_` blends (`SACI_01` -> `_01` and `_02`, `SACI_03`
-> `_02`, `SACI_04` -> `_03`, `SACI_05` -> `_04`). The other eight --
`CURUPIRA_01`, `_04`, `_05`, `_08`, `CURUPIRA_SACI_03`, `CU_SA_IA_BO_01`, `_02`
and `_MA_01` -- are nested by the later blends. That is not a contradiction, and
it has a consequence: a bad fit on one of them does not stay put, it takes every
blend above it down too. The round email ranks a dual-role slot's failure above
a leaf slot's for that reason (`notify.py` reads `reused_slots()`).

**`SACI_03` fits the 20-day label but was backtested on the 60-day one.** Its
`fixed_params` come from a `target_binned_return_60` study and are kept, because
they say which shape of booster suits the feature set; but Numerai pays on
`target_binned_return_20`, so that is what it fits. Ask
`registry.backtest_target(name)` -- never `fit["target"]` -- when you want to know
which label a slot's *numbers* came from. Its 0.093018 does not describe the live
slot, and no re-score exists for it.

Only the slots in `src.live.stake.STAKED` carry stake, a small minority of the
account, so a comparison aimed at deployment should say which side of that line
a model falls on. Read the names from there rather than repeating them or their
amounts here -- both move whenever stake does, and a hardcoded list goes stale
without failing.

### Live state is dynamic -- read it, do not memorise it

Which slots are on the account and which fell through to a fallback changes from
round to round, so both come from the newest `results/live/<round>/round_report.json`
via `src.analysis.slots.live_state()`. The report's `live` column prints it:
`ok`, `prev!` (reused last round's predictions), `example!` (fell all the way
through to Numerai's own), `failed!`, `not-sub`. Check the column, not this
paragraph.

Rounds 1368-1380 (2026-10-01 to 10-20) had 20 of 25 submitted slots, all three
staked ones included, uploading the `CURUPIRA_02` SIBLING. The cause was
`causal_target_encoding`, which summed NaN targets into its prefix sums. Once
`train.parquet` lagged live by more than the 20-period lag (32 days, and Numerai
serves it that way), every live row's lookback reached unresolved live-history
rows, so `add_tenc` raised for every tenc-bearing feature set. This was fixed on
2026-10-21 by counting a NaN target as no observation. Research encodings are
byte-identical, since the train file has no NaN target. Scores from those 13
rounds describe `CURUPIRA_02`, not the slot named on them.

A second fault was hiding behind the first. When a token crosses
`ONEHOT_MIN_DATES`, the one-hot vocabulary grows, and every one-hot cache's
signature goes stale. The stale-cache rung then handed the old fit the new,
wider column list. That was certain to fail: LightGBM refused 777 columns for a
fit made on 776. `load_blend` rejected the whole blend outright, so a skipped
`CURUPIRA_05_60` refit took down the 12 slot blends that contain it.
`refit.usable_columns` now predicts a cached fit on the columns it was fitted
on, which are always still present because the vocabulary only grows.

Rounds 1338 and 1339 had `CURUPIRA_SACI_03` and `_04` on the benchmark. Both
causes are fixed: `_03` because `load_slot_blend` could not assemble a slot blend
nested inside another one (`CURUPIRA_08`), and `_04` because `allfeat_pp` does not
contain `cumlag3`'s lag columns, which its member `SACI_05` needs. Note the round
report blamed all three of `_03`'s members; only `CURUPIRA_08` ever failed, and
the note now lists the real ones.

`models/live/registry.json` is generated from `src/live/registry.py`, holds
`ORDER` only, and is rewritten only by `python3 -m src.live.registry --write`.
Nothing in a round calls that, so the file goes stale; it once held 8 entries
against the source's 18. Edit the `.py`.

### Slots whose recorded figure is not their series

`src/analysis/slots.py` resolves every slot in `ORDER` to a scored series and
self-checks against the registry (`python3 -m src.analysis.slots`). On
2026-10-22, **34 of 37 reproduce `backtest.payout` to 1e-6.** Two of the three
that do not are deliberate, and the report footnotes both:

- `CURUPIRA_05_60` -- registry 0.074914 is its predictions RE-SCORED on the payable
  20-day label, out of `results/cache/cross_target_payout.csv`. Its stored series
  (`t60_c05_blend`) is on the 60-day label and pays 0.092201. Same slot, two scales.
- `CURUPIRA_07` -- its `backtest_run` is the string `"(screened, not run as a single
  arm)"`, and 0.084782 is a payout-SERIES average. The prediction-level blend exists
  as `ridge_glmnet/xae_c07_base` and pays 0.098594.

The third, `IA_03`, is not deliberate. Its registry block was typed in rounded
(payout 0.088000 against the series' 0.087997, sharpe and drawdown to two
decimals). The self-check's note calls it a 20-day re-score, which it is not:
that note is the fallback text for any mismatch without a reason of its own.

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

## MLP training compute

For Timmermann–Vulicevic epoch schedules, FLOP budgets, and compute scaling,
use [`/timmermann-mlp-compute`](.claude/skills/timmermann-mlp-compute/SKILL.md).
