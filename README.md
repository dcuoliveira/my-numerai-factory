# Numerai Crypto research pipeline

A walk-forward research and live-submission pipeline for the
[Numerai Crypto tournament](https://numer.ai/): dataset builders, a model zoo behind one
sklearn-wrapper contract, era-aware cross-validation, CORR/MMC scoring, and a live round
runner that fills the submission slots every morning from cron.

Derived from [numerai/example-scripts](https://github.com/numerai/example-scripts).
Need help? [Find us on Discord.](https://discord.gg/numerai)

Read [CLAUDE.md](CLAUDE.md) for the working conventions: where results live, how to report
them, which slots are deployed, and the traps worth knowing before you quote a number.

## Setup

`setup.sh` is the installer — there is no `requirements.txt`. It installs the packages
`src/` imports, downloads the base dataset, and runs a smoke fit.

```bash
bash setup.sh                 # base install + crypto/v2.0
bash setup.sh --with-derived  # ...plus the derived feature chain (target encoding ->
                              #    one-hot -> rolling means -> spreads -> interactions)
bash setup.sh --all           # ...plus the Numerai Classic datasets
bash setup.sh --check         # verify an existing install, change nothing
```

Skills live in `.claude/skills/` and Claude Code picks them up automatically.

## Run

One walk-forward backtest, written to `results/{forecast,per_period,metrics,hyperparams}/`:

```bash
python3 -m src.run_forecast \
    --tournament crypto --model lgb --data-version crypto/v2.0 \
    --cv-scheme rolling --train-window 6month --refit-every 6month \
    --n-iter 50 --n-splits 5 --seed 42 --run-name my_first_run
```

## Experiments

One shell script per study in `experiments/`, each documenting in its own header what it
runs, what it is measured against, and what would count as a result. Common knobs:
`DRY_RUN=1` to print the commands without running them, `ONLY=` / `SKIP=` to select arms,
`SHARD=/OF=` to split a study across concurrent runners, `SMOKE=1` for a fast pass.

Two indexes, and which one to read depends on the question:

```bash
python3 -m src.analysis.experiment_db          # where every study GOT TO (derived, never stale)
python3 -m src.analysis.experiment_db --arms E09
```

[experiments/EXPERIMENTS.md](experiments/EXPERIMENTS.md) is the durable record and holds the
FINDINGS, written once a study is over. `experiment_db` is the operational view: its
`STUDIES` table is the hand-edited source of truth for a study's intent, and everything else
— arm counts, whether a runner is alive — is derived from `results/` and `/proc` on every
query. Neither replaces the other.

## Reporting

Reporting a set of results means producing a document, not just a table:

```bash
python3 -m src.analysis.report_html --model lgb --runs <run> [<run> ...]   # interrogable
python3 -m src.analysis.report_pdf  --model lgb --runs <run> [<run> ...]   # archival
python3 -m src.analysis.results_table --model lgb --runs <run> --sort-by payout
```

All three read the same arithmetic from `src/analysis/report_data.py` and produce identical
rows. See CLAUDE.md for the column definitions and the three things that are easy to get
wrong (payout is rebuilt per period; `calmar` is not the textbook one; never mix targets).

## Live

`src/live/` fills the submission slots. `src/live/registry.py` is the single source of truth
for what each slot is. The round runs from cron three times each morning (a first pass and
two retries) behind a `flock`, with a fallback ladder per slot; the daily stake report
follows it.

```bash
python3 -m src.live.run_round --dry-run     # rehearse; writes a round report, so not in a live tree
python3 -m src.live.refit --all --force     # refit every slot, outside the submission window
```

## Layout

```
src/
├── data_mani/     dataset download + derived feature builders
├── prediction/    the model zoo behind one sklearn-wrapper contract (models.py)
├── analysis/      experiment_db, the report renderers, slot resolution, stake
├── live/          registry, round runner, submitter, refit scheduler
└── tests/
experiments/       one runner per study + EXPERIMENTS.md (the findings)
results/           forecasts, per-period tables, metrics, reports   (gitignored)
data/              downloaded + built parquet datasets              (gitignored)
models/            fitted live estimators, rewritten every refit    (gitignored)
.claude/skills/    agent workflows, one SKILL.md each
```

## Skills

| Skill | Use it for |
|---|---|
| `crypto-experiment-reporting` | Run and report arm-based experiments in this pipeline |
| `arm-board-reporting` | Turn a finished set of runs into a defensible comparison board |
| `safe-crontab-edits` | Touch the crontab — it drives two independent live pipelines |
