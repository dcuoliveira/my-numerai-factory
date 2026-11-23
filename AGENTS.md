# Repository instructions for Codex

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

## Register every experiment before launch

`results/experiments.db` (`python3 -m src.analysis.experiment_db`) indexes
every numbered study; the tracked source of truth is `STUDIES` in
`src/analysis/experiment_db.py`, and findings go in `experiments/EXPERIMENTS.md`.

Every experiment gets a number and a registration BEFORE it launches -- no
exceptions for "quick" reruns, add-on runs, GPU queues, or anything outside
`results/metrics/` (e.g. a study that writes its own output directory). An
unregistered job cannot be found later by `experiment_db` or by grep, which
is how the 1M-parameter volatility MLP runs ran for a day with no study id
(registered after the fact as E182). Before starting any run:

1. Take the next free id: `grep -oE '"E[0-9]+"' src/analysis/experiment_db.py | sort -V | tail -1`, plus one.
2. Add a `STUDIES["E<n>"]` entry to `src/analysis/experiment_db.py`
   (title, idea, runner, prefix, models, expected_arms). If the study writes
   outside `results/metrics/`, say where in `idea`.
3. Add a `### E<n>: <title> (registered <date>)` section to
   `experiments/EXPERIMENTS.md` with the user's request, design, launch
   command, host and output paths.
4. Commit both before (or together with) launching, and tell the user the id.

If you find a running or finished job with no id, register it the same way
before reporting on it.

## Numerai Crypto missing values

When creating, extending, rebuilding, testing or evaluating a Numerai Crypto
dataset with potentially missing candidate features, use
[`$numerai-crypto-missing-values`](.agents/skills/numerai-crypto-missing-values/SKILL.md)
before implementation. Before accepting or interpreting the resulting experiment,
use [`$numerai-crypto-missing-values-audit`](.agents/skills/numerai-crypto-missing-values-audit/SKILL.md).

## MLP training compute

For Timmermann–Vulicevic epoch schedules, FLOP budgets, and compute scaling,
use [`$timmermann-mlp-compute`](.agents/skills/timmermann-mlp-compute/SKILL.md).

## Forecasting model embedding standardization

Whenever using Chronos, t0, TabPFN-TS or Timestral embeddings H, use these
user-recommended defaults unless the user specifies another method or requests
a standardization comparison:

| Model family | Recommended H standardization |
| --- | --- |
| Chronos (including Chronos-2 / `chronos2`) | `rolling` |
| t0 | `rolling` |
| TabPFN-TS / `tabpfnts` | `asset` |
| Timestral | `robust` |

The recommended way to use H is **design C**: concatenate standardized H
across a fixed training asset universe, with consistent asset slots and each
asset included once, as input to each per-asset prediction model. Use C by
default alongside the model-specific standardization above, unless the user
specifies another design or requests a design comparison.

Apply these defaults to downstream H processing; preserve the pretrained model's
native input normalization. Use [`$tsfm-embedding-standardization`](.agents/skills/tsfm-embedding-standardization/SKILL.md)
for the causal fitting and reporting details. This is a user preference, not a
claim that these methods universally outperform the alternatives.

## Reusable TSFM signal experiment

When asked to run the **TSFM signal experiment**, or the hidden-state / per-asset
ridge / cross-target residualization and ensemble workflow on a supplied dataset,
use [`$tsfm-signal-experiment`](.agents/skills/tsfm-signal-experiment/SKILL.md).
For example: "Run the TSFM signal experiment on crypto/my-selected-signals with
Chronos-2, t0 and Timestral." The skill covers both20d/60d training and scoring,
raw/residual standalone and ensemble forecasts, common-asset payout reports and
top-three ensemble plots. It generalizes the E206–E211 workflow; do not substitute
momentum-specific paths, features or dates for the supplied dataset.

## Adding a model to the stake book

When the user passes a newly added model and asks to add it to the allocation /
stake pipeline, or asks whether it changes the top 3, use
[`$stake-book-update`](.agents/skills/stake-book-update/SKILL.md). It runs
`python3 -m src.analysis.book_update <SLOT>`. That compares the previous top 3
(the last momentum watch, and the book without the slot) with the recent top 3
(the book with it) on E55's roll-250 resolved-payout ranking, and writes a PDF.
`--write` appends the slot to `factor_momentum_stake.BOOK`. It refuses when the
slot's backtest ends before the book's, because `BOOK` is joined on common dates
and the live watch would lose its tail.
