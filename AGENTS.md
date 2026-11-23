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

Apply these defaults to downstream H processing; preserve the pretrained model's
native input normalization. Use [`$tsfm-embedding-standardization`](.agents/skills/tsfm-embedding-standardization/SKILL.md)
for the causal fitting and reporting details. This is a user preference, not a
claim that these methods universally outperform the alternatives.
