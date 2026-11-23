---
name: stake-book-update
description: Add a new model (registry slot) to the momentum stake book that decides capital allocation, and report whether the top 3 changes -- previous top 3 (last watch / book without it) against the recent top 3 with it -- as a PDF. Use whenever the user passes a newly added model and asks to add it to the allocation / stake pipeline, asks whether a new slot changes the top 3 or top 1, or asks to re-rank the book by recent performance for staking.
---

# Stake book update

Invoke with `/stake-book-update <SLOT> [<SLOT> ...]`, or ask to "add <SLOT> to the
stake book and tell me if the top 3 changed".

## What the routine is

Capital goes where E55's **momentum watch** says. It ranks every model in
`src/analysis/factor_momentum_stake.py::BOOK` by trailing-250 mean payout
(`0.1*corr + mmc`) on RESOLVED rounds only (shift 20, the 20-day label lag),
puts the whole stake on top-1 ("top1 roll250", the E55 out-of-sample pick),
stores the ranking in `results/live/momentum_top1.json` and e-mails when top-1
changes (`python3 -m src.analysis.factor_momentum_stake --watch`, wrapped by
`src/analysis/cron_momentum_watch.sh`). The deployed mean-variance solver in
`src/live/stake.py` is the static alternative; it is not what this routine updates.

## Steps

1. Check the interpreter: `/home/danielco/miniconda3/bin/python3`.
2. Resolve the name. It must be a slot in `src/live/registry.MODELS` with a
   scored `backtest_run`. If the user gives an experiment arm (`model/run`) that
   is not a slot, say so and stop: define the slot first. A slot backtested on
   the 60-day label is refused by the tool (other payout scale).
3. Run the report (no `--write` yet):

   ```bash
   /home/danielco/miniconda3/bin/python3 -m src.analysis.book_update <SLOT>
   ```

   It writes `results/reports/crypto/book_update/<date>_<slot>.{html,json,pdf}`.
   When the local Chromium fails, it prints the PDF on lovelace.
4. Read the four top-3 columns:
   - **previous (watch)**: what the last watch stored.
   - **previous (now)**: the book without the candidate on today's full panel.
   - **previous (window)** vs **recent**: without and with the candidate on the
     candidate's own rows. This pair is the candidate's effect and nothing else.
   A difference between watch and now is drift, not the candidate.
5. Check coverage. `BOOK` is joined on COMMON dates, so a candidate whose
   backtest stops early would cut the tail of every model's panel and move the
   live watch back in time. The tool marks this "NOT safe to add" and blocks
   `--write`. Then report the window comparison, say it describes the
   candidate's period and not today, and offer to extend the candidate's
   backtest through the book's last date. That is a new run, so register it
   first as a numbered study (CLAUDE.md). Use `--force` only if the user
   explicitly accepts the truncation.
6. When it is safe, add it:

   ```bash
   /home/danielco/miniconda3/bin/python3 -m src.analysis.book_update <SLOT> --write
   ```

   This appends the slot to `BOOK` and leaves the watch state alone, so if top-1
   changed, the next scheduled watch sends the stake-move e-mail. Run
   `python3 -m pytest -q src/tests/test_book_update.py`. Then commit
   `factor_momentum_stake.py` and push (CLAUDE.md).
7. Report in chat:
   - the previous and recent top 3 side by side, and the candidate's rank and signal;
   - whether top-1 moves, which means a stake move;
   - the paired Newey-West t against each previous top-3 member. Rankings use
     point estimates, so a gap at |t| < 2 is not an established difference.
     Say so even when the watch would act on it.
   Attach the PDF.

## Caveats to keep

- Raw payout (no payout factor or clipping): read ratios, not NMR.
- `calmar` / `dd` in the table are on the same resolved 250-row window, so they
  are comparable down the column.
- The watch currently omits some submitted slots (for example the `_R_01`
  resid slots). This routine only adds what it is given; it does not audit
  `BOOK`.
