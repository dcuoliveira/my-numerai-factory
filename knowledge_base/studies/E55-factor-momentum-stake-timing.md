# E55 — Factor-momentum stake timing across the submitted book

- **State:** not started (scout 0/?); a non-model analysis study (no `models`), so it writes no results/metrics
- **Runner:** src/analysis/factor_momentum_stake.py
- **Models / datasets:** none — treats the book's per-period payout panel as the data; the 24 submitted models are the "factors"
- **Artifacts on disk:** 0.0 MB across 0 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** THIN
- **Sources read:** EXPERIMENTS.md §none; memory: e28-e29-prefix-collision.md; metrics: none on disk

## Main question
Apply Gupta-Kelly "Factor Momentum Everywhere" to STAKE SIZING: each submitted model is a factor, its per-period payout its return, and the bet is to tilt stake toward models that did well on recently RESOLVED rounds, walk-forward. Can momentum-timed stake beat static equal weight once the real frictions (20-day decision lag, withdraw-delay ladder, 0.1-NMR/round floor) are modelled?

## What was tried
Per the registered idea: equal-capital (weights sum to 1) z-tilt allocation with signal `.shift(20)` (decide on resolved payouts only) and a withdraw-delay ladder so realised weight = trailing-20 mean of the target; long-only + optional weight floor for the 0.1-NMR floor. Sweep rolling lookbacks {20,40,60,90,120,180,250} and EWMA halflives {10,20,40,80,120}. Baselines: static equal weight, and full-stack-on-the-causal-best (topk=1, ladder-timed). Select-on-train / report-on-held-out, paired Newey-West lag 20, turnover measured; panel is the book's per-period payout (pseudo-OOS, ratios not NMR). The analysis never launched (scout 0/?).

## Trained parameters
Not applicable — a stake-allocation analysis, not a model fit. config JSON not on disk; no runs produced.

## What was found
THIN — no EXPERIMENTS.md section, no corroborating memory finding, and no metrics JSON exist in the artifacts read. The only memory reference (e28-e29-prefix-collision.md) concerns an unrelated bookkeeping bug: E55's empty `models: []` throws an IndexError in `test_every_declared_study_resolves_its_own_prefix`, not anything about stake-timing science.

The study's registered idea text ITSELF asserts a result, reproduced here verbatim but UNCORROBORATED by any sanctioned source read in this task: "on the held-out third momentum BEATS equal weight across every lookback (t +2.2..+4.1) and best-single is strongest (+0.0092, t +3.95), lookback-insensitive, EWMA ~= rolling, turnover tiny (0.004-0.04, ladder-smoothed) -- but the edge is regime-concentrated (full-window t only 0.5-1.8), so promising on the recent regime, not established across history." Treat this as a claim pending an EXPERIMENTS.md/report artifact, not an established finding. Nothing to delete (0 result bytes).
