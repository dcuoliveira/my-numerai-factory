# E145 — Chronos-2-Synth forecast-head features: registration

Registered 2026-11-07, before any E145 feature was built or any arm fitted or scored. Everything
below is frozen; a change after scoring must be marked as such in the report and does not move a
verdict.

## Question

Do Chronos-2-Synth's own forecasts (the forecasting head's quantiles, not a hidden state) of a
coin's 20-day label carry information beyond the 22 v2.0 features?

## Features (frozen)

Per row (coin, D): context = 256 grid steps ending at D; target = `target_binned_return_20` with
every label after D − 35 days blanked; cov branches add the 22 v2.0 features through D as
past-only covariates (no future values). Forecast 20 steps after D; q10/q50/q90 kept. A row needs
32 resolved labels in its context, else no forecast (NaN, 0.5 at load, routed to v2.0).
Revision `3607918a9fd027d5c465d8213e46b98e2c041cea`, fp32 inference with TF32 off.

Branches: ZS_COV, ZS_AR (frozen model); FT_COV, FT_AR (one LoRA fine-tune per block 2..11, r 8,
alpha 16, default targets, lr 1e-5, 1000 steps, seed 145, cut at the block's first train date;
blocks 0-1 have too few labels before their cut and get no FT model).

Five summaries per branch: median_mean, median_terminal, median_slope, uncertainty_mean,
asymmetry_mean. Deltas ZS_DELTA = ZS_COV − ZS_AR, FT_DELTA = FT_COV − FT_AR, on raw summaries.
Primary scaling: within-date percentile rank (continuous). Robustness: five-bin rank
(`bin_ties_preserved`, 5/20/50/20/5).

## Arms (frozen)

B0 · ZS_COV_HEAD · FT_COV_HEAD · ZS_DELTA_HEAD · FT_DELTA_HEAD · ZS_FULL_HEAD (ZS_COV+ZS_DELTA) ·
FT_FULL_HEAD (FT_COV+FT_DELTA) · ALL_HEAD (all four) · HEAD_ONLY (all four, no v2.0) ·
AR controls ZS_AR_HEAD, FT_AR_HEAD (B0 + the AR summaries; used only to read "relative to AR") ·
controls P_HIST (tenure), P_WARM (binned share of the head columns observed), P_OBS (rank of the
share of the 256-step context with a resolved label), ALL_H (ALL_HEAD + tenure, vs P_HIST) ·
placebo P_NM ("this coin never has a forecast anywhere in the file", LOOK-AHEAD, never a candidate).

Direct legs: within-date rank of ZS_COV median_mean, FT_COV median_mean, ZS_DELTA median_mean,
FT_DELTA median_mean, sign fixed positive. Rank-blended with each model's B0 prediction at
w ∈ {0, .05, .1, .2, .3, .5}; w chosen by dev routed CORR only.

Models: ridge (there is no `ridge_ktz`; ridge needs no prior), `lgb_ktz`, `mlp_ktz` (linear MLP), fixed
configs, 5 seeds for lgb/mlp, 1 for ridge. Equal-budget retune (one seed) of B0, ZS_COV_HEAD,
FT_COV_HEAD and ALL_HEAD.

## Windows

Walk-forward 130 train / 22 purge / 130 predict, 12 blocks. **Primary dev window: blocks 2–10**
(the dates where every family exists). Blocks 0–10 reported too. Block 11 (2026-02-20 to
2026-08-20) is REUSED evidence, already scored by earlier studies, not a virgin test. The 28
labelled dates after 2026-08-20 and all live dates stay unread.

## Decision rule (per family × model, primary = continuous rank, routed R_SLOT)

A family is a **genuine candidate** only if all hold:

1. dev (blocks 2–10) routed CORR delta vs B0, Newey-West lag 20: t ≥ 2.0;
2. dev routed payout delta not materially harmful: t > −2 and delta > −0.003;
3. dev CORR delta positive in ≥ 4 of 5 seeds (lgb, mlp) and in ≥ 6 of the 9 dev blocks;
4. not explained by tenure / warm-up / missingness: dev CORR delta of the family over P_HIST,
   P_WARM and P_OBS all positive (unrouted, matched seeds);
5. block 11 (reused) routed CORR delta > 0;
6. equal-budget retune (where run) keeps the dev CORR delta positive;
7. the five-bin robustness dataset keeps the dev CORR delta positive.

Beating an AR arm counts for nothing; every criterion is against B0. With about 21 family × model
cells, a t between 2.0 and 2.9 is reported as within multiplicity.

Verdicts:
- **REJECT**: dev routed CORR t ≤ −2, or dev routed payout t ≤ −2, or block-11 routed CORR t ≤ −2.
- **FORWARD TEST ONLY**: a genuine candidate (1–7). The highest status reachable now.
- **NO IMPROVEMENT**: neither.
- **ADOPT**: only after a frozen forward test (below) passes.

A direct leg's blend is a candidate when its dev-chosen w > 0, the blend's dev routed CORR delta
has t ≥ 2, payout is not harmful (as 2.) and block 11 keeps the sign.

## Forward test (if any candidate)

Frozen before reading: the family, model config, blend weight, routing rule. Primary metric:
routed CORR delta vs B0 per date. Read only once ≥ 120 resolved forward dates exist after
2026-08-20 (about six independent 20-day periods); ADOPT if the delta is positive with NW t ≥ 1.5
and payout delta t > −1. The first 28 dates alone never decide.

## Closing rule

If no forecast-head arm beats v2.0 under this rule, close the Chronos-2 feature line: no further
hidden-state dimensions, LoRA settings or downstream model families.
