# E27 — FARM forecast combination vs the simple average

- **State:** done (24/24 arms)
- **Runner:** experiments/e27_farm_combination_crypto.sh
- **Models / datasets:** `ridge_glmnet` combination wrapper (`run_farm_ensemble`, `src/prediction/farm.py`) over 5 finished member forecasts; two 5-member pools — `decorr5` (most-decorrelated glmnet/lgb/mlp + the two E26 sign-constrained GRRs) and `strong5` (each family's best-CORR run + the same GRRs)
- **Artifacts on disk:** 103.1 MB across 24 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E27 — findings" (lines 358–418) and §"E27 validation — the same estimators reproduce the paper on FRED-MD" (419–459); memory: farm-combination-replication.md (also e61-sur-ridge.md, ktz-priors-and-wrappers.md, litmom-eda-signal-verdict.md); metrics: results/metrics/crypto/ridge_glmnet/e27_strong5_farm1_l2boost.json

## Main question
Replicating Lee & Lee (2025) "Solving the Forecast Combination Puzzle": does FARM — which treats the simple average as a common factor and regresses its own error on the members' idiosyncratic deviations d_i = f_i − f_bar, adding back what is predictable (FARM1; FARM2/3 first strip ≤2 PCA / supervised-PCA factors) — beat the 1/N equal-weight rank average on `numerai_corr` and payout? It is run as a MEASURED null: the prior is that FARM cuts squared-error loss but not rank correlation, because minimising MSE compresses the cross-sectional spread the ranking is paid for.

## What was tried
24 arms = two pools × 12 arms. Each pool holds 5 forecasters; the 12 arms are `equal` (the 1/N baseline every arm is paired against), `farm1/2/3 × {ridge, l2boost, postalasso}`, and the paper's `GR` and `DS` reference combiners. Each arm is paired against the equal-weight blend of the IDENTICAL members on the dates they share, Newey-West lag 20, over 510 periods. A standalone FRED-MD replication harness (`farm_paper.py` / `experiments/farm_paper_replication.py`, data at `data/fredmd/`) ran the same estimators on the paper's own macro-forecasting experiment (5 targets × 3 horizons) as a positive control. `POOLS=highN` rebuilds the paper's ~134-member high-dimensional pool.

## Trained parameters
From `e27_strong5_farm1_l2boost.json`: model `ridge_glmnet` (combination wrapper); members `ridge_glmnet/w6r6_lobo_tenc`, `lgb/e02_concat_trio`, `mlp/a2_mlp_fenccausal`, `gen_ridge/e26_pos`, `gen_ridge/e26_neg`; method `farm1`, beta_method `l2boost`; update_every 22, lag 20, min_history 60, roll 252; n_periods 510. The `equal` baseline (`e27_strong5_equal.json`) has the same five members with method `equal`. These combine fixed, already-finished member forecasts — there is no per-model hyperparameter search in the arm itself.

## What was found
FINDINGS-BACKED. **The paper's method reproduces on the paper's metric and does not transfer to ours — no FARM/GR/DS arm beats 1/N on `numerai_corr` at |t| ≥ 2 in either pool.**
- `decorr5`: all 11 non-equal arms are below equal on both CORR and payout (best case Δpayout −0.00245, t −1.04 for farm1_l2boost; equal CORR 0.16257, payout 0.09353).
- `strong5`: the only gain is MMC, CORR is a dead tie. Best arm `farm1_l2boost` scores CORR 0.16638 vs equal 0.16636 (ΔCORR +0.00003, t +0.01); its entire Δpayout +0.00314 (t +1.32, under the house bar) is MMC (0.07951 → 0.08264). Drawdown identical (1.68).
- The winning estimator is the one that deviates LEAST from 1/N (L2-Boosting at step 0.001) — the puzzle restating itself inside the method built to solve it. FARM2/FARM3 buy nothing over FARM1, and NOT because r=0: on strong5 FARM3 selects r=2 at 57/60 updates (mean r 1.95) and still lands at/below FARM1.
- Validation: on the paper's FRED-MD experiment the same estimators reproduce (cell-level relMSFE correlates 0.934 with Table 1; h=1 anchors near-exact, e.g. FARM1·L2-Boost CPIAUCSL 0.872 vs paper 0.883; 11/13 pattern checks pass). On the 134-member pool relMSFE falls to 0.836–0.864 (DM t +6.6..+9.2) while CORR falls 0.159 → 0.11–0.14 (paired t −3.1..−3.8). So the crypto null is a real data/metric property, not a bug.
- Taken with E10/E23/E25 (60 finished combiner arms), the combiner question stays closed: nothing beats a plain 1/N rank average at the house bar. Operational corollary: the hand-set equal weights in the CURUPIRA slot blends remain correct.
