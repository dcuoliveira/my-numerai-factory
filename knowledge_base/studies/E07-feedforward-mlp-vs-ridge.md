# E07 — A feedforward net against the deployed ridge, both tasks

- **State:** done (2/2 arms, 2 scout)
- **Runner:** experiments/e07_mlp_crypto.sh
- **Models / datasets:** sklearn mlp on crypto/v2.0 (raw 22), as regression and as classification, at the CURUPIRA cadence
- **Artifacts on disk:** 9.1 MB across 2 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/mlp/e07_mlpreg_raw.json
- Note: `payout` = 0.1*CORR_mean + MMC_mean from stored summaries.

## Main question
First non-linear, non-tree function class on the board: a sklearn MLP on the raw 22
features, run both as regression (against CURUPIRA_02's backtest) and as classification
(against its clf twin). The estimator is the only thing varying.

## What was tried
2 arms: `e07_mlpreg_raw` (regression) and `e07_mlpclf_raw` (classification), both on
crypto/v2.0, 510 periods, 20-day label. Unlike the E01-E06 boosters, these run a REAL
random search (n_iter 50, n_splits 5) over MLP alpha (grid 360), batch_size (24),
learning_rate_init (240) and hidden_layer_sizes.

## Trained parameters
From results/metrics/crypto/mlp/e07_mlpreg_raw.json: model mlp, task regression,
target_binned_return_20, data_version crypto/v2.0, n_features 22, cv_scheme rolling,
n_splits 5, train_window 130, refit_every 130, embargo 20, seed 42, search_type random,
n_iter 50, no fixed_params. Refit-0 best_params: alpha 8.227241, batch_size 2048,
hidden_layer_sizes [64, 32], learning_rate_init 0.008886238. (The clf twin chose
alpha 0.000855, batch_size 1024, hidden [64], lr 0.005541.) Runtime ~92 min (reg),
~124 min (clf).

## What was found
No written finding; from the metrics (510 periods): regression clearly beats
classification. `e07_mlpreg_raw` CORR 0.158119, MMC 0.074617, payout 0.090428;
`e07_mlpclf_raw` is far weaker, CORR 0.070290, MMC 0.047306, payout 0.054335. The MLP
regression config chosen here (alpha 8.23, [64,32], lr 0.0089, bs 2048) is the pinned
recipe reused by E13. No paired t-stat against the ridge baseline in available artifacts.
