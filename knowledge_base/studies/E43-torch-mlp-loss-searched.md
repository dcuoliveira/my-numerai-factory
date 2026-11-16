# E43 — torch MLP training-loss study, config SEARCHED (best payout)

- **State:** done (12/12 arms)
- **Runner:** experiments/e43_torch_mlp_loss_search_crypto.sh
- **Models / datasets:** model `torch_mlp_loss`; crypto/v2.0 (raw 22) and crypto/v2.0-fenccausal (44), `target_binned_return_20`
- **Artifacts on disk:** 66.9 MB across 12 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/torch_mlp_loss/e43_corr_fenccausal.json, results/metrics/crypto/torch_mlp_loss/e43_mse_raw.json

## Main question
E42's companion: the same 6 loss arms x raw/fenccausal, but each loss runs a FULL search over the torch_mlp grid and finds its own config — the best-achievable-per-loss read, where E42 holds the config fixed to isolate the loss's effect on learning. Paired Newey-West vs the mse control, ranked against the book.

## What was tried
12 arms = 6 losses (mse control, corr, bmc-benchmark, bmc-proxy, sharpe, calmar) x 2 feature sets (raw 22, fenccausal 44). Unlike E42, each arm searches the torch_mlp grid rather than pinning a config. Runs: e43_{mse,corr,bmcbench,bmcproxy,sharpe,calmar}_{raw,fenccausal}.

## Trained parameters
From results/metrics/crypto/torch_mlp_loss/e43_corr_fenccausal.json (representative): model `torch_mlp_loss`, data_version crypto/v2.0-fenccausal, n_features 44, target target_binned_return_20, cv_scheme rolling, train_window 130, refit_every 130, embargo 20, n_splits 5, n_iter 40 (randomized search), seed 42, val_frac 0.3, inner_cv purged, fixed_params null (searched, not pinned), model_options.loss "corr". The raw mse control (e43_mse_raw.json) matches on cadence (n_iter 40, n_splits 5, refit_every 130, seed 42) with model_options.loss "mse", n_features 22, fixed_params null. The specific winning hyperparameters chosen by each search are not stored in the metrics JSON (fixed_params is null).

## What was found
No written finding; from the metrics JSONs the searched configs lift CORR above E42's pinned runs, and the mse control is strong. e43_corr_fenccausal: CORR mean 0.083757 (sharpe 0.931480), MMC mean 0.037776, n = 510. e43_mse_raw (control): CORR mean 0.150695 (sharpe 1.540043), MMC mean 0.064909, n = 516 — the strongest of the two opened arms. A full paired-NW ranking of all 12 arms vs the mse control is not recorded in available artifacts; only per-arm summary metrics exist.
