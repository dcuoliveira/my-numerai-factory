# E60 — MLP fit-60 / score-20: the best-MMC MLP setups re-tuned on 60d

- **State:** done (5/5 full arms + 5 scouts = 10 runs)
- **Runner:** experiments/e60_mlp_fit60_score20_crypto.sh
- **Models / datasets:** models `mlp`, `torch_mlp`, `mlp_ktz2`; arms mlp/fenccausal(44), mlp/tenc-onehot(763), torch_mlp/tenc(23), mlp/raw(22), mlp_ktz2/fenccausal
- **Artifacts on disk:** 29.92 MB across 10 forecast runs (5 full + 5 scout) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/mlp_ktz2/e60_ktz2_fenccausal.json, results/metrics/crypto/mlp/e60_mlp_fenccausal.json (+ torch_tenc, mlp_onehot, mlp_raw summarised)

## Main question
Does training an MLP on the smoother 60-day signal pay better on the payable 20-day label? E60 FITS on target_binned_return_60 but the headline deliverable RE-SCORES the predictions against target_binned_return_20 (the inverse of SACI_03, which fits 20 and scores 60).

## What was tried
The MLP sibling of E57/E58/E59. The five MLP setups that led MMC per feature set are re-tuned FROM SCRATCH (fresh search, not the pinned 20-day config): mlp/fenccausal, mlp/tenc-onehot, torch_mlp/tenc, mlp/raw, and mlp_ktz2/fenccausal (the E33-reprior wider grid). Each fits at embargo 60 (~470 periods, ~8 effective windows -> NW lag 60); predictions then re-scored against the 20-day label via src/analysis/cross_target.py. The runner also reports native 60d metrics (comparable to E57-59's 60d board). Each full arm has a matching era-step-6 scout.

## Trained parameters
From the metrics JSONs (native 60d fit): cv_scheme rolling, embargo 60, train_window 130, refit_every 130, n_splits 5, seed 42, n_iter 50, target target_binned_return_60, no fixed_params. Representative winning arm e60_ktz2_fenccausal (model mlp_ktz2, n_features 44); best_params (refit 0): {mlp__activation 'identity', mlp__alpha 10.0, mlp__batch_size 1024, mlp__hidden_layer_sizes [32], mlp__learning_rate_init 0.00316} — i.e. a near-linear (identity-activation) single-32-unit net was selected.

## What was found
No written finding; from the metrics (native 60-day fit, n=470/478), ranked by the repo payout(0.1·CORR+MMC):
- e60_ktz2_fenccausal (mlp_ktz2): CORR 0.13195, MMC 0.06800, payout 0.08120
- e60_mlp_onehot (mlp):           CORR 0.11251, MMC 0.06412, payout 0.07537
- e60_mlp_raw (mlp, n=478):       CORR 0.12012, MMC 0.05448, payout 0.06649
- e60_torch_tenc (torch_mlp):     CORR 0.11225, MMC 0.06422, payout 0.07545
- e60_mlp_fenccausal (mlp):       CORR 0.08090, MMC 0.03418, payout 0.04227

The headline 20-day re-scored MMC/payout (the study's actual deliverable, vs the idea's stated fit-20 MLP MMC baselines: fenccausal 0.0825, tenc-onehot 0.0781, tenc 0.0775, raw 0.0746) is produced by cross_target.py and is NOT stored in the metrics JSONs read, so the fit-60-vs-fit-20 verdict cannot be quoted from the artifacts here. On the native 60-day scale the mlp_ktz2 fenccausal arm (identity activation) leads at MMC 0.0680 — below the rf/lgb/catboost 60-day winners (E57-59, MMC ~0.078-0.083). Per-period Sharpe/drawdown and the 20-day re-score must be recomputed via cross_target.py / the report tooling.
