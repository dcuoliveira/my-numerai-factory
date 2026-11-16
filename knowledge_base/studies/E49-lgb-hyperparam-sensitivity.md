# E49 — LightGBM hyperparameter sensitivity (Kalfa-Timmermann-van der Zwan)

- **State:** done (316/316 arms)
- **Runner:** experiments/e27_hyperparam_sensitivity_crypto.sh
- **Models / datasets:** model `lgb`; crypto/v2.0 (raw 22), `target_binned_return_20`. Artifacts keep the `ktz_` handle (not `e49_`).
- **Artifacts on disk:** 1505.5 MB across 316 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none assigned (ktz-priors-and-wrappers.md, read for E48, attributes the lgb_ktz grid to "E27"=this study); metrics: results/metrics/crypto/lgb/ktz_nl0016_lr0p100_ne08000_cs1p00.json

> **Reclaimability caveat (deletion-relevant).** This study's own recorded idea states: "Findings were never written; until they are, this study's forecasts are not reclaimable." No EXPERIMENTS.md section was ever written. The sweep is also the largest un-indexed artifact set in the repo — it ran under the id E27 (taken by a concurrent FARM study), so its 316 arms belonged to no STUDIES entry and a disk cleanup would have read them as orphaned. Do NOT delete on the basis of this file alone; the science was not written up.

## Main question
Run the "Overhyped?" (Kalfa-Timmermann-van der Zwan) sensitivity framework on the BOOSTER — the LightGBM sibling of E17 (mlp) and E48 (rf) — to map the lgb surface so the lgb_ktz grid can be set from evidence.

## What was tried
A 4-axis num_leaves x learning_rate x n_estimators x colsample factorial, 315 cells plus the ktz_bridge_canonical arm the runner writes last (316 total), on the raw 22 of crypto/v2.0, then the paper's readings via src.analysis.hyper_sensitivity (invoked as --prefix ktz_). Published report: results/reports/crypto/ktz_top10_vs_stacked.html. Artifacts keep the `ktz_` handle because it predates the numbering.

## Trained parameters
From results/metrics/crypto/lgb/ktz_nl0016_lr0p100_ne08000_cs1p00.json (one representative cell): model `lgb`, data_version crypto/v2.0, n_features 22, target target_binned_return_20, cv_scheme rolling, train_window 130, refit_every 130, embargo 20, n_splits 2, n_iter 1 (pinned), seed 42, val_frac 0.3, inner_cv purged. fixed_params for this cell: num_leaves 16, learning_rate 0.1, n_estimators 8000, colsample_bytree 1.0, min_child_samples 100, reg_lambda 10, subsample 0.8, subsample_freq 1, max_depth -1, deterministic true, force_row_wise true. Across the 316 cells num_leaves/learning_rate/n_estimators/colsample sweep the factorial (run names ktz_nl<..>_lr<..>_ne<..>_cs<..>).

## What was found
No written finding exists (the study idea explicitly states findings were never written). From the one cell opened: ktz_nl0016_lr0p100_ne08000_cs1p00 CORR mean 0.050096 (sharpe 0.658930, max_drawdown 1.3406), MMC mean 0.020080, n = 515 — one factorial cell, not a ranked result. The compiled lgb_ktz grid recorded in ktz-priors-and-wrappers.md ("lgb_ktz (E27) — 432 cells: learning_rate x colsample x num_leaves x n_est") is the only downstream distillation on record; the sensitivity readings (marginals, fANOVA, priors) are not recorded in available artifacts.
