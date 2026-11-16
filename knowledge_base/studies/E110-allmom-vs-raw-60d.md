# E110 — all-momentum dataset vs raw on the 60-day target (allmom60 vs raw60)

- **State:** done (10/10 arms)
- **Runner:** experiments/e110_allmom_t60_crypto.sh
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz; allmom60 on crypto/v2.0-allmom-fillmask (57f) vs raw60 on crypto/v2.0 (22f); scored on target_binned_return_60
- **Artifacts on disk:** 47.2 MB across 10 forecast runs (plus 10 metrics + 10 hyperparams + 10 per_period) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none; memory: e108-e111-allmom-pipeline.md; metrics: results/metrics/crypto/catboost_ktz/e110_catboost_ktz_allmom60.json, results/metrics/crypto/ridge/e110_ridge_raw60.json

## Main question
The 60-day twin of E108: does the all-momentum dataset beat raw on the 60-day target? Prior: momentum was null vs raw at 20d (E64–E67), but slower signals (12-1, Carver trend) might fit a 60-day horizon better. A matched raw-v2.0@60 baseline did not exist for 4 of the 5 model classes, so it had to be trained here.

## What was tried
Two lanes, 5 models each tuned from scratch on target_binned_return_60 (embargo auto → 60): raw60 on crypto/v2.0 (22f, the matched 60-day baseline) and allmom60 on crypto/v2.0-allmom-fillmask (57f = raw + feature_mom_ 17 + feature_xsmom_ 3 + feature_tf_ 15). Shared protocol otherwise (rolling 130/130, 5-fold, seed 42, n_iter 50). Reported per model (raw60/allmom60 rows plus allmom60-vs-raw60 paired, Newey-West lag 20). 60-day numbers are NOT comparable to 20-day tables. Feeds E111 (meta residualisation + ensembles), which also rescores these forecasts on the payable 20-day label.

## Trained parameters
From results/metrics/crypto/ridge/e110_ridge_raw60.json config: model ridge; data_version crypto/v2.0; cv_scheme rolling; train_window 130; refit_every 130; embargo 60; inner_cv purged; n_splits 5; val_frac 0.3; n_iter 50; search_type random; metric corr; target target_binned_return_60; task regression; n_features 22; seed 42; fixed_params null. CORR.mean 0.17861, MMC.mean 0.07571, n=486. The allmom60 catboost twin (e110_catboost_ktz_allmom60.json) is identical config with model catboost_ktz, data_version crypto/v2.0-allmom-fillmask, n_features 57, embargo 60, search_jobs 1; CORR.mean 0.15510, MMC.mean 0.05459, n=486.

## What was found
MEMORY-BACKED. No dedicated EXPERIMENTS.md section. The memory records E110 only as the 60-day stage that feeds E111, with no standalone paired verdict written; it notes the raw@60 baseline had to be trained (no matched arms existed for lgb/cat/rf/mlp, only an old ridge t60_c02_raw). From the metrics opened, on the 60-day target raw60 ridge (CORR 0.17861) is above allmom60 catboost (0.15510); a full per-model paired delta is not recorded in the available artifacts. E111 (which rescores these on the payable 20-day label) is where the verdict lands: allmom trails raw after residualisation too, and no allmom ensemble beats its best member or the book.
