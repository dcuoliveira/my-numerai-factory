# E50 — One LightGBM tune per book dataset, with the E49 priors (lgb_ktz)

- **State:** done (9/9 arms)
- **Runner:** experiments/e29_lgb_ktz_tune_crypto.sh
- **Models / datasets:** model `lgb_ktz`; one tune per book dataset (raw, tenc, interact, fenccausal, cumlag3, tenc_onehot, onehot_rollsel, allfeat, allfeatpp), `target_binned_return_20`. Prefix `e29_` in the lgb_ktz tree.
- **Artifacts on disk:** 40.6 MB across 9 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/lgb_ktz/e29_tenc.json, results/metrics/crypto/lgb_ktz/e29_tenc_onehot.json

> **Deployment tie (deletion-relevant).** Per this study's recorded idea, `e29_tenc_onehot` is `SACI_06`'s `backtest_run` in src/live/registry.py — an un-indexed run backing a deployed slot. Registered RETROSPECTIVELY: the runner declares EXPERIMENT=E29, but STUDIES' E29 is a different (subset-tuning/FARM) study, so these 9 arms originally resolved to no entry.

## Main question
The lgb twin of E28: one from-scratch LightGBM tune per submitted-model dataset, with the E49 sensitivity priors compiled into the search grid, to produce a tuned lgb_ktz forecaster on every book feature set.

## What was tried
9 arms = one from-scratch RandomizedSearchCV of `lgb_ktz` per book dataset: raw22, tenc23, interact36, fenccausal44, cumlag3-88, tenc_onehot763, onehot_rollsel786, allfeat788, allfeatpp812. The E49 sensitivity priors are compiled into the search grid (the lgb_ktz grid searched, not pinned).

## Trained parameters
From results/metrics/crypto/lgb_ktz/e29_tenc.json (representative): model `lgb_ktz`, data_version crypto/v2.0-tenc, n_features 23, target target_binned_return_20, cv_scheme rolling, train_window 130, refit_every 130, embargo 20, n_splits 5, n_iter 60 (randomized search), search_jobs 3, seed 42, val_frac 0.3, inner_cv purged, fixed_params null (the lgb_ktz grid IS the prior; the winning cell is not stored in the JSON). e29_tenc_onehot.json matches on cadence (n_iter 60, n_splits 5, refit_every 130, seed 42) with data_version crypto/v2.0-tenc-onehot, n_features 763. The feature count (n_features) is what sweeps across the 9 arms.

## What was found
No written finding; from the two arms opened these are strong tuned forecasters in the ~0.157 CORR range. e29_tenc: CORR mean 0.157205 (sharpe 1.779873, max_drawdown 0.8106), MMC mean 0.072328, n = 510. e29_tenc_onehot: CORR mean 0.157183 (sharpe 1.838710, max_drawdown 0.6194), MMC mean 0.074273, n = 510 — the one backing SACI_06. A ranking across all 9 datasets and any paired-vs-book comparison are not recorded in available artifacts.
