# E52 — One rf_ktz tune per book dataset, with the E48 priors

- **State:** done (12/9 arms, scout prefix e52scout_)
- **Runner:** experiments/e52_rf_ktz_tune_crypto.sh
- **Models / datasets:** model `rf_ktz`; one tune per book dataset (raw22, tenc23, interact36, fenccausal44, cumlag3-88, tenc_onehot763, onehot_rollsel786, allfeat788, allfeatpp812), `target_binned_return_20`
- **Artifacts on disk:** 50.4 MB across 12 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/rf_ktz/e52_raw.json, results/metrics/crypto/rf_ktz/e52_fenccausal.json

## Main question
The rf sibling of E28 (mlp_ktz) and E50 (lgb_ktz), and the multi-dataset scale-up of E51 (raw+fenc only): a from-scratch RandomizedSearchCV of `rf_ktz` — the E48 priors as a 60-cell grid — on ALL NINE datasets the deployed book trains on, so every book slot gets an RF counterpart tuned on its own feature set. The pool a paired-vs-book comparison and any RF blend draws from.

## What was tried
One tune per book dataset (9 planned; 12 runs on disk incl. scouts). Grid: n_estimators pinned 100 (E48 red herring), max_depth {3,4,6}, max_features {0.05..0.30}, min_samples_leaf {300,1000,2000,4000} (E48 ceiling opened). --n-iter 40, 5-fold CV, CURUPIRA cadence (rolling 130/130, embargo 20, seed 42). SERIAL search x n_jobs (one joblib level — E51's fenc arm hung on the nested search-jobs>1 x n_jobs>1 loky path), parallelised at the DATASET level. Per the study idea the search picks leaf 300 (lighter than E48's OOS marginal of 1000 — a CV/OOS nuance).

## Trained parameters
From results/metrics/crypto/rf_ktz/e52_raw.json: model `rf_ktz`, data_version crypto/v2.0, n_features 22, target target_binned_return_20, cv_scheme rolling, train_window 130, refit_every 130, embargo 20, n_splits 5, n_iter 40, search_jobs 1 (serial), seed 42, val_frac 0.3, inner_cv purged, fixed_params null (the rf_ktz grid IS the prior; the chosen cell is not stored in the JSON). e52_fenccausal.json: same cadence, data_version crypto/v2.0-fenccausal, n_features 44. The feature set (data_version/n_features) is what sweeps across the arms.

## What was found
No written finding; from the metrics JSONs the per-dataset rf_ktz tunes land at ~0.154 CORR, matching E51. e52_raw: CORR mean 0.154655 (sharpe 1.621754, max_drawdown 0.9804), MMC mean 0.063855, n = 518. e52_fenccausal: CORR mean 0.153018 (sharpe 1.771979, max_drawdown 0.7406), MMC mean 0.078063, n = 510 — these e52_fenccausal summary values are byte-identical to E51's e51_fenc arm (same CORR/MMC/std/drawdown/n), i.e. the fenccausal rf_ktz tune reproduces across the two studies. A ranking across all 9 datasets and the paired-vs-book comparison are not recorded in available artifacts.
