# E51 — RF tune from scratch with the E48 priors (rf_ktz)

- **State:** done (2/2 arms, scout prefix e51scout_)
- **Runner:** experiments/e51_rf_ktz_tune_crypto.sh
- **Models / datasets:** model `rf_ktz`; crypto/v2.0 (raw 22) and crypto/v2.0-fenccausal (44), `target_binned_return_20`
- **Artifacts on disk:** 9.5 MB across 2 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/rf_ktz/e51_raw.json, results/metrics/crypto/rf_ktz/e51_fenc.json

## Main question
The rf sibling of E28 (mlp_ktz) / E50 (lgb_ktz): a from-scratch RandomizedSearchCV of `rf_ktz` — the E48 sensitivity priors compiled into a search grid — on the raw 22 and on fenccausal (44, the feature set the book's best neural slot uses), to produce a deployable RF. A paired-vs-book comparison is the stated next step.

## What was tried
2 arms (raw, fenc). The grid encodes E48's four findings: n_estimators PINNED at 100 (fANOVA red herring, share 0.0002); max_depth a shallow gate {3,4,6}; max_features a dense-low sweep {0.05..0.30} (0.6/1.0 dropped); min_samples_leaf heavy {300,1000,2000,4000} with the ceiling OPENED past E48's binding max of 1000 (its marginal was monotone to the grid edge — the E07/E17 alpha-ceiling trap). 60-cell grid, every cell in E48's good region, searched at --n-iter 40, CURUPIRA cadence (rolling 130/130, embargo 20, n_splits 5, seed 42).

## Trained parameters
From results/metrics/crypto/rf_ktz/e51_raw.json: model `rf_ktz`, data_version crypto/v2.0, n_features 22, target target_binned_return_20, cv_scheme rolling, train_window 130, refit_every 130, embargo 20, n_splits 5, n_iter 40, search_jobs 8, seed 42, val_frac 0.3, inner_cv purged, fixed_params null (the rf_ktz grid IS the prior; the winning cell is not stored in the JSON). e51_fenc.json: same cadence with data_version crypto/v2.0-fenccausal, n_features 44, search_jobs 1 (the fenc arm hung on the nested search-jobs>1 x n_jobs>1 loky path, per the E52 idea). Per the study idea the search picked min_samples_leaf 300 (lighter than E48's OOS marginal of 1000 — a CV/OOS nuance; the chosen leaf value is not in the JSON).

## What was found
No written finding; from the metrics JSONs the rf_ktz tunes land at ~0.153 CORR (close to the top MLP/lgb slots). e51_raw: CORR mean 0.153724 (sharpe 1.596543, max_drawdown 1.0177), MMC mean 0.064717, n = 518 — the study idea records this as CORR 0.154 / payout 0.080 (payout is not stored in the metrics JSON). e51_fenc: CORR mean 0.153018 (sharpe 1.771979, max_drawdown 0.7406), MMC mean 0.078063, n = 510. The fenc arm has notably higher MMC and lower drawdown than raw. A paired-vs-book result is not recorded in available artifacts.
