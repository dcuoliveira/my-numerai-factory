# E122 — Training window of the best standalone RF (e57_tenc_onehot), 1 week to 6 months

- **State:** done (7/7 arms)
- **Runner:** `experiments/e122_rf_window_crypto.sh`
- **Models / datasets:** rf_ktz (E48 priors) on `crypto/v2.0-tenc-onehot` (763 feat), the **60-day** label `target_binned_return_60`; payout-best standalone random forest, run `e57_tenc_onehot`
- **Artifacts on disk:** 28.17 MB across 7 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §(table row, line 113 — no dedicated section, status "running", expectation but no written result); memory: none; metrics: `results/metrics/crypto/rf_ktz/e122_w130.json`

## Main question
The RF twin of E117/E119/E120/E121: how much history should each refit use? Only the rolling window varies (5/10/15/20/22/65/130 periods); the w130 arm reproduces `e57_tenc_onehot`. Unlike the other window studies this one fits and scores on the 60-day label.

## What was tried
Seven window arms (w5…w130). Held constant: data `crypto/v2.0-tenc-onehot` (763 feat), the 40-draw KTZ search (`n_iter=40`) with 5 inner folds, refit cadence 130, embargo auto = 60 (forced by the 60-day label), seed 42, target `target_binned_return_60`, cv rolling/purged. Windows below 130 use `--tune-window 130` (a <70-period block can't hold its own purged fold at embargo 60; search on 130 periods, folds train on the window's length, winner refit on the window). Each window paired NW lag 20 vs w130. 4 refits per arm; n=470 scored periods (fewer than the 20-day studies' 510 — the 60-day label costs ~40 dates).

## Trained parameters
From `results/metrics/crypto/rf_ktz/e122_w130.json`: model `rf_ktz`, data_version `crypto/v2.0-tenc-onehot`, n_features 763, n_iter 40, refit_every 130, embargo 60, train_window 130, seed 42, target `target_binned_return_60`, fixed_params null. 4 refits. Best_params refit 0: `bootstrap` true, `max_depth` 6, `max_features` 0.15, `min_samples_leaf` 300, `n_estimators` 100; refit 3: max_depth 3, max_features 0.3, min_samples_leaf 300, n_estimators 100. Only `train_window` swept across the other six arms.

## What was found
No written finding in EXPERIMENTS.md (status "running"; the row states only the expectation that short windows lose). NUMBERS-ONLY, from the metrics JSONs (60-day-label scores; payout = 0.1·CORR + MMC rebuilt from summary means — ranking-grade only). All arms n=470:
- w5 CORR 0.133865, MMC 0.063141, pay 0.076528
- w10 CORR 0.107559, MMC 0.040931, pay 0.051687
- w15 CORR 0.127063, MMC 0.057145, pay 0.069852
- w20 CORR 0.120169, MMC 0.057725, pay 0.069742
- w22 CORR 0.108489, MMC 0.040339, pay 0.051188
- w65 CORR 0.106510, MMC 0.047855, pay 0.058506
- **w130 (best) CORR 0.161295, MMC 0.082645, pay 0.098774**

Direction: w130 is the best arm (CORR 0.161295, MMC 0.082645, pay 0.098774 — consistent with the quoted `e57_tenc_onehot` 0.0988 payout / 0.0826 MMC at w130). The shorter-window ladder is NOT monotonic here (w5 at CORR 0.134 outscores w10/w22/w65), unlike the strictly monotonic 20-day LightGBM/CatBoost/MLP studies, but w130 still dominates. No paired t-statistics recorded in available artifacts.
