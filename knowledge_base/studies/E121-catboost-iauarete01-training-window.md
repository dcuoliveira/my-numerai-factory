# E121 — Training window of the best standalone CatBoost (IAUARETE_01), 1 week to 6 months

- **State:** done (7/7 arms)
- **Runner:** `experiments/e121_catboost_train_window_crypto.sh`
- **Models / datasets:** catboost_ktz on `crypto/v2.0-fenccausal` (44 feat = raw 22 + 22 causal per-symbol means); the payout- and MMC-best single CatBoost of E32, deployed as component IAUARETE_01 (`catboost_ktz/e32_sel_fenccausal`)
- **Artifacts on disk:** 34.88 MB across 7 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §(table row, line 112 — no dedicated section, status "registered", no written finding); memory: none; metrics: `results/metrics/crypto/catboost_ktz/e121_w130.json`

## Main question
The CatBoost twin of E117/E119/E120: how much history should each refit use? Only the rolling window varies (5/10/15/20/22/65/130 periods); the w130 arm reproduces `e32_sel_fenccausal` / IAUARETE_01.

## What was tried
Seven window arms (w5…w130). Held constant: data `crypto/v2.0-fenccausal` (44 feat), the 12-cell KTZ grid searched exhaustively (`--n-iter 12`) at every refit, refit cadence 130, embargo 20, seed 42, target `target_binned_return_20`, cv rolling/purged. Windows below 130 use `--tune-window 130` (search on the 130 periods ending at the same date, folds train on the window's length, winner refit on the window). The KTZ grid is static so `grid_from` is ignored. Each window paired NW lag 20 vs w130. 4 refits per arm; n=510 scored periods.

## Trained parameters
From `results/metrics/crypto/catboost_ktz/e121_w130.json`: model `catboost_ktz`, data_version `crypto/v2.0-fenccausal`, n_features 44, n_iter 12, refit_every 130, embargo 20, train_window 130, seed 42, target `target_binned_return_20`, model_options `{"iterations": 2000}`, fixed_params null. 4 refits. Best_params refit 0: `depth` 2, `iterations` 2000, `l2_leaf_reg` 10.0, `learning_rate` 0.005, `min_data_in_leaf` 100, `rsm` 0.1; refit 3 identical except `rsm` 0.2. Only `train_window` swept across the other six arms.

## What was found
No written finding in EXPERIMENTS.md (status "registered"). NUMBERS-ONLY, from the metrics JSONs (payout = 0.1·CORR + MMC rebuilt from summary means — ranking-grade only, per CLAUDE.md). All arms n=510:
- w5 CORR 0.062199, MMC 0.023781, pay 0.030001
- w10 CORR 0.058216, MMC 0.021036, pay 0.026858
- w15 CORR 0.060331, MMC 0.023583, pay 0.029616
- w20 CORR 0.067473, MMC 0.024042, pay 0.030789
- w22 CORR 0.077805, MMC 0.027713, pay 0.035494
- w65 CORR 0.116509, MMC 0.050104, pay 0.061755
- **w130 (best) CORR 0.156464, MMC 0.077381, pay 0.093027**

Direction: monotonic in history — more training history is strictly better, w130 the best by a wide margin (same shape as E119/E120). w130 CORR 0.156464 / pay 0.093027 is consistent with IAUARETE_01's quoted 0.09303. No paired t-statistics recorded in available artifacts.
