# E119 — Training window of the best standalone MLP (IA_01), 1 week to 6 months

- **State:** done (7/7 arms; 5 scouts)
- **Runner:** `experiments/e119_ia01_train_window_crypto.sh`
- **Models / datasets:** mlp (sklearn MLPRegressor) on `crypto/v2.0-fenccausal`, 44 features (raw 22 + 22 causal per-symbol means); IA_01's setup
- **Artifacts on disk:** 62.19 MB across 12 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §(table row, line 110 — no dedicated section, status "registered", no written finding); memory: none; metrics: `results/metrics/crypto/mlp/e119_w130.json`

## Main question
E117's question (how much history should each refit use?) asked of IA_01 — the payout-best of the standalone 20-day MLP runs. Everything held except the rolling training window: 5 / 10 / 15 / 20 / 22 / 65 / 130 periods. The w130 arm is meant to reproduce IA_01 / `a2_mlp_fenccausal`.

## What was tried
Seven window arms (w5…w130) plus scouts. Held constant: data `crypto/v2.0-fenccausal` (44 feat), 50-draw random search (`n_iter=50`) with 5 inner purged splits, refit cadence 130, embargo 20, seed 42, target `target_binned_return_20`, era_step 1, cv rolling. Hyperparameters searched from scratch at every refit; windows below 130 use `--tune-window 130` (search on the 130 periods ending where the window ends, folds that train on the window's length, winner refit on the window). w130 uses no tune-window (searches its own window). Scout arms include `e119scout_pin_*` and `e119scout_pinx_*` (n_iter=1 pins, search_jobs 1/5/8) and `e119scout_w5` (128 periods). 4 refits per arm; n=510 scored periods.

## Trained parameters
From `results/metrics/crypto/mlp/e119_w130.json`: model `mlp`, data_version `crypto/v2.0-fenccausal`, n_features 44, cv_scheme rolling / inner_cv purged, n_splits 5, n_iter 50, refit_every 130, embargo 20, train_window 130, seed 42, target `target_binned_return_20`, fixed_params null. 4 refits. Searched best_params refit 0: `mlp__alpha` 2.5514, `mlp__batch_size` 2048, `mlp__hidden_layer_sizes` [64, 32], `mlp__learning_rate_init` 0.005541; refit 3: alpha 8.2272, batch_size 2048, hidden [64, 32], lr 0.008886. Across arms only `train_window` swept (5→130); the n_iter=1 scout pins (`e119scout_pin_sj1/sj5`) reach the highest numbers.

## What was found
No written finding in EXPERIMENTS.md (status "registered"). NUMBERS-ONLY, from the metrics JSONs (CORR / MMC / payout where payout = 0.1·CORR + MMC rebuilt from the stored summary means — ranking-grade only, per CLAUDE.md; not the per-period payout). All arms n=510:
- w5 CORR 0.052823, MMC 0.025227, pay 0.030509
- w10 CORR 0.044937, MMC 0.018484, pay 0.022978
- w15 CORR 0.060792, MMC 0.018382, pay 0.024461
- w20 CORR 0.045014, MMC 0.009953, pay 0.014455
- w22 CORR 0.057359, MMC 0.013918, pay 0.019654
- w65 CORR 0.082546, MMC 0.031744, pay 0.039999
- **w130 (best) CORR 0.154566, MMC 0.076005, pay 0.091461**

Direction: more history is strictly better — w130 dominates every shorter window by a wide margin (CORR roughly triples from w5 to w130). The searched w130 scores CORR 0.154566; the n_iter=1 pin scouts (`e119scout_pin_sj1/sj5`) reach CORR 0.160734 / MMC 0.080160 / pay 0.096234 at n=510. No paired t-statistics are recorded in available artifacts.
