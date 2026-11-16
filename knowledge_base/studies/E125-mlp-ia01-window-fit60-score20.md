# E125 — E119's training-window ladder for IA_01, fitted on the 60-day label, scored on the 20-day one

- **State:** partial (3/7 arms per manifest snapshot; 4 window metrics files now on disk: w5, w22, w65, w130)
- **Runner:** `experiments/e125_ia01_train_window_t60_crypto.sh` (via E119's runner, `E119_TARGET=60`)
- **Models / datasets:** mlp on `crypto/v2.0-fenccausal` (IA_01's setup, 44 feat); fitted/tuned on `target_binned_return_60`
- **Artifacts on disk:** 15.28 MB across 3 forecast runs (manifest snapshot; disk now holds 4 window arms) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §(table row, line 116 — no dedicated section, status "registered", no result); memory: none; metrics: `results/metrics/crypto/mlp/e125_w130.json`

## Main question
The MLP twin of E124: does fitting E119's IA_01 window-ladder on the smoother 60-day label and deploying on the 20-day payable label beat training directly on 20, and does it depend on the window? `e125_w130` is meant to reproduce `mlp/e60_mlp_fenccausal`.

## What was tried
Same model, data and windows as E119 — mlp on `crypto/v2.0-fenccausal` (44 feat), rolling windows 5/10/15/20/22/65/130, the 50-draw search from scratch at every refit, windows < 130 tuned on the 130 periods ending where the window ends — but FITTED and TUNED on `target_binned_return_60` with embargo 60 (two inner folds, not four). refit 130, seed 42, cv rolling/purged. Deliverable: every arm RE-SCORED on `target_binned_return_20` via `cross_target.score_on_payable_target`, consolidated with E119's fit-20 arms on the 470 shared dates, paired NW lag 20 and 60.

## Trained parameters
From `results/metrics/crypto/mlp/e125_w130.json`: model `mlp`, data_version `crypto/v2.0-fenccausal`, n_features 44, n_iter 50, refit_every 130, embargo 60, train_window 130, seed 42, target `target_binned_return_60`, fixed_params null. 4 refits. Best_params refit 0 and refit 3 identical: `mlp__alpha` 10.0, `mlp__batch_size` 2048, `mlp__hidden_layer_sizes` [64], `mlp__learning_rate_init` 0.008886. Only `train_window` swept across the other arms.

## What was found
No written finding in EXPERIMENTS.md (status "registered"). NUMBERS-ONLY. CAVEAT: the metrics JSONs score each arm on the 60-day label; the payable deliverable (20-day re-score via `cross_target`, consolidated with E119) is NOT present in these JSONs or any source read here. 60-day-label figures (payout = 0.1·CORR + MMC from summary means — ranking-grade only), all arms n=470:
- w5 CORR 0.026914, MMC −0.002776, pay −0.000084
- w22 CORR 0.037445, MMC 0.007463, pay 0.011208
- w65 CORR 0.062942, MMC 0.027828, pay 0.034123
- **w130 (best) CORR 0.080896, MMC 0.034176, pay 0.042265**

Direction (on the 60-day label, from the arms present): monotonic — more history is strictly better; the shortest window (w5) is essentially flat/negative (MMC −0.0028). The MLP's 60-day-label CORR (w130 0.081) is far below the catboost/rf 60-day arms, and the 20-day re-scored comparison that is the study's point is not recorded in available artifacts.
