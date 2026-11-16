# E127 — E121's CatBoost training-window ladder, fitted on the 60-day label, scored on the 20-day one

- **State:** not started (0 full arms; 1 scout run on disk, `e127scout_w5`)
- **Runner:** `experiments/e127_catboost_train_window_t60_crypto.sh` (via E121's runner, `TW_TARGET=60`)
- **Models / datasets:** catboost_ktz on `crypto/v2.0-fenccausal` (IAUARETE_01's setup, 44 feat); fitted/tuned on `target_binned_return_60`
- **Artifacts on disk:** 4.36 MB across 1 forecast run (scout only) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §(table row, line 118 — no dedicated section, status "registered", no result); memory: none; metrics: `results/metrics/crypto/catboost_ktz/e127scout_w5.json`

## Main question
The CatBoost twin of E124 (ridge), E125 (MLP) and E126 (lgb): does fitting CatBoost on the smoother 60-day horizon and deploying on the 20-day payable label beat fitting on 20, and does it depend on the window?

## What was tried
Planned: E121's arms — `catboost_ktz` on `crypto/v2.0-fenccausal` (44 feat), rolling windows 5/10/15/20/22/65/130, the exhaustive 12-cell KTZ grid searched from scratch every refit, `--tune-window 130` below 130, refit 130 — but FITTED and TUNED on `target_binned_return_60` with embargo 60. At embargo 60 a 130-period tuning block keeps only the last two of its five inner folds (exactly production's own 60-day search, `CURUPIRA_05_60`). Every arm to be RE-SCORED on `target_binned_return_20` via `cross_target.score_on_payable_target` and consolidated with E121's fit-20 arms (`src.analysis.e127_report`), paired NW lag 20 and 60 and vs the deployed `e121_w130` (IAUARETE_01). Only a single scout arm (w5) exists on disk.

## Trained parameters
From the one scout file `results/metrics/crypto/catboost_ktz/e127scout_w5.json`: model `catboost_ktz`, data_version `crypto/v2.0-fenccausal`, n_features 44, n_iter 12, refit_every 130, embargo 60, train_window 5, tune_window 130, seed 42, target `target_binned_return_60`, model_options `{"iterations": 300}`, fixed_params null. 4 refits. Best_params refit 0: `depth` 2, `iterations` 300, `l2_leaf_reg` 10.0, `learning_rate` 0.005, `min_data_in_leaf` 100, `rsm` 0.2; refit 3: same but `learning_rate` 0.02. (The scout uses iterations 300, lighter than E121's 2000.)

## What was found
No written finding in EXPERIMENTS.md (status "registered"); the full ladder is not started. NUMBERS-ONLY from the single scout, and on the 60-day label only (the payable 20-day re-score that is the study's deliverable is NOT in this JSON or any source read here). `e127scout_w5` (train_window 5, n=470): CORR 0.139908, MMC 0.062460, payout (0.1·CORR + MMC from summary means, ranking-grade only) 0.076451. One scout arm cannot establish a window trend or the 20-day re-scored comparison; those are not recorded in available artifacts.
