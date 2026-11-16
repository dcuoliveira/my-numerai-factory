# E59 — catboost_ktz 60-day tune of E32's top-5-MMC datasets

- **State:** done (5/5 arms)
- **Runner:** experiments/e59_catboost_ktz_tune_t60_crypto.sh
- **Models / datasets:** model `catboost_ktz` on catboost's five highest-MMC E32 datasets — fenccausal(44), tenc(23), allfeatpp(812), allfeat(788), tenc_onehot(763)
- **Artifacts on disk:** 21.17 MB across 5 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/catboost_ktz/e59_fenccausal.json, results/metrics/crypto/catboost_ktz/e59_tenc.json (+ allfeatpp, allfeat, tenc_onehot summarised)

## Main question
Does CatBoost's 20-day MMC strength (from E32) carry to the 60-day label and reach the 60-day book (study-stated: SACI_03 MMC 0.0788 / payout 0.0930, CURUPIRA_05_60 MMC 0.0804 / payout 0.0922)? CORR is NOT comparable to the 20-day E32 arms — read against the 60-day book only.

## What was tried
The CatBoost sibling of E57 (rf) and E58 (lgb). Catboost's own per-dataset MMC ordering picks a different five (keeps tenc where rf/lgb take interact). The ONLY intended change from E32's sel arms is the target (target_binned_return_60): same 12-cell KTZ grid (depth {2,4} x lr {.005,.01,.02} x rsm {.1,.2}; iterations 2000, l2_leaf_reg 10, min_data_in_leaf 100 pinned), same exhaustive `--n-iter 12` search, same rolling 130/130 cadence, n_splits 5, seed 42. `--embargo auto` derives 60 (thins inner CV to ~2 usable folds); ~470 periods. 5 arms.

## Trained parameters
From the metrics JSONs: model `catboost_ktz`, task regression, cv_scheme rolling, embargo 60, train_window 130, refit_every 130, n_splits 5, seed 42, n_iter 12, val_frac 0.3, target target_binned_return_60, 4 refits each, model_options {iterations 2000}. Representative winning arm e59_allfeatpp (n_features 812). Both inspected arms picked the same corner (fenccausal and tenc, refit 0): {depth 2, iterations 2000, l2_leaf_reg 10.0, learning_rate 0.005, min_data_in_leaf 100, rsm 0.1} — the shallowest, slowest, lowest-rsm cell of the grid.

## What was found
No written finding; from the metrics (all 5 arms, n=470, 60-day label), ranked by MMC:
- e59_allfeatpp:  CORR 0.13806, MMC 0.08002, payout(0.1·CORR+MMC) 0.09383
- e59_tenc:       CORR 0.15499, MMC 0.07910, payout 0.09460
- e59_allfeat:    CORR 0.13419, MMC 0.07728, payout 0.09070
- e59_tenc_onehot:CORR 0.12644, MMC 0.07340, payout 0.08605
- e59_fenccausal: CORR 0.11213, MMC 0.05592, payout 0.06713

Direction: the top catboost_ktz 60-day arm (allfeatpp, MMC 0.08002) essentially reaches the study-stated 60-day book MMC (SACI_03 0.0788, CURUPIRA_05_60 0.0804); tenc is close behind (0.0791) with the highest CORR of the five. CatBoost's MMC carries to 60-day at a level comparable to rf_ktz (E57). Per-period Sharpe/drawdown and paired deltas vs the book must be recomputed via the report tooling (the "payout" shown is the exact mean of the 0.1·CORR+MMC linear combination only).
