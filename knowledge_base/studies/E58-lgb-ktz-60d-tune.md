# E58 — lgb_ktz 60-day tune of E29's top-5-MMC datasets

- **State:** done (5/5 arms)
- **Runner:** experiments/e58_lgb_ktz_60d_tune_crypto.sh
- **Models / datasets:** model `lgb_ktz` on the five highest-MMC E29 datasets — fenccausal(44), interact(36), allfeat(788), tenc_onehot(763), allfeatpp(812)
- **Artifacts on disk:** 19.58 MB across 5 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/lgb_ktz/e58_fenccausal.json, results/metrics/crypto/lgb_ktz/e58_allfeat.json (+ interact, tenc_onehot, allfeatpp summarised)

## Main question
Does LightGBM's 20-day MMC strength (from E29) carry to the 60-day label and reach the 60-day book (study-stated: SACI_03 MMC 0.0788 / payout 0.0930, CURUPIRA_05_60 MMC 0.0804 / payout 0.0922)? Secondary paired read vs E01's 60d m50 pins.

## What was tried
The LightGBM sibling of E57 and the 60-day twin of E29's five best arms. The ONLY change from E29 is the target (target_binned_return_60): same lgb_ktz grid (E27 priors, L=lr*n_est<=40), same 60-draw from-scratch RandomizedSearchCV, same rolling 6-month/6-month (130/130) cadence, n_splits 5, seed 42, default python3. `--embargo auto` derives 60; ~470 periods. Excludes the leaky non-causal global encoding (fencglobal) and all blend/ensemble arms. 5 arms, one per dataset.

## Trained parameters
From the metrics JSONs: model `lgb_ktz`, task regression, cv_scheme rolling, embargo 60, train_window 130, refit_every 130, n_splits 5, seed 42, n_iter 60, val_frac 0.3, target target_binned_return_60, 4 refits each, no fixed_params. Representative winning arm e58_allfeatpp (n_features 812). Example best_params (e58_fenccausal, refit 0): {colsample_bytree 0.11, learning_rate 0.00123, max_depth -1, min_child_samples 100, n_estimators 4000, num_leaves 2, reg_lambda 10.0, subsample 0.8, subsample_freq 1} — the signature very-slow-shrinkage, 2-leaf KTZ booster.

## What was found
No written finding; from the metrics (all 5 arms, n=470, 60-day label), ranked by MMC:
- e58_allfeatpp:  CORR 0.15811, MMC 0.07777, payout(0.1·CORR+MMC) 0.09358
- e58_tenc_onehot:CORR 0.14628, MMC 0.07481, payout 0.08944
- e58_allfeat:    CORR 0.13668, MMC 0.06977, payout 0.08343
- e58_interact:   CORR 0.13850, MMC 0.06619, payout 0.08004
- e58_fenccausal: CORR 0.11832, MMC 0.06350, payout 0.07533

Direction: the best lgb_ktz 60-day arm (allfeatpp, MMC 0.07777) falls just short of the study-stated 60-day book MMC targets (0.0788 / 0.0804); LightGBM's MMC carries less cleanly to 60-day than rf_ktz did in E57 (where tenc_onehot reached 0.0826). Per-period Sharpe/drawdown and paired deltas vs the book/E01 must be recomputed via the report tooling (the "payout" shown is the exact mean of the 0.1·CORR+MMC linear combination only).
