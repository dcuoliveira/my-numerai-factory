# E57 — rf_ktz 60-day tune of E52's top-5-MMC datasets

- **State:** done (5/5 arms)
- **Runner:** experiments/e57_rf_ktz_60d_tune_crypto.sh
- **Models / datasets:** model `rf_ktz` on the five highest-MMC E52 datasets — fenccausal(44), interact(36), allfeat(788), allfeatpp(812), tenc_onehot(763)
- **Artifacts on disk:** 20.59 MB across 5 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/rf_ktz/e57_fenccausal.json, results/metrics/crypto/rf_ktz/e57_allfeat.json (+ allfeatpp, tenc_onehot, interact summarised)

## Main question
Does random forest's 20-day MMC strength (from E52) carry to the 60-day label, and does it reach the 60-day book (study-stated book targets: SACI_03 MMC 0.0788 / payout 0.0930, CURUPIRA_05_60 MMC 0.0804 / payout 0.0922)?

## What was tried
The 60-day twin of E52's five highest-MMC arms. The ONLY change from E52 is the target (target_binned_return_60 instead of _20): same rf_ktz grid (E48 priors), same 40-draw from-scratch RandomizedSearchCV, same rolling 130/130 cadence, n_splits 5, seed 42, same sklearn 0.24 venv. `--embargo auto` derives 60 from the horizon; 60-day runs score ~470 periods. 5 arms, one per dataset.

## Trained parameters
From the metrics JSONs: model `rf_ktz`, task regression, cv_scheme rolling, embargo 60, train_window 130, refit_every 130, n_splits 5, seed 42, n_iter 40, val_frac 0.3, target target_binned_return_60, 4 refits each, no fixed_params. Representative winning arm e57_tenc_onehot (n_features 763). Example best_params (e57_fenccausal, refit 0): {bootstrap True, max_depth 4, max_features 0.1, min_samples_leaf 300, n_estimators 100}; the search swept max_depth {3..6}, max_features {0.05..0.3}, min_samples_leaf {300..4000}, n_estimators fixed 100.

## What was found
No written finding; from the metrics (all 5 arms, n=470, 60-day label), ranked by MMC:
- e57_tenc_onehot: CORR 0.16129, MMC 0.08264, payout(0.1·CORR+MMC) 0.09877
- e57_allfeatpp: CORR 0.16928, MMC 0.08139, payout 0.09832
- e57_allfeat:   CORR 0.15831, MMC 0.08043, payout 0.09626
- e57_interact:  CORR 0.15761, MMC 0.07584, payout 0.09160
- e57_fenccausal:CORR 0.12859, MMC 0.07058, payout 0.08344

The top rf_ktz arm (tenc_onehot, MMC 0.08264) edges past both study-stated 60-day book MMC targets (SACI_03 0.0788, CURUPIRA_05_60 0.0804); three of five arms exceed 0.080 MMC. Direction: RF's MMC strength does appear to carry to the 60-day label at the dataset level. Per-period Sharpe/drawdown and paired deltas against the book must be recomputed via the report tooling (the "payout" here is the exact mean of 0.1·CORR+MMC, a linear combination of the two stored means — not a Sharpe-bearing quantity).
