# E98 — crbmc + FULL interactions vs raw (and vs xcurated E97): v2.0-crbmc-xfull-fillmask (2856f)

- **State:** done (5/5 arms)
- **Runner:** experiments/e98_crbmc_xfull_crypto.sh (reuses mommask_core.sh, VARIANT=xfull)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on crypto/v2.0-crbmc-xfull-fillmask (2856f), paired vs raw v2.0 (22f, reused e72_*_base) and vs crbmc-xcurated (E97)
- **Artifacts on disk:** 26.6 MB across 5 forecast runs, plus 5 metrics + 5 hyperparams + 5 per_period (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/catboost_ktz/e98_catboost_ktz_xfull.json (+ the other 4 arms and e72_*_base)

## Main question
The full-interaction twin of E97 and the biggest feature set run to date. Does the full pairwise expansion (2774 interactions on the 82-column crbmc base) drown ridge/MLP while leaving the trees roughly flat, and does the E97↔E98 contrast show curation beating brute-force expansion?

## What was tried
crypto/v2.0-crbmc-xfull-fillmask (2856f) = crbmc base (82) + 2774 full pairwise interactions (feature_x2_). 5 models from scratch at the shared protocol, 5 arms sequential (budget ~8-10h), reported 3-way vs raw and vs crbmc-xcurated (E97). Only the model class varies.

## Trained parameters
From results/metrics/crypto/ridge/e98_ridge_xfull.json config (representative): model ridge; data_version crypto/v2.0-crbmc-xfull-fillmask; target target_binned_return_20; cv_scheme rolling; train_window 130; refit_every 130; embargo 20; era_step 1; n_splits 5; val_frac 0.3; n_iter 50; search_type random; metric corr; n_features 2856; seed 42. All five arms share this protocol at n_features 2856; swept: model only.

## What was found
NUMBERS-ONLY. No written finding. From the metrics (mean CORR / MMC; level readings, block n=523, raw n=521) the full expansion clearly hurts, most at the linear/NN end: ridge collapses to 0.08404 / 0.03556 vs raw 0.14155 / 0.05801; mlp collapses to 0.06622 / 0.02723 vs raw 0.12865 / 0.05003. The trees survive but still trail raw — lgb 0.13586 / 0.04952 vs 0.14712 / 0.05925, rf 0.13145 / 0.04283 vs 0.14705 / 0.05887, catboost 0.12746 / 0.04664 vs 0.14924 / 0.05821. Best arm: lgb (CORR 0.13586). Against E97 (the 98f curated twin), every arm is lower under the full expansion — ridge 0.084 vs 0.117, lgb 0.136 vs 0.143, catboost 0.127 vs 0.143, rf 0.131 vs 0.149, mlp 0.066 vs 0.083 — so on these level readings curation beats brute-force expansion across the board. Direction, un-paired: HURTS vs raw, worst for ridge/MLP, matching the drowning prior (cf. E70 925f, E85 1506f). No paired NW delta vs raw in the sources read.
