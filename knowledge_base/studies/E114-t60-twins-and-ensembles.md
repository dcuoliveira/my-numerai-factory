# E114 — 60-day twins of the E112 picks + 3/4-member ensembles with the book

- **State:** done (3/3 arms)
- **Runner:** experiments/e114_selected_t60_ensembles_crypto.sh (+ e114_ensembles.py / e114_allmom10_... screening)
- **Models / datasets:** lgb_ktz and catboost_ktz on crypto/v2.0-allmom-fillmask-tenc (58f), mlp_ktz on crypto/v2.0-allmom-fillmask-roll (228f); scored on target_binned_return_60, then ensembled on the payable 20-day label with CU_SA_IA_BO_02 and CU_SA_IA_BO_MA_R_01
- **Artifacts on disk:** 14.5 MB across 3 forecast runs (plus 3 metrics + 3 hyperparams + 3 per_period) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none; memory: e112-allmom-encodings.md; metrics: results/metrics/crypto/catboost_ktz/e114_catboost_ktz_tenc60.json

## Main question
Do the E112 picks fit better on the 60-day horizon, and does any 3/4-member ensemble of a pick's variants with the deployed book beat the book or either slot?

## What was tried
Fit the three E112 picks on target_binned_return_60 (embargo auto → 60) on the SAME datasets/protocol: lgb_ktz and catboost_ktz on crypto/v2.0-allmom-fillmask-tenc (58f), mlp_ktz on crypto/v2.0-allmom-fillmask-roll (228f) → e114_<m>_{tenc,roll}60. Paired in-target vs E110's allmom60 twins. Then, per model, every 3- and 4-member equal-weight rank average of six items — the model plain@20d, RESID@20d, plain@60d, RESID@60d (E107 transform, s=0.5, meta >= 5 days old), CU_SA_IA_BO_02, CU_SA_IA_BO_MA_R_01 — 35 combos × 3 models, scored on the payable 20-day label over one common window, paired (Newey-West lag 20) vs each slot alone and vs the best member, with calmar.

## Trained parameters
From results/metrics/crypto/catboost_ktz/e114_catboost_ktz_tenc60.json config: model catboost_ktz; data_version crypto/v2.0-allmom-fillmask-tenc; cv_scheme rolling; train_window 130; refit_every 130; embargo 60; inner_cv purged; n_splits 5; val_frac 0.3; n_iter 50; search_type random; metric corr; target target_binned_return_60; task regression; n_features 58; seed 42; fixed_params null; search_jobs 1. CORR.mean 0.15596, MMC.mean 0.06068, n=486. The lgb tenc60 and mlp roll60 twins share the protocol (lgb on the 58f tenc set, mlp on the 228f roll set).

## What was found
MEMORY-BACKED. No EXPERIMENTS.md section. The memory (E114 note, 2026-10-22) records: the tenc 60d twins are +0.005/+0.006 in-target vs the allmom60 twins (not significant). Across the 105 screened 3/4-member equal-weight ensembles of {plain20, resid20, plain60, resid60, CU_SA_IA_BO_02, CU_SA_IA_BO_MA_R_01}, NONE beats CU_SA_IA_BO_MA_R_01 on payout or either slot on calmar; every combo is below MA_R_01 at t −2 to −4. Direction: null-to-negative — the 60d twins add nothing deployable and no ensemble improves on the book. Fit in ~11 minutes.
