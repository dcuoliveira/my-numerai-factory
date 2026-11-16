# E112 — symbol encodings on top of all-momentum: fenccausal / tenc / rolling means / all vs allmom

- **State:** done (20/20 arms)
- **Runner:** experiments/e112_allmom_encodings_crypto.sh (4 concurrent lanes via mommask_core.sh; combined report experiments/e112_report.py)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on four supersets of crypto/v2.0-allmom-fillmask: -fenccausal (114f), -tenc (58f), -roll (228f), -all (286f); base e108_*_allmom reused
- **Artifacts on disk:** 103.3 MB across 20 forecast runs (plus 20 metrics + 20 hyperparams + 20 per_period) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none; memory: e112-allmom-encodings.md; metrics: results/metrics/crypto/lgb_ktz/e112_lgb_ktz_tenc.json, results/metrics/crypto/catboost_ktz/e112_catboost_ktz_fenc.json

## Main question
Which of the three per-symbol encodings adds to crypto/v2.0-allmom-fillmask (57f, E108) when stacked on it? Prior: fenccausal is what makes IA_01 work on the raw 22 (MLP); tenc and rolling means rarely moved trees on raw; the 286f stack risks ridge dilution (E82/E85).

## What was tried
Four supersets, each differing from the 57f allmom base ONLY by the encoding: -fenccausal (114f, +57 feature_<f>_symmean = causal expanding mean of each feature over the symbol's strictly earlier dates, shrunk m=50), -tenc (58f, +feature_sym_tenc = mean of the symbol's own 20d target over resolved dates <= t-20, shrunk m=50), -roll (228f, +171 feature_<f>_rm{5,20,60} per-symbol trailing means incl today), -all (286f, all three). 4 datasets × 5 models from scratch at the shared protocol (rolling 130/130, embargo auto=20, 5-fold, val_frac 0.3, seed 42, n_iter 50, target_binned_return_20) = 20 arms; base reused (e108_<m>_allmom), each arm paired (Newey-West lag 20) vs its own model's allmom twin. A 7-agent leak audit found no leak (datasets are the repo builders at their defaults).

## Trained parameters
From results/metrics/crypto/lgb_ktz/e112_lgb_ktz_tenc.json config: model lgb_ktz; data_version crypto/v2.0-allmom-fillmask-tenc; cv_scheme rolling; train_window 130; refit_every 130; embargo 20; inner_cv purged; n_splits 5; val_frac 0.3; n_iter 50; search_type random; metric corr; target target_binned_return_20; n_features 58; seed 42; fixed_params null; search_jobs 4. CORR.mean 0.15050, MMC.mean 0.06351, n=526. The catboost fenc twin (e112_catboost_ktz_fenc.json): data_version crypto/v2.0-allmom-fillmask-fenccausal, n_features 114, model catboost_ktz; CORR.mean 0.12938, MMC.mean 0.05348, n=526.

## What was found
MEMORY-BACKED. No EXPERIMENTS.md section (the memory notes a row was written, but the manifest records no line range). Final (single seed, 496 periods, done 2026-10-21): only tenc helps. lgb +0.0071 (t +2.88, calmar 0.038→0.062) and catboost +0.0059 (t +2.75, calmar 0.044→0.062), both via MMC. Calmar improves for all 5 models (shallower drawdown). fenc lowers CORR in 4 of 5 (catboost ΔCORR −0.016, t −2.54); roll never helps (mlp −0.0245, t −2.46); all is flat (the tenc gain is drowned). Ridge results: payout vs base 0.06234 → fenc 0.06383, tenc 0.06349, roll 0.05808, all 0.05697. Best arms are ~0.080 vs the book's 0.1047 (435 days). Caveat recorded: ridge alpha pins at 65.79 (top of a grid capped at 100) in most refits, so a null ridge result is weak evidence. Blend test (2026-10-22, e112_blend_with_slots.py, 435 days): lgb/cat tenc and mlp+roll added at 10–50% into CU_SA_IA_BO_02 and CU_SA_IA_BO_MA_R_01 — calmar fell in all 80 blends, best payout +0.0012 (t +0.36, noise); not deployable as a blend leg. (E114 fits the 60d twins and screens ensembles.)
