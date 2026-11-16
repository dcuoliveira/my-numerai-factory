# E82 — kitchen-sink combo (raw+vol+infl+carver+ix, 753f) vs raw v2.0

- **State:** done (5/5 arms)
- **Runner:** experiments/e82_vol_infl_carver_ix_combo_crypto.sh (shared mommask_core.sh, VARIANT=combo)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on `crypto/v2.0-vol-infl-carver-ix-fillmask` (753f)
- **Artifacts on disk:** 26.4 MB across 5 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §E82 row (status "registered", no finding cell); memory: e82-e85-combo-fenccausal.md; metrics: results/metrics/crypto/catboost_ktz/e82_catboost_ktz_combo.json

## Main question
Does stacking every individually-null engineered block finally beat raw v2.0?

## What was tried
`crypto/v2.0-vol-infl-carver-ix-fillmask` (753f), a clean superset of raw: raw 22 + `feature_vol_` 22 (E74) + `feature_macro_bei_` 12 (inflation, E80/E81) + `feature_tf_` 15 (Carver, E72) + `feature_ix_` 682 (interactions, dominating the count). fillmask = missing cells pre-filled to 0.5 (== the loader's own fill). 5 models from scratch at the shared protocol, reported 3-way vs raw v2.0 (`e72_*_base`) and momswap-macro (`e70_*_base`), both reused.

## Trained parameters
From results/metrics/crypto/catboost_ktz/e82_catboost_ktz_combo.json:
model catboost_ktz; data_version `crypto/v2.0-vol-infl-carver-ix-fillmask`; n_features 753; cv_scheme rolling, inner_cv purged; train_window 130, refit_every 130, embargo 20, min_train 0, era_step 1; n_splits 5, val_frac 0.3; seed 42; n_iter 50; search_type random; metric corr; target target_binned_return_20; fixed_params none. catboost refit-0 best_params: depth 2, iterations 2000, l2_leaf_reg 10.0, learning_rate 0.005, min_data_in_leaf 100, rsm 0.1. Same protocol across all 5 arms; model class varied.

## What was found
MEMORY-BACKED (e82-e85-combo-fenccausal.md). **Every arm is BELOW raw.** The huge interaction count drowns the linear/MLP models: ridge t −3.38, catboost t −3.46, mlp t −3.29 vs raw; lgb is least-bad (CORR 0.140, not significant). Ranked dead-last vs the book: best single arm lgb payout 0.065, dpay −0.027 (t −2.38) vs SACI_06. **No 2–3 model ensemble helps** — all blends 0.063–0.065 payout, not significant vs the best single, all below raw (0.073) and far below the book (0.104); tree arms mutually correlated, ridge/mlp decorrelated-but-weaker. Priced blend scratch left on disk as lgb_ktz/e82ens_{lr,lrc,lrm,lcm,all}. From the metrics (own 521-period windows, standalone): best lgb 0.03615 / CORR 0.13995, then rf 0.03536, catboost 0.03279, ridge 0.02805, mlp 0.02620.
