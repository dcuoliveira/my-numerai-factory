# E81 — inflation block on raw baseline: raw+12 BEI (34f) vs raw v2.0

- **State:** done (5/5 arms)
- **Runner:** experiments/e81_inflonly_vs_raw_crypto.sh (shared mommask_core.sh, VARIANT=inflraw)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on `crypto/v2.0-infl` (34f)
- **Artifacts on disk:** 26.4 MB across 5 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §E81 row (status "registered", no finding cell); memory: none dedicated (referenced only as "E81 = same 12 BEI on raw" in the E80 index line); metrics: results/metrics/crypto/mlp_ktz/e81_mlp_ktz_inflraw.json

## Main question
The clean isolation of inflation's marginal value on the deployed baseline: does raw 22 + the 12 break-even-inflation features beat raw v2.0? Better-controlled sibling of E80, which buried the same 12 BEI features in the 61f momswap-macro stack.

## What was tried
`crypto/v2.0-infl` (rebuilt 2026-09-29) = raw 22 + 12 BEI features (`feature_macro_bei_*` levels + 20d/60d changes) = 34f, a clean superset of raw. The 12 BEI columns are day-level (inert for ridge on the within-date metric) but the raw 22 restore the cross-section, so ridge ≈ raw and only trees/MLP can use inflation as a regime tilt. 5 models from scratch at the shared protocol, reported 3-way vs raw v2.0 (`e72_*_base`) and momswap-macro-infl (`e80_*_infl`), both reused.

## Trained parameters
From results/metrics/crypto/mlp_ktz/e81_mlp_ktz_inflraw.json:
model mlp_ktz; data_version `crypto/v2.0-infl`; n_features 34; cv_scheme rolling, inner_cv purged; train_window 130, refit_every 130, embargo 20, min_train 0, era_step 1; n_splits 5, val_frac 0.3; seed 42; n_iter 50; search_type random; metric corr; target target_binned_return_20; fixed_params none. mlp refit-0 best_params: activation identity, alpha 0.01, batch_size 1024, hidden_layer_sizes [32], learning_rate_init 0.0375. Same protocol across all 5 arms; model class varied.

## What was found
NUMBERS-ONLY — no written finding exists; paired deltas vs raw are not recorded in the available artifacts. From the metrics (own 521-period windows, standalone): the four non-MLP arms are near-identical and tightly clustered — lgb 0.06943 / CORR 0.14340, ridge 0.07132 / CORR 0.14067, rf 0.06985, catboost 0.06889 — while mlp trails at payout 0.05481 / CORR 0.11087. The clustering (ridge ≈ trees) matches the design prior that the day-level BEI columns add little on a within-date metric. No paired t-stat against raw v2.0 is available on disk to establish help/null/hurt beyond these standalone levels.
