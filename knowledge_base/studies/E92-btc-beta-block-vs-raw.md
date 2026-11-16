# E92 — BTC-beta block vs raw: crypto/v2.0-beta (raw 22 + 3 BTC-beta) vs v2.0

- **State:** done (5/5 arms, + 5 scout screens)
- **Runner:** experiments/e92_beta_vs_raw_crypto.sh
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on crypto/v2.0-beta (25f), paired vs raw v2.0 (22f, reused e72_*_base)
- **Artifacts on disk:** 28.7 MB across 10 forecast runs (5 full e92_* + 5 e92scout_*), plus 10 metrics + 10 hyperparams + 10 per_period (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none; memory: e90-e94-statarb-block-sweep.md; metrics: results/metrics/crypto/catboost_ktz/e92_catboost_ktz_beta.json (+ the other 4 arms and e72_*_base)

## Main question
Does a coin's rolling market beta to BTC add signal to the raw panel? crypto/v2.0-beta is a strict superset of crypto/v2.0 — the raw 22 plus exactly 3 columns, feature_beta_btc_30d/_60d/_90d (each coin's rolling OLS slope of its returns on Bitcoin over 30/60/90 days). A clean leave-the-rest-fixed ablation of the BTC-beta block.

## What was tried
5 models tuned from scratch on v2.0-beta at the shared protocol, so each arm pairs validly (Newey-West lag 20) against its raw twin e72_*_base, which is reused rather than re-run. Only the 5 beta arms were fit. A scout screen (e92scout_*, ~0.9 MB each) preceded the full runs. The single varying factor across the 5 cells is the model class; the dataset and protocol are fixed.

## Trained parameters
From results/metrics/crypto/catboost_ktz/e92_catboost_ktz_beta.json config (representative; identical across arms except model): model catboost_ktz; data_version crypto/v2.0-beta; target target_binned_return_20; task regression; cv_scheme rolling; train_window 130; refit_every 130; embargo 20; era_step 1; n_splits 5; val_frac 0.3; n_iter 50; search_type random; metric corr; n_features 25; seed 42; fixed_params null. The ridge arm shares the same protocol (n_features 25, fixed_params null). What swept across the other arms is only the model (ridge / lgb_ktz / rf_ktz / mlp_ktz).

## What was found
MEMORY-BACKED. No dedicated EXPERIMENTS.md section. The e90-e94 sweep memory does not record an E92-specific paired delta (it notes E92 was run by a concurrent session) but its standing conclusion covers this block: "recombining the price-derived 22 (momentum/vol/carver/interactions/reversal/coint/beta/BAB) does NOT beat raw v2.0." The metrics I read agree: every beta arm's mean CORR sits below its raw twin — ridge 0.12840 vs raw 0.14155, lgb 0.13570 vs 0.14712, catboost 0.13266 vs 0.14924, rf 0.13951 vs 0.14705, mlp 0.12351 vs 0.12865; MMC is likewise lower in all five (e.g. rf 0.05239 vs raw 0.05887). The best beta arm (rf, CORR 0.13951) still trails raw-rf. These are level readings — the beta arms carry n=523 periods and e72_*_base n=521, so they are not the paired NW deltas the runner's report computes, but the gap is one-directional and well outside a two-period window difference. Direction: the BTC-beta block is null-to-negative vs raw, matching the memory's prior (beta is a market/regime signal with ~zero standalone within-date IC).
