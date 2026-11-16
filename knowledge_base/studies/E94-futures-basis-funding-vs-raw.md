# E94 — futures-basis & funding block vs raw: crypto/v2.0-basis-fillmask (32f) vs v2.0

- **State:** done (5/5 arms)
- **Runner:** experiments/e94_basis_vs_raw_crypto.sh (reuses mommask_core.sh, VARIANT=basis)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on crypto/v2.0-basis-fillmask (32f), paired vs raw v2.0 (22f, reused e72_*_base) and the vol block (e74_*_vol)
- **Artifacts on disk:** 25.7 MB across 5 forecast runs, plus 5 metrics + 5 hyperparams + 5 per_period (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: e90-e94-statarb-block-sweep.md (frames the question; records E94 as "results pending"), report-calmar-convention.md (SMTP note only); metrics: results/metrics/crypto/mlp_ktz/e94_mlp_ktz_basis.json, results/metrics/crypto/lgb_ktz/e94_lgb_ktz_basis.json (+ the other 3 arms and e72_*_base)

## Main question
The first genuinely new data class in this program — perp funding + spot-futures basis, not a re-transform of the 22. Does a direct carry / short-constraint measure (crowded longs via persistently positive funding, where high-beta overpricing should concentrate) beat raw? The e90-e94 memory flags this as the test of whether genuinely new data breaks the "nothing beats raw" pattern.

## What was tried
crypto/v2.0-basis-fillmask = raw 22 + 10 features: feature_basis_pct, basis_ma_20d, basis_z_20d, basis_z_60d, basis_chg_5d, basis_chg_20d, feature_funding_rate, funding_ma_7d, funding_ma_30d, funding_z_30d. Per-symbol, binned (~5 levels/date), point-in-time; a coin with no listed perp has no funding and is neutral-filled (fillmask = loader fill), and that absence is itself the hard-to-short signal. 5 models from scratch at the shared protocol, reported 3-way vs raw and the vol block. Only the model class varies.

## Trained parameters
From results/metrics/crypto/catboost_ktz/e94_catboost_ktz_basis.json config (representative): model catboost_ktz; data_version crypto/v2.0-basis-fillmask; target target_binned_return_20; cv_scheme rolling; train_window 130; refit_every 130; embargo 20; era_step 1; n_splits 5; val_frac 0.3; n_iter 50; search_type random; metric corr; n_features 32; seed 42; fixed_params null. The ridge arm is the same protocol at n_features 32. Swept: model only.

## What was found
NUMBERS-ONLY. No EXPERIMENTS.md section, and the e90-e94 memory recorded E94 as "results pending" (launched 2026-10-06). From the metrics (mean CORR / MMC, block vs its raw twin, level readings over near-identical windows — block n=523, raw n=521): lgb 0.14995 / 0.06133 vs raw 0.14712 / 0.05925 (the one arm above raw on both CORR and MMC, +0.003 / +0.002); rf 0.14648 / 0.05977 ≈ raw 0.14705 / 0.05887; catboost 0.14228 / 0.05461 below; ridge 0.13486 / 0.05284 below; mlp 0.10678 / 0.03903 well below raw 0.12865 / 0.05003. Best arm: lgb (CORR 0.14995, CORR dd 0.527, stored pnl-series mean 0.03719 sharpe 1.34). Direction, un-paired: null-to-marginal — basis/funding lifts only lgb slightly on this level comparison and leaves the others at or below raw; no paired Newey-West delta vs raw was in the sources read, so the "does new data beat raw" verdict is not established here beyond the level readings.
