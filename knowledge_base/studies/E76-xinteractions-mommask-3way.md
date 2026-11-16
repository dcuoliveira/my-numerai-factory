# E76 — xinteractions mommask, 3-way vs xinteractions + v2.0

- **State:** done (5/5 arms)
- **Runner:** `experiments/e76_xinteractions_mommask_crypto.sh` (thin wrapper on shared `mommask_core.sh`)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on `crypto/v2.0-momswap-macro-xinteractions-mommask` (219f); compared 3-way vs reused `e71_*_xint` and `e72_*_base` (raw v2.0 22f)
- **Artifacts on disk:** 22.8 MB across 5 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none (planning-table row, status "registered"); memory: mommask-sweep-e75-e77.md, e79-fillmask-equals-loader.md; metrics: results/metrics/crypto/catboost_ktz/e76_catboost_ktz_mommask.json (+ all 5 mommask arms; reused E71/E72 bases)

## Main question
The xinteractions analog of the mommask sweep. `crypto/v2.0-momswap-macro-xinteractions-mommask` (219f) = the 219-feature xinteractions set (momswap-macro + macro-mom × crypto-mom products) with features masked to neutral 0.5 concentrated on momentum-missing rows. Diagnostic: masking under-informed rows should not help.

## What was tried
Only the 5 mommask arms run from scratch at the shared protocol; reported 3-way against reused `e71_*_xint` and `e72_*_base` (raw v2.0). `e71_ridge_xint` (missing from E71) was run separately to complete the ridge cell. Shared `mommask_core.sh`. Prior: null-or-worse.

## Trained parameters
From the metrics JSONs: `cv_scheme=rolling`, `train_window=130`, `refit_every=130`, `n_splits=5`, `embargo=20`, `val_frac=0.3`, `metric=corr`, `search_type=random`, `n_iter=50`, `seed=42`, `target=target_binned_return_20`. `fixed_params` = null (ktz priors in the search grid). `n_features`: 219 (mommask), vs 219 (E71 xint, as recorded on disk) / 22 (E72 raw). n=520.

## What was found
No per-study written finding (e79 memory covers the general mommask<fill direction). From the metrics, mommask is below BOTH the xinteractions parent and raw v2.0 on every model. CORR (mommask / E71 xint / E72 raw): ridge 0.100429 / 0.112212 / 0.141550; lgb 0.126377 / 0.150747 / 0.147116; catboost 0.124171 / 0.138949 / 0.149235; rf 0.127413 / 0.133176 / 0.147053; mlp 0.099529 / 0.109938 / 0.128652. Payout (mommask / E71 xint): ridge 0.049473 / 0.053008; lgb 0.060984 / 0.072720; catboost 0.061334 / 0.064717; rf 0.061223 / 0.061668; mlp 0.049447 / 0.058519. So mommask trails the already-weak xint parent on all 5 models and trails raw v2.0 by more (roughly −0.015 to −0.04 CORR vs raw). Masking only removes signal, as predicted. Paired Newey-West t-stats are not recorded in the artifacts.
