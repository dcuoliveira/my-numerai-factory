# E77 — carver mommask, 3-way vs carver + v2.0

- **State:** done (5/5 arms)
- **Runner:** `experiments/e77_carver_mommask_crypto.sh` (thin wrapper on shared `mommask_core.sh`)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on `crypto/v2.0-momswap-macro-carver-mommask` (64f); compared 3-way vs reused `e72_*_carver` (carver 64f) and `e72_*_base` (raw v2.0 22f)
- **Artifacts on disk:** 22.7 MB across 5 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none (planning-table row, status "registered"); memory: mommask-sweep-e75-e77.md; metrics: results/metrics/crypto/catboost_ktz/e77_catboost_ktz_mommask.json (+ all 5 mommask arms; reused E72 carver/base)

## Main question
The carver analog of the mommask sweep. `crypto/v2.0-momswap-macro-carver-mommask` (64f) = the carver set (momswap-macro + 15 Carver/Clenow trend-following features) with features masked to neutral 0.5 concentrated on momentum-missing rows. Diagnostic: masking under-informed rows should not help.

## What was tried
Only the 5 mommask arms run from scratch at the shared protocol; reported 3-way against reused `e72_*_carver` (carver 64f) and `e72_*_base` (raw v2.0 22f). Shared `mommask_core.sh`. Prior: null-or-worse.

## Trained parameters
From the metrics JSONs: `cv_scheme=rolling`, `train_window=130`, `refit_every=130`, `n_splits=5`, `embargo=20`, `val_frac=0.3`, `metric=corr`, `search_type=random`, `n_iter=50`, `seed=42`, `target=target_binned_return_20`. `fixed_params` = null (ktz priors in the search grid). `n_features`: 64 (mommask), vs 64 (E72 carver) / 22 (E72 raw). n=520.

## What was found
No per-study written finding. From the metrics, mommask is below BOTH the carver parent and raw v2.0 on every model. CORR (mommask / E72 carver / E72 raw): ridge 0.117889 / 0.133413 / 0.141550; lgb 0.131686 / 0.151201 / 0.147116; catboost 0.130524 / 0.149120 / 0.149235; rf 0.124027 / 0.152466 / 0.147053; mlp 0.113538 / 0.131058 / 0.128652. Payout (mommask / E72 carver): ridge 0.060845 / 0.068139; lgb 0.061128 / 0.072067; catboost 0.061543 / 0.071387; rf 0.056465 / 0.073215; mlp 0.058755 / 0.069462. So mommask − carver is roughly −0.015 to −0.028 CORR on all 5 models and below raw v2.0 as well; masking under-informed rows only removes signal, as predicted. Paired Newey-West t-stats are not recorded in the artifacts.
