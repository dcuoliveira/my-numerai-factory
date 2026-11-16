# E78 — vol mommask, 3-way vs vol + v2.0

- **State:** done (5/5 arms)
- **Runner:** `experiments/e78_vol_mommask_crypto.sh` (thin wrapper on shared `mommask_core.sh`)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on `crypto/v2.0-vol-mommask` (44f); compared 3-way vs reused `e74_*_vol` (vol 44f) and `e72_*_base` (raw v2.0 22f)
- **Artifacts on disk:** 22.8 MB across 5 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none (planning-table row, status "registered"); memory: e74-vol-estimators.md (describes the E78 gate), e82-e85-combo-fenccausal.md; metrics: results/metrics/crypto/rf_ktz/e78_rf_ktz_mommask.json (+ all 5 mommask arms; reused E74 vol / E72 raw)

## Main question
The volatility analog of the mommask sweep, keyed on the vol block instead of momentum. `crypto/v2.0-vol-mommask` (44f) = the vol set (raw 22 + 22 `feature_vol_` estimators) with the WHOLE row set to neutral 0.5 wherever any of the 22 vol features is missing (`build_mommask --gate created --require all --neutralize all`). 90,164/514,696 rows gated (17.5%) — the no-spot names plus rows with <120d history (gates less than litmom-mommask's 34.4%, whose `feature_mom_ret_12_1` needs 360d). Diagnostic: masking under-informed rows should not help; a drop quantifies how much the vol block carries the no-spot/young names.

## What was tried
Only the 5 mommask arms run from scratch at the shared protocol; reported 3-way against reused `e74_*_vol` (vol 44f) and `e72_*_base` (raw v2.0 22f). Shared `mommask_core.sh`. Prior: null-or-worse.

## Trained parameters
From the metrics JSONs: `cv_scheme=rolling`, `train_window=130`, `refit_every=130`, `n_splits=5`, `embargo=20`, `val_frac=0.3`, `metric=corr`, `search_type=random`, `n_iter=50`, `seed=42`, `target=target_binned_return_20`. `fixed_params` = null (ktz priors in the search grid). `n_features`: 44 (mommask), vs 44 (E74 vol) / 22 (E72 raw). n=519 (mlp arm n=520).

## What was found
No per-study written finding. From the metrics, mommask is below BOTH the vol parent and raw v2.0 on every model. CORR (mommask / E74 vol / E72 raw): ridge 0.125058 / 0.145434 / 0.141550; lgb 0.129792 / 0.149565 / 0.147116; catboost 0.134574 / 0.154524 / 0.149235; rf 0.133098 / 0.151009 / 0.147053; mlp 0.082192 / 0.127541 / 0.128652. Payout (mommask / E74 vol): ridge 0.059217 / 0.075193; lgb 0.058532 / 0.071448; catboost 0.064099 / 0.075861; rf 0.062373 / 0.070250; mlp 0.040348 / 0.068097. So mommask − vol is roughly −0.02 to −0.045 CORR across the models; the mlp arm collapses hardest (CORR 0.082192, c.sharpe 0.69, dd(corr) 4.45). Gating under-informed rows only removes signal — the vol block does carry real information on the no-spot/young names. Paired Newey-West t-stats are not recorded in the artifacts.
