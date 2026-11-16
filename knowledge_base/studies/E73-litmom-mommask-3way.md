# E73 — litmom with base masked on momentum-missing rows (mommask), 3-way

- **State:** done (5/5 arms)
- **Runner:** `experiments/e73_litmom_mommask_crypto.sh`
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on `crypto/v2.0-litmom-mommask` (39f); compared 3-way vs reused E65 arms on `crypto/v2.0` (base) and `crypto/v2.0-litmom`
- **Artifacts on disk:** 26.9 MB across 10 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none (planning-table row, status "registered"); memory: e73-litmom-mommask.md, e74-vol-estimators.md (notes the on-disk recipe neutralises the whole gated row); metrics: results/metrics/crypto/ridge/e73_ridge_mommask.json (+ all 5 mommask arms, + E65 base/litmom for the 3-way)

## Main question
`crypto/v2.0-litmom-mommask` = litmom (39f) with features set to neutral 0.5 in ~17% of cells, CONCENTRATED on momentum-missing rows (P(base masked | momentum missing)=0.996 vs 0.263 when present). Masking only removes information (→0.5), so it is leakage-safe. The question is diagnostic, not a search for a win: how much does hiding the base block on the no-spot rows (which already lean entirely on the base) cost numerai_corr? A significant drop quantifies how much the base block carries the no-spot names.

## What was tried
Only the 5 mommask arms run (from scratch, at exactly E65's protocol), then a 3-way comparison reusing E65's `e65_*_base` (v2.0 raw 22) and `e65_*_litmom` (litmom 39f) arms. (Caveat from the e74 memory: the on-disk mommask recipe fully neutralises the gated row — base AND the gate block → 0.5 — so the E73 header's "momentum columns identical" is inaccurate for the gated cells.) Prior: masking info should not help; null-or-worse expected.

## Trained parameters
From the metrics JSONs: `cv_scheme=rolling`, `train_window=130`, `refit_every=130`, `n_splits=5`, `embargo=20`, `val_frac=0.3`, `metric=corr`, `search_type=random`, `n_iter=50`, `seed=42`, `target=target_binned_return_20`. `fixed_params` = null (ktz priors in the search grid). `n_features`: 39 (mommask) vs 22 (E65 base) / 39 (E65 litmom). n=520.

## What was found
No written finding; from the metrics, mommask is well below BOTH the raw-22 base and litmom on every model — masking the base on momentum-missing rows clearly degrades, confirming the base block carries the no-spot names. CORR (mommask / E65 base / E65 litmom): ridge 0.112006 / 0.144228 / 0.136542; lgb 0.123042 / 0.147709 / 0.145546; catboost 0.125453 / 0.151131 / 0.148397; rf 0.127726 / 0.149624 / 0.154826; mlp 0.105157 / 0.142588 / 0.132460. Payout (mommask / E65 base): ridge 0.048444 / 0.073615; lgb 0.056129 / 0.074292; catboost 0.059778 / 0.074376; rf 0.061294 / 0.077261; mlp 0.053507 / 0.073571. So mommask − base is roughly −0.02 to −0.04 CORR and −0.003 to −0.009 payout across all 5 models; the drop is largest for ridge and mlp. The study behaves as designed: it quantifies the cost of hiding the base block, it is not a candidate to deploy. Paired Newey-West t-stats are not recorded in the artifacts.
