# E75 — momswap-macro mommask, 3-way vs momswap-macro + v2.0

- **State:** done (5/5 arms)
- **Runner:** `experiments/e75_momswap_macro_mommask_crypto.sh` (thin wrapper on shared `mommask_core.sh`)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on `crypto/v2.0-momswap-macro-mommask` (49f); compared 3-way vs reused `e70_*_base` (momswap-macro 49f) and `e72_*_base` (raw v2.0 22f)
- **Artifacts on disk:** 26.8 MB across 10 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none (planning-table row, status "registered"); memory: mommask-sweep-e75-e77.md, e79-fillmask-equals-loader.md (general mommask<fill finding); metrics: results/metrics/crypto/catboost_ktz/e75_catboost_ktz_mommask.json (+ all 5 mommask arms; reused E70/E72 bases)

## Main question
The momswap-macro analog of E73's mommask (E74 was taken by the vol study). `crypto/v2.0-momswap-macro-mommask` (49f) = momswap-macro with features set to neutral 0.5 concentrated on momentum-missing rows (P(base masked | mom missing)=0.961). Unlike E73 it masks ALL groups: base ~17% / momentum ~10% / macro ~34% of cells (5.27M extra neutral cells). Diagnostic: does masking under-informed rows help (it should not), and how much is lost?

## What was tried
Only the 5 mommask arms run from scratch at the shared protocol; reported 3-way against reused `e70_*_base` (momswap-macro 49f) and `e72_*_base` (raw v2.0 22f). Shared `mommask_core.sh`. Prior: null-or-worse.

## Trained parameters
From the metrics JSONs: `cv_scheme=rolling`, `train_window=130`, `refit_every=130`, `n_splits=5`, `embargo=20`, `val_frac=0.3`, `metric=corr`, `search_type=random`, `n_iter=50`, `seed=42`, `target=target_binned_return_20`. `fixed_params` = null (ktz priors in the search grid). `n_features`: 49 (mommask), vs 49 (E70 base) / 22 (E72 raw). n=520.

## What was found
No per-study written finding; the e79 memory states the general direction ("per-cell fill beats whole-row gating in 20/20 cells... mommask is the whole-row gate"). From the metrics, mommask is below BOTH the momswap-macro parent and raw v2.0 on every model. CORR (mommask / E70 momswap-macro / E72 raw): ridge 0.119710 / 0.138147 / 0.141550; lgb 0.134431 / 0.144594 / 0.147116; catboost 0.129504 / 0.146148 / 0.149235; rf 0.134037 / 0.152047 / 0.147053; mlp 0.109109 / 0.123118 / 0.128652. Payout (mommask / E70 parent): ridge 0.060760 / 0.070622; lgb 0.063803 / 0.067404; catboost 0.059993 / 0.069673; rf 0.065823 / 0.073145; mlp 0.059313 / 0.061629. So mommask − parent is roughly −0.01 to −0.02 CORR on all 5 models (largest for ridge/mlp), confirming masking information only removes signal. Paired Newey-West t-stats are not recorded in the artifacts.
