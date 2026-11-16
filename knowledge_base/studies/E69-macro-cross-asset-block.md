# E69 — Macro / cross-asset information set on v2.0-momswap

- **State:** done (19/10 arms — 10 model arms + 9 ensemble/blend arms)
- **Runner:** `experiments/e69_macro_momswap_crypto.sh`
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz × {base `crypto/v2.0-momswap` (31f), macro `crypto/v2.0-momswap-macro` (49f)}
- **Artifacts on disk:** 80.2 MB across 29 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none (planning-table row, status "registered"); memory: e69-macro-momswap.md, inflation-breakeven-block.md, e80-infl-breakeven-macro.md; MEMORY.md index (E69 blend finding); metrics: results/metrics/crypto/catboost_ktz/e69_catboost_ktz_macro.json (+ all full arms and blends)

## Main question
The E68 sibling with a different information set: instead of a cross-sectional momentum regime, add EXTERNAL MACRO state. On top of `crypto/v2.0-momswap` (31f), add 18 `feature_macro_*` (gold/silver returns 20d/60d, S&P returns 5d/20d/60d, DXY dollar-index momentum 5d/20d/60d, US Treasury yields y1/y2/y5/y10/y30, curve slopes 1s10s/2s10s/5s30s), causal-percentile normalised, broadcast point-in-time (D−1). Does macro state beat the base at paired |t|≥2 on a tree/MLP?

## What was tried
10 model arms = 5 models × {base 31f, macro 49f}, tuned from scratch. (Note from memory: the study was restarted after the user added DXY momentum, taking macro from 46f→49f; the on-disk arms are the 49f rebuild.) Like xsmom, macro features are day-level / within-date constant → ridge is a designed negative control. Plus 9 ensemble/blend arms (`e69_macro_ens`, `e69_bl_cusa_*` book-blends at various weights).

## Trained parameters
From the metrics JSONs: `cv_scheme=rolling`, `train_window=130`, `refit_every=130`, `n_splits=5`, `embargo=20`, `val_frac=0.3`, `metric=corr`, `search_type=random`, `n_iter=50`, `seed=42`, `target=target_binned_return_20`. `fixed_params` = null (ktz priors in the search grid). `n_features`: base 31, macro 49; model arms n=520; blend arms n≈448 and `data_version=null`.

## What was found
MEMORY-BACKED (MEMORY.md index: "ridge inert; DONE; b5 blend (0.8 cusa + 0.2 rf_macro) lifts cu_sa_ia_bo_02 CORR +0.0043 t+1.80 at flat payout (best blend across E67/E69)"). The macro feature block on its own does NOT beat the base: ridge is exactly inert (`e69_ridge_macro` CORR 0.138147 = `e69_ridge_base` 0.138147); trees are slightly negative — ΔCORR (macro − base) lgb −0.00490, catboost −0.00303, rf −0.00136, Δpayout all mildly negative; the mlp "gain" is again the base mlp arm collapsing (CORR 0.086724, dd 3.55). The deployable result is at the BLEND level: the book slot `cu_sa_ia_bo_02` rank-averaged 0.8/0.2 with the macro RF arm lifts CORR by +0.0043 (t +1.80) at flat payout — the best blend found across E67/E69, but t +1.80 is below the house |t|≥2 bar. The matching on-disk arm is `e69_bl_cusa_rf_82` (CORR 0.164956, MMC 0.085729, payout 0.102224, n=448). Takeaway: macro as a feature block is inert/null for the models directly; its only positive signal is a small, below-threshold MMC/CORR diversification when blended with the book.
