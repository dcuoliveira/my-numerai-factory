# E32 — catboost_ktz grid ENSEMBLE vs SEARCH per book dataset

> ⚠️ **PRODUCTION — DO NOT DELETE:** e32_sel_fenccausal / e32_sel_allfeat back IAUARETE_01 / IAUARETE_02 (in registry.ORDER). These forecasts are the backtest references `src.analysis.slots` resolves; retention protects them (K1). Deleting them breaks the registry self-check.

- **State:** done (18/18 arms)
- **Runner:** experiments/e32_catboost_ktz_ensemble_crypto.sh
- **Models / datasets:** `catboost_ktz_ens` (`CatBoostGridEnsemble`) vs `catboost_ktz` (search), two arms per book dataset over one 12-cell CatBoost KTZ grid
- **Artifacts on disk:** 79.3 MB across 18 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E32 — findings" (lines 700–732), §"E32 vs the deployed book" (756–767), §"Read against E31 and E30" (768–785); memory: ktz-priors-and-wrappers.md; metrics: results/metrics/crypto/catboost_ktz/e32_sel_raw.json, results/metrics/crypto/catboost_ktz_ens/e32_ens_allfeat.json

## Main question
The CatBoost member of the E30(lgb)/E31(mlp) family: ensemble the ktz grid instead of searching it. Does rank-averaging all 12 cells (`catboost_ktz_ens`) beat searching the SAME 12 and keeping the CV-best (`catboost_ktz`)? Prior from E10/E23/E25: averaging buys RISK (sharpe, drawdown) over LEVEL.

## What was tried
18 arms = 9 ENS + 9 SEL, one pair per book dataset. The grid: depth{2,4} × lr{.005,.01,.02} × rsm{.1,.2}, with iterations 2000, l2_leaf_reg 10, min_data_in_leaf 100 pinned (E27 booster priors on CatBoost's names, deliberately compact so the ensemble can fit ALL 12). ENS fits all 12 on each 130/130 block and rank-averages (`--n-iter 1 --n-splits 2`, selection removed; block-level rank validated ≈ per-period in `test_pipeline`); SEL searches the same 12 exhaustively (`--n-iter 12 --n-splits 5`) and keeps the CV-best. Both wrappers read the grid from `CatBoostGridEnsemble`, so it is average-12 vs pick-1-of-12. Headline is the paired ENS − SEL delta (NW lag 20); each ENS also ranked vs its deployed slot.

## Trained parameters
SEL example `e32_sel_raw.json`: model `catboost_ktz`, data `crypto/v2.0`, n_features 22, model_options {iterations 2000}, n_iter 12, n_splits 5, train_window 130, refit_every 130, embargo 20, seed 42, target `target_binned_return_20`; CORR 0.152371, pnl 0.037692, n_periods 516. ENS example `e32_ens_allfeat.json`: model `catboost_ktz_ens`, data `crypto/v2.0-tenc-onehot-spread-interact`, n_features 788, n_iter 1, n_splits 2; CORR 0.145451, MMC 0.073649, pnl 0.039553, n_periods 510. Arms sweep data_version across the 9 book feature sets.

## What was found
FINDINGS-BACKED. **Averaging the catboost_ktz grid LOSES to searching it on all 9 datasets — the opposite of E31's MLP result.** Payout is negative on every dataset, clearing |t| ≥ 2 on 4 (fenccausal −3.53, interact −2.99, tenc −2.93, cumlag3 −2.00); CORR negative on all 9, significant on 7. The loss is not redeemed by risk: the ensemble's payout Sharpe is LOWER than the pick's on all 9, so the "averaging buys risk over level" prior does not hold here.
- Mechanism (flip side of E31): rank-averaging beats CV selection only when (a) selection is NOISY and (b) members are DIVERSE. The 12 catboost_ktz cells fail both — near-identical (ENS↔SEL within-period rank corr 0.949–0.974, vs 0.03–0.75 for the book's cross-model blends) and CatBoost's symmetric oblivious trees make the CV pick reliable. So only the dilution cost remains.
- Vs the book (n=458): best CatBoost arm `e32_sel_fenccausal` 0.09328 lands ~9th of the 37-arm block, +0.0032 (t +0.51, not significant) over payout baseline `SACI_06`, well below the blends. ENS arms fall to the BOTTOM: e32_ens_raw −0.0148 (t −2.53), e32_ens_cumlag3 −0.0139 (t −3.00), e32_ens_interact −0.0109 (t −2.66). No E32 arm is a deploy candidate.
- Caveat for generalising: grids differ in size (MLP 120 vs CatBoost 12), so "MLP vs CatBoost" is confounded with "120 vs 12"; the clean claim is the within-study one — for this 12-cell grid, averaging is a strict loss to searching.
