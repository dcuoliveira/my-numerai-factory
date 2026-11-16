# E30 — lgb_ktz grid ENSEMBLE per book dataset (fit all, rank-average)

- **State:** done (9/9 arms)
- **Runner:** experiments/e30_ktz_gridensemble_lgb_reg_crypto.sh
- **Models / datasets:** `lgb` grid-ensemble (`GridEnsembleRegressor`), one rank-average per book dataset (raw, tenc, interact, fenccausal, cumlag3, tenc_onehot, onehot_rollsel, allfeat, allfeatpp)
- **Artifacts on disk:** 50.2 MB across 9 forecast runs (the 9 deployable `e30_<ds>_ens`; the 495 raw cells are intermediates — their forecasts reclaimed as each ensemble is built, cell metrics reclaimed after the report, per_period kept for the oracle) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E30 — findings" (lines 543–588) and §"Read against E31 and E30" (768–785); memory: e61-sur-ridge.md, e28-e29-prefix-collision.md; metrics: results/metrics/crypto/lgb/e30_raw_ens.json

## Main question
For each of the 9 book datasets, fit EVERY cell of the `lgb_ktz` good-corner grid (54 cells = num_leaves{2,4,8} × lr{.001,.003,.01} × colsample{.05,.10,.20} × n_est{2k,4k}, with L = lr·n_est ≤ 40) as pinned walk-forwards and rank-average their OOS predictions — instead of letting the search SELECT one. The lgb member of the E30(lgb)/E31(mlp)/E32(catboost) grid-ensemble family, paired against the E29 search-and-pick per dataset.

## What was tried
9 deployable ensemble arms. Each `e30_<ds>_ens` is a per-period rank average of all 54 pinned cells on that dataset. Paired vs the E29 pick per dataset, Newey-West lag 20, on the block's common window. Also compared against the oracle best single cell (needs test scores, not achievable live). The grid is the E27 booster KTZ prior on LightGBM's names.

## Trained parameters
From `e30_raw_ens.json`: combination "per-period rank average" over 54 members named `e30_raw_nl{2,4,8}_lr{001,003,01}_cs{05,10,20}_ne{2k,4k}` — i.e. num_leaves × learning_rate × colsample × n_estimators grid as above; n_periods 516; CORR 0.148121, MMC 0.061389, pnl 0.036749 (raw arm). Each member is a pinned lgb walk-forward (train/refit 130, embargo 20, seed 42, target `target_binned_return_20`); the per-dataset arm sweeps the data_version.

## What was found
FINDINGS-BACKED. **Bagging the whole good-corner lgb_ktz grid is statistically indistinguishable from E29's search-and-pick — no dataset clears |t| ≥ 2 in either direction.** Unlike E31 (MLP), where ensembling recovered a real gap, the LightGBM arms are already so close in the good corner that searching (E29), pinning and bagging all land in the same place.
- Per-dataset ENS − E29 payout deltas: raw −0.00227 (t −0.69), tenc −0.00029 (−0.12), interact −0.00288 (−1.02), fenccausal +0.00270 (+0.75), cumlag3 −0.00459 (−1.29), tenc_onehot +0.00282 (t +1.52, the largest), onehot_rollsel +0.00392 (+1.19), allfeat +0.00309 (+0.88), allfeatpp +0.00165 (+0.52). ENS edges the pick on the 5 richer sets and loses on the 4 leaner, all inside noise.
- The oracle stays out of reach: ENS is below the best single cell on all 9 (short by 0.0015–0.0070); averaging regresses toward the middle of the grid.
- Standout is `fenccausal`: best c.sharpe (1.330) and by far the lowest drawdown (0.850 vs 1.1–1.9 elsewhere) — a risk-not-level profile, the same causal per-symbol encoding the deployed `IA_01`/`CU_SA_IA_BO_01` lean on. Not deployable as-is: no arm beat its E29 baseline at |t| ≥ 2. (The FARM-over-full-grid study that once held id E30 was renumbered E36.)
