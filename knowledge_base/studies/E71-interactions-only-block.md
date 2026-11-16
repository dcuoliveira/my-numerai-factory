# E71 — Interactions-only block on v2.0-momswap-macro (the E70 follow-up)

- **State:** done (10/10 arms; scout 1)
- **Runner:** `experiments/e71_xinteractions_momswap_macro_crypto.sh`
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz × {base `crypto/v2.0-momswap-macro` (49f), xint `crypto/v2.0-momswap-macro-xinteractions`}
- **Artifacts on disk:** 51.3 MB across 11 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none (planning-table row, status "registered"); memory: e71-xinteractions-momswap-macro.md, e79-fillmask-equals-loader.md; metrics: results/metrics/crypto/catboost_ktz/e71_catboost_ktz_base.json (+ all full arms)

## Main question
E70 dumped the full 925f allfeat block and it was null-to-negative (the 752 onehot dummies crushed ridge/MLP). E71 strips it to JUST the 24-column interact block: 6 accel + 6 adapted products (`feature_mom_ret_28d` stands in for the removed `feature_momentum`) + 12 momentum×macro-momentum products (crypto momentum gated by the macro regime — a per-symbol cross term that CAN move a within-date rank, so ridge is no longer a pure negative control). Does the interactions-only block firm up the boosters' E70 hint once the dummy noise is removed?

## What was tried
10 arms = 5 models × {base 49f, xint}, from scratch. Dataset built via `build_momswap_eng.py --blocks interact`. The first launch's scout was stopped when the session ended (only 1 arm ran); relaunched full. Decision criterion: xint beats base at paired |t|≥2.

## Trained parameters
From the metrics JSONs: `cv_scheme=rolling`, `train_window=130`, `refit_every=130`, `n_splits=5`, `embargo=20`, `val_frac=0.3`, `metric=corr`, `search_type=random`, `n_iter=50`, `seed=42`, `target=target_binned_return_20`. `fixed_params` = null (ktz priors in the search grid). `n_features`: base 49. **Discrepancy to note:** every on-disk `*_xint` metrics config records `data_version=crypto/v2.0-momswap-macro-xinteractions` with `n_features=219`, whereas the e71 memory and EXPERIMENTS.md describe the study as run at 73f (49 base + 24 interact). The e79 memory explains the dataset was rebuilt 73f→219f AFTER E71 ran; the on-disk metrics therefore record the 219f count. Treat the xint feature count as unresolved between the artifacts (73f in prose, 219f in the metrics config). n=520.

## What was found
No written finding of E71's own base-vs-xint result in the artifacts. From the metrics (xint − base, mean per-period): a tighter null than E70, mostly negative. ΔCORR: ridge −0.02594, mlp −0.01318, catboost −0.00720, rf −0.01887, lgb +0.00615 (only lgb positive). Δpayout: ridge −0.01761, mlp −0.00311, catboost −0.00496, rf −0.01148, lgb +0.00532. Best xint arm `lgb_ktz_xint` CORR 0.150747 / payout 0.072720 (just above its base 0.144594 / 0.067404). The interactions-only block does not establish a gain: only lgb is positive and the linear/MLP/rf arms are clearly hurt. Paired Newey-West t-stats are not recorded in the artifacts.
