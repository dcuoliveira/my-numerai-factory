# E66 — Swap the base momentum proxies for the literature momentum block

- **State:** done (10/10 arms)
- **Runner:** `experiments/e66_momswap_litmom_crypto.sh`
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz × {base `crypto/v2.0` (raw 22f), swap `crypto/v2.0-litmom` with the 8 base momentum proxies excluded = 31f}
- **Artifacts on disk:** 54.8 MB across 20 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none (planning-table row only, status "registered"); memory: e66-momentum-proxy-swap.md, litmom-momentum-feature-pipeline.md; metrics: results/metrics/crypto/mlp_ktz/e66_mlp_ktz_swap.json (+ all 10 full arms)

## Main question
Not "does adding literature momentum help" (E64/E65 found that null-to-negative, because the new features are redundant with the base's own momentum proxies) but the SUBSTITUTION question: if you DROP the 8 base momentum proxies (`feature_{momentum,relative_strength_index,sharpe_ratio,bollinger}_{20,60}d`) and REPLACE them with the 17 literature momentum features, is the resulting 31-feature set better than the raw 22?

## What was tried
10 arms = 5 models × 2 feature sets. `base` = `crypto/v2.0` (raw 22). `swap` = `crypto/v2.0-litmom` run with `--feature-exclude '^feature_(momentum|relative_strength_index|sharpe_ratio|bollinger)_[0-9]+d$'` → 31 features (14 non-momentum base + 17 `feature_mom_*`). Both arms share rows/dates/seed; they differ in the momentum representation and in feature count (confound: the swap both removes native proxies and adds literature features, so it answers "is 31 better than raw 22", not "do the new ones add"). Decision criterion: swap beats base at paired Newey-West |t|≥2 (lag 20).

## Trained parameters
From the metrics JSONs (identical config across arms): `cv_scheme=rolling`, `train_window=130`, `refit_every=130`, `n_splits=5` (inner_cv purged), `embargo=20`, `val_frac=0.3`, `era_step=1`, `metric=corr`, `search_type=random`, `n_iter=50`, `seed=42`, `target=target_binned_return_20`, `search_jobs=8`. `fixed_params` = null for every arm (the `_ktz` sensitivity priors are compiled into each model's search grid, not stored as fixed params). `n_features`: base 22, swap 31. 4 refits per arm, n=520 scored periods.

## What was found
No written finding (MEMORY index records "DONE (10/10, 1h44m) — results not yet summarized"). From the metrics (per-model, swap − base, mean per-period): the swap does NOT beat the raw 22. ΔCORR: ridge −0.00479, lgb +0.00065, catboost −0.00072, rf −0.00178, mlp −0.00300 (4/5 negative). Δpayout: ridge −0.00218, lgb −0.00366, catboost −0.00053, rf −0.00569, mlp −0.00361 (5/5 negative). Best overall arm is the raw-22 `rf_ktz_base` (CORR 0.154306, MMC 0.064287, payout 0.079718, n=520). The paired Newey-West t-stats (the study's actual decision rule) are not recorded in the available artifacts; the mean deltas are null-to-slightly-negative, consistent with the stated prior (null-or-worse).
