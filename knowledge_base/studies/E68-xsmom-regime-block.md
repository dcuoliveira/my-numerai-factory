# E68 — Cross-sectional momentum-regime block on v2.0-momswap

- **State:** done (10/10 arms)
- **Runner:** `experiments/e68_xsmom_momswap_crypto.sh`
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz × {base `crypto/v2.0-momswap` (31f), xsmom `crypto/v2.0-momswap-xsmom` (34f)}
- **Artifacts on disk:** 57.0 MB across 20 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none (planning-table row, status "registered"); memory: e68-xsmom-regime.md; metrics: results/metrics/crypto/mlp_ktz/e68_mlp_ktz_xsmom.json (+ all 10 full arms)

## Main question
Does a market-REGIME descriptor of momentum help on top of the momswap set (31f)? `build_xsmom` collapses the raw (un-binned) momentum cross-section into 3 day-level numbers per date — the cross-sectional mean, variance and skew of the momentum distribution across coins (per-signal moment → causal percentile → average → causal percentile), broadcast point-in-time (D−1) as `feature_xsmom_{mean,var,skew}`.

## What was tried
10 arms = 5 models × {`crypto/v2.0-momswap` (31f), `crypto/v2.0-momswap-xsmom` (34f = 31 + 3)}, tuned from scratch. The 3 features are IDENTICAL across the cross-section on a date, so they carry ~zero standalone within-era numerai_corr and are a designed negative control for ridge (a day-constant cannot move a within-date rank); only trees/MLP can use them via regime gating/interactions. Judge in the paired comparison, not on standalone IC. Decision criterion: xsmom beats base at paired |t|≥2 on a tree/MLP.

## Trained parameters
From the metrics JSONs: `cv_scheme=rolling`, `train_window=130`, `refit_every=130`, `n_splits=5`, `embargo=20`, `val_frac=0.3`, `metric=corr`, `search_type=random`, `n_iter=50`, `seed=42`, `target=target_binned_return_20`. `fixed_params` = null (ktz priors in the search grid). `n_features`: base 31, xsmom 34. n=520.

## What was found
No written finding (MEMORY index: "ridge INERT (negative control); launched 2026-09-24"). From the metrics (xsmom − base, mean per-period): ridge is EXACTLY inert as designed — `e68_ridge_xsmom` CORR 0.138147, payout 0.070622 vs `e68_ridge_base` CORR 0.138147, payout 0.070622 (identical to 6 dp), confirming the day-constant features cannot move a linear within-date rank. Trees are null-to-slightly-negative: ΔCORR lgb −0.00253, catboost −0.00074, rf −0.00100; Δpayout lgb −0.00286, catboost −0.00139, rf +0.00011 (small, mostly negative). The mlp arm shows a large apparent ΔCORR +0.05194, but this is an artifact of the base mlp arm collapsing (`e68_mlp_ktz_base` CORR 0.086724, c.sharpe 0.83, dd(corr) 3.55 — a degenerate run), not xsmom adding signal (the xsmom mlp arm CORR 0.138666 is in line with the other models' base). No model shows a genuine xsmom gain. Paired Newey-West t-stats are not recorded; the result is a null, matching the stated prior.
