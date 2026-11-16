# E74 — Realised-volatility estimator block vs the raw 22

- **State:** done (10/10 arms)
- **Runner:** `experiments/e74_vol_vs_raw_crypto.sh`
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz × {base `crypto/v2.0` (raw 22f), vol `crypto/v2.0-vol` (44f)}
- **Artifacts on disk:** 46.5 MB across 10 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none (planning-table row, status "registered"); memory: e74-vol-estimators.md, e90-e94-statarb-block-sweep.md (characterises E74 as "weakly +, ns"); metrics: results/metrics/crypto/mlp_ktz/e74_mlp_ktz_vol.json (+ all 10 full arms)

## Main question
Does a proper realised-volatility block help on top of the base's two crude volatility features? `crypto/v2.0-vol` (44f) = raw 22 + 22 `feature_vol_*` estimators computed from spot OHLCV over a hyperparameter grid: squared-returns RV (`sqret_{5,10,20,40,60,120}d`), EWMA/RiskMetrics (`ewma_lam{90,94,97,99}`), Parkinson (`park_{5,10,20,40,60,120}d`), Yang–Zhang (`yz_{5,10,20,40,60,120}d`). Coverage ~82–89% of panel rows. The block must add over the base `feature_volatility_{20,60}d`.

## What was tried
10 arms = 5 models × {raw `crypto/v2.0` (22f), `crypto/v2.0-vol` (44f)}, from scratch. Built by the first OHLC-reading feature module (`volatility_features.py`/`build_vol.py`). Redundancy analysis (from memory): mean off-diag pooled |corr| 0.69; range estimators nearly collapse in crypto (park_20 vs yz_20 = 0.94 — 24/7 trading, no overnight gap; sqret_20 vs park_20 = 0.83); the WINDOW axis is the real diversity (sqret 5d vs 120d = 0.44); block vs base `feature_volatility_20d` ≈ 0.65–0.76. Decision criterion: vol beats raw at paired |t|≥2.

## Trained parameters
From the metrics JSONs: `cv_scheme=rolling`, `train_window=130`, `refit_every=130`, `n_splits=5`, `embargo=20`, `val_frac=0.3`, `metric=corr`, `search_type=random`, `n_iter=50`, `seed=42`, `target=target_binned_return_20`. `fixed_params` = null (ktz priors in the search grid). `n_features`: base 22, vol 44. n=521.

## What was found
MEMORY-BACKED (e90-e94 memory: "Secondary 'vol' = the leading prior add-on (E74, weakly +, ns)" — i.e. weakly positive, not significant). The metrics confirm the "weakly +" direction: this is the one block in this batch with a consistent (if small) positive delta. ΔCORR (vol − base): catboost +0.00529, rf +0.00396, ridge +0.00388, lgb +0.00245, mlp −0.00111. Δpayout: lgb −0.00251, catboost +0.00272, rf −0.00333, ridge +0.00303, mlp +0.00520. Best arm `catboost_ktz_vol` CORR 0.154524 / MMC 0.060408 / payout 0.075861 (vs catboost base 0.149235 / 0.058215 / 0.073138). So the vol block gives a small +CORR/+payout on 4/5 models (mlp ~flat), but the memory records it as not significant — it did not clear the |t|≥2 bar, and later studies (E90–E96) treat it only as the leading-but-ns prior add-on. The paired Newey-West t-stats are not in the artifacts.
