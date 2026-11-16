# E72 — Carver trend-following stack vs the raw 22

- **State:** done (11/10 arms — 10 model arms + `e72_raw5`)
- **Runner:** `experiments/e72_carver_vs_raw_crypto.sh`
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz × {base `crypto/v2.0` (raw 22f), carver `crypto/v2.0-momswap-macro-carver` (64f)}
- **Artifacts on disk:** 59.7 MB across 21 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none (planning-table row, status "registered"); memory: e72-carver-vs-raw.md; metrics: results/metrics/crypto/catboost_ktz/e72_catboost_ktz_carver.json (+ all full arms)

## Main question
Bottom-line test of the whole engineering stack against the original raw 22 — the deployment question "is the engineered 64 better than the raw 22", not a single-block ablation. `crypto/v2.0-momswap-macro-carver` (64f) = momentum swap (14 base + 17 literature momentum) + 18 macro/cross-asset (incl. DXY momentum) + 15 Robert Carver-style trend-following `feature_tf_*` (EWMAC crossovers 2_8..32_128, breakouts 20/40/80/160, Bollinger-z 20/40/80, Clenow 60/90, new-high-20-in-5).

## What was tried
10 arms = 5 models × {raw `crypto/v2.0` (22f), carver (64f)}, from scratch. A whole-stack comparison: many blocks differ at once (the raw momentum proxies are replaced by literature momentum, plus macro and trend-following added). Plus `e72_raw5` (a raw ensemble reference arm). Decision criterion: carver beats raw at paired |t|≥2.

## Trained parameters
From the metrics JSONs: `cv_scheme=rolling`, `train_window=130`, `refit_every=130`, `n_splits=5`, `embargo=20`, `val_frac=0.3`, `metric=corr`, `search_type=random`, `n_iter=50`, `seed=42`, `target=target_binned_return_20`. `fixed_params` = null (ktz priors in the search grid). `n_features`: base 22, carver 64. n=520 (carver arms) / 521 (raw base arms).

## What was found
No written finding in the artifacts. From the metrics (carver − base, mean per-period): a mixed null — no decisive win for the 64f stack. ΔCORR: lgb +0.00409, rf +0.00541, mlp +0.00241, catboost −0.00012, ridge −0.00814. Δpayout: lgb −0.00189, rf −0.00036, mlp +0.00657, catboost −0.00175, ridge −0.00403. Best carver arm `rf_ktz_carver` CORR 0.152466 / payout 0.073215 (vs raw rf 0.147053 / 0.073579) — and no carver arm beats the best raw-22 single arm on payout (`lgb_ktz_base` 0.0740). On CORR the tree/MLP arms are marginally positive and ridge negative; on payout four of five models are slightly negative (−0.0004 to −0.0040) and only mlp gains (+0.0066). This is consistent with the stated prior that no engineered block has beaten the raw 22 on this panel at |t|≥2; the paired Newey-West t-stats that would settle it are not recorded in the available artifacts. (NOTE: this study's `e72_*_base` raw-22 arms are the reused baseline for the E75/E77/E78 mommask studies and others.)
