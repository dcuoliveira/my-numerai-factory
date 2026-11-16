# E113 — row ablation of E109's feature_fm_ block on the fenccausal ridge

- **State:** done (6/6 arms)
- **Runner:** experiments/e113_fmrep_ablation_crypto.sh
- **Models / datasets:** ridge on crypto/v2.0-fenccausal-fmrep, 44 base features always kept, one feature_fm_ sub-block added per arm via --feature-exclude; compared vs e109_ridge_fenc (no block) and e109_ridge_fenc_fm (whole block), both reused
- **Artifacts on disk:** 33.0 MB across 6 forecast runs (plus 6 metrics + 6 hyperparams + 6 per_period) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E113 — E109's damage to the ridge comes from the time-series block; TabPFN's own predictions are neutral"; memory: e109-foundation-repr-block.md; metrics: results/metrics/crypto/ridge/e113_ridge_ts.json

## Main question
Which part of E109's 41-column foundation-model block makes the fenccausal ridge worse (full block: payout −0.0170, t −2.35)?

## What was tried
Ridge on crypto/v2.0-fenccausal-fmrep, the 44 base features always kept, ONE sub-block added per arm via --feature-exclude '^feature_fm_(?!<sub-block>)': row 1 TabPFN point predictions (3 cols), rows 1–4 point + distribution + bin probabilities + disagreement (12), rows 2–4 alone (9, the note's proposed first study), row 5 TabPFN embedding PCs (8), rows 6–7 per-block heads + context variants (7), rows 8–9 TimesFM/Chronos-2 on spot (14). 6 arms at the shared protocol, each paired (Newey-West lag 20, 510 periods) vs e109_ridge_fenc (no block, payout 0.08622) and vs e109_ridge_fenc_fm (whole block, 0.06923).

## Trained parameters
From results/metrics/crypto/ridge/e113_ridge_ts.json config (the TimesFM/Chronos-2 rows-8–9 arm): model ridge; data_version crypto/v2.0-fenccausal-fmrep; feature_exclude "^feature_fm_(?!(tsfm|chr)_)"; cv_scheme rolling; train_window 130; refit_every 130; embargo 20; inner_cv purged; n_splits 5; val_frac 0.3; n_iter 50; search_type random; metric corr; target target_binned_return_20; n_features 58 (44 base + 14 TS cols); seed 42; fixed_params null. CORR.mean 0.13892, MMC.mean 0.06188, n=510. The other 5 arms differ only in the feature_exclude regex selecting their sub-block.

## What was found
FINDINGS-BACKED. The time-series block is the main cause. Per sub-block, Δpayout vs no-block (t): TabPFN point preds (3) +0.0011 (+0.37); rows 1–4 (12) +0.0015 (+0.31); rows 2–4 (9) +0.0004 (+0.10); embedding PCs (8) −0.0045 (−0.69); heads+contexts (7) −0.0054 (−1.46); TimesFM/Chronos-2 (14) −0.0105 (−2.63); whole block (41) −0.0170 (−2.35). Rows 8–9 (TimesFM/Chronos-2 on each coin's own price path) are the only sub-block that hurts payout at |t|>=2 on its own, costing MMC −0.0092 (t −2.55) and CORR −0.0128 (t −3.23). Embeddings and heads cost CORR at |t|≈2 but payout only at |t|<1.5. Harms roughly add up (single-block deltas sum to −0.019 vs −0.017 for the whole block). TabPFN's own outputs (rows 1–4) are payout-neutral (+0.0004 to +0.0015, all |t|<0.4) and trade a little CORR (−0.004..−0.008, |t|<1.9) for MMC and depth: on equal n, calmar improves 0.062→0.072 (rows 1–4) and 0.079 (rows 2–4), burn 60→56 for rows 2–4. Verdict: drop rows 5–9 (they only cost); rows 2–4 are the one candidate left — a risk play (calmar +0.017 at flat payout), worth a calmar-paired check, and a separate study since none is live-wired.
