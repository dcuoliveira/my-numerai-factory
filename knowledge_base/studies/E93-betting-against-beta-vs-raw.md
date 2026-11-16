# E93 — betting-against-beta (short-constraint) block vs raw: crypto/v2.0-bab (31f) vs v2.0

- **State:** done (5/5 arms)
- **Runner:** experiments/e93_bab_vs_raw_crypto.sh (reuses mommask_core.sh, VARIANT=bab)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on crypto/v2.0-bab (31f), paired vs raw v2.0 (22f, reused e72_*_base) and the vol block (e74_*_vol)
- **Artifacts on disk:** 24.2 MB across 5 forecast runs, plus 5 metrics + 5 hyperparams + 5 per_period (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none; memory: e90-e94-statarb-block-sweep.md; metrics: results/metrics/crypto/mlp_ktz/e93_mlp_ktz_bab.json (+ the other 4 arms and e72_*_base)

## Main question
Operationalizes the Miller / Hong-Sraer "speculative betas" anomaly for a short-constrained crypto market: do high-beta coins that are also hard to short (illiquid) and high-disagreement (high vol) earn lower forward returns, so a short-constraint-conditioned beta block beats raw? Distinct from E92 (beta-only, 25f) in carrying the interaction terms, the novel content.

## What was tried
crypto/v2.0-bab = raw 22 + 9 BAB features: 3 windows (30/60/90d) × {beta_btc, beta × Amihud-illiquidity, beta × realised-vol}. Products formed from per-date centered percentiles (robust to Amihud skew), then binned; per-symbol, point-in-time (lag 1), a clean superset of raw. 5 models from scratch at the shared protocol, reported 3-way vs raw v2.0 and vs the vol block. Only the model class varies across the 5 arms.

## Trained parameters
From results/metrics/crypto/catboost_ktz/e93_catboost_ktz_bab.json config (representative): model catboost_ktz; data_version crypto/v2.0-bab; target target_binned_return_20; cv_scheme rolling; train_window 130; refit_every 130; embargo 20; era_step 1; n_splits 5; val_frac 0.3; n_iter 50; search_type random; metric corr; n_features 31; seed 42; fixed_params null. The ridge arm is the same protocol at n_features 31. Swept: model only.

## What was found
MEMORY-BACKED. No EXPERIMENTS.md section. The e90-e94 sweep memory records E93 explicitly: "**WORST block tested — HURTS all 5 models at |t|≥2** (ridge -0.019 payout t-4.4; even lgb/mlp). Interaction cols are low-signal noise for a within-date rank; and the hardest-to-short no-spot coins (18.5%) are neutral-filled, so we're blind exactly where the effect should live." The metrics agree on direction: every BAB arm's mean CORR is below raw — ridge 0.12039 vs 0.14155, lgb 0.13097 vs 0.14712, catboost 0.13175 vs 0.14924, rf 0.13543 vs 0.14705, mlp 0.10234 vs 0.12865 (mlp the worst, CORR dd 5.15). MMC is lower across all five (mlp 0.03017 vs raw 0.05003). Level readings (block n=523 vs raw n=521), but the memory's payout delta (ridge -0.019, t-4.4) is the authoritative paired result. Direction: HURTS.
