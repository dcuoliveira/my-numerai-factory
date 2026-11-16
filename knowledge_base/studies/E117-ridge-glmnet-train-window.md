# E117 — Training window of the best single ridge, 1 week to 6 months

- **State:** done (7/7 arms)
- **Runner:** experiments/e117_train_window_crypto.sh
- **Models / datasets:** ridge_glmnet on crypto/v2.0-tenc-onehot-rollsel-spread-interact (allfeat_pp) minus the tenc column (811 features); the payout-best single ridge (w6r6_lobo_tenc) with only the rolling training window varied
- **Artifacts on disk:** 33.1 MB across 7 forecast runs (plus 7 metrics + 7 hyperparams + 7 per_period) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E117 / E124 — six months of history fit on the 20-day label is the best ridge" (line 1438; dedicated section with per-window fit-20/fit-60 payout tables and fit60−fit20 Newey-West deltas) and index row (line 108; status "done", finding recorded); memory: none; metrics: results/metrics/crypto/ridge_glmnet/e117_w130.json, e117_w22.json (all 7 arms' CORR/MMC read)

## Main question
How much history should each refit use? Vary only the rolling training window (5 / 10 / 15 / 20 / 22 / 65 / 130 periods ≈ 1, 2, 3, 4 weeks, 1, 3, 6 months) on the payout-best single ridge, holding everything else. This is how-much-history, NOT freshness (E116's axis).

## What was tried
Seven arms, each the glmnet ridge on allfeat_pp (feature_exclude '^feature_sym_tenc$', 811 features) with only the rolling training window changed. Lambda is searched from scratch at every refit; the glmnet path is built from the training window. The purged inner CV cannot fit inside a window of <=22 periods, so those windows use --tune-window 130 (the search runs on the 130 periods ending where the window ends, with rolling folds trained on the window's own length, and the winner refitted on the window alone), so every arm selects λ on the same four validation slices. w130 (tune_window null) reproduces w6r6_lobo_tenc. Refit cadence stays 130. A pinned-lambda first version was superseded before reporting.

## Trained parameters
From results/metrics/crypto/ridge_glmnet/e117_w22.json config: model ridge_glmnet; data_version crypto/v2.0-tenc-onehot-rollsel-spread-interact; feature_exclude "^feature_sym_tenc$"; cv_scheme rolling; train_window 22; tune_window 130; refit_every 130; embargo 20; inner_cv purged; n_splits 5; val_frac 0.3; n_iter 50; search_type random; metric corr; target target_binned_return_20; task regression; n_features 811; seed 42; fixed_params null; search_jobs 8. The w130 arm (e117_w130.json) is identical with train_window 130 and tune_window null. train_window swept 5/10/15/20/22/65/130 across the 7 arms.

## What was found
FINDINGS-BACKED by EXPERIMENTS.md §E117/E124 (line 1438) and its index row (line 108). The 7 metrics JSONs give CORR.mean (MMC.mean), all n=510: w5 0.04506 (0.01020), w10 0.04256 (0.00965), w15 0.07828 (0.02446), w20 0.08199 (0.03019), w22 0.08834 (0.03320), w65 0.11785 (0.04535), w130 0.15944 (0.07185). The best arm is w130 (6 months, which reproduces w6r6_lobo_tenc): CORR 0.15944, MMC 0.07185, n=510 — roughly 3.5× the CORR of the 1–2-week windows; the shortest windows (5–10 periods) collapse to CORR ≈ 0.043–0.045.

FINDING (EXPERIMENTS.md §E117/E124): history dominates everything else. Payout rises monotonically with the training window on the 20-day label — w130 (6 months) pays 0.08955 (dd 1.73, calmar 0.052) — and every window under six months is significantly worse than six months (t −2.46 to −3.21 on E117's own 510 dates; −2.18 to −2.82 on the 470 dates shared with E124). On short windows the λ search lands on the top of the glmnet path on most refits, shrinking nearly everything away given little data. Meta-residualising the finished forecasts (added 2026-10-24, E107 transform, strength 0.5) lifts the six-month arm to payout 0.09725 (t +3.50), the best row in the joint table; on short windows it does almost nothing.
