# E99 — coint+rev+basis+macro stack vs raw: v2.0-coint-rev-basis-macro-fillmask (68f) vs v2.0

- **State:** done (5/5 arms)
- **Runner:** experiments/e99_coint_rev_basis_macro_vs_raw_crypto.sh (reuses mommask_core.sh, VARIANT=crbm)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on crypto/v2.0-coint-rev-basis-macro-fillmask (68f), paired vs raw v2.0 (22f, reused e72_*_base) and the vol block (e74_*_vol)
- **Artifacts on disk:** 26.5 MB across 5 forecast runs, plus 5 metrics + 5 hyperparams + 5 per_period (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/catboost_ktz/e99_catboost_ktz_crbm.json, results/metrics/crypto/ridge/e99_ridge_crbm.json (+ the other 3 arms and e72_*_base)

## Main question
Pools the per-symbol stat-arb/carry signals with the day-level macro regime block: does coint+rev+basis+funding plus 18 macro columns beat raw? Ridge is inert on the macro columns (constant within date) and uses only the 28 per-symbol features; trees/MLP can regime-gate on macro.

## What was tried
crypto/v2.0-coint-rev-basis-macro-fillmask (68f, clean superset of raw) = raw 22 + coint 4 + rev 14 + basis 6 + funding 4 + macro 18 (gold/silver/S&P/DXY/US-yield-curve, day-level). Sibling of E95 (coint+rev+basis) and E96 (+mom+carver), swapping in macro+funding. 5 models from scratch at the shared protocol, reported 3-way vs raw and the vol block. Only the model class varies.

## Trained parameters
From results/metrics/crypto/ridge/e99_ridge_crbm.json config (representative): model ridge; data_version crypto/v2.0-coint-rev-basis-macro-fillmask; target target_binned_return_20; cv_scheme rolling; train_window 130; refit_every 130; embargo 20; era_step 1; n_splits 5; val_frac 0.3; n_iter 50; search_type random; metric corr; n_features 68; seed 42. All five arms share this protocol at n_features 68; swept: model only.

## What was found
NUMBERS-ONLY. No written finding. From the metrics (mean CORR / MMC; level readings, block n=523, raw n=521): lgb 0.14941 / 0.05501 just above raw lgb 0.14712 / 0.05925 on CORR (MMC below); rf 0.14855 / 0.05564 ≈ raw rf 0.14705 / 0.05887; catboost 0.13270 / 0.04266 below raw 0.14924; ridge 0.13018 / 0.04730 below raw 0.14155; mlp 0.11233 / 0.03414 below raw 0.12865. Best arm: lgb (CORR 0.14941). A confirming internal check: the ridge arm's CORR/MMC/dd are byte-identical to E95's ridge (0.13018 / 0.04730 / dd 1.792), as expected since ridge ignores the 18 macro columns and sees the same 28 per-symbol features as E95's coint+rev+basis. Direction, un-paired: null — macro+funding lifts only lgb a hair and leaves everything else at or below raw, with MMC below raw throughout. No paired NW delta vs raw in the sources read.
