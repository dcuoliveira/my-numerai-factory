# E97 — crbmc + CURATED interactions vs raw (and vs xfull E98): v2.0-crbmc-xcurated-fillmask (98f)

- **State:** done (5/5 arms)
- **Runner:** experiments/e97_crbmc_xcurated_crypto.sh (reuses mommask_core.sh, VARIANT=xcur)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on crypto/v2.0-crbmc-xcurated-fillmask (98f), paired vs raw v2.0 (22f, reused e72_*_base) and vs crbmc-xfull (E98)
- **Artifacts on disk:** 26.5 MB across 5 forecast runs, plus 5 metrics + 5 hyperparams + 5 per_period (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/catboost_ktz/e97_catboost_ktz_xcur.json (+ the other 4 arms and e72_*_base)

## Main question
Curated-vs-full interaction expansion on the crbmc combined-signal base: does a curated 16-interaction set hold up where the full expansion (E98, 2856f) drowns the linear/NN models, and does either beat raw?

## What was tried
crypto/v2.0-crbmc-xcurated-fillmask (98f) = crbmc base (raw 22 + coint 4 + rev 14 + basis 6 + funding 4 + mom 17 + carver/tf 15 = 82) + 16 curated pairwise interactions (feature_x2_). 5 models from scratch at the shared protocol, reported 3-way vs raw and vs crbmc-xfull (E98). Only the model class varies.

## Trained parameters
From results/metrics/crypto/ridge/e97_ridge_xcur.json config (representative): model ridge; data_version crypto/v2.0-crbmc-xcurated-fillmask; target target_binned_return_20; cv_scheme rolling; train_window 130; refit_every 130; embargo 20; era_step 1; n_splits 5; val_frac 0.3; n_iter 50; search_type random; metric corr; n_features 98; seed 42. All five arms share this protocol at n_features 98; swept: model only.

## What was found
NUMBERS-ONLY. No written finding. From the metrics (mean CORR / MMC; level readings, block n=523, raw n=521): rf 0.14941 / 0.05865 slightly above raw rf 0.14705 / 0.05887 (CORR +0.002, MMC ≈); catboost 0.14252 / 0.05212 below raw 0.14924; lgb 0.14282 / 0.05168 below raw 0.14712; ridge 0.11684 / 0.04263 below raw 0.14155; mlp collapses to 0.08341 / 0.02445 vs raw 0.12865 / 0.05003. Best arm: rf (CORR 0.14941, CORR dd 0.875). Direction, un-paired: the curated interaction set holds the trees near raw (rf a hair above) but hurts ridge and badly hurts the MLP; MMC is at or below raw everywhere. No paired NW delta vs raw, and the E97↔E98 contrast cell is not in the sources read. Consistent with the prior that curation drowns the linear/NN models less than the full expansion.
