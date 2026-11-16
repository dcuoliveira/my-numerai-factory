# E100 — crbmc interactions from RAW products (xraw) vs BINNED products (E101 xbin) + raw

- **State:** done (5/5 arms)
- **Runner:** experiments/e100_crbmc_xraw_crypto.sh (reuses mommask_core.sh, VARIANT=xraw)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on crypto/v2.0-crbmc-xraw-fillmask (1469f), paired vs crbmc-xbin (E101) and raw v2.0 (22f, reused e72_*_base)
- **Artifacts on disk:** 26.7 MB across 5 forecast runs, plus 5 metrics + 5 hyperparams + 5 per_period (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none; memory: report-calmar-convention.md; metrics: results/metrics/crypto/mlp_ktz/e100_mlp_ktz_xraw.json, results/metrics/crypto/ridge/e100_ridge_xraw.json (+ the other 3 arms, E101 arms and e72_*_base)

## Main question
An A/B on interaction CONSTRUCTION: does forming an interaction as the product of the RAW (continuous, pre-bin) parent values, THEN rank-binning, carry more usable signal than forming it from the already-binned parents (E101)? The two share the same 1469 columns but different values (corr to xbin −0.16..−0.49), since binning parents first discards the magnitude the product needs.

## What was tried
crypto/v2.0-crbmc-xraw-fillmask (1469f = crbmc base 82 + 1387 feature_x2_), each interaction = product of raw parent values then rank-binned. 5 models from scratch at the shared protocol, reported 3-way vs crbmc-xbin (E101, the headline A/B) and vs raw. Only the model class varies.

## Trained parameters
From results/metrics/crypto/ridge/e100_ridge_xraw.json config (representative): model ridge; data_version crypto/v2.0-crbmc-xraw-fillmask; target target_binned_return_20; cv_scheme rolling; train_window 130; refit_every 130; embargo 20; era_step 1; n_splits 5; val_frac 0.3; n_iter 50; search_type random; metric corr; n_features 1469; seed 42. All five arms share this protocol at n_features 1469; swept: model only.

## What was found
MEMORY-BACKED. No EXPERIMENTS.md section. The report-calmar-convention memory records the E100/E101 crbmc-interaction arms as the motivating example for always printing calmar: "the crbmc interaction arms (E100/E101) sit below raw on payout but post HIGHER calmar (lgb crbmc-xbin calmar 0.077 vs raw-lgb 0.055)" — i.e. they trail raw on level but earn it more smoothly. The metrics agree on level (mean CORR; block n=523, raw n=521): lgb 0.14720 ≈ raw lgb 0.14712; the rest below raw — rf 0.13796, catboost 0.13072, ridge collapses to 0.08277, mlp collapses to 0.07675 (vs raw mlp 0.12865). Best arm: lgb (CORR 0.14720, MMC 0.05485). Against the E101 binned-product twin (CORR per model): ridge xraw 0.08277 < xbin 0.09276, mlp xraw 0.07675 < xbin 0.08629, catboost xraw 0.13072 < xbin 0.13251, but lgb xraw 0.14720 > xbin 0.14524 and rf xraw 0.13796 ≈ xbin 0.13763. So raw-product interactions are slightly better for the boosters and worse for ridge/MLP; the difference is small. Direction: below raw on level, calmar-positive per the memory; raw-vs-binned product is a wash with a faint lgb edge for raw-product.
