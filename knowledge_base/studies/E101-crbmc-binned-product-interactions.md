# E101 — crbmc interactions from BINNED products (xbin) vs RAW products (E100 xraw) + raw

- **State:** done (5/5 arms)
- **Runner:** experiments/e101_crbmc_xbin_crypto.sh (reuses mommask_core.sh, VARIANT=xbin)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on crypto/v2.0-crbmc-xbin-fillmask (1469f), paired vs crbmc-xraw (E100) and raw v2.0 (22f, reused e72_*_base)
- **Artifacts on disk:** 26.3 MB across 5 forecast runs, plus 5 metrics + 5 hyperparams + 5 per_period (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none; memory: report-calmar-convention.md; metrics: results/metrics/crypto/catboost_ktz/e101_catboost_ktz_xbin.json, results/metrics/crypto/ridge/e101_ridge_xbin.json (+ the other 3 arms, E100 arms and e72_*_base)

## Main question
The A/B twin of E100: does forming an interaction as the product of the already-BINNED {0,.25,.5,.75,1} parent values (a product of cross-sectional ranks, magnitude discarded), THEN rank-binning, carry more or less usable signal than the raw-product construction (E100)? Same 1469 columns as xraw, different values (corr to xraw −0.16..−0.49).

## What was tried
crypto/v2.0-crbmc-xbin-fillmask (1469f), each interaction = product of binned parent values then rank-binned. 5 models from scratch at the shared protocol, reported 3-way vs crbmc-xraw (E100) and vs raw. Only the model class varies.

## Trained parameters
From results/metrics/crypto/ridge/e101_ridge_xbin.json config (representative): model ridge; data_version crypto/v2.0-crbmc-xbin-fillmask; target target_binned_return_20; cv_scheme rolling; train_window 130; refit_every 130; embargo 20; era_step 1; n_splits 5; val_frac 0.3; n_iter 50; search_type random; metric corr; n_features 1469; seed 42. All five arms share this protocol at n_features 1469; swept: model only.

## What was found
MEMORY-BACKED. No EXPERIMENTS.md section. The report-calmar-convention memory names this arm directly: the E100/E101 crbmc-interaction arms "sit below raw on payout but post HIGHER calmar (lgb crbmc-xbin calmar 0.077 vs raw-lgb 0.055)" — the datapoint that made prop calmar a required report column. The metrics agree on level (mean CORR; block n=523, raw n=521): lgb 0.14524 just below raw lgb 0.14712; rf 0.13763, catboost 0.13251 below raw; ridge 0.09276 and mlp 0.08629 collapse (vs raw 0.14155 / 0.12865). Best arm: lgb (CORR 0.14524, MMC 0.05216). Against the E100 raw-product twin (CORR per model): binned-product is a little kinder to the linear/NN models (ridge 0.09276 > 0.08277, mlp 0.08629 > 0.07675, catboost 0.13251 > 0.13072) and a little worse for lgb (0.14524 < 0.14720) and ~tied for rf. Direction: below raw on payout/level, calmar-positive per the memory; binned-vs-raw product is a wash, with binned slightly less damaging to ridge/MLP.
