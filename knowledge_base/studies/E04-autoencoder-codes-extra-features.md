# E04 — Autoencoder codes as extra features, on LightGBM

- **State:** done (3/3 arms)
- **Runner:** experiments/e04_autoencoder_lgb_reg_crypto.sh
- **Models / datasets:** lgb; crypto/v2.0-aetuned (22 raw + 8 autoencoder codes = 30), a PCA counterpart, and raw crypto/v2.0 (22)
- **Artifacts on disk:** 15.0 MB across 3 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/lgb/e04_ae.json
- Note: `payout` = 0.1*CORR_mean + MMC_mean from stored summaries.

## Main question
Do 8 bottleneck-autoencoder codes add anything to the raw 22 features? One model, one
thing varying — the extra columns. A PCA arm serves as a linear-compression control.

## What was tried
3 arms: `e04_ae` (raw 22 + 8 AE codes, n_features 30), `e04_pca` (raw + PCA codes,
n_features 30), `e04_raw` (22). Same pinned booster and cadence as E01/E02. 510 periods,
20-day label.

## Trained parameters
From results/metrics/crypto/lgb/e04_ae.json: model lgb, regression,
target_binned_return_20, data_version crypto/v2.0-aetuned, n_features 30, cv_scheme
rolling, n_splits 2, train_window 130, refit_every 130, embargo 20, seed 42,
n_iter 50 (nominal). LightGBM booster PINNED via `fixed_params` (every searched bound has size 1, so `n_iter=50` is nominal): colsample_bytree 0.070213, learning_rate 0.002572, max_depth 8, n_estimators 4327, num_leaves 255, reg_lambda 1.757511, subsample 0.816153, subsample_freq 1.

## What was found
No written finding; from the metrics (510 periods): the codes do not help. raw pays
0.083272 at the highest CORR (0.149754). `e04_pca` 0.084174 (CORR 0.148699) is a hair
above raw on payout but below on CORR; `e04_ae` is below both on every metric
(payout 0.081731, CORR 0.146588, MMC 0.067073). On these numbers neither autoencoder
nor PCA codes beat raw by a meaningful margin; no paired t-stats in available artifacts.
