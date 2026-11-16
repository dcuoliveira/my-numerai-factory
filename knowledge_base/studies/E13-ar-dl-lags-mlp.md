# E13 — Do autoregressive and distributed lags help the MLP?

- **State:** done (4/4 arms)
- **Runner:** experiments/e13_adl_mlp_crypto.sh
- **Models / datasets:** one PINNED sklearn MLP on crypto/v2.0 (raw), v2.0-adl and the AR / DL splits
- **Artifacts on disk:** 119.6 MB across 4 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/mlp/e13_adl.json
- Note: `payout` = 0.1*CORR_mean + MMC_mean from stored summaries.

## Main question
Four arms on ONE pinned MLP (the config e07_mlpreg_raw chose) differing only in columns:
contemporaneous only (raw), + lags of the TARGET (AR), + lags of every FEATURE (DL), +
both (ADL). AR and DL are split so a move in ADL says which half caused it. A target lag
under 20 periods would be lookahead (the label is a 20-day forward return), so build_adl
refuses one. DL was already tried on lgb and did not help (E05/E06); the AR half is new.

## What was tried
4 arms: `e13_raw` (22), `e13_ar` (raw + target AR lags), `e13_dl` (raw + feature lags),
`e13_adl` (both, n_features 91). 514 periods, 20-day label. Unlike E07-E12 these do NOT
search — n_iter 1 with the MLP pinned to e07's winning config so only the columns vary.

## Trained parameters
From results/metrics/crypto/mlp/e13_adl.json: model mlp, regression,
target_binned_return_20, data_version crypto/v2.0-adl, n_features 91, cv_scheme rolling,
n_splits 5, train_window 130, refit_every 130, embargo 20, seed 42, n_iter 1,
fixed_params mlp__alpha 8.227241, mlp__batch_size 2048, mlp__hidden_layer_sizes [64,32],
mlp__learning_rate_init 0.008886238 (the pinned e07_mlpreg recipe). raw/ar/dl share the
same pinned MLP on their respective column sets.

## What was found
No written finding; from the metrics (514 periods) neither lag block helps, and the full
ADL stack is the worst. By CORR, raw is best (0.153804) > ar (0.148688) > dl (0.149419)
> adl (0.139919). By payout the order is nearly flat and mixed: dl 0.082855 edges raw
0.082438 (dl has lower CORR but higher MMC 0.067913 vs 0.067057), ar 0.080974, adl 0.074879.
On these numbers adding target-AR or feature-DL lags does not lift the MLP (ADL clearly
hurts); the new AR half is no better than the DL half already known to fail on lgb.
No paired t-stats in available artifacts.
