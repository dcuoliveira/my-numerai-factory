# E95 — coint+rev+basis stack vs raw: crypto/v2.0-coint-rev-basis-fillmask (50f) vs v2.0

- **State:** done (5/5 arms, + 5 scout screens)
- **Runner:** experiments/e95_coint_rev_basis_vs_raw_crypto.sh
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on crypto/v2.0-coint-rev-basis-fillmask (50f), paired vs raw v2.0 (22f, reused e72_*_base)
- **Artifacts on disk:** 31.6 MB across 10 forecast runs (5 full e95_* + 5 e95scout_*), plus 10 metrics + 10 hyperparams + 10 per_period (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/catboost_ktz/e95_catboost_ktz_crb.json (+ the other 4 arms and e72_*_base)

## Main question
Does the combined stat-arb / microstructure stack beat raw where each block alone did not? crypto/v2.0-coint-rev-basis-fillmask is a strict superset of raw: the 22 plus 28 features in four families — coint (4: ADF-120d, half-life-120d, residual-z 60d/120d), rev (14: short-term reversal rsi/stoch/cci/amihud/ivol/max1/max5/skew/str_{1,7,14}d/dist_ma_{20,50}d/bbz_20d), basis (6: futures-spot) and funding (4: perp).

## What was tried
This is the combination of the individually-tested blocks E91 (coint), E94 (basis/funding) and E90 (reversal). 5 models tuned from scratch at the shared protocol so each cell pairs validly (NW lag 20) against its raw twin e72_*_base (reused). fillmask = missing cells pre-filled to 0.5 (the loader's own fill), so as loaded it is a clean raw+28 superset. A scout screen (e95scout_*) preceded the full runs. Only the model class varies.

## Trained parameters
From results/metrics/crypto/ridge/e95_ridge_crb.json config (representative): model ridge; data_version crypto/v2.0-coint-rev-basis-fillmask; target target_binned_return_20; cv_scheme rolling; train_window 130; refit_every 130; embargo 20; era_step 1; n_splits 5; val_frac 0.3; n_iter 50; search_type random; metric corr; n_features 50; seed 42. All five arms share this protocol at n_features 50; swept: model only.

## What was found
NUMBERS-ONLY. No written finding. From the metrics (mean CORR / MMC, level readings; block n=523, raw n=521): rf 0.14868 / 0.05527 vs raw 0.14705 / 0.05887 (CORR +0.002, MMC below); lgb 0.14798 / 0.05468 vs raw 0.14712 / 0.05925 (CORR +0.001, MMC below); catboost 0.14398 / 0.05216 below raw 0.14924; ridge 0.13018 / 0.04730 below raw 0.14155; mlp 0.11861 / 0.04079 below raw 0.12865. Best arm: rf (CORR 0.14868, CORR dd 0.550). Direction, un-paired: null — the stacked 28-feature add-on lifts CORR a hair on lgb/rf and sits below raw on ridge/catboost/mlp, and MMC is below raw in every arm. No paired NW delta vs raw was in the sources read. Consistent with the prior that each constituent block was individually null-or-marginal (E90/E91/E94); stacking them does not clear the bar on this level comparison.
