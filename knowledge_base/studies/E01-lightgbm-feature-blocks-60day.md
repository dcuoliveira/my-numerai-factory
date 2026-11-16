# E01 — LightGBM feature blocks on the 60-day target

- **State:** done (23/23 arms)
- **Runner:** experiments/e01_blocks60_lgb_reg_crypto.sh
- **Models / datasets:** lgb; one data_version per feature block (crypto/v2.0 raw 22, v2.0-fenccausal 44, v2.0-fencglobal 44, v2.0-onehot, allfeat/tenc/spread/interact/rollsel variants)
- **Artifacts on disk:** 96.6 MB across 23 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E01 — findings"; memory: none; metrics: results/metrics/crypto/lgb/e01_fencglobal_m313.json

## Main question
Does any engineered feature block beat raw when the label is the **60-day** return
(`target_binned_return_60`) instead of the 20-day one? The same 23 arms, same pinned
booster and same cadence are re-run with only the label changed, to test whether the
20-day feature-block results transfer to the longer horizon.

## What was tried
23 arms = the feature blocks (raw, fenccausal, fencglobal, onehot, tenc, tenc_onehot,
tenc_onehot_rollsel, onehot_rollsel, spread, interact, onehot_interact, allfeat,
allfeatpp, allfeatppp) crossed with two LightGBM min_child_samples leaf sizes, m313
and m50. All on `target_binned_return_60`. Significance read Newey-West at **lag 60**
(a 60-day target overlaps ~98% between consecutive daily periods, so 470 periods are
~8 effectively-independent windows and nulls are expected).

## Trained parameters
From results/metrics/crypto/lgb/e01_fencglobal_m313.json: model lgb, task regression,
target target_binned_return_60, data_version crypto/v2.0-fencglobal, n_features 44,
cv_scheme rolling, n_splits 2, train_window 130, refit_every 130, embargo 60,
era_step 1, val_frac 0.3, seed 42, search_type random, metric corr, 4 refits.
LightGBM booster PINNED via `fixed_params` (every searched bound has size 1, so `n_iter=50` is nominal): colsample_bytree 0.070213, learning_rate 0.002572, max_depth 8, n_estimators 4327, num_leaves 255, reg_lambda 1.757511, subsample 0.816153, subsample_freq 1. The m50 arms set min_child_samples 50 instead of 313; raw arms use
crypto/v2.0 (22 features).

## What was found
FINDINGS-BACKED. Nulls dominate, as expected at lag 60. Payout vs raw (470 periods):
`fencglobal` m313 pays 0.162860, +0.078671 over raw (t +3.65) but this is a lookahead
artifact that strengthens with horizon (its whole-sample token profile partly encodes
whether a token survived; +2.83 on 20-day vs +3.65 here) — not deployable.
`interact` REVERSES sign between labels: +0.005710 (t +2.99) on 20-day, −0.006687
(t −3.56) here; combined with not reproducing at leaf 50, the 20-day result reads as
conditional on label and leaf size (SACI_01 is deployed on that 20-day result).
`fenccausal` is consistently harmful on both labels (−2.15, −2.24; here payout 0.064105
vs raw 0.084189). Every leaf-50 wide arm is positive (+0.83 to +1.10 t) but none
significant. The 60-day label is far riskier: max drawdowns 4.7–7.9 vs 0.2–1.2 for the
same datasets on the 20-day label.
