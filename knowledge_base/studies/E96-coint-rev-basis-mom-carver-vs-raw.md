# E96 — coint+rev+basis+mom+carver stack vs raw: crypto/v2.0-coint-rev-basis-mom-carver-fillmask (82f) vs v2.0

- **State:** done (5/5 arms, + 5 scout screens)
- **Runner:** experiments/e96_coint_rev_basis_mom_carver_vs_raw_crypto.sh
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on crypto/v2.0-coint-rev-basis-mom-carver-fillmask (82f), paired vs raw v2.0 (22f, reused e72_*_base)
- **Artifacts on disk:** 31.1 MB across 10 forecast runs (5 full e96_* + 5 e96scout_*), plus 10 metrics + 10 hyperparams + 10 per_period (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/catboost_ktz/e96_catboost_ktz_crbmc.json (+ the other 4 arms and e72_*_base)

## Main question
The widest stat-arb + trend add-on: does E95's coint+rev+basis (50f) PLUS the momentum and Carver trend blocks (the "crbmc" base, 82f) beat raw? Prior: the individual blocks were each null-or-marginal, and a 60-column add-on is in the width where ridge gets diluted (cf. E82/E85) and E92/E95 found null-to-negative.

## What was tried
crypto/v2.0-coint-rev-basis-mom-carver-fillmask = raw 22 + 60 features in six families: coint (4), rev (14), basis (6), funding (4), mom (17, literature momentum incl. Tzotchev robust trend), tf (15, Carver/Clenow trend-following). fillmask == the loader's own 0.5 fill, so a clean raw+60 superset. 5 models from scratch at the shared protocol, each cell paired vs its raw twin e72_*_base (reused). Scout screen (e96scout_*) first. Only the model class varies.

## Trained parameters
From results/metrics/crypto/ridge/e96_ridge_crbmc.json config (representative): model ridge; data_version crypto/v2.0-coint-rev-basis-mom-carver-fillmask; target target_binned_return_20; cv_scheme rolling; train_window 130; refit_every 130; embargo 20; era_step 1; n_splits 5; val_frac 0.3; n_iter 50; search_type random; metric corr; n_features 82; seed 42. All five arms share this protocol at n_features 82; swept: model only.

## What was found
NUMBERS-ONLY. No written finding. From the metrics (mean CORR / MMC; level readings, block n=523, raw n=521): every arm sits at or below its raw twin — lgb 0.14528 / 0.05355 vs raw 0.14712 / 0.05925; catboost 0.14403 / 0.05307 vs raw 0.14924 / 0.05821; rf 0.14361 / 0.05263 vs raw 0.14705 / 0.05887; ridge 0.11944 / 0.04400 vs raw 0.14155 / 0.05801; mlp 0.11575 / 0.04021 vs raw 0.12865 / 0.05003. Best arm: lgb (CORR 0.14528, CORR dd 0.834). Direction, un-paired: null-to-negative — the 60-column add-on leaves the trees a touch below raw and pulls ridge down ~0.022 CORR; MMC is below raw in all five. No paired NW delta vs raw in the sources read. Matches the dilution prior for wide add-ons.
