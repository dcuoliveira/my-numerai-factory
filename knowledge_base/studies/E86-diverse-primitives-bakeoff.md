# E86 — diverse-primitives bake-off: 7 raw groups x 5 models from scratch, rank + ensembles

- **State:** done (35/35 arms)
- **Runner:** experiments/e86_group_bakeoff_crypto.sh
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on 7 primitive feature groups of `crypto/v2.0` (feature-include regexes)
- **Artifacts on disk:** 111.3 MB across 35 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §E86 row (status "registered", no finding cell); memory: none; metrics: results/metrics/crypto/ridge/e86_ridge_volatility.json, results/metrics/crypto/rf_ktz/e86_rf_ktz_momentum.json

## Main question
Which model class best exploits which primitive feature group of the raw 22, and does a cross-group/model blend beat the single best cell? (No single group is expected to match full raw.)

## What was tried
The raw-22 primitive families that partition the 22 exactly: momentum (2), volatility (4 = volatility+bollinger), rsi (2), sharpe (2), volume (4), size (4 = market_cap), price_ma (4 = close_). Each is a `--feature-include` regex on `crypto/v2.0` (one dataset → one clean common window). 7 groups × 5 models = 35 from-scratch runs at the shared protocol, each intended to be paired (NW lag 20) vs the full raw 22 (`e72_*_base`); ensemble screen over the 35 arms + book via report_html afterwards. The referenced `g1_diverse_primitives.json` never reached disk, so the groups are those primitive families.

## Trained parameters
From results/metrics/crypto/rf_ktz/e86_rf_ktz_momentum.json: model rf_ktz; data_version `crypto/v2.0`; feature_include `momentum` (n_features 2); cv_scheme rolling, inner_cv purged; train_window 130, refit_every 130, embargo 20; n_splits 5, val_frac 0.3; seed 42; n_iter 50; search_type random; metric corr; target target_binned_return_20; rf refit-0 best_params: bootstrap True, max_depth 6, max_features 0.15, min_samples_leaf 300, n_estimators 100. n_features varies per group (2–4); same protocol across all 35 cells, model class and feature-group swept.

## What was found
NUMBERS-ONLY — no written finding or paired-delta artifact on disk. From the metrics (own ~522-period windows, standalone), the **volatility group carries the most standalone signal**: it takes the top three cells — ridge_volatility payout 0.06660 / CORR 0.13681 / MMC 0.05292, catboost_volatility 0.06043 / CORR 0.14043, rf_volatility 0.05787 — but size is close behind (rf_size 0.05680, ridge_size 0.05662, lgb_size 0.05621, catboost_size 0.05453) and outranks mlp_volatility 0.05402 and lgb_volatility 0.05198. The worst cell is mlp_rsi at payout −0.01550 (then ridge_momentum −0.01130, mlp_momentum −0.01099). Whether any blend beat the single best cell is not recorded in the available artifacts.
