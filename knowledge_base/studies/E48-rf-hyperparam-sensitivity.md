# E48 — Random-forest hyperparameter sensitivity (Kalfa-Timmermann-van der Zwan)

- **State:** done (192/192 arms)
- **Runner:** experiments/e48_rf_hyperparam_sensitivity_crypto.sh
- **Models / datasets:** model `rf` (bagged forest); crypto/v2.0 (raw 22), `target_binned_return_20`
- **Artifacts on disk:** 892.7 MB across 192 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none; memory: ktz-priors-and-wrappers.md, e28-e29-prefix-collision.md, no-working-interpreter-build-a-venv.md; metrics: results/metrics/crypto/rf/e48_ne300_d12_mf03_ml10.json

## Main question
Run the "Overhyped?" (Kalfa-Timmermann-van der Zwan) hyperparameter-sensitivity framework — previously run on the MLP (E17) and LightGBM (E27/E49) — on a BAGGED FOREST, a model class the book had never tried. Map the forest surface once so a future rf_ktz grid can be set from evidence, not sklearn defaults.

## What was tried
One pinned walk-forward per cell of a 3x4x4x4 factorial = 192 cells: n_estimators {100,300,1000} x max_depth {3,6,12,None} x max_features {0.1,0.3,0.6,1.0} x min_samples_leaf {1,10,100,1000}, on the raw 22 of crypto/v2.0. A forest is bagging not boosting, so there is NO learning-rate axis: max_features carries the Numerai colsample idiom and min_samples_leaf the heavy-leaf idiom, while n_estimators + max_depth stand in for the boosters' lr + num_leaves. bootstrap and max_samples pinned at the forest default. Then the paper's four readings (marginal distributions, functional-ANOVA importance, good/bad corners, priors); two-way claim tested = max_features x min_samples_leaf. n_splits 2 (pinned cells record only an unused cv_score). Read with src.analysis.hyperparam_sensitivity_rf. Operational note (memory): E48 ran under its own venv `/home/danielco/.e48venv/bin/python`.

## Trained parameters
From results/metrics/crypto/rf/e48_ne300_d12_mf03_ml10.json (one representative cell): model `rf`, data_version crypto/v2.0, n_features 22, target target_binned_return_20, cv_scheme rolling, train_window 130, refit_every 130, embargo 20, n_splits 2, n_iter 1 (pinned, no search), seed 42, val_frac 0.3, inner_cv purged. fixed_params for this cell: max_depth 12, max_features 0.3, min_samples_leaf 10, n_estimators 300. Across the 192 cells these four fixed_params sweep the factorial above (one cell per run name e48_ne<..>_d<..>_mf<..>_ml<..>).

## What was found
MEMORY-BACKED. The distilled outcome is recorded in ktz-priors-and-wrappers.md as the compiled rf_ktz (E48) search grid: "60 cells: depth {3,4,6} x max_features {.05..0.3} x min_samples_leaf {300..4000}; --n-iter ~40" — i.e. E48 concluded shallow depth, dense-low max_features, and heavy leaves. The four underlying findings are recorded in the E51/E52 study ideas (derived metadata): n_estimators is a fANOVA red herring (share ~0.0002, pinned at 100), max_depth shallow won monotonically {3,4,6}, max_features dense-low {0.05-0.30} (0.6/1.0 dropped, no good-cell support at 1.0), and min_samples_leaf monotone heavy to the grid edge (ceiling later OPENED past E48's binding max of 1000 to {300,1000,2000,4000}). From the one cell opened here: e48_ne300_d12_mf03_ml10 CORR mean 0.123411 (sharpe 1.562413, max_drawdown 0.3872), MMC mean 0.052353, n = 517. The best E48 cell reached CORR ~0.1561 per the E51 idea (that specific best-cell number is not independently in EXPERIMENTS.md or an opened JSON). No dedicated EXPERIMENTS.md findings section exists for this study.
