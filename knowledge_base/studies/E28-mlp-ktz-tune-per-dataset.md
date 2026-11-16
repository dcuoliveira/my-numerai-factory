# E28 — One MLP tune per book dataset, with the E17 priors (mlp_ktz)

- **State:** done (9/9 arms; 2 scout)
- **Runner:** experiments/e28_mlp_ktz_tune_crypto.sh
- **Models / datasets:** `mlp_ktz` (E17 sensitivity priors compiled into a search grid), one from-scratch tune on each of the 9 datasets the book trains on (raw, tenc, tenc_onehot, allfeat, allfeatpp, interact, cumlag3, onehot_rollsel, fenccausal)
- **Artifacts on disk:** 0.1 MB across 9 forecast runs (forecast files already reclaimed; only metrics/hyperparams/per_period for 2 arms remain on disk — bytes.forecast = 0, count = 0) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"Ensemble (E31) vs the searched pick (E28)" (lines 612–637) and §"E34 (mlp_ktz2 tune) vs E28 (mlp_ktz tune)" (854–873) — E28 has no standalone findings section; its result is recorded as the baseline in these two paired comparisons; memory: e28-e29-prefix-collision.md; metrics: results/metrics/crypto/mlp_ktz/e28_tenc_onehot.json

## Main question
For each of the 9 datasets the deployed book trains on, run ONE from-scratch hyperparameter search of `mlp_ktz` — the E17 Kalfa-Timmermann-van der Zwan sensitivity priors compiled into a grid (dense learning-rate sweep, architecture as a coarse shallow/deep gate, alpha near-pinned). Does an evidence-primed, lr-focused tune match or beat the earlier broad-grid MLP with less budget and no alpha wandering?

## What was tried
9 arms, one `mlp_ktz` RandomizedSearchCV per book dataset. Protocol (from the configs): rolling CV, train_window 130, refit_every 130, embargo 20, inner purged 5-fold CV (n_splits 5), random search n_iter 80, seed 42, target `target_binned_return_20`, val_frac 0.3. The grid draws only from E17's good region (120 cells: 20-level lr × 3 archs × 2 alpha, identity pinned). Named-id note: E28 and the separate E29 study both use the `e28_` prefix (concurrent-session id collision; separated by model tree).

## Trained parameters
From `e28_tenc_onehot.json`: model `mlp_ktz`, data_version `crypto/v2.0-tenc-onehot`, n_features 763, cv_scheme rolling, inner_cv purged, n_splits 5, n_iter 80, train_window 130, refit_every 130, embargo 20, seed 42, target `target_binned_return_20`, fixed_params null (the grid is searched, not pinned), model_options {}. The 9 arms sweep the data_version across the 9 book feature sets; protocol and seed are held constant.

## What was found
FINDINGS-BACKED (as the baseline in the E31/E34 comparisons). **A single searched `mlp_ktz` lands near the bottom of the book — the book's strength is blending, not one tuned MLP.** E28's best arm `e28_fenccausal` scored payout 0.07816 (CORR, from the JSON, e.g. `e28_tenc_onehot` CORR 0.110495), sitting near the bottom of the 28-arm block, above only `SACI_05`. Per-dataset E28 payouts (from the paired tables): tenc 0.07489, allfeatpp 0.07211, fenccausal 0.07765, tenc_onehot 0.07345, cumlag3 0.07077, interact 0.07098, raw 0.07163, onehot_rollsel 0.06753, allfeat 0.06532. E31 later showed that rank-averaging the whole grid (rather than searching it) beats this pick on 7/9 datasets, and E34 showed the E17 alpha-0.1 ceiling was the bottleneck (re-priored `mlp_ktz2` lifted 7/9 by +0.004 to +0.022 payout). E28 itself is best read as the "search-and-pick" reference those two studies improve on.
