# E17 — MLP hyperparameter sensitivity (Kalfa-Timmermann-van der Zwan)

- **State:** done (160/160 arms; 145 forecast runs on disk)
- **Runner:** experiments/e17_hyperparam_sensitivity_crypto.sh
- **Models / datasets:** `mlp` (sklearn) on crypto/v2.0 (22 raw features). A 5×2×4×4 factorial = 160 cells: architecture × activation × learning rate × alpha.
- **Artifacts on disk:** 660.02 MB across 145 forecast runs (largest study in this batch) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none; memory: ktz-priors-and-wrappers.md, e28-e29-prefix-collision.md; metrics: results/metrics/crypto/mlp/e17_a32_identity_lre3_ale1.json

## Main question
Apply the "Overhyped?" (Kalfa–Timmermann–van der Zwan) HP-sensitivity framework to the sklearn MLP: map the whole hyperparameter surface once so the production search grid can be set from evidence rather than one pinned draw. The four readings are marginal distributions, functional-ANOVA importance (which HPs are worth tuning), good/bad corner cases, and an lr/alpha prior.

## What was tried
One pinned walk-forward per cell of the 240-cell (quoted as 5×2×4×4) factorial — nothing searched, each cell fixes its own architecture, activation, learning rate and alpha (the run name encodes them, e.g. `a32_16_8_identity_lre1_ale1`). Architectures descend from [32] through deeper funnels to [32,16,8,4,2]; activation identity vs relu; lr and alpha each on a 4-level grid.

## Trained parameters
From results/metrics/crypto/mlp/e17_a32_identity_lre3_ale1.json: model `mlp`, data_version `crypto/v2.0`, cv_scheme `rolling`, embargo 20, train_window 130, refit_every 130, target `target_binned_return_20`, n_features 22, seed 42, n_iter 1 (pinned), n_splits 5, val_frac 0.3. fixed_params for that representative cell: `{mlp__activation identity, mlp__alpha 0.1, mlp__batch_size 1024, mlp__hidden_layer_sizes [32], mlp__learning_rate_init 0.001}`. Every other cell pins a different point of the same axes.

## What was found
MEMORY-BACKED. The study's deliverable, per ktz-priors-and-wrappers.md: E17 produced the `mlp_ktz` wrapper — a compact 120-cell search grid (a dense 20-level learning-rate axis × 3 archs × 2 alpha) that draws only from the study's GOOD region so a RandomizedSearchCV fine-tunes rather than explores, used at `--n-iter ~80`; `mlp_ktz2` is an E33 re-prior, and `*_ens` variants ensemble the grid instead of searching. No dedicated EXPERIMENTS.md finding section exists. The 160-cell metric sweep (NUMBERS, n=515 for completed cells) shows identity-activation cells clustered at the top: best CORR e17_a32_16_8_4_identity_lre1_ale1 CORR 0.150052 / MMC 0.064183; best MMC e17_a32_identity_lre2_ale1 CORR 0.144671 / MMC 0.069490; top identity cells span ~0.13–0.15 CORR. relu cells are lower, and the deepest relu nets at the smallest lr collapse — several arms land at CORR ≤ 0 with truncated windows (e.g. e17_a32_16_relu_lre1_ale2 CORR −0.00818 at n=373), the known relu-MLP fragility. Bookkeeping caveat (e28-e29-prefix-collision.md): E17 was part of an earlier prefix-collision class (E17/E37), since fixed; this is a tracking note, not a scientific result.
