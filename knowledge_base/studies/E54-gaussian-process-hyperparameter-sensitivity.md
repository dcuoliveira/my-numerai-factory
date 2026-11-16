# E54 — Gaussian Process hyperparameter sensitivity (Kalfa-Timmermann-van der Zwan)

- **State:** not started (0/96 scouted)
- **Runner:** experiments/e54_gp_hyperparam_sensitivity_crypto.sh
- **Models / datasets:** model `gp` on the raw 22 of crypto/v2.0
- **Artifacts on disk:** 0.0 MB across 0 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** THIN
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: none on disk

## Main question
Map the GP hyperparameter surface once (the "Overhyped?" KTZ framework, as E17/E27/E48/E49 did for MLP/boosters/RF) so that a `gp_ktz` grid can be set from evidence rather than guessed. Two-way claim of interest: length_scale x alpha (function smoothness vs observation noise, the GP analogue of the paper's lr x alpha).

## What was tried
Planned 96 cells = a 4x2x3x4 factorial: kernel {Matern-1/2, -3/2, -5/2, RBF} x {isotropic, ARD} x length_scale {0.25, 1.0, 4.0} x alpha {1e-3, 1e-2, 1e-1, 1.0}, regression only (GPC has no alpha), on the raw 22. Each cell a pinned walk-forward with the kernel FROZEN (optimizer=None) so a pinned cell measures the pinned value, not the marginal-likelihood basin; subsample 6000 and normalize_y True pinned; n_splits 2. Analysis via src.analysis.hyperparam_sensitivity_gp (marginals, fANOVA, corners, priors). The study never launched: 0 of 96 scout arms ran.

## Trained parameters
config JSON not on disk; no runs were produced (0/96). No forecast headers exist.

## What was found
THIN — the study was never started (state "not started", scout 0/96) and produced no forecasts, metrics, or written finding. No EXPERIMENTS.md section, memory file, or metrics JSON exists for it in the artifacts read. Nothing scientific to record and nothing to delete.
