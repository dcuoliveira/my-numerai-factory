# E26 — Sign-constrained GRR on allfeat_pp: all betas positive, all betas negative

- **State:** done (5/3 arms; 3 single-fit arms + 2 ensembles, 5 runs on disk)
- **Runner:** experiments/e26_sign_constraint_crypto.sh
- **Models / datasets:** `gen_ridge` on allfeat_pp (crypto/v2.0-tenc-onehot-rollsel-spread-interact, 812 features). Arms: `free` (unconstrained), `neg` (all betas ≤ 0), `pos` (all betas ≥ 0), plus ensembles `ens_all` and `ens_posneg`.
- **Artifacts on disk:** 19.74 MB across 5 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/gen_ridge/e26_free.json

## Main question
Does constraining the generalised-ridge coefficients to a single sign (all positive, or all negative) help or hurt on the 812-feature allfeat_pp set — and can an ensemble of the two constrained fits recover anything the unconstrained fit has?

## What was tried
Three constrained single fits (free / neg / pos) on the same 812-feature dataset, searched, plus two 1/N-style ensembles over them (ens_all = all three, ens_posneg = pos + neg).

## Trained parameters
From results/metrics/crypto/gen_ridge/e26_free.json: model `gen_ridge`, data_version `crypto/v2.0-tenc-onehot-rollsel-spread-interact`, cv_scheme `rolling`, embargo 20, train_window 130, refit_every 130, target `target_binned_return_20`, n_features 812, seed 42, n_iter 50, n_splits 5, val_frac 0.3, fixed_params None. The sign constraint is encoded by the arm; the two `ens_*` arms are ensembles (their config data_version is null, written by the ensemble path).

## What was found
No written finding; from the metrics, sign constraints hurt CORR. `free` (unconstrained) is CORR 0.159912 / MMC 0.071664 — the top single-fit CORR in this batch. Constraining signs degrades it: `neg` (all betas negative) 0.141564 / 0.055407, `pos` (all betas positive) 0.106876 / 0.069913. The ensembles recover MMC but not free's CORR: ens_posneg CORR 0.151202 / MMC 0.073202 (study-best MMC), ens_all 0.155140 / 0.072796. So the free fit already uses both signs optimally; forcing one sign costs CORR, and blending the two constrained fits regains MMC level but still trails free on CORR. No paired t-stats recorded.
