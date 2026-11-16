# E20 — Exponential recency weighting on every submitted model's dataset

- **State:** done (18/18 arms)
- **Runner:** experiments/e20_expdecay_genridge_crypto.sh
- **Models / datasets:** `gen_ridge` on 9 datasets (raw, tenc, fenccausal, interact, cumlag3, tenc-onehot, allfeat, onehot-rollsel, allfeat_pp), each in two variants: exponential recency decay (`exp_`) and uniform weighting (`unif_`, the control).
- **Artifacts on disk:** 91.15 MB across 18 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/gen_ridge/e20_unif_fenccausal.json

## Main question
Does weighting recent training rows more heavily (exponential decay) beat flat/uniform weighting for the generalised-ridge models, across every dataset a submitted slot fits on?

## What was tried
18 arms = 9 datasets × {exp, unif}. Each arm is a searched gen_ridge fit; the only thing that varies within a dataset pair is the sample-weighting scheme. Within-pair comparison (exp vs unif on the same dataset).

## Trained parameters
From results/metrics/crypto/gen_ridge/e20_unif_fenccausal.json: model `gen_ridge`, data_version `crypto/v2.0-fenccausal`, cv_scheme `rolling`, embargo 20, train_window 130, refit_every 130, target `target_binned_return_20`, n_features 44, seed 42, n_iter 50, n_splits 5, val_frac 0.3, fixed_params None (searched). The decay factor of the `exp_` arms is not exposed in this config block; the `unif_` arm is the flat-weight control.

## What was found
No written finding; from the metrics, exponential decay did not help — uniform weighting wins or ties on essentially every dataset. CORR, unif vs exp: interact 0.155160 vs 0.151970; tenc 0.150167 vs 0.148628; fenccausal 0.152440 vs 0.138301; raw 0.143822 vs 0.134609; cumlag3 0.148360 vs 0.146021; tenc-onehot 0.122856 vs 0.115543; allfeat 0.121544 vs 0.115592; allfeat_pp 0.122069 vs 0.116530; onehot-rollsel 0.121748 vs 0.114787. MMC tracks the same direction. Best arm overall is unif_interact CORR 0.155160 / MMC 0.070850. So recency-weighting the training window buys nothing here and slightly hurts on several sets. No paired t-stats recorded.
