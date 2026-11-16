# E16 — Cumulative per-symbol differences on a lag ladder to one year

- **State:** done (12/12 arms; 9 scout runs)
- **Runner:** experiments/e16_cumdiff_mlp_crypto.sh
- **Models / datasets:** `mlp` on 6 lag rungs (1, 5, 22, 66, 132, 264 days), each in two variants: differences alone (`dXonly`) and raw 22 plus differences (`dX`). Control is e15_mlp_raw, reused.
- **Artifacts on disk:** 79.56 MB across 12 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/mlp/e16_d264only.json

## Main question
Does handing the MLP per-symbol DIFFERENCES rather than levels help? A ladder (not every integer lag) because 264 orders would add 5,808 columns. The target difference is anchored at the 20-period horizon (unanchored it correlates +0.2805 with the label, anchored −0.0114).

## What was tried
Twelve arms: each rung cumulative over those below it, in diff-only and raw+diff variants, searched. Compared to the reused raw control e15_mlp_raw (CORR 0.151659).

## Trained parameters
From results/metrics/crypto/mlp/e16_d264only.json: model `mlp`, data_version `crypto/v2.0-e16d264only`, cv_scheme `rolling`, embargo 20, train_window 130, refit_every 130, target `target_binned_return_20`, n_features 138, seed 42, n_iter 50, n_splits 5, val_frac 0.3, fixed_params None. Each arm's dataset is its own `crypto/v2.0-e16d<rung>[only]` version; feature count grows with the rung.

## What was found
No written finding; from the metrics, differences do not help the MLP. The raw+diff arms at best reproduce the raw control and degrade as the ladder lengthens: d1 CORR 0.149385, d5 0.146825, d22 0.122967, d132 0.097276, d66 0.092531, d264 0.084023 — all at or below raw's 0.151659. The diff-only arms are far weaker across the board: d1only 0.018024, d5only 0.015662, d22only 0.042912, d66only 0.022753, d132only 0.041899, d264only 0.018880. So levels already carry what the differences would, and longer-horizon cumulative differences actively cost CORR. No paired t-stats recorded.
