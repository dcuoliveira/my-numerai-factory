# E15 — The MLP feature table: 7 raw groups + 6 derived blocks, twice

- **State:** done (28/28 arms; 28 scout runs)
- **Runner:** experiments/e15_mlp_feature_groups_crypto.sh
- **Models / datasets:** `mlp` and `torch_mlp`, each on 14 feature sets. The 7 groups that partition the raw 22 (price, liquidity, oscillator, volatility, momentum, size, bollinger), plus derived blocks tenc, spread, interact, fenccausal, rollsel, cumlag3 — and raw itself.
- **Artifacts on disk:** 115.46 MB across 28 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/mlp/e15_mlp_tenc.json, results/metrics/crypto/torch_mlp/e15_torch_mlp_tenc.json

## What was tried / main question
Map which feature groups and derived blocks carry the MLP's signal, fitting every set by BOTH backends and fully searching each arm. Two backends because E09 showed they agree on raw, so a block that moves ONE backend but not the other is search noise — a cheaper guard than a t-statistic against the one-bad-draw failure E12 and E14 hit. First study run wholly on the reverted 1e1 alpha grid.

## Trained parameters
From results/metrics/crypto/mlp/e15_mlp_tenc.json: model `mlp`, data_version `crypto/v2.0-tenc`, cv_scheme `rolling`, embargo 20, train_window 130, refit_every 130, target `target_binned_return_20`, n_features 23, seed 42, n_iter 50, n_splits 5, val_frac 0.3, fixed_params None (searched). The torch_mlp twin uses the same recipe/dataset (e15_torch_mlp_tenc: n_features 23, same cv/window/seed). Each arm swaps only its feature set.

## What was found
No written finding; from the metrics. The full and derived-block arms cluster near CORR 0.15–0.16; the best single arm is mlp_fenccausal CORR 0.16172 / MMC 0.082492 (n=510), with torch_mlp_tenc 0.160115 and torch_mlp_rollsel 0.161080 close behind. The two backends agree on raw (mlp 0.151659 vs torch_mlp 0.144935). The 7 raw sub-groups are much weaker and spread wide: volatility 0.1399 is the only one near full-raw; then size ~0.098, price ~0.052, liquidity ~0.050, momentum ~0.030, oscillator ~0.016, bollinger ~0.012 (MMC ≈ 0). So the MLP's signal lives in the full raw set and the fenccausal/tenc/rollsel/interact derived blocks, not in any single raw group except volatility. No paired t-stats recorded.
