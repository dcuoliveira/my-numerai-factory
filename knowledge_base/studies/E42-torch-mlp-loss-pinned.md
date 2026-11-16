# E42 — torch MLP training-loss study, config PINNED (learning dynamics)

- **State:** done (12/12 arms, 3 scout)
- **Runner:** experiments/e42_torch_mlp_loss_dynamics_crypto.sh
- **Models / datasets:** model `torch_mlp_loss`; crypto/v2.0 (raw 22) and crypto/v2.0-fenccausal (44), `target_binned_return_20`
- **Artifacts on disk:** 67.1 MB across 12 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §(see collision note below); memory: none; metrics: results/metrics/crypto/torch_mlp_loss/e42_bmcproxy_fenccausal.json, results/metrics/crypto/torch_mlp_loss/e42_mse_raw.json

> **ID-collision warning (deletion-relevant).** The manifest matched this study to an EXPERIMENTS.md section headed "E42 — cross-sectional rank allocator does NOT beat the static mv book", but that section describes a DIFFERENT study — the NasdaqJockey rank-allocator in `src/analysis/xs_stake.py` (arms B0-B3/A1-A3, bootstrap ΔSharpe). That is NOT this torch-MLP-loss study (runner `experiments/e42_torch_mlp_loss_dynamics_crypto.sh`, model `torch_mlp_loss`). Two studies share the id E42. Do NOT treat the xs_stake finding as this study's result. This file is therefore NUMBERS-ONLY from the metrics JSONs.

## Main question
Does the training OBJECTIVE change what the torch MLP learns and how stably, when the architecture/optimiser config is held fixed? The config is pinned so only the loss varies, isolating the loss's effect on learning dynamics (per-epoch curve, convergence, pred_std collapse, train/val gap). E43 is the companion that searches a config per loss.

## What was tried
12 arms = 6 losses x 2 feature sets. Losses: mse (control), corr, bmc vs the sparse real meta model (benchmark), bmc vs the full-coverage meta_proxy, sharpe, calmar. Feature sets: raw (crypto/v2.0, 22 feat) and fenccausal (crypto/v2.0-fenccausal, 44 feat). bmc-bench vs bmc-proxy is the "optimise on MMC-only vs estimate with BMC" contrast. All arms share one pinned config; evaluated on the board and via learning dynamics (loss_dynamics.py). 3 scout arms also ran.

## Trained parameters
From results/metrics/crypto/torch_mlp_loss/e42_bmcproxy_fenccausal.json (representative): model `torch_mlp_loss`, data_version crypto/v2.0-fenccausal, n_features 44, target target_binned_return_20, cv_scheme rolling, train_window 130, refit_every 130, embargo 20, n_splits 2, n_iter 1 (pinned, no search), seed 42, val_frac 0.3, inner_cv purged. fixed_params: mlp__alpha 0.001, mlp__batch_size 1024, mlp__hidden_layer_sizes [64, 32], mlp__learning_rate_init 0.001. model_options.loss = "bmc" with bmc_reference "meta_proxy" (proxy_alpha 1.0). The raw mse control (e42_mse_raw.json) shares the identical pinned fixed_params with model_options.loss = "mse", n_features 22. Across arms, only `model_options.loss` (and bmc_reference for proxy vs bench) varies; the pinned fixed_params are constant.

## What was found
No written finding applies to this study (the matched EXPERIMENTS.md section is a different study — see collision warning). From the metrics JSONs, the two opened arms: e42_bmcproxy_fenccausal CORR mean 0.067864 (sharpe 0.764477), MMC mean 0.033452, n = 510; e42_mse_raw (control) CORR mean 0.070692 (sharpe 1.001709), MMC mean 0.029492, n = 516. Pinned-config arms land in the ~0.07 CORR range; a full loss-by-loss ranking with learning-dynamics reading is not recorded in available artifacts.
