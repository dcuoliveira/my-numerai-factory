# E33 — MLP sensitivity v2: width + alpha opened, on 3 datasets

> ⚠️ **PRODUCTION — DO NOT DELETE:** e33_fenc_a128_64_relu_lre3_al10 backs IA_02 (member of CU_SA_IA_BO_02 / CU_CA_IA_01). These forecasts are the backtest references `src.analysis.slots` resolves; retention protects them (K1). Deleting them breaks the registry self-check.

- **State:** partial (1626/1728 arms; 11 scout)
- **Runner:** experiments/e33_hyperparam_sensitivity_v2_crypto.sh
- **Models / datasets:** `mlp` pinned walk-forwards; raw (22), fenccausal (44), tenc_onehot (763)
- **Artifacts on disk:** 104.7 MB across 6 forecast runs on disk (most of the 1626 pinned cells are intermediates; metrics/per_period/hyperparams retained for 1637 cells per the manifest counts) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E33 — findings" (lines 786–844), §"E34 / E35 — findings" (845–853), §"E47 — deploying the E33 winner: IA_02 + two MLP blends" (979–1025); memory: ktz-priors-and-wrappers.md; metrics: results/metrics/crypto/mlp/e33_fenc_a256_identity_lre3_al100.json

## Main question
Re-run E17's Kalfa-Timmermann-van der Zwan MLP sweep with the two axes `IA_01` proved matter now OPENED: the first hidden layer unfrozen (width 32/64/128/256) and alpha reaching 100 (E17 stopped at 0.1). `IA_01` — the best MLP slot, an sklearn MLP on fenccausal — selects width 64–128 and alpha 8–10, both OUTSIDE E17's grid, so E17 could not have found it. Produce new priors (for the E34 tune and E35 ensemble) and locate IA_01's region.

## What was tried
1728 pinned walk-forwards = 12 arch × 2 activation × 4 lr × 6 alpha on each of raw(22) / fenccausal(44) / tenc_onehot(763), analysed PER DATASET (different scales; `scratchpad/e33_priors.py`). Each cell a pinned walk-forward (not searched). Alpha opened to {…,100}; activation {identity, relu}; width to 256. ~118 of 1728 cells collapsed (high-lr deep/wide relu) and are filled payout-0 (they map the bad corner). Run across deepfive+deepsix (shared FS, OF=12, resume-enabled); 1626/1728 completed.

## Trained parameters
Representative cell `e33_fenc_a256_identity_lre3_al100.json`: model `mlp`, data `crypto/v2.0-fenccausal`, n_features 44, fixed_params {activation identity, alpha 100, batch_size 1024, hidden_layer_sizes [256], learning_rate_init 0.001}, n_iter 1, n_splits 2, train_window 130, refit_every 130, embargo 20, seed 42, target `target_binned_return_20`; CORR 0.153988, MMC 0.079062, pnl 0.041692, n_periods 510. Across arms the sweep varies activation, alpha, lr, and hidden_layer_sizes within each of the 3 datasets.

## What was found
FINDINGS-BACKED. **The transfer problem was an alpha-ceiling problem.** With alpha opened to ~10, all three datasets agree on the same priors.
- ALPHA is the axis E17 got wrong: alpha 10 is decisively best on every dataset (fenccausal median 0.087 at alpha 10 vs 0.051–0.055 at alpha ≤0.1; raw 0.076; one-hot 0.086). E17's 0.1 ceiling sat entirely below the good region. Alpha 100 falls off again (peak ~1–10).
- WIDTH barely matters (medians flat across 32/64/128/256); IA_01's width 64–128 was incidental — the alpha it paired with is what made it work, so E17's frozen width-32 was not the error.
- DEPTH shallow (depth 1 > 2 > 3); LR low, peak 1e-3 (both match E17). ACTIVATION is the one split: identity has the higher median (robust); relu has the higher peak — fenccausal best relu w128 lr1e-3 alpha10 = 0.0949, one-hot best relu w32 alpha10 = 0.0938.
- IA_01's region is now inside the box and confirmed best; the top fenccausal cell (0.0949) is far above the old ktz-grid oracle (0.091, E31). Residual gap to IA_01 (~0.098–0.102) is lr granularity (IA_01 fine-searched to ~0.009, between E33's 1e-3 and 1e-2 nodes).
- New priors feeding E34/E35: alpha {1,10} (~100× E17's 0.1), activation {identity, relu}, lr 1e-4..1e-2 peak 1e-3, width 32–128, depth 1–2. Downstream deploy (E47): the top cell `128,64 / relu / lr 1e-3 / alpha 10` was wired live as the pinned MLP component `IA_02`, with two new slot-blends (`CU_SA_IA_BO_02`, `CU_CA_IA_01`).
