# E35 — Ensemble the mlp_ktz2 (E33-reprior) grid per book dataset

- **State:** done (9/9 arms; 1 scout)
- **Runner:** experiments/e35_mlp_ktz2_ensemble_crypto.sh
- **Models / datasets:** `mlp_ktz2_ens` (grid ensemble of the E33-corrected 120-cell box), one rank-average per book dataset
- **Artifacts on disk:** 40.7 MB across 9 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E34 / E35 — findings (the E33 priors, tuned and ensembled)" (lines 845–853), §"E35 (ensemble of the corrected grid) -- two comparisons" (874–888); memory: none; metrics: results/metrics/crypto/mlp_ktz2_ens/e35_raw.json

## Main question
E31 redone on the E33 priors: fit all 120 cells of the `mlp_ktz2` grid (alpha {1,10}, {identity,relu}, lr 1e-4..1e-2, width 32–128, depth 1–2) and rank-average, vs the E34 search-and-pick over the same grid. The ensemble-vs-pick question (E28/E31) re-asked on the corrected box that now contains IA_01's region.

## What was tried
9 arms, each the per-period rank average of all 120 corrected-grid cells per 130/130 rolling block. Selection removed: `--n-iter 1 --n-splits 2`. Two paired comparisons: vs E31 (old-prior ensemble) and vs E34 (pick over the SAME corrected grid), Newey-West lag 20; also against the book.

## Trained parameters
From `e35_raw.json`: model `mlp_ktz2_ens`, data `crypto/v2.0`, n_features 22, n_iter 1, n_splits 2, cv_scheme rolling, train_window 130, refit_every 130, embargo 20, seed 42, target `target_binned_return_20`, fixed_params null, model_options {}; CORR 0.153022, MMC 0.065569, pnl 0.037534, n_periods 516. Arms sweep data_version across the 9 book feature sets; the full 120-cell corrected grid is fit per block.

## What was found
FINDINGS-BACKED. **Correcting the box lifted the ensemble everywhere, but its edge over the (now-good) search is smaller than in E31.**
- vs E31 (old-prior ensemble): all 9 positive, +0.004 to +0.023 payout, largest on the wide one-hot sets (tenc_onehot +0.023, allfeat +0.021, allfeatpp +0.018).
- vs E34 (pick over the same corrected grid), ensemble − pick: onehot_rollsel +0.0157 (t +2.26), raw +0.0117 (t +1.51), allfeatpp +0.0147 (t +1.66), cumlag3 +0.0078; tenc/interact/fenccausal slightly negative (|t|<0.5); tenc_onehot/allfeat ~flat. The ensemble's edge over the pick is SMALLER than in E31 for a clean reason: the corrected search (E34) is now good, so there is less selection noise for the ensemble to hedge. It still helps where the pick is unreliable (onehot_rollsel, raw, allfeatpp) and is neutral where E34 already found a strong cell.
- Overall (E33→E34→E35 arc): correcting the alpha ceiling moved both the searched MLP and the grid ensemble ~0.01–0.02 payout into mid-book, but neither fully reaches IA_01 (0.102); the residual gap is lr granularity plus the book's top being multi-member blends.
