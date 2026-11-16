# E123 — Training window of a FROM-SCRATCH-SEARCHED LightGBM (concat_trio feature set), 1 week to 6 months

- **State:** not started (0 arms; scout 0/7)
- **Runner:** `experiments/e123_train_window_lgb_search_crypto.sh`
- **Models / datasets:** lgb (searched afresh per window, NOT pinned) on the group-CONCAT subset of `crypto/v2.0-onehot` (748 feat)
- **Artifacts on disk:** 0.0 MB across 0 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** THIN
- **Sources read:** EXPERIMENTS.md §(table row, line 114 — no dedicated section, status "registered", no result); memory: none; metrics: none on disk

## Main question
The from-scratch-search twin of E120 (and the true LightGBM analogue of E117, which searched its model per window). When the model may get SIMPLER as its history shrinks, does a short window still lose, or does a re-tuned booster recover the payout that pinning the 4327-tree booster (E120) threw away?

## What was tried
Planned: same base as E120 — `lgb` on the group-CONCAT subset of `crypto/v2.0-onehot` (748 feat), refit 130, embargo 20, seed 42, windows 5/10/15/20/22/65/130 — but the booster searched afresh at every window (`--n-iter 60 --n-splits 5`) instead of pinned. The plain `lgb` grid spans concat_trio on every axis (n_est 2k–30k, lr 0.001–0.01, colsample 0.05–0.30, num_leaves 31–1023) EXCEPT `min_child_samples` (grid floor 1000 vs concat_trio's 50), so w130 is NOT expected to reproduce the pin. Windows < 130 use `--tune-window 130`; w130 searches its own window. Intended comparisons: paired NW lag 20 vs the searched w130, cross-paired vs the pinned E120 arm at each window (does searching beat pinning?), and vs the deployed pinned concat_trio (`e120_w130` = 0.09074).

## Trained parameters
config JSON not on disk; no runs exist. No fitted parameters recorded in available artifacts.

## What was found
THIN. The study is registered but NOT started — 0 runs, 0 bytes, scout 0/7, and no finding in EXPERIMENTS.md. Nothing is reclaimable and no result is recorded in available artifacts.
