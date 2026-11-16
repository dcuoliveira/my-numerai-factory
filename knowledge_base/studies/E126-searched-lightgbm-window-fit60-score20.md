# E126 — E123's training-window ladder for the searched LightGBM, fitted on the 60-day label, scored on the 20-day one

- **State:** not started (0 arms; scout 0/7)
- **Runner:** `experiments/e126_train_window_lgb_search_t60_crypto.sh`
- **Models / datasets:** lgb (searched afresh per window) on the group-CONCAT subset of `crypto/v2.0-onehot` (748 feat); fitted/tuned on `target_binned_return_60`
- **Artifacts on disk:** 0.0 MB across 0 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** THIN
- **Sources read:** EXPERIMENTS.md §(table row, line 117 — no dedicated section, status "registered", no result); memory: none; metrics: none on disk

## Main question
The LightGBM twin of E124 (ridge) and E125 (MLP), and the 60-day sibling of E123: does training the searched booster on the smoother 60-day horizon and deploying on the 20-day payable label beat training directly on 20, and does it depend on the window?

## What was tried
Planned: E123's arms — `lgb` on the group-CONCAT subset of `crypto/v2.0-onehot` (748 feat), rolling windows 5/10/15/20/22/65/130, the `--n-iter 60 --n-splits 5` search from scratch at every refit, `--tune-window 130` below 130, refit 130 — but FITTED and TUNED on `target_binned_return_60` with embargo 60 (`--embargo auto`). Every arm to be RE-SCORED on `target_binned_return_20` via `cross_target.score_on_payable_target` and consolidated with E123's fit-20 arms in one table (`experiments/e126_consolidate.py`), each window paired NW lag 20 on shared dates and vs the deployed pinned concat_trio (0.09074) re-scored the same way.

## Trained parameters
config JSON not on disk; no runs exist. No fitted parameters recorded in available artifacts.

## What was found
THIN. The study is registered but NOT started — 0 runs, 0 bytes, scout 0/7, and no finding in EXPERIMENTS.md. Nothing is reclaimable and no result is recorded in available artifacts.
