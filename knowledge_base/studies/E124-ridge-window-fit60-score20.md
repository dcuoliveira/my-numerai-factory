# E124 — E117's training-window ladder, fitted on the 60-day label, scored on the 20-day one

- **State:** partial (4/7 arms per manifest snapshot; 7 window metrics files now on disk)
- **Runner:** `experiments/e124_train_window_t60_crypto.sh`
- **Models / datasets:** ridge_glmnet on allfeat_pp − tenc = `crypto/v2.0-tenc-onehot-rollsel-spread-interact` (811 feat); fitted/tuned on `target_binned_return_60`
- **Artifacts on disk:** 17.46 MB across 6 forecast runs (manifest snapshot; disk now holds all 7 window arms w5…w130) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E117 / E124 — six months of history fit on the 20-day label is the best ridge" (line 1438; dedicated section with fit-20/fit-60 payout tables and fit60−fit20 Newey-West deltas) and index row (line 115; status "done", finding recorded); memory: none; metrics: `results/metrics/crypto/ridge_glmnet/e124_w130.json`

## Main question
Does fitting E117's ridge window-ladder on the smoother 60-day label and deploying on the 20-day payable label beat training directly on 20, and does it depend on the window? Deliverable is the 20-day RE-SCORE via `cross_target.score_on_payable_target`, consolidated with E117's seven fit-20 arms in one table.

## What was tried
Same model, data and windows as E117 — ridge_glmnet on allfeat_pp − tenc (811 feat), rolling windows 5/10/15/20/22/65/130, lambda searched from scratch at every refit, windows < 130 tuned on the 130 periods ending where the window ends — but FITTED and TUNED on `target_binned_return_60` with the embargo it forces (60). Two inner folds instead of four (exactly production's own 60-day search). refit 130, seed 42, cv rolling/purged, n_iter 50. Paired t intended at Newey-West lag 20 and lag 60.

## Trained parameters
From `results/metrics/crypto/ridge_glmnet/e124_w130.json`: model `ridge_glmnet`, data_version `crypto/v2.0-tenc-onehot-rollsel-spread-interact`, n_features 811, n_iter 50, refit_every 130, embargo 60, train_window 130, seed 42, target `target_binned_return_60`, fixed_params null. 4 refits. Searched best_params (single ridge alpha): refit 0 `alpha` 137.40; refit 3 `alpha` 15353.03 (the λ moves by two orders of magnitude across refits). Only `train_window` swept across the other arms.

## What was found
FINDINGS-BACKED. CAVEAT on the JSONs: the metrics JSONs score each arm on the 60-day label, so the per-arm figures below are the fit-60 intermediate, not the payable 20-day re-score. The 20-day re-score — the point of the study — IS recorded in EXPERIMENTS.md §E117/E124 (line 1438), consolidated with E117's seven fit-20 arms on 470 common dates (2024-07-25 to 2026-05-13). 60-day-label figures (payout = 0.1·CORR + MMC from summary means — ranking-grade only), all arms n=470:
- w5 CORR 0.113304, MMC 0.047933, pay 0.059263
- w10 CORR 0.109517, MMC 0.041592, pay 0.052544
- w15 CORR 0.120505, MMC 0.034370, pay 0.046421
- w20 CORR 0.104686, MMC 0.032514, pay 0.042983
- w22 CORR 0.101437, MMC 0.031665, pay 0.041808
- w65 CORR 0.097604, MMC 0.040096, pay 0.049856
- **w130 (best) CORR 0.146932, MMC 0.069154, pay 0.083848**

Direction (on the 60-day label): w130 is best; the shorter windows are flatter and non-monotonic (w5 > w65 on CORR).

FINDING (EXPERIMENTS.md §E117/E124, line 1438; index row line 115, status "done"): on the payable 20-day re-score, fitting on the 60-day label helps only where there is too little history to fit the 20-day one — fit60−fit20 payout +0.03335 at 1 week (t +1.14) and +0.02911 at 2 weeks (t +1.01), cutting dd(pay) from 8.96 to 1.88 at one week — and hurts from 4 weeks up, with the crossover at about three weeks. Nothing clears |t| > 2 (0 of 7), and nothing beats six months fit on 20 (payout 0.08955, dd 1.73, calmar 0.052); six months fit on 60 pays 0.07784 with three times the drawdown.
