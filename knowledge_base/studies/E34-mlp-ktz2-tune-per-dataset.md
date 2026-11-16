# E34 — One MLP tune per book dataset with the E33 priors (mlp_ktz2)

- **State:** done (9/9 arms; 1 scout)
- **Runner:** experiments/e34_mlp_ktz2_tune_crypto.sh
- **Models / datasets:** `mlp_ktz2` (E33 re-priored grid), one from-scratch tune per book dataset
- **Artifacts on disk:** 43.5 MB across 9 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E34 / E35 — findings (the E33 priors, tuned and ensembled)" (lines 845–853), §"E34 (mlp_ktz2 tune) vs E28 (mlp_ktz tune)" (854–873); memory: none; metrics: results/metrics/crypto/mlp_ktz2/e34_tenc.json

## Main question
E28 redone on the E33 re-sweep priors instead of E17's. The one change that matters: alpha {1,10} (~100× E17's 0.1 ceiling, the bug that kept IA_01's region out of the box), activation un-pinned to {identity,relu}, lr low peak 1e-3, width 32–128 (weak axis), depth 1–2. From-scratch search of `mlp_ktz2` on each of the 9 book datasets. Paired vs the E28 (old-prior) pick and against the book — does the corrected box reach IA_01?

## What was tried
9 arms, one `mlp_ktz2` RandomizedSearchCV per book dataset. Protocol (from configs): rolling CV, train_window 130, refit_every 130, embargo 20, inner purged 5-fold CV (n_splits 5), random search n_iter 80, seed 42, target `target_binned_return_20`. The only change vs E28 is the grid (the E33-corrected box). Paired vs E28 per dataset, Newey-West lag 20.

## Trained parameters
From `e34_tenc.json`: model `mlp_ktz2`, data `crypto/v2.0-tenc`, n_features 23, n_iter 80, n_splits 5, cv_scheme rolling, train_window 130, refit_every 130, embargo 20, seed 42, target `target_binned_return_20`, fixed_params null (grid searched), model_options {}; CORR 0.152041, MMC 0.075442, pnl 0.039274, n_periods 510. Arms sweep data_version across the 9 book feature sets; the mlp_ktz2 grid (alpha {1,10}, activation {identity,relu}, lr 1e-4..1e-2, width 32–128, depth 1–2) is held fixed.

## What was found
FINDINGS-BACKED. **Correcting the alpha ceiling worked: the searched MLP jumped ~0.004–0.022 payout over its E28 predecessor on 7/9 datasets, climbing into mid-book — but still does not fully reach IA_01 (0.102).**
- E34 − E28 (payout Δ, t): allfeat +0.02246 (+1.44), interact +0.01821 (+1.66), tenc +0.01576 (t +1.94, closest to significant), tenc_onehot +0.01516 (+0.98), fenccausal +0.01342 (+1.45), cumlag3 +0.00877 (+0.62), onehot_rollsel +0.00452 (+0.28), allfeatpp −0.00016 (−0.01), raw −0.00246 (−0.29).
- 7/9 positive, mostly on the datasets whose optimum sat outside E17's box. On ~26 effective windows the consistency and magnitude are the signal. This confirms E33: the 0.1 alpha ceiling was the bottleneck. `raw` is the one arm that did not move — its E28 optimum was already inside the old box.
- Residual ~0.010–0.015 gap to IA_01 is lr granularity (IA_01 fine-searched to ~0.009 vs the grid's 0.01 node) plus the book's very top now being multi-member blends a single-model tune cannot structurally match.
