# E61 — Multi-symbol multivariate (SUR-style) ridge vs the stacked ridge

- **State:** done (36/36 arms)
- **Runner:** experiments/e61_sur_ridge_crypto.sh
- **Models / datasets:** models `sur_ridge` and `ridge_glmnet`, regression + classification, across all 9 book datasets (36 arms)
- **Artifacts on disk:** 174.10 MB across 36 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none; memory: e61-sur-ridge.md, e62-sur-trees.md, litmom-eda-signal-verdict.md; metrics: results/metrics/crypto/sur_ridge/e61_sur_reg_allfeatpp.json, results/metrics/crypto/sur_ridge/e61_sur_reg_fenccausal.json (+ full 36-arm board summarised)

## Main question
Does modelling the cross-symbol error covariance sharpen the ranking? The deployed linear slots fit ONE shared beta on the stacked panel (errors i.i.d.). `sur_ridge` keeps the shared beta but partials out k common cross-symbol factors from the residuals (feasible-GLS / Seemingly-Unrelated-Regressions). n_factors=0 is an EXACT stacked-ridge control, so the value is entirely in searched k>0 beating ridge_glmnet on numerai_corr at paired |t|>=2.

## What was tried
36 from-scratch tunes on one protocol: {sur_ridge, ridge_glmnet} x {regression, classification} x 9 datasets (raw/tenc/interact/fenccausal/cumlag3/tenc_onehot/onehot_rollsel/allfeat/allfeatpp). Every sur_ridge arm pairs against its matched stacked ridge; reg pairs against clf. Symbol routed to fit() (WANTS_SYMBOL) with the date; predict stays x'beta. The n_factors the search KEEPS is the diagnostic. Perf note (memory): first build used a full 460x460 eigh per fit (2.5h/arm), fixed to top-k randomized_svd (~90x, equivalent).

## Trained parameters
From the metrics JSONs: cv_scheme rolling, embargo 20, train_window 130, refit_every 22, n_splits 5, seed 42, n_iter 50, target target_binned_return_20, no fixed_params. Representative best-payout arm e61_sur_reg_allfeatpp (n_features 812), best_params (refit 0): {alpha 100000.0, n_factors 2} — the search KEEPS k=2 factors. Memory records the search keeps k>0 in ~80% of refits.

## What was found
FINDING (null) — from memory file e61-sur-ridge.md: the cross-symbol covariance does NOT help the ranking. sur_ridge beats ridge_glmnet on numerai_corr in 0/18 cells at |t|>=2 (reg mean ΔCORR -0.0073, clf +0.0004). The factor is selected in CV (~80% of refits keep k>0) but dies out-of-sample — the same lesson as FARM/E27/E30: modelling the error structure sharpens the fit's own loss, not the paid metric. On WIDE datasets (763-812 feat) SUR nudges MMC/payout up (not significant, t~0.9-1.5) while LOWERING CORR — a shift toward meta-model contribution, not a ranking gain. Regression >> classification for both models (~+0.06 CORR); the binary up/down reframing is costly.

Consistent with the metrics board read here: within the regression arms, matched sur vs glm pairs are near-ties on CORR (e.g. fenccausal glm 0.13644 vs sur 0.12513; allfeatpp glm 0.13600 vs sur 0.12652 — glm higher on CORR), while sur lifts MMC on the wide sets (allfeatpp sur MMC 0.07018 vs glm 0.05027). Verdict: do not deploy sur_ridge over ridge_glmnet. Report artifact: results/reports/crypto/e61_sur_vs_stacked.html.
