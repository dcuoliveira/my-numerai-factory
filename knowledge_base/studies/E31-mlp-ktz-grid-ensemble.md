# E31 — mlp_ktz grid ENSEMBLE per book dataset (fit all, rank-average)

- **State:** done (9/9 arms; 1 scout)
- **Runner:** experiments/e31_mlp_ktz_ensemble_crypto.sh
- **Models / datasets:** `mlp_ktz_ens` (`GridEnsembleRegressor`), one rank-average of the 120-cell E17-prior grid per book dataset
- **Artifacts on disk:** 40.7 MB across 9 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E31 — findings" (lines 589–611), §"Ensemble (E31) vs the searched pick (E28)" (612–637), §"E31 vs the deployed book" (638–652), §"Read against E31 and E30" (768–785); memory: e28-e29-prefix-collision.md; metrics: results/metrics/crypto/mlp_ktz_ens/e31_raw.json

## Main question
For each of the 9 book datasets, fit EVERY cell of the `mlp_ktz` grid (120 = E17 priors: 20 lr × 3 arch × 2 alpha × identity) and rank-average their OOS predictions instead of letting inner CV SELECT one. E28 showed a single searched `mlp_ktz` loses to the book; does ensembling a model's own hyperparameter grid — the same rank-average operator the deployed CURUPIRA_SACI/CU_SA blends use — recover the gap? Paired against the E28 pick per dataset (same data/cadence/seed).

## What was tried
9 arms, each the per-period rank average of all 120 cells on a 130/130 rolling block. Selection is removed, not searched: `--n-iter 1 --n-splits 2` (2 is the CV floor). The combiner ranks each member over the whole predicted block (not per period, since predict never sees period labels) — validated ≈ per-period `run_ensemble` on the narrow sets (per-period scored-CORR difference < 5e-3) and exact live. Paired vs E28, Newey-West lag 20; also ranked against the deployed book on the report common window (n=458).

## Trained parameters
From `e31_raw.json`: model `mlp_ktz_ens`, data_version `crypto/v2.0`, n_features 22, n_iter 1, n_splits 2, cv_scheme rolling, train_window 130, refit_every 130, embargo 20, seed 42, target `target_binned_return_20`, fixed_params null, model_options {}; CORR 0.143853, MMC 0.061582, pnl 0.035928, n_periods 516. Arms sweep data_version across the 9 book feature sets; each fits the full 120-cell E17-prior grid per block.

## What was found
FINDINGS-BACKED. **Averaging the whole mlp_ktz grid beats picking the search-selected best on 7 of 9 datasets, and moves the single MLP from the bottom of the book into its lower middle — but not up to the deployed neural slot.** Mechanism: on a ~26-effective-window CV the selected "best" config is a noisy pick, and an equal-weight rank-average over 120 cells is immune to that selection variance.
- E31 − E28 (payout Δ, t; CORR Δ, t): tenc +0.00903 (t +2.25) / +0.01575 (t +3.13); interact +0.01022 (+1.49) / +0.02984 (t +3.03); cumlag3 +0.01205 (+1.46) / +0.01494 (+1.88); fenccausal +0.00569 (+0.77) / +0.01377 (+1.78); raw +0.00435 (+0.84) / +0.00941 (+1.44); onehot_rollsel +0.00311 (+1.50) / +0.00472 (+1.73); allfeat +0.00228 (+0.61) / +0.00761 (+1.85); allfeatpp −0.00360 (−0.79) / −0.00040 (−0.07); tenc_onehot −0.00642 (−1.86) / −0.00853 (t −2.10, in the direction of HARM). Direction positive on 7/9; the CORR lift is the reliable part (tenc and interact the only arms significantly improving CORR); on payout only tenc clears |t| ≥ 2.
- The exception is the widest one-hot sets (allfeatpp flat, tenc_onehot a loss) — exactly where E28's pick was already strong and the block-vs-per-period combiner is unvalidated.
- Vs the deployed book (n=458): best submitted MLP = best overall `CU_SA_IA_BO_01` 0.10550; pure MLP `IA_01` 0.10160. `e31_tenc` (0.08499) climbed to just below `CURUPIRA_02` (0.08528), above `CURUPIRA_01`/`SACI_04`/`SACI_05` — lower-middle of the 28-arm block — but still −0.017 below `IA_01`. The book's edge remains blending ACROSS models and feature sets, not across one model's hyperparameters.
