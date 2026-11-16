# E10 — Walk-forward stacking: a fitted meta-learner over member predictions

- **State:** partial (16/22 arms, 12 scout)
- **Runner:** experiments/e10_stack_crypto.sh
- **Models / datasets:** ridge and mlp meta-learners over member out-of-sample predictions (crypto/v2.0-e10stackR / -e10stackM), plus 1/N, NNLS and base-model controls
- **Artifacts on disk:** 48.3 MB across 20 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E10 — findings" and §"E10 — Wolpert's k-fold inner CV (follow-up)"; memory: farm-combination-replication.md (E27, cross-references E10 as part of the measured combiner null); metrics: results/metrics/crypto/ridge/e10_meta_ridge_stackR.json

## Main question
Every blend on the board is a FIXED rank average. Does a meta-learner fitted on the
members' out-of-sample predictions do better — ridge (signed linear) or MLP (non-linear)
— against 1/N and constrained NNLS on the same members and dates? Each meta is tuned
under both inner CVs: the book's purged chronological split and Wolpert's original
shuffled k-fold (`_kf` arms), to price the purge.

## What was tried
Meta-learners (ridge, mlp) over two within-class member sets (stackR, stackM), each under
two inner CVs (purged and `_kf` k-fold), plus controls: 1/N equal (ctl_eq), constrained
NNLS (ctl_nnls), and base ridge/mlp arms. Paired Newey-West lag 20 on the 151 dates all
arms share. Cross-class stacking (`stackX`) was NOT run, so the study says nothing about
members from different function classes (the case the CURUPIRA slot blends exercise).
State partial: 16/22 arms.

## Trained parameters
From results/metrics/crypto/ridge/e10_meta_ridge_stackR.json: model ridge, regression,
target_binned_return_20, data_version crypto/v2.0-e10stackR, n_features 2 (two members),
cv_scheme expanding, train_window None, refit_every 22, n_splits 5, embargo 20, seed 42,
n_iter 20, alpha grid size 100; refit-0 alpha 28.480. The MLP meta (e10_meta_mlp_stackR)
searched the standard MLP grid and refit-0 chose alpha 12.1547, batch_size 4096,
hidden [256,128], lr 0.001194. The purged alpha ceiling differed by set — 100.0 on
stackR vs 10.0 on stackM.

## What was found
FINDINGS-BACKED. Fitting the combination bought nothing. Against a fixed 1/N rank average
every meta is negative and no healthy arm reaches |t|=2 (payout deltas: ridge stackR
−0.001714 t −1.12; mlp stackR −0.013486 t −2.00; ridge stackM −0.001511 t −0.55;
mlp stackM −0.000062 t −0.13). Against constrained NNLS the deltas are ~zero
(+0.000064 to +0.001604, |t|<=0.77) — signed coefficients and an intercept buy nothing
over non-negative weights. No stack beats the better of its own two members. A borderline
corr-only pair (meta_ridge_stackM vs NNLS t +1.98) does not survive Bonferroni (~2.6) and
vanishes in payout (t +0.77).

A secondary finding: the MLP meta `e10_meta_mlp_stackR` silently collapsed to one constant
value on 44 of 151 dates (refit blocks 2 and 4) without crashing — both dead blocks chose
identical hyperparameters [(64,32), alpha 19.41, lr 0.0055] whose CV scores looked normal;
a narrow-input (2-column) failure, not general MLP fragility, and it does not implicate
IA_01. The follow-up re-tuned the metas under Wolpert's shuffled k-fold: where both fits
are healthy the inner CV changes nothing (3 of 4 arms |t|<=1.55); the one large `_kf`
delta (mlp stackR +0.009928, t +1.46) is a repaired degenerate draw, not better tuning.
Headline unchanged: under either inner CV a fitted meta does not beat 1/N or NNLS.
