# E25 — Weighted and selection combiners over a diverse pool

- **State:** done (16/16 arms)
- **Runner:** experiments/e25_weighted_combiners_crypto.sh
- **Models / datasets:** `ridge_glmnet` tree. 8 schemes × 2 member pools: pool5 (5 members incl. LSTM) and tab4 (4 tabular members). Schemes: equal (1/N control), shrunk_tangency, inverse_variance, complete_subset, bagged_caruana, granger_ramanathan, caruana, stacking_nnls.
- **Artifacts on disk:** 63.92 MB across 16 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E25 — findings" (L311–357); memory: farm-combination-replication.md; metrics: results/metrics/crypto/ridge_glmnet/e25_pool5_granger_ramanathan.json

## Main question
E23 predicted that a combiner able to DOWNWEIGHT a weak-but-orthogonal member (the LSTM) would capture the diversity 1/N could not. E25 tests eight walk-forward weighting/selection schemes against the 1/N blend of the identical members to measure exactly that.

## What was tried
Each scheme run over both pools via `run_weighted_ensemble`, paired against the 1/N (`equal`) blend of the same members, 503 common dates, Newey-West lag 20. The study also records each scheme's mean weight on the LSTM, to see whether it up- or down-weighted the weak member.

## Trained parameters
Ensemble/combiner blends: the metrics config block is null except seed 42 (no model/target/data_version/cv — these are combiners over the deployed single models, not fresh fits). n_periods 503 for every arm. The only "parameters" are the scheme and its pool; weights are estimated per walk-forward step inside run_weighted_ensemble.

## What was found
Not one weighting or selection scheme beat 1/N — all 14 deltas negative (14/14). The single strongest signal in the study is that a fitted combiner, stacking_nnls on pool5, is WORSE than equal weights at −0.008034 payout (t −2.01); nothing clears the 16-test Bonferroni bar (|t|≈2.9). Adding the orthogonal member makes weighting HARDER: 6 of 7 schemes are more negative on pool5 than tab4 (granger_ramanathan −0.0052 vs −0.0009; complete_subset −0.0037 vs −0.0005). E23's mechanism prediction was WRONG and this measures it: the schemes OVERweight the weak LSTM (mean weight 0.173–0.359 vs 1/N's 0.200; granger_ramanathan 0.359, complete_subset 0.270), chasing its decorrelation at the cost of level (it pays 0.061 vs the best member's 0.083). Wiring verified: `equal` reproduces the E23 control to 0.000000. Taken with E10 and E23 (~30 combiner arms total), nothing beats a plain equal-weight rank average — the combiner is not the bottleneck, and the deployed CURUPIRA slot blends' hand-set equal weights are not leaving money on the table. Every pool5 arm contains the LSTM and cannot run live. (Metric-JSON cross-check: pool5_equal CORR 0.157732, tab4_equal 0.157823; every scheme arm sits at or below its pool's equal.)
