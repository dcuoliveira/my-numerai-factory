# E23 — Cross-class 1/N blends: does diversity buy what fitting could not?

- **State:** done (4/4 arms)
- **Runner:** experiments/e23_diverse_blend_crypto.sh
- **Models / datasets:** `ridge_glmnet` tree is where the blends are recorded, but the arms are fixed 1/N rank averages over deployed single models: all5 (ridge+mlp+lgb+glmnet+lstm), tab2 (glmnet+lgb), tab4 (ridge+mlp+lgb+glmnet), div3 (glmnet+lgb+lstm).
- **Artifacts on disk:** 11.75 MB across 4 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E23 — findings" (L270–310); memory: farm-combination-replication.md; metrics: results/metrics/crypto/ridge_glmnet/e23_all5.json

## Main question
E10 showed fitting a combiner doesn't beat 1/N when members are homogeneous. E23 asks the complementary question: does a FIXED equal-weight blend capture the benefit of DIVERSITY — specifically, does adding the orthogonal LSTM (0.35–0.41 decorrelation from the tabular members) to a tabular blend help?

## What was tried
Four fixed 1/N rank averages (no fitting). Every figure on the 503 dates all rows share, each paired Newey-West lag 20. The mechanism test is the pair div3−tab2 (does adding the LSTM to a tabular pair help) and all5−tab4.

## Trained parameters
These are ensemble blends written by `run_ensemble`, not model fits: the metrics config block is present but all fields are null (no model / target / data_version / cv_scheme recorded — consistent with the repo note that run_ensemble writes no target into a blend's config). n_periods 503 (all5, tab4) to 510 (div3, tab2). No trainable hyperparameters; members are the named deployed single models at equal weight.

## What was found
A null that refutes the diversity hypothesis that motivated it. (1) The level gain over the best member is real but NOT established: all5 − ridge = +0.003669 payout (+4.4%, t +0.47); at 503 dates / ~26 effective windows SE is 0.0078 and no arm clears |t|=2 (best tab4, t +0.53). (2) The orthogonal member contributes nothing: div3 − tab2 = −0.000943 (t −0.19) — adding the LSTM made it slightly WORSE; all5 − tab4 = +0.000833 (t +0.26). (3) Why: the LSTM pays 0.0607 vs the best member's 0.0835 (27% less), so at 1/N its level drag cancels its decorrelation benefit — diversity only pays if the weak member is downweighted, which 1/N cannot do. (4) Two members as good as four: tab2 0.086336 ≈ tab4 0.086319. Deployability: div3 and all5 contain the LSTM and cannot run live; tab2/tab4 can, and neither beats the book. Read with E25, which tested combiners that CAN downweight. (Metric-JSON cross-check, CORR/MMC: all5 0.157732/0.071379, tab4 0.157823/0.070537, tab2 0.157112/0.071916, div3 0.153482/0.071362.)
