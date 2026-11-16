# E08 — The MLP on the four one-hot datasets the book runs on

- **State:** done (8/8 arms, 13 scout)
- **Runner:** experiments/e08_mlp_onehot_crypto.sh
- **Models / datasets:** mlp and torch_mlp on one-hot-bearing datasets (crypto/v2.0-onehot-rollsel ~786f, v2.0-tenc-onehot-spread-interact ~788f, allfeat, allfeatpp)
- **Artifacts on disk:** 39.6 MB across 8 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/mlp/e08_mlp_onehot_rollsel.json
- Note: `payout` = 0.1*CORR_mean + MMC_mean from stored summaries.

## Main question
Can an MLP use the 740 one-hot ticker dummies? Ridge cannot form a per-ticker
interaction from them and LightGBM discards them outright (fe_lgb_reg_onehot came back
byte-identical to raw), but a dense first layer over a one-hot is an embedding lookup.
A scaler bug that sent rare dummies to z=197 was fixed (ScaleContinuousOnly, 197.48 -> 2.29)
before this ran.

## What was tried
8 arms = {mlp, torch_mlp} x 4 one-hot datasets (allfeat, allfeatpp, tenc_onehot,
onehot_rollsel). Scoped down from 12 to 4 because E15 covered the rest; these are the
last of the 9 datasets the submitted book trains on. 510 periods, 20-day label,
real search (n_iter 50, n_splits 5).

## Trained parameters
From results/metrics/crypto/mlp/e08_mlp_onehot_rollsel.json: model mlp, regression,
target_binned_return_20, data_version crypto/v2.0-onehot-rollsel, n_features 786,
cv_scheme rolling, n_splits 5, train_window 130, refit_every 130, embargo 20, seed 42,
n_iter 50, no fixed_params. Refit-0 best_params: alpha 8.227241, batch_size 2048,
hidden_layer_sizes [64,32], lr 0.008886238. The torch_mlp allfeat arm (n_features 788)
chose alpha 0.000122, batch_size 1024, hidden [32], lr 0.003070 and took ~600 min.

## What was found
No written finding; from the metrics (510 periods): the narrower one-hot datasets work,
the wide "allfeat/allfeatpp" ones hurt. Best `e08_mlp_onehot_rollsel` payout 0.092143
(CORR 0.155924, MMC 0.076550); `e08_torch_mlp_tenc_onehot` 0.090045 (CORR 0.155421);
`e08_mlp_tenc_onehot` 0.088577; `e08_mlp_allfeat` 0.088890. The degraded arms are
`e08_mlp_allfeatpp` 0.049293 (CORR 0.080699) and `e08_torch_mlp_allfeat` 0.062677
(CORR 0.102747) — the torch and sklearn MLPs disagree sharply on the widest sets. No
paired t-stats in available artifacts.
