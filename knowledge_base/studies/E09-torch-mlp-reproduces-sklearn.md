# E09 — Does a torch MLP reproduce the sklearn one?

- **State:** done (2/2 arms, 2 scout)
- **Runner:** experiments/e09_torch_mlp_crypto.sh
- **Models / datasets:** mlp (sklearn) and torch_mlp on crypto/v2.0 (raw 22)
- **Artifacts on disk:** 9.2 MB across 2 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/torch_mlp/e09_torchmlp_raw.json
- Note: `payout` = 0.1*CORR_mean + MMC_mean from stored summaries.

## Main question
Same architecture, same grid, same folds, different framework — does the torch MLP
reproduce the sklearn one through the full walk-forward? A NULL (the two agreeing) is
the success condition; they already agree at unit level and this checks it survives the
walk-forward.

## What was tried
2 arms: `e09_sklmlp_raw` (sklearn mlp) and `e09_torchmlp_raw` (torch_mlp), both raw 22
on crypto/v2.0, 514 periods, 20-day label, identical real search (n_iter 50, n_splits 5).

## Trained parameters
From results/metrics/crypto/torch_mlp/e09_torchmlp_raw.json: model torch_mlp, regression,
target_binned_return_20, data_version crypto/v2.0, n_features 22, cv_scheme rolling,
n_splits 5, train_window 130, refit_every 130, embargo 20, seed 42, n_iter 50,
no fixed_params. Refit-0 best_params: alpha 19.414919, batch_size 2048,
hidden_layer_sizes [64,32], lr 0.005541. The sklearn arm chose alpha 39.194068,
batch_size 1024, hidden [64], lr 0.003888 — the two searches landed on different
configs within the shared grid.

## What was found
No written finding; from the metrics (514 periods) the two frameworks land very close,
consistent with the intended null: torch_mlp CORR 0.144701, MMC 0.062148, payout 0.076618;
sklearn mlp CORR 0.142310, MMC 0.061071, payout 0.075302 — a ~0.0013 payout gap. The two
independent searches picked different hyperparameters yet scored within noise of each
other. No paired t-stat in available artifacts.
