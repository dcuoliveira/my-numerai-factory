# E14 — E13's ADL question again, each arm tuned for its own width

- **State:** done (2/2 arms; 2 scout runs)
- **Runner:** experiments/e14_adl_searched_crypto.sh
- **Models / datasets:** `mlp` on two sets — e14_raw (crypto/v2.0, 22 features) and e14_adl (crypto/v2.0-adl, 91 features, the autoregressive-distributed-lag expansion).
- **Artifacts on disk:** 39.57 MB across 2 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/mlp/e14_adl.json

## Main question
E13 pinned a config chosen on 22 columns and applied it to 91, so its negative ADL result partly measured "wrong shape" rather than "lags do not help". Here raw and adl each get a full 50-draw search, removing the width handicap; agreement between the two designs is worth more than either alone.

## What was tried
Two arms, both searched from scratch. Each arm a full 50-draw random search (n_iter 50) so the wide arm is not handicapped by a pinned shape — at the cost of the search noise E12 measured at ~0.024 payout spread.

## Trained parameters
From results/metrics/crypto/mlp/e14_adl.json: model `mlp`, data_version `crypto/v2.0-adl`, cv_scheme `rolling`, embargo 20, train_window 130, refit_every 130, target `target_binned_return_20`, n_features 91, seed 42, n_iter 50, n_splits 5, val_frac 0.3, fixed_params None (searched), search_type random. The raw arm is the same recipe on crypto/v2.0 (22 features); its config JSON was not separately opened.

## What was found
No written finding; from the metrics the raw 22-feature arm clearly beats the 91-feature ADL arm even with each fully searched for its own width: e14_raw CORR 0.151659 / MMC 0.065085 (n=514) vs e14_adl CORR 0.110019 / MMC 0.045533 (n=514). This agrees in direction with E13's negative ADL result — the lag expansion does not help the MLP here — but no paired t-stat is recorded, and payout is not stored directly in the metrics JSON.
