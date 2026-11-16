# E21 — Cluster regularisation on allfeat_pp: shrink some blocks harder than others

- **State:** done (11/11 arms)
- **Runner:** experiments/e21_cluster_reg_crypto.sh
- **Models / datasets:** `gen_ridge` on allfeat_pp (crypto/v2.0-tenc-onehot-rollsel-spread-interact, 812 features). Arms apply different L2 shrinkage to different feature blocks.
- **Artifacts on disk:** 52.11 MB across 11 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/gen_ridge/e21_int_x10.json

## Main question
On the 812-feature allfeat_pp set, does penalising some feature blocks (interactions, one-hot, derived) harder than others — rather than one flat penalty — improve the generalised-ridge fit?

## What was tried
11 arms, all on the same 812-feature dataset: `flat` (uniform shrinkage control), `int_x10`/`int_x100` (interaction block penalty ×10/×100), `onehot_x10`/`onehot_x100`, `derived_x10`, `rawfree`/`rawfree_oh10`/`rawfree_hard` (leave the raw block lightly/un-penalised at varying strengths), `searched`, and `fuse`.

## Trained parameters
From results/metrics/crypto/gen_ridge/e21_int_x10.json: model `gen_ridge`, data_version `crypto/v2.0-tenc-onehot-rollsel-spread-interact`, cv_scheme `rolling`, embargo 20, train_window 130, refit_every 130, target `target_binned_return_20`, n_features 812, seed 42, n_iter 50, n_splits 5, val_frac 0.3, fixed_params None. The per-block penalty multipliers are encoded by the arm, not in this config block.

## What was found
No written finding; from the metrics, differential block shrinkage barely moves the needle. The flat control is CORR 0.122069 / MMC 0.060387. Best CORR is e21_onehot_x100 0.125098 (+0.003 over flat, but MMC drops to 0.052045); best MMC is e21_fuse 0.062185 (CORR 0.122763). Several arms sit at or below flat (rawfree_hard 0.116048, rawfree 0.116978, searched 0.117544, int_x10/x100 ~0.121). No arm separates cleanly from the flat penalty. No paired t-stats recorded.
