# E89 — SPONGE-cluster bake-off on the 1526f fenccausal-pairs combo: 25 clusters x 5 models

- **State:** partial (86/125 arms)
- **Runner:** experiments/e89_sponge_bakeoff_fenccausal_pairs.sh
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on SPONGE clusters of `crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal-pairs` (1526f)
- **Artifacts on disk:** 397.9 MB across 86 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md — no E89 row exists; memory: none; metrics: results/metrics/crypto/mlp_ktz/e89_mlp_ktz_sponge_00.json

## Main question
The E87/E88 bake-off on the biggest dataset: which model best exploits which signed cluster of the 1526f stack (the 1506f fenccausal combo + 20 pair features), and does a cross-cluster blend diversify? Sibling of E87 (753f, 10 clusters) and E88 (1506f, 18 active clusters).

## What was tried
`crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal-pairs` (1526f = the 1506f fenccausal combo + 20 pair `feature_sprd_*` features), with 25 SPONGE signed-spectral clusters from its own sponge_feature_groups.json (k=25, partitioning all 1526; sizes 406/220/216/182/154/.../1). 25 clusters × 5 models = 125 from-scratch runs intended at the shared protocol, each selected via a `--feature-include` anchored alternation. On disk 86 of the 125 cells are present, so the study is partial.

## Trained parameters
From results/metrics/crypto/mlp_ktz/e89_mlp_ktz_sponge_00.json: model mlp_ktz; data_version `crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal-pairs`; n_features 406 (cluster sponge_00); cv_scheme rolling, inner_cv purged; train_window 130, refit_every 130, embargo 20; n_splits 5, val_frac 0.3; seed 42; n_iter 50; search_type random; metric corr; target target_binned_return_20; feature_include = the cluster's explicit feature-name alternation (feature_ix_* / feature_tf_* / feature_sprd_* etc., including `_symmean` twins). mlp refit-0 best_params: activation identity, alpha 0.01, batch_size 1024, hidden_layer_sizes [32], learning_rate_init 0.00273. n_features varies by cluster; same protocol across all cells.

## What was found
NUMBERS-ONLY — no written finding; partial run. From the metrics (521-period windows, standalone), the best entry is a blend, `e89_topblend` payout 0.08190 / CORR 0.14805 / MMC 0.06709; the best single cell is the ~26-feature cluster sponge_20 across model classes — ridge_sponge_20 0.06831 / CORR 0.14178, catboost_sponge_20 0.06368, lgb_sponge_20 0.06211, rf_sponge_20 0.06157. The worst cell is mlp_sponge_05 at −0.02134. Paired deltas vs the full combo and the full ensemble screen are not on disk; a prediction-level `e89_topblend` artifact exists but its paired delta is not recorded.
