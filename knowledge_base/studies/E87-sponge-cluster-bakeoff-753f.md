# E87 — SPONGE-cluster bake-off: signed-spectral clusters x 5 models, rank + ensembles

- **State:** partial (35/50 arms — 7 of 10 clusters present on disk: sponge_00/03/05/06/07/08/09)
- **Runner:** experiments/e87_sponge_bakeoff.py
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on SPONGE clusters of `crypto/v2.0-vol-infl-carver-ix-fillmask` (753f)
- **Artifacts on disk:** 152.9 MB across 35 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md — no E87 row exists; memory: none; metrics: results/metrics/crypto/lgb_ktz/e87_lgb_ktz_sponge_07.json (via rep run e87_mlp_ktz_sponge_00)

## Main question
The E86 idea with data-driven feature groups: tune every model class from scratch on each of the 10 SPONGE signed-spectral clusters (Cucuringu et al. AISTATS 2019; signed Pearson affinity, k=10 by silhouette) of the 753f combo, rank the (model × cluster) cells, and screen cross-cluster blends. No single cluster is expected to beat the full combo (which itself trails raw).

## What was tried
10 clusters (sizes 330/176/132/45/24/22/12/4/4/4) partitioning all 753 features, each selected via a `--feature-include` anchored alternation of its exact feature names; 10 clusters × 5 models = 50 from-scratch runs intended at the shared protocol, each paired (NW lag 20) vs the full combo (`e82_*_combo`). Clusters run concurrently (CLUSTER_JOBS=5). On disk only 7 clusters' cells are present (35 runs), so the study is partial — sponge_01/02/04 cells are absent.

## Trained parameters
From the representative metrics JSON results/metrics/crypto/mlp_ktz/e87_mlp_ktz_sponge_00.json: model mlp_ktz; data_version `crypto/v2.0-vol-infl-carver-ix-fillmask`; n_features 330 (cluster sponge_00); cv_scheme rolling, inner_cv purged; train_window 130, refit_every 130, embargo 20; n_splits 5, val_frac 0.3; seed 42; n_iter 50; search_type random; metric corr; target target_binned_return_20; feature_include = the cluster's explicit feature-name alternation (bollinger + feature_ix_* interaction + feature_tf_* trend names). mlp refit-0 best_params: activation identity, alpha 0.01, batch_size 1024, hidden_layer_sizes [32,16,8], learning_rate_init 0.0209. n_features varies by cluster; same protocol across all cells.

## What was found
NUMBERS-ONLY — no written finding; partial run. From the metrics (own 521-period windows, standalone), the compact 24-feature cluster **sponge_07** is the strongest cluster for four of five model classes (ridge, catboost, lgb, mlp; for rf, sponge_03 edges it 0.06549 vs 0.06363): lgb_sponge_07 payout 0.06572 / CORR 0.14501 / MMC 0.05121, ridge_sponge_07 0.06973, catboost_sponge_07 0.06727, rf_sponge_07 0.06363, mlp_sponge_07 0.04992. The 12-feature sponge_03 is next overall. The worst cell is mlp_sponge_08 at −0.01635 (then mlp_sponge_06 −0.01475, ridge_sponge_08 −0.01377). Paired deltas vs the full combo, the ensemble screen, and the 3 missing clusters are not on disk.
