# E29 — Per-subset from-scratch tuning + FARM over the full pool

- **State:** done (18/18 arms)
- **Runner:** experiments/e28_subset_tuning_farm_crypto.sh
- **Models / datasets:** `ridge_glmnet`, `mlp`, `lgb`, `gen_ridge` (GRR); feature subsets raw / tenc / onehot / rollsel / spread / interact / fenccausal; then a FARM combination over a 24-model pool
- **Artifacts on disk:** 126.1 MB across 18 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E29 — findings" (lines 490–542); memory: e28-e29-prefix-collision.md, ktz-priors-and-wrappers.md; metrics: results/metrics/crypto/ridge_glmnet/e28_farm_equal.json, results/metrics/crypto/gen_ridge/e28_grr_symfe.json

## Main question
Two parts. (1) Fill the 3 untuned (family × subset) cells a strict config scan found missing — `ridge_glmnet × rollsel`, `mlp × onehot`, `lgb × onehot` — with from-scratch searches, plus 3 generalised-ridge (GRR) sign/structure arms on one-hot. (2) Run FARM (equal, farm1/2/3 × {ridge, l2boost, postalasso}, gr, ds) over a genuinely high-dimensional 24-model pool (best searched run per family × 7 subsets + the 3 GRRs) — the regime the Lee-Lee paper targets — to re-test E27's finding that 1/N wins.

## What was tried
18 arms: 3 new per-subset tunes + 3 GRR base arms + 12 FARM combination arms. Tuning protocol (rolling walk-forward): train=refit=130, embargo 20, inner purged 5-fold CV, 50-draw random search, seed 42, target `target_binned_return_20`. GRR arms on `crypto/v2.0-onehot`: β≥0, β≤0, and onehot-block-unpenalised (p1e-6, symbol fixed effects). FARM arms combine the 24-member pool; paired vs the 1/N blend of identical members, Newey-West lag 20, 510 periods. (Launched as E28, filed as E29 after an id collision; runs keep the `e28_` prefix, separated by model tree.)

## Trained parameters
Per-subset tune example `e28_grr_symfe.json`: model `gen_ridge`, data `crypto/v2.0-onehot`, n_features 762, model_options {alpha_hi 2, alpha_lo −2, group_specs ["onehot:p1e-6"], structures ["identity"]}, n_iter 50, n_splits 5, train_window 130, refit_every 130, embargo 20, seed 42. FARM baseline `e28_farm_equal.json`: method `equal`, beta_method `ridge`, update_every 22, lag 20, min_history 60, roll 252, over 24 members spanning lgb (w6r6_lgb_reg, e14_fenccausal/interact/rollsel/spread/tenc, e28_lgb_onehot), mlp (a2_mlp_fenccausal, e07_mlpreg_raw, e15_mlp_interact/rollsel/spread/tenc, e28_mlp_onehot), ridge_glmnet (7 runs incl. e28_glmnet_rollsel), and the 3 GRRs; n_periods 510.

## What was found
FINDINGS-BACKED. **(1) The three new tunes filled in as expected, and (2) 1/N wins again — no FARM arm beats equal-weight.**
- New tunes: `mlp / e28_mlp_onehot` CORR 0.1554 (on par with glmnet-on-rollsel and the raw-feature MLPs — confirms E08's dense-first-layer-as-embedding premise), `ridge_glmnet / e28_glmnet_rollsel` 0.1554, `lgb / e28_lgb_onehot` 0.1479 (lower, consistent with the booster discarding the dummies). GRR arms: e28_grr_neg (β≤0) 0.1387, e28_grr_pos (β≥0) 0.1054, e28_grr_symfe (unpenalised onehot) 0.0960 — weakest, as expected with 740 collinear free columns.
- FARM over the 24-member pool: equal CORR 0.1619, payout 0.0904 (baseline). Best arm farm2·l2boost ΔCORR −0.0022 (t −0.99), a tie-to-worse; every ridge/postalasso/GR/DS arm significantly WORSE (ΔCORR t −2.0 to −2.4; ds ΔCORR −0.0125 t −2.35). L2-Boost (shrinks hardest to equal) is again least-bad.
- This closes the combiner question at scale: with the FRED-MD validation, FARM's failure here is the metric (rank correlation), not pool size or implementation. The hand-set equal weights in the deployed CURUPIRA blends remain correct.
