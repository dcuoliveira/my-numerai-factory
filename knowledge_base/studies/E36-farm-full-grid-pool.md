# E36 — FARM over the full (model × feature-group) grid

- **State:** done (12/12 arms)
- **Runner:** experiments/e30_gridfarm_crypto.sh
- **Models / datasets:** `ridge_glmnet` FARM combination over a 39-member pool (best searched run per family × 12 non-onehot feature-groups + 3 E29 GRRs)
- **Artifacts on disk:** 55.5 MB across 12 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none (no dedicated section — only the E30 findings note that this study was renumbered E36); memory: e28-e29-prefix-collision.md (bookkeeping only, no result); metrics: results/metrics/crypto/ridge_glmnet/e30_farm_equal.json, e30_farm_farm1_l2boost.json, e30_farm_ds.json, e30_farm_farm1_postalasso.json

## Main question
The (model × feature-group) singles grid was already near-complete, but E29's FARM used only the 7 top-level blocks, never the 7 raw-groups. Run FARM (equal, farm1/2/3 × {ridge,l2boost,postalasso}, gr, ds) over the FULL grid — the best searched run per (family × 12 non-onehot feature-groups) = 36 base models plus the 3 E29 GRRs = 39 — reusing existing tuned runs. Does a 39-member high-dimensional pool (the paper's regime) let any fitted combiner beat 1/N? (Onehot excluded.)

## What was tried
12 FARM combination arms over the same 39-member pool. Combination config (from the JSONs): method ∈ {equal, farm1/2/3 × ridge/l2boost/postalasso, gr, ds}, beta/update as in the FARM family, update_every 22, lag 20, min_history 60, roll 252. The members are finished forecasts: 12 ridge_glmnet (w6r6 grp + reg runs + e28_glmnet_rollsel), 12 mlp (e15_mlp_* groups), 12 lgb (w6r6 grp + e14 blocks), and gen_ridge e28_grr_pos/neg/symfe.

## Trained parameters
From `e30_farm_equal.json`: model `ridge_glmnet` combination, 39 members, method `equal`, beta_method `ridge`, update_every 22, lag 20, min_history 60, roll 252; n_periods 501. `e30_farm_farm1_postalasso.json`: method `farm1`, beta_method `postalasso`, same 39-member pool and combination params. The members themselves are the pre-tuned per-group single-model runs (not re-fit here).

## What was found
NUMBERS-ONLY — no written finding exists for E36; summarized from the metric JSONs. **The 1/N (equal) arm leads every FARM arm sampled, matching the E27/E29 pattern** (nothing beats the simple average):
- `e30_farm_equal` (1/N): CORR 0.151841, MMC 0.066253, payout 0.081437, n 501.
- `e30_farm_farm1_l2boost`: CORR 0.136491, MMC 0.055406, payout 0.069056, n 501.
- `e30_farm_farm1_postalasso`: CORR 0.111405, MMC 0.049765, payout 0.060905, n 501.
- `e30_farm_ds`: CORR 0.116211, MMC 0.051098, payout 0.062719, n 501.
No paired t-stats are recorded for E36 in EXPERIMENTS.md; the equal arm is best on all three metrics in this sample, and (as in E27/E29) the L2-Boost variants are the least-bad fitted combiners — on payout FARM2·L2-Boost 0.06962 is a hair above FARM1·L2-Boost 0.06906 (the one that shrinks hardest toward equal weights), both far below equal's 0.08144. Treat the "equal wins" reading as consistent with the measured E27/E29 findings but not independently t-tested here.
