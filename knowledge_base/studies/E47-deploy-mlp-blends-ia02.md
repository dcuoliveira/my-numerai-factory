# E47 — Deploy the E33 winner: a live pinned MLP (IA_02) + two blends

> ⚠️ **PRODUCTION — DO NOT DELETE:** e47_cusaiabo02 / e47_cucaia01 back the UPLOADED slots CU_SA_IA_BO_02 / CU_CA_IA_01. These forecasts are the backtest references `src.analysis.slots` resolves; retention protects them (K1). Deleting them breaks the registry self-check.

- **State:** done (2/2 arms)
- **Runner:** experiments/e47_deploy_mlp_blends_crypto.sh
- **Models / datasets:** per-period rank-average blends priced under models `lgb` (e47_cusaiabo02) and `ridge_glmnet` (e47_cucaia01); the pinned MLP leg is `mlp/e33_fenc_a128_64_relu_lre3_al10` on fenccausal
- **Artifacts on disk:** 5.6 MB across 2 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E47 — deploying the E33 winner: `IA_02` + two MLP blends" (lines 979-1025); memory: none; metrics: results/metrics/crypto/lgb/e47_cusaiabo02.json, results/metrics/crypto/ridge_glmnet/e47_cucaia01.json

## Main question
Not a sweep — a DEPLOYMENT. E33 found the fenccausal MLP peaks at alpha ~10 (the regularisation E17's 0.1 ceiling could not reach). E47 wires that cell live as the pinned MLP component `IA_02` and builds/prices two slot-blends on it, to confirm they are deployable improvements over their members.

## What was tried
E33's top fenccausal cell (128,64 / relu / lr 1e-3 / alpha 10) is wired as pinned MLP component `IA_02` (src/live/registry.py, fit_override.fixed_params, NOT_SUBMITTED; refits live like IA_01 but pins its config). Two slot-blends are built on it and priced as REAL per-period rank-averages of their members' finished forecasts (the deployable number, not the screening proxy which misreads dd(pay) as 0.28 where the real figure is ~1.0), each over 470 periods vs its own best member:
- CU_SA_IA_BO_02 = IA_02 + CURUPIRA_05_60 + CU_SA_IA_BO_01 (a version bump of CU_SA_IA_BO_01, a 2nd neural leg).
- CU_CA_IA_01 = IA_02 + IAUARETE_01 (CatBoost/Caipora) + CURUPIRA_05_60.
Rationale: mlp-vs-mlp blends are worthless (E33 cells correlate rho 0.80-0.98; 0 of 35 blends beat their best member); the value is mlp perpendicular to the tabular book (rho ~ 0.1).

## Trained parameters
These are per-period rank-average blends, equal 1/3 weights, not model fits. From results/metrics/crypto/lgb/e47_cusaiabo02.json: combination "per-period rank average", weights [0.333, 0.333, 0.333], members ["mlp/e33_fenc_a128_64_relu_lre3_al10" (= IA_02), "ridge_glmnet/t60_c05_blend" (= CURUPIRA_05_60), "boto01_slot_mlpfenc" (= CU_SA_IA_BO_01)]. From results/metrics/crypto/ridge_glmnet/e47_cucaia01.json: same combination/weights, members ["mlp/e33_fenc_a128_64_relu_lre3_al10" (IA_02), "catboost_ktz/e32_sel_fenccausal" (= IAUARETE_01), "t60_c05_blend" (= CURUPIRA_05_60)]. The underlying IA_02 config (128,64 / relu / lr 1e-3 / alpha 10) is recorded in EXPERIMENTS.md and the registry, not in these blend JSONs.

## What was found
FINDINGS-BACKED (EXPERIMENTS.md §E47). Both blends were deployed. Over 470 periods vs each blend's best member:
- Best member CU_SA_IA_BO_01: CORR 0.16817, payout 0.10712, c.sharpe 1.92, dd(pay) 1.16, calmar 0.093, burn 38.
- CU_SA_IA_BO_02: CORR 0.16118, payout 0.10635, c.sharpe 2.03, dd(pay) 1.02, calmar 0.104, burn 30 — matches its best member on payout (-0.0008, noise) and improves EVERY risk metric. Deploy for risk, not level.
- CU_CA_IA_01: CORR 0.16003, payout 0.10422, c.sharpe 2.05, dd(pay) 1.08, calmar 0.097, burn 31 — OUT-LEVELS all three members (0.10422 vs ~0.095) and improves risk (mlp/catboost/ridge legs genuinely diverse); the level gain is a point estimate, not a paired |t|>=2 result.

Metrics JSONs corroborate the CORR/MMC: e47_cusaiabo02 CORR mean 0.161177, MMC mean 0.090233, n = 470; e47_cucaia01 CORR mean 0.160034, MMC mean 0.088221, n = 470. Caveats (EXPERIMENTS.md): IA_02 is one config/one seed; both blends inherit their members' fallbacks (CU_SA_IA_BO_02 carries CURUPIRA_SACI_03's example-fallback risk) and both hold the 60-day CURUPIRA_05_60 (so 470 periods). Status: wired and validated — IA_02 fits live in ~22s, both blends reproduce their registry backtest to 1e-6; account slots cu_sa_ia_bo_02 and cu_ca_ia_01 exist and the next open round submits both.
