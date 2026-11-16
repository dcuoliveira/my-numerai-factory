# E107 — Residualise every model on a 5-day-old meta model, s = 0.5 (no training)

- **State:** done (38/38 arms)
- **Runner:** experiments/e107_residualise_everything_crypto.py
- **Models / datasets:** meta_resid transform over 38 models — the walk-forward ridge on fenccausal, the from-scratch MSE and −CORR networks (seeds 42/43), and every 20-day registry slot; no new dataset
- **Artifacts on disk:** 125.6 MB across 38 forecast runs (plus 38 metrics + 38 per_period series) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E107 — residualising every model does not make them converge; the gain scales with each model's meta overlap"; memory: e106-meta-residualise.md, e108-e111-allmom-pipeline.md, live-resid-slots-and-tenc-fix.md; metrics: results/metrics/crypto/meta_resid/e107_cu_sa_ia_bo_01_s050.json

## Main question
E106's deployable transform (s=0.5, meta >= 5 days old) applied to every model at once: do all models converge to the same residualised payout? If so, model choice is settled and work moves to the residualisation/portfolio layer.

## What was tried
No training. Per day on the 470 days every model shares (2024-07-25..2026-05-13): `rank(gauss(rank(p)) − 0.5·β·gauss(rank(m)))`, m = latest meta model >= 5 days old, scored against the same-day meta model. SACI_03 and CURUPIRA_05_60 (60-day) left out. Gate: s=0 reproduces each source's stored per-period CORR and MMC — but 16 blends' stored series are not reproduced by their own forecasts (up to 7e-3/day, <=4e-5 on the mean, from near-ties in rank averages written at different times), so the gate was set against the rescored forecast. Convergence defined as: the cross-model spread collapses and the share of model pairs differing at |t|>=2 falls to the level of two seeds of one network.

## Trained parameters
No model fit. From results/metrics/crypto/meta_resid/e107_cu_sa_ia_bo_01_s050.json config: model meta_resid; study E107; source_label CU_SA_IA_BO_01; source_run "lgb/boto01_slot_mlpfenc"; meta_lag_days 5; meta_source "data/crypto/v2.0/historical_meta_models.parquet"; strength 0.5; target target_binned_return_20; task regression; transform "rank(gauss(rank(p)) − s * beta * gauss(rank(m_lag))), per day"; n_periods 470. This arm scores CORR.mean 0.15559, MMC.mean 0.10193. The 38 arms vary only by source model.

## What was found
FINDINGS-BACKED. Every model gains but they do NOT converge; the gain scales with each model's meta overlap. Payout gains run +0.0030 (CURUPIRA_05, SACI_02) to +0.0104 (CU_SA_IA_BO_01), all t>=2.2; median gain blends +0.0075, single models +0.0063, ridge and networks +0.0058. No convergence: payout range across models 0.0319→0.0384, std 0.0078→0.0092, pairs differing at |t|>=2 27%→29%, before/after ranking correlation 0.98 (two seeds of one network differ at |t| 0.4–0.7). The gain tracks β (Spearman 0.82; gain ≈ −0.010 + 0.033·β), so residualisation stretches the order: CU_SA_IA_BO_01 goes 0.1071→0.1175 (top of book, beating 24 of 37 others at t>=2); residualised networks (0.090–0.091) stay level with the residualised walk-forward ridge (0.087; t −1.0, −1.5), ~0.027 below CU_SA_IA_BO_01 (t +4.4..+4.9). Cost: calmar falls for 34 of 38 (median 0.088→0.074), median dd(pay) 1.09→1.39; for CU_SA_IA_BO_01 calmar 0.093→0.082. Mean pairwise prediction correlation falls 0.81→0.76. Verdict: model choice is not settled; residualisation is an additive layer, not one that makes models interchangeable. Deployment (2026-10-21): two residualised slot blends went live — Pool A `CU_SA_IA_BO_MA_R_01` (formerly curupira_02) = 50/50 RESID(CU_SA_IA_BO_01)+RESID(CU_SA_IA_BO_MA_01), payout 0.1187 (+0.0110, t +3.73 vs CU_SA_IA_BO_MA_01); Pool B `CU_R_IA_MA_60_01` (formerly curupira_03) = 1/4 each RESID(CURUPIRA_05)+IA_01+MA_FENC+CURUPIRA_05_60, payout 0.1121 (+0.0044, t +1.76). Live transform is src.prediction.meta_resid.
