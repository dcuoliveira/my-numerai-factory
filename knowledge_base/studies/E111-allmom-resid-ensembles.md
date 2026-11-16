# E111 — meta residualisation + ensembles of the all-momentum models (20d and 60d)

- **State:** done (22/21 arms)
- **Runner:** experiments/e111_allmom_resid_ensembles_crypto.py
- **Models / datasets:** meta_resid transform + rank-average ensembles over allmom20 (E108), raw20 (E72 base), allmom60/raw60 (E110) and the book CU_SA_IA_BO_02; 60-day forecasts scored in-target AND on the payable 20-day label
- **Artifacts on disk:** 78.5 MB across 22 forecast runs (plus 23 metrics + 23 per_period) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none; memory: e112-allmom-encodings.md, e108-e111-allmom-pipeline.md; metrics: results/metrics/crypto/meta_resid/e111_raw20_mlp_s050.json, results/metrics/crypto/meta_resid/e111_ens_allmom10_s050.json

## Main question
Does E107's deployable meta residualisation lift the all-momentum models, and does any ensemble of the residualised all-momentum models beat its best member or the residualised book?

## What was tried
No training. E107's transform (s=0.5, meta >= 5 days old, imported from e107 so the arithmetic is identical; gate s=0 reproduces the repo scorer to 1e-9) applied to the held-out forecasts of allmom20 (E108), raw20 (E72 base), allmom60 + raw60 (E110) and CU_SA_IA_BO_02; 60-day forecasts scored in-target and on the payable 20-day label (joined from crypto/v2.0). Part B: every 2/3/4-member equal-weight rank-average of the 10 residualised allmom models (5 @20d + 5 @60d-on-20d), paired (Newey-West lag 20) vs best member and vs the residualised book. Part C: residualised book + best ensemble at 10/20/30%. Validation reported 20d-only (2026-10-21).

## Trained parameters
No model fit. From results/metrics/crypto/meta_resid/e111_raw20_mlp_s050.json config: model meta_resid; study E111; source_run "mlp_ktz/e72_mlp_ktz_base"; meta_lag_days 5; strength 0.5; target target_binned_return_20; transform "rank(gauss(rank(p)) − s * beta * gauss(rank(m_lag))), per day"; n_periods 507; CORR.mean 0.10759, MMC.mean 0.05426. The 10-member allmom ensemble (e111_ens_allmom10_s050.json): source_run "lgb_ktz/e111_ens_allmom10", members = the 5 E108 allmom20 + 5 E110 allmom60 runs listed in its config, assembled_by experiments/e111_allmom10_ensemble.py; CORR.mean 0.11913, MMC.mean 0.05741, n=486.

## What was found
MEMORY-BACKED. No EXPERIMENTS.md section; the memory records the 20d-only validation (435 common meta days): residualisation lifts every arm (raw +0.002..+0.005; book CU_SA_IA_BO_02 +0.0082, t +4.7) BUT calmar FALLS for every arm (book 0.103 → 0.094) — it buys payout at the cost of drawdown. allmom still trails raw after residualisation (ridge −0.018 t −3.3, mlp −0.016 t −2.7). No allmom ensemble beats its best member or the residualised book (ensemble vs resid book −0.036, t ≈ −2.2); book + ensemble at 10% is +0.0003 (not significant). Note on the "target30" request: there is no target_binned_return_30 (only 20/60), so it was treated as 20d. Arm count 22/21 (one extra over the expected 21; the ens_allmom10 run is a separate assembled artifact).
