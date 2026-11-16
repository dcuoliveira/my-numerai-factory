# E116 — ridge refit cadence × lambda rule: every period / 1 / 3 / 6 months (fenccausal, live lag)

- **State:** done (12/12 arms, scout 12)
- **Runner:** experiments/e116_ridge_lambda_cadence_crypto.py (reuses src/prediction/lambda_selection.py)
- **Models / datasets:** ridge on crypto/v2.0-fenccausal (44f), rolling 130, E115's 51-λ grid, embargo 21 (live train/live gap); 3 λ rules × 4 refit cadences
- **Artifacts on disk:** 82.6 MB across 24 forecast runs (12 full + 12 scout; 24 metrics + 24 hyperparams + 24 per_period) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E116 — refitting LESS often is better: payout rises monotonically from every period to every 6 months, for every λ rule"; memory: e116-ridge-refit-cadence.md; metrics: results/metrics/crypto/ridge/e116_tscv_mse_r1.json, results/metrics/crypto/ridge/e116_preq_payout_r130.json

## Main question
E115 refit monthly and trailed the 6-monthly house ridge, but cadence, grid and rule all differed. Here grid and rule are fixed and ONLY the refit cadence moves — is a staler model worse or better?

## What was tried
Grid (51 alphas 1e-3..1e7) and rule fixed, only refit cadence varies: r1 (every period = every live round, the fastest a live round can refit), r22 (1 month), r65 (3 months), r130 (registry.FIT's 6-month live cadence), for three E115 rules: tscv_corr, preq_payout, tscv_mse → 12 arms. Embargo 21 (not 20) so training ends 22 periods before the predicted date (the live train/live gap); scored dates unchanged (510). Paired per arm vs r130, rules vs tscv_corr per cadence, and r22 embargo-21 vs E115's r22 embargo-20; Newey-West lag 20, with calmar.

## Trained parameters
From results/metrics/crypto/ridge/e116_tscv_mse_r1.json config: model ridge; study E116; data_version crypto/v2.0-fenccausal; cv_scheme rolling; train_window 130; refit_every 1; embargo 21; live_gap_periods 22; n_features 44; n_splits 5; search_type grid; n_iter 51; lambda_rule "tscv", lambda_loss "mse", metric mse; target target_binned_return_20; seed null; warmup_blocks 1019; first_oos_period 2020-07-30; alpha_grid = 51 points 1e-3..1e7; fixed_params null. CORR.mean 0.12875, MMC.mean 0.04671, n=510. The preq_payout r130 arm (e116_preq_payout_r130.json): refit_every 130, lambda_rule "prequential", lambda_loss "payout", warmup_blocks 7; CORR.mean 0.15899, MMC.mean 0.07338.

## What was found
FINDINGS-BACKED. Refitting LESS often is better: payout rises monotonically r1 < r22 < r65 < r130 for all three rules. Payout (calmar): tscv_corr 0.0577(0.036)/0.0672(0.045)/0.0821(0.064)/0.0848(0.066); preq_payout 0.0623(0.044)/0.0664(0.047)/0.0847(0.082)/0.0893(0.073); tscv_mse 0.0596(0.034)/0.0665(0.039)/0.0871(0.069)/0.0900(0.071). Paired vs r130, Δpayout (t): r1 −0.0271(−2.85)/−0.0270(−2.92)/−0.0304(−3.13); r22 −0.0176(−1.92)/−0.0229(−2.69)/−0.0235(−2.96); r65 −0.0027/−0.0046/−0.0029 (all |t|<0.6). Burn improves with slower refit (105–109 at r1 vs 34–47 at r130). It is NOT about λ: the ex-post best fixed λ shows the same order (payout 0.067→0.070→0.088→0.093), and at r1 tscv_corr switches λ 245 times (79 edge picks) yet the stable prequential rule (34 switches) is just as bad. The λ rule still does not matter at any cadence (every rule-vs-tscv_corr |t|<=1.76, extending E115's null). The live-lag embargo did not matter (r22 emb 21 vs E115 r22 emb 20, |t|<=1.24). E115's gap to the house ridge was all cadence: at r130 the arms match e109_ridge_fenc (Δpayout −0.0015..+0.0038, |t|<0.7). Best arms: tscv_mse_r130 (0.0900, calmar 0.071, burn 44) and preq_payout_r130 (0.0893, 0.073, 47). Open (untested): "older training data predicts better" vs r130 phase luck (only 4 fits) — not separated. Verdict: keep the live 6-monthly cadence (registry.FIT refit_every 130); refitting every round is significantly worse. Scouts ran at era_step 4, not reportable. r1 walk-forward took 80 min.
