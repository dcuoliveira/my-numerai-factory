# E18 — TabNet at library defaults, on every submitted slot's dataset

- **State:** partial (5/9 arms)
- **Runner:** experiments/e18_tabnet_crypto.sh
- **Models / datasets:** `tabnet` (dreamquark-ai pytorch-tabnet, added to prediction/models.py). 5 narrow arms run: raw (crypto/v2.0, 22f), tenc (crypto/v2.0-tenc), interact, fenccausal (44f), cumlag3. The 4 wide arms (763–812f: tenc_onehot, allfeat, onehot_roll, allfeat_pp) were deliberately NOT run.
- **Artifacts on disk:** 24.41 MB across 5 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E18 — findings" (L147–178); memory: none; metrics: results/metrics/crypto/tabnet/e18_fenccausal.json

## Main question
Is dreamquark-ai's TabNet (an attention-based deep-tabular net whose fit survives joblib, so it is deployable through the normal refit path, unlike the LSTM) competitive on this panel? A viability probe at pinned library defaults — one arm per distinct dataset a submitted slot trains on — not a tuned model.

## What was tried
One arm per training dataset, parameters PINNED to library defaults (nothing searched). n_splits=2 not 5 because with fixed params the inner CV score is unused and the per-period result is byte-identical either way; no eval_set, to keep the fit leak-free. Each arm paired against the deployed slot trained on the identical dataset, per-period payout, Newey-West lag 20, scored on the 510-period tail (the deployed-slot basis).

## Trained parameters
From results/metrics/crypto/tabnet/e18_fenccausal.json: model `tabnet`, data_version `crypto/v2.0-fenccausal`, cv_scheme `rolling`, embargo 20, train_window 130, refit_every 130, target `target_binned_return_20`, n_features 44, seed 42, n_iter 1 (pinned), n_splits 2, val_frac 0.3. fixed_params: `{net__batch_size 1024, net__gamma 1.3, net__lambda_sparse 0.001, net__learning_rate 0.02, net__mask_type sparsemax, net__max_epochs 100, net__momentum 0.02, net__n_a 8, net__n_d 8, net__n_independent 2, net__n_shared 2, net__n_steps 3, net__patience 10, net__virtual_batch_size 128}`. Other arms reuse this spec on their own dataset.

## What was found
Default-parameter TabNet is NOT competitive — it loses to the deployed slot on every dataset, significantly (a loss, not a null: |t| 2.0–5.3). Paired per-period payout deltas, NW lag 20 (from EXPERIMENTS.md §E18): raw −0.0287 (t −3.7) vs CURUPIRA_02 ridge, and −0.0257 (t −3.9) vs SACI_04 lgb; tenc −0.0145 (t −2.0) vs CURUPIRA_01 glmnet; interact −0.0324 (t −4.8) vs SACI_01 lgb; fenccausal −0.0428 (t −5.3) vs IA_01 MLP; cumlag3 −0.0255 (t −3.1) vs SACI_05 lgb. The sharpest datum: on fenccausal a plain MLP on the same 44 columns nearly doubles TabNet's CORR (0.162 vs 0.095), so the data is learnable and TabNet's inductive bias is what fails — ~39k rows / ~130 effective periods and a low-SNR target are too little to identify an attention-masked multi-step feature-transformer, and the 22 shipped features are already quantile-binned (the raw-feature selection TabNet exists to learn is already done). The 4 wide arms were not run because no larger column set closes a t −3 to −7 deficit. Board: results/reports/crypto/report_20260829T142026Z.html. (Metric-JSON cross-check: best arm e18_tenc CORR 0.117965 / MMC 0.057877 / n=510; e18_raw 0.109019; e18_fenccausal 0.095366.)
