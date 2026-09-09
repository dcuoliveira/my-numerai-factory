# Experiment index

One row per numbered study. The ID is the reference handle: every artefact a
study writes carries it, so `e01_*` retrieves the whole thing — runs, logs and
report — in one glob, years later.

| ID | Name | Runner | What it asks | Arms | Status |
|---|---|---|---|---|---|
| **E01** | LightGBM feature blocks on the 60-day target | `e01_blocks60_lgb_reg_crypto.sh` | Does any feature block beat raw when the label is `target_binned_return_60` instead of `_20`? Same 23 arms, same pinned estimator, same cadence — the label is the only thing varying. | 23 | done, 23/23, 38 min |
| **E07** | A feedforward net against the deployed ridge, both tasks | `e07_mlp_crypto.sh` | Does changing the FUNCTION CLASS do what eleven feature blocks could not? `mlp` on the raw 22 at the CURUPIRA cadence, as regression against `ridge/w6r6_fixed_reg` (CURUPIRA_02's own backtest run) and as classification against `ridge/w6r6_fixed_clf`. Neither baseline is refitted; the estimator is the only thing varying. | 2 | running |
| **E08** | The MLP on the four one-hot datasets the book runs on | `e08_mlp_onehot_crypto.sh` | Can an MLP use the 740 ticker dummies ridge can't interact and LightGBM discards? Scoped from 12 datasets to 4 (E15 covered the rest); completes MLP coverage of the deployed book. Was blocked on a scaler bug sending rare dummies to z=197. | 8 | running |
| **E09** | Does a torch MLP reproduce the sklearn one? | `e09_torch_mlp_crypto.sh` | `mlp` and `torch_mlp` on raw 22 at the CURUPIRA cadence, sharing one grid (`TorchMLPWrapper` references `MLPWrapper`'s constants, so they cannot drift). A NULL is the success condition — the two already agree at unit level, so this asks whether the agreement survives the walk-forward. | 2 | running |
| **E11** | Does a learned ticker embedding earn its place in the MLP? | `e11_symbol_embedding_crypto.sh` | Two torch MLP arms on raw 22, differing only in whether an `nn.Embedding` over the ticker is present. Built the way the LSTM's is (padding row for unseen codes, index from the training block only), so the two models agree on what a symbol embedding is. | 2 | running |
| **E12** | How wide should the ticker embedding be? | `e12_embed_dim_crypto.sh` | Six arms — no embedding, then 4/8/16/32/64 — each width pinned via `--model-options` while everything else is still searched. Gives a curve and five paired deltas where E11's single search gives one winner and no interval. | 6 | running |
| **E13** | Do autoregressive and distributed lags help the MLP? | `e13_adl_mlp_crypto.sh` | Four arms on one pinned MLP: contemporaneous, +target lags (AR), +feature lags (DL), +both (ADL). Lags of a 20-day forward target must be ≥20 periods or they are lookahead — `build_adl` refuses shorter. | 4 | running |
| **E14** | E13's ADL question with each arm tuned for its own width | `e14_adl_searched_crypto.sh` | E13 pinned a 22-column config onto 91 columns; here raw and adl each run a full 50-draw search. Read together with E13 — the two designs fail in opposite directions. | 2 | running |
| **E15** | The MLP feature table — 7 raw groups + 6 derived blocks, on both backends | `e15_mlp_feature_groups_crypto.sh` | 14 feature sets × sklearn and torch = 28 fully searched arms. Cross-backend agreement is the guard: a block that moves one backend and not the other is search noise. | 28 | running |
| **E16** | Cumulative differences on a lag ladder to one year | `e16_cumdiff_mlp_crypto.sh` | Six rungs (1/5/22/66/132/264), each cumulative over those below, as differences alone and as raw+differences. Target differences anchored at the 20-period horizon — unanchored they correlate +0.28 with the label. | 12 | registered |
| **E17** | MLP hyperparameter sensitivity, after the 'Overhyped?' paper | `e17_hyperparam_sensitivity_crypto.sh` | One pinned walk-forward per cell of a 5×3×4×4 NN factorial (240), then marginal distributions, fANOVA importance, corner cases and an lr/alpha prior. Finds which HPs are worth tuning and which ranges guarantee good/bad OOS. | 240 | running |
| **E28** | One MLP tune per book dataset with the E17 priors (`mlp_ktz`) | `e28_mlp_ktz_tune_crypto.sh` | For each of the 9 datasets the book trains on, one from-scratch search over a grid that encodes E17's priors — dense learning rate, coarse architecture gate, near-pinned alpha. | 9 | running |
| **E29** | One LightGBM tune per book dataset with the E27 priors (`lgb_ktz`) | `e29_lgb_ktz_tune_crypto.sh` | For each of the 9 datasets the book trains on, one from-scratch search over a grid that encodes E27's priors — dense learning rate, then colsample, coarse num_leaves, near-pinned n_estimators (levels enforce lr≥cs>nl≥n_est), with L=lr·n_est≤40 keeping the sweep in E27's good corner. | 9 | running |
| **E30** | Ensemble the lgb_ktz grid instead of searching it | `e30_ktz_gridensemble_lgb_reg_crypto.sh` | Per book dataset, fit all 54 cells of a good-corner grid (nl{2,4,8}×lr{.001,.003,.01}×cs{.05,.10,.20}×ne{2000,4000}, L≤40) as pinned walk-forwards, then rank-average their predictions — "bag over hyperparameters" vs E29's search-and-pick. 486 cells + 9 ensembles. | 9 | done |
| **E31** | Ensemble the mlp_ktz grid instead of searching it | `e31_mlp_ktz_ensemble_crypto.sh` | The MLP counterpart of E30. Per book dataset, fit all 120 cells of the E17-prior grid (20 lr × 3 arch × 2 alpha × identity) and rank-average their predictions — "bag over hyperparameters" vs E28's search-and-pick. A single meta-estimator per dataset (block-level rank-average, validated ≈ per-period run_ensemble); runs at `--n-iter 1 --n-splits 2` since selection is removed. Paired vs the E28 pick per dataset, and against the book (`IA_01`, `CU_SA_IA_BO_01`). | 9 | registered |
| **E33** | MLP sensitivity v2 — width + alpha opened, 3 datasets | `e33_hyperparam_sensitivity_v2_crypto.sh` | E17's KTZ sweep re-run with the first hidden layer UNFROZEN (width 32/64/128/256) and alpha opened to 100 — the two axes IA_01 (best MLP slot, sklearn mlp on fenccausal) selects OUTSIDE E17's grid (width 64–128, alpha 8–10). 12 arch × 2 act × 4 lr × 6 alpha = 576 cells on each of raw(22)/fenccausal(44)/tenc_onehot(763) = 1728 pinned walk-forwards, analysed per dataset. Feeds new priors to the E34 tune and E35 ensemble. | 1728 | done — **alpha 0.1→~10 is the fix, width a red herring**; all 3 datasets agree; IA_01's region now in-box and best (fenc top relu/w128/alpha10 0.0949) |
| **E34** | One MLP tune per book dataset with the E33 priors (`mlp_ktz2`) | `e34_mlp_ktz2_tune_crypto.sh` | E28 redone on the corrected box: `mlp_ktz2` grid (alpha {1,10}, {identity,relu}, lr peak 1e-3, width 32–128, depth 1–2), one from-scratch search per book dataset. Paired vs the E28 old-prior pick and against the book — does the corrected box reach IA_01? | 9 | registered |
| **E35** | Ensemble the mlp_ktz2 (E33-reprior) grid per book dataset | `e35_mlp_ktz2_ensemble_crypto.sh` | E31 redone on the corrected box: fit all 120 `mlp_ktz2` cells and rank-average, vs the E34 search-and-pick over the same grid. Ensemble-vs-pick re-asked on the box that now contains IA_01's region. `--n-iter 1 --n-splits 2`; paired vs E34 and against the book. | 9 | registered |
| **E42** | torch MLP training-loss study — PINNED (learning dynamics) | `e42_torch_mlp_loss_dynamics_crypto.sh` | Does a different training LOSS change what the torch MLP learns and how stably? 6 losses — mse(control), corr, bmc vs the sparse real meta model (benchmark) and vs the full-coverage meta_proxy, sharpe, calmar — on raw+fenccausal, all at ONE pinned config so only the loss varies. New `torch_mlp_loss` wrapper (loss knob via `--model-options`, per-period losses via a trailing period/reference passthrough column, reusing `nn/losses.py`). Evaluated on the board AND on learning dynamics (`loss_dynamics.py`: per-epoch curve, epochs-to-converge, pred_std collapse, train/val gap). bmc-bench vs bmc-proxy = "optimise on MMC-only vs estimate with BMC". | 12 | registered |
| **E43** | torch MLP training-loss study — SEARCHED (best payout) | `e43_torch_mlp_loss_search_crypto.sh` | E42's companion: same 6 loss arms × raw/fenccausal, but each loss runs a full search and finds its own config — best-achievable-per-loss, where E42 pins the config to isolate the loss's effect on learning. Paired NW vs the mse control. | 12 | registered |
| **E32** | Ensemble the catboost_ktz grid vs searching it | `e32_catboost_ktz_ensemble_crypto.sh` | The CatBoost counterpart of E30/E31. Per book dataset, two arms over ONE 12-cell CatBoost KTZ grid (depth{2,4} × lr{.005,.01,.02} × rsm{.1,.2}; iterations 2000, l2_leaf_reg 10, min_data_in_leaf 100 pinned — the E27 booster priors on CatBoost's names, compact so the ensemble can fit ALL of it): `catboost_ktz_ens` rank-averages all 12 (one deployable `CatBoostGridEnsemble`, `--n-splits 2`) vs `catboost_ktz` which searches the same 12 and keeps the CV-best (`--n-iter 12`, exhaustive). Both wrappers read the grid from `CatBoostGridEnsemble`, so ensemble-of-12 vs pick-1-of-12 is exact. Headline is the paired ENS − SEL delta (NW lag 20); each ENS arm is also ranked against its deployed slot. Prior: averaging buys risk (sharpe, drawdown) over level. | 18 | registered |
| **E36** | FARM over the full (model × feature-group) grid | `e30_gridfarm_crypto.sh` | FARM (equal, farm1/2/3 × {ridge,l2boost,postalasso}, gr, ds) over the best searched run per (family × 12 non-onehot feature-groups) = 36 base models + the 3 E29 GRR arms = 39 — the paper's high-dimensional regime, testing whether any fitted combiner beats 1/N over the FULL grid (E27's FARM used only the 7 top-level blocks). Runs carry the `e30_farm_` prefix. Was numbered E30 in `experiment_db` until 2026-09-05; renumbered to free E30 for the lgb grid-ensemble it collided with (E34/E35 reserved by E33). Findings not yet written. | 12 | done |
| **E37** | Cumulative per-symbol differences to a year, on ridge_glmnet | `e17_cumdiff_glmnet_crypto.sh` | E16's question asked of the linear model, and asked as a BASIS question. Ladder 1/2/3/5/10/21/63/130/261 (261 dates = one calendar year here, measured), cumulative, alone and beside the raw 22. span{f, f−f₋ₖ} == span{f, f₋ₖ}, so an unpenalised fit could not tell differences from lags — ridge can, because ‖b‖² does not commute with a change of basis. Levels 3/6/9 carry a featdiff/lag pair at identical span to isolate that. Caveat: an order-261 difference is real for only 61% of rows. Was numbered E17 in `experiment_db` until 2026-09-05, when a second `"E17"` dict key (the MLP hyperparameter-sensitivity study, which keeps E17) silently overwrote it and hid this study from the tracker; renumbered to E37 (E34/E35 reserved by E33, E36 the renamed FARM grid), and the MLP study narrowed to the `e17_a32_` prefix so `study_of()` routes both cleanly. | 26 | registered |
| **E18** | TabNet at library defaults, on every submitted slot's dataset | `e18_tabnet_crypto.sh` | First deep-tabular architecture on the board. `tabnet` (dreamquark-ai pytorch-tabnet, added to `prediction/models.py` as an sklearn-contract wrapper — its fitted model survives joblib, so it is deployable through the normal refit path, unlike the LSTM). One arm per DISTINCT training dataset a submitted slot fits on: raw, tenc, interact, fenccausal, cumlag3, tenc_onehot, allfeat, onehot_roll, allfeat_pp (9). Parameters PINNED to library defaults (n_d=n_a=8, n_steps 3, gamma 1.3, lambda_sparse 1e-3, lr 0.02, 100 epochs) — a viability probe, not a tuned model, on a panel small and noisy for a net this expressive. `n_splits=2` because with fixed params the inner CV only makes an unused score (per-period result byte-identical to 5, verified); no `eval_set`, so the fit is leak-free. | 9 | 5 narrow done, 0 failed; **default TabNet HURTS on every dataset** (see findings). 4 wide arms NOT run — the viability gate closed. |
| **E10** | Walk-forward stacking: a fitted meta-learner over member predictions | `e10_stack_crypto.sh` | Every blend on the board is a FIXED rank average. Does a meta-learner FITTED on the members' out-of-sample predictions beat one? `ridge` (signed linear) and `mlp` (non-linear) as level-1 over three member sets — ridge-only, MLP-only, and both — against two controls on the identical members and dates: `run_ensemble` 1/N and `run_weighted_ensemble --scheme stacking_nnls` (constrained NNLS). Each meta is tuned under BOTH inner CVs — the book's purged chronological split and Wolpert's original shuffled **k-fold** (`_kf` arms) — which is legitimate because the level-1 features are already out-of-sample; the paired `_kf`-vs-purged delta prices the purge. Scored on the ~150-date tail all arms share. | 22 | running (within-class `stackR`+`stackM`, purged done 12/16; kfold `_kf` arms launching; `stackX` not run) → **done, 16/22**: within-class purged + kfold complete; `stackX` (6 arms) not run |
| **E13** | Autoregressive distributed lags against the best raw ridge | `e13_adl_crypto.sh` | Does the target's OWN past add anything, at a legitimate >=20-period remove? The DL half is already null across 30 arms; the AR half is new and is the half that can leak. 8 specifications x 2 ridge estimators, controls `ridge/w6r6_fixed_reg` and `ridge_glmnet/w6r6_glmnet_reg`. Carries a deliberate LOOKAHEAD control (`leak`, y_{t-1}, corr 0.839 with the label) so the size of the trap is a measurement, not a hazard. | 16 | running |
| **E20** | Exponential recency weighting on every submitted model's dataset | `e20_expdecay_genridge_crypto.sh` | Does weighting recent dates more help? Generalised Ridge with W = exponential decay against W = I, two arms on each of the 9 datasets behind a single-model slot; structure pinned and the same alpha grid both sides, so W is the only difference. half_life searched in DATES, floored at 32 — a 21-date half-life leaves 2.9 effective independent windows against 6.5 uniform. | 18 | registered |
| **E21** | Cluster regularisation on allfeat_pp — shrink some blocks harder than others | `e21_cluster_reg_crypto.sh` | Every slot shrinks all 812 columns with one alpha; this gives Δ's diagonal a per-block multiplier. Three mechanisms: (a) penalise derived blocks more, (b) penalise raw LESS — which changes the λ→∞ limit from a constant to the fit on raw alone, (c) fuse within clusters instead of shrinking them. Blocks partition exactly: raw 22, spread 11, interact 14, rollsel 24, tenc 1, onehot 740. | 11 | registered |
| **E22** | The one-hot block — the penalty, the key, the granularity | `e22_onehot_reg_crypto.sh` | Three questions E21 left open, one change at a time: penalise the dummies 10×, once, or not at all (unpenalised = symbol fixed effects); key on `symbol` or on `ucid` (30 tickers are reused by different coins, 3 were renamed — 2.48% of rows, and nothing in src/ had ever used ucid); and 746 identities vs three low/mid/large cap buckets. Plus arms where the one-hot replaces the raw 22. | 13 | registered |
| **E23** | Cross-class 1/N blends: does diversity buy what fitting could not? | `e23_diverse_blend_crypto.sh` | E10 showed a FITTED meta beats nothing when members are 0.86 correlated. The tabular book is one block (ridge/mlp/lgb/glmnet 0.72–0.92); the torch LSTM is the only orthogonal member (0.35–0.41). With **no fitting at all**, how much does a plain 1/N rank average gain from adding a decorrelated member? `tab2`/`tab4` controls omit the LSTM to separate diversity from member count. **Not deployable**: any arm containing the LSTM cannot run live. | 4 | done, 4/4, seconds — **NULL**: the orthogonal member adds nothing (see findings) |
| **E24** | Can TabPFN be pushed past its E19 default? Three Tier-1 levers | `e24_tabpfn_tuning_crypto.sh` | E19 ran TabPFN handicapped (context 4096, n_estimators 4). Three one-change arms vs that baseline: (1) FULL STRENGTH ctx 8192 / n_est `auto` (memory_saving_mode auto fits 8 GB); (2) CLASSIFIER HEAD — `tabpfn_clf`, TabPFNClassifier on the 5 target bins returning E[y\|x], same ctx/estimators so only the head differs; (3) TEMPERATURE swept {0.5,0.75,1.25} on fenccausal. GPU only (deepfive RTX 2070). | 11 | done, 11/11, 299 min — full strength ~NULL; classifier head **+0.012 t+3.03 on raw only** (hurts fenccausal); temperature ~NULL. Strengthened TabPFN moves the blend swap from losing (t−2.67) to **tied** (t−1.89/−0.80) but still does NOT beat CU_SA_IA_BO_01 (see findings) |
| **E38** | TimesFM zero-shot univariate forecast of each symbol's target | `src/analysis/timesfm_zeroshot.py` | First TIME-SERIES foundation model, and first univariate/per-symbol arm. The ORIGINAL TimesFM 1.0 (200M, Das et al. 2023), loaded from `google/timesfm-1.0-200m-pytorch` via the v1.2.6 torch backend (no JAX; isolated `/scratch` venv reusing base CUDA torch), zero-shot. One symbol at a time, forecast that coin's own `target_binned_return_20` history; forecast = signal. Leak wall at the 20-period embargo (context only dates ≤ t−20 on the global grid), horizon 20, signal = mean of the horizon. Bespoke inference (TimesFM is a sequence forecaster, outside the row-wise MODELS pipeline); scored by `timesfm_score.py` on tabpfn/e19_fenccausal's 510 dates. | 1 | done, 139,684 forecasts — **NULL**: CORR +0.0119 (t +0.82 vs zero), MMC +0.0125 (t +1.06), sharpe 0.12; positive point estimate, indistinguishable from zero, dead last vs the book (see findings) |
| **E39** | The tabular reframing of E38: TabPFN zero-shot on ALL features | `run_forecast (tabpfn on crypto/v2.0-tenc-onehot-rollsel-spread-interact)` | E38's direct counterpart. Same problem, same 510-date window, but reframed from a univariate time series to a TABULAR one and given to TabPFN instead of TimesFM, zero-shot: each row is its full feature vector, and the model predicts the target in-context from past resolved rows. ALL 812 features (allfeat_pp = raw+tenc+onehot+rollsel+spread+interact), context 2048 (vs E19's memory-capped 1024). Isolates the FRAMING. | 1 | done, 20 min — **the reframing wins decisively**: CORR 0.1345 (vs E38's 0.0119), payout 0.0718, c.sharpe 1.07; paired vs E38 CORR **+0.123 (t +6.23)**, MMC +0.046 (t +3.36), payout +0.058 (t +3.77). Context 2048 also beat E19's 1024 (0.1345 vs 0.1238). Edge is CROSS-SECTIONAL, not autoregressive. Still below TabPFN's best narrow arm (fenccausal ~0.147) — 812 feats is past its ~500 design range. |
| **E40** | TEMPO zero-shot univariate forecast of each symbol's target | `src/analysis/tempo_zeroshot.py` | The controlled model-swap sibling of E38: same per-symbol, univariate, leak-free zero-shot design, TEMPO in place of TimesFM, so E38-vs-E40 isolates the MODEL (not the framing). TEMPO (Cao et al., ICLR 2024, arXiv:2310.04948) — GPT-2-backbone ~80M, STL trend/seasonal/residual + RevIN + soft prompt pool, all inside `predict()`. Pretrained zero-shot weights Melady/TEMPO TEMPO-80M_v1.pth (HF), via pip `timeagi` --no-deps in a /scratch venv reusing base torch 2.7 (dodges its torch==1.13 pin; smoke-tested clean on torch 2.x / transformers 5.x). Context = coin's own target_binned_return_20 ≤ t−20, seq_len 336, horizon 20, signal = mean. Scored by `timesfm_score.py --model tempo` on tabpfn/e19_fenccausal's 510 dates. | 1 | done, 139,684 forecasts, ~31 min — **NULL, if anything slightly NEGATIVE**: CORR −0.0171 (t −1.58 vs zero), MMC −0.0027, payout −0.0044, sharpe −0.20; dead last vs the book (dpay −0.096, t −6.19). Paired vs E38 (TimesFM) CORR **−0.029 (t −2.48)** — TEMPO is significantly WORSE than TimesFM. Two different TS foundation models both ≈0 confirms the FAILURE IS THE FRAMING, not the model: univariate autoregressive extraction gets nothing where the tabular/cross-sectional framing (E39 TabPFN) got CORR 0.134. |
| **E42** | Cross-sectional rank-based dynamic stake allocator (NasdaqJockey, Out of Sample Ep 5) | `src/analysis/xs_stake.py` | Type-P (procedure). Does a time-varying CROSS-SECTIONAL rank allocator over the full 24-slot stable beat the deployed static 3-slot mv book? Rank each slot cross-sectionally on {level, recent, sharpe[, momentum-'future']}, build the composite from the RANKS, stake the top-K. Arms: B0 static-mv (control), B1 equal, B2 single-best, B3 rolling-mv, A1 rank, A2 rank+future, A3 hybrid (rank-SELECT then mv-SIZE). Walk-forward lag-20, select-on-train / report-on-held-out, equal-capital (weights sum to 1). Judged by the classic pipeline's stats.py: bootstrap ΔSharpe (block 20, 5000 draws) of the held-out payout path NET of turnover, + stats.decide. Panel is SIMULATED pseudo-OOS (no payout factor / no clipping) -> ratios, not NMR. | 7 | done -- **INCONCLUSIVE / NULL** (see findings): no arm beats B0 with a CI clear of 0; the rank arms raise drawdown without adding Sharpe. |
| **E41** | Fine-tuned TEMPO — does adapting the weights lift E40's null? | `src/analysis/tempo_finetune.py` | E40 (TEMPO zero-shot) scored CORR −0.017, null; E41 fine-tunes TEMPO and re-evaluates on the SAME 510 dates so E40↔E41 isolates fine-tuning. Parameter-efficient (GPT-2 frozen; LoRA + heads + RevIN affine train), warm-started from Melady/TEMPO and fine-tuned in-memory (never reconstructed — the .pth loads strict=False). Training windows: (336→20) slices over each symbol's resolved target series, horizon-end ≤ leak cutoff 2024-06-27; loss = MSE on the first 20 output steps. A **2×2 matrix**: scheme {single global, walk-forward refit-every 130} × selection {validation MSE, validation cross-sectional CORR}. Eval half reuses E40's `build_tasks`/`forecast_signal`; scored by `timesfm_score.py --model tempo`. Prior: fine-tuning can't reach the cross-sectional edge (E39), so a null across arms is the likely, still-informative outcome. | 4 | done, 4/4 — **NULL across the whole matrix**: every arm stays negative (CORR −0.011 to −0.015 vs E40's −0.017), negative c.sharpe, dead last vs the book (dpay ≈ −0.10, t ≈ −7). Paired vs E40 every delta is positive but **none significant**: sg_mse +0.0032 (t +0.56), sg_corr +0.0021 (t +0.36), wf_mse +0.0037 (t +0.74), wf_corr +0.0057 (t +1.02). Faint consistent ordering (walk > single; CORR-select helps in the walk cell) but the best arm (walk+CORR) reaches only CORR −0.0114 (t −1.24 vs zero, still negative). **Fine-tuning does not help — the weights aren't the problem, the univariate framing is** (E39 tabular got 0.134 on the same dates). Confirms E38/E40. |
| **E44** | Fine-tuned TabPFN on fenccausal — can adapting the weights beat E19? | `src/analysis/tabpfn_finetune.py` | The tabular counterpart of E41, on the framing that WORKS. E19 ran TabPFN zero-shot (frozen, in-context) on crypto/v2.0-fenccausal (44 feat) → CORR 0.145, the strongest foundation-model arm on the board. E44 fine-tunes the TabPFN v2 transformer (tabpfn 8.5.0 finetuning API) on the 345,496 fenccausal rows on/before the leak cutoff 2024-06-27, saves the adapted checkpoint, then re-runs the EXACT E19 walk-forward (rolling 130/130, embargo 20, seed 42, n_est 4, ctx 4096) with `model_path` → the fine-tuned .ckpt, so E19↔E44 isolates fine-tuning. Unlike E41 (univariate, fine-tuning couldn't reach the cross-sectional edge), here the framing already works, so a genuine lift over 0.145 is plausible. Scored on tabpfn/e19_fenccausal's 510 dates; paired NW vs E19. | 1 | done, ~42min FT + 12min eval — **fine-tuning HURT, significantly**: CORR 0.14504 → **0.11788**, payout 0.08124 → 0.06801; paired vs E19 CORR **−0.0272 (t −3.02)**, MMC −0.0105 (t −1.45), payout −0.0132 (t −1.63). The opposite of the hope. TabPFN's edge IS its meta-learned in-context prior (trained across millions of synthetic tasks); fine-tuning ~11M params on one noisy, non-stationary crypto target erodes that prior — overfitting to spurious structure and, because the FT loss is CRPS+MSE, compressing the cross-sectional spread the ranking pays for (the E27 MSE-vs-CORR decoupling again). So across BOTH framings fine-tuning fails: univariate (E41) couldn't reach the edge, tabular (E44) actively damages the pretrained prior. **Zero-shot TabPFN (E19) is not just good — it beats fine-tuning it.** |
| **E46** | FARM over the from-scratch ktz SEARCH-PICK models (lgb E29 + catboost E32-sel + mlp E34) vs 1/N | `e46_ktz_searchpick_farm_crypto.sh` | Combine the 27 best-tuned (model × dataset) forecasters — the config each family's from-scratch KTZ-prior search PICKED per dataset — with the Lee–Lee FARM combiner, vs the plain 1/N of the same 27. (Earlier plan FARMed the raw HP grid *cells*, but E30/E32 reclaim those and the surviving mlp E33 sweep is junk-heavy; the search-*pick* models are the clean pool, all already on disk.) POOL: `lgb_ktz/e29_<ds>` (60-draw search) + `catboost_ktz/e32_sel_<ds>` (12-cell exhaustive) + `mlp_ktz2/e34_<ds>` (80-draw search on the E33-reprior wider grid), 9 datasets = 27 base forecasters. FARM (equal=1/N baseline, farm1/2/3 × {ridge,l2boost,postalasso}, gr, ds) over four pools: per-family (lgb 9, cat 9, mlp 9) and pooled (all 27). Paired NW lag 20 vs each pool's own 1/N; best arm ranked against the book. Prior (E27/E29/E36): 1/N wins on this panel. | 48 | done — **NULL, 1/N wins every pool**: no FARM arm beats the equal rank-average; `l2boost` ties it (mlp `farm1_l2boost` +0.0009, t+0.40 the only arm above 1/N anywhere), while the `ridge`/`postalasso`/`gr`/`ds` arms OVERFIT the deviation regression on ~26 windows and lose up to t−2.3 (lgb). The only real signal is that pooling all 27 lifts the SIMPLE average (all-1/N payout 0.0911 > lgb 0.0898 / cat 0.0906 / mlp 0.0864) — diversity helps via 1/N, not FARM. Vs the book (common 457): pooled 1/N 0.0928 (t+0.74 vs SACI_06) ≈ best single submitted, but BELOW every deployed slotblend (0.099–0.105); FARM's best arm is slightly worse (0.0911). **But SUBSET SELECTION does what reweighting can't**: a curated 4-member diverse blend `e46_blend_cf_ca_mt_mi` (catboost fenc+allfeat + mlp tenc+interact, NO lgb) priced through `run_ensemble` scores payout **0.0980, c.sharpe 1.94, burn(d) 37** — beats the full-27 1/N (+0.006), the best single (+0.005, t+1.68 vs SACI_06) and ~half the deployed slots (CURUPIRA_08, SACI_02/01, CURUPIRA_07, IAUARETE_01), ≈ CURUPIRA_SACI_03; still below the top tier (CU_SA_IA_BO_01/BOITATA/CAIPORA 0.102–0.105). The payout-SERIES screen HID this (said no subset beats its best member) — the real per-period rank-average exploits cross-sectional diversity the series-average can't see. FARM/peLASSO missed it because they reweight the full pool rather than DROP members. |
| **E25** | Weighted and selection combiners over a diverse pool | `e25_weighted_combiners_crypto.sh` | E10 showed fitting beats nothing when members are homogeneous; E23 showed a fixed 1/N cannot exploit a weak-but-orthogonal member because it cannot downweight anything. This is the untested cell: 8 walk-forward weighting/selection schemes (`equal`, `inverse_variance`, `complete_subset`, `caruana`, `bagged_caruana`, `shrunk_tangency`, `granger_ramanathan`, `stacking_nnls`) over the same 5-member pool E23 blended, each paired against E23's 1/N on identical members and dates. A tabular-only pool repeats every scheme WITHOUT the LSTM, so any gain can be attributed to handling the orthogonal member rather than to weighting per se. 16 tests → read |t| ≳ 2.9, not 2. | 16 | done, 16/16, 20 min — **NULL, and one-directional**: all 14 schemes LOSE to 1/N (see findings) |
| **E26** | Sign-constrained GRR on allfeat_pp — all betas ≥ 0, all betas ≤ 0 | `e26_sign_constraint_crypto.sh` | The one GRR knob that shrinks the feasible set rather than the penalty: a box-constrained QP by coordinate descent, optimality on KKT. Both constraints bind (139–217 of 394 live coords), measured before launch, so neither arm is a no-op. Alpha floored at 1e-2 in all three arms — a constrained solve at the 1e-4 default cannot reach optimality. | 3 | registered |
| **E27** | LightGBM hyperparameter sensitivity (Kalfa-Timmermann-van der Zwan) | `e27_hyperparam_sensitivity_crypto.sh` | Which hyperparameters actually move payout, which corner values lock in good/bad models, and what range is a sound prior? A 315-cell factorial (num_leaves x learning_rate x n_estimators x colsample) marginalised the paper's way. | 315 | running |
| **E47** | Deploy the E33 winner: pinned MLP `IA_02` + two blends | `e47_deploy_mlp_blends_crypto.sh` | Not a sweep — a DEPLOYMENT. Wire E33's top fenccausal cell (128,64/relu/lr1e-3/**α10**) live as the pinned MLP component `IA_02` (`NOT_SUBMITTED`), then build and **price** (real per-period rank-average, not the screened proxy) two blends on it: `CU_SA_IA_BO_02` = `IA_02`+`CURUPIRA_05_60`+`CU_SA_IA_BO_01` (a version bump — same Curupira+Saci+Iara footprint, 2nd neural leg) and `CU_CA_IA_01` = `IA_02`+`IAUARETE_01`+`CURUPIRA_05_60` (Curupira+Caipora+Iara). Does a 2nd, differently-regularised neural draw improve the book? | 2 | done — **deployed**; both slots on the account, priced (see findings), submit at the next open round |
| **E48** | Random-forest hyperparameter sensitivity (Kalfa-Timmermann-van der Zwan) | `e48_rf_hyperparam_sensitivity_crypto.sh` | The 'Overhyped?' sweep E17 ran on the MLP and E27 on LightGBM, now on a BAGGED FOREST — the first random forest the book has tried. One pinned walk-forward per cell of a 3×4×4×4 factorial: `n_estimators` {100,300,1000} × `max_depth` {3,6,12,None} × `max_features` {0.1,0.3,0.6,1.0} × `min_samples_leaf` {1,10,100,1000} = 192 cells on the raw 22 of crypto/v2.0, then marginals, fANOVA importance, corners and priors via `src.analysis.hyperparam_sensitivity_rf`. A forest is bagging not boosting, so there is **no learning-rate axis**: `max_features` carries the Numerai colsample idiom (the most distinctive setting), `min_samples_leaf` the heavy-leaf idiom, and the two-way claim tested is `max_features × min_samples_leaf` (the forest analogue of the paper's lr × alpha). `bootstrap`/`max_samples` pinned at the forest default. Maps the surface once so a future `rf_ktz` grid can be set from evidence. | 192 | running |

## What each ID owns

```
results/{metrics,forecast,per_period,hyperparams}/crypto/<model>/e01_*
logs/e01_*.log            one per arm
logs/e01_report.txt       the final table
experiments/e01_*.sh      the runner
e01.txt                   console log of the run
```

A study that scouts first owns a second prefix. E07's `--era-step 4` wiring runs
are `e07scout_*`, deliberately NOT `e07_*`: a downsampled arm moves the split
boundary, so it must never be globbed up alongside the full-resolution ones and
reported against the same baseline.

## Naming rule

`E<NN>` plus a one-line description, recorded here before the study launches.

An ID beats a descriptive prefix because a description stops being unique the
moment a second study shares it — `b60_lgb_reg_*` names a configuration, and the
next 60-day LightGBM study would match the same glob. It also beats a codename:
`E01` sorts, increments, and tells you the order things were run in.

Run names inside a study stay descriptive: `e01_<arm>_m<leaf>` reads as
"study E01, this arm, this leaf size".

## E18 — findings

**Default-parameter TabNet is not competitive on this panel — it loses to the
deployed slot on every dataset it was run on, significantly.** Five narrow arms
(raw, tenc, interact, fenccausal, cumlag3), library defaults pinned (n_d=n_a=8,
n_steps 3, gamma 1.3, lambda_sparse 1e-3, lr 0.02, 100 epochs), scored on the
510-period tail — the same basis as the deployed slots. Paired against the slot
trained on the identical dataset, per-period payout, Newey-West lag 20:

| dataset | TabNet | deployed slot | Δ payout | t |
|---|---|---|---|---|
| raw | 0.0556 | CURUPIRA_02 ridge 0.0842 | −0.0287 | −3.7 |
| raw | 0.0556 | SACI_04 lgb 0.0813 | −0.0257 | −3.9 |
| tenc | 0.0697 | CURUPIRA_01 glmnet 0.0841 | −0.0145 | −2.0 |
| interact | 0.0566 | SACI_01 lgb 0.0890 | −0.0324 | −4.8 |
| fenccausal | 0.0558 | IA_01 **MLP** 0.0987 | −0.0428 | −5.3 |
| cumlag3 | 0.0512 | SACI_05 lgb 0.0767 | −0.0255 | −3.1 |

This is a significant LOSS, not a null: |t| 2.0–5.3, well outside the ~26
effective-window floor. The sharpest datum is `fenccausal`, where a plain MLP on
the same 44 columns nearly DOUBLES TabNet's CORR (0.162 vs 0.095) — so the data
is learnable; TabNet's specific inductive bias is what fails. The mechanism:
~39k rows / ~130 effective periods and a low-SNR target are far too little to
identify an attention-masked, multi-step feature-transformer, and the 22 shipped
features are already quantile-binned — the raw-feature selection TabNet exists to
learn has been done by the feature engineering. The 4 wide arms (763–812 feat)
were NOT run: at defaults TabNet loses by t −3 to −7 on the narrow sets, and no
larger column set closes that. If pursued, the only lever is a lambda_sparse/lr
tuning study (the grid is in `TabNetWrapper`), but a 0.03–0.04 payout deficit is
a large hole for hyperparameters alone. Board:
`results/reports/crypto/report_20260829T142026Z.html`.

## E10 — findings

**Fitting the combination bought nothing.** Four meta-learners over two
within-class member sets, each paired on the 151 dates all arms share,
Newey-West lag 20. Payout deltas:

| meta | set | vs 1/N | vs NNLS | vs its best member |
|---|---|---:|---:|---:|
| ridge | stackR | −0.001714 (t −1.12) | +0.000064 (t +0.28) | −0.004603 (t −1.26) |
| mlp | stackR | −0.013486 (t −2.00) ☠ | −0.011708 (t −1.86) ☠ | −0.016375 (t −2.07) ☠ |
| ridge | stackM | −0.001511 (t −0.55) | +0.000155 (t +0.77) | −0.002425 (t −0.42) |
| mlp | stackM | −0.000062 (t −0.13) | +0.001604 (t +0.65) | −0.000977 (t −0.27) |

1. **Against a fixed 1/N rank average, every meta is negative** and no healthy
   arm reaches |t| = 2. The fixed averages already on the board are not leaving
   money on the table.
2. **Against constrained NNLS the deltas are ~zero** (+0.000064 to +0.001604,
   |t| ≤ 0.77). Signed coefficients and an intercept buy nothing over
   non-negative weights, so the constraint was not costing anything. Table 2
   was the real bar and it is a clean null.
3. **No stack beats the better of its own two members.** Each lands between its
   members — the signature of a combiner that is merely averaging. That is the
   question a combiner has to answer before the others matter.
4. A borderline `corr`-only pair (`meta_ridge_stackM` vs NNLS, t +1.98; vs
   `mlp_fenc`, t +2.03) does not survive Bonferroni (~2.6 for four meta arms)
   and does not appear in payout (t +0.77). Not a result.

**The MLP meta silently collapsed, and that is the more useful finding.**
`e10_meta_mlp_stackR` produced ONE CONSTANT value for every symbol on 44 of its
151 dates — refit blocks 2 and 4. It did not crash; it wrote a well-formed
forecast. Three details:

* Both dead blocks chose the *identical* hyperparameters — `(64,32)`,
  `alpha 19.41`, `lr 0.0055`. Their **CV scores looked normal** (0.1489,
  0.1716), so the search could not see it: the net died when refit on the full
  block, not in the folds.
* The alpha grid is data-scaled and its ceiling differed — **100.0 on stackR
  against 10.0 on stackM** — so only stackR's search could reach the lethal
  region. stackM picked `(64,32)` too, at alpha 8.2, and lived.
* It is a NARROW-INPUT failure, not general MLP fragility: the base MLP arms on
  22 and 44 features had zero dead blocks. A high L2 penalty can zero out a
  2-column input in a way it cannot zero out 44. Worth a look before trusting a
  future MLP meta, but it does not implicate `IA_01`.

**Two reporting traps this study exposed**, both of which turn the failure into
a merely-mediocre number:

* `summarize` drops the undefined periods, so the metrics json reports
  `n_periods=107` for an arm scored on 151. Building payout as
  `0.1*corr["mean"] + mmc["mean"]` then adds means over DIFFERENT denominators.
  Scoring a dead period as the 0 it earned gives payout 0.023459; dropping it
  gives 0.033105, flattering the arm by 41%.
* Ranking `pay` down a table that mixes a 151-date arm with a 510-date one
  ranks the WINDOW. The recent tail is far weaker for everything: 1/N pays
  0.0870 over its own 503 dates and **0.0369** over the meta's 151. Always
  carry a common-window column.

Caveat on scope: `stackX` (cross-class) was not run, so this says nothing about
whether stacking helps when members come from different function classes —
which is the case the CURUPIRA slot blends actually exercise.

### E10 — Wolpert's k-fold inner CV (follow-up)

The meta-learners were re-tuned under Wolpert's ORIGINAL inner CV — shuffled
random k-fold instead of the purged chronological split — with the OUTER
walk-forward untouched, so both stay pseudo-OOS (`--inner-cv kfold`, `_kf` arms).
Paired `_kf`-vs-purged, same arm, same 151 dates, lag 20:

| arm | payout Δ | t | dead (kf/purged) |
|---|---:|---:|---|
| ridge stackR | −0.000011 | −0.72 | 0 / 0 |
| ridge stackM | −0.000043 | −1.55 | 0 / 0 |
| mlp stackM | −0.002497 | −0.46 | 0 / 0 |
| mlp stackR | +0.009928 | +1.46 | **0 / 44** |

1. **The inner CV changes nothing where both fits are healthy.** Three of four
   arms are null (|t| ≤ 1.55); the purge was buying nothing on the meta level,
   which makes sense — the level-1 features are already out-of-sample, so there
   is little forward-target overlap left for k-fold to exploit.
2. **The one large delta is a repaired failure, not better tuning.** The purged
   `mlp stackR` collapsed to a constant on 44/151 periods (a degenerate
   hyperparameter config its CV could not see — the fold scores looked normal).
   K-fold happened to pick a live config, so `_kf` scores +0.0099. In the clean
   case where BOTH fits are healthy (`mlp stackM`) k-fold is if anything slightly
   WORSE (−0.46). So k-fold is not a better tuner; it dodged one degenerate draw.
3. **The headline conclusion is unchanged.** Even healthy, the k-fold MLP meta
   still loses to both controls (vs 1/N −0.0036 t −0.76; vs NNLS −0.0018 t −0.45).
   Under either inner CV, a fitted meta does not beat 1/N or NNLS.
4. **Cost note.** These four arms ran 35h wall-clock against a box at load 65–100
   (other studies); the ridge metas are ~65 min uncontended.

## E23 — findings

**A null, and it refutes the diversity hypothesis that motivated it.**
Four fixed 1/N rank averages, every figure on the 503 dates all rows share,
paired Newey-West lag 20.

| row | members | payout |
|---|---|---:|
| `all5` | ridge+mlp+lgb+glmnet+lstm | 0.087152 |
| `tab2` | glmnet+lgb | 0.086336 |
| `tab4` | ridge+mlp+lgb+glmnet | 0.086319 |
| `div3` | glmnet+lgb+lstm | 0.085393 |
| ridge | best single member | 0.083482 |
| lstm | weakest member | 0.060720 |

1. **The level gain over the best member is real but NOT established.**
   `all5` − `ridge` = **+0.003669 payout (+4.4%), t +0.47**. The point estimate
   reproduces exactly, but at 503 dates / ~26 effective windows the standard
   error is 0.0078. No arm in table 1 clears |t| = 2 (best is `tab4`, t +0.53).
2. **The orthogonal member contributes NOTHING — the mechanism test is null.**
   `div3` − `tab2` = −0.000943 (t −0.19): adding the LSTM to a tabular pair made
   it slightly WORSE. `all5` − `tab4` = +0.000833 (t +0.26). So whatever small
   gain a blend has comes from ordinary variance reduction among the TABULAR
   members, not from the LSTM's 0.35–0.41 decorrelation.
3. **Why: orthogonal but weak is not enough at equal weight.** The LSTM pays
   0.0607 against the best member's 0.0835 — 27% less. At 1/N its level drag
   roughly cancels its decorrelation benefit. Diversity only pays if the weak
   member is downweighted, which 1/N cannot do BY CONSTRUCTION.
4. **Two members is as good as four.** `tab2` (0.086336) ≈ `tab4` (0.086319).
   Piling on correlated tabular members adds nothing, consistent with the
   0.72–0.92 block structure.

**Read together with E10 this sharpens the picture.** E10: fitting doesn't beat
1/N when members are homogeneous. E23: 1/N doesn't capture diversity when members
differ sharply in STRENGTH. The untested cell is a combiner that can downweight a
weak-but-orthogonal member — which is what `run_weighted_ensemble`'s NNLS /
caruana / complete_subset schemes do, and is the natural next probe.

**Deployability:** `div3` and `all5` contain the LSTM and cannot run live
(no torch live path). `tab2`/`tab4` can, and neither beats the book.

## E25 — findings

**Not one weighting or selection scheme beat 1/N. All 14 deltas are negative.**
Every figure on the 503 dates all rows share, each arm paired against the 1/N
blend of the IDENTICAL members, Newey-West lag 20.

| scheme | pool5 Δpayout (t) | tab4 Δpayout (t) | mean w(lstm) |
|---|---:|---:|---:|
| shrunk_tangency | −0.001184 (−0.95) | −0.001168 (−0.72) | 0.191 |
| inverse_variance | −0.002327 (−1.33) | −0.001370 (−1.05) | 0.184 |
| complete_subset | −0.003671 (−1.88) | −0.000497 (−0.58) | 0.270 |
| bagged_caruana | −0.004834 (−1.42) | −0.002646 (−1.47) | 0.173 |
| granger_ramanathan | −0.005163 (−1.37) | −0.000910 (−0.55) | 0.359 |
| caruana | −0.007668 (−1.47) | −0.004803 (−1.15) | 0.208 |
| stacking_nnls | −0.008034 (**−2.01**) | −0.004245 (−1.66) | 0.224 |

1. **The sign is unanimous.** 14 of 14 comparisons negative. Nothing clears the
   16-test Bonferroni bar (|t| ≈ 2.9) in either direction, but the single
   strongest signal in the study is that a fitted combiner (`stacking_nnls`) is
   WORSE than equal weights at t −2.01. This is the classic "1/N is hard to beat"
   result: estimating weights from ~500 noisy overlapping periods costs more in
   estimation error than optimal weights could return.
2. **Wiring is verified.** `equal` reproduces the E23 control to 0.000000 with
   zero variance, so `run_weighted_ensemble` and `run_ensemble` agree bit-for-bit
   and the deltas above are attributable to the scheme alone.
3. **Adding the orthogonal member makes weighting HARDER, not easier.** 6 of 7
   schemes are more negative on pool5 than tab4 (e.g. granger_ramanathan −0.0052
   vs −0.0009; complete_subset −0.0037 vs −0.0005). The fifth member adds
   estimation noise that exceeds what its diversity returns.
4. **The mechanism prediction in E23 was WRONG, and this measures it.** E23 said
   these schemes would help by *downweighting* the weak LSTM. They do the
   opposite: mean weight 0.173–0.359 against 1/N's 0.200, with
   granger_ramanathan at 0.359 and complete_subset at 0.270. The schemes chase
   its decorrelation and overweight it, which costs level because it pays 0.061
   against the best member's 0.083. Suggestively, the least-bad arm
   (shrunk_tangency, −0.0012) is one of the three that slightly downweighted it.

**Taken with E10 and E23 this closes the combiner question.** Three studies, ~30
combiner arms — a fitted meta-learner (E10), a fixed 1/N over a diverse pool
(E23), and eight walk-forward weighting/selection schemes (E25) — and nothing
beats a plain equal-weight rank average. **The combiner is not the bottleneck and
further work on it is not indicated.** The operational corollary is positive: the
hand-set equal weights in the deployed CURUPIRA slot blends are not leaving money
on the table, and every alternative tested here is worse.

**Deployability:** every pool5 arm contains the LSTM and cannot run live.

## E27 — findings

**The paper's method reproduces on the paper's metric and does not transfer to
ours. No FARM arm beats 1/N on `numerai_corr` at |t| ≥ 2 in either pool.** 24
arms, each paired against the equal-weight blend of the IDENTICAL members on the
dates they share, Newey-West lag 20.

| pool | arm | CORR | MMC | payout | Δpayout (t) | ΔCORR (t) |
|---|---|---:|---:|---:|---:|---:|
| decorr5 | **equal** | 0.16257 | 0.07728 | 0.09353 | baseline | baseline |
| decorr5 | farm1_l2boost | 0.15876 | 0.07520 | 0.09108 | −0.00245 (−1.04) | −0.00381 (−1.62) |
| decorr5 | ds | 0.15708 | 0.07425 | 0.08996 | −0.00357 (−1.16) | −0.00549 (−1.76) |
| decorr5 | farm1_ridge | 0.15692 | 0.07418 | 0.08988 | −0.00366 (−1.17) | −0.00565 (−1.77) |
| decorr5 | farm1_postalasso | 0.15665 | 0.07404 | 0.08970 | −0.00383 (−1.19) | −0.00592 (−1.80) |
| strong5 | farm1_l2boost | 0.16638 | **0.08264** | **0.09928** | +0.00314 (+1.32) | +0.00003 (+0.01) |
| strong5 | ds | 0.16287 | 0.08078 | 0.09707 | +0.00093 (+0.27) | −0.00349 (−0.99) |
| strong5 | farm1_ridge | 0.16279 | 0.08079 | 0.09707 | +0.00093 (+0.27) | −0.00357 (−1.02) |
| strong5 | farm1_postalasso | 0.16263 | 0.08055 | 0.09681 | +0.00067 (+0.19) | −0.00373 (−1.05) |
| strong5 | **equal** | 0.16636 | 0.07951 | 0.09614 | baseline | baseline |

1. **On `decorr5` the sign is unanimous and negative** — all 11 FARM/GR/DS arms
   below the simple average on both CORR and payout, best case −1.04 on payout.
   With weak-but-decorrelated members there is nothing for the deviation
   regression to find that 1/N has not already got.
2. **On `strong5` the ONLY gain is MMC, and CORR is a dead tie.**
   `farm1_l2boost` scores CORR 0.16638 against equal's 0.16636 — ΔCORR +0.00003
   at t +0.01, a tie to four decimals — while MMC goes 0.07951 → 0.08264. The
   entire +0.00314 payout gain is contribution over the meta model, not
   correlation, and at t +1.32 it does not clear the house |t| ≥ 2 bar. Drawdown
   is identical (1.68) and p.sharpe barely moves (1.16 → 1.17).
3. **The winner is the estimator that deviates LEAST from 1/N.** L2-Boosting at
   step 0.001 is by construction a heavily-shrunk near-equal solution, and it is
   the best arm in both pools. Ridge and Post-ALasso, which move further, do
   worse. That is the forecast combination puzzle restating itself inside the
   method built to solve it.
4. **FARM2/FARM3 buy nothing over FARM1, and NOT because there are no factors to
   find.** Paired against FARM1 with the same estimator, 10 of 12 comparisons sit
   within ±0.0003 payout, none is significantly positive, and the two largest
   moves are negative (`strong5 farm3_l2boost` −0.00256, t −1.87). The tempting
   explanation — that the CV just picks r = 0 — is **wrong here**: on `strong5`
   FARM3 selects r = 2 at 57 of 60 updates (mean r 1.95) and still lands at or
   below FARM1, and FARM2 on `decorr5` averages r 0.74. The factors are extracted
   and they do not pay. That reproduces the paper's conclusion #3 while
   supporting its stated *reason* — estimating the extra factors and their
   loadings costs more than the signal they carry — rather than the weaker
   reading that no such factors exist.
5. **Why it fails here even though it works there.** FARM minimises SQUARED
   error; the book is paid on cross-sectional rank correlation. Lowering MSE
   pulls the combination toward the conditional mean, which compresses the very
   spread the per-period ranking is scored on. The divergence is explicit in the
   paper's own high-dimensional regime, a 134-member pool: the three FARM1 arms
   cut relative MSFE to 0.836–0.864 at Diebold-Mariano t +6.6 to +9.2, an
   unambiguous win on the paper's loss, while CORR falls from equal's 0.159 to
   0.11–0.14 at paired t −3.1 to −3.8, an unambiguous loss on ours. `POOLS=highN`
   rebuilds that pool and reproduces the CORR/payout half; the relative-MSFE
   figures come from the standalone replication harness, because
   `run_farm_ensemble` deliberately scores through the repo path and does not
   emit the paper's loss. **A method can reproduce faithfully on its own loss and
   still be worthless here** — which is the reason to judge any imported combiner
   on `numerai_corr` and payout before believing its published metric.

### E27 validation — the same estimators reproduce the paper on FRED-MD

A null is only worth trusting if the code is correct, and the crypto book cannot
supply a positive control — it has no structure FARM can exploit, so "FARM does
not help" and "my FARM is broken" look identical. So the estimators were run
UNCHANGED (`src/prediction/farm.py`, imported by `src/prediction/farm_paper.py`)
on the paper's own experiment: FRED-MD macro forecasting in level space, five
targets × three horizons, reproducing Lee & Lee (2025) Table 1. Data vendored at
`data/fredmd/` (the canonical St. Louis Fed host S3-blocks this box; mirror + SHA256
in the README). Runner: `experiments/farm_paper_replication.py`; tests:
`src/tests/test_farm_paper.py` (8, incl. a synthetic positive control where FARM1
beats the average >2× by construction).

**It reproduces.** Across the 165 non-equal cells, the reproduced relative MSFE
correlates **0.934** with the paper's (mean |diff| 0.049; FARM1 cells 0.033). The
`h=1` anchors land almost exactly — FARM1·L2-Boost CPIAUCSL 0.872 (paper 0.883),
PCEPI 0.864 (0.866), UNRATE 0.883 (0.877) — GR blows up to 1.11–2.21 (paper
1.14–2.06), L2-Boost ≤ Ridge, FARM2/3 do not beat FARM1, and relative MSFE decays
with horizon exactly as published. 11 of 13 automated pattern checks pass (the two
misses are `h=2` significance edges, present but just under p<0.10, as expected
from vintage and predictor-selection differences).

**Conclusion:** the estimators are faithful, so the E27 crypto result is a genuine
property of the data and the rank-correlation metric — not an implementation bug.
FARM works exactly where the paper says it does (squared-error loss, N large
relative to T, level space) and not on this book, for the reason stated in
finding #5.

**Taken with E10, E23 and E25 the combiner question stays closed.** Four studies
and 60 finished combiner arms (16 + 4 + 16 + 24) — a fitted meta-learner, a fixed
1/N over a diverse pool, eight weighting/selection schemes, and now the
factor-adjusted family that is allowed to move a rank rather than merely reweight
— and nothing beats a plain equal-weight rank average at the house bar. The one
thing E27 adds that the others could not: it shows the failure is **not lack of
expressiveness**, because FARM demonstrably fits better in its own loss and still
loses on the metric that pays.

**Deployability:** every arm is deployable — all five members have live refit
paths and no LSTM is involved — but none is worth deploying. The hand-set equal
weights in the CURUPIRA slot blends remain correct.

## E01 — findings

Newey-West at **lag 60**, not 20: a 60-day target overlaps ~98% between
consecutive daily periods, so 470 periods are roughly **eight** effective
independent windows. Nulls are the expected outcome.

| arm | leaf | payout | Δ vs raw | t |
|---|---:|---:|---:|---:|
| `fencglobal` ▲ | 313 | 0.162860 | +0.078671 | **+3.65** |
| `allfeatpp` | 50 | 0.093312 | +0.014690 | +1.10 |
| `tenc_onehot` | 50 | 0.093018 | +0.014397 | +1.06 |
| `raw` | 313 | 0.084189 | baseline | — |
| `interact` | 313 | 0.077502 | −0.006687 | **−3.56** |
| `fenccausal` | 313 | 0.064105 | −0.020084 | **−2.24** |

1. **`interact` reverses sign between labels.** +0.005710 (t = +2.99) on the
   20-day target, −0.006687 (t = −3.56) here. Same columns, same pinned model,
   same cadence. Combined with it not reproducing at leaf 50 (+2.99 → +0.36),
   the 20-day result reads as conditional on the label and the leaf size rather
   than a property of the columns. **SACI_01 is deployed on that 20-day result.**
2. **`fencglobal` gets worse, not better.** +2.83 on 20-day, +3.65 here — its
   lookahead strengthens with horizon, exactly as `build_feature_encoding`'s
   docstring predicts: 62% of tokens do not survive three years, and a
   whole-sample profile partly states whether the token survived. Still not
   deployable.
3. **`fenccausal` is consistently harmful** on both labels (−2.15, −2.24).
4. **Every leaf-50 wide arm is positive** (+0.83 to +1.10) and none significant.
5. **The 60-day label is far riskier.** Max drawdowns run 4.7–7.9 against
   0.2–1.2 for the same datasets on the 20-day label.

## E29 — findings

(Filed as E29 — launched as E28, but a concurrent session took that id for an
unrelated `mlp_ktz` study; the runs keep the `e28_` prefix and the DB counter
filters by model tree, so the two do not cross-count. 18/18 arms.)

**Three untuned cells filled from scratch (50-draw purged-CV search, seed 42):**

| new tune | dataset | CORR |
|---|---:|---:|
| `mlp / e28_mlp_onehot` | v2.0-onehot (762) | 0.1554 |
| `ridge_glmnet / e28_glmnet_rollsel` | v2.0-rollsel (46) | 0.1554 |
| `lgb / e28_lgb_onehot` | v2.0-onehot (762) | 0.1479 |

The MLP on the pure one-hot scores **0.1554 — on par with glmnet on rollsel and
with the raw-feature MLPs** — confirming E08's premise that a dense first layer
turns the 740 ticker dummies into an embedding lookup. LightGBM on the same
one-hot lands lower (0.1479), consistent with the book's standing result that the
booster discards the dummies. **Three GRR arms** on v2.0-onehot: `e28_grr_neg`
(β≤0) 0.1387, `e28_grr_pos` (β≥0) 0.1054, `e28_grr_symfe` (onehot block
unpenalised, p1e-6 — symbol fixed effects) 0.0960; the unpenalised-dummy arm is
weakest, as expected when 740 collinear columns fit freely.

**FARM over the 24-model pool (best searched run per family × 7 subsets + 3 GRRs):
1/N wins again. No arm beats equal-weight.** Paired vs the 1/N blend of the
identical members, Newey-West lag 20, 510 periods:

| arm | CORR | payout | Δpayout (t) | ΔCORR (t) |
|---|---:|---:|---:|---:|
| **equal** | 0.1619 | 0.0904 | baseline | baseline |
| farm2·l2boost | 0.1598 | 0.0901 | −0.0003 (−0.10) | −0.0022 (−0.99) |
| farm1·l2boost | 0.1589 | 0.0885 | −0.0018 (−0.75) | −0.0030 (−1.37) |
| farm3·l2boost | 0.1563 | 0.0882 | −0.0022 (−0.69) | −0.0056 (−1.92) |
| farm·ridge (×3) | 0.1510 | 0.0846 | −0.0058 (−1.20) | −0.0109 (−2.03) |
| ds | 0.1494 | 0.0826 | −0.0078 (−1.65) | −0.0125 (−2.35) |
| postalasso (×3) / gr | 0.143 | 0.079 | −0.011 (−1.5) | −0.019 (−2.2 … −2.3) |

1. **The 24-member per-subset + sign-constrained-GRR pool is the paper's
   high-dimensional regime (N large vs T), and still nothing beats 1/N.** The
   best arm (farm2·l2boost) is a statistical tie-to-worse (ΔCORR −0.0022, t
   −0.99); every ridge/postalasso/GR/DS arm is significantly WORSE (ΔCORR t −2.0
   to −2.4). L2-Boost — the estimator that shrinks hardest toward equal weights —
   is again the least-bad, exactly as in E27 and the paper.
2. **This closes the combiner question at scale.** E10/E23/E25/E27 established it
   on 5–37 members; E29 confirms it on a genuinely diverse 24-member pool spanning
   3 families × 7 feature subsets plus 3 GRR structures. With the FRED-MD
   validation (the same estimators DO win on the paper's MSE loss), the reading is
   settled: **FARM's failure here is the metric, not the pool size or the
   implementation** — minimising squared error cannot buy rank correlation.
3. **Operational corollary unchanged:** the hand-set equal weights in the deployed
   CURUPIRA slot blends remain correct; no fitted combiner tried across five
   studies improves on them.

## E30 — findings

**Bagging the whole good-corner lgb_ktz grid is statistically indistinguishable
from E29's search-and-pick — no dataset clears |t| ≥ 2 in either direction.**
Where E31 (mlp) found ensembling recovers a real gap over the searched pick, the
LightGBM book does not: its arms are already so close in the good corner that
searching, pinning (E29) and now bagging all land in the same place.

Per dataset, grid-ensemble (ENS = equal-weight per-period rank average of all 54
cells) vs the E29 pick, paired Newey-West lag 20, on the block's common window:

| dataset | ENS pay | c.sharpe | dd(pay) | oracle best | E29 pay | ENS − E29 (t) |
|---|---|---|---|---|---|---|
| raw | 0.076201 | 1.076 | 1.907 | 0.079575 | 0.078470 | −0.002269 (t −0.69) |
| tenc | 0.087756 | 1.265 | 1.307 | 0.091553 | 0.088049 | −0.000292 (t −0.12) |
| interact | 0.086847 | 1.267 | 1.339 | 0.090920 | 0.089724 | −0.002877 (t −1.02) |
| fenccausal | 0.090718 | 1.330 | 0.850 | 0.097669 | 0.088014 | +0.002704 (t +0.75) |
| cumlag3 | 0.084004 | 1.202 | 1.326 | 0.088994 | 0.088591 | −0.004587 (t −1.29) |
| tenc_onehot | 0.092806 | 1.302 | 1.181 | 0.094359 | 0.089991 | +0.002815 (t +1.52) |
| onehot_rollsel | 0.087700 | 1.112 | 1.781 | 0.089202 | 0.083778 | +0.003921 (t +1.19) |
| allfeat | 0.092827 | 1.291 | 1.042 | 0.095568 | 0.089739 | +0.003088 (t +0.88) |
| allfeatpp | 0.090615 | 1.154 | 1.534 | 0.093032 | 0.088968 | +0.001647 (t +0.52) |

1. **No significance either way.** ENS edges the pick on level on the 5 richer
   feature sets (fenccausal, tenc_onehot, onehot_rollsel, allfeat, allfeatpp) and
   loses on the 4 leaner ones, but every delta sits inside the noise band — the
   largest is tenc_onehot at t +1.52. Bagging neither reliably helps nor hurts.
2. **The oracle stays out of reach.** ENS is below the best single cell on all 9
   (short by 0.0015–0.0070). The oracle needs test scores to pick and is not
   achievable live; averaging does not recover it, it regresses toward the middle
   of the grid.
3. **The one arm that stands out is `fenccausal`** — best c.sharpe (1.330) and by
   far the lowest drawdown (0.850 vs 1.1–1.9 elsewhere): the risk-not-level
   profile, and the same causal per-symbol encoding the deployed `IA_01` /
   `CU_SA_IA_BO_01` lean on. If any piece of E30 graduates it is this one, on the
   drawdown — and it already lives in the book via that encoding.
4. **Not deployable as-is:** no arm beat its E29 baseline at the house |t| ≥ 2 bar.

Bookkeeping: the 54 cells/dataset are intermediates in the `lgb` tree — forecasts
reclaimed as each ensemble is built, cell metrics json reclaimed after this report
(per_period series kept for the oracle) — so `experiment_db` counts the 9
deployable `e30_<ds>_ens` arms, not the 495 raw cells. HTML board:
`results/reports/crypto/e30_gridensemble_vs_pick.html`. (The FARM-over-full-grid
study that previously held the id E30 in `experiment_db` was renumbered **E36**;
E34/E35 are reserved for E33's follow-ups.)

## E31 — findings

**Averaging the whole mlp_ktz grid beats picking the search-selected best on 7 of
9 datasets, and moves the single MLP from the bottom of the book into its lower
middle — but not up to the deployed neural slot.** The mechanism is the one E28
predicted: on a ~26-effective-window CV the selected "best" config is a noisy
pick, and an equal-weight rank-average over the 120 cells is immune to that
selection variance. (Equal-weight averaging dilutes toward the mean member by
construction — so where selection is reliable the ensemble would only match that
mean and could not win; the real-data gain is therefore attributable to
selection-noise immunity, not to averaging per se. This is an argument from the
"averaging dilutes the mean" principle, not a recorded synthetic result.)

Method: `mlp_ktz_ens` (`GridEnsembleRegressor`) fits all 120 E17-prior cells
(20 lr × 3 arch × 2 alpha × identity) on each rolling 130/130 block and
rank-averages their OOS predictions — the same per-period rank operator the
deployed CURUPIRA_SACI/CU_SA blends use, applied to one model's grid instead of
to distinct feature sets. No inner-CV selection: `--n-iter 1 --n-splits 2` (2 is
the CV floor — n_splits=1 skips its only fold). The combiner ranks each member
over the whole predicted block, not per period, because predict never sees period
labels; that approximation is validated ≈ per-period `run_ensemble` on the narrow
sets (per-period scored-CORR difference < 5e-3) and is exact live (one period per round).

### Ensemble (E31) vs the searched pick (E28) — paired Newey-West, lag 20

| dataset | e28 pay | e31 pay | Δpay | t(pay) | ΔCORR | t(CORR) |
|---|---|---|---|---|---|---|
| tenc | 0.07489 | 0.08392 | +0.00903 | **+2.25** | +0.01575 | **+3.13** |
| cumlag3 | 0.07077 | 0.08282 | +0.01205 | +1.46 | +0.01494 | +1.88 |
| interact | 0.07098 | 0.08120 | +0.01022 | +1.49 | +0.02984 | **+3.03** |
| fenccausal | 0.07765 | 0.08334 | +0.00569 | +0.77 | +0.01377 | +1.78 |
| raw | 0.07163 | 0.07598 | +0.00435 | +0.84 | +0.00941 | +1.44 |
| onehot_rollsel | 0.06753 | 0.07064 | +0.00311 | +1.50 | +0.00472 | +1.73 |
| allfeat | 0.06532 | 0.06760 | +0.00228 | +0.61 | +0.00761 | +1.85 |
| allfeatpp | 0.07211 | 0.06851 | −0.00360 | −0.79 | −0.00040 | −0.07 |
| tenc_onehot | 0.07345 | 0.06704 | −0.00642 | −1.86 | −0.00853 | **−2.10** |

The direction is positive on 7 of 9 and the CORR lift is the reliable part —
`tenc` (+3.13) and `interact` (+3.03) are the only arms that significantly
IMPROVE CORR (|t|≥2, positive), most others sit +1.7 to +1.9; `tenc_onehot` also
clears |t|≥2 on CORR but at t−2.10, in the direction of HARM. On
payout only `tenc` clears the bar (+2.25); the rest are the |t|<2 that ~26
windows make the expected outcome, but the consistency of sign is itself the
signal. **The exception is the widest one-hot sets:** `allfeatpp` flat and
`tenc_onehot` a genuine loss (CORR t−2.10) — exactly the two where E28's pick was
already strong and where the block-vs-per-period combiner is unvalidated. Whether
the `tenc_onehot` loss is real or a combiner artifact is being cross-checked
against per-period `run_ensemble`.

### E31 vs the deployed book (report common window, n=458)

Best submitted MLP = best overall = `CU_SA_IA_BO_01` (0.10550); pure MLP `IA_01`
0.10160 — unchanged from the E28 report. What moved is the single MLP's rank:
E28's best arm (`e28_fenccausal`, 0.07816) sat near the bottom, above only
`SACI_05`. **`e31_tenc` (0.08499) has climbed to just below `CURUPIRA_02`
(0.08528), now above `CURUPIRA_01`, `SACI_04` and `SACI_05`** — roughly the
lower-middle of the 28-arm block. The rest of the narrow E31 arms
(`cumlag3` 0.08318, `fenccausal` 0.08317, `interact` 0.08066, `raw` 0.07957) land
just below it; the four wide one-hot arms stay at the bottom (0.067–0.072).

So ensembling recovers a real slice of the E28 gap — but `e31_tenc` is still
−0.017 below `IA_01` and −0.021 below `CU_SA_IA_BO_01`. **The book's edge remains
blending ACROSS models and feature sets, not across one model's hyperparameters.**

### Ensemble vs its own best member — the oracle (fenccausal, all 120 members reconstructed)

`scratchpad/e31_members.py` drives the pipeline's own `forecast()` and intercepts
the OOS predict to recover all 120 members exactly (self-check vs the launched arm:
max|diff| 1.1e-15). On fenccausal:

| | payout |
|---|---|
| oracle — best single cell (#76: identity, (32,), lr 2.1e-2, alpha 0.1) | **0.09109** |
| ENSEMBLE (all 120, block-rank) | 0.08334 |
| E28 CV-searched pick | 0.07765 |
| mean member | 0.07351 |
| worst member | 0.03964 |

So the arm-board skill's mandated question — **does the ensemble beat its own best
member? — is NO on fenccausal: it loses to the oracle by −0.00775.** But it beats
the MEAN member by +0.00983 and beats 84% of members (101/120). Read together:
the CV-pick-to-oracle gap is 0.0134 (0.09109 − 0.07765) — that is the price of
selection noise — and the ensemble recovers ~42% of it (0.08334 − 0.07765 =
0.00569) by being immune to the pick, while leaving the rest on the table by
diluting the best cell. **The ktz grid DOES contain a strong fenccausal config
(0.09109, and squarely inside the priors); CV just cannot reliably find it and the
equal-weight ensemble waters it down.** Even that oracle (0.09109) stays below
IA_01 (~0.098–0.102), consistent with IA_01's relu/wide/high-alpha region sitting
outside the ktz box entirely — the motivation for E33/E34/E35. A CV-weighted or
top-k ensemble (up-weighting good cells rather than averaging all) is the natural
lever the equal-weight design leaves unpulled.

The combiner cross-check on fenccausal is clean: block-rank vs per-period
`run_ensemble` differ by |Δpayout| 1.9e-4, so the in-estimator block-level ranking
is faithful here.

### Caveats and what is NOT in these artifacts

- **The 4 wide one-hot arms still rest on the unvalidated block≈per-period
  equivalence** (the fenccausal cross-check above validates only a narrow-ish
  44-col set). Live is unaffected (one period per round); the backtest blocks are
  the exposure.
- **Equal weight includes the lr-tail members** (2e-4 underfit, 5e-2 noisy) at full
  weight. If a dataset underperforms, a CV-weighted or lr-trimmed blend is the
  follow-up — not evidence that ensembling fails.

Artifacts: `results/reports/crypto/e31_ensemble_vs_book.html`;
`results/{metrics,forecast}/crypto/mlp_ktz_ens/e31_*`; runner
`experiments/e31_mlp_ktz_ensemble_crypto.sh`. Sibling studies: E30 (lgb), E32
(catboost) ask the same question of the other two families.

## E32 — findings

**Averaging the catboost_ktz grid LOSES to searching it on all 9 datasets — the
opposite of E31's MLP result — because the two conditions that make grid-ensembling
pay are both absent for this booster.** Payout is negative on every dataset and
clears |t|≥2 on 4 (fenccausal −3.53, interact −2.99, tenc −2.93, cumlag3 −2.00);
CORR is negative on all 9 and significant on 7. And the loss is not redeemed by
risk: the ensemble's payout Sharpe is LOWER than the pick's on ALL 9 datasets; its
drawdown is modestly shallower on 6 of 9 (the one place averaging shows its usual
hand), but that is swamped by the Sharpe loss — so the programme's "averaging buys
risk over level" prior (E10/E23/E25) does not hold here. No E32 arm is a deploy
candidate.

The mechanism is the flip side of E31's. Equal-weight rank-averaging beats CV
selection only when (a) the selection is NOISY — so immunity to the noisy pick is
worth something — and (b) the members are DIVERSE enough that the diluted-toward-
the-mean average is not much worse than the best cell. The 12 catboost_ktz cells
fail both: they are near-identical (ENS↔SEL within-period rank corr **0.949–0.974**,
against the 0.03–0.75 of the book's cross-model blends), and CatBoost's symmetric
oblivious trees make its CV pick reliable enough that averaging just regresses the
strong cells toward the weak low-lr/shallow ones. Where E31's 120 diverse MLP cells
had selection noise to be immune to, E32's 12 correlated booster cells have almost
none — so only the dilution cost remains.

Method: `catboost_ktz_ens` (`CatBoostGridEnsemble`) fits all 12 cells
(depth{2,4} × lr{.005,.01,.02} × rsm{.1,.2}; iterations 2000, l2_leaf_reg 10,
min_data_in_leaf 100 pinned) on each rolling 130/130 block and rank-averages their
OOS predictions (block-level rank, validated ≈ per-period in `test_pipeline`);
`--n-iter 1 --n-splits 2`, selection removed. The control `catboost_ktz` searches
the SAME 12 cells exhaustively (`--n-iter 12 --n-splits 5`) and keeps the CV-best —
both wrappers read the grid from `CatBoostGridEnsemble`, so it is average-12 vs
pick-1-of-12 with nothing else varying.

### Ensemble (ens) vs the searched pick (sel) — paired Newey-West, lag 20

Δ = ens − sel; negative = the ensemble is worse. Sorted by t(payout).

| dataset | sel pay | ens pay | Δpay | t(pay) | ΔCORR | t(CORR) | ens↔sel ρ |
|---|---|---|---|---|---|---|---|
| fenccausal | 0.09303 | 0.08143 | −0.01160 | **−3.53** | −0.01457 | **−3.96** | 0.949 |
| interact | 0.08727 | 0.08074 | −0.00653 | **−2.99** | −0.01004 | **−3.99** | 0.965 |
| tenc | 0.09067 | 0.08318 | −0.00749 | **−2.93** | −0.01016 | **−3.09** | 0.955 |
| cumlag3 | 0.08520 | 0.07881 | −0.00639 | **−2.00** | −0.00928 | **−2.63** | 0.950 |
| allfeatpp | 0.09058 | 0.08600 | −0.00458 | −1.88 | −0.00989 | **−3.41** | 0.974 |
| onehot_rollsel | 0.08827 | 0.08389 | −0.00439 | −1.52 | −0.01068 | **−3.01** | 0.967 |
| raw | 0.07793 | 0.07409 | −0.00383 | −1.51 | −0.00720 | **−2.25** | 0.953 |
| allfeat | 0.08934 | 0.08819 | −0.00115 | −0.32 | −0.00299 | −0.67 | 0.959 |
| tenc_onehot | 0.08842 | 0.08753 | −0.00090 | −0.29 | −0.00438 | −1.20 | 0.973 |

Every point estimate is negative. Payout clears the |t|≥2 bar on 4 of 9; CORR — the
more reliable axis, as in E31 — is significantly worse on 7 of 9. MMC is negative
on all 9 (significant on tenc/interact/fenccausal). The two nulls (allfeat,
tenc_onehot) are the widest one-hot sets, where sel and ens are most redundant
(ρ 0.959, 0.973) and both models are least separable — a null there is "the two are
the same", not "averaging is fine".

### E32 vs the deployed book (report common window, n=458)

Best submitted = `CU_SA_IA_BO_01` 0.10550; `IA_01` 0.10160 — the neural slots, as in
E31. The best CatBoost arm of the 18 is `e32_sel_fenccausal` (0.09328), which lands
about 9th of the 37-arm block — with the deployed single-model slots, +0.0032
(t +0.51, not significant) over the block's payout baseline `SACI_06`, and well below
the blends. The ens arms fall to the BOTTOM of the block: `e32_ens_raw` −0.0148
(t −2.53), `e32_ens_cumlag3` −0.0139 (t −3.00), `e32_ens_interact` −0.0109 (t −2.66)
against that baseline. So CatBoost on a single feature set — searched OR ensembled —
sits at the level of the book's single-model slots, never the blends; ensembling
its grid moves it the wrong way.

### Read against E31 and E30

E31 (MLP): ensemble BEAT the pick on 7/9. E32 (CatBoost): ensemble LOST on 9/9. Same
operator, opposite verdicts — and the difference is exactly the noisy-selection /
member-diversity account above, not the model being "worse" (searched CatBoost
0.093 is competitive with searched MLP 0.093). Two caveats before generalising: the
grids differ in SIZE (MLP 120 cells, CatBoost 12 — the CatBoost grid is deliberately
small so the ensemble can fit all of it, which itself limits both diversity and
selection noise), so "MLP vs CatBoost" is confounded with "120 vs 12"; and E30 (lgb,
54 cells) is the tie-breaker on that axis and should be read alongside once it lands.
The clean, unconfounded claim is the within-study one: for THIS 12-cell CatBoost
grid, averaging is a strict loss to searching.

Artifacts: `results/reports/crypto/e32_catboost_ktz.html` (interactive board vs the
book, prediction-correlation matrix, screened ensembles);
`results/{metrics,forecast,per_period}/crypto/{catboost_ktz_ens,catboost_ktz}/e32_*`;
runner `experiments/e32_catboost_ktz_ensemble_crypto.sh`; report `logs/e32_report.txt`.

## E33 — findings

**The transfer problem was an alpha-ceiling problem.** E17 capped alpha at 0.1
and froze the first layer at 32; its priors then failed on fenccausal (IA_01 lives
at alpha 8-10, relu, width 64-128, all outside E17's box). E33 re-swept with those
axes opened, on three datasets, and the result is unusually clean: **with alpha
opened to 10, all three datasets agree on the same priors.**

Cross-dataset best-median value per axis (1728 cells, collapse corner filled 0):

| axis | raw | fenccausal | tenc_onehot |
|---|---|---|---|
| alpha | **10** | **10** | **10** |
| lr | 1e-3 | 1e-3 | 1e-3 |
| depth | 1 | 1 | 1 |
| activation (median) | identity | identity | identity |
| width | 64 | 32 | 64 |

- **ALPHA is the axis E17 got wrong, not width.** Alpha 10 is decisively best on
  every dataset (fenccausal median 0.087 at alpha 10 vs 0.051-0.055 at alpha
  <=0.1; raw 0.076; one-hot 0.086). E17's 0.1 ceiling sat entirely below the good
  region. Alpha 100 falls off again (median collapses), so the peak is ~1-10.
- **WIDTH barely matters.** Medians are flat across 32/64/128/256 on all three
  (raw 0.062-0.066, fenccausal 0.059-0.063). IA_01's width 64-128 was incidental;
  the alpha it paired with is what made it work. So E17's frozen width-32 was not
  the error it looked like.
- **DEPTH: shallow.** depth 1 > 2 > 3 everywhere. LR: low, peak 1e-3. Both match
  E17.
- **ACTIVATION is the one split.** identity has the higher MEDIAN (robust) on all
  three, but relu has the higher PEAK -- the top-5 cells on fenccausal and one-hot
  are ALL relu+alpha10 (fenccausal best: relu w128 lr1e-3 alpha10 = **0.0949**;
  one-hot best: relu w32 alpha10 = 0.0938). relu's median is dragged down by the
  collapse corner (high-lr deep relu), which alpha 10 suppresses. So identity is
  the safe default; relu+alpha10 is where the ceiling is.

**IA_01's region is now inside the box and confirmed best.** E33's top fenccausal
cell (relu, width 128, alpha 10, 0.0949) is IA_01's neighbourhood and far above the
old ktz-grid oracle (0.091, E31). The remaining gap to IA_01 (~0.098-0.102) is lr
granularity -- IA_01 searched lr finely and landed at ~0.009, between E33's 1e-3
and 1e-2 grid nodes.

**New priors (for the E34 tune and E35 ensemble), vs the old ktz box:**

| axis | old ktz (E17) | new (E33) |
|---|---|---|
| alpha | {0.01, 0.1} | **{1, 10}** (peak 10; ~100x higher) |
| activation | identity pinned | **{identity, relu}** (relu holds the peak) |
| lr | dense ~2e-4..5e-2 | 1e-4..1e-2, peak 1e-3 |
| width | 32 | 32-128 (weak axis) |
| depth | <=3 | 1-2 |

The single change that matters is alpha 0.1 -> ~10. Un-pinning activation lets the
grid reach the relu+alpha10 peak; width can stay small.

Caveats: ~118 of 1728 cells collapsed (high-lr deep/wide relu) and are filled
payout-0 -- they map the bad corner, they are not missing data. Analysis is
per-dataset (`scratchpad/e33_priors.py`); the shared E17 tool could not decompose
width from depth. Run across deepfive+deepsix (shared FS, OF=12, resume-enabled).

## E34 / E35 — findings (the E33 priors, tuned and ensembled)

**Correcting the alpha ceiling worked: both the searched MLP (E34) and the grid
ensemble (E35) jumped ~0.01-0.02 payout over their old-prior predecessors (E28,
E31) and climbed into mid-book -- but neither fully reaches IA_01 (0.102).** The
residual ~0.010-0.015 gap is lr granularity (IA_01 fine-searched to lr~0.009;
the grid's nearest node is 0.01) plus the book's very top now being multi-member
blends a single-model approach cannot structurally match.

### E34 (mlp_ktz2 tune) vs E28 (mlp_ktz tune) -- paired NW lag 20

| dataset | E28 pay | E34 pay | Δpay | t |
|---|---|---|---|---|
| allfeat | 0.06532 | 0.08778 | +0.02246 | +1.44 |
| interact | 0.07098 | 0.08918 | +0.01821 | +1.66 |
| tenc | 0.07489 | 0.09065 | +0.01576 | **+1.94** |
| tenc_onehot | 0.07345 | 0.08861 | +0.01516 | +0.98 |
| fenccausal | 0.07765 | 0.09108 | +0.01342 | +1.45 |
| cumlag3 | 0.07077 | 0.07955 | +0.00877 | +0.62 |
| onehot_rollsel | 0.06753 | 0.07205 | +0.00452 | +0.28 |
| allfeatpp | 0.07211 | 0.07195 | -0.00016 | -0.01 |
| raw | 0.07163 | 0.06917 | -0.00246 | -0.29 |

7/9 positive, +0.004 to +0.022, mostly on the datasets whose optimum was outside
E17's box. tenc is closest to significant (+1.94); on ~26 effective windows the
consistency and magnitude are the signal. **This confirms E33: the 0.1 alpha
ceiling was the bottleneck.** raw is the one arm that did not move -- its E28
optimum was already inside the old box.

### E35 (ensemble of the corrected grid) -- two comparisons

vs E31 (old-prior ensemble): **all 9 positive**, +0.004 to +0.023, largest on the
wide one-hot sets (tenc_onehot +0.023, allfeat +0.021, allfeatpp +0.018).
Correcting the box lifted the ensemble everywhere.

vs E34 (pick over the SAME corrected grid) -- ensemble-vs-pick, paired NW:
onehot_rollsel +0.0157 (**t+2.26**), raw +0.0117 (t+1.51), allfeatpp +0.0147
(t+1.66), cumlag3 +0.0078; tenc/interact/fenccausal slightly negative (|t|<0.5);
tenc_onehot/allfeat ~flat. **The ensemble's edge over the pick is SMALLER than in
E31**, and for the clean reason: the corrected search (E34) is now good, so there
is less selection noise for the ensemble to hedge. It still helps where the pick
is unreliable (onehot_rollsel, raw, allfeatpp) and is neutral where E34 already
found a strong cell.

### Against the book (report common window, n=457)

Book top now: CU_SA_IA_BO_01 0.1055, CURUPIRA_BOITATA_01 0.1045, CURUPIRA_CAIPORA_01
0.1021, IA_01 0.1016 (new slots since the E31 report).

- **E34** best is fenccausal 0.09247 (12th of 31, above ~8 submitted slots) but
  HIGH VARIANCE: 4 arms fall to 0.066-0.079 (raw, cumlag3, onehot_rollsel,
  allfeatpp significantly below mid-book, t-2.5 to -2.9).
- **E35** is TIGHTLY CLUSTERED 0.084-0.091 -- the ensemble lifts the weak datasets
  (allfeatpp 0.066->0.086, raw 0.072->0.084) and trims the strong (fenccausal
  0.092->0.089), so EVERY arm lands mid-book and NONE is significantly below
  baseline. Best is tenc_onehot 0.09061 (6th, above SACI_06).

**The tune-vs-ensemble tradeoff, cleanly:** E34 buys a higher peak with higher
variance; E35 buys robustness (uniform mid-book, no weak arm) at a slightly lower
peak. Same lesson as E28/E31, but both shifted up ~0.015 by the alpha fix.

**Deploy implication:** the corrected MLP (E34 fenccausal 0.092, or the robust E35
~0.089) is now genuinely mid-book and beats several deployed leaf slots, but does
NOT beat IA_01 or the top blends. As a standalone slot it is not an upgrade; its
value is as a blend member (like IA_01 in CU_SA_IA_BO_01) -- and whether an
mlp_ktz2 arm decorrelates from IA_01 enough to help a blend is the natural next
check (prediction-corr, not payout).

Artifacts: results/reports/crypto/{e34_tune,e35_ensemble}_vs_book.html;
results/{metrics,forecast}/crypto/mlp_ktz2{,_ens}/e3{4,5}_*.

## E42 — cross-sectional rank allocator does NOT beat the static mv book

Evaluated NasdaqJockey's live sizing idea (Out of Sample Ep 5, ~26-29 min):
run a large model stable, rank each model cross-sectionally on several
dimensions, build the composite from the RANKS not the scores, stake the top
ranks and zero the rest, re-ranked every round. Ported to crypto as
`src/analysis/xs_stake.py` and back-tested walk-forward on the 24-slot
per-period panel (470 dates, 313 train / 157 held-out), judged with the classic
pipeline's `stats.py` battery.

**Headline: NULL / INCONCLUSIVE.** Held-out, net of a 0.005/unit turnover cost,
no arm beats the deployed static mv book (B0) with a bootstrap ΔSharpe CI clear
of zero:

| arm | net Sharpe | max_dd | effN | ΔSharpe vs B0 (95% CI) | verdict |
|---|---|---|---|---|---|
| B0 static-mv (incumbent) | 0.97 | 0.87 | 2.8 | — | — |
| B1 equal | 1.10 | 0.81 | 24 | +0.14 [-0.01, +0.45] | INCONCLUSIVE |
| B2 single-best | 1.50 | 0.09 | 1.0 | +0.54 [-0.66, +1.53] | INCONCLUSIVE (artifact) |
| B3 rolling-mv | 1.10 | 0.83 | 7.2 | +0.13 [-0.03, +0.44] | INCONCLUSIVE |
| A1 rank | 0.99 | 1.11 | 6.4 | +0.02 [-0.12, +0.35] | INCONCLUSIVE |
| A2 rank+future | 0.95 | 1.09 | 6.5 | -0.02 [-0.14, +0.24] | INCONCLUSIVE |
| A3 hybrid | 1.00 | 1.10 | 3.2 | +0.03 [-0.13, +0.37] | INCONCLUSIVE |

**What it shows about the idea, beyond the null:**
- **The rank arms (A1/A2/A3) RAISE drawdown** (max_dd ~1.1 vs B0's 0.87) without
  adding Sharpe. The predicted failure: ranking each model independently is
  correlation-blind, and this stable is ~2 effective independent bets
  (participation ratio 1.9 of 24, median pairwise corr 0.75), so the top-K by
  rank piles into one cluster.
- **The "future" momentum term HURTS** (A2 net Sharpe 0.95 < A1 0.99), consistent
  with forward model-performance being ~unpredictable at this horizon.
- **The hybrid (A3) fixes the concentration** (effN 3.2 vs A1's 6.4 after mv
  sizing — mv correctly down-weights the redundant members) but still does not
  beat B0.
- **B2 single-best is a cautionary artifact, not a win.** Its flashy 1.50 Sharpe /
  0.09 drawdown comes from chasing whichever slot had the smoothest *simulated*
  recent Sharpe -- and 46% of the held-out window it parks in `CURUPIRA_05_60`
  and `CURUPIRA_07`, which are NOT stakeable (`NOT_SUBMITTED` / no run). Its
  block-bootstrap CI [-0.66, +1.53] refuses to bless it; the naive Wilcoxon
  p_holm 0.000 is exactly the autocorrelation overstatement the block bootstrap
  corrects.

**Power.** MDE(mean payout delta, 80% power) vs B0 is 0.014 against a payout mean
of 0.057 -- with 20-day overlapping targets the eval window is ~8 effective
independent blocks, so a real difference would have to be large to register.
Friedman across arms is p<0.0001 (the arms DO differ) but that does not say any
beats the incumbent. Pre-declared as a risk/Sharpe study, not a return study.

**Caveat that bounds all of the above.** The panel is SIMULATED pseudo-OOS
(rolling 130-period purged CV), with no payout factor and no per-round clipping,
so levels are not NMR and only arm-vs-arm ratios are trustworthy. Realized live
payouts (`results/reports/crypto/history/`) are still too short and unresolved to
re-run this against; that is the natural follow-up once ~Nov 2026 rounds resolve.

**Verdict: do NOT adopt.** On the evidence available, the rank allocator is not
an upgrade over the deployed mv book and adds drawdown; the deployed static mv
(and plain equal-weight / rolling-mv) are as good or better on the sensitive
axis. No live code was touched (offline `xs_stake.py` only).

Artifacts: `src/analysis/xs_stake.py`, `src/tests/test_xs_stake.py`;
`python3 -m src.analysis.xs_stake --report [--plot out.png]`.

## E47 — deploying the E33 winner: `IA_02` + two MLP blends

Not a sweep, a **deployment**. E33 found the fenccausal MLP peaks at alpha ~10
(the regularisation E17's 0.1 ceiling could not reach); E47 puts that cell to
work. Its top cell — `128,64 / relu / lr 1e-3 / alpha 10` — is wired live as the
pinned MLP component **`IA_02`** (`src/live/registry.py`,
`fit_override.fixed_params`, `NOT_SUBMITTED`, refits live like `IA_01` but pins
its config instead of searching). Two slot-blends are built on it and **priced as
real per-period rank-averages** of their members' finished forecasts — the
deployable number, not the screening proxy, which read `dd(pay)` 0.28 where the
real figure is ~1.0.

**The screen said mlp-vs-mlp blends are worthless** (the E33 cells correlate
ρ 0.80–0.98; 0 of 35 blends beat their best member on anything). **The value is
mlp ⟂ tabular book** (ρ ≈ 0.1), so both deployed blends pair the neural draw with
tabular anchors. Priced over 470 periods, each vs its own best member:

| slot | members (⅓ each) | CORR | payout | c.sharpe | dd(pay) | calmar | burn |
|---|---|---|---|---|---|---|---|
| `CU_SA_IA_BO_01` (best member) | — | 0.16817 | 0.10712 | 1.92 | 1.16 | 0.093 | 38 |
| **`CU_SA_IA_BO_02`** | `IA_02`+`CURUPIRA_05_60`+`CU_SA_IA_BO_01` | 0.16118 | 0.10635 | **2.03** | **1.02** | **0.104** | **30** |
| **`CU_CA_IA_01`** | `IA_02`+`IAUARETE_01`+`CURUPIRA_05_60` | 0.16003 | 0.10422 | **2.05** | 1.08 | 0.097 | 31 |

- **`CU_SA_IA_BO_02`** matches its best member on payout (−0.0008, noise) and
  improves *every* risk metric — a version bump of `CU_SA_IA_BO_01` (same
  Curupira+Saci+Iara footprint, a second neural leg). Deploy for risk, not level.
- **`CU_CA_IA_01`** OUT-LEVELS all three members (0.10422 vs ~0.095) *and*
  improves risk — its mlp/catboost/ridge legs are genuinely diverse, so blending
  adds level here. The level gain is a point estimate, not a paired |t|≥2 result.

**Naming.** `CU_SA_IA_BO_02` is a version bump (no new creature). `CU_CA_IA_01`
= Curupira + **Ca**ipora (the CatBoost leg `IAUARETE_01`, prefix CA not IA to
avoid colliding with the neural Iara, exactly as `CURUPIRA_CAIPORA_01`) + Iara.
Both account slots (`cu_sa_ia_bo_02`, `cu_ca_ia_01`) exist.

**Caveats.** `IA_02` is one config, one seed (like `IA_01`'s 0.16172); the blends
inherit their members' fallbacks (`CU_SA_IA_BO_02` carries `CURUPIRA_SACI_03`'s
example-fallback risk); both carry the 60-day `CURUPIRA_05_60`, so 470 periods.

**Status.** Wired and validated — `IA_02` fits live in ~22 s, both blends resolve
and reproduce their registry backtest to 1e-6, live/report tests green. No open
round at deploy time (1349 closed); the next open round submits both.

Artifacts: `experiments/e47_deploy_mlp_blends_crypto.sh` (reprices the two
blends), `results/{per_period,metrics,forecast}/crypto/{lgb,ridge_glmnet}/e47_*`;
live wiring in `src/live/registry.py` (`IA_02`, `CU_SA_IA_BO_02`, `CU_CA_IA_01`).
