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
| **E45** | Fine-tune TabPFN with a METRIC-ALIGNED objective (CORR, MMC) | `src/analysis/tabpfn_finetune_metric.py` | E44 fine-tuned with the library CRPS+MSE and it HURT; part of the damage might be objective mismatch (MSE compresses the cross-sectional spread the ranking pays for). E45 fine-tunes toward the PAYABLE metric via a custom differentiable loop over TabPFN's bar-distribution mean: (corr) minimise −pearson(pred, target); (mmc) minimise −BMC(pred ⟂ meta, target) on the meta-covered rows. Real meta only exists from 2024-05-07, so the leak-safe (≤2024-06-27) meta-available window is just ~38 dates — the MMC arm is data-starved. Same leak-safe pool as E44, save→model_path round-trip, E19 walk-forward eval; paired NW vs E19 and E44. | 2 | done — **still HURTS; it was prior-corruption, not the loss**: CORR-objective → CORR 0.11170 (paired vs E19 −0.0334, t −2.57); MMC-objective → 0.08488 (t −3.81). Both significantly below zero-shot E19 (0.145), and the CORR arm ≈ E44's MSE arm (0.118) — changing the objective did not rescue it. Zero-shot TabPFN beats every fine-tuned variant (MSE/CORR/MMC). |
| **E56** | Chronos-2 with the feature cross-section as covariates | `src/analysis/chronos2_covariates.py` | The missing cell of the FM arc: a TIME-SERIES FM that ingests covariates. Univariate TS-FMs got ~0 (TimesFM E38 +0.012, TEMPO E40 −0.017, no features); tabular TabPFN on the feature cross-section got 0.145 (E19). Chronos-2 (amazon/chronos-2, 120M, arXiv:2510.15821, `chronos-forecasting≥2.0` `Chronos2Pipeline`) is the first TS-FM with exogenous-covariate support (its "group attention"). Per scored era, ONE `predict_df` panel call over all window-complete symbols (group attention across the cross-section): context = every symbol's (target+44 fenccausal features) up to the leak wall T−20 on a contiguous OWN-STEP index (Chronos needs a regular grid, gappy calendar → own-step, the E38/E40 convention); features for eras T−19..T passed as KNOWN-FUTURE covariates (features at t are observed at t; only the 20-day target is embargoed) so the current cross-section conditions the forecast; era-T forecast = signal. Zero-shot, /scratch venv reusing base cu118 torch (HF_HUB_DISABLE_XET=1). ~86% cross-section coverage/era (~245 symbols). Scored on tabpfn/e19_fenccausal's 510 dates. | 1 | done, 125,163 forecasts, ~1hr, 0.67 GB/era — **NULL**: CORR +0.0025 (t +0.16 vs zero), MMC +0.0115, payout +0.0117, c.sharpe 0.02. Feeding features as covariates did NOT beat the univariate TS-FMs (paired vs E38 dCORR −0.009 t −1.01; vs E40 +0.020 t +1.68 — both ns) and sits **8.7σ below** TabPFN (dCORR −0.143, t −8.72). Confirms the edge is PURELY cross-sectional: Chronos-2 covariates are time-ALIGNED per-series features (+ group attention), a structurally weak match to a per-row cross-sectional signal — even covariate support can't reach the tabular edge. |
| **E83** | Frozen TabPFN backbone + a trained 5-class head (E[y\|x] readout) | `src/analysis/tabpfn_head.py` | The GENTLE adaptation E44/E45 lacked. Full fine-tuning corrupted TabPFN's meta-learned prior and HURT (E44 −0.027 t−3.02; E45 also worse). Here the backbone is FROZEN (vendored checkpoint untouched); only a NEW head trains — `Linear(softmax'd ~5000-bin bar-distribution → 5 classes {0,.25,.5,.75,1}) → softmax → E[y\|x]`. The signal is the CONTINUOUS bounded expectation, NOT a hard label (numerai_corr is rank-based; a hard 5-level output would tie ~245 coins/era and kill rank resolution — 'discrete' lives in the head + CE loss, not the emitted number). Head trained w/ cross-entropy on the leak-safe pre-2024-06-27 pool (CE 1.54→1.33 over 4 epochs). Eval = per-date walk-forward (context 130 dates before T−20, subsample 4096); ONE forward/date yields BOTH e83_head (head) and e83_head_base (raw bar-dist mean = matched n_est=1 zero-shot, validated ≈ E19). Numbered E83 (concurrent session took E82). | 2 | done, 153k rows/arm — **NEUTRAL, not a win, but crucially not corrupting**: head CORR 0.12207 vs matched base 0.11129, paired **+0.0108 (t +0.82)**, MMC −0.0014 (t −0.13), payout **−0.0003 (t −0.02)** — a tiny insignificant CORR nudge, flat payout. Head vs E19 (n_est=4 zero-shot) payout −0.028 (t −2.40), still well below. Freezing the backbone AVOIDS the E44/E45 prior corruption (head ≈ neutral, not damaging), the head emits a valid bounded E[y\|x], but it can't manufacture a real gain: the frozen backbone already produces a near-optimal ranking, so the head just re-learns the natural re-binning (≈ the mean) with too weak a crypto tilt to help. **Zero-shot TabPFN (E19) remains the best.** |
| **E84** | TabPFN zero-shot AND frozen+head on each of CU_SA_IA_BO_01's 8 feature-sets | `/home/danielco/e84_chain.sh` | Sweeps the 8 feature-sets composing the deployed slot CU_SA_IA_BO_01 (raw 22, fenccausal 44, allfeat_pp 812, and the 5 ridge-quintet BLOCKS onehot 740 / tenc 1 / interact 14 / size 4 / price 4, last five as feature-include regexes). Per set: (a) TabPFN zero-shot at E19's config (n_est 4, ctx 4096; raw/fenccausal/allfeat_pp reuse e19_*; onehot at reduced n_est 1/ctx 1024 — 740-dim is pathologically slow), (b) the E83 frozen-backbone+head (emits matched n_est=1 base). 13 new runs, ~3h. Reordered after the full-config onehot zero-shot stalled the sequential chain for hours. | 13 | done — **(1) best zero-shot = fenccausal 0.14504 ≈ raw 0.14299**: the causal per-symbol means barely help TabPFN (they carry IA_01's MLP), and MORE features HURT (allfeat_pp 812 → 0.124 < raw). Blocks weak-but-real (size/tenc/price/interact 0.06–0.08; a single tenc column reaches 0.082); onehot ~null 0.015. **(2) head vs base**: NEUTRAL on every strong set (raw/fenccausal/allfeat_pp |t|<1.1, confirms E83 generalizes); significantly HELPS only onehot (+0.0046 t+3.48, payout t+2.64 — recovers the ticker fixed-effect the mean underuses, but stays 0.033); HURTS price (−0.0225 t−2.43). 8 tests → bar ~|t|2.7, so onehot is the one robust head-win. **No feature-set where the head makes a good TabPFN better.** |
| **E102** | LoRA fine-tune of TabPFN with a per-date cross-sectional CORR loss | `src/analysis/tabpfn_lora.py` | The correctly-aligned fine-tune E44/E45/E83 never achieved on all 3 axes: rank-aligned (differentiable cross-sectional CORR loss, the numerai_corr stand-in), cross-sectional/listwise (each episode = ONE date's full coin cross-section, context = its fenccausal history ≤ T−20 wall, query = date-T cross-section), prior-preserving (only LoRA r=8 on attention q/k/v/out = **294,912 of 11.4M params**, backbone frozen). Readout = bar-dist mean; LoRA MERGED into the backbone then eval is the ORDINARY E19 walk-forward via model_path (E19↔E102 isolates the adaptation). Leak-safe ≤2024-06-27. peft 0.21.2, base env. | 1 | done, ~56m train + 12m eval — **HURT**: the LoRA LEARNED in-sample (training cross-sectional corr 0.129→0.167 over 3 epochs, so the scheme works as designed) but did NOT generalize — eval CORR 0.13593 vs E19 0.14504, paired CORR −0.0091 (t −0.90), **MMC −0.0170 (t −2.12), payout −0.0179 (t −2.00)** — significantly worse on MMC/payout. Overfit the non-stationary pre-2024 regime. **Definitively closes the fine-tuning question**: across full-FT (E44/E45, hurt), frozen head (E83, neutral), and now the rank-aware prior-preserving LoRA (E102, hurt), NOTHING beats zero-shot. The frozen meta-learned prior already reads the cross-section via the pre-binned features; any weight update overfits a low-SNR non-stationary target and erodes it. **Zero-shot TabPFN (E19) is the ceiling.** |
| **E103** | E102 LoRA fine-tune per Tier-1 (crypto-derived-structural) feature set | `/home/danielco/e103_chain.sh` | E102's LoRA scheme (per-date cross-sectional CORR loss, r=8 adapters on attention q/k/v/out, backbone frozen, merged then evaluated as the ordinary E19 walk-forward rolling 130/130 embargo 20 seed 42 n_est 4 ctx 4096) applied to each **Tier-1** feature set — those built by STRUCTURAL processing of the raw 22, no external data. **7 new arms**: raw `v2.0` (22f), `lag3` (44f), `cumdiff3` (69f), `ar` (25f), `interact` (36f), `tenc` (23f), `aetuned` (30f); `fenccausal` (44f) is `e102_lora`, reused as the 8th screen member. Dropped: `allfeat` 812f (TabPFN stall/OOM hazard on 8 GB — froze E84), `e10stack` 2f (degenerate), and the cumulative-lags family (`cumlag2`/`cumlag3` OOM the 8 GB cards at 66–88f; theme covered by `lag3` + `cumdiff3`). Trained on deepfive (2× idle GPU, shared `/home`), two parallel lanes. THEN (2) screen ensembles of the 8 LoRA arms for one beating the top single arm; (3) screen ensembles of the winners with the submitted book + e102. Hypothesis (from E102 + the e102×book 4-way): each single arm underperforms E19 zero-shot, but the LoRA objective DECORRELATES them, so a blend — and a blend with the book — may add a RISK (drawdown/calmar) edge, not a payout-level one. Leak-safe ≤2024-06-27, paired NW. | 7 (+1 OOM) | done, 7/7 (cumdiff3 backfilled solo with a 2 h cap after timing out at 1 h) — **STEP 1: every LoRA arm is weak and below e102.** Single-arm payout 0.030–0.039 / CORR 0.105–0.111 for six of them, and `cumdiff3` far worse at 0.0244 / 0.050 — its training loss never fell (1−corr 0.945→0.948) and it is the one arm decorrelated from e102 (ρ 0.24), but too weak to use: `e102+cumdiff3` pays 0.0436, t −3.52 vs e102 alone (e102 fenccausal 0.0634/0.136; E19 zero-shot 0.145) — LoRA hurts the level on *every* Tier-1 processing, reconfirming E44/E45/E102. **STEP 2: no LoRA-arm ensemble beats the top single arm (e102).** All 154 combos of the 8 arms (sizes 2–4) dilute: the arms are mutually correlated (ρ 0.59–0.81, cumdiff3 excepted) and e102-dominated, so blending averages down (t vs best member −2.8 to −3.2); 0 beat-their-best. **STEP 3: no {LoRA+book} ensemble beats the deployed best (CU_SA_IA_BO_01 payout 0.10712) on PAYOUT** — 0/375 on the series proxy, confirmed at the real prediction level (`run_ensemble`): best e102 blend `e102+BO_01+IA_01+CAIPORA+C05` = payout 0.10538, below 0.10712; e103 arms strictly hurt (dpay t −4 to −5). **BUT e102 is a genuine RISK diversifier** — solo it has the pool's lowest payout drawdown (0.34) and highest calmar (0.217); added to the book it ~holds payout (−0.0017, noise) while cutting dd ~28% (1.157→0.827), lifting calmar ~37% (0.093→0.127) and CORR (0.1682→0.1687). The earlier e102×book 4-way (with the 60-day decorrelator `CURUPIRA_05_60`) remains the strongest risk variant: payout *tied* at 0.10711, dd 0.638. **Verdict: LoRA fine-tuning across 7 feature processings cannot make a level-competitive TabPFN arm; its only value is as a decorrelated low-drawdown risk leg in a book blend, and the extra Tier-1 arms add nothing beyond e102 (weaker + correlated).** OOM casualties on the 8 GB cards: cumlag2/3 (66–88f) and the cumdiff3 timeout (69f, slow); allfeat 812f and e10stack 2f excluded by design. **Named in `src/live/registry.py`**: `TUPA_01` (E19 zero-shot TabPFN), `TUPA_02` (e102 LoRA, component) and `TU_CU_SA_IA_BO_CA_01` (the 5-way risk blend, backtest run `tabpfn/tu_cu_sa_ia_bo_ca_01`) — all three in the new `registry.BACKTEST_ONLY` set: resolvable by name, in no `ORDER`, never fitted or uploaded live until a GPU TabPFN fit/predict path is validated. |
| **E46** | FARM over the from-scratch ktz SEARCH-PICK models (lgb E29 + catboost E32-sel + mlp E34) vs 1/N | `e46_ktz_searchpick_farm_crypto.sh` | Combine the 27 best-tuned (model × dataset) forecasters — the config each family's from-scratch KTZ-prior search PICKED per dataset — with the Lee–Lee FARM combiner, vs the plain 1/N of the same 27. (Earlier plan FARMed the raw HP grid *cells*, but E30/E32 reclaim those and the surviving mlp E33 sweep is junk-heavy; the search-*pick* models are the clean pool, all already on disk.) POOL: `lgb_ktz/e29_<ds>` (60-draw search) + `catboost_ktz/e32_sel_<ds>` (12-cell exhaustive) + `mlp_ktz2/e34_<ds>` (80-draw search on the E33-reprior wider grid), 9 datasets = 27 base forecasters. FARM (equal=1/N baseline, farm1/2/3 × {ridge,l2boost,postalasso}, gr, ds) over four pools: per-family (lgb 9, cat 9, mlp 9) and pooled (all 27). Paired NW lag 20 vs each pool's own 1/N; best arm ranked against the book. Prior (E27/E29/E36): 1/N wins on this panel. | 48 | done — **NULL, 1/N wins every pool**: no FARM arm beats the equal rank-average; `l2boost` ties it (mlp `farm1_l2boost` +0.0009, t+0.40 the only arm above 1/N anywhere), while the `ridge`/`postalasso`/`gr`/`ds` arms OVERFIT the deviation regression on ~26 windows and lose up to t−2.3 (lgb). The only real signal is that pooling all 27 lifts the SIMPLE average (all-1/N payout 0.0911 > lgb 0.0898 / cat 0.0906 / mlp 0.0864) — diversity helps via 1/N, not FARM. Vs the book (common 457): pooled 1/N 0.0928 (t+0.74 vs SACI_06) ≈ best single submitted, but BELOW every deployed slotblend (0.099–0.105); FARM's best arm is slightly worse (0.0911). **But SUBSET SELECTION does what reweighting can't**: a curated 4-member diverse blend `e46_blend_cf_ca_mt_mi` (catboost fenc+allfeat + mlp tenc+interact, NO lgb) priced through `run_ensemble` scores payout **0.0980, c.sharpe 1.94, burn(d) 37** — beats the full-27 1/N (+0.006), the best single (+0.005, t+1.68 vs SACI_06) and ~half the deployed slots (CURUPIRA_08, SACI_02/01, CURUPIRA_07, IAUARETE_01), ≈ CURUPIRA_SACI_03; still below the top tier (CU_SA_IA_BO_01/BOITATA/CAIPORA 0.102–0.105). The payout-SERIES screen HID this (said no subset beats its best member) — the real per-period rank-average exploits cross-sectional diversity the series-average can't see. FARM/peLASSO missed it because they reweight the full pool rather than DROP members. |
| **E55** | Factor-momentum stake timing across the submitted book | `src/analysis/factor_momentum_stake.py` | Gupta–Kelly "Factor Momentum Everywhere" applied to STAKE SIZING — each of the 24 submitted models is a factor, its payout its return; tilt stake toward models that did well on recently RESOLVED rounds, walk-forward. Models the real Numerai frictions: the 20-day DECISION LAG (`.shift(20)`) AND the WITHDRAW-DELAY LADDER (realised weight = trailing-20 mean of the target, since a stake change glides in over ~20 rounds), plus the 0.1-NMR/round floor. Equal-capital z-tilt allocation; sweep rolling lookbacks {20,40,60,90,120,180,250} and EWMA halflives {10,20,40,80,120}; baselines equal-weight and full-stack-on-causal-best; select-on-train/report-on-held-out, paired NW lag 20. Panel = book payout series (pseudo-OOS, ratios not NMR). | 84 | done — **momentum BEATS equal weight, universally OOS**: full 84-arm grid (rules ztilt k{.5,1,2} / topk K{1,3,6} / rankprop × 12 signal windows), **84/84 arms beat equal on the held-out third, 73/84 clear t≥2**. OOS-selected (max train calmar → held-out) = **`top1 roll250`** (full stack on causal-best, 250-day lookback): held-out +0.0066 **t+5.04**, and it also clears the full-window at **t+2.69**. The earlier "regime-concentrated" worry is resolved by lookback: the **9 arms that clear t≥2 on the FULL window are the LONG-lookback ones** (roll250 / ewma120), concentrated or diffuse — short windows win big held-out but fail full-window (regime-specific). Concentration (topk) > diffuse tilt (ztilt) on level; higher κ > lower. Turnover 0.006–0.04 (ladder-smoothed → withdraw friction negligible, implementable). Ratios not NMR; one held-out split. |
| **E25** | Weighted and selection combiners over a diverse pool | `e25_weighted_combiners_crypto.sh` | E10 showed fitting beats nothing when members are homogeneous; E23 showed a fixed 1/N cannot exploit a weak-but-orthogonal member because it cannot downweight anything. This is the untested cell: 8 walk-forward weighting/selection schemes (`equal`, `inverse_variance`, `complete_subset`, `caruana`, `bagged_caruana`, `shrunk_tangency`, `granger_ramanathan`, `stacking_nnls`) over the same 5-member pool E23 blended, each paired against E23's 1/N on identical members and dates. A tabular-only pool repeats every scheme WITHOUT the LSTM, so any gain can be attributed to handling the orthogonal member rather than to weighting per se. 16 tests → read |t| ≳ 2.9, not 2. | 16 | done, 16/16, 20 min — **NULL, and one-directional**: all 14 schemes LOSE to 1/N (see findings) |
| **E26** | Sign-constrained GRR on allfeat_pp — all betas ≥ 0, all betas ≤ 0 | `e26_sign_constraint_crypto.sh` | The one GRR knob that shrinks the feasible set rather than the penalty: a box-constrained QP by coordinate descent, optimality on KKT. Both constraints bind (139–217 of 394 live coords), measured before launch, so neither arm is a no-op. Alpha floored at 1e-2 in all three arms — a constrained solve at the 1e-4 default cannot reach optimality. | 3 | registered |
| **E27** | LightGBM hyperparameter sensitivity (Kalfa-Timmermann-van der Zwan) | `e27_hyperparam_sensitivity_crypto.sh` | Which hyperparameters actually move payout, which corner values lock in good/bad models, and what range is a sound prior? A 315-cell factorial (num_leaves x learning_rate x n_estimators x colsample) marginalised the paper's way. | 315 | running |
| **E47** | Deploy the E33 winner: pinned MLP `IA_02` + two blends | `e47_deploy_mlp_blends_crypto.sh` | Not a sweep — a DEPLOYMENT. Wire E33's top fenccausal cell (128,64/relu/lr1e-3/**α10**) live as the pinned MLP component `IA_02` (`NOT_SUBMITTED`), then build and **price** (real per-period rank-average, not the screened proxy) two blends on it: `CU_SA_IA_BO_02` = `IA_02`+`CURUPIRA_05_60`+`CU_SA_IA_BO_01` (a version bump — same Curupira+Saci+Iara footprint, 2nd neural leg) and `CU_CA_IA_01` = `IA_02`+`IAUARETE_01`+`CURUPIRA_05_60` (Curupira+Caipora+Iara). Does a 2nd, differently-regularised neural draw improve the book? | 2 | done — **deployed**; both slots on the account, priced (see findings), submit at the next open round |
| **E48** | Random-forest hyperparameter sensitivity (Kalfa-Timmermann-van der Zwan) | `e48_rf_hyperparam_sensitivity_crypto.sh` | The 'Overhyped?' sweep E17 ran on the MLP and E27 on LightGBM, now on a BAGGED FOREST — the first random forest the book has tried. One pinned walk-forward per cell of a 3×4×4×4 factorial: `n_estimators` {100,300,1000} × `max_depth` {3,6,12,None} × `max_features` {0.1,0.3,0.6,1.0} × `min_samples_leaf` {1,10,100,1000} = 192 cells on the raw 22 of crypto/v2.0, then marginals, fANOVA importance, corners and priors via `src.analysis.hyperparam_sensitivity_rf`. A forest is bagging not boosting, so there is **no learning-rate axis**: `max_features` carries the Numerai colsample idiom (the most distinctive setting), `min_samples_leaf` the heavy-leaf idiom, and the two-way claim tested is `max_features × min_samples_leaf` (the forest analogue of the paper's lr × alpha). `bootstrap`/`max_samples` pinned at the forest default. Maps the surface once so a future `rf_ktz` grid can be set from evidence. | 192 | done, 192/192 (0 fail) — **all three Numerai idioms transfer to the forest, and `n_estimators` is a red herring.** fANOVA shares: `min_samples_leaf` 0.31 + `max_depth` 0.28 + their interaction **0.25** dominate; `max_features` 0.12; **`n_estimators` 0.0002** (100 trees = 1000, so pin it at 100). All three axes monotone the Numerai way: **shallow** (depth 3 med 0.076 > None 0.064), **heavy leaf** (1000 med 0.078 > 1 0.067), **low colsample** (mf 0.1 med 0.077 > 1.0 0.067). Best cell `e48_ne100_d3_mf01_ml1000` payout **0.0808** / CORR 0.1512; best CORR **0.1561** at mf0.3/ml1000/d3 — ≈ the book's best MLP slot IA_01 (0.158) on the raw 22, though not yet a paired/matched comparison. Worst cell (deep/full-colsample/no-leaf) payout 0.033 / CORR 0.061 — the KTZ bad corner, dispersion left-skewed (−1.5). The depth×leaf interaction is substitution: a deep tree is fine IFF the leaf floor is heavy. Ranking holds across both period-halves. **Prior for an `rf_ktz` grid:** pin `n_estimators`=100; `max_features` {0.1,0.3} (drop 1.0 — 0/48 support among good cells); `min_samples_leaf` heavy (favour 1000); `max_depth` shallow {3,6} (or let the leaf floor cap complexity). Report: `logs/e48_report.txt`. |
| **E51** | RF tuned from scratch with the E48 priors (`rf_ktz`) | `e51_rf_ktz_tune_crypto.sh` | The rf sibling of E28 (`mlp_ktz`) / E50 (`lgb_ktz`): a from-scratch `RandomizedSearchCV` of `rf_ktz` — the E48 sensitivity findings compiled into a search grid — on the raw 22 and on fenccausal (44). Grid: `n_estimators` **pinned 100** (E48 red herring), `max_depth` {3,4,6}, `max_features` {0.05..0.30} (dense-low colsample; 0.6/1.0 dropped), `min_samples_leaf` {300,1000,2000,4000} — the E48 leaf ceiling **opened past its binding max of 1000** (its marginal ran to the grid edge). 60-cell all-good grid, `--n-iter` 40, 5-fold inner CV, CURUPIRA cadence (rolling 130/130, embargo 20, seed 42). Produces a deployable RF; the honest test is a paired vs-book comparison (next). | 2 | running — raw **done: CORR 0.15372 / payout 0.08009 / c.sharpe 1.60** (search picks depth3-6/mf0.1-0.3/**leaf300**/100trees — lighter leaf than E48's OOS marginal of 1000, a CV/OOS nuance); fenc re-running (1st attempt killed mid-run after refit 2 on the nested-loky path). Superseded by **E52** (all 9 book datasets). |
| **E52** | One `rf_ktz` tune per book dataset, with the E48 priors | `e52_rf_ktz_tune_crypto.sh` | The rf sibling of E28 (`mlp_ktz`) / E50 (`lgb_ktz`), scaling E51 from 2 to **all 9** book datasets (raw22, tenc23, interact36, fenccausal44, cumlag3-88, tenc_onehot763, onehot_rollsel786, allfeat788, allfeatpp812). From-scratch `RandomizedSearchCV` of `rf_ktz` (E48 priors: n_est pinned 100, depth {3,4,6}, max_features {0.05..0.30}, min_samples_leaf {300..4000} ceiling opened), `--n-iter` 40, 5-fold CV, CURUPIRA cadence. **Serial search × n_jobs** (one joblib level — E51's fenc hung on the nested `search-jobs>1 × n_jobs>1` path), parallelised at the dataset level (`RFK_POOL`). Gives every book slot an RF counterpart tuned on its own features — the pool a paired-vs-book comparison and any RF blend draws from. | 9 | running |
| **E57** | `rf_ktz` 60-day tune of E52's top-5-MMC datasets | `e57_rf_ktz_60d_tune_crypto.sh` | The 60-day twin of E52's best arms. E52 ranked the 9 rf_ktz tunes by MMC on `target_binned_return_20`; E57 re-tunes the **five highest-MMC** datasets — allfeat(788), fenccausal(44), tenc_onehot(763), allfeatpp(812), interact(36) — on the **60-day** label. Only the target changes: same rf_ktz grid (E48 priors), same 40-draw from-scratch search, same rolling 130/130 / 5-fold / seed 42, run on the **same sklearn 0.24** as E52 so only the label differs. `--embargo auto`→60; 60-day runs score ~470 periods. Tests whether RF's 20-day MMC strength carries to the 60-day label and reaches the 60-day book (SACI_03 MMC 0.0788/pay 0.0930, CURUPIRA_05_60 MMC 0.0804/pay 0.0922). | 5 | running |
| **E58** | `lgb_ktz` 60-day tune of E29's top-5-MMC datasets | `e58_lgb_ktz_60d_tune_crypto.sh` | The LightGBM sibling of E57 and the 60-day twin of E29's best arms. E29 ranked the 9 lgb_ktz tunes by MMC on `target_binned_return_20`; E58 re-tunes the **five highest-MMC** datasets — tenc_onehot(763), allfeat(788), interact(36), allfeatpp(812), fenccausal(44), the **same five** the rf tune (E57) picked — on the **60-day** label. Only the target changes: same lgb_ktz grid (E27 priors, L=lr·n_est≤40), same 60-draw from-scratch `RandomizedSearchCV`, same rolling 6month/6month / 5-fold / seed 42 / default python3 so only the label differs. `--embargo auto`→60; 60-day runs score ~470 periods. Excludes the leaky non-causal global encoding (fencglobal, ~0.14 MMC an artefact) and all blend/ensemble arms (nothing to tune). Tests whether LightGBM's 20-day MMC strength carries to the 60-day label and reaches the 60-day book (SACI_03 MMC 0.0788/pay 0.0930, CURUPIRA_05_60 MMC 0.0804/pay 0.0922); secondary paired read vs E01's 60d m50 pins. | 5 | done, 5/5 in 85 min (470 periods each), report `report_20260914T195441Z.html`. **allfeatpp REACHES the book**: payout 0.09358 vs SACI_03 0.09302 (paired NW dpay +0.0006, t +0.03 — indistinguishable), MMC 0.0778 tied, and does it with ~HALF the drawdown (dd(pay) 5.89 vs ~11) and higher p.sharpe (0.90 vs 0.72) — matches on level, better on risk. **fenccausal COLLAPSES on 60d**: the only arm significantly below the book (dpay −0.0177, t −2.24), worst payout 0.07533 / drawdown 11.85 — its 20-day MMC edge (the deployed IA_01 feature set) is horizon-specific and does NOT carry. tenc_onehot/allfeat/interact trail the book but ns (|t|≤0.79), all with lower drawdowns than the book. Search landed in E27's good corner (nl 2–4, low lr/cs) on 60d too. Caveat: 470 periods at ~98% overlap → few effective windows, |t|<2 expected. |
| **E59** | `catboost_ktz` 60-day tune of E32's top-5-MMC datasets | `e59_catboost_ktz_tune_t60_crypto.sh` | The CatBoost sibling of E57 (rf) and E58 (lgb): the 60-day twin of E32's best catboost picks. E32 ranked the 9 catboost_ktz search-and-pick tunes by MMC on `target_binned_return_20`; E59 re-tunes the **five highest-MMC by catboost's own ranking** — fenccausal(44), allfeatpp(812), tenc(23), allfeat(788), tenc_onehot(763) — on the **60-day** label. (The set differs from the rf/lgb tunes' — each family's per-dataset MMC ordering differs; catboost keeps tenc where they take interact.) Only the target changes: same 12-cell KTZ grid (depth{2,4}×lr{.005,.01,.02}×rsm{.1,.2}; iters 2000, l2 10, min_data_in_leaf 100 pinned), same exhaustive `--n-iter 12` search, same rolling 130/130 / 5-fold / seed 42. `--embargo auto`→60 (forced by the label), thinning the inner CV to ~2 usable folds; 60-day runs score ~470 periods. CORR is **not** comparable to the 20-day E32 arms — read against the 60-day book (SACI_03 MMC 0.0788/pay 0.0930, CURUPIRA_05_60 MMC 0.0804/pay 0.0922). Tests whether catboost's 20-day MMC strength carries to the 60-day label. | 5 | registered |
| **E53** | Gaussian Process viability test — full book, reg + clf | `e53_gp_test_crypto.sh` | First GP on the board and the gate before the E54 KTZ sweep. One `gp` arm per (dataset × task) across all 9 book datasets × {regression (posterior mean on the 5-binned target), classification (GPC pos-class prob, target binarised at 0.5)} = 18 arms. A subsampled **exact** GP (`src/prediction/gp.py`): training rows capped (reg 6000, clf 1500) via `--model-options` (a GP is O(n³); a 130-date block is ~40k rows), kernel frozen, outer search tunes family + length-scale (+ alpha for reg). Light `--n-iter 12` at the deployable cadence (rolling 130/130, embargo 20, n_splits 5, seed 42); serial search × arm pool (E52 anti-loky-hang). Does the wrapper run on both tasks, and is reg or clf worth carrying forward? A viability probe, not a tuned model. | 18 | registered |
| **E54** | Gaussian Process hyperparameter sensitivity (Kalfa-Timmermann-van der Zwan) | `e54_gp_hyperparam_sensitivity_crypto.sh` | The 'Overhyped?' sweep (E17 MLP, E27/E49 boosters, E48 forest) now on a GP — the first kernel model on the board. One pinned walk-forward per cell of a 4×2×3×4 factorial: kernel {Matérn-1/2,-3/2,-5/2, RBF} × {iso, ARD} × `length_scale` {0.25,1.0,4.0} × `alpha` {1e-3,1e-2,1e-1,1.0} = 96 cells on raw 22 of crypto/v2.0, then marginals/fANOVA/corners/priors via `src.analysis.hyperparam_sensitivity_gp`. Kernel **frozen** (`optimizer=None`) so a pinned cell means what it says; `subsample` 6000 and `normalize_y` True pinned; `n_splits` 2 (pinned CV score unread); regression only (GPC has no `alpha`). Two-way claim: `length_scale × alpha` (smoothness vs noise — the GP analogue of lr×alpha / max_features×min_samples_leaf). The length-scale/alpha priors it reads off feed BOTH the `gp_ktz` search grid AND the GP's own kernel prior. | 96 | registered |
| **E60** | MLP **fit-60 / score-20**: the best-MMC MLP setups re-tuned on 60d | `e60_mlp_fit60_score20_crypto.sh` | The MLP sibling of E57 (rf) / E58 (lgb) / E59 (catboost), with one deliberate difference: those FIT AND SCORE on the 60-day label; E60 **fits on 60d but scores on the payable 20-day one** (the inverse of SACI_03, which fits 20 / scores 60). The five MLP setups that led MMC per feature set are re-tuned **from scratch** — a fresh `--n-iter 50` search, not the pinned 20-day config: `mlp`/fenccausal(44), `mlp`/tenc-onehot(763), `torch_mlp`/tenc(23), `mlp`/raw(22), `mlp_ktz2`/fenccausal (the E33-reprior wider grid). Each fits `target_binned_return_60` at embargo 60 (forced by the horizon, ~470 periods ≈ 8 effective windows → NW lag 60), then its predictions are **re-scored** against `target_binned_return_20` via `src.analysis.cross_target` — the same meta model every deployed slot is graded on. Deliverable = 20-day MMC/payout vs each dataset's fit-20 MLP already on disk (fenccausal 0.0825, tenc-onehot 0.0781, tenc 0.0775, raw 0.0746): does training on the smoother 60-day signal pay better on the 20-day label? Also reports native 60d metrics, comparable to E57-59's board. | 5 | registered |
| **E61** | Multi-symbol multivariate (SUR-style) ridge vs the stacked ridge | `e61_sur_ridge_crypto.sh` | The deployed linear slots fit **one shared β on the stacked panel** — every (symbol,date) row independent, errors i.i.d. (`ridge_glmnet`). `sur_ridge` (`src/prediction/sur_ridge.py`) keeps the shared β but **partials out k common cross-symbol factors** from the residuals — the feasible-GLS / SUR adjustment under a factor covariance — exploiting that coins' errors co-move within a date. `n_factors=0` is an **exact stacked-ridge control**, so the value is the searched k>0. The row's **symbol** is routed to `fit()` (`WANTS_SYMBOL`) with the date; neither is needed at predict (stays `x'β`). **36 arms** = {sur_ridge, ridge_glmnet} × {regression, classification} × 9 book datasets, one protocol (rolling 130/130, embargo 20, 5-fold, 50-draw, seed 42), so every sur_ridge arm pairs against its matched stacked ridge and reg pairs against clf. A result = sur_ridge beating ridge_glmnet on numerai_corr at paired |t|≥2; prior (E27/E30) is a null — modelling the error covariance sharpens β in MSE but need not move rank corr. The **n_factors the search keeps** is the diagnostic. | 36 | **done (null)** — 36 arms, 0 failed, 667 min. sur_ridge beats ridge_glmnet on numerai_corr in **0/18** cells at \|t\|≥2 (reg mean ΔCORR −0.0073, clf +0.0004). The search **keeps k>0 in ~80% of refits** — the cross-symbol factor is selected in CV but does not survive to out-of-sample rank corr (the E27/E30 pattern). On wide datasets sur nudges MMC/payout up (not significant) while **lowering** CORR. Regression ≫ classification for both models (~+0.06 CORR). The stacked pooled ridge is not improved by modelling the cross-symbol error covariance. |
| **E62** | SUR cross-symbol factor-GLS on trees (lgb/rf/catboost) vs stacked | `e62_sur_trees_crypto.sh` | E61/`sur_ridge` generalised to trees: `SurAugment` (`src/prediction/sur_ridge.py`) adds the k common cross-symbol factor scores as **extra features** at fit (the tree explains away common shocks) and predicts with them **zeroed**; `n_factors=0` = exact plain-tree control, factors from a cheap stage-1 ridge's residuals. **108 arms** = 3 families × 2 tasks × 9 datasets × {sur, stacked}, matched protocol (rolling 130/130, embargo 20, 5-fold, 50-draw, seed 42): `sur_lgb` vs `lgb_ktz`, `sur_rf` vs `rf_ktz`, `sur_catboost` vs `catboost_ktz`(reg)/`catboost`(clf). Result = sur beats stacked on numerai_corr at paired \|t\|≥2; prior (E61 null, E27/E30) is the factor is picked in CV but dies out-of-sample. | 108 | stopped 55/108 (host reboot 2026-09-26; sur_catboost clf crashed) — **null**: no pair clears the adjusted bar; catboost reg's CORR edge is search budget, not factors (see findings) |
| **E63** | Literature crypto momentum features vs the raw 22 | `e63_momentum_features_crypto.sh` | The 22 shipped features are rank-binned and hide the price path. `build_momentum` rebuilds **17 momentum features from spot prices** — from the crypto momentum literature (Liu & Tsyvinski 2021, Grobys & Sapkota 2019, Tzouvanas 2019, Li 2021, Liu & Tsyvinski & Wu 2022, Begušić 2019) plus **Tzotchev (2018) robust trend-following** at six crypto horizons (7/14/30/60/120/240d + composite) — puts them on Numerai's exact `{0,.25,.5,.75,1}` scale (`src/data_mani/binning.py`) and merges them **point-in-time** (row D uses spot closes ≤ D−1, the last completed bar known at submission; `attach_signals_asof`), building `data/crypto/v2.0-litmom`. **6 arms**, matched protocol (rolling 130/130, embargo 20, 5-fold, 50-draw, seed 42): {`lgb`, `ridge_glmnet`} × {base `v2.0`, mom `v2.0-litmom`, momonly `^feature_mom_`}. Result = base+mom beats matched base on numerai_corr at paired \|t\|≥2. New skills: `crypto-derived-features`, `crypto-feature-binning`, `crypto-price-data`. | 6 | registered |
| **E64** | KTZ-prior model bake-off: litmom vs base across 5 model classes | `e64_ktz_litmom_bakeoff_crypto.sh` | E63 asked "does litmom help?" for two models; E64 widens it to **five function classes, each tuned FROM SCRATCH under its KTZ (Kalfa–Timmermann–van der Zwan) priors**: a linear model (`ridge` — **no ktz variant**, its fixed log-alpha grid IS the evidenced search, the linear control), two boosters (`lgb_ktz` E27, `catboost_ktz` E29/E32), a bagged forest (`rf_ktz` E48) and a feedforward net (`mlp_ktz` E17). **10 arms** = 5 models × {base `crypto/v2.0` (22f), litmom `crypto/v2.0-litmom` (22+17 momentum)}, matched cadence (rolling 130/130, embargo auto=20, 5-fold, val_frac 0.3, seed 42, `target_binned_return_20`, metric corr). `n_iter` **per model** at each grid's evidenced budget {ridge 50, lgb_ktz 50, catboost_ktz 12, rf_ktz 40, mlp_ktz 80}, identical across a model's two arms so the **within-model** pairing is exact. Result = litmom beats matched base on numerai_corr at paired \|t\|≥2 (lag 20); the clean finding is **consistency** across function classes. A null across all five, with intervals, is equally a finding — E63's pre-run EDA put the incremental IC at ~0.015 (medium-horizon returns). | 10 | registered |
| **E65** | E64 re-run on the updated litmom (22+17) at uniform n_iter=50 | `e65_ktz_litmom_v2_bakeoff_crypto.sh` | **E64 re-run**, two changes. (1) **litmom grew** from 22+10 (32 cols) to **22+17 (39 cols)** — the original 10 momentum + a 7-feature trend block (`feature_mom_trend_7d/14d/30d/60d/120d/240d` + `feature_mom_trend_robust`, the Tzotchev robust trend). (2) the search budget is **uniform `n_iter=50`** for every model (E64 used per-model 50/50/12/40/80); `catboost_ktz`'s 12-cell grid clamps 50→12 (a full sweep). **10 arms** = 5 models × {base `crypto/v2.0` (22f), litmom `crypto/v2.0-litmom` (39f)}, same cadence as E64 (rolling 130/130, embargo auto=20, 5-fold, val_frac 0.3, seed 42, `target_binned_return_20`, metric corr); base arms re-run at 50 so the within-model pairing is exact. Result = litmom beats matched base at paired \|t\|≥2; also read **E65 vs E64** — did the added trend block change any model's verdict? (E64 on 22+10 was a null across all 5.) | 10 | registered |
| **E66** | Swap the base momentum proxies for the literature momentum block | `e66_momswap_litmom_crypto.sh` | E64/E65 found **adding** litmom is null-to-negative because the new momentum features are redundant with the base's **own** momentum proxies — `feature_momentum`, `rsi`, `sharpe_ratio`, `bollinger` (each 20d/60d; the new features load 0.5–0.81 on them). E66 asks the **substitution** question: drop those **8 base momentum proxies** and replace them with the **17 literature momentum** features. **10 arms** = 5 models × {base `crypto/v2.0` (raw 22), swap (`v2.0-litmom` minus the 8 proxies via `--feature-exclude` = 14 non-momentum base + 17 mom = **31f**)}, uniform `n_iter=50`, same cadence (rolling 130/130, embargo auto=20, 5-fold, seed 42, `target_binned_return_20`). Result = swap beats the raw-22 base at paired \|t\|≥2 — the literature block does the momentum job better than the crude proxies. Confounds remove-native with add-literature (inherent to a swap); prior is null-or-worse. | 10 | registered |
| **E67** | Old momentum proxies vs new literature momentum, momentum-only, from scratch | `e67_old_vs_new_momentum_crypto.sh` | The clean head-to-head E64–E66 kept confounded by the rest of the panel. Races the two momentum blocks **alone**, each tuned from scratch — **no other features**: `old` = the 8 base momentum proxies (`momentum`/`rsi`/`sharpe`/`bollinger` × 20d/60d), `new` = the 17 literature momentum (`feature_mom_*`), both selected from `crypto/v2.0-litmom` via `--feature-include`. **10 arms** = 5 models × {old (8f), new (17f)}, uniform `n_iter=50`, same cadence. Result = new beats old at paired \|t\|≥2; levels are toy (8–17f, not comparable to the book) — read the delta. **DONE**; ensemble `e67_new_ens` built + priced vs `cu_sa_ia_bo_02` (momentum-only tops ~0.10 CORR / 0.04 payout, ~⅓ of the deployed blend; no blend beats it). | 10 | done |
| **E68** | Cross-sectional momentum-regime block on v2.0-momswap | `e68_xsmom_momswap_crypto.sh` | Does a market-**regime** descriptor of momentum help on top of the momswap set (14 base + 17 momentum, 31f)? `build_xsmom` collapses the raw (un-binned) momentum cross-section into **3 day-level numbers per date** — the cross-sectional **mean, variance, skew** of the momentum distribution across coins (per-signal moment of each of the 17 raw signals → causal percentile → average → causal percentile), broadcast **point-in-time** (D−1) as `feature_xsmom_{mean,var,skew}` → `crypto/v2.0-momswap-xsmom` (34f). Because they are **identical across the cross-section** on a date, they carry ~zero standalone within-era `numerai_corr` and are **inert for ridge** (the negative control — a day-constant cannot move a within-date rank); only trees/MLP can use them via regime gating/interactions. **10 arms** = 5 models × {base `v2.0-momswap`, xsmom `v2.0-momswap-xsmom`}, tuned from scratch, uniform `n_iter=50`, same cadence (rolling 130/130, embargo auto=20, 5-fold, seed 42, `target_binned_return_20`). Result = xsmom beats base at paired \|t\|≥2 on a tree/MLP; prior is a null. | 10 | registered |
| **E69** | Macro / cross-asset information set on v2.0-momswap | `e69_macro_momswap_crypto.sh` | The **E68 sibling with a different information set** — E68 added a cross-sectional momentum *regime* block, E69 adds *external macro* state. On top of `v2.0-momswap` (31f), add the **18 shared macro/cross-asset features** (`feature_macro_*`: gold & silver returns 20d/60d, S&P returns 5d/20d/60d, **DXY dollar-index momentum 5d/20d/60d**, US Treasury yields y1/y2/y5/y10/y30, curve slopes 1s10s/2s10s/5s30s; `build_macro.py` → `crypto/v2.0-momswap-macro`, 49f), causal-percentile normalised across time, broadcast **point-in-time** (D−1). Like xsmom, macro is **identical across the cross-section** on a date → ~zero standalone within-era `numerai_corr` and **inert for ridge** (negative control); only trees/MLP can use it via regime gating/interactions. **10 arms** = 5 models × {base `v2.0-momswap`, macro `v2.0-momswap-macro`}, tuned from scratch, uniform `n_iter=50`, same cadence (rolling 130/130, embargo auto=20, 5-fold, seed 42, `target_binned_return_20`). Runs concurrently with E68. Result = macro beats base at paired \|t\|≥2 on a tree/MLP; prior is a null. | 10 | registered |
| **E70** | Engineered allfeat-style block on v2.0-momswap-macro | `e70_eng_momswap_macro_crypto.sh` | Does the `allfeat_pp` recipe help on top of momswap-macro? `build_momswap_eng.py` adds (retargeted to the swapped set, which dropped RSI/base-momentum): **tenc**(1, causal lag-20 symbol target-encoding), **onehot**(752 symbol dummies), **rollsel**(87 = rolling means {5,20,60} of the 17 momentum + 10 macro-return + 2 volatility; RSI absent so dropped), **spread**(12, term-structure diffs within each momentum/macro family), **interact**(24 = accel + adapted products with `feature_mom_ret_28d` standing in for the removed `feature_momentum`, **plus 12 momentum×macro-momentum** products = crypto trend gated by the macro regime) → `crypto/v2.0-momswap-macro-eng` (925f). **10 arms** = 5 models × {base `v2.0-momswap-macro` (49f), eng (925f)}, from scratch, uniform `n_iter=50`, same cadence. Onehot/tenc = symbol identity, mom×macro = regime gating — both help via **interactions** (trees/MLP). Result = eng beats base at paired \|t\|≥2; prior is a null. | 10 | registered |
| **E71** | Interactions-only block on v2.0-momswap-macro (the E70 follow-up) | `e71_xinteractions_momswap_macro_crypto.sh` | E70 dumped the full 925f allfeat block and it was null-to-negative (the 752 onehot dummies crushed ridge/MLP on CORR at \|t\|>2; boosters only a slight non-sig +MMC). E71 strips it to **just the 24-column interact block** (`build_momswap_eng.py --blocks interact` → `crypto/v2.0-momswap-macro-xinteractions`, 73f): 6 accel + 6 adapted products + **12 momentum×macro-momentum** products (crypto momentum gated by the macro regime — a per-symbol cross term that CAN move a within-date rank, so ridge is no longer a pure negative control). **10 arms** = 5 models × {base `v2.0-momswap-macro` (49f), xint (73f)}, from scratch, uniform `n_iter=50`, same cadence. Tests whether the boosters' E70 hint firms up once the dummy noise is removed. Result = xint beats base at paired \|t\|≥2; prior a tighter null. | 10 | registered |
| **E72** | Carver trend-following stack vs the raw 22 | `e72_carver_vs_raw_crypto.sh` | Bottom-line test of the whole engineering stack vs the original raw 22. `crypto/v2.0-momswap-macro-carver` (64f) = momentum swap (14 base + 17 literature momentum) + 18 macro/cross-asset (incl. DXY momentum) + **15 Robert Carver-style trend-following** features (`feature_tf_*`: EWMAC crossovers 2_8..32_128, breakouts 20/40/80/160, Bollinger-z 20/40/80, Clenow 60/90, new-high-20-in-5). Paired against `crypto/v2.0` (raw 22, base). **10 arms** = 5 models × {base v2.0, carver}, from scratch, uniform `n_iter=50`, same cadence. Whole-stack comparison (the deployment question 'is the engineered 64 better than the raw 22'), not a single-block ablation. Result = carver beats raw at paired \|t\|≥2; prior is a null. | 10 | registered |
| **E73** | litmom with base masked on momentum-missing rows (mommask), 3-way | `e73_litmom_mommask_crypto.sh` | `crypto/v2.0-litmom-mommask` = litmom (39f) with the 22 **base** features set to neutral 0.5 in ~17% of cells, **concentrated on momentum-missing rows** (P(base masked ∣ momentum missing)=0.996 vs 0.263 when present; ~11 of 22 base masked per masked row); the 17 momentum columns are identical to litmom. So on the ~22% no-spot rows that already lean entirely on the base block, ~half the base is also hidden (leakage-safe — masking only removes info). **5 arms** = 5 models on mommask, from scratch, uniform `n_iter=50` at **exactly E65's protocol**, compared **3-way** vs `crypto/v2.0` (`e65_*_base`) and `crypto/v2.0-litmom` (`e65_*_litmom`) reused from E65 (only the 5 mommask arms run). Prior: masking base info should not help numerai_corr; a null-or-worse (esp. vs litmom) expected, a significant drop quantifies how much the base carries the no-spot names. | 5 | registered |
| **E74** | Realised-volatility estimator block vs the raw 22 | `e74_vol_vs_raw_crypto.sh` | Does a proper volatility block help over the base's two crude volatility features? `crypto/v2.0-vol` (44f) = the raw 22 + **22 realised-volatility estimators** (`feature_vol_*`) from spot OHLCV over a hyperparameter grid — four estimators × their lookback: **squared-returns** realised variance (`feature_vol_sqret_{5,10,20,40,60,120}d`), **EWMA/RiskMetrics** variance (`feature_vol_ewma_lam{90,94,97,99}`), **Parkinson** high/low range (`feature_vol_park_{5,10,20,40,60,120}d`), **Yang–Zhang** full-OHLC (`feature_vol_yz_{5,10,20,40,60,120}d`, the only classical estimator robust to drift + opening gaps). Per-symbol, raw from spot, attached as-of ≤ D−1, rank-binned per date (`build_vol.py` → `volatility_features.py`, the first feature module that reads OHLC not close alone). Coverage ~82–89% of panel rows (no-spot memecoins neutral-filled). **10 arms** = 5 models × {base `v2.0` (22f), vol `v2.0-vol` (44f)}, from scratch, uniform `n_iter=50`, same cadence (rolling 130/130, embargo auto=20, 5-fold, seed 42, `target_binned_return_20`). Isolate the block with `--feature-include '^feature_vol_'`. Result = vol beats raw at paired \|t\|≥2; prior is a null, and the base already carries `feature_volatility_{20,60}d` so the block must add over those. | 10 | registered |
| **E75** | momswap-macro mommask, 3-way vs momswap-macro + v2.0 | `e75_momswap_macro_mommask_crypto.sh` | The momswap-macro analog of E73's mommask (E74 was taken by the vol study). `crypto/v2.0-momswap-macro-mommask` (49f) = momswap-macro with features masked to neutral 0.5 concentrated on momentum-missing rows (P(base masked∣mom missing)=0.961) — unlike E73 it masks **all groups**: base ~17%, momentum ~10%, macro ~34% of cells (real→0.5, info removed; 5.27M extra neutral cells), leakage-safe. **5 arms** on mommask, from scratch, `n_iter=50` at the shared protocol, compared **3-way** vs momswap-macro (`e70_*_base`) and v2.0 (`e72_*_base`), reused. Shared `mommask_core.sh`. Prior null-or-worse. | 5 | registered |
| **E76** | xinteractions mommask, 3-way vs xinteractions + v2.0 | `e76_xinteractions_mommask_crypto.sh` | `crypto/v2.0-momswap-macro-xinteractions-mommask` (219f) = the 219-feature xinteractions set (momswap-macro + 170 macro-mom×crypto-mom products) with features masked to neutral 0.5 concentrated on momentum-missing rows. **5 arms** from scratch, `n_iter=50`, compared **3-way** vs xinteractions (`e71_*_xint`) and v2.0 (`e72_*_base`), reused; `e71_ridge_xint` (missing from E71) run separately to complete the ridge cell. Shared `mommask_core.sh`. Prior null-or-worse. | 5 | registered |
| **E77** | carver mommask, 3-way vs carver + v2.0 | `e77_carver_mommask_crypto.sh` | `crypto/v2.0-momswap-macro-carver-mommask` (64f) = the carver set (momswap-macro + 15 Carver/Clenow trend-following) with features masked to neutral 0.5 concentrated on momentum-missing rows. **5 arms** from scratch, `n_iter=50`, compared **3-way** vs carver (`e72_*_carver`) and v2.0 (`e72_*_base`), reused. Shared `mommask_core.sh`. Prior null-or-worse. | 5 | registered |
| **E78** | vol mommask, 3-way vs vol + v2.0 | `e78_vol_mommask_crypto.sh` | The **volatility analog** of the mommask sweep, keyed on the vol block. `crypto/v2.0-vol-mommask` (44f) = the vol set (raw 22 + 22 `feature_vol_` estimators) with the **whole row set to neutral 0.5 wherever any of the 22 vol features is missing** (`build_mommask --gate created --require all --neutralize all`, the same recipe as E73/E75–E77 but gating on `feature_vol_`). **90,164/514,696 rows gated (17.5%)** — the no-spot names + rows with <120d history (the longest vol window; gates less than litmom-mommask's 34.4%, whose `feature_mom_ret_12_1` needs 360d). Masking only removes info (real→0.5), leakage-safe. **5 arms** from scratch, `n_iter=50` at the shared protocol, compared **3-way** vs vol (`e74_*_vol`) and v2.0 (`e72_*_base`), reused (E74 must be on disk for the vol cells). Shared `mommask_core.sh`. Prior null-or-worse; a drop quantifies how much the vol block carries the no-spot/young names. | 5 | registered |
| **E79** | xinteractions un-gated (fillmask == raw 219f), 3-way vs mommask + v2.0 | `e79_xinteractions_fillmask_crypto.sh` | The **one non-redundant fillmask**. `build_mommask --mode fill` replaces **only** the missing (NaN) cells with 0.5 and keeps present values — which is **exactly** what the loader (`fill_missing_features`, 0.5) already does, so a `-fillmask` dataset is **byte-identical AS LOADED to its raw parent** (verified for all 5). Four fillmask sets therefore reproduce E65/E70/E72/E74 exactly and are **not** re-run. xinteractions is the exception: `crypto/v2.0-momswap-macro-xinteractions` was rebuilt **73f→219f after E71 ran at 73f**, so no un-gated 219f run exists (E76 ran the 219f **gated** block but reused the 73f `e71_*_xint` as "parent" — a feature-count mismatch). E79 runs 5 models from scratch on `crypto/v2.0-momswap-macro-xinteractions-fillmask` (219f, == raw 219f as loaded), `n_iter=50` at the shared protocol, and reports a clean **3-way**: v2.0 (`e72_*_base`) / mommask (`e76_*_mommask`, gated) / fillmask (this, un-gated). Answers whether cell-level fill beats whole-row gating on the 219f block, and supplies the proper un-gated 219f baseline. Shared `mommask_core.sh` (`VARIANT=fillmask`). | 5 | registered |
| **E80** | inflation-expectations (breakeven) macro block vs v2.0 + momswap-macro | `e80_infl_momswap_macro_crypto.sh` | The direct sibling of **E69 (macro)**, keyed on inflation. `crypto/v2.0-momswap-macro-infl` (61f) = momswap-macro (49f) + **12 breakeven-inflation (BEI) features**: market-implied inflation expectations at 5y/10y/30y/5y5y-forward + their 20d/60d changes. `feature_macro_` prefix, **day-level broadcast** (median 1 distinct value/date, 0 NaN) — so like E69's macro it is **inert for ridge** on a within-date rank metric (negative control) and can help only via tree/MLP interactions. **5 arms** from scratch, `n_iter=50` at the shared protocol, reported **3-way**: vs raw v2.0 (`e72_*_base`, the requested baseline) and vs momswap-macro (`e70_*_base`, to isolate the 12 BEI features' marginal value), both reused. Shared `mommask_core.sh` (`VARIANT=infl`). Prior: ~zero standalone within-era IC; null-or-marginal like every macro/engineered block. | 5 | registered |
| **E81** | inflation block on raw baseline: raw+12 BEI (34f) vs raw v2.0 | `e81_inflonly_vs_raw_crypto.sh` | `crypto/v2.0-infl` (rebuilt 2026-09-29) = raw 22 + **12 breakeven features** (`feature_macro_bei_*` levels + 20d/60d changes) = **34f**, a clean **superset** of raw. The clean isolation of inflation's marginal value on the **deployed baseline** — the better-controlled sibling of E80 (which buried the same 12 BEI feats in the 61f momswap-macro stack). The BEI columns are **day-level** (inert for ridge on the within-date metric) but the raw 22 restore the cross-section (289 distinct vectors/date, **not degenerate**), so ridge ≈ raw and only trees/MLP can use inflation as a regime tilt. *(An earlier build was inflation-only/12f and within-date degenerate; rebuilt to append onto raw before this ran.)* **5 arms** from scratch, `n_iter=50`, reported 3-way vs raw v2.0 (`e72_*_base`) and momswap-macro-infl (`e80_*_infl`), reused. Shared `mommask_core.sh` (`VARIANT=inflraw`). Prior: null, like every macro/inflation block on this within-date metric. | 5 | registered |
| **E82** | kitchen-sink combo (raw+vol+infl+carver+ix, 753f) vs raw v2.0 | `e82_vol_infl_carver_ix_combo_crypto.sh` | Does stacking every individually-null engineered block finally beat raw? `crypto/v2.0-vol-infl-carver-ix-fillmask` (**753f**), a clean superset of raw: raw 22 + `feature_vol_` 22 (E74) + `feature_macro_bei_` 12 (inflation, E80/E81) + `feature_tf_` 15 (Carver, E72) + `feature_ix_` **682** (interactions — dominates the count). fillmask = missing cells pre-filled to 0.5 (== the loader's own fill; no non-fillmask twin, so this is the canonical form as loaded). **Not** within-date degenerate (300 distinct vectors/date). **5 arms** from scratch, `n_iter=50`, reported 3-way vs raw v2.0 (`e72_*_base`) and momswap-macro (`e70_*_base`, a mid-size stack for reference — not nested), reused. Shared `mommask_core.sh` (`VARIANT=combo`). Prior (E70 925f, E71/E79 219f xint): a huge interaction count **drowns ridge** (t<−2.7) and hurts MLP while trees stay ~flat; unlikely to beat raw. | 5 | registered |
| **E85** | combo + causal per-symbol means (1506f) vs combo (753f) + v2.0 | `e85_combo_fenccausal_crypto.sh` | Does a **causal per-symbol expanding-mean twin** of every kitchen-sink feature help over the kitchen sink? `crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal` (**1506f**), a clean superset of the E82 combo (753f): appends a `_symmean` column for **every** combo feature (raw 22→44, vol 22→44, bei 12→24, tf 15→30, ix 682→**1364**). Each `_symmean` = mean of that symbol's **own past** (a row never enters its own encoding) → leak-safe; the fenccausal idea behind IA_01 (best MLP slot), applied to the whole stack. Biggest feature set run to date. **5 arms** from scratch, `n_iter=50`, reported 3-way vs combo (`e82_*_combo`, the requested baseline) and v2.0 (`e72_*_base`), reused. Shared `mommask_core.sh` (`VARIANT=fenc`). *(Numbered E85 — a concurrent session took E83/E84 (TabPFN) mid-work; id-collision hazard, cf. E74.)* Prior: the combo already trailed raw; 1506f unlikely to beat it — watch the mlp arm. | 5 | registered |
| **E86** | diverse-primitives bake-off: 7 raw groups × 5 models from scratch, rank + ensembles | `e86_group_bakeoff_crypto.sh` | Tune every model class **from scratch on each primitive feature group** of the raw 22, rank the (model × group) cells, screen ensembles. (The referenced `g1_diverse_primitives.json` never reached disk, so the groups are the raw-22 primitive families that partition the 22 exactly: `momentum`(2), `volatility`(4), `rsi`(2), `sharpe`(2), `volume`(4), `size`(4), `price_ma`(4).) Each is a `--feature-include` regex on `crypto/v2.0` (one dataset → one clean common window). **7 groups × 5 models = 35 runs** at the shared protocol (rolling 130/130, embargo 20, 5-fold, seed 42, n_iter 50, `target_binned_return_20`), each paired (NW lag 20) vs the full raw 22 (`e72_*_base`); ensemble screen over the 35 arms + book via `report_html`. Groups run concurrently (~2.5h). Prior: no single primitive should match full raw — the questions are which model best exploits which primitive, and whether a cross-group/model blend beats the single best. | 35 | registered |
| **E88** | SPONGE-cluster bake-off on the 1506f causal-combo: 25 clusters × 5 models from scratch, rank + ensembles | `e88_sponge_fenccausal_bakeoff_crypto.py` | The E87 idea on the **bigger causal-combo**. Tune every model class **from scratch on each of the 25 SPONGE signed-spectral clusters** (Cucuringu et al. AISTATS 2019; signed Pearson affinity, k=25 by silhouette) of `crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal` (**1506f**), from the user's `data/.../sponge_feature_groups.json`. The 1506f twin of **E87** (753f, 10 clusters, **without** the causal per-symbol means): the universe here adds a leak-safe causal expanding per-symbol mean (`_symmean`) for every combo feature — the `fenccausal` idea behind `IA_01`, the best MLP slot — so each cluster mixes a primitive with its own symbol-level twin. The 25 clusters partition all 1506 feats **exactly** (0 missing, 0 overlap; sizes 330/308/198/165/119/66/48/46/45/44/23/22/22/22/16/8/8/4/3/3/2/1/1/1/1); each selected via a `--feature-include` anchored alternation of its exact feature names. **7 of the 25 are EXCLUDED** from the fitted roster — the pure day-level break-even-inflation clusters (`sponge_02/03/06/14/19/21/24`, 12 raw `feature_macro_bei_*` feats): a BEI series is one macro value per date broadcast to every coin, so those clusters are **constant within each date** → within-date `numerai_corr` is undefined (ridge/lgb/catboost/rf can't score a candidate; mlp fits only init noise ≈ 0). The runner detects this **data-drivenly** (within-date `nunique` on a ~30-date sample) and **names** the exclusion in the report; it reproduces the scout failures exactly and matches the E80/E81 BEI-inert prior. Leaves **18 active clusters × 5 models = 90 runs** at the shared protocol (rolling 130/130, embargo 20, 5-fold, seed 42, n_iter 50, `target_binned_return_20`) — identical to E85/E86/E87 so every cell is comparable to the book. Each cell paired (NW lag 20) vs the FULL 1506f combo (`e85_*_fenc`, reused); ensemble screen over the 90 `e88_*` cells + book via `report_html`. Clusters run concurrently (`CLUSTER_JOBS=5`). Prior: no single cluster beats the full combo (which itself trails raw) — the questions are which model best exploits which signed cluster, and whether a cross-cluster/model blend diversifies. | 90 | **done** (90/90, 21 h, 0 failures; 7 BEI clusters excluded as within-date degenerate, see runner). **(1) The signal concentrates in the volatility cluster.** `sponge_09` (48 f, `feature_vol_*` + base) reproduces the FULL 1506 f combo across every model — paired NW vs `e85_<m>_fenc`: ridge payout +0.0136 (t +0.65; CORR +0.040 t +1.76), lgb +0.0057 (t +0.53), catboost +0.0006 (t +0.06), rf −0.0026 (t −0.32), mlp +0.0164 (t +0.90; **CORR +0.046 t +2.32**). The 16 f volume/price cluster `sponge_07` nearly matches too. The ~1300 `feature_ix_` interaction + `_symmean` features add nothing standalone and *dilute* ridge/mlp (full-combo mlp 0.044 < vol-cluster mlp 0.061). **(2) Model × cluster:** rf/lgb/catboost extract the big signed-interaction clusters (`sponge_05` 330 f, `_01` 308 f) where ridge & mlp collapse to ≈ 0; on the compact vol/volume/price clusters all five agree. The pure-`ix` clusters (`sponge_16`/`20`/`17`) are ≈ null. **(3) No single cell beats the deployed book:** best cell `rf/sponge_07` payout 0.0709 (446-period common window) vs `SACI_06` 0.0917 and `CU_SA_IA_BO_01` 0.107 — a feature-attribution result, not a new slot, consistent with the E82/E85 prior (the combo trails raw). **(4) A cluster-blend beats the kitchen sink.** A prediction-level rank-average of 5 rf cluster-cells (`sponge_07+09+05+01+11`) pays **0.0804** (own 521-window) = **+0.0139 vs the full combo `e85_rf_ktz_fenc` (t +2.09), MMC +0.0126 (t +2.13)** — selecting-and-blending a few signed clusters beats dumping all 1506 f into one model; on the 446 book-window it reaches 0.0895 ≈ `SACI_06` (dpay −0.002, t −0.17). Notably the payout-SERIES screen MISSED this (0/7140 blends beat their best member at \|t\|>2) — a screen-vs-pricing divergence (the blend's gain is MMC, which the series-average screen cannot see). Board: `results/reports/crypto/e88_sponge_board.html`; blends priced as `rf_ktz/e88_rf_blend5` etc. |
| **E92** | BTC-beta block vs raw: `crypto/v2.0-beta` (raw 22 + 3 BTC-beta) vs `v2.0`, 5 models from scratch | `e92_beta_vs_raw_crypto.sh` | Does a coin's rolling market **beta to BTC** help the raw panel? `crypto/v2.0-beta` is a **strict superset** of `crypto/v2.0` — the raw 22 plus exactly 3 new features `feature_beta_btc_30d/_60d/_90d` (rolling beta to Bitcoin over 30/60/90 d). A clean leave-the-rest-fixed ablation: the only difference between an `e92_<m>_beta` arm and its reused raw baseline `e72_<m>_base` is those 3 columns. **5 models × 1 dataset = 5 runs** tuned from scratch at the shared protocol (rolling 130/130, embargo 20, 5-fold, seed 42, n_iter 50, `target_binned_return_20`, era 1) — identical to `e72_*_base`, so each cell pairs validly (NW lag 20) vs its raw twin. Raw side **reused** (`e72_*_base`), not re-run. Prior: BTC-beta is a market/regime signal with ~zero standalone within-date IC (like the macro/inflation blocks E69/E80/E81), so expect null-to-marginal on ridge, with at most a small tree/MLP interaction lift. | 5 | **done** (5/5, 72 min, 0 failures). **Adding BTC-beta HURTS — it does not help any model.** Paired beta − raw (NW lag 20, ~521 common periods), payout Δ: ridge **−0.0117 (t −3.63)**, catboost **−0.0134 (t −3.23)**, lgb **−0.0127 (t −2.08)** — all HURT at \|t\|>2; rf −0.0080 (t −1.97, MMC Δ t −2.01) borderline-negative; mlp −0.0048 (t −0.47) null. **Every model's point estimate is negative; none positive.** CORR and MMC move the same way (ridge CORR −0.0145 t −4.52, catboost CORR −0.0177 t −4.32). Even under a 5-test Bonferroni bar (~\|t\|2.6) ridge and catboost stay significant. So `crypto/v2.0-beta` is **worse** than `crypto/v2.0`: the 3 `feature_beta_btc_{30,60,90}d` columns carry ~no within-date cross-sectional signal (beta-to-BTC is near-common across coins on a date) yet add dimensions the models chase, diluting the strong raw panel — hits the linear model hardest (ridge −3.63). **Verdict: do not adopt v2.0-beta; keep v2.0.** Report `logs/e92_report.txt`; arms `e92_*_beta` vs reused `e72_*_base`. |
| **E95** | coint+rev+basis stack vs raw: `crypto/v2.0-coint-rev-basis-fillmask` (50f) vs `v2.0`, 5 models from scratch | `e95_coint_rev_basis_vs_raw_crypto.sh` | Does the **combined** stat-arb / microstructure stack beat raw where each block alone did not? `crypto/v2.0-coint-rev-basis-fillmask` is a **strict superset** of `v2.0` — raw 22 + **28 features** in four families: **coint** (4: ADF-120d, half-life-120d, residual-z 60/120d), **rev** (14: short-term reversal — rsi/stoch/cci/amihud/ivol/max1/max5/skew/str_{1,7,14}d/dist_ma_{20,50}d/bbz_20d), **basis** (6: futures-spot basis pct/ma/chg/z) and **funding** (4: perp funding rate/ma/z). `fillmask` = missing cells pre-filled to 0.5 (== the loader's own fill), so as loaded it's a clean raw+28 superset. This is the **combination** of the individually-tested blocks E91 (coint), E94 (basis/funding) and E90 (reversal). **5 models × 1 dataset = 5 runs** from scratch at the shared protocol (rolling 130/130, embargo 20, 5-fold, seed 42, n_iter 50, `target_binned_return_20`, era 1) — identical to `e72_*_base`, so each pairs validly (NW lag 20) vs its raw twin. Raw side **reused** (`e72_*_base`), not re-run. Prior: each constituent was null-or-marginal; the open question is whether stacking them clears the bar, or — like the wide engineered combos (E82/E85) and the BTC-beta block (E92, which HURT at \|t\|>2) — a 28-column add-on just dilutes the raw panel. | 5 | **done** (5/5, 88 min, 0 failures). **The combined coint+rev+basis stack does not help raw — it hurts or is null, never positive.** Paired crb − raw (NW lag 20, ~521 periods), payout Δ: ridge **−0.0136 (t −3.15)**, catboost **−0.0077 (t −2.86)**, mlp **−0.0114 (t −2.28)** — all HURT at \|t\|>2; lgb −0.0052 (t −1.68) and rf −0.0043 (t −1.31) null. **Every point estimate negative; none positive.** Nuance: for the trees CORR is *flat* (lgb +0.0004 t +0.13, rf +0.0009 t +0.27) while the drag is in MMC (lgb −1.85, rf −1.49) — the 28 stat-arb/microstructure cols don't change tree ranking but shave contribution over the meta model; ridge/mlp are hurt outright. Confirms the width-dilution pattern (E82/E85/E92): stacking the individually-null blocks (coint E91 + reversal E90 + basis/funding E94) does not rescue them. **Verdict: keep v2.0.** Report `logs/e95_report.txt`; arms `e95_*_crb` vs reused `e72_*_base`. |
| **E96** | coint+rev+basis+mom+carver stack vs raw: `crypto/v2.0-coint-rev-basis-mom-carver-fillmask` (82f) vs `v2.0`, 5 models from scratch | `e96_coint_rev_basis_mom_carver_vs_raw_crypto.sh` | The widest stat-arb + trend add-on. `crypto/v2.0-coint-rev-basis-mom-carver-fillmask` is a **strict superset** of `v2.0` — raw 22 + **60 features** in six families: **coint** (4), **rev** (14), **basis** (6), **funding** (4) [= the E95 28] **+ mom** (17, literature momentum incl. Tzotchev robust trend; cf. E63/E66/E67) **+ tf** (15, Carver/Clenow trend-following; cf. E72). `fillmask` == loader's own 0.5 fill → clean raw+60 superset; only those 60 columns differ from `e72_<m>_base`. **5 models × 1 dataset = 5 runs** from scratch at the shared protocol (rolling 130/130, embargo 20, 5-fold, seed 42, n_iter 50, `target_binned_return_20`, era 1) — identical to `e72_*_base`, so each pairs validly (NW lag 20) vs its raw twin. Raw side **reused** (`e72_*_base`). Prior: each constituent block was null-or-marginal vs raw (E90/E91/E94/E66/E67/E72); a 60-column add-on is squarely in the width where ridge gets diluted (E82/E85) and E92/E95 found null-to-negative — the question is whether momentum + trend rescue it on the tree/MLP side or sink it further. | 5 | **done** (5/5, 93 min, 0 failures). **Momentum + Carver do NOT rescue it — the widest add-on hurts or is null, never positive.** Paired crbmc − raw (NW lag 20, ~521 periods), payout Δ: ridge **−0.0180 (t −2.92)**, catboost **−0.0068 (t −2.13)** HURT at \|t\|>2; lgb −0.0066 (t −1.56), rf −0.0074 (t −1.82), mlp −0.0123 (t −1.74) null. **Every point estimate negative; none positive.** vs the 28-feat E95 stack the drag is the SAME or slightly worse — ridge degrades further (E95 −0.0136 → E96 **−0.0180**, the worst linear dilution of any add-on tested), the rest ≈ unchanged. So the extra 32 columns (mom 17 + tf 15), each individually null vs raw (E66/E67/E72), add nothing on top of the already-diluting coint/rev/basis stack. Confirms the pattern across E92/E95/E96 (and E82/E85): **no engineered raw add-on beats raw v2.0 — width dilutes, ridge worst.** **Verdict: keep v2.0.** Report `logs/e96_report.txt`; arms `e96_*_crbmc` vs reused `e72_*_base`. |
| **E104** | Day-indexed minibatches vs row-shuffled minibatches, on the sklearn-init torch MLP | `e104_daybatch_torch_mlp_crypto.py` | Does it matter **how a training window is cut into minibatches**? One pinned model — the from-scratch torch MLP with sklearn's exact initial weights that tracks `mlp/e33_fenc_a64_32_16_identity_lre3_al100` (identity `(64,32,16)`, alpha 100, fenccausal 44f, rolling 130/130) — and only the batch scheme varies: `row` shuffles the ROWS and slices 1024 (baseline); `day3` shuffles the DAYS and a batch is every stock of 3 shuffled days (~900 rows, size-matched); `day1` is one day per batch. A batch of whole days has no fixed row count (241..300 stocks a day across the panel), so the loss gives each **day** 1/k and the L2 term — which sklearn divides by rows-in-batch — is pinned to alpha/1024. Controls `rowfix`, `day3lit`, `day1lit` separate "batches are days" from "the penalty moved". 10 seeds per arm because this arm is numerically degenerate; the test is paired NW lag 20 on the seed-mean per-period delta. Two more controls, `row900` and `row300`, are row batches at the day arms' own size. A second pass takes `row`, `row900` and `day3` to 100 seeds for boxplots. | 350 | done, 350/350 — composition is a null at 10 and at 100 seeds; batch size is not (see findings) |
| **E105** | Learning-rate schedulers on the day3 batch scheme (torch MLP, sklearn init) | `e105_lr_schedulers_daybatch_crypto.py` | E104's `day3` arm unchanged, with **only the learning rate over training varied**. `const` (1e-3 throughout) is the baseline, gated bit for bit against `e104_day3`. Seven of torch's own schedulers, each peaking at 1e-3 over a 60-epoch horizon: `StepLR`, `ExponentialLR`, `CosineAnnealingLR`, warm-up + cosine, cosine warm restarts, `OneCycleLR`, `ReduceLROnPlateau`. Every schedule also lowers the *average* rate, so two controls hold it constant at 5e-4 and 2.5e-4 to separate the schedule's shape from "less learning rate". 100 seeds per arm (42–141), paired by seed against `const`. | 1000 | done, 1000/1000, 37 + 35 min — five annealing schedulers beat the baseline and a plain lower rate; `OneCycleLR` best (see findings) |
| **E106** | Residualise the −CORR network on the meta model (no training) | `e106_meta_residualise_crypto.py` | Does moving the −CORR network's predictions off the meta model buy payout? Per day on the 547 meta days, `rank(gauss(rank(p)) − s·β·gauss(rank(m)))`, the space MMC is computed in, with s swept 0→1 in 21 steps. The same-day meta model is an **oracle** (it is published after the round closes); four deployable arms residualise on the latest meta model ≥1/5/20/30 days old. Scored against the same-day meta model, paired against s = 0 of the same seed. Seeds 42 and 43. | 202 | done, 202/202, ~3 min — payout rises and survives a 5-day-old meta model (+0.0069, t +2.9; seed 43 t +2.8); dead by 30 days; costs drawdown above s ≈ 0.5 (see findings) |
| **E107** | Residualise every model on a 5-day-old meta model, s = 0.5 (no training) | `e107_residualise_everything_crypto.py` | Do all models converge to the same payout once residualised? E106's deployable transform (s = 0.5, latest meta model ≥ 5 days old) applied to the walk-forward ridge, the MSE and −CORR networks (seeds 42/43) and every 20-day slot in the registry, on the 470 days they all share. Converge = the cross-model spread collapses and model pairs stop differing at \|t\| ≥ 2 beyond what two seeds of one network show. If so, model choice is settled and the work moves to the residualisation / portfolio layer. | 38 | done, 38/38, ~2 min on deepfour — **no convergence**: every model gains (+0.003 to +0.010, all t ≥ 2.2) but the ranking holds (ρ 0.98) and the spread widens (see findings) |
| **E108** | All momentum/trend families in one dataset vs raw | `e108_allmom_vs_raw_crypto.sh` | First dataset to hold **every** momentum/trend family at once. `crypto/v2.0-allmom-fillmask` (57f), assembled with `build_combo` (union of already-built, already-binned blocks onto the raw v2.0 spine — no recomputation, leak-safe): raw 22 (incl the 2 base `feature_momentum_`) + `feature_mom_` 17 (litmom: cumret horizons, 12-1, MAX, vol-scaled, Tzotchev robust trend) + `feature_xsmom_` 3 (cross-sectional mean/var/skew, day-level) + `feature_tf_` 15 (Carver/Clenow: EWMAC, breakout, Bollinger-z, Clenow). Coverage mom 64.9% / xsmom 94.6% / tf 78.7%. **5 models from scratch**, `n_iter=50`, 3-way vs raw v2.0 (`e72_*_base`) and the vol block (`e74_*_vol`); report carries `calmar`. Prior: momentum null-to-negative vs raw (E64–E67); pooling every variant lets a tree pick among them, but the raw 22 already absorb most of it. | 5 | running |
| **E110** | All-momentum vs raw on the 60-day target | `e110_allmom_t60_crypto.sh` | The 60-day twin of E108. Two concurrent lanes, 5 models each tuned from scratch on `target_binned_return_60` (embargo auto → 60): **raw60** on `crypto/v2.0` — the matched 60-day baseline, which did not exist for 4 of the 5 model classes (pairing a 60-day arm against a 20-day one would rank the label) — and **allmom60** on `crypto/v2.0-allmom-fillmask` (57f). Report: raw60/allmom60 rows with calmar, allmom60 vs raw60 paired (NW lag 20). 60-day numbers are not comparable to 20-day tables. Feeds E111. | 10 | running |
| **E111** | Meta residualisation + ensembles of the all-momentum models | `e111_allmom_resid_ensembles_crypto.py` | No training. E107's deployable transform (s = 0.5, meta model ≥ 5 days old; imported, gate s = 0 ≡ repo scorer to 1e-9) on allmom20 (E108), raw20 (E72), allmom60 + raw60 (E110) and the book `CU_SA_IA_BO_02`; 60-day forecasts scored in-target **and** on the payable 20-day label. Then every 2/3/4-member rank-average of the 10 residualised allmom models on the 20-day label, paired vs best member and vs the residualised book; then book + best ensemble at 10/20/30%. Auto-launched when E110 finishes. | 21 | queued (20d validation done) |
| **E109** | Foundation-model representations as features, on allfeat_pp / fenccausal / raw | `e109_fmrep_crypto.sh` | Builds the 9 representations of `experiments/features-foundation.md` **causally** and merges them as one 41-column `feature_fm_` block onto three bases: `crypto/v2.0-fmrep` (raw 22+41), `crypto/v2.0-fenccausal-fmrep` (44+41), `crypto/v2.0-tenc-onehot-rollsel-spread-interact-fmrep` (allfeat_pp 812+41). Block: TabPFN v2 on fenccausal (rows 1-5: mean/median/mode, std/IQR/skew, 5 bin probabilities, estimator disagreement, 8 embedding PCs), TabPFN on one block alone (row 6: tenc/interact/size/price/raw), 30- and 390-date context variants (row 7), TimesFM and Chronos-2 on each coin's log close <= D-1 (row 8: 20-day q50/spread/skew) and Chronos-2 encoder embedding (row 9: 8 PCs). TabPFN context = the 130 dates ending 20 before each 20-date refresh block (labels resolved, no row in its own context); embedding PCA bases fitted once on a burn-in that ends before the first training date; every column rank-binned per date. Builders `src/data_mani/{foundation_repr,foundation_repr_ts,build_fmrep}.py`, tests `test_foundation_repr.py`. **30 arms** = 3 bases x 5 models x {with, without}; *without* is the SAME dataset under `--feature-exclude '^feature_fm_'`, so the block is the only difference. Shared protocol (rolling 130/130, embargo 20, 5-fold, n_iter 50, seed 42). All three on the 1,701-date spine to 2026-07-08 (raw v2.0 cut there: the TabPFN inputs end there), 510 scored periods. Prior (pre-run EDA): TabPFN means are ~84% a per-date linear function of the 44 inputs but add partial IC +0.04 (t +2.6..3.6) over the fenccausal ridge's own prediction; size/price/onehot heads and the TS models on the LABEL were null. Not live-wired. **DONE: the block helps no model on any base; payout falls in 15/15 pairs, at \|t\|≥2 for ridge (raw, fenccausal) and the raw MLP.** | 30 | done |
| **E112** | Symbol encodings on top of all-momentum | `e112_allmom_encodings_crypto.sh` | Which per-symbol encoding adds to `crypto/v2.0-allmom-fillmask` (57f, E108)? Four supersets, each differing from the base **only** by the encoding: `-fenccausal` (114f, +57 `_symmean`: causal expanding mean of each feature over the symbol's strictly earlier dates, shrunk m=50), `-tenc` (58f, +`feature_sym_tenc`: mean of the symbol's own 20d target over resolved dates ≤ t−20, shrunk m=50), `-roll` (228f, +171 `_rm{5,20,60}`: per-symbol trailing means incl. today), `-all` (286f, all three). 4 datasets × 5 models from scratch at the shared protocol = **20 arms**; base **reused** (`e108_<m>_allmom`), each arm paired (NW lag 20) vs its model's allmom twin; report carries `calmar`. Four lanes concurrently via `mommask_core.sh`. Prior: fenccausal is what makes IA_01 work on raw; tenc/rolling rarely moved trees on raw; 286f risks ridge dilution (E82/E85). **Done 2026-10-21 (142 min, 0 failed; 496 common periods, single seed).** A leak audit found NO look-ahead: each dataset = the repo builder at its defaults. Minor caveats: the priors are full-sample means; lag 20 is about 2 calendar days short (the calendar-correct tenc has *higher* IC); the encodings key on symbol, not ucid (31 reused tickers, ~1.6% of rows). **Only `tenc` helps** — payout lgb +0.0071 (t +2.88) and catboost +0.0059 (t +2.75), carried by MMC (t +3.03 / +2.89); ridge +0.0012, mlp +0.0060 (t +1.36), rf −0.0011. Calmar improves for **all 5** (lgb .038→.062, cat .044→.062, rf .040→.053) through a shallower dd(pay). `fenc` lowers CORR in 4/5 (catboost t −2.54), so the IA_01 effect does not transfer to the 57f momentum base. `roll` never helps (rf CORR t −2.05; mlp payout −0.0245, t −2.46). `all` is flat: the 228 extra columns drown tenc's gain. Best arm (rf_fenc 0.0802 / lgb_tenc 0.0800 on 435 days) is still below the book's 0.1047 (t −1.34). Report `logs/e112_report.txt`, `results/reports/crypto/e112_encodings.html`. **Blend addendum (2026-10-22, `e112_blend_with_slots.py`; 435 common days; gate: both slots reproduce their registry payout to 1e-5):** lgb+tenc, cat+tenc, their 50/50 mix and mlp+roll were each added at 10–50% to `CU_SA_IA_BO_02` and to `CU_SA_IA_BO_MA_R_01`, as a plain or a residualised leg. **Calmar falls in every one of the 80 blends.** The best payout is RESID(lgb+tenc) at 20% into `CU_SA_IA_BO_02`, +0.0012 (t +0.36, noise). Every blend into `MA_R_01` lowers payout. A plain tenc leg raises CORR (+0.0037, t +2.17 at 10% into `MA_R_01`) but gives up more MMC than it gains. Not deployed. | 20 | done |
| **E113** | Row ablation of E109's `feature_fm_` block on the fenccausal ridge | `e113_fmrep_ablation_crypto.sh` | E109's whole 41-column block cost the fenccausal ridge payout −0.0170 (t −2.35); which part does it? Ridge on `crypto/v2.0-fenccausal-fmrep`, the 44 base features always kept, ONE sub-block added per arm via `--feature-exclude '^feature_fm_(?!<sub-block>)'`: row 1 TabPFN point predictions (3), rows 1–4 (12), rows 2–4 alone (9 — `features-foundation.md`'s proposed first study), row 5 TabPFN embedding PCs (8), rows 6–7 per-block heads + context variants (7), rows 8–9 TimesFM/Chronos-2 on spot (14). Shared protocol; each arm paired (NW lag 20, 510 periods) vs `e109_ridge_fenc` (no block) and `e109_ridge_fenc_fm` (whole block), both reused. Runs on deepfive's CPUs (deepsix is busy with E112). **DONE: the harm is the TimesFM/Chronos-2 block (payout −0.0105, t −2.63); TabPFN rows 1–4 are payout-neutral.** | 6 | done |
| **E114** | 60-day twins of the E112 picks + 3/4-member ensembles with the book | `e114_selected_t60_ensembles_crypto.sh` | Fits the three E112 picks on `target_binned_return_60` (embargo auto → 60), same datasets and protocol: lgb_ktz and catboost_ktz on `-tenc` (58f), mlp_ktz on `-roll` (228f) → `e114_<m>_{tenc,roll}60`, paired in-target vs E110's `allmom60` twins. Then, **per model**, every 3- and 4-member equal-weight rank average of six items: plain@20d, RESID@20d, plain@60d, RESID@60d (E107 transform, s = 0.5, meta ≥ 5 days old), `CU_SA_IA_BO_02`, `CU_SA_IA_BO_MA_R_01`. That is 35 combos × 3 models, scored on the payable 20-day label over one common window, paired (NW lag 20) vs each slot alone and vs the best member, with calmar. **Done 2026-10-22 (fits 11 min; gate: both slots reproduce their registry payout to 1e-5; 435 common days).** 60-day twins, in-target vs allmom60: lgb+tenc +0.0052 (t +1.12), cat+tenc +0.0062 (t +1.31), mlp+roll −0.0141 (t −1.46). On the payable 20-day label they are weak (0.048–0.057). **No ensemble of the 105 beats `CU_SA_IA_BO_MA_R_01` on payout (0.1200) or either slot on calmar (0.103 / 0.086).** The best is RESID20(lgb+tenc) + BO_02 + MA_R_01 at 0.1115, which is −0.0085 vs MA_R_01 (t −2.04) and calmar 0.078. Every combo is below MA_R_01 at t −2.0 to −4.0. Report `logs/e114_report.txt`, `results/reports/crypto/e114/ensembles.csv`. | 3 | done |
| **E115** | Ridge λ by prequential loss vs by time-series CV (fenccausal, monthly refit) | `e115_ridge_prequential_crypto.py` | Should ridge's λ be chosen on the forecasts the model has ACTUALLY made out of sample, or on folds cut from the current training window? Ridge on `crypto/v2.0-fenccausal` (44f), rolling 130, refit every 22 periods (one month), embargo 20, 510 scored periods. One FIXED λ grid (51 log-spaced, 1e-3..1e7, sklearn scale, bracketing the ~0.02..2e4 X'X spectrum of a 130-period window), every λ fitted at every refit (one SVD, `src/prediction/lambda_selection.py`), and two rules pick which λ predicts the block: **prequential** = argmin of each λ's mean per-period loss over all its own out-of-sample forecasts resolved by that refit (position < train_end; 46 warm-up blocks from 2020-08 give it ~1,000 periods of history before the first scored block); **tscv** = argmin of mean held-out loss over 5 purged folds inside the training window, the pipeline's own rule (the tscv/corr arm reproduces `hyper_params_search` on the same grid prediction for prediction, `src/tests/test_lambda_selection.py`). Each rule with loss = −CORR, = MSE and (added 2026-10-23) = −payout, i.e. −(0.1·CORR + MMC) against the real meta model, so 6 arms from one pass that share every fit and differ only in the λ each block used → `ridge/e115_{preq,tscv}_{corr,mse,payout}`. The meta model starts 2024-05-07, 17 periods before the first scored block's training ends, so early payout picks stand on thin evidence. A block with none takes that rule's CORR pick, and the report counts those blocks. Report: preq − tscv per loss and corr − mse per rule, paired NW lag 20 on CORR/MMC/payout, with calmar and burn; λ path per arm (median, spread, switches, grid-edge hits); and the ex-post FIXED-λ CORR curve, which says whether any rule CAN matter. `e109_ridge_fenc` (130/130, fixed 1e-4..1e2 grid) is printed as context only, since cadence and grid both differ. **DONE 2026-10-22 (15 min): null.** Prequential − TSCV payout −0.0031 (t −0.59) with −CORR, +0.0004 (t +0.12) with MSE. Both rules land within 0.002 CORR of the best fixed λ in hindsight (0.1373 at λ ≈ 4e3, against 0.1046 at least squares). Prequential holds λ ≈ 2.5e3 almost throughout; TSCV switches 20 times in 24 blocks. See findings below. Payout-utility arms added 2026-10-23: null under prequential (+0.0010, t +0.21, lowest dd(pay) 1.43) and HARMFUL under TSCV (−0.0028, t −2.09), because they shrink harder on thin meta-model history. | 6 | done |
| **E116** | Ridge refit cadence × λ rule: every period / 1 / 3 / 6 months (fenccausal, live lag) | `e116_ridge_lambda_cadence_crypto.py` | E115 refit monthly and trailed the 6-monthly house ridge by ~0.02 payout, but cadence, λ grid and selection rule all differed at once. Here the grid (E115's 51 λ, 1e-3..1e7) and the rule are held fixed and ONLY the refit cadence moves: r1 (every period, i.e. every live round — the fastest a live round can refit, since `train.parquet` gains one resolved date per round), r22 (1 month), r65 (3 months), r130 (6 months, `registry.FIT`'s live cadence). This is done for three E115 rules — tscv_corr, preq_payout, tscv_mse — giving 12 arms `ridge/e116_{tscv_corr,preq_payout,tscv_mse}_r{1,22,65,130}`. **Embargo 21, not 20:** live, `train.parquet` ends 22 periods before the round's date (45 of 52 recorded rounds; the rest had stale data), and the walk-forward trains through start − embargo − 1, so 21 reproduces the live gap. Embargo 20 is one period optimistic, which a refit every period would apply to every day. The 510 scored dates are unchanged. Report: per arm each cadence paired against r130, the rules against tscv_corr at each cadence, an embargo check (r22 against E115's monthly arm at embargo 20), Newey-West lag 20 with calmar and burn; the λ path per arm and cadence; and the ex-post fixed-λ oracle per cadence. Prior: the early raw-ridge cadence sweep's inverted U (monthly 0.127 < quarterly 0.145 < six-monthly 0.153 > yearly 0.144, quoted in `src/live/refit.py`). **DONE 2026-10-23: slower is better.** Payout rises r1 < r22 < r65 < r130 for all three rules, and the ex-post best fixed λ follows the same order, so it is the cadence, not λ. Every period costs 0.027–0.030 payout against 6-monthly (t −2.85 to −3.13); monthly costs 0.018–0.023 (t −1.9 to −3.0); quarterly ties. The rule is null at every cadence; embargo 21 vs 20 is null. At r130 the arms match `e109_ridge_fenc`, so E115's gap was all cadence. See findings below. | 12 | done |
| **E117** | Training window of the best single ridge, 1 week to 6 months | `e117_train_window_crypto.sh` | ridge_glmnet on allfeat_pp minus tenc (w6r6_lobo_tenc, 811 feat), only the rolling window varied: 5/10/15/20/22/65/130 periods, λ searched from scratch every refit. Windows ≤22 periods cannot hold a purged fold, so they use the new `--tune-window 130`: search on the 130 periods ending at the same date with folds that train on the window's length, then refit on the window alone. Cadence held at 130 — how much history, not freshness (E116). FINDING: payout rises monotonically with history; every window under 6 months is significantly worse (t −2.46 to −3.21 on 510 dates), and on short windows the λ search lands on the top of the path on most refits. See findings below. | 7 | done |
| **E118** | 34 causal model-representation blocks, one at a time, on the best ridge / LightGBM / CatBoost | `e118_fmx_crypto.sh` | Adds each of 34 representation blocks ALONE to the payout-best clean standalone run of three families (510 periods, 1,701-date spine): `ridge_glmnet/w6r6_lobo_tenc` (allfeat_pp − tenc, 811f, 0.08779), `lgb_ktz/e29_tenc_onehot` (tenc_onehot, 763f, 0.08999), `catboost_ktz/e109_catboost_ktz_fenc` (fenccausal, 44f, 0.09303); baselines reused. "Clean" = regression, era-step 1, 20-day-scored, on a data version some live slot was backtested on; E116's custom-λ ridge (0.0900) is excluded as not reproducible through `run_forecast`. Blocks `feature_xNN_`: TabPFN on fenccausal x01–x05, x07–x20 (point, distribution, disagreement, embedding, heads, attention familiarity, rank uncertainty, drift, neighbours, residual learner, crowd, 60-day term, extremeness, nowcasts, self-context, regime context, pairwise, local drivers, evidence age); x06 TimesFM/Chronos-2 on spot; x22–x35 the book's RF / CatBoost / LightGBM / MLP / block ridges refit walk-forward from 2023-05 (predictions, tree/stage/seed spread, leaves, SHAP, hidden units, member spread). All causal and live-computable (context ends 20 dates — 60 for x13 — before each 20-date block; source models searched and refit every 130 in their own window; spot as-of D−1; reductions on past rows only; rank-binned per date). Builders `src/data_mani/{fmx_tabpfn,fmx_models,build_fmx}.py`. Plus 3 `x00` arms (each base re-run on its `-fmx` dataset), the in-study "without" every block pairs against. | 105 | DONE 2026-10-28 (x33 lgb re-run after its lane died 10-24): **null overall** -- 6 payout cells at \|t\| ≥ 2 of 102 (chance ~4.6), none past Bonferroni (~3.3). Hits: x29 LightGBM stage profile on ridge +0.00272 (t +3.17, CORR t +3.51) and catboost +0.00250 (t +2.02); x10 embedding neighbours on catboost +0.00209 (t +3.02); down: ridge x07 (−2.68), x10 (−2.35), lgb x20 (−2.00). x06 TimesFM/Chronos-2 confirms E109/E113: ridge −0.0015 (t −1.03), lgb +0.0008 (+0.18), catboost −0.0090 (t −1.97). Payout up on all three families: x17, x29, x34 (all small). TabPFN blocks mildly hurt ridge, lean positive on catboost. Bases reproduce their reused runs exactly. `src.analysis.e118_report`, results/reports/crypto/e118_summary.csv. x29 carried into E130. |
| **E119** | Training window of the best standalone MLP (IA_01), 1 week to 6 months | `e119_ia01_train_window_crypto.sh` | E117 asked of IA_01 (`mlp/a2_mlp_fenccausal`, payout-best standalone MLP, fenccausal 44 feat): only the rolling window varied, 5/10/15/20/22/65/130 periods, hyperparameters searched from scratch every refit, windows below 130 tuned with `--tune-window 130`. Cadence held at 130. The 130 arm must reproduce IA_01. | 7 | done 2026-10-24, 7/7. **6 months is clearly best; every window of 3 months or less is significantly worse** (510 dates, NW lag 20 vs w130): w5 0.0305 (t −3.07), w10 0.0230 (−3.38), w15 0.0245 (−2.95), w20 0.0145 (−4.88), w22 0.0197 (−4.38), w65 0.0374 on 470d (−3.39), w130 **0.0915**. Short windows pin α to the grid top. **w130 does NOT reproduce IA_01** (0.0915 vs 0.0987, t −1.06): the fit reproduces to 1.5e-4, but the inner-CV scores of identical candidates moved since 2026-08-23 (0.0561→0.0500 at refit 0) and flipped the winner at refits 0 and 2; not threads, host, search_jobs or scaler dtype (all bit-identical) — IA_01's 0.0987 is one draw of a search sensitive to 1e-4 perturbations. Residualised (E107 transform, s 0.5 lag 5): only w130 gains (+0.0083, t +4.80 → 0.1032, calmar 0.080→0.071). Reports: `logs/e119_report.txt`, `logs/e125_resid_report.txt`, `results/reports/crypto/e119_board.html`. |
| **E120** | Training window of the best standalone LightGBM (concat_trio), 1 week to 6 months | `e120_train_window_lgb_crypto.sh` | The LightGBM twin of E117/E119. E117 asked of the payout-best standalone LightGBM — `lgb/e03_concat_trio`, the canonical pinned booster (num_leaves 255, 4327 trees, lr 0.00257, colsample 0.07) on the group-CONCAT subset of `crypto/v2.0-onehot` (market-cap + momentum + volatility + per-symbol one-hot, 748 feat), 20-day payout 0.09074, the best standalone `lgb`. Only the rolling window varied: 5/10/15/20/22/65/130 periods. The model is PINNED (`--fixed-params`), so unlike E117/E119 there is **no hyperparameter search and no `--tune-window`** — every arm is the same booster refit on that much history, cleanly paired to the w130 control. Cadence held at 130, tree count held at 4327 — how much history, not freshness (E116) and not capacity. The 130 arm reproduces `e03_concat_trio`; each window paired (NW lag 20) vs w130. | 7 | done, 7/7 in 10 min. **The deployed 6-month window is clearly best — more history is strictly better, monotonic, every shorter window significantly worse** (NW lag 20 vs w130): w5 0.03223 (t −3.46), w10 0.02625 (t −3.98), w15 0.02945 (t −4.01), w20 0.03644 (t −3.62), w22 0.04018 (t −3.45), w65 0.06244 (t −2.91), w130 **0.09074** (control). payout climbs monotonically with window, p.sharpe 0.27→1.24, dd(pay) ~8→~1. A pinned 4327-tree booster on a week–month of data overfits badly (sharper than E117's ridge, being higher-capacity). w130 reproduces `e03_concat_trio` to 0.00e+00. Payout still rising at the 130 ceiling → a >130 (e.g. 260/1yr) arm is the natural follow-up. |
| **E121** | Training window of the best standalone CatBoost (IAUARETE_01), 1 week to 6 months | `e121_catboost_train_window_crypto.sh` | The CatBoost twin of E117/E119/E120. The payout- and MMC-best single catboost of E32 — `catboost_ktz` on fenccausal (`crypto/v2.0-fenccausal`, 44 feat = raw 22 + 22 causal per-symbol means), deployed as the component **IAUARETE_01** (backtest run `catboost_ktz/e32_sel_fenccausal`). Only the rolling window varied: 5/10/15/20/22/65/130 periods; the 12-cell KTZ grid searched exhaustively (`--n-iter 12`) every refit; windows below 130 use `--tune-window 130` (search on the 130 periods ending at the same date, folds train on the window's length, winner refit on the window). Unlike ridge there's no lambda-path scaling — the KTZ grid is static, so `grid_from` is ignored. Cadence held at 130 — how much history, not freshness (E116). The 130 arm reproduces `e32_sel_fenccausal` / IAUARETE_01; each window paired (NW lag 20) vs w130. | 7 | registered |
| **E122** | Training window of the best standalone RANDOM FOREST (e57_tenc_onehot), 1 week to 6 months | `e122_rf_window_crypto.sh` | The RF twin of E117/E119/E120/E121. The payout- and MMC-best single random forest — `rf_ktz` (E48 priors) on `crypto/v2.0-tenc-onehot` (763 feat), the **60-day** label, run `e57_tenc_onehot` (payout 0.0988 / MMC 0.0826 at w130). Only the rolling window varied: 5/10/15/20/22/65/130 periods; the 40-draw KTZ search runs every refit; windows below 130 use `--tune-window 130` (embargo auto=60, so a <70-period block can't hold its own purged fold — search on the 130 periods ending at the same date, folds train on the window's length, winner refit on the window). Cadence held at 130 — how much history, not freshness (E116). The 130 arm reproduces `e57_tenc_onehot`; each window paired (NW lag 20) vs w130. Expect short windows to lose and the heavy leaf floor {300..4000} to be unusable below a few periods. | 7 | running |
| **E123** | Training window of a FROM-SCRATCH-SEARCHED LightGBM (concat_trio feature set), 1 week to 6 months | `e123_train_window_lgb_search_crypto.sh` | The from-scratch-search twin of **E120**, and the true LightGBM analogue of E117/E119 (which searched their models per window; E120 pinned). Same base as E120 — `lgb` on the group-CONCAT subset of `crypto/v2.0-onehot` (748 feat), refit 130, embargo 20, seed 42, windows 5/10/15/20/22/65/130 — but the booster is **searched afresh at every window** (`--n-iter 60 --n-splits 5`) instead of pinned to `e03_concat_trio`. Answers what E120 could not: when the model may get *simpler* as its history shrinks, does a short window still lose, or does a re-tuned booster recover the payout pinning the 4327-tree booster threw away? The plain `lgb` grid spans concat_trio on every axis (n_est 2k–30k, lr 0.001–0.01, colsample 0.05–0.30, num_leaves 31–1023, and `min_child_samples`, whose range `fit_grid` rescales from the training block every refit via `models.scaled_leaf_range` — 20–780 at 130 periods, so concat_trio's 50 is inside). A 60-draw random search need not land on the pin; whether it finds something as good is part of the question. Windows < 130 use `--tune-window 130` (the E117 mechanism); w130 searches its own window. Reported paired (NW lag 20) vs the searched w130, cross-paired vs the **pinned** E120 arm at each window (does searching beat pinning?), and vs the deployed pinned concat_trio (`e120_w130` = 0.09074). | 7 | registered |
| **E124** | E117's training-window ladder, fitted on the 60-day label, scored on the 20-day one | `e124_train_window_t60_crypto.sh` | Same model, data, windows and from-scratch λ search as E117, but fitted and tuned on target_binned_return_60 (embargo 60). Predictions re-scored on the payable 20-day label via `cross_target`, then all 14 arms (E117 fit-20 + E124 fit-60) in one table on one common window. Two inner folds, not four — exactly production's 60-day search. FINDING: fitting on 60 helps only where there is too little history to fit 20 — +0.033 at 1 week, +0.029 at 2 weeks, cutting dd(pay) 8.96 → 1.88 — and hurts from 4 weeks up; 0 of 7 beyond t = ±2, and nothing beats 6 months fit on 20. First launch lost 5 arms to a full /home quota; rerun resumably. See findings below. | 7 | done |
| **E125** | E119's training-window ladder for IA_01, fitted on the 60-day label, scored on the 20-day one | `e125_ia01_train_window_t60_crypto.sh` | The MLP twin of E124. E119's arms -- `mlp` on fenccausal (IA_01's setup, 44 feat), rolling windows 5/10/15/20/22/65/130, the 50-draw search from scratch every refit, `--tune-window 130` below 130, refit 130 -- FITTED and TUNED on `target_binned_return_60` with embargo 60 (two inner folds, not four). Every arm RE-SCORED on `target_binned_return_20` by `cross_target.score_on_payable_target` and consolidated with E119's fit-20 arms on the 470 shared dates, paired NW lag 20 and 60. Arms run through E119's runner (`E119_TARGET=60`); `e125_w130` must reproduce `mlp/e60_mlp_fenccausal`. | 7 | done 2026-10-24, 7/7. **Fitting on 60 days hurts the 20-day payout at every window**, worst at 6 months: w130 fit-60 0.0360 vs fit-20 0.0949 (t −3.47 lag 20 / −3.28 lag 60); shorter windows −0.001..−0.030 (all |t|<2, but every fit-60 row is below its fit-20 twin). Every fit-60 arm is significantly below the fit-20 6-month reference (t −3.5..−7.7). Residualising does not help any fit-60 arm (w5 −0.0010, t −2.20) and lowers calmar in 14/14 rows. e125_w130 reproduces e60_mlp_fenccausal to 2.5e-15; w65 and w130 pick identical hyperparameters (embargo 60 truncates both windows' inner folds to the same 18/44 periods). Pairwise: fit-20 vs fit-60 predictions ρ median 0.21 (payout ρ 0.30, tail to −0.42), same-label 0.53/0.59, twins 0.99/1.00 (`results/cache/e125_pair_corr.csv`). Report `src.analysis.e125_resid_report`. |
| **E126** | E123's training-window ladder for the searched LightGBM, fitted on the 60-day label, scored on the 20-day one | `e126_train_window_lgb_search_t60_crypto.sh` | The LightGBM twin of E124 (ridge) and E125 (MLP), and the 60-day sibling of **E123**. E123's arms — `lgb` on the group-CONCAT subset of `crypto/v2.0-onehot` (748 feat), rolling windows 5/10/15/20/22/65/130, the `--n-iter 60 --n-splits 5` search from scratch every refit, `--tune-window 130` below 130, refit 130 — but **fitted and tuned on `target_binned_return_60`** with embargo 60 (`--embargo auto`). Crypto pays on the 20-day label, so every arm is **re-scored on `target_binned_return_20`** by `cross_target.score_on_payable_target` and consolidated with E123's fit-20 arms in one table (`python3 -m src.analysis.e126_report`: 7 windows × fit 20/60 × raw / meta-residualised at the live strength 0.5 = 28 rows, the e124/e127 layout, plus the 378-pair prediction- and payout-correlation histogram tagged across / window / twin), each window paired (NW lag 20) on the dates the pair shares and vs the deployed pinned concat_trio (0.09074) re-scored the same way. Asks whether training the booster on the smoother 60-day horizon and deploying on the 20-day payable label beats training directly on 20, and whether that depends on the window. | 7 | registered |
| **E127** | E121's CatBoost training-window ladder, fitted on the 60-day label, scored on the 20-day one | `e127_catboost_train_window_t60_crypto.sh` | The CatBoost twin of E124 (ridge), E125 (MLP) and E126 (lgb). E121's arms — `catboost_ktz` on `crypto/v2.0-fenccausal` (IAUARETE_01's setup, 44 feat), rolling windows 5/10/15/20/22/65/130, the exhaustive 12-cell KTZ grid searched from scratch every refit, `--tune-window 130` below 130, refit 130 — but **fitted and tuned on `target_binned_return_60`** with the embargo it forces (60, not 20). At embargo 60 a 130-period tuning block keeps only the **last two** of its five inner folds — exactly production's own 60-day search (`CURUPIRA_05_60`), kept for fidelity. Every arm **re-scored on `target_binned_return_20`** by `cross_target.score_on_payable_target` and consolidated with E121's fit-20 arms in one table (`src.analysis.e127_report`), each window paired NW lag 20 and 60 and vs the deployed `e121_w130` (IAUARETE_01) re-scored the same way. Arms run through E121's runner (`TW_TARGET=60`); the longest window relates to `catboost_ktz/e59_fenccausal`. Asks whether fitting CatBoost on the smoother 60-day horizon and deploying on the 20-day payable label beats fitting on 20, and whether that depends on the window. | 7 | registered |
| **E128** | E122's RF training-window ladder fitted on the 20-day label; with E122, raw and meta-residualised, all scored on 20 | `e128_rf_window_t20_crypto.sh` | The fit-20 half E122 (fit 60) lacks, so RF gets the same table as ridge (E117/E124), MLP (E119/E125) and catboost (E121/E127). E122's arms — `rf_ktz` (E48 priors) on `crypto/v2.0-tenc-onehot` (763 feat), windows 5/10/15/20/22/65/130, the 40-draw search from scratch every refit, `--tune-window 130` below 130, refit 130 — **fitted and tuned on `target_binned_return_20`** (embargo 20). Arms run through E122's runner (`TW_TARGET=20`). Deliverable `python3 -m src.analysis.e128_report`: 7 windows × fit 20/60 × raw / meta-residualised (live strength 0.5, 5-day lag, applied to finished forecasts) = 28 rows on the 20-day label, paired NW lag 20 (and 60 for fit-60 vs fit-20) and vs the 6-month fit-20 raw arm, plus the 378-pair prediction/payout correlation histogram (across / window / twin). Already known from E122 alone on the 20-day label: w130 fit-60 pays 0.0841, every shorter window hurts at \|t\|≥2. | 7 | done 2026-10-24, 7/7 (hopper+lovelace 3-core slots, ~75 min). **Six months fit on 20 is the best RF, and residualising it is the best row**: raw 0.0945 (calmar 0.077, best), resid 0.1024 (+0.0079, t +4.04). Fit-60 vs fit-20: 0 of 7 windows beyond \|t\|=2 — fit-60 is ahead at 2 weeks–1 month (+0.008..+0.026) and behind at 1 week, 3 months, 6 months (w130 −0.0104, t −0.59); e122_w130 resid 0.0906 is indistinguishable from the reference (t −0.22). Residualising raises payout in 12/14 arms (5 at t>2, both 6-month arms t ≈ +4) and lowers calmar in 8/14. Every window under 6 months loses to the reference. Pairs: across-label prediction ρ median 0.27 but payout ρ median −0.22 (67% of across pairs negative); the two 6-month arms ρ 0.75 pred / 0.20 payout; same-label 0.59/0.66; twins 0.97/0.99 (`results/cache/e128_pair_corr.csv`). Report `logs/e128_report.txt`. |
| **E130** | Causal leaf-value features of CatBoost (fenccausal + x29) and RF (allfeat) added to the best ridge and MLP | `e130_treeleaf_crypto.sh` | Do two of the book's best tree models carry structure that the best linear and neural models miss? `src/data_mani/build_treeleaf.py` re-runs `catboost_ktz` on fenccausal + E118's x29 block (`e118_catboost_x29`) and `rf_ktz` on allfeat (MA_ALLFEAT, `e52_allfeat`) through `forecast()` -- fit 20, rolling 130 / refit 130 / embargo 20, searched from scratch at every refit -- with the validation start moved back 2 blocks. Every row gets the distribution of its trees' leaf VALUES (mean, std, q10/25/50/75/90): 7 `feature_lfcat_*` + 7 `feature_lfrf_*` columns, rank-binned within date, merged as `crypto/v2.0-fenccausal-treeleaf` and `crypto/v2.0-tenc-onehot-rollsel-spread-interact-fmx-treeleaf`. Causal as in E129. Arms, each a 50-draw purged search from scratch: `ridge_glmnet` on allfeat_pp − tenc + x29 (`e118_ridge_x29`'s protocol) and `mlp` on fenccausal (IA_01's), each as control / +cat / +rf / +both, plus the MLP control and +both again at seed 43 for a noise floor. Paired NW lag 20 against the same-seed in-study control; a result is \|t\| ≥ 2 on payout. Prior: E129's RF leaves hurt IA_01 (t −2.44). | 10 | registered 2026-10-26 |
| **E129** | Causal leaf-value features of the 6-month RF added to the 8 models inside CU_SA_IA_BO_MA_R_01 and CU_SA_IA_BO_02 | `e129_rfleaf_members_crypto.sh` | Does the structure of E128's w130 random forest (rf_ktz, tenc-onehot, fit 20, 6-month window) help the book's two best slots? `src/data_mani/build_rfleaf.py` re-runs that forest through the pipeline's own `forecast()` with the validation start moved back 8 blocks, and gives every row the DISTRIBUTION of its 100 trees' leaf values (mean, std, q10/25/50/75/90) plus the forest's prediction residualised on the 5-day-lagged meta model: 8 `feature_rfleaf_*` columns, binned, merged as `crypto/v2.0[-...]-rfleaf`. Causal: a row's leaves come from its own walk-forward block's forest, trained only on rows that had resolved, so no row is ever scored by a forest that trained on it. Leaf IDs are deliberately not used, since the forest is refit every 130 periods and an ID means something different in each refit. The 6 single members are re-run on their exact recorded protocol as control and +rfleaf, and CURUPIRA_05/_05_60 gain a sixth, leaf-only ridge block. `src.analysis.e129_report` reassembles both slots from the controls and from the +rfleaf arms with one assembler, which first reproduces the stored slot series. Paired NW lag 20 on the 20-day payout; a result is \|t\| ≥ 2 for a slot or member. | 14 | done 2026-10-24 (14/14) — **null-to-negative for both slots**: R_01 −0.0071 (t −2.17), BO_02 −0.0025 (t −1.04); IA_01 hurt (t −2.44); only the leaf-only ridge BLOCKS help (C05 t +2.24, C05_60 t +3.36), diluted to nothing in the slots (see findings) |
| **E131** | litmom-fillind vs v2.0: fixed-config model × feature-group factorial | `e131_litmom_fillind_factorial.py` | Does `crypto/v2.0-litmom-fillind` (60f = 22 base + 17 `feature_mom_` + 21 `feature_isna_`: 17 momentum, 4 base) add out-of-sample value over `crypto/v2.0` on the 20-day label? Three families (ridge, lgb_ktz, mlp_ktz) each at ONE fixed previously-searched config, refit per chronological fold, across B0 / B0C / B1 / M1 / M2 / FULL / ADDED_ONLY (+ leave-one-family-out, alt-fill control) on identical aligned rows; 22-date purge; untouched final test block; 5 seeds for stochastic families; `numerai_tools` numerai_corr; paired deltas with NW/block-bootstrap CIs; custom contribution relative to B0; CV-chosen B0+ADDED_ONLY blend weight tested once on the final block. Audits first (schema, base integrity, indicators, point-in-time, missingness, config provenance). Report `results/reports/crypto/e131_litmom_fillind/E131_does_litmom_help_REJECT.html`. | **REJECT.** FULL−B0 dev (1,430 dates, NW lag 20): ridge −0.0111 (t −2.38), lgb_ktz +0.0005 (t +0.13), mlp_ktz −0.0170 (t −2.99); test block all negative; avg −0.0092 CI [−0.0169,−0.0015]. Raw momentum hurts ridge/mlp (t≈−2.9); momentum and base indicators unsupported (lgb M2−M1 +0.0036 dev reverses on test t −2.04). Equal-budget retuning does not rescue (ridge t −2.22, mlp t −3.01; lgb +0.009 pre-test, −0.010 test t −3.28). Missingness channel carries survivorship look-ahead (spot coverage matched 2026-09-22): P_NM placebo +dev / test t −3.6..−5.4. Causal tenure control P_HIST +0.0085 avg (t +3.53) dev, positive-ns test. Lead: lgb B0+0.3·ADDED_ONLY blend test +0.0063 (t +2.33), promising-unconfirmed. mlp MISS_ONLY unfittable (0 cols). Test block was reused from E64/E65. | done 2026-10-28 |
| **E132** | feed-fillind datasets vs v2.0, neutral 0.5 for coins with no features | `e132_feed_fillind_factorial.py`, `e132_analysis.py` | E131's protocol (same 513,496 keys, 12 blocks, fixed configs, 5 seeds, test block 2026-02-20..08-20, already scored by E65/E131) on `crypto/v2.0-litmom-feed-fillind`, `-momswap-feed-fillind`, `-momswap-macro-carver-feed-fillind`. All three share 480,833 keys; the other 6.4% are coins with no price, which score at a neutral prediction (per-date rank, then 0.5) instead of leaving the universe. Dropped rows average target 0.431 against 0.504, so the drop is itself look-ahead: B0F (raw v2.0 on the feed rows) separates row handling from features, P_DROP sizes the survivorship channel, P_HIST controls tenure. Plus group ablations, equal-budget retunes (`e132_<m>_tune*`), CV-chosen ADDED blends. | 24 x 3 + 12 | DONE 2026-10-31: **REJECT, all 3 datasets on all 3 models.** Against v2.0 with no-feature coins at 0.5: dev ΔCORR ridge −0.033..−0.034, lgb −0.018..−0.022, mlp −0.028..−0.033 (all \|t\| ≥ 4.2); test all negative, \|t\| ≥ 2.0. Row handling alone −0.0175 (t −8.0): neutral rows −0.013, refit on feed rows −0.004; they average target 0.431 vs 0.504 and only 14/245 symbols survive, so the row set is look-ahead. Features alone (vs B0F): ridge/mlp hurt on dev (t −2.5..−3.7), lgb null, test not significant. Retunes: 18/18 negative vs v2.0, 15 at \|t\| ≥ 2. Blends fail; E131's lgb lead does not repeat (test t +1.5). P_HIST +0.0088 dev (t +3.8) but mixed on test, weakening E131's lead. 5.57% of scored keys are neutral. ~25 wrong-coin price series uncorrected. `results/reports/crypto/e132_feed_fillind/E132_do_feed_fillind_datasets_help_REJECT.html`. |
| **E133** | Own-history features vs v2.0: fixed-config model × horizon factorial | `e133_histfe_factorial.py` | Does `crypto/v2.0-histfe` (374f = 22 shipped + 352 `feature_hfe_`: own-history rank 7/14/30/90/364d, MA difference 3/7, 7/14, 14/30, 30/90, 91/364d, days since max, days since change; per ucid, calendar windows, 60% coverage, tie-preserving Numerai bins) add out-of-sample value on the 20-day label? Three families at the E131 fixed configs on identical rows, arms B0 / SHORT / LONG / COMBO / TSC / FULL / ADDED_ONLY / F_RANK / F_MA / F_TSMAX / P_HIST, 11 dev blocks (train 130, purge 22, predict 130), 5 seeds, equal-budget retune on B0 and FULL. Paired deltas with NW lag 20 and block bootstrap, redundancy, blends by leave-one-block-out. Dates from 2026-02-20 are never read: the final test period stays untouched, so no verdict can exceed PROMISING BUT UNCONFIRMED. Report `results/reports/crypto/e133_histfe/`. | **REJECT** on dev, test unread. CORR vs B0, 1,430 dates: ridge SHORT −0.0164 (t −4.5), LONG −0.0238 (−5.1), COMBO −0.0302 (−5.8); mlp_ktz −0.0397 (−8.0), −0.0298 (−6.9), −0.0447 (−9.4); lgb_ktz −0.0014 (−0.6), −0.0130 (−3.1), −0.0119 (−2.9). Payout, 468 dates: ridge and mlp negative, lgb small and positive (+0.0058..+0.0071, |t|<2). Only lgb F_RANK clears t 2 on payout (+0.0123, t +3.2, below a Bonferroni ~3.6) and it ties tenure P_HIST (−0.0005, t −0.1). P_HIST lifts all three models (CORR t +4.8, +3.1, +2.6). Retune and leave-one-block-out blends give no rescue; the mlp retune collapses on alpha=100 blocks. Drawdown falls on most ridge/mlp arms, so calmar rises with payout lower. Numerai's exact bin rule found, (rank-1)/(n-1) with inclusive .05/.25/.75/.95 cuts, 38,720 of 38,720 cross-sections; `binning.rank_to_bins` matches 91.13%. `experiments/e133_followup.py`, `e133_report.py`; report `results/reports/crypto/e133_histfe/E133_do_own_history_features_help_REJECT.html`. |
| **E134** | feed-fillind datasets as live would use them: route each coin by what it has | `e134_route_feature_sets.py`, `e134_report.py` | Registered before scoring, 2026-10-31. Would these datasets beat v2.0 under the rule we would run live? No refit: E132's saved predictions. Primary rule: raw v2.0 ranks all coins, and the richer model reorders only coins with a row in its dataset, inside the rank slots they already hold, so a no-feed coin keeps its v2.0 rank. Arms: L/S/C `_FULL` and indicator-free `L_M1`/`S_M1`/`C_RAW`, ridge/lgb_ktz/mlp_ktz, seed-paired. Sensitivity: pooled merge, and routing only coins with at least half the proposed features observed. Also the 3 retunes against E131's tuned B0. Fixed bar: ADOPT = dev CORR t >= +2, test delta > 0, payout not significantly negative on either window. REJECT = dev t <= -2. PROMISING = test t >= +2 without a dev REJECT. Else NO IMPROVEMENT. **DONE 2026-10-31: REJECT for deployment, all 3 datasets.** Of 18 primary cells: 12 REJECT (all ridge and MLP), 6 NO IMPROVEMENT (all LightGBM), none PROMISING or ADOPT. Routing cuts E132's dev loss by roughly 40-70% (FULL dev ΔCORR ridge -0.0184..-0.0200, lgb -0.0053..-0.0089, mlp -0.0144..-0.0186), and test is 17 of 18 negative, with L_M1 ridge and MLP at |t| >= 2. Covered-coin deltas match, so the remaining gap is real ordering, not routing. R_POOL and R_HALF do not change any reading; the one payout cell at |t| >= 2 positive, R_HALF lgb S_FULL dev +0.0097 (t +2.0), is negative on test. Retunes: 16 of 18 negative against tuned B0, 7 at |t| >= 2. LightGBM dev calmar improves (0.041-0.049 vs 0.023, shallower drawdown) and does not hold on test. Report `results/reports/crypto/e134_route/` (one PDF per dataset). Caveat: covered set is still chosen by today's feed, so the numbers may flatter the candidates. |
| **E135** | own-history features as live would use them: route each coin by what it has | `e135_route_histfe.py`, `e135_report.py` | Registered before any test date was read, 2026-11-01. Would `crypto/v2.0-histfe` beat v2.0 under E134's live rule? E133's 11 dev blocks plus one new fit of the final block, 2026-02-20..08-20, train 130 / purge 22 / predict 130. This spends the block E133 held back, because E134's bar needs it; E65/E131/E132/E134 already scored those dates on other features, so it is not virgin, and it selects nothing here. histfe has no feed mask, so covered means the arm's OWN history features are observed that date, which is warm-up and known at the time. Primary rule R_SLOT (at least one observed): raw v2.0 ranks all coins, the candidate reorders only covered coins inside the rank slots they hold. Sensitivity: R_POOL, and R_HALF (at least half observed). Primary arm FULL on ridge/lgb_ktz/mlp_ktz, seed-paired against B0 on all keys. Secondary, same bar: SHORT, LONG, COMBO, TSC, F_RANK, F_MA, F_TSMAX; control P_HIST (tenure, nothing to route). ADDED_ONLY is left out: it drops the base features, so it is not a deployment candidate. Retune: E133's tuned B0 and FULL plus the final block under the same grid. Fixed bar: ADOPT = dev CORR t >= +2, test delta > 0, payout not significantly negative on either window. REJECT = dev t <= -2. PROMISING = test t >= +2 without a dev REJECT. Else NO IMPROVEMENT. Report `results/reports/crypto/e135_route_histfe/`, PDF `E135_histfe_routed_REJECT.pdf`. | **DONE 2026-11-01: REJECT for deployment.** B0 on the new block reproduces E132's to 1.2e-7. FULL ΔCORR dev / test (NW t), R_SLOT: ridge −0.0295 (−5.3) / −0.0346 (−3.3); lgb_ktz −0.0110 (−2.7) / +0.0042 (+0.5); mlp_ktz −0.0422 (−7.3) / −0.0512 (−4.5), REJECT on all 3 under all 3 rules. Routing does little: 0.3% of FULL's keys take the fallback under R_SLOT, 5.1% under R_HALF. Of 24 non-control test cells, 3 reach t >= +2 and 12 reach t <= −2. PROMISING by the bar, neither confirmed: lgb F_RANK (dev −0.0073, t −1.96, 3 of 5 seeds past −2; test +0.0063, t +2.31; NO IMPROVEMENT under R_HALF, test t +1.99) and lgb TSC (dev −0.0011, t −0.4; test +0.0093, t +2.4; test payout t +0.9). lgb F_RANK's payout is same-sign, +0.0128 (t +3.3) / +0.0037 (t +1.3), and calmar beats B0 on both windows (0.050 vs 0.023, 0.035 vs 0.028), but it ties tenure on dev payout. P_HIST meets ADOPT by the letter on all 3 models (dev t +2.6..+4.8), yet test deltas are +0.0006..+0.0019 with |t| < 0.5: control, not confirmed, needs its own study. Retune: every ΔCORR negative, ridge test t −3.2, mlp test t −3.2, lgb test t −0.9. Caveats: this spent the last histfe-unread window, so the PROMISING cells can only be confirmed on later dates; mlp_ktz linear; proxy MMC; blends not re-scored. |
| **E136** | histfe: complete the feature-set evaluation procedure | `e136_histfe_evaluation.py`, `e136_report.py` | Registered before launch, 2026-11-01. Runs only the steps of the `feature-set-evaluation` skill that E133/E135 left open, and reuses their 12-block predictions, because refitting deterministic arms reproduces them. New fits, 12 blocks at E133's fixed configs and seeds: NO_RANK, NO_MA, NO_TSMAX, NO_TSC (leave one family out of FULL); FULL_H, F_RANK_H, TSC_H (block + tenure, read against P_HIST); P_WARM (B0 + the binned share of history features observed), a causal placebo for how much of a gain is only depth of history, since histfe has no look-ahead flag for a P_NM arm. ADDED_ONLY gets its final block. Blends: rank B0 + w*ADDED_ONLY, w from E.W chosen on all dev dates, by CORR and separately by payout, then scored on test. Audit: rebuild the block on the real panel truncated at 3 dates and compare with the dataset; overlap of warm-up with E132's look-ahead no_feed flag. Scoring is plain on all 300 coins, as the procedure specifies; the routed reading stays E135's. Verdict vocabulary from the skill: ADOPT = dev t >= +2, test same sign, survives retune and controls. REJECT = significantly harmful or leakage-driven. PROMISING BUT UNCONFIRMED = a test-only success. Else NO IMPROVEMENT. The final block was already read by E135, so it is not untouched, and it selects nothing here. Report `results/reports/crypto/e136_histfe_evaluation/E136_does_histfe_help_REJECT.html`. | **DONE 2026-11-01: REJECT.** Audit clean: 0 of 422,400 cells differ on the truncated real panel; warm-up vs no_feed correlation 0.10. FULL plain ΔCORR dev / test: ridge −0.0300 (−5.3) / −0.0346 (−3.3); lgb −0.0110 (−2.7) / +0.0044 (+0.5); mlp −0.0428 (−7.3) / −0.0512 (−4.6). ADDED_ONLY hurts all 3. Ablations: no single family explains the harm; no lgb family reaches |t| 2. Blends: weights from dev, nothing reaches t 2 on either window (best lgb w 0.10, test CORR +0.0054, t +1.3). Placebo P_WARM earns lgb dev payout +0.0099 (t +2.6), and F_RANK beats it by only +0.0025 (t +0.6), so depth of history explains most of F_RANK's dev payout gain. Tenure: test deltas at most +0.0019, |t| < 0.5. PROMISING by the bar, test only: lgb SHORT and TSC. lgb F_RANK flips with the rule: dev t −2.09 plain (REJECT), −1.96 routed (PROMISING). Dev CORR starts 2020 and dev payout May 2024: on payout's 468 dates lgb F_RANK has CORR +0.0082 (t +2.2). Lead, read after the fact: lgb F_RANK + tenure vs tenure, payout dev +0.0065 (t +1.7), test +0.0070 (t +2.7), 5 of 5 seeds positive on both; but full-dev CORR −0.0091 (t −2.7) marks it REJECT by the bar, vs v2.0 test payout only +0.0020 (t +0.4), and it hurts ridge and mlp. Not compared with the placebo. 28 labelled dates after 2026-08-20 were deliberately left unread: about one independent period, no meta model. |
| **E137** | fenccausal-binned: full feature-set evaluation procedure | `e137_fenccausal_binned_evaluation.py`, `e137_report.py` | Registered before launch, 2026-11-01, run on lovelace. Does `crypto/v2.0-fenccausal-binned` (44f = 22 shipped + 22 `_symmean`, causal expanding per-symbol means, pseudo-count 50, rank-binned per date) beat v2.0? Base features and targets equal the E133 panel on all 468,000 keys, so B0, P_HIST and tuned B0 are reused from E133/E135 and a ridge refit checks that. Protocol: 12 blocks (train 130, purge 22, predict 130), E133's fixed configs, target_binned_return_20, seeds 5 for lgb_ktz and mlp_ktz. Arms: FULL; ADDED_ONLY; CONT (same means unbinned, recomputed in the run because `v2.0-fenccausal` stops at 2026-07-08; FULL vs CONT is the cost of binning); F_PRICE, F_CAP, F_VOL, F_SIGNAL and their leave-one-out NO_*; H20, H60; FULL_H vs P_HIST; P_GLOBAL, the binned whole-sample means, a look-ahead placebo; tuneFULL vs tuned B0. Blends rank B0 + w*ADDED_ONLY, w chosen on dev. Routing R_SLOT and R_HALF on FULL. Audit: truncated rebuild, reused tickers, full-frame prior. Primary cells FULL on 3 models. Bar: ADOPT = dev CORR t >= +2, test delta > 0, payout not significantly negative on either window, surviving tenure, retune and placebo. REJECT = dev t <= -2 or leakage-driven. PROMISING BUT UNCONFIRMED = test t >= +2 without a dev REJECT. Else NO IMPROVEMENT. Test 2026-02-20..08-20 selects nothing and is not virgin. The 28 labelled dates after it stay unread. Report `results/reports/crypto/e137_fenccausal_binned/`. | DONE 2026-11-01, **NO IMPROVEMENT, REJECT on ridge**. FULL vs v2.0: ridge dev CORR -0.0096 (t -2.3), test +0.0042 (t +0.8); lgb -0.0012 (t -0.3), test -0.0087 (t -1.9); mlp -0.0054 (t -1.4), test +0.0030 (t +0.6). ADDED_ONLY REJECT on all three; no feature-set arm reaches ADOPT; the only PROMISING cell of 39 is ridge F_VOL (test t +2.0, dev flat). Audit clean: 0 of 11,488,312 cells on rebuild, 0 of 26,400 at four cut dates, whole-frame prior moved no bin; 1.7% of keys mix two coins (ticker-keyed). Signal means carry the CORR harm. Binning costs nothing: FULL beats CONT on dev CORR for lgb (+0.0109, t +3.2) and mlp (+0.0109, t +3.1), and CONT is REJECT on both. mlp FULL payout test +0.0135 (t +3.2) fails the tenure control (FULL_H vs P_HIST -0.0077) and is fit noise-sized: one extra feature moves test payout by -0.018. P_GLOBAL look-ahead is worth +0.05 CORR, the causal block zero or less. tuneFULL vs tuneB0 on mlp is unreadable: large alpha collapses the net to a constant in 6 and 4 of 12 blocks. Routing is a no-op (every key has the block). Do not deploy; no forward candidate. |
| **E138** | carver-macro: full feature-set evaluation procedure | `e138_carver_macro_evaluation.py`, `e138_report.py` | Registered before launch, 2026-11-04. Does `crypto/v2.0-carver-macro` (55f = all 22 shipped + 15 `feature_tf_` Carver/Clenow trend signals + 18 shared `feature_macro_`; builder `src/data_mani/build_carver_macro.py`) beat v2.0? Missing-value standard: every v2.0 row kept, trend block recomputed from the current spot file (close of D-1, at most 4 days stale, binned per date over the panel), NaN left in the file (0.5 at load), no `feature_isna_*` indicators; scoring both plain on all 300 coins and ROUTED (E134 R_SLOT primary for deployment: v2.0 ranks every coin, the candidate reorders only coins whose block is observed; R_HALF sensitivity). Base features and target equal the E133 panel on all 468,000 keys (checked cell for cell), so B0, P_HIST and tuned B0 (ridge, lgb) are reused from E133/E135; ridge B0CHK refit matches to 8.9e-7. Protocol: 12 blocks (train 130, purge 22, predict 130), E133 fixed configs, target_binned_return_20, 5 seeds lgb_ktz/mlp_ktz. Arms: FULL; ADDED_ONLY; F_TF; F_MACRO; NO_EWMAC, NO_BREAKOUT, NO_BBZ, NO_CLENOW (incl. newhigh20in5), NO_RATES (yields+slopes), NO_RISK (gold/silver/S&P/DXY); FULL_H vs P_HIST; P_WARM (binned share of trend block observed, causal); P_NM (coin never has a trend value in the whole file: LOOK-AHEAD placebo); tuneFULL vs tuneB0 with the mlp alpha grid capped at 1 (E137 collapse), so mlp tuneB0 is refitted. Blends rank B0 + w*ADDED_ONLY, w chosen on dev. Audit: rebuild both blocks from panel/spot/macro cut at 5 dates. Primary cells: FULL routed R_SLOT on 3 models. Bar as E137: ADOPT = dev CORR t >= +2, test delta > 0, payout not significantly negative on either window, surviving tenure, retune and placebos; REJECT = dev t <= -2 or leakage-driven; PROMISING BUT UNCONFIRMED = test t >= +2 without a dev REJECT; else NO IMPROVEMENT. Prior: E72 (carver vs raw) and E69 (macro, ridge-inert) were null on older datasets; E132/E134 rejected momswap-macro-carver-feed-fillind, which lacked the 8 base momentum proxies. Test 2026-02-20..08-20 selects nothing and is not virgin; the 28 labelled dates after it stay unread. Report `results/reports/crypto/e138_carver_macro/`. | DONE 2026-11-04, **NO IMPROVEMENT** (no ADOPT; macro REJECT on lgb, trend REJECT on mlp). Audit clean: 0 of 49,500 cells at 5 cut dates; trend observed on 92.2% of keys, 5.8% no-feed coins, about 8% of coins routed. FULL routed R_SLOT vs v2.0, CORR dev / test: ridge −0.0024 (t −1.0) / +0.0001; lgb −0.0072 (t −1.7) / +0.0043 (t +1.6); mlp −0.0050 (t −1.7) / −0.0032 (t −0.9). Macro inert on ridge (F_MACRO = v2.0 to 1e-5, date-constant); lgb F_MACRO dev −0.0113 (t −2.3), test +0.0063 (t +2.6): regime split fitted to dev, carried by yields/slopes (FULL vs NO_RATES dev −0.0084, t −2.8). Trend: mlp F_TF REJECT (dev −0.0065 t −2.1, test −0.0080 t −2.9). Only lead: lgb F_TF routed, test CORR +0.0045 (t +1.6), payout +0.0044 (t +2.0), 5/5 seeds positive on test, 1/5 on dev, does not clear P_WARM. Unrouted gains (lgb FULL test +0.0112, t +2.3) shrink under routing. P_NM look-ahead flag lgb dev +0.0054 (t +3.1), test null; tenure dev-only again. Capped mlp retune no longer collapses (min spread 0.02); tuneFULL vs tuneB0 test payout +0.0127 (t +2.9), one seed, not a lead. Blends pick w 0 to 0.05. Do not deploy. Report `results/reports/crypto/e138_carver_macro/E138_does_carver_macro_help_NO_IMPROVEMENT.pdf`. |
| **E139** | carver-macro-infl: full feature-set evaluation procedure | `e139_carver_macro_infl_evaluation.py` | Registered before launch, 2026-11-04, run on dostoievski + tolstoi. Does `crypto/v2.0-carver-macro-infl` (67f = E138's 55 + 12 shared `feature_macro_bei_*`: break-even 5y/10y/30y/5y5y levels and their 20/60-day changes, causal trailing percentile, D-1, 7 days stale; built by `build_inflation` onto E138's dataset) beat v2.0, and does inflation add anything given carver-macro? Same missing-value standard, protocol, routing (R_SLOT on trend coverage) and bar as E138; machinery imported from E138's runner. Arms: CM = E138 FULL (reused parent); FULL; ADDED_ONLY; F_INFL; NO_BEILVL; NO_BEICHG; FULL_H vs P_HIST; P_WARM and P_NM reused from E138; tuneFULL vs tuneB0 (mlp grid capped at alpha 1). Key reads: FULL vs B0 and FULL vs CM. Prior: E80/E81 inflation null. A date-constant column cannot reorder coins in a linear model, so F_INFL should be inert on ridge and the linear mlp. Audit: block rebuilt from the macro store cut at 5 dates; every other cell equals E138's. Report `results/reports/crypto/e139_carver_macro_infl/`. | DONE 2026-11-04, **NO IMPROVEMENT, REJECT on lgb and mlp; inflation harmful.** Audit clean (0 of 18,000 block cells at 5 cut dates; 0 of 28.8M parent cells differ). FULL routed vs v2.0 CORR dev / test: ridge −0.0024 / +0.0001 (identical to CM: date-constant columns are inert in a linear fit); lgb −0.0115 (t −2.6) / +0.0036; mlp −0.0078 (t −2.2) / −0.0029. FULL vs CM (inflation's margin): ridge 0; lgb dev CORR −0.0043 (t −2.7), payout lower on 5/5 seeds on both windows; mlp dev payout −0.0118 (t −4.7) but per-seed −0.021..+0.009, mixed sign on test = fit noise. lgb F_INFL dev −0.0144 (t −3.6), test payout −0.0027 (t −2.0): regime split fitted to dev. Levels' margin (FULL vs NO_BEILVL) REJECT on lgb/mlp. mlp tuneFULL vs tuneB0 PROMISING by letter (test payout +0.0168, t +2.9), but E138's retune without inflation gave the same, so it is the retune, not the block. Third inflation null after E80/E81: do not carry the block. Report `results/reports/crypto/e139_carver_macro_infl/E139_does_carver_macro_infl_help_NO_IMPROVEMENT.pdf`. |
| **E140** | carver-macro-infl-xinteractions: full feature-set evaluation procedure | `e140_carver_macro_infl_xinteractions_evaluation.py` | Registered before launch, 2026-11-04, run on deepfour + deepsix. Does `crypto/v2.0-carver-macro-infl-xinteractions` (337f = E139's 67 + 270 `feature_ix_<trend>_X_<macro>`: raw trend signal x signed raw macro momentum, 15 trend x (10 metals/S&P/DXY returns + 8 break-even changes), binned per date AFTER the product, so each is the trend ranking sign-flipped by the macro regime; builder `src/data_mani/build_trend_xinteractions.py`; NaN wherever the trend signal is, no rows dropped, no indicators) beat v2.0, and do the interactions add anything given carver-macro-infl? E138 protocol, routing and bar. Arms: CMI = E139 FULL (reused parent); FULL; ADDED_ONLY; F_IX; IX_RET (150 return interactions); IX_BEI (120 break-even interactions); FULL_H vs P_HIST; P_WARM/P_NM reused from E138; tuneFULL vs tuneB0. Key reads: FULL vs B0 and FULL vs CMI. Prior: E71/E79 xinteractions on momswap-macro were null. Audit: interactions rebuilt from panel/spot/macro cut at 5 dates, plus the sign-flip identity. Report `results/reports/crypto/e140_carver_macro_infl_xinteractions/`. | DONE 2026-11-04, **REJECT on all three models.** Audit clean (0 of 405,000 cells at 5 cut dates; coverage = trend block, 90.9%). FULL routed vs v2.0 CORR dev / test: ridge −0.0449 (t −6.3) / −0.0289 (t −2.8); lgb −0.0191 (t −3.9) / −0.0059; mlp −0.0540 (t −8.2) / −0.0303 (t −2.9). FULL vs CMI REJECT on all 3 on both windows (lgb test payout −0.0100, t −4.5, 5/5 seeds negative). IX_RET and IX_BEI each REJECT on ridge/mlp, not above the parent on lgb; retune does not rescue; blends pick w 0. Mechanism: each binned interaction is its trend signal sign-flipped by the macro regime (mean |Spearman| 0.986), so 270 near-copies with date-varying sign that linear models average into noise. Close the macro-interaction line (also E71/E79). Report `results/reports/crypto/e140_carver_macro_infl_xinteractions/E140_does_carver_macro_infl_xinteractions_help_REJECT.pdf`. |
| **E141** | macrobeta: rolling coin betas to macro/inflation factors, full feature-set evaluation | `e141_macrobeta_evaluation.py` | Registered before launch, 2026-11-04. Does `crypto/v2.0-macrobeta` (54f = 22 shipped + 32 `feature_mbeta_<factor>_<w>d`; builder `src/data_mani/build_macro_betas.py`) beat v2.0? Each beta is the univariate rolling OLS slope of a coin's close-to-close log return on the one-day change of one factor, on the macro business-day calendar (weekend crypto moves fall into Monday), over 60 and 120 macro days, at least 80% of the window observed; factors: log returns of gold (GLD), silver (SLV), S&P, DXY; changes of y1/y2/y5/y10/y30, slopes 2s10s/1s10s/5s30s, break-evens 5y/10y/30y/5y5y. Causal: beta_t uses data <= t, attached to row D as-of <= D-1 (7 days stale), then rank-binned per date over the panel. Same missing-value standard (no drop, no indicators, NaN -> 0.5 at load, routed). Motivation: E138/E139 showed the raw date-constant macro/infl columns are exactly inert in linear models and harmful regime splits in lgb; a coin's sensitivity is cross-sectional. Arms: FULL; ADDED_ONLY; F_PRICE (8), F_RATES (16), F_INFL (8); W60, W120; NO_PRICE/NO_RATES/NO_INFL; FULL_H vs P_HIST; P_WARM (binned share of betas observed); P_NM (coin never has a beta: look-ahead placebo); tuneFULL vs tuneB0 (mlp alpha capped at 1). E138 protocol and bar; routing covers a coin when its betas exist that date. Built 2026-11-04: betas match direct OLS exactly on spot checks; mean |rho| among betas 0.24; max |rho| with a shipped feature 0.26 (S&P beta vs volatility_60d). Report `results/reports/crypto/e141_macrobeta/`. | registered **DONE 2026-11-06: REJECT on all 3.** Routed FULL vs v2.0 dev CORR: ridge -0.0181 (t -3.8), lgb -0.0145 (t -3.1, 5/5 seeds; test -0.0245 t -2.5), mlp -0.0189 (t -4.0). Betas alone (ADDED_ONLY) carry ~no rank signal; price betas worst (F_PRICE REJECT dev+test on ridge/lgb). Rate betas lower CORR but raise payout via MMC (lgb dev payout +0.0183 t +3.2; mlp test +0.0151 t +3.5) -- fails the bar on CORR. W60~W120. FULL < P_WARM on dev for every model. Blend weights all 0. Audit clean (0/48,000). Closes the macro line (raw E138/E139, interactions E140, betas E141). Report `results/reports/crypto/e141_macrobeta/E141_does_macrobeta_help_REJECT.pdf`. |
| **E142** | future-trends: look-ahead positive control for the feature-set evaluation | `e142_future_trends_evaluation.py` | Registered before launch, 2026-11-05. A SANITY CHECK, not a candidate. `crypto/v2.0-future-trends` (23f = 22 shipped + `feature_future_ret_20d`, builder `src/data_mani/build_future_trends.py`) adds each coin's spot return over the NEXT 20 calendar days, close(D+20)/close(D)-1, rank-binned per date; 89% of rows observed, per-date Spearman with target_binned_return_20 0.56. Arms FULL and ADDED_ONLY vs reused B0 on E138's protocol (12 walk-forward blocks, 5 seeds for lgb/mlp, routing, blends). Expected: a margin far beyond any honest block, read as LEAKAGE. Report results/reports/crypto/e142_future_trends/E142_does_the_evaluation_catch_lookahead_PASS.pdf. | DONE 2026-11-05, PASS. FULL vs B0 unrouted ΔCORR dev/test: ridge +0.396 (t +38) / +0.350 (t +8.5), lgb +0.401 (t +50) / +0.344 (t +8.8), mlp +0.397 (t +41) / +0.351 (t +8.4); payout +0.38..+0.42; no drawdown on any FULL series. Best honest FULL ΔCORR dev in E136-E141 is negative for every model. The bar reads it as ADOPT: it measures size and sign, not provenance, so only the point-in-time audit and look-ahead placebos stop a leak. ADDED_ONLY CORR is 0.532 for all three models (one 5-bin column, so every monotone model ranks the same); FULL is slightly lower on CORR (the 22 dilute it) and higher on payout. Window probe: the target best matches close(D+2)->close(D+30) (per-date Spearman 0.88), not D->D+20 (0.56). |
| **E143** | c2syn: Chronos-2-Synth pre-head representations (with / without covariates), full feature-set evaluation | `e143_c2syn_evaluation.py` | Registered before launch, 2026-11-06. Does `crypto/v2.0-c2syn` (54f = 22 shipped + 16 `feature_c2cov_pcNN` + 16 `feature_c2ar_pcNN`) add value over v2.0? Each coin's panel history (label + 22 features on the panel-date grid) goes zero-shot through `autogluon/chronos-2-synth` (Chronos-2's 120M architecture trained on synthetic series only): **cov** = label with the 22 features as covariates in one group, **ar** = label alone. The kept vector is the encoder's last hidden state of the label variate at the output patch holding date D's step, i.e. the input of the forecasting head (`src/data_mani/chronos2_syn_repr.py`, fp16 autocast, cosine to fp32 >= 0.99999). Point in time: labels only from dates <= D-35 days (target resolves ~D+30, train is served 32 days behind); features after the wall enter as known-future covariates; context 256 steps, >= 32 resolved labels or the row is left out. 768-d states -> 16 PCs per mode on a past-only burn-in basis, rank-binned per date with Numerai's exact rule (`build_c2syn.py`). Arms on E138's protocol: FULL, ADDED_ONLY, F_COV, F_AR, FULL_H vs P_HIST, P_WARM, P_NM, equal-budget retune; routing on the block's coverage. **Findings: NO IMPROVEMENT overall, REJECT on ridge and the MLP.** Audit clean (5 dates rebuilt from cut panels: 0 of 2,082,816 state values differ). Routed FULL vs v2.0: ridge dev CORR -0.0060 (t -2.2), dev payout -0.0102 (t -2.1); mlp_ktz dev CORR -0.0079 (t -2.5); lgb_ktz dev null, test CORR -0.0060 (t -1.9), payout -0.0072 (t -2.0), 5/5 seeds lower on test. 13/32 PCs have raw per-date IC |t|>=2 (best +-0.06, t +-7; max |rho| with a shipped feature 0.30), but ADDED_ONLY is -0.04..-0.08 dev CORR on every model: standalone IC, no marginal value. F_COV beats F_AR on dev CORR on all three (t +2.0..+2.6) and F_AR is the worst block (mlp dev CORR t -5.7, lgb test payout t -3.8), but F_COV is still <= v2.0 everywhere. FULL_H loses to tenure on every model; dev-picked blends w 0-0.10, lgb's 0.10 loses on test (t -2.8); retune does not rescue it. Closes TSFM representations as features (with E109/E113/E118 x06). Report `results/reports/crypto/e143_c2syn/E143_does_c2syn_help_NO_IMPROVEMENT.pdf`. | DONE 2026-11-06 |
| **E144** | c2ft: walk-forward LoRA fine-tuned Chronos-2-Synth pre-head representations (with / without covariates), full feature-set evaluation | `e144_c2ft_evaluation.py` | Registered before launch, 2026-11-06. E143 with `autogluon/chronos-2-synth` LoRA fine-tuned (r 8, lr 1e-5, 1000 steps, prediction length 35, context 256, early-stopped on each series' last 35 steps) instead of zero-shot. A fine-tune makes its labels in-sample, so there is one model pair (cov / ar) per walk-forward block, trained on the panel cut at the block's FIRST TRAIN DATE (`src/data_mani/chronos2_ft_repr.py`): every label it saw resolved before the block's training window. The block's train and test rows are represented by its own models; PCA basis on its train rows, 16 PCs per mode, rank-binned (`build_c2ft.py`). Features differ by block (`data/crypto/v2.0-c2ft/blocks.parquet`); `train.parquet` carries each row's test-block values for routing. Blocks 0 and 1 have no model (block 1's cut, 2020-07-29, leaves too few labels for the 134-step minimum): 0.5, routed to v2.0, so features start 2021-08-27. Same arms, controls and bar as E143. **Finding: NO IMPROVEMENT by the bar, REJECT on the MLP (routed dev CORR −0.0055, t −2.1), and harmful in substance on the other two:** LightGBM test CORR −0.0101 (t −4.4), payout −0.0108 (t −5.2), 5/5 seeds lower, worse than E143 zero-shot (−0.0060); ridge dev payout −0.0093 (t −2.5). LightGBM's dev looks positive (tuneFULL +0.0080 dev payout, t +2.0) and breaks on test (−0.0200, t −5.3): per-block bases change PC meaning every block (pooled IC 1/32 PCs at |t|≥2 vs E143's 13/32). F_COV less harmful than F_AR again; blends w=0 (ridge/MLP), LightGBM dev-picked 0.20 loses on test. Audit clean: same saved block model on cut vs uncut panel, 0 of 2,076,672 values differ (states stored from the fp32 merge differ from the fp16 reload by rounding only, cosine ≥ 0.99995). Closes Chronos-2 representations as features; only the fine-tuned FORECAST (quantiles) is left untested. PDF `results/reports/crypto/e144_c2ft/E144_does_c2ft_help_NO_IMPROVEMENT.pdf`. | DONE 2026-11-07 |
| **E145** | c2head: Chronos-2-Synth FORECAST-head features (zero-shot and walk-forward LoRA, covariate-aware and label-only, COV-minus-AR deltas), full feature-set evaluation | `e145_c2head_evaluation.py` | Registered 2026-11-07 before any fit; rules frozen in `experiments/e145_registration.md`. The head's 20-step q10/q50/q90 forecast of `target_binned_return_20` per row (`src/data_mani/chronos2_head.py`): 256-step context ending at D, labels after D−35 days blanked, 22 v2.0 features through D as past-only covariates, fp32 inference at revision 3607918. ZS_COV/ZS_AR frozen; FT_COV/FT_AR one LoRA fine-tune per block 2–11 whose training windows are built exactly like inference rows. Five summaries per branch (median mean/terminal/slope, mean q90−q10, asymmetry) plus ZS/FT COV−AR deltas, within-date percentile rank (primary) and five-bin (robustness) (`build_c2head.py`). Arms ZS/FT × COV/DELTA/FULL, ALL_HEAD, HEAD_ONLY, AR controls, P_HIST/P_WARM/P_OBS controls, P_NM look-ahead placebo, direct median legs blended with B0; primary dev window blocks 2–10, block 11 reused; 28 post-test dates unread. | registered 2026-11-07 |
| **E146** | c2dims: every binned hidden dimension of the four Chronos-2-Synth states (zero-shot / fine-tuned x cov / ar), full feature-set evaluation | `e146_c2dims_evaluation.py` | Registered before launch, 2026-11-11, before any E146 arm was fitted or scored. Dataset `crypto/v2.0-c2syn-c2ft-dims` (`src/data_mani/build_c2syn_c2ft.py --raw --no-pc`, stored on /scratch, symlinked): 22 shipped + 3,072 features, all 768 dims of the encoder states E143 (zero-shot) and E144 (per-block LoRA fine-tuned) reduced to 16 PCs, each rank-binned with Numerai's rule (zero-shot within date, fine-tuned within block x date; fine-tuned dims differ by block, `blocks.parquet`; blocks 0-1 have no fine-tuned model). Groups: shipped momentum (momentum, sharpe), reversion (RSI, bollinger), volume, volatility, price level (close), size (market cap); c2cov, c2ar, c2ftcov, c2ftar (768 each). Arms: FULL, ADDED_ONLY, F_C2COV / F_C2AR / F_C2FTCOV / F_C2FTAR, F_ZS, F_FT, FULL_H vs P_HIST, P_WARM, P_NM (look-ahead placebo), NO_MOM / NO_REV / NO_VOLU / NO_VOLA / NO_PRICE / NO_SIZE (v2.0 minus a shipped group, for scale), E143/E144 PC arms reused (dims vs PCs paired), tuneFULL vs tuned B0 (mlp alpha capped at 1). Protocol and bar as E144: 12 walk-forward blocks (130/22/130), fixed configs ridge / lgb_ktz / mlp_ktz (linear), 5 seeds lgb+mlp, target_binned_return_20, routed R_SLOT as the deployment score, NW lag 20 + block bootstrap, proxy MMC. Block 11 is reused evidence, not virgin; the 28 labelled dates after 2026-08-20 stay unread. Skipped and named: FULL minus each shipped group (3,000+ columns each) and FULL minus each single state. Audit: the raw-state truncation audits of E143/E144 re-run, plus re-binning the raw states on sample dates against the dataset. Report `results/reports/crypto/e146_c2dims/E146_does_c2dims_help_REJECT.pdf`. | DONE 2026-11-11: **REJECT on all three models.** Routed dev CORR vs v2.0: ridge −0.0514 (t −7.3), mlp −0.0489 (t −7.5), lgb −0.0095 (t −2.5); lgb test CORR −0.0190 (t −5.3), payout −0.0169 (t −5.0), 5/5 seeds lower; ridge/mlp negative in all 12 blocks. Every state is worse as 768 dims than as its own 16 PCs (ridge dev CORR −0.034..−0.041, t −7..−8). Fine-tuned ≈ zero-shot; both families together worse than zero-shot alone. lgb dev calmar 0.070 vs 0.023 (drawdown only) does not survive test. Retune, tenure, warm-up controls all against. Dims cost ridge/mlp about as much as deleting the shipped volatility group (the only shipped group worth > 0.008 dev CORR). Audit clean (truncation + re-binning identity 0 cells differ). Closes Chronos-2 hidden states as features (PCs or dims, zero-shot or fine-tuned). |
| **E147** | Group-only importance on `crypto/v2.0-c2syn-c2ft`: ridge + MLP tuned from scratch on each feature group alone | `e147_group_importance.py` | Registered before launch, 2026-11-11. 14 groups + ALL22 reference: MOM (momentum, sharpe 4), REV (RSI, bollinger 4), VOLU (4), VOLA (2), PRICE (close 4), SIZE (market cap 4), C2COV / C2AR / C2FTCOV / C2FTAR (16 PCs each), the same four as 768 raw dims (_FULL). Standalone importance, not marginal value over v2.0. Per block hyperparameters picked on E133's inner split (last 40 train dates after a 22-date purge): ridge alpha {1..10000}; MLP (32,16) activation {identity, relu} x alpha {0.01, 1} x lr {0.01, 0.05} picked on seed 0, refit seeds 0-4 (alpha 100 dropped: E137 collapse). 12 walk-forward blocks (130/22/130), target_binned_return_20, fine-tuned columns per block from blocks.parquet, missing = 0.5. Scored on all 468,000 keys (constant prediction = CORR 0) and on COMMON rows (fine-tuned state observed, blocks 2-11) where every group sees identical rows; proxy MMC; NW lag 20 + block bootstrap; block 11 reported apart. Report `results/reports/crypto/e147_group_importance/E147_group_importance.pdf`. | DONE 2026-11-11 (standalone importance, no verdict bar). Common rows, dev: VOLATILITY alone ≈ all 22 on CORR (ridge 0.1225 vs 0.1170, t +0.7; MLP +0.019, t +2.2) but short on test (−0.013, t −2.5), MMC and drawdown (calmar 0.013 vs 0.031). SIZE alone: CORR 0.064 but MMC 0.059 ≈ all 22, payout ties all 22, best calmar of any input (0.124 ridge). Chronos-2 groups: ridge CORR 0.066-0.091 alone (t 11-14), above every shipped group but volatility, yet MMC 0.005-0.029 and payout 0.047-0.062 below all 22: meta-model-redundant, consistent with E143/E144/E146. 768 dims vs 16 PCs: ridge +0.009..+0.012, MLP −0.018..−0.024 (overfits). cov > ar; fine-tuned ≥ zero-shot in 6/8 cells. Momentum/reversion/price/volume weak alone (ridge 0.010-0.043). Ridge ≥ MLP on 11/14 groups. |
| **E148** | c2resid: zero-shot Chronos-2 on the BTC+ETH beta-residual return path, with BTC / ETH / leave-one-out size-peer index as past covariates, against ridge on the same inputs, C2 residual-only, C2 raw-only and C2 grouped context | `e148_c2resid_evaluation.py` | Registered 2026-11-12 before any forecast was scored. Paths (`src/data_mani/c2resid_paths.py`): daily spot log returns clipped ±1; residual = r − b·r_BTC − c·r_ETH with b, c from OLS on [1, r_BTC, r_ETH] over days t−120..t−1 (≥60 obs); row D sees days ≤ D−1 (house submit lag); size bucket = shipped market_cap_avg_20d bin on D; peer index = leave-one-out mean raw return of the bucket's other coins. Signals = forecast of the residual return summed over days D+3..D+30 (target ≈ close(D+2)→close(D+30), E142): C2 zero-shot (chronos-2-synth, E143's revision, ctx 256, q50 span sum) on RAW path alone / RESIDUAL path alone / residual + BTC, ETH, peer (COV, the headline arm) / one group per (date, size bucket) holding every member's residual path + BTC + ETH (GRP); RIDGE = causal rolling ridge (refit every 22 panel dates on the trailing 365 days of rows whose 30-day label resolved, RidgeCV) on window sums of the same four series + residual×market interactions, target = the same forward residual sum, ranked within date; RESMOM = 20-day residual momentum (no fit). Dataset `crypto/v2.0-c2resid` = v2.0 + 6 binned signals (Numerai rule within date). Part A, standalone: each signal ranked per date vs target_binned_return_20, common rows; decomposition RES−RAW (residualising), COV−RES and GRP−RES (peer/market information), COV−RIDGE and GRP−RIDGE (C2 encoding), RIDGE−RESMOM. Part B, marginal: E146's protocol — 12 walk-forward blocks (130/22/130), fixed ridge / lgb_ktz / mlp_ktz, 5 seeds, routed R_SLOT deployment score, NW lag 20 + block bootstrap, proxy MMC; arms FULL, ADDED_ONLY, F_<signal> ×6, FULL_H vs P_HIST, P_NM look-ahead placebo, tuneFULL vs tuneB0. Block 11 reused, not virgin. Audit: rebuild paths from spot truncated at D−1 and re-forecast D on 3 dates. Report `results/reports/crypto/e148_c2resid/E148_does_c2resid_help_NO_IMPROVEMENT.pdf`. | DONE 2026-11-12: **NO IMPROVEMENT, REJECT on ridge and MLP.** Routed FULL dev CORR: ridge −0.0042 (t −2.2), mlp −0.0052 (t −2.2), lgb −0.0004; no single signal raises dev CORR on any model. C2 residual signals give small dev payout gains (RES +0.0039 t +2.7 ridge, +0.0049 t +3.0 lgb) but no more than the warm-up placebo (+0.0048 / +0.0107); FULL < P_WARM on dev CORR on all models. Decomposition (Part A, standalone): residualising is the one component that pays (RES − RAW payout +0.038 t +4.0 dev, +0.024 t +3.1 test, via MMC); BTC/ETH/peer covariates null (COV − RES −0.004 t −1.3); grouped context null (GRP − RES −0.001); C2 vs ridge on same inputs null (+0.014 t +0.5). Same ordering at ~1/10 size inside v2.0. Ridge signal: best standalone test CORR (0.051) but REJECT as a feature on all 3. Audit clean (spot cut at D−1: all 4 C2 arms identical on 3 dates; ridge inputs/peer/momentum 0 cells differ). |
| **E149** | vol-causal: the proposed realised-volatility block (feature_vol_*: 22 levels + 132 own-history ranks), full feature-set evaluation | `e149_vol_causal_evaluation.py` | Registered 2026-11-12 before any fit. Dataset `crypto/v2.0-vol-causal` (`src/data_mani/build_vol_causal.py`, built 2026-11-12 by another session): 22 shipped + 154 proposed columns — close-to-close 5-120d, EWMA λ .90-.99, Parkinson and Yang-Zhang 5-120d, each also ranked against the coin's own 7-365-day history; as-of D-1, causal wrong-coin identity gate, Numerai binning; builder truncation audit 0 of 258,000 cells differ at 5 dates. **Candidate = feature_vol_* only; the shipped feature_volatility_20d/60d are base features, in B0 and every arm.** Arms: FULL, ADDED_ONLY, F_LVL, F_TSR, F_SQRET / F_EWMA / F_PARK / F_YZ (levels + ranks of one family), F_SHORT / F_LONG, F_SWAP (FULL minus the shipped volatility), FULL_H vs P_HIST, P_WARM, P_NM, tuneFULL vs tuneB0. E146 protocol: 12 walk-forward blocks (130/22/130), fixed ridge / lgb_ktz / mlp_ktz, 5 seeds, routed R_SLOT deployment score, NW lag 20 + block bootstrap, proxy MMC; block 11 reused, not virgin. Report `results/reports/crypto/e149_vol_causal/` (PDF). | DONE 2026-11-12: REJECT on ridge (dev CORR −0.0138 t −2.9) and MLP (−0.0179 t −4.3), LightGBM null dev / harmed test; no sub-block helps; F_SWAP shows the block cannot replace the shipped volatility. |
| **E150** | macrovol: coin volatility x macro volatility regimes, matched covered-sample evaluation | `e150_macrovol_evaluation.py` | Registered before launch, 2026-11-12. `crypto/v2.0-vol-causal-macrovol` = vol-causal (176f) + 578 `feature_mvx_`: per macro source (gold GLD, S&P 500, DXY, 10y yield in bp, VIX level, VIX vol-of-vol) the interactions zc*zm, abs(zc-zm), zc in high (pm>=0.8) / low (pm<=0.2) macro-vol regimes for each of the 22 crypto estimators (causal probit-of-percentile normalisation, crypto 365 calendar days, macro 252 trading days, strictly prior), and coin-level cov/corr/beta/resvol/r2 over 60/120 trading days. Builders `build_vol_macrovol.py`, `fetch_macro_vol.py` (raw downloads kept, hashed). B_COV, VC_COV and every candidate trained on ONE common covered mask (69% of keys), routed R_SLOT through B_ALL, compared routed vs routed. Controls: scalar products, circular (504 trading days) and future (+20, look-ahead) placebos, LOO per source, with/without exposures, regime split, equal-budget retune; PARSIM chosen on dev by a frozen rule. Report results/reports/crypto/e150_macrovol/. | completed: corrected dev rerun REJECT on all three models; no PARSIM source |
| **E151** | volumealpha: point-in-time expected trading volume and liquidity (Trading Volume Alpha analogue), full feature-set evaluation | `e151_volumealpha_evaluation.py` | Registered before any build or fit, 2026-11-12; frozen rules in `experiments/e151_registration.md`. Dataset `crypto/v2.0-volumealpha-causal` (v2.0 + continuous-then-binned blocks, NaN where unavailable, `cov_*` masks). Sources, free only: Binance archive spot/perp klines incl. delisted, USD-M OI metrics and funding; 8 other CEX REST venues (currently listed only); Yahoo aggregator volume; Coin Metrics community and DefiLlama by cmcId (revised, not vintage); Snapshot votes (point-in-time). Blocks: TECH (paper's 8: lagged MA 1/5/22/252 of return and log dollar volume), EXT, FUND (size/surprise/leverage/value/beta/age), EVENT, XASSET, OOF volume forecasts (ridge/LGB/MLP/32-16-8, quarterly expanding refits), COST proxies. Arms B_COV, TECH, EXT, FUND, EVENT, XASSET, OOF, COST, RAW, MODEL, PARSIM + source ablations + sparse sub-family arms + P_HIST/P_WARM; ridge / lgb_ktz / mlp_ktz, routed R_SLOT vs B_COV on a matched mask. Named gaps: per-token DEX volume, unlocks/emissions, upgrades, airdrops, launches, index rebalances, liquidations. Report `results/reports/crypto/e151_volumealpha/`. | registered |
| **E152** | Strict-vintage volume-alpha admission and evaluation | `e152_volumealpha_causal_evaluation.py` | Registered 2026-11-12; `e152_registration.md`. Complements E151 without changing its fetchers: verified publication/revision timestamps, dated chain/contract mappings, contemporaneous USD conversion, source-family ablations; eight paper technical columns kept separate from extensions, fundamentals/events/cross-asset, OOF ridge/GBM/MLP volume forecasts and cost scenarios. Full base panel preserved; candidate and B_COV trained on identical covered rows; B_ALL rank-slot routing; 130/22/130 periods, equal searches, seeds 0–4. Forward test from 2026-11-13 after dev-only selection freezes. | DATASET COMPLETE 2026-11-13 (user-authorized inferred reconstruction): 524,596 base rows and every original column preserved exactly; 70 added features, three 8-feature source ablations (Binance, other CEX, Coinbase-direct-USD conversion). Two-day lag; 31 reused tickers excluded; 66 known bad HYPE/Gate bars removed; original retrievals and source hashes retained. 9,458,676 OOF model rows checked, no publication/train-boundary violations; 14,989 real-source technical cells recomputed at five cutoffs, zero mismatches. Causal per-fit training-range stability bounds eliminate extrapolation explosions. Ensemble volume R2 vs causal trailing baseline: 0.103 / 0.128 / 0.167 on direct 1/5/20-calendar-day targets. Reconstructed identity/availability, not verified historical vintages. No return-feature comparison or alpha claim. Report `results/reports/crypto/e152_volumealpha_causal_build/dataset_report.pdf`. |
| **E153** | TabNet with a wide KTZ-style search, raw + fenccausal | `e153_tabnet_ktz_crypto.sh` | Follow-up to E18 (default TabNet lost everywhere, t −2.0..−5.3). New `tabnet_ktz` wrapper opens every pytorch-tabnet axis: width 8–64 (n_a=n_d), n_steps 2–7, gamma 1.0–2.0, n_independent/n_shared 1–3, lambda_sparse 1e-6..1e-1, BN momentum 0.005–0.3, sparsemax/entmax, lr 3e-4..5e-2 (20 log levels), batch 1024–4096, ghost 128–512, epochs 30–150 (~114M cells). Random search n_iter 60, n_splits 3, seed 42, rolling 130/130, embargo 20, no eval_set (leak-free), CPU 4×4 threads. Arms e153_raw (v2.0, 22f) and e153_fenccausal (44f); pair against e18_* and CURUPIRA_02/SACI_04 / IA_01. | 2 | Registered and launched 2026-11-13 via src.cluster |
| **E154** | `numerai_crypto_c2synth_zero_shot_return_rank`: Chronos-2-Synth zero-shot forecast of daily log returns, summed over the target interval, ranked | `src/data_mani/c2synth_zeroshot.py --arm return_rank` | Frozen `autogluon/chronos-2-synth` @ 3607918a, per-token groups, fp32. Predeclared input: daily spot log returns through C-1; score = sum of q50 over days C+3..C+30 (interval close(C+2)->close(C+30)). Context {64,128,256} picked on validation (blocks 9-10, 260 rounds); test = block 11 (130 rounds, not virgin: many studies scored it). Baseline 28-day momentum; paired vs B0 lgb_ktz and CURUPIRA_05, block-20 bootstrap. Output `results/reports/crypto/e154_c2synth_return_rank/`. | 0 | **done 2026-11-13** (deepfour, unit nmr-e154_c2synth_return_rank, 4.2 min, 390 rounds, 89-91% coverage) — **NULL as a signal.** ctx 256 chosen on validation (CORR 0.022; 64/128 negative). Test CORR **0.0089** (NW t +0.36, 55% positive rounds) vs B0 0.129 (t +4.1). Common subset: A − B0 test **−0.098** (t −2.77, boot [−0.168, −0.023]); vs CURUPIRA_05 −0.083 (t −4.5). Beats mom28 on validation (+0.076 t +4.4) but not on test (+0.031 t +1.0). Reproduces E148 `raw` exactly at ctx 256. Report `results/reports/crypto/e154_e155_c2synth_comparison/report.pdf`. |
| **E155** | `numerai_crypto_c2synth_zero_shot_bin_forecast`: Chronos-2-Synth zero-shot forecast of each token's own target-bin label series | `src/data_mani/c2synth_zeroshot.py --arm bin_forecast` | Same checkpoint, rounds, universe and scoring as E154. Labels on their original dates, daily calendar with gaps as NaN, label d usable at round C iff d <= C-32 days; forecast from the last available label date through C (h = C-L, skip h>64), continuous q50. Baselines: last label, mean of last five. Predeclared blend rule with E154 (rank corr < 0.5 and blend beats better arm on validation). Output `results/reports/crypto/e155_c2synth_bin_forecast/`. | 0 | **done 2026-11-13** (deepsix, unit nmr-e155_c2synth_bin_forecast, 4.7 min, 87-88% coverage) — **NULL.** ctx 256 chosen on validation (CORR 0.012). Test CORR **0.0148** (t +1.39) vs B0 0.129. Common subset B − B0 test **−0.093** (t −2.07, boot [−0.173, +0.008]). No edge over its baselines on test (vs last label +0.005 t +0.5; vs mean of 5 −0.002). Blend rule: rank corr with E154 0.38 (<0.5) but blend validation CORR 0.0297 < A's 0.0352 → not complementary, blend NOT scored on test. Same report as E154. |
| **E158** | tabnet_ktz hyperparameter importance map | `e158_tabnet_ktz_hparams_crypto.sh` | E153's grid/walk-forward, keeping EVERY candidate: new opt-in `run_forecast --save-cv-results` (per-fold inner-CV CORR + MMC on the real meta model, `results/tuning/crypto/tabnet_ktz/<run>_cv.csv`) and `--reseed-per-refit` (fresh draw per refit; E153 re-scored the same 60 points each refit). n_iter 120, n_splits 3, seeds 42/43, raw + fenccausal (4 arms). Analysis `src.analysis.hparam_importance`: eta² vs permutation null, Kendall W across refits, interactions, RF-surrogate cross-check, sensitivity curves, suggested walk-forward grid. Goal: narrow the space for later walk-forward searches. (Drafted as E154; that id was taken by a concurrent session.) | 4 | Registered and launched 2026-11-13 via src.cluster |
| **E183** | TabNet with epochs tied to model size (TV schedule) | `e183_tabnet_tv_crypto.sh` | New `tabnet_ktz_tv` (src/prediction/tabnet_tv.py, isolated from tabnet.py so running E158 workers are untouched): no max_epochs axis; each candidate trains E = floor(0.2·P^0.8)+10 epochs, P = trainable params of ITS architecture on THIS dataset's columns (asserted equal to the fitted network). Space: width 8–16 (n_a=n_d), steps 2–5, GLU 1–2, rest as tabnet_ktz → TV 136–837 epochs raw / 174–906 fenccausal (E153/E158 used 30–150, below TV everywhere). E158 capture (--save-cv-results, --reseed-per-refit), n_iter 40, n_splits 3, seeds 42/43, raw + fenccausal; 2 torch threads/worker (1.1–2.3 s/epoch measured). Launched with a one-off `src.cluster --extra-cores 10` (new opt-in flag, default 0). | 4 | Registered and launched 2026-11-17 |
| **E187** | tabnet_ktz_tv on every deployed dataset | `e187_tabnet_tv_deployed_crypto.sh` | E183 recipe (TV epochs tied to model size, n_iter 40, seed 42) on the 8 deployed feature sets not yet covered: tenc, interact, cumlag3, onehot_roll, tenc_onehot, allfeat, allfeat_pp, allfeat_pp_fenc; raw/fenccausal reused from E183; allfeat_pp_lag3 skipped (no research dataset). | 8 | Registered and launched 2026-11-18 |
| **E159** | volumealpha routed: does E152's 70-column volume-alpha block help v2.0? | `e159_volumealpha_evaluation.py` | Registered before launch, 2026-11-13. Dataset `crypto/v2.0-volumealpha-causal` (E152 inferred reconstruction, 2-day lag, not verified vintages). E149's routed protocol (E138/E139 machinery): FULL 22+70, ADDED_ONLY, F_TECH/F_LIQ/F_FUND/F_XASSET/F_FCST/F_COST, F_RAW vs F_MODEL, FULL_H vs P_HIST, P_WARM, P_NM, tuneFULL vs tuneB0; ridge / lgb_ktz / mlp_ktz, R_SLOT routed on each arm's coverage; test = block 11 (not E152's forward test). Chain + email: `e159_orchestrate.py`. Report `results/reports/crypto/e159_volumealpha/`. | running |

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

## E104 — day-indexed minibatches do NOT differ from row-shuffled ones; batch SIZE does

**Question.** Same model, same data, same walk-forward — does it matter whether a
minibatch is 1024 shuffled *rows* or every stock of a few shuffled *days*? The model
is the from-scratch torch MLP with sklearn's exact initial weights that tracks
`mlp/e33_fenc_a64_32_16_identity_lre3_al100` (identity `(64,32,16)`, alpha 100, lr
1e-3, fenccausal 44f, rolling 130/130, embargo 20). 8 arms × 10 seeds (42–51), all
n=510 on one window, so calmar is comparable throughout.

**Gate.** `e104_row_s42` reproduces `torch_scratch/e33repl_scratch_SKINIT_*` bit for
bit (max |Δcorr| over 510 periods = 0.0e+00).

| arm | a batch is | L2 ÷ | const seeds | CORR mean | CORR median | payout mean | calmar (median) | steps | \|W\| |
|---|---|---|---|---|---|---|---|---|---|
| `row` (baseline) | 1024 rows | rows in batch | 1 | 0.06411 | 0.11283 | 0.03984 | 0.131 | 2501 | 5.8e-17 |
| `rowfix` | 1024 rows | 1024 | 1 | 0.06434 | 0.12958 | 0.04219 | 0.248 | 2492 | 1.6e-15 |
| `row900` | 900 rows | 1024 | 1 | 0.06117 | 0.11633 | 0.03838 | 0.185 | 2600 | 2.6e-16 |
| **`day3`** | 3 days (900 rows) | 1024 | 1 | 0.06110 | 0.11859 | 0.03845 | 0.170 | 2606 | 1.8e-16 |
| `day3lit` | 3 days | rows in batch | 1 (+2 partial) | 0.04373 | 0.04040 | 0.02507 | 0.075 | 2530 | 2.0e-17 |
| `row300` | 300 rows | 1024 | **10** | 0 | 0 | 0 | — | 4121 | 7.3e-26 |
| `day1` | 1 day (300 rows) | 1024 | **10** | 0 | 0 | 0 | — | 4098 | 1.8e-34 |
| `day1lit` | 1 day | rows in batch | **10** | 0 | 0 | 0 | — | 3585 | 3.1e-55 |

`const` = seeds predicting one constant in every period; `steps` = optimiser steps
before the stop; `|W|` = median norm of the net linear map the stack composes to.

Paired per period, Newey-West lag 20, on the seed-mean payout delta. `t` is over
time with the ten seeds taken as given; `seed t` treats each seed's mean delta as
one observation (9 df) and is the bar a claim about the *scheme* has to clear.

| comparison | what it isolates | Δ payout | NW se | t | seed t | seeds won |
|---|---|---|---|---|---|---|
| `day3` − `row` | the new scheme vs the baseline | −0.00139 | 0.00171 | −0.81 | −0.55 | 6/10 |
| `day3` − `row900` | **composition only** (same size, same steps) | +0.00007 | 0.00074 | +0.09 | +0.07 | 5/10 |
| `day1` − `row300` | composition only, 300 rows | 0 exactly | — | — | — | both constant |
| `row900` − `rowfix` | batch size only, 900 vs 1024 | −0.00381 | 0.00054 | −7.03 | −1.16 | 3/10 |
| `row300` − `rowfix` | batch size only, 300 vs 1024 | −0.04219 | 0.00537 | −7.85 | −2.40 | 2/10 |
| `rowfix` − `row` | the L2 pin alone | +0.00235 | 0.00121 | +1.95 | +0.62 | 6/10 |
| `day3lit` − `row` | the literal port of the day scheme | −0.01477 | 0.00216 | −6.85 | −2.23 | 3/10 |

- **What a batch is made of has no detectable effect.** At an identical batch size,
  step count and objective, days vs rows is +0.00007 payout, interval about
  [−0.0014, +0.0015]. At 300 rows both are a constant on every seed.
- **Batch size does, through the number of optimiser steps.** Every arm is decaying
  toward the origin; a smaller batch takes more steps before the stop (4,100 against
  2,500) and the residual map is 10–40 orders of magnitude smaller. At seed 42 the
  surviving prediction holds 44 distinct values per day under `rowfix`, 23 `row900`,
  14 `day3`, 10 `row`, 4 `day3lit` and 1 at 300 rows — the float64 spacing above 0.5
  is 1.1e-16 and the whole score lives in the last few of those steps.
- **The seed picks the outcome, not the scheme.** `row`, `rowfix`, `row900` and `day3`
  land in the same state on every seed: five work (~0.13), 46 is a constant in all,
  47 and 49 are sign-flipped in all.
- **The penalty has to be pinned once batches stop being a fixed size.** sklearn
  divides the L2 term by rows-in-batch, so 900-row day batches run a 14% stronger
  penalty; that alone costs 0.0148 payout (`day3lit`) where the pinned `day3` costs
  nothing.
- **One seed gets this wrong.** On seed 42 alone `day3` − `row` is +0.00824 payout,
  t +4.18. `rowfix` − `row` on the same seed is +0.00753, t +3.63, with no day
  batching at all, and days vs rows there is +0.00105, t +0.63. Across ten seeds the
  L2-pin gain itself is seed t +0.62.

**Caveats.** This arm does not learn in the usual sense — alpha 100 shrinks the map
to float noise — so the null is about a model decaying to zero, and says little
about day batches for one that fits. Every day in all four training windows holds
exactly 300 stocks, so a changing stock count was exercised by the runner's
self-test (thinned to 1–300 a day) and by the design choices it forced, not by the
data. Forecasts were written for seed 42 only (11 MB each); the report against the
book is on its 470-period common window.

Artifacts: `experiments/e104_daybatch_torch_mlp_crypto.py` (`selftest`, `report`),
`results/{per_period,metrics}/crypto/torch_scratch/e104_*`, `logs/e104_report.txt`,
`results/reports/crypto/report_20261010T062519Z.html`.

### E104 at 100 seeds — the distributions of CORR, MMC and dd(pay)

`row`, `row900` and `day3` extended to seeds 42–141, the same seeds in every arm
(270 more runs, 19 min; `E104_TAG=s100`). All n=510.

| arm | const | negative | CORR ≥ 0.10 | CORR q25 / median / q75 | CORR mean | MMC q25 / median / q75 | MMC mean | dd(pay) q25 / median / q75 |
|---|---|---|---|---|---|---|---|---|
| `row` (baseline) | 14 | 25 | 30 | −0.0001 / 0.0181 / 0.1226 | 0.03983 | −0.0010 / 0.0109 / 0.0615 | 0.02056 | 0.35 / 2.88 / 7.90 |
| `row900` | 20 | 22 | 35 | 0.0000 / 0.0276 / 0.1296 | 0.04518 | 0.0000 / 0.0079 / 0.0647 | 0.02202 | 0.35 / 1.47 / 6.83 |
| `day3` | 20 | 22 | 35 | 0.0000 / 0.0250 / 0.1277 | 0.04371 | 0.0000 / 0.0093 / 0.0648 | 0.02201 | 0.35 / 1.97 / 7.30 |

Counts are seeds out of 100. dd(pay) is over the seeds that are not a constant
predictor: a constant has no drawdown because it places no bet.

Paired by seed, `day3` minus the row arm (t on 99 df; Wilcoxon signed-rank, two-sided):

| base | metric | mean | median | seeds up / same / down | paired t | Wilcoxon p |
|---|---|---|---|---|---|---|
| `row` | CORR | +0.00388 | +0.0003 | 54 / 12 / 34 | +1.51 | 0.118 |
| `row` | MMC | +0.00145 | 0 | 49 / 12 / 39 | +1.20 | 0.229 |
| `row` | dd(pay) | −0.374 | 0 | 41 / 12 / 47 | −1.06 | 0.199 |
| `row900` | CORR | −0.00148 | 0 | 39 / 19 / 42 | −0.89 | 0.139 |
| `row900` | MMC | −0.00001 | 0 | 37 / 19 / 44 | −0.01 | 0.447 |
| `row900` | dd(pay) | +0.198 | 0 | 49 / 19 / 32 | +1.02 | 0.059 |

- **Against its size-matched twin the day scheme is the same distribution.** Same
  count of constants (20) and negatives (22), and 93 of 100 seeds land in the same
  state (constant / negative / positive) under both.
- **Against the baseline as it stands, `day3` is a little ahead and not reliably
  so.** `row900` has the same lead over `row`, so it belongs to the batch size and
  the pinned penalty, not to the days. On the first 10 seeds this pair read −0.0014
  payout; on 100 it reads +0.0018. The sign moved with the sample.
- **Ten seeds flattered this arm.** Seeds 42–51 have a median CORR of 0.113; all
  100 have 0.018. Only 30 of 100 baseline seeds reach CORR 0.10, 14 are a constant
  and 25 are negative. The seed-42 row (0.1297), which is the one that tracks the
  sklearn arm, sits at the 79th percentile of its own configuration.
- dd(pay) spans two orders of magnitude between seeds (0.2 to 30), which is the
  difference between a working seed and a sign-flipped one, not between schemes.

Artifacts: `results/plots/crypto/e104_seed_boxplots_s100.png` (mode `boxplot`),
`logs/e104_seed_metrics_s100.csv`, `logs/e104_boxplot_stats_s100.txt`,
`logs/e104_report_s100.txt`.

## E105 — annealing the learning rate rescues the day3 arm; `OneCycleLR` is best

**Question.** E104's `day3` arm unchanged — torch from scratch, sklearn init,
identity `(64,32,16)`, alpha 100, a batch = every stock of 3 shuffled days — with
only the learning rate over training varied. 10 arms × 100 seeds (42–141, shared by
every arm), all n=510. Seven of torch's own schedulers peaking at 1e-3 over a
60-epoch horizon, and two constant lower rates as controls, because every schedule
also lowers the average rate.

**Gate.** `e105_const` reproduces `e104_day3` bit for bit on all 100 seeds.

| arm | const seeds | negative | CORR ≥ 0.10 | CORR median | CORR mean | MMC mean | payout mean | dd(pay) | calmar | last lr |
|---|---|---|---|---|---|---|---|---|---|---|
| `const` 1e-3 (baseline) | 20 | 22 | 35 | 0.025 | 0.0437 | 0.0220 | 0.0264 | 1.97 | 0.014 | 1e-3 |
| `lr5e4` constant | 0 | 29 | 37 | 0.055 | 0.0499 | 0.0257 | 0.0307 | 4.46 | 0.010 | 5e-4 |
| `lr25e5` constant | 0 | 25 | 41 | 0.055 | 0.0548 | 0.0283 | 0.0338 | 3.76 | 0.012 | 2.5e-4 |
| `step` StepLR | 0 | 12 | 50 | 0.099 | 0.0785 | 0.0401 | 0.0479 | 2.29 | 0.026 | 8.6e-5 |
| `exp` ExponentialLR | 0 | 9 | 56 | 0.112 | 0.0905 | 0.0463 | 0.0553 | 1.66 | 0.036 | 2.8e-5 |
| `cosine` CosineAnnealingLR | 0 | 13 | 47 | 0.096 | 0.0751 | 0.0384 | 0.0459 | 2.55 | 0.023 | 2.7e-5 |
| `warmcos` warm-up + cosine | 0 | 10 | 58 | 0.110 | 0.0887 | 0.0451 | 0.0539 | 1.50 | 0.048 | 1.0e-5 |
| `restarts` warm restarts | 5 | 27 | 38 | 0.041 | 0.0498 | 0.0253 | 0.0303 | 2.65 | 0.012 | 9.1e-4 |
| **`onecycle` OneCycleLR** | 0 | **7** | **69** | **0.118** | **0.1021** | **0.0517** | **0.0620** | **1.16** | **0.067** | 4e-9 |
| `plateau` ReduceLROnPlateau | 0 | 24 | 39 | 0.059 | 0.0534 | 0.0279 | 0.0332 | 4.12 | 0.010 | 1.3e-4 |

Counts are seeds out of 100. dd(pay) and calmar are medians over each arm's own
non-constant seeds; `last lr` is the median rate at the stop.

Paired by seed (t on 99 df). Nine arms are tested against `const`, so the bar is
about |t| ≥ 2.9.

| arm | Δ payout vs `const` | seed t | seeds up | Δ payout vs `lr25e5` | seed t |
|---|---|---|---|---|---|
| `onecycle` | +0.0356 | +9.18 | 79 | +0.0282 | +7.86 |
| `exp` | +0.0289 | +7.97 | 81 | +0.0215 | +6.32 |
| `warmcos` | +0.0275 | +7.14 | 77 | +0.0202 | +6.20 |
| `step` | +0.0215 | +6.07 | 75 | +0.0142 | +4.46 |
| `cosine` | +0.0195 | +5.46 | 73 | +0.0121 | +3.81 |
| `lr25e5` | +0.0074 | +2.61 | 64 | — | — |
| `plateau` | +0.0068 | +2.43 | 55 | −0.0006 | −0.24 |
| `lr5e4` | +0.0043 | +1.68 | 52 | — | — |
| `restarts` | +0.0039 | +2.11 | 59 | −0.0035 | −1.31 |

- **Five schedulers beat the baseline by a wide margin, and beat a plain lower rate
  too.** So it is the shape of the schedule, not the lower average: `lr25e5` spends
  the same summed learning rate as `step`, `cosine`, `warmcos` and `onecycle`
  (1.2–1.35) and gets between a fifth and a third of their gain.
- **`OneCycleLR` is the best arm**, ahead of `exp` (+0.0067 payout, t +3.13) and
  `warmcos` (+0.0080, t +6.93). `exp` and `warmcos` are level (t +0.71); both beat
  `step`, which beats `cosine`.
- **What orders the arms is the rate training ENDS on.** Spearman of last lr against
  mean payout across the ten arms is −0.93 (summed lr: −0.75). The five winners end
  at 9e-5 or below; `plateau`, `restarts` and the constant rates end at 1.3e-4 or
  above and sit with the controls.
- **Two different failures, two different cures.** Any lower rate removes the
  constant predictors (20 → 0): the baseline's map ends at 1.7e-16 and its
  predictions hold 44 distinct values a day, the others end at 1e-12 or larger and
  hold all 300. Only annealing toward zero removes the sign flips: of the baseline's
  22 negative seeds `onecycle` turns 19 positive, `warmcos` 18, `exp` 17, `step` 14,
  `cosine` 13 — against 5–6 for the constant rates and 2 for `restarts`.
- **Drawdown improves in distribution, not seed by seed.** On the 80 seeds where the
  baseline is a live model, median dd(pay) goes 1.97 → 0.96 (`onecycle`), 0.98
  (`warmcos`), 1.10 (`exp`) and calmar 0.014 → 0.088 / 0.067 / 0.065; but `onecycle`
  is lower on exactly 40 of the 80. The gain is the rescued sign-flipped seeds
  (dd ~20 → ~1), not shallower drawdowns on seeds that already worked. The constant
  lower rates make drawdown worse (median 2.9–3.0, higher on 48–50 of 80).

**Caveats.** The objective is unchanged and its minimiser is still the zero map; a
schedule helps by stopping the optimiser short of it, so this is early stopping in
learning-rate time, not a better fit, and says nothing yet about a model that is
not over-regularised. `ReduceLROnPlateau` at the stop rule's own 1e-4 never fires
(torch compares with the last accepted best, the stop rule with the running best)
and was the baseline bit for bit; it runs at 1e-3. Horizons are 60 epochs because
constant-rate training stops at ~53; `OneCycleLR` also cycles beta1. The first 50
seeds flattered the schedulers (`onecycle` 1 negative of 50, then 6 of the next 50).

Artifacts: `experiments/e105_lr_schedulers_daybatch_crypto.py` (`gate`, `report`,
`boxplot`), `results/{per_period,metrics}/crypto/torch_scratch/e105_*` (no
forecasts), `logs/e105_report.txt`, `logs/e105_seed_metrics.csv`,
`results/plots/crypto/e105_seed_boxplots.png`.

## E106 — residualising on the meta model raises payout, and a 5-day-old meta model keeps most of it

No training: the −CORR network's stitched predictions (identity 64-32-16, alpha 0, epoch 500, seeds 42/43) are moved off the meta model per day, `rank(gauss(rank(p)) − s·β·gauss(rank(m)))`, and scored against the same-day meta model on the 547 meta days. The gate holds: at s = 0 the source per-period CORR and MMC are reproduced to 1e-12. Reaching that needed the target cast back to float32, because a float64 target moves `numerai_corr` by about 4e-9 on identical ranks.

**Before residualising:**
- The network's per-day rank correlation with the meta model is 0.52, and β has a median of 0.53.
- The meta model is published within about 1.5–2.5 days: a file downloaded on Saturday 2026-08-08 at 13:30 ended at Thursday 2026-08-06. So the ≥5-day arm is the safe deployable one.

**Best strength per arm.** The payout gain over s = 0 is paired, Newey-West lag 20, given as seed 42 / seed 43:

| meta model used | best s | payout gain | t | strengths with t ≥ 2 |
|---|---|---|---|---|
| same day (oracle) | 0.75 / 0.70 | +0.0081 / +0.0075 | +3.6 / +3.5 | all |
| ≥ 1 day old | 0.80 / 0.75 | +0.0073 / +0.0068 | +3.3 / +3.2 | all |
| ≥ 5 days old | 0.80 / 0.80 | +0.0069 / +0.0064 | +2.9 / +2.8 | all |
| ≥ 20 days old | 0.85 / 0.80 | +0.0053 / +0.0050 | +1.7 / +1.8 | s ≤ 0.70 |
| ≥ 30 days old | 0.75 / 0.70 | +0.0021 / +0.0019 | +0.7 / +0.7 | none |

- **The mechanism:** MMC re-ranks and gaussianises before it orthogonalises, and does not rescale, so removing the meta component first scales up the part MMC pays on. Under the oracle at s = 0.75, CORR falls 0.145 → 0.110 (t −6.7) and MMC rises 0.070 → 0.082 (t +5.4).
- **It buys level, not smoothness.** At the 5-day lag, calmar holds up to s ≈ 0.5. At s = 0.75, dd(pay) goes 1.51 → 1.78 and calmar 0.056 → 0.051. The 20-day lag lowers drawdown instead (dd 1.45, calmar 0.061 at s = 0.5).
- **Against the book** (470-day common window): the 5-day, s = 0.75 arm pays 0.0919, tying the best single submitted model SACI_06 (0.0920, t −0.02). Its calmar is 0.051 against 0.087.
- **Caveats:** one source model; about 27 independent windows; the strength is picked from 21 steps, although every s ≤ 0.7 clears t ≥ 2 up to a 20-day lag. `src/live` does not download the meta model, so deploying this needs a per-round fetch.
- **Artefacts:** `results/reports/crypto/e106_meta_residualise.html` (frontier and sweep), `e106_board.html` (against the book), and `e106_meta_residualise/sweep.csv`.
- **Why it works** (2026-10-19, the walk-forward diagnostics rebuilt on MMC minus ridge: `results/reports/crypto/torchscratch_{corr,mse}_500ep_wf_s42_walkforward_mmc.html`).
  - On the 547 meta days, the networks rank coins almost exactly like the same-window ridge (per-day correlation 0.97). Their departure from it leans toward the meta model (correlation +0.09).
  - MMC does not pay for that part. Daily, MMC gap ≈ −0.0024 + 0.71 × CORR gap, so the −CORR network's CORR edge over the ridge (+0.0054, t +2.5) is only +0.0014 on MMC.
  - Both networks beat the ridge on CORR in 4–5 of the 5 covered folds, but on MMC in 2 (r09 and r11 for both).
  - Residualising removes exactly that meta-leaning part.

## E107 — residualising every model does not make them converge; the gain scales with each model's meta overlap

E106's deployable transform (s = 0.5, the latest meta model at least 5 days old) applied to 38 models on the 470 days they all share (2024-07-25 → 2026-05-13): the walk-forward ridge, the MSE and −CORR networks (seeds 42/43), and every 20-day slot in the registry. SACI_03 and CURUPIRA_05_60 are 60-day and were left out. At s = 0 the pipeline equals the repo scorer on each model's own forecast, exactly.

Sixteen blends' stored per-day series are not reproduced by their own stored forecasts: by up to 7e-3 a day and at most 4e-5 on the mean, through near-ties in rank averages written at different times. The gate was therefore set against the rescored forecast, not the stored series.

**Every model gains.**
- Payout gains run from +0.0030 (CURUPIRA_05, SACI_02) to +0.0104 (CU_SA_IA_BO_01), all at t ≥ 2.2.
- Median gain: blends +0.0075, single models +0.0063, ridge and networks +0.0058.

**They do not converge.**

| | before | after |
|---|---|---|
| payout range across models | 0.0319 | 0.0384 |
| payout std across models | 0.0078 | 0.0092 |
| model pairs differing at \|t\| ≥ 2 | 27% | 29% |

- The two seeds of one network differ at |t| 0.4–0.7.
- The before/after ranking correlates 0.98.

**The gain scales with the meta overlap.** Each model's gain tracks β, its projection on the lagged meta model (Spearman 0.82; gain ≈ −0.010 + 0.033 × β), and better models carry more of it. So residualising stretches the ranking:
- CU_SA_IA_BO_01 goes from 0.1071 to 0.1175, the top of the book. Residualised, it beats 24 of the other 37 at t ≥ 2.
- The residualised networks (0.090–0.091) stay level with the residualised walk-forward ridge (0.087; t −1.0 and −1.5), and about 0.027 below CU_SA_IA_BO_01 (t +4.4 to +4.9).

**Cost:** calmar falls for 34 of 38 models (median 0.088 → 0.074) and median dd(pay) rises from 1.09 to 1.39. For CU_SA_IA_BO_01, calmar goes 0.093 → 0.082. On the book, s = 0.5 trades depth for level, so a live slot's strength should be chosen on calmar too.

**Diversity:** the mean pairwise correlation between models' predictions falls from 0.81 to 0.76.

**Verdict:** model choice is not settled. Residualisation is a layer that adds to every model and stretches the order, not one that makes models interchangeable.

**Artefacts:** `results/reports/crypto/e107_residualise_everything.html` (figure, reading and table) and `e107_board.html` (the 38 residualised models against the book).

**Combinations, and what went live (2026-10-21).** The residualised models, together with the plain ones, were screened as 2-, 3- and 4-member per-period rank averages, then priced with `src.run_ensemble` over 470 periods.
- **Pool A (residualised only):** the best combination is `meta_resid/e107px_cu_sa_cu_sa` = 50/50 RESID(CU_SA_IA_BO_01) + RESID(CU_SA_IA_BO_MA_01). It has payout 0.1187, the highest measured on the book, at +0.0110 (t +3.73) against CU_SA_IA_BO_MA_01. It costs depth: dd(pay) is 1.39 against 1.10, calmar 0.085 against 0.098, and CORR is lower.
- **Pool B (residualised, plain and 60-day mixed):** the best combination is `meta_resid/e107pxB_curu_IA_0_MA_F_CURU` = 1/4 each of RESID(CURUPIRA_05), IA_01, MA_FENC and CURUPIRA_05_60. It has payout 0.1121, p.sharpe 1.663 and dd(pay) 0.907, at +0.0044 (t +1.76) against CU_SA_IA_BO_MA_01.

Both are deployed as slot blends. Pool A is `CU_SA_IA_BO_MA_R_01` (account slot formerly `curupira_02`) and Pool B is `CU_R_IA_MA_60_01` (formerly `curupira_03`). The live transform is `src.prediction.meta_resid`, and `src/tests/test_meta_resid.py` pins it to this script's copy and to the rank-average construction.

## E109 — foundation-model representations help no model on any base; they hurt the ridge and the raw MLP

The 41-column `feature_fm_` block (the 9 representations of `features-foundation.md`, built causally) against each base without it. Each pair runs on the same dataset and the same 510 periods (2024-07-25 → 2026-07-08), so the block is the only difference. Paired Newey-West at lag 20.

**Payout, with minus without (t):**

| model | raw (22) | fenccausal (44) | allfeat_pp (812) |
|---|---|---|---|
| ridge | **−0.0178 (−2.33)** | **−0.0170 (−2.35)** | −0.0012 (−0.41) |
| lgb_ktz | −0.0046 (−0.61) | −0.0037 (−0.52) | −0.0050 (−1.29) |
| catboost_ktz | −0.0038 (−0.79) | −0.0060 (−1.00) | −0.0043 (−1.22) |
| rf_ktz | −0.0053 (−0.76) | −0.0026 (−0.66) | −0.0044 (−1.24) |
| mlp_ktz | **−0.0294 (−2.81)** | −0.0090 (−0.86) | −0.0083 (−1.30) |

- Payout falls in all 15 pairs and MMC in all 15. Nothing rises.
- CORR rises in three pairs (lgb raw +0.001, lgb fenccausal +0.003, rf fenccausal +0.004), all at |t| < 1.
- The CORR damage is concentrated in the same three arms: ridge raw −0.028 (t −3.84), ridge fenccausal −0.028 (t −4.05), MLP raw −0.037 (t −3.46).
- The trees are flat: they take the block and gain nothing. The linear ridge and the MLP give it weight and lose.
- On allfeat_pp's 853 columns the block is diluted and every effect is small.

**The prior did not hold, and why it probably did not.** The pre-run EDA found TabPFN means adding partial IC +0.04 (t +2.6 to +3.6) over the fenccausal ridge. That EDA used E19's mean. The causal mean scores CORR 0.1095 on the same 510 dates against E19's 0.145, with a per-date rank correlation of only 0.745 between the two. Its block-mean CORR swings with sd 0.082 over 27 refresh blocks. Each 20-date refresh draws a new 4,096-row context subsample, so what the column means shifts within every 130-date downstream training window. That fits a fixed linear weight failing; it is a plausible mechanism, not a tested one. It also says E19's 0.145 is a favourable draw of only 4 contexts, which matters for `TUPA_01`, which points at E19.

**Checks.**
- The replay of E19's contexts reproduces its saved prediction to 2.9e-6, so the extraction (quantiles, bin probabilities, per-estimator means, embeddings) sits on the same model.
- Each context ends 20 dates before its block. 8 causality tests are in `src/tests/test_foundation_repr.py`.
- The "without" arms reproduce deployed backtests exactly: `e109_rf_ktz_fenc` = `MA_FENC`, `e109_catboost_ktz_fenc` = `IAUARETE_01`, `e109_ridge_raw` = `CURUPIRA_02`.
- Each variant minus the block is asserted identical to its base.
- TabPFN coverage is 100% of the rows any model reads; the 3% gap is the 20 embargo dates the loader drops. TS coverage is 85%, from the spot feed.

**Verdict:** do not deploy. As built, the representations carry no incremental payout for any model class here.

**Not done.** Two follow-ups would explain the failure. A row ablation on ridge/fenccausal would find which part of the block does the harm; the note's proposed first study (rows 2–4 alone) is one arm of it. Stabilising TabPFN would test the mechanism, by averaging several context subsamples per refresh or using a larger context.

**Artefacts:** `results/reports/crypto/e109_fmrep.html` (all 30 arms against the book, on the book's 470-period window) and `logs/e109_report.txt` (paired with/without on 510). Builders: `src/data_mani/{foundation_repr,foundation_repr_ts,build_fmrep}.py`, cache `data/crypto/fmrep_cache/`.

## E113 — E109's damage to the ridge comes from the time-series block; TabPFN's own predictions are neutral

Ridge on `crypto/v2.0-fenccausal-fmrep`, the 44 base features always kept, one sub-block of E109's `feature_fm_` block added per arm. Each arm is paired (Newey-West lag 20, 510 periods) against the ridge with no block, `e109_ridge_fenc` (payout 0.08622). The whole block gave 0.06923 (t −2.35).

| sub-block (rows) | cols | CORR | MMC | payout | Δpayout vs no block (t) | ΔCORR (t) | calmar | burn |
|---|---|---|---|---|---|---|---|---|
| none (`e109_ridge_fenc`) | 0 | 0.15175 | 0.07105 | 0.08622 | — | — | 0.062 | 60 |
| TabPFN point preds (1) | 3 | 0.14815 | 0.07247 | 0.08729 | +0.0011 (+0.37) | −0.0036 (−1.19) | 0.065 | 68 |
| TabPFN rows 1–4 | 12 | 0.14377 | 0.07331 | 0.08769 | +0.0015 (+0.31) | −0.0080 (−1.88) | 0.072 | 67 |
| TabPFN rows 2–4 (the note's first study) | 9 | 0.14523 | 0.07214 | 0.08667 | +0.0004 (+0.10) | −0.0065 (−1.46) | **0.079** | 56 |
| TabPFN embedding PCs (5) | 8 | 0.13825 | 0.06794 | 0.08176 | −0.0045 (−0.69) | **−0.0135 (−2.13)** | 0.052 | 82 |
| per-block heads + contexts (6–7) | 7 | 0.14410 | 0.06637 | 0.08078 | −0.0054 (−1.46) | **−0.0076 (−2.00)** | 0.050 | 67 |
| TimesFM / Chronos-2 on spot (8–9) | 14 | 0.13892 | 0.06188 | 0.07577 | **−0.0105 (−2.63)** | **−0.0128 (−3.23)** | 0.052 | 66 |
| whole block (`e109_ridge_fenc_fm`) | 41 | 0.12342 | 0.05689 | 0.06923 | **−0.0170 (−2.35)** | **−0.0283 (−4.05)** | 0.041 | 113 |

- **The time-series block is the main cause.** TimesFM and Chronos-2 on each coin's own price path, rows 8–9, are the only sub-block that hurts payout at |t| ≥ 2 on their own. They cost MMC −0.0092 (t −2.55) as well as CORR.
- **The embeddings and the heads add smaller harms.** Both cost CORR at |t| ≈ 2 but payout only at |t| < 1.5.
- **The harms roughly add up.** The single-block payout deltas sum to −0.019, against −0.017 for the whole block.
- **TabPFN's own outputs are neutral on payout.** Point predictions, distribution, bin probabilities and disagreement (rows 1–4) pay +0.0004 to +0.0015 (all |t| < 0.4).
- **They trade a little CORR for MMC and depth.** CORR falls 0.004–0.008 (|t| < 1.9) and MMC rises. On equal n, calmar improves from 0.062 to 0.072 (rows 1–4) and 0.079 (rows 2–4), and burn falls from 60 to 56 for rows 2–4.
- **The pre-run EDA's prior was too strong.** It expected the TabPFN mean to add partial IC over the ridge. The causal mean adds no payout and costs CORR at t −1.2.

**Verdict:** drop rows 5–9; they only cost. Rows 2–4 are the one candidate left, and they are a risk play rather than a level play: calmar +0.017 on 510 periods at flat payout. They are worth a calmar-paired check before any deployment talk. That would be a separate study, since none of this is live-wired.

**Artefacts:** `results/reports/crypto/e113_fmrep_ablation.html` (the 8 rows against the book) and `logs/e113_report.txt` (paired deltas against both the no-block and the whole-block ridge).


## E62 — factor-augmented trees do not beat their plain twins; the one apparent edge is search budget

**Status:** 55 of 108 arms finished, giving 25 complete SUR-vs-stacked pairs on raw, tenc, interact, fenccausal and cumlag3. Nothing is running.
- **Wide datasets:** the 48 arms on tenc_onehot, onehot_rollsel, allfeat and allfeatpp never produced a result. deepsix rebooted uncleanly at 12:11 on 2026-09-26, during arm 61 (sur_lgb_reg_tenc_onehot, refit 12 of 24). run_forecast writes only at the end, so there is nothing partial to resume.
- **sur_catboost classifiers:** all 5 crashed at refit 0. sklearn `clone` rejects CatBoost when `l2_leaf_reg` is an `np.float64`. The plain CatBoostWrapper grid uses `np.logspace` for that parameter; the ktz grid casts to float, which is why the regression arms run. The 4 wide-dataset classifier arms would hit the same error.

**Method:** each pair was compared with paired Newey-West at lag 20 on its own 510 periods (519 on raw). All 50 arms are scored on target_binned_return_20. The repo's `paired_delta.py` reproduces every delta to 1e-16.

| family | task | CORR Δ, t by dataset (raw / tenc / interact / fenccausal / cumlag3) | payout Δ t, same order |
|---|---|---|---|
| lgb | reg | +0.53 / −1.98 / −0.12 / −0.79 / −1.92 | −0.13 / −1.62 / −0.16 / −1.11 / **−2.21** |
| lgb | clf | −0.17 / +0.52 / +0.55 / +0.87 / +0.88 | −1.18 / −0.33 / −0.00 / −0.20 / +0.46 |
| rf | reg | +0.09 / +0.19 / −1.19 / −0.14 / +1.51 | −0.52 / +0.22 / −0.91 / −1.60 / +0.43 |
| rf | clf | −0.47 / −0.76 / −1.60 / −0.17 / −0.03 | −0.68 / −0.24 / −1.23 / +0.60 / +0.00 |
| catboost | reg | +1.71 / +1.19 / **+2.41** / +1.68 / +1.90 | +1.22 / +0.95 / +1.77 / +0.21 / +1.26 |

- **No pair clears the multiplicity bar.** Bonferroni over 25 pairs puts it at |t| ≥ 3.1. The largest of all 75 tests is +2.41, and the worst is lgb reg cumlag3 at payout t −2.21 and mmc t −2.23. lgb and rf lose payout in 15 of 20 pairs, and every positive among them has |t| ≤ 0.60.
- **The SUR arms are near-duplicates of their twins.** Within-date prediction correlation is 0.93–0.96 for regression and 0.80–0.95 for classification. CV keeps k > 0 in about 90% of refits, which is the E61 pattern: the factor gets picked in CV and dies out-of-sample.
- **The CatBoost "edge" is search budget, not factors.**
  - The plain catboost_ktz grid has 12 cells, so its 50-draw search clamps to 12 candidates. The SUR arm searches 50 of the 60 cells that the 12 base cells × k {0,1,2,3,5} make up. CatBoost is the only family where SUR gets the bigger menu.
  - With the base cell held fixed (58 refits where SUR kept k > 0 and the same base cell as the control), adding factors gives CORR t +0.87 and payout t −0.11.
  - The gain lives in the 50 refits where the bigger menu also moved the base configuration (mostly to rsm 0.2 and depth 4).
  - The 5 datasets amount to about 2 independent looks, so a 5/5 sign pattern has a null probability near 0.18, not 1/32. The pooled CORR t is about 1.9 and the pooled payout t 1.09.
  - The lower dd(pay) on 4 of 5 datasets is descriptive only.
- **Against the book**, the best arm (sur_catboost reg tenc, payout 0.072 on 449 periods) is far below the book's slot blends (0.106–0.117).

**Verdict:** modelling the cross-symbol error factor does nothing for trees, as it did nothing for the ridge (E61). Do not resume E62 to finish the wide datasets. Doing so would need the CatBoost clone fix, and nothing in the 25 pairs suggests the wide sets would differ.

**Artefacts:** `results/reports/crypto/e62_sur_trees_reg.html` and `e62_sur_trees_clf.html` (boards against the book, on a common 449-period window).

## E115 — prequential and time-series CV pick very different λ and score the same; selecting on payout does not raise payout

Ridge on `crypto/v2.0-fenccausal` (44f), rolling 130, refit every 22 periods (one month), embargo 20, 510 scored periods, one fixed grid of 51 λ (1e-3..1e7; the X'X spectrum is 0.024..2.0e4). Every λ is fitted at every refit, and the six arms differ only in which λ each of the 24 blocks used. The two payout arms were added 2026-10-23; rerunning reproduced the four earlier arms byte for byte. The prequential rule had 992 resolved out-of-sample periods at the first scored block and 1,498 at the last. Paired comparisons are Newey-West lag 20 on the 510 common periods.

| arm | CORR | MMC | payout | c.sharpe | dd(pay) | calmar | burn | median λ | distinct λ / switches |
|---|---|---|---|---|---|---|---|---|---|
| prequential, −CORR | 0.13566 | 0.05175 | 0.06532 | 1.29 | 1.74 | 0.038 | 93 | 2,512 | 2 / 1 |
| prequential, MSE | 0.13670 | 0.05260 | 0.06627 | 1.31 | 1.74 | 0.038 | 91 | 2,512 | 1 / 0 |
| TSCV, −CORR | 0.13555 | 0.05482 | 0.06837 | 1.47 | 1.52 | 0.045 | 87 | 11,080 | 14 / 20 |
| TSCV, MSE | 0.13675 | 0.05224 | 0.06591 | 1.40 | 1.71 | 0.038 | 87 | 3,246 | 10 / 19 |
| prequential, −payout | 0.12974 | 0.05338 | 0.06636 | 1.47 | **1.43** | **0.046** | 87 | 25,120 | 8 / 17 |
| TSCV, −payout | 0.12945 | 0.05262 | 0.06556 | 1.38 | 1.52 | 0.043 | 103 | 579,200 | 14 / 18 |
| `e109_ridge_fenc` (context, refit 130) | 0.15175 | 0.07105 | 0.08622 | 1.60 | 1.38 | 0.062 | 60 | 38 | — |

| paired | ΔCORR (t) | ΔMMC (t) | Δpayout (t) |
|---|---|---|---|
| prequential − TSCV, −CORR loss | +0.00011 (+0.02) | −0.00307 (−0.66) | −0.00306 (−0.59) |
| prequential − TSCV, MSE loss | −0.00005 (−0.02) | +0.00036 (+0.14) | +0.00035 (+0.12) |
| −CORR − MSE, prequential | −0.00104 (−1.78) | −0.00085 (−1.70) | −0.00095 (−1.71) |
| −CORR − MSE, TSCV | −0.00120 (−0.37) | +0.00258 (+0.97) | +0.00246 (+0.83) |
| prequential − TSCV, −payout loss | +0.00030 (+0.09) | +0.00076 (+0.28) | +0.00079 (+0.26) |
| −payout − (−CORR), prequential | −0.00592 (−1.07) | +0.00163 (+0.36) | +0.00104 (+0.21) |
| −payout − MSE, prequential | −0.00696 (−1.33) | +0.00078 (+0.18) | +0.00009 (+0.02) |
| −payout − (−CORR), TSCV | **−0.00610 (−2.58)** | −0.00220 (−1.94) | **−0.00281 (−2.09)** |
| −payout − MSE, TSCV | −0.00730 (−1.72) | +0.00038 (+0.14) | −0.00035 (−0.11) |

- **The rule does not matter.** Prequential − TSCV is null under all three losses. The 95% payout intervals are [−0.0132, +0.0071] with the −CORR loss, [−0.0054, +0.0062] with MSE and [−0.0051, +0.0067] with −payout.
- **λ does matter, and both rules find it.**
  - The ex-post fixed-λ CORR curve runs from 0.1046 at least squares up to a peak of 0.1373 at λ ≈ 3,981, then down to 0.1234 at the X'y limit.
  - Every arm sits within 0.0017 CORR of that peak, so hindsight would barely beat any of them. Selection recovers about 95% of the 0.033 gap between least squares and the peak.
  - This is not a null by construction: the curve has room, and both rules use it.
- **They get there differently.**
  - Prequential locks onto λ ≈ 2,512 after the warm-up. The MSE arm never moves; the −CORR arm drops to 1,585 for 3 blocks. A month of new evidence barely shifts ~1,000 periods of history.
  - TSCV re-decides from 4 folds of 26 periods every month: 14 distinct λ, 20 switches in 24 blocks, an IQR of 2.7 decades.
  - TSCV/−CORR made 6 grid-edge picks: 5 at 1e7, where CORR has already flattened, so that edge does not bind, and 1 at 1e-3.
  - The two rules agree on 4% of blocks with −CORR and 12% with MSE.
- **TSCV/−CORR has the best risk profile, but not significantly.** Calmar is 0.045 against 0.038, dd(pay) 1.52 against 1.74, and MMC +0.003. Its payout edge over prequential is t +0.59, so this is descriptive only.
- **−CORR against MSE makes no difference.** It is |t| < 1.8 under both rules. Under prequential both losses choose the same λ in 21 of 24 blocks.
- **Choosing on payout does not raise payout.**
  - Under prequential it is null: +0.0010 (t +0.21, 95% CI [−0.0088, +0.0109]) against the −CORR arm. It buys risk rather than level: dd(pay) 1.43, the lowest of the six arms, calmar 0.046 and c.sharpe 1.47.
  - Under TSCV it HURTS: −0.0028 payout (t −2.09, CI [−0.0055, −0.0002]) and −0.0061 CORR (t −2.58) against TSCV/−CORR. It also has the most losing days (burn 103 against 87).
  - Of the 15 comparisons this is the one clearing |t| ≥ 2, so read it as a warning rather than an established effect.
- **Why payout selection is weak here: it shrinks harder on far less evidence.**
  - The ex-post fixed-λ payout curve peaks at λ ≈ 1.6e4 (0.0694). That is 4× heavier than CORR's optimum, because shrinkage trades CORR for MMC. The curve runs from 0.0468 at least squares to 0.0674 at the limit, so it is flatter on the heavy side.
  - The payout arms pick heavier still. Prequential's median is 2.5e4 with 17 switches; TSCV's is 5.8e5 with an IQR of 3.95 decades, 8 picks at the 1e7 edge (where payout has flattened) and 1 at 1e-3.
  - Both run on thin evidence, because the meta model starts 2024-05-07. At the first block prequential had 17 periods with a meta model against 992 for the CORR arms. TSCV had 17, 39, 61, 83 and then 104 inner periods over the first five blocks. No block needed the CORR fallback.
  - Every arm still lands within 0.003 payout of that peak. The best is TSCV/−CORR at 0.0684, ahead of both payout-chosen arms.
- **Monthly refit trails the 6-monthly house ridge** — context, not a tested result.
  - Every arm trails `e109_ridge_fenc` by 0.015–0.016 CORR (t −1.25 to −1.38) and 0.018–0.021 payout (t −1.5 to −1.9).
  - This is not an E115 artefact. The pipeline's own monthly fenccausal ridge, `ridge_glmnet/e61_glm_reg_fenccausal`, scores CORR 0.13644, payout 0.06830 and calmar 0.044 — the level of TSCV/−CORR. The 6-monthly glmnet twin (`w6r6_glmnet_reg_fenccausal`) scores 0.14357.
  - Cadence and grid both differ from the context run, so this gap is an observation, not a comparison.

**Verdict:** choosing ridge's λ prequentially instead of by in-window TSCV changes nothing in level on fenccausal at monthly cadence, under any of the three losses. Both rules land within 0.002 CORR of the best fixed λ in hindsight. Using payout itself as the selection utility does not help: it is null under prequential (with the lowest drawdown of the six arms) and costs payout under TSCV (t −2.09). The meta model's short history makes payout a noisy utility here. Selecting on CORR gets as close to the payout optimum as selecting on payout does. Prequential's advantages are stability (one λ for two years against 20 switches) and cost (one fit per λ per refit, no inner folds). If a ridge λ is ever re-selected live, prequential gives the same score with a λ that does not jump. On this setup the bigger lever is refit cadence, not λ selection.

**Artefacts:** `logs/e115_report.txt` (also emailed); `results/reports/crypto/e115/lambda_path.csv` (each block's λ per arm) and `fixed_lambda_curve.csv`; runs `ridge/e115_{preq,tscv}_{corr,mse,payout}`; scouts `ridge/e115scout_*` (era_step 4, not reportable).

## E117 / E124 — six months of history fit on the 20-day label is the best ridge; a 60-day label only rescues windows too short to use

**Setup.** ridge_glmnet on allfeat_pp minus `tenc` (811 features, the payout-best
single ridge), rolling training windows of 5 / 10 / 15 / 20 / 22 / 65 / 130 periods,
refit every 130, λ searched from scratch at every refit. Windows under 130 cannot
hold a purged fold, so they were tuned with the new `--tune-window 130`: the search
runs on the 130 periods ending where the window ends, with folds that each train on
the window's length, and the winner is refitted on the window alone. E117 fits the
20-day label; E124 fits and tunes the 60-day one (embargo 60, two inner folds).
**Every arm is scored on the payable 20-day label** through
`cross_target.score_on_payable_target`, which reproduces E117's native numbers to
4e-9 (CORR) and 1e-14 (MMC). 470 common dates, 2024-07-25 to 2026-05-13.

| window | payout fit-20 | payout fit-60 | fit60−fit20 (t, lag 20) | dd fit-20 | dd fit-60 |
|---|---|---|---|---|---|
| 1 week | 0.01933 | 0.05268 | +0.03335 (+1.14) | 8.96 | 1.88 |
| 2 weeks | 0.02017 | 0.04929 | +0.02911 (+1.01) | 8.72 | 2.38 |
| 3 weeks | 0.03943 | 0.04764 | +0.00821 (+0.35) | 7.87 | 4.68 |
| 4 weeks | 0.04352 | 0.03894 | −0.00458 (−0.22) | 7.85 | 5.59 |
| 1 month | 0.04600 | 0.03774 | −0.00826 (−0.40) | 7.85 | 5.87 |
| 3 months | 0.05692 | 0.04031 | −0.01661 (−0.63) | 4.13 | 4.94 |
| **6 months** | **0.08955** | 0.07784 | −0.01171 (−0.49) | **1.73** | 5.17 |

**History dominates everything else.** Payout rises monotonically with the window on
the 20-day label, and every window under six months is significantly worse than six
months (t −2.18 to −2.82 here; −2.46 to −3.21 on E117's own 510 dates).

**The 60-day label crosses over at about three weeks.** Below it, the slower label is
a better target than the one paid on: it is a smoother function of the same features,
so a fit on a week of data is less noise — drawdown falls from 8.96 to 1.88 at one
week. Above it, fitting the paid label wins. Nothing at |t| > 2, and nothing beats
six months fit on 20 (0.08955, dd 1.73, calmar 0.052); six months fit on 60 pays
0.07784 with three times the drawdown.

**The λ search runs out of path on short windows.** On the 20-day label it lands on
the top of the glmnet path on most refits under three months (`85.6^ 47.8^` at one
week): given little data the best it finds is to shrink nearly everything away. On
the 60-day label the ceiling binds at refit 2 for almost every window, which reads
as a regime where the slow label was unpredictable rather than as a window effect.

**Meta-residualised (added 2026-10-24).** Every arm re-scored after the live
transform -- `src.prediction.meta_resid`, strength 0.5, meta model at least 5 days
old -- applied to the finished forecasts, nothing refitted. The application
reproduces E107's stored residualised CURUPIRA_04 to 5e-16. Residualising raises
payout in 12 of 14 arms, but clears t > 2 only at six months: fit-20 +0.00770
(t +3.50, 0.08955 → **0.09725, the best row in the table**) and fit-60 +0.00431
(t +2.21). On short windows it does almost nothing (+0.0002 at one week), which is
E107's rule that the gain scales with a model's overlap with the meta model: a
model fitted on a week of data says little the meta model says. It trades CORR for
MMC (six months fit-20: CORR 0.1576 → 0.1409, MMC 0.0738 → 0.0832) and lowers
calmar in 10 of 14 (0.052 → 0.045, dd 1.73 → 2.15), as E107 found. The ranking of
windows and of labels is unchanged by it.

**Read with E116.** E116 held the window at 130 and varied the CADENCE; refitting
less often was better, monotonically to six months. E117/E124 hold the cadence and
vary the WINDOW; more history is better, monotonically to six months. The deployed
130 / 130 configuration is the best point on both axes measured.

## E116 — refitting LESS often is better: payout rises monotonically from every period to every 6 months, for every λ rule

Ridge on `crypto/v2.0-fenccausal` (44f), rolling 130-period window, E115's 51-λ grid, embargo 21 (training ends 22 periods before the predicted date, the live train/live gap), the same 510 scored dates as E115. Only the refit cadence moves. Paired comparisons are Newey-West lag 20 on 510 common dates.

| payout (calmar) | r1, every period | r22, 1 month | r65, 3 months | r130, 6 months |
|---|---|---|---|---|
| tscv_corr | 0.0577 (0.036) | 0.0672 (0.045) | 0.0821 (0.064) | 0.0848 (0.066) |
| preq_payout | 0.0623 (0.044) | 0.0664 (0.047) | 0.0847 (**0.082**) | 0.0893 (0.073) |
| tscv_mse | 0.0596 (0.034) | 0.0665 (0.039) | 0.0871 (0.069) | **0.0900** (0.071) |
| ex-post best fixed λ (oracle) | 0.0673 | 0.0696 | 0.0879 | 0.0928 |

| paired vs r130, Δpayout (t) | r1 | r22 | r65 |
|---|---|---|---|
| tscv_corr | **−0.0271 (−2.85)** | −0.0176 (−1.92) | −0.0027 (−0.39) |
| preq_payout | **−0.0270 (−2.92)** | **−0.0229 (−2.69)** | −0.0046 (−0.59) |
| tscv_mse | **−0.0304 (−3.13)** | **−0.0235 (−2.96)** | −0.0029 (−0.51) |

- **Cadence is the lever, and slower is better.**
  - Payout rises monotonically r1 < r22 < r65 < r130 for all three rules, and calmar and burn improve with it (burn 105–109 at r1 against 34–47 at r130).
  - A refit every period, the fastest a live round allows, costs 0.027–0.030 payout against the live 6-monthly cadence (t −2.85 to −3.13). The monthly refit costs 0.018–0.023 (t −1.92 to −2.96). Quarterly is indistinguishable from 6-monthly (|t| < 0.6).
  - This reproduces the early raw-ridge cadence sweep's left side (monthly 0.127 < quarterly 0.145 < six-monthly 0.153), now with λ re-chosen at every refit.
- **It is not about λ.**
  - The ex-post best FIXED λ shows the same ordering (payout 0.067 → 0.070 → 0.088 → 0.093). So it is how recent the training window is, not noisy λ re-selection, that hurts.
  - At r1, tscv_corr's λ switches 245 times in 510 blocks (79 grid-edge picks), yet the stable prequential rule (34 switches) is just as bad.
- **The λ rule still does not matter at any cadence.** Every rule-vs-tscv_corr comparison is |t| ≤ 1.76, extending E115's null to all four cadences.
- **The live-lag embargo did not matter.** r22 at embargo 21 against E115's r22 at embargo 20 is |t| ≤ 1.24 for all three rules.
- **E115's gap to the house ridge was all cadence.** At r130 the arms match `e109_ridge_fenc` (Δpayout −0.0015 to +0.0038, |t| < 0.7), even with λ ~2.5e3–2e4 against e109's ≤ 100.
- **Best arms:** tscv_mse_r130 (payout 0.0900, calmar 0.071, burn 44) and preq_payout_r130 (0.0893, 0.073, 47). preq_payout_r65 has the best calmar (0.082, dd(pay) 1.03). None of them differs from e109 at |t| ≥ 2.
- **An open question, untested here:** why should a staler model predict better? At r130 the training window is on average ~86 periods old when it predicts, at r1 always 21. This study cannot separate "older training data predicts better" (e.g. recent factor returns reversing at ~1-month lag) from phase luck — r130 is only 4 fits whose dates happen to fall where they do. A refit-every-period run with the window lagged by extra periods, or r130 at shifted start dates, would separate the two.

**Verdict:** keep the live 6-monthly cadence (`registry.FIT` refit_every 130). Refitting every round, the fastest live allows, is significantly worse for every λ rule tested, and so is monthly. Quarterly ties 6-monthly. The λ rule is not a lever at any cadence.

**Artefacts:** `logs/e116_report.txt` (also emailed), `logs/e116_r{1,22,65,130}.log`; `results/reports/crypto/e116/{lambda_path,fixed_lambda_curve}_r<N>.csv`, `fit_r<N>.json`, `table.csv`; runs `ridge/e116_{tscv_corr,preq_payout,tscv_mse}_r{1,22,65,130}`; scouts `ridge/e116scout_*` (era_step 4, not reportable). Walk-forward time: r1 80 min, the others 3.5–5.4 min.

## E129 — the 6-month RF's leaf values do not help CU_SA_IA_BO_MA_R_01 or CU_SA_IA_BO_02

**Setup.** `src/data_mani/build_rfleaf.py` re-ran E128's w130 forest through the pipeline's own `forecast()`, with the validation start moved back 8 × 130 periods. That covers the history from 2021-01-28, which is 84.5% of rows. On E128's blocks 2 and 3 it reproduces E128's stored forecast to 2e-16. Blocks 0 and 1 cannot match: E128's own training windows straddle the 20 embargo dates that `make_validation_split` drops, while here they are contiguous. Every row gets the distribution of its 100 trees' leaf values (mean, std, q10/25/50/75/90) plus the forest's prediction residualised on the 5-day-lagged meta model, all binned. That is 8 `feature_rfleaf_*` columns, from a forest that never trained on the row.

The six single members were re-run on their recorded protocol, as control and as +rfleaf. Five controls reproduce the stored originals exactly once SACI_04 runs on the frozen `crypto/v2.0-asof`, because the live round keeps growing `data/crypto/v2.0`. IA_01's 50-draw MLP search does not reproduce: 0.0915 against 0.0987, per-period corr 0.84. So the IA_01 pair is matched only to itself, and its search noise is of the same order as the effect.

`src.analysis.e129_report` reassembles both slots from the stored configs. Built from the stored runs, the assembler reproduces both slot series to 2e-5.

| member / slot | control | +rfleaf | Δ payout (t) | Δ CORR (t) |
|---|---|---|---|---|
| IA_01 | 0.0915 | 0.0724 | −0.0191 (**−2.44**) | −0.0277 (−3.26) |
| IA_02 | 0.0949 | 0.0966 | +0.0017 (+0.56) | +0.0015 (+0.49) |
| SACI_04 | 0.0833 | 0.0857 | +0.0024 (+0.76) | −0.0004 (−0.14) |
| MA_FENC | 0.0934 | 0.0937 | +0.0003 (+0.13) | +0.0036 (+1.39) |
| MA_ALLFEAT | 0.0955 | 0.0929 | −0.0026 (−0.79) | −0.0006 (−0.21) |
| CURUPIRA_04 | 0.0872 | 0.0903 | +0.0031 (+1.08) | +0.0045 (+1.55) |
| CURUPIRA_05 (+ leaf block) | 0.0863 | 0.0910 | +0.0047 (**+2.24**) | +0.0133 (+5.61) |
| CURUPIRA_05_60 (+ leaf block, 470) | 0.0749 | 0.0850 | +0.0101 (**+3.36**) | +0.0214 (+5.29) |
| **CU_SA_IA_BO_MA_R_01** (470) | 0.1158 | 0.1087 | −0.0071 (**−2.17**) | −0.0081 (−2.59) |
| **CU_SA_IA_BO_02** (470) | 0.1050 | 0.1025 | −0.0025 (−1.04) | −0.0015 (−0.65) |

- **As inputs, the leaves do nothing for the five tree, ridge and pinned-MLP members** (|t| ≤ 1.1), and they hurt the searched MLP IA_01.
- **The two ridge block-blends gain, but that is an ensemble effect, not a feature effect.** A sixth ridge on the leaf block is essentially the 6-month RF's prediction added to a 5-ridge blend.
- **It does not survive into the slots.** Post-hoc variants (`results/reports/crypto/e129_slot_variants.csv`) cost R_01 −0.0018 (t −1.34) for everything except IA_01 and −0.0007 (t −0.79) for blocks only. BO_02 loses −0.0005 (t −0.27) and −0.0006 (t −0.48). Drawdown rises in every variant: R_01's dd(pay) goes from 1.41 to 1.50–1.80.

**Verdict:** do not deploy. The slots already hold the RF and ridge information the leaves carry: MAPINGUARI_01's RF legs and five ridge blocks sit inside both. Adding the 6-month RF's own leaves is redundant at the slot level and harmful through IA_01.

**Artefacts:** `results/reports/crypto/e129_rfleaf.html` (board), `e129_members.csv`, `e129_slots.csv`, `e129_slot_variants.csv`; datasets `crypto/v2.0-asof[-rfleaf]`, `v2.0-fenccausal-rfleaf`, `v2.0-tenc-onehot-spread-interact-rfleaf`, `v2.0-tenc-onehot-rollsel-spread-interact-rfleaf`; leaf panel `data/crypto/rfleaf_cache/`.


### E150 observed-only development rerun (2026-11-12)

User requested comparison of the created datasets. Correct the daily exposure alignment into separate `-retrospective` datasets; preserve legacy files. New runner outputs `results/reports/crypto/e150_macrovol_observed_dev/`. Every candidate uses complete observed feature rows with matched v2.0 and vol-causal comparators; high/low gates are separate arms, never neutral-imputed. Run ridge / LightGBM / MLP, fixed five stochastic seeds plus equal six-candidate inner searches, source/LOO/exposure/gate arms and scalar/circular/look-ahead controls. Only development blocks are fitted/scored: the old final block is already contaminated. Future placebo excluded from selection; parsimonious rule requires improvement across models. Macro source vintages remain unverified and gold remains the legacy GLD proxy, so this is exploratory and cannot authorize adoption. Jobs: `e150_observed.jobs`; preparer: `e150_observed_prepare.py`; automated selection/scoring/PDF: `e150_observed_finish.py`. Frozen inputs and definitions live in the output registration.

### E150 observed-only rerun findings (2026-11-13)

Completed all 12 training batches and scoring: 1,038 prediction jobs, 1,349,400 date/arm/seed score rows, 1,300 development dates; historical-meta MMC proxy on 468 dates. **REJECT the combined macro block on development for all three models.** With equal six-candidate searches, comparing ALL with its own matched covered-sample vol-causal comparator after slot routing:

| Model | Mean CORR delta | Newey-West t | Mean MMC-proxy delta | MMC t |
|---|---:|---:|---:|---:|
| ridge | -0.016489 | -4.09 | -0.001730 | -0.46 |
| LightGBM | -0.008810 | -2.46 | +0.001212 | +0.25 |
| identity MLP | -0.031118 | -4.87 | -0.024405 | -3.49 |

The searched candidate covered 66.99% of full-universe rows on average; uncovered coins retain explicit baseline rank predictions. None of the six individual macro families passes the frozen cross-model selection rule, so PARSIM selects no macro source. The combined block's per-period CORR win rates are 35.4%, 41.9%, 29.9%; worst-block mean CORR deltas are -0.0582, -0.0745, -0.1179. This is development-only, retrospective evidence: GLD is still an ETF proxy, macro publication vintages are unverified and the inherited crypto universe is reconstructed. The previously scored final block was excluded. No deployment or stake change is warranted.

The initial single-worker scorer was superseded by `e150_finish_scoring.py`, using batched matrix calculations and sharing work for identical predictions. Numerical parity against numerai_tools was checked for all distinct columns on 33 real model/seed/date batches (max error 3.33e-16); float32 targets, ties, constant predictions and missing meta-model values have regression checks. Block-bootstrap draws/truncation are unchanged and all archived selection t-statistics reproduce. Optimized scoring and report generation completed in 79 seconds. The original runner, models, inputs and selection were preserved.

PDF: `results/reports/crypto/e150_macrovol_observed_dev/E150_does_macrovol_help_REJECT_DEVELOPMENT.pdf` (37 pages, layout checked). CSVs: `headline.csv`, `paired_deltas.csv`, `levels.csv`, `regime_deltas.csv`, `seed_deltas.csv`, `coverage_by_year.csv`, `by_block.csv`; raw scores `scores_by_date_seed.parquet`. Completion and validation: `COMPLETE.json`, `scoring_batches/*.validation.json`.

### E156: simulate adding the shipped raw-volatility features (registered 2026-11-13)

User request: `v2.0-exrawvolfeat` versus `v2.0`, as though the direct volatility columns were initially unavailable. Freeze the current official v2.0 input; create a 20-feature dataset by dropping exactly `feature_volatility_20d` and `feature_volatility_60d`, preserving all other columns, rows, index and missing values. Predeclared arms: EXRAWVOL (20), ADD20 (21), ADD60 (21), FULL (22). Primary paired direction is FULL minus EXRAWVOL; positive means adding the shipped volatility columns helps. Sharpe ratios and Bollinger bands remain, so this is their conditional feature value.

Ridge, LightGBM and identity MLP: identical full-universe rows, target 20, eleven purged 130/22/130 development folds. Standard v2.0 input neutral fill on all arms, no macro/spot/feed gates or routed subset. Fixed configs plus identical six-candidate searches, each repeated over the same five stochastic seeds (ridge: seed 0). Inner validation is 40 dates after a 22-date purge. Paired Newey-West lag 20, moving-block bootstrap length 20 / 2,000 draws; proxy MMC and payout on the same meta-covered dates, seed stability and downside. Final previously evaluated dates on/after 2026-02-20 excluded. No untouched-test or deployment claim. Dataset and runner hashes frozen before fitting. Runner: `experiments/e156_rawvol_addback.py`; outputs: `results/reports/crypto/e156_rawvol_addback/`.

### E156 findings: shipped volatility add-back (completed 2026-11-13)

**PROMISING on development across all three models:** adding the original two official volatility features to the other 20 v2.0 features materially improves CORR. This is a counterfactual column-removal/add-back test, with all panel coins and no macro or external feed coverage gate. Same labels, dates, inner searches and seeds for both sides.

| Model | Searched FULL − EXRAWVOL mean CORR | Newey-West t | Mean proxy MMC difference | MMC t | CORR win rate |
|---|---:|---:|---:|---:|---:|
| Ridge | +0.047677 | +5.07 | +0.028246 | +1.68 | 73.6% |
| LightGBM | +0.028167 | +5.02 | +0.010983 | +1.57 | 68.6% |
| Identity MLP | +0.054868 | +6.52 | +0.026962 | +1.91 | 75.5% |

1,430 development CORR dates / 468 historical-meta MMC-proxy dates; five matched stochastic seeds on both fixed and searched comparisons (ridge seed 0). CORR gains are positive in all 11 development blocks for each model and in every stochastic seed's full-window mean. Worst-block mean CORR improvements: +0.02217 ridge, +0.00225 LightGBM, +0.00349 MLP. Adding each horizon alone also helps: searched ADD20 CORR deltas +0.03420 / +0.02304 / +0.03872; ADD60 +0.03747 / +0.01980 / +0.04613. Both together are strongest in every model. Proxy MMC gains are positive but none meets |t| >= 2; payout t is +1.80 / +1.78 / +2.06, so the broad evidence is stronger for CORR than contribution/payout.

Dataset `crypto/v2.0-exrawvolfeat` has exactly 20 features, 524,596 rows, and all retained values/index/missing cells copied exactly from a frozen v2.0 snapshot. Raw columns removed: `feature_volatility_20d` and `feature_volatility_60d`. Sharpe and Bollinger proxies remain: this is conditional column value, not removal of all volatility information. Full v2.0 fixed predictions reproduce E133 benchmarks: maximum difference 1.37e-6 ridge, 0 LightGBM, 1.11e-16 MLP. New scoring passed numerai_tools parity checks on sampled real dates, including float32 targets and missing meta values. All 12 training batches completed; 125,840 date/model/arm/seed scores were produced. Unit tests for feature selection and paired direction passed. The local snap Chromium export failed; the same completed analysis was successfully exported on lovelace with no model or score changes (`e156_export_report.py`).

Previously scored final dates on/after 2026-02-20 were excluded, so no untouched-test or new deployment claim is made. PDF (23 pages, layout checked): `results/reports/crypto/e156_rawvol_addback/E156_does_adding_raw_volatility_help_PROMISING_DEVELOPMENT.pdf`. Companion artifacts: `registration.json`, `baseline_reproduction.json`, `scoring_parity.json`, `headline.csv`, `paired_deltas.csv`, `paired_by_seed.csv`, `levels.csv`, `by_block.csv`, `scores_by_date_seed.parquet`, `COMPLETE.json`.

## E173 — Matched spot-volume feature ablation (registered 2026-11-14)

User-defined fixed A0–A5 adds: relative1dayUSD return; log latest base volume/prior20median and log latest3mean/preceding14mean; raw surprise×relative return; venue surge breadth/count;20day medianUSD spotvolume×return. Percentile ranks and training-only1/99% clipped continuous surprise, CS surprise deviation and interaction. Matched venue-pairs throughout each calendar comparison; no fills in saved features, neutral fill only at estimator input. Stored E152 inferred availability exactD−2, source price/identity checks retained; unverified vintages/survivorship disclosed. Nine venues vs Binance/Coinbase/OKX/KuCoin (user confirmed) on common A5 coverage. Separate Yahoo-only20dayUSD-volume history, A1+liquidity vsA1.

Frozen E133/E135 raw22 baseline/target,12 chronological130/22/130folds, fixed ridge/LGBKTZ/identityMLP parameters, stochasticseeds0–4. Exact468000validationrows allarms; original baseline predictions reused after exact feature/target check. NumeraiCORR validation score, dailySpearman, proxyMMC/payout, pairedHAC20 andblock20bootstrap2000 vsA0 and incremental; byblock/seed, featurecoverage, common9vs4. No retuning peruser fixed-settings specification. Exploratory reused testblock11, no live/stake writes or emails requested. Manifest/configs/predictions saved. Real-source prefix audits before fit.

Runner `experiments/e173_spot_volume_comparison.py`, features `src/data_mani/spot_volume_comparison.py`, results `results/reports/crypto/e173_spot_volume/`. Findings pending.

E173 COMPLETE 2026-11-14: all three models scored; PDF `results/reports/crypto/e173_spot_volume/E173_spot_volume_comparison_EXPLORATORY.pdf` (61 pages). Nine-venue A5 vs A0 NumeraiCORR dev/test: LightGBM +0.006675(t2.86)/+0.002569(t0.95), MLP -0.009171(t-3.97)/-0.010031(t-4.39), ridge -0.004118(t-1.80)/+0.002007(t0.67). Model-specific exploratory lead, not universal volume gain. LightGBM strongest incremental group A2 vs A1 +0.001353(t2.18) dev, +0.00073 test, positive8/11devblocks and4/5seeds. MLP A3 vs A2 recovers+0.00500dev but full model remains belowbaseline. Outputs include common9vs4, separateYahoo, coverage, paired block/seed deltas, configs/predictions. Historical-prefix audit87414rows zerochanges; batch/scalar evaluator identity checked. PDFfirstpage visuallychecked.


## E181 — Sklearn-initialised Torch KTZ hidden states as ridge features (registered 2026-11-15)

User requests hidden-size and processing sweeps, one E164-style synthesis PDF, causal fresh ridge fits, unpenalised one-hot coefficients, lifecycle notifications. Source encoder uses fenccausal44; readout uses allfeat_pp minus target encoding plus x29 (813 original features). Architectures: (8), (16), (32), (64), (128), (32,16), (32,16,8). Forty KTZ learning-rate/alpha cells per architecture, identity activation, actual sklearn NumPy initialisation draws in Torch. Inner68/purge22/validate40 selects encoder and 100-point glmnet path index; path rebuilt at fresh outer130/purge22/predict130 fits. Seeds42–44. Added states: raw, standardised, contemporaneous CS five bins, training-fitted quantile five bins, standard states+squares, hidden-only+asset effects, MLP+ridge residual correction. Existing 752-ish asset dummies are also unpenalised by exact within-asset demeaning; unseen assets receive pooled intercept. Fresh FE baseline differs intentionally from historical penalised-dummy E118. Dimensions/processing are exploratory; dates>=2026-02-20 excluded at read. Frozen upstream causal x29/fenccausal data reused with hashes; no historical-vintage proof. Report: common meta-covered CORR/MMC/payout/DD/calmar/burn, per-seed/block, cumulative payout by architecture, paired HAC20, full registration and hidden-rank/head reconstruction audits. Start/failure/completion notifications use project SMTP; completion attaches single synthesis PDF. Runner experiments/e181_mlp_hidden_ridge.py; output results/reports/crypto/e181_mlp_hidden_ridge/.

E174–E180 launch/audit update2026-11-15: all seven feature builds and real-source prefix audits PASSED. E174260767rows/285tasks; E175256046rows/285tasks; E176258297rows/241tasks; E177/E178/E179261145rows/197tasks each; E180260767rows/219tasks. Initial exact-float own-history ties split semivariance windows by<2e-18 raw variance; now use explicit relative1e-12+absolute1e-18 numerical ties, regression test added (17checks pass), rebuild all affected data. Preserved rawCS20EWMA fit only after exact input-identity check. All source/code revisions recorded. Parallel dependency queue `experiments/volatility_proxy_supervisor.py`, one-core watcher adds result-derived findings to the same perproxyPDF; no mail. Running code left immutable; final visual review remains pending until PDFs exist.

E174–E180 COMPLETE2026-11-15: all1,621registered tasks and seven synthesis PDFs completed, no model task failures. All28canonical20/60payout delta95%block-bootstrap intervals versus matched simple-return variance include zero; no improvement established by these primary comparisons. Full11window implementation/calendar ridge sweeps and canonical six-representation ridge/identityMLP sweeps retained, no returnholdout opened. Each task completed8supported outer folds; first3gridfolds skipped for insufficient source/warm-up coverage, never shortened purge. Every headline metric shares468dates and complete seed coverage within its proxy. SevenPDFs visuallyspotchecked and text/definition/metric identities verified; final_manifest.json records hashes, page counts, cohort sizes and CSV hashes. Reports include explicit formulas, outcome-derived findings, grids, controls, curves, HAC20/bootstrap2000, seed/block/coverage diagnostics and complete tables. Repeated raw candidate/asset-ID lists remain adjacent JSON/CSV artifacts. One transient SSH255 in final Yang–Zhang rendering retried successfully; no fits rerun. Experiment index now derives these seven arm counts and final metrics from checkpoint artifacts, not hand-maintained status.


### E182: volatility replacement, ~1M-parameter MLP at 1x and 10x compute (registered 2026-11-17)

User request: run the new overparameterised (~1M parameter) model and the overcomputation (10x epoch schedule) model on v2.0 and on v2.0 with the replacement volatility features; add both to the report tables and the cumulative-payout add-ons, keeping the previous 10x MLP.

Design: `src.analysis.volatility_replacement --mlp-device cuda --mlp-widths 984 984 --mlp-recipe-multiplier {1,10} --reuse-non-mlp results/volatility_replacement_mlp_10x`, outputs `results/volatility_replacement_mlp_1m_1x/` and `_10x/`, queued by `results/volatility_replacement_mlp_1m_1x/queue.sh`, then `volatility_replacement_report --output results/volatility_replacement_mlp_10x --addon <run>` after each. Same 11 development folds, matched coverage and rank-slot routing as the parent study (`experiments/volatility_proxies.md`). Launched 2026-11-16 on `westeros` and registered after launch.

Status 2026-11-17 16:54: the 1x run (PID 52391) was on fold 0 of 11, at epoch 4,200 of 12,626 on the third input set. The 10x run had not started. The queue wrapper had died (`python: command not found`), so the 10x run and the add-on report will need a manual launch once the 1x run finishes.

Stopped 2026-11-22 at the user's request, not finished. `westeros` rebooted on 2026-11-21 at 10:11, which killed the 1x run after folds 0-2 of 11. A resume then failed because `data/crypto/v2.0/train.parquet` had been refreshed to 2026-10-21 (526,696 rows against the frozen 525,196), and the CUDA checkpoint signature pins the old file's sha256. The 10x run never started and no add-on report was rendered. Checkpoints for folds 0-3 remain in `results/volatility_replacement_mlp_1m_1x/checkpoints/`. Resuming would need a frozen panel (rows dated on or before 2026-10-14) passed with `--panel`, plus a content-based input check in place of the byte hash.

### E183: TabNet with epochs tied to model size, TV schedule, raw + fenccausal (registered 2026-11-17)

User request: add 10 CPU cores per machine as a one-off over the cluster limit, and create and run a version of E158 under a new registry entry, with TabNet on a new hyperparameter space whose epoch counts come from the Timmermann-Vulicevic (TV) computation rule for each sampled model size and dataset, without disturbing the TabNet experiments already running.

Design: new model `tabnet_ktz_tv` (`src/prediction/tabnet_tv.py`, `TabNetKtzTVWrapper` in `src/prediction/models.py`). It has no `max_epochs` axis. Each candidate trains E = floor(0.2 * P^0.8) + 10 epochs, where P is the trainable parameter count of the network ITS sampled settings build on the dataset's column count (asserted equal to the fitted network). Space: width 8-16 (n_a = n_d), steps 2-5, GLU 1-2 per kind, every other axis as `tabnet_ktz`; TV is 136-837 epochs on raw and 174-906 on fenccausal (E153/E158 searched 30-150). The model lives in its own module so running `tabnet_ktz` loky workers, which unpickle `TabNetSkRegressor`, see no constructor change. Walk-forward and per-candidate capture are E158's (`--save-cv-results --reseed-per-refit`), n_iter 40, n_splits 3, seeds 42/43, 2 torch threads per search worker (measured 1.1-2.3 s/epoch on 39k rows; scout 16 min). Analysis: `src.analysis.hparam_importance` on `results/tuning/crypto/tabnet_ktz_tv/e183_*_cv.csv`, and E183 vs E158 for TV-length vs short training.

Launch: `python3 -m src.cluster run --jobs experiments/e183_cluster.jobs --extra-cores 10 --exclude lovelace` (new opt-in `--extra-cores`, default 0; lovelace excluded because it already carries E158 raw_s43 outside the manager). Arms `e183_{raw,fenccausal}_s{42,43}`, 18/18/18/12 cores, on the hosts the manager picks (deepfour/deepfive/deepsix/hamilton at plan time). Outputs `results/{metrics,forecast,per_period,hyperparams}/crypto/tabnet_ktz_tv/e183_*`, `results/tuning/crypto/tabnet_ktz_tv/e183_*_cv.csv`, logs `logs/cluster/e183_*.out`. First launched as "E182" 2026-11-17 and stopped after ~15 min when a concurrent session registered E182; relaunched as E183.



### E184: TabPFN + new output layer, LoRA fine-tuned on MMC for the Timmermann-Vulicevic epoch count (registered 2026-11-17)

User request: take TabPFN, add a layer at the end, fine-tune it with LoRA under an MMC objective, and run it for the full number of epochs the Timmermann-Vulicevic (TV) compute approximation recommends, on the GPU.

Design (`src/analysis/tabpfn_lora_head_mmc.py`): LoRA r=8 / alpha 16 on every attention q/k/v/out projection (E102's adapter set, backbone frozen, lr 1e-5), plus a NEW zero-initialised `Linear(192 -> 1)` (lr 1e-4) on each query row's final hidden state -- the vector TabPFN's own `output_projection` decodes -- added as a residual to the bar-distribution mean, so step 0 is exactly zero-shot. This reads the hidden state rather than E83's decoded ~5000-bin distribution. Loss: 1 - BMC against the real meta model (E45's `mmc_loss`), one date's cross-section per step (context = the 130 dates before the T-20 leak wall, subsampled to 4096 rows; E102's episode shape), gradient clip 1.0. Training dates: leak-safe (<= 2024-06-27, E19's first scored date - 20) AND meta-covered -- only 38 dates (2024-05-07..2024-06-27), 11,400 query rows, because the real meta model starts 2024-05-07 (E45's data-starvation caveat). Epochs: E = floor(0.2 * P^0.8) + 10 with P = 295,105 trainable scalars (LoRA 294,912 + layer 193) -> **4,763 epochs, 180,994 steps**, early stopping off. One epoch = one pass over the 38 dates in a fresh order. Smoke on deepfive: 375 s/epoch -> **~21 days** on one RTX 2070, plus ~9 evals. Expect heavy overfitting of 38 dates; that is part of what the run measures.

Eval: per-date walk-forward on E19's 510 scored dates (n_estimators 1, same context subsample at every checkpoint) at epochs 1, 3, 10, 30, 100, 300, 1000, 3000 (descriptive, never selected on) and 4,763 (the pre-specified result). Arms under `results/{forecast,per_period,metrics}/crypto/tabpfn_head/`: `e184_ep<N>` (LoRA + layer), `e184_lora_ep<N>` (same forward, layer dropped), `e184_zs` (adapters disabled = matched n_est=1 zero-shot, written once). Compare paired NW vs `e184_zs`, vs E19 (n_est 4) and vs E45 mmc / E102 / E83.

Launch: launched 2026-11-17 by hand on deepfive as `systemd-run --user --unit=nmr-e184_tabpfn_lora_head_mmc taskset -c 18-21 bash logs/cluster/e184_tabpfn_lora_head_mmc.cmd.sh` (GPU 0) -- `src.cluster` refused placement because E183 already held deepfive's 30% CPU share; the user approved going over it for one GPU job. The job file `experiments/e184_tabpfn_lora_head_mmc.jobs` is the manager form for a relaunch (unit `nmr-e184_tabpfn_lora_head_mmc`, log `logs/cluster/e184_tabpfn_lora_head_mmc.out`). Host deepfive, 1 GPU. Resumable from `models/tabpfn/e184/state.pt` (saved every 5 epochs and at each eval); relaunching the same job line continues.

Launch: `python3 -m src.cluster run --jobs experiments/e184_tabpfn_lora_head_mmc.jobs --gpu-slots 1 --only deepfive` (unit `nmr-e184_tabpfn_lora_head_mmc`, log `logs/cluster/e184_tabpfn_lora_head_mmc.out`). Host deepfive, 1 GPU. Resumable from `models/tabpfn/e184/state.pt` (saved every 5 epochs and at each eval); relaunching the same job line continues.
### E185: IA_01..IA_04 sklearn MLPs at 1x and 10x TV epochs (registered 2026-11-17)

User request: retrain the four live MLP legs at 10x the Timmermann-Vulicevic reference epochs and compare against the current recipe.

Design: `results/ia_mlp_10x/run.py`, sklearn `MLPRegressor` with the pinned registry configs (IA_01 from `models/live/IA_01.json`), 4 rolling holdouts of 130 train / 22 gap / 30 validation / 130 test dates (520 scored dates, 2024-10-17 to 2026-10-14), seed 42. Arms: current recipe (`before`), 1x TV reference epochs (`reference`), 10x final, 10x selected by 5-epoch median validation R². Report `results/ia_mlp_10x/build_report.py` -> `mlp_10x_report.pdf`. Ran 2026-11-16 on `westeros`; registered after the fact.

Finding: 10x final lowers proxy payout for all four models (paired interval excludes zero for IA_02, IA_03, IA_04); 10x selected recovers most of the gap but beats the current recipe only for IA_01 (+0.67 cumulative payout over 520 dates).

### E186: IA_01..IA_04 as torch MLPs trained on an MMC loss at TV reference epochs (registered 2026-11-17)

User request: use the torch-based MLP with sklearn initialisation, an MMC loss, the datasets of IA_01..IA_04 and the epochs from the Timmermann-Vulicevic compute rule; design and run it.

Design: `experiments/e186_ia_torch_mmc.py`. Data, features, folds and pinned hyperparameters identical to E185. Network built by `TorchMLPRegressor._build` (sklearn Glorot-uniform on weights and biases, float64, relu, Adam with sklearn betas/eps, sklearn L2 `0.5*alpha*sum W^2/batch`). Epochs fixed at the TV reference `max(floor(0.2 P^0.8),1)+10` (IA_01 78, IA_02 426, IA_03 72, IA_04 402), no early stopping, final weights scored. Batches are whole dates (~batch_size rows, dates reshuffled each epoch). Arms: `torch_mmc` (per-date negative BMC vs the real meta model, `nn.losses.BMCLoss` via `PerPeriodLoss`) and `torch_mse` (half-MSE control in the same loop); seeds 42-46. Fold 0's training window starts before the meta model (2024-05-07): on a date without meta the reference is constant and the loss is per-date CORR (34% of fold-0 rows have meta; folds 1-3 have 100%). Scoring identical to E185 (CORR full cross-section; proxy MMC and payout on meta rows), so E185's `before`/`reference` arms are paired on the same 520 dates. Per-epoch validation CORR/MMC, loss split (data / L2), prediction std and weight norm logged to `curves.csv`.

Launch: `systemd-run --user --unit nmr-e186 python3 experiments/e186_ia_torch_mmc.py --device cpu --workers 12` on `westeros` (12 of 48 cores; CPU because the nets are tiny and per-date loss steps are launch-bound). Output `results/e186_ia_torch_mmc/` (fits.csv, curves.csv, daily_scores.csv, manifest.json), log `logs/e186.log`. The panel is pinned to 2026-10-14 (E185's last date; train.parquet later gained 2026-10-15) so all arms share the same 520 dates. Report: `python3 experiments/e186_report.py` -> `results/e186_ia_torch_mmc/e186_report.pdf`.

Finding (2026-11-17, 160 fits, 5 seeds): **training on the MMC loss at TV epochs is worse than the deployed recipe for all four models.** Proxy payout, deployed sklearn recipe -> torch MMC (paired delta, 95% block-bootstrap interval): IA_01 0.0520 -> 0.0458 (-0.0062 [-0.0213, +0.0094]), IA_02 0.0871 -> 0.0357 (-0.0514 [-0.0762, -0.0286]), IA_03 0.0783 -> 0.0319 (-0.0464 [-0.0621, -0.0306]), IA_04 0.0752 -> 0.0330 (-0.0422 [-0.0607, -0.0245]). With the torch loop held fixed, MMC vs MSE is clearly lower for IA_01, IA_02 and IA_03 and higher only for IA_04, where the MSE control itself collapsed (alpha 10 at lr 3.2e-3 drives weight norm to ~1e-3; payout 0.0132). The torch MSE control reproduces the sklearn recipe for IA_01-03 (within ±0.006, intervals include 0). Mechanism: under the MMC loss, mean validation MMC peaks at EPOCH 1 for every model (0.063-0.071) and decays to 0.016-0.031 by the TV epoch, i.e. the per-date BMC objective overfits within a few passes, and the TV schedule (72-426 epochs) runs far past it. Not tested: an early-stopped or few-epoch MMC arm, a smaller alpha/lr for the scale-free loss, or a CORR+MMC mixture.

### E187: tabnet_ktz_tv on every dataset behind a deployed slot (registered 2026-11-18)

Request: "Take the models using TV limit design and run an experiment that applies this model the datasets we currently use behind the deployed models. Test the model on all datasets."

Design: E183's `tabnet_ktz_tv` unchanged (width 8-16, steps 2-5, GLU 1-2, E = floor(0.2*P^0.8)+10 epochs from each candidate's own parameter count on the dataset's columns), E183 walk-forward and capture, n_iter 40, n_splits 3, seed 42. Arms and datasets: tenc `v2.0-tenc` (23), interact `v2.0-interact` (36), cumlag3 `v2.0-cumlag3` (88), onehot_roll `v2.0-onehot-rollsel` (786), tenc_onehot `v2.0-tenc-onehot` (763), allfeat `v2.0-tenc-onehot-spread-interact` (788), allfeat_pp `v2.0-tenc-onehot-rollsel-spread-interact` (812), allfeat_pp_fenc `v2.0-tenc-onehot-rollsel-spread-interact-fenccausal` (834). raw and fenccausal are E183's `e183_{raw,fenccausal}_s42`. Not run: allfeat_pp_lag3 (878, CURUPIRA_SACI_04), which has no prebuilt research dataset. TV epochs on the wide sets are 1,059-2,944, so those arms run many times longer than E183's. Compare paired against the deployed slots on each dataset.

Launch: `python3 -m src.cluster run --jobs experiments/e187_cluster.jobs` (9-core slots, 4 search workers x 2 threads; hamilton/hopper/lovelace x2, deepfour, deepsix at plan time). Outputs `results/{metrics,forecast,per_period,hyperparams}/crypto/tabnet_ktz_tv/e187_*`, `results/tuning/crypto/tabnet_ktz_tv/e187_*_cv.csv`, logs `logs/cluster/e187_*.out`.

### E188: compute scaling laws for IARA-family MLPs, MSE+L2 vs MMC (registered 2026-11-18)

User request: train a grid of MLPs on the Numerai Crypto target, each to its Timmermann-Vulicevic compute limit, under (A) MSE + explicit L2 and (B) MMC; fit statistical and financial scaling laws; per-run training reports and a cross-run scaling-law report (full spec in the conversation, sections 0-11). Feature set chosen by the user: fenccausal only (IA_01/IA_02's 44 columns). Compute plan chosen: full spec, GPU multiplexed.

Measured (section 0): every fold has 39,000 train rows / 130 eras, 9,000 / 30 validation, 39,000 / 130 test, exactly 300 rows per era; fold 0 has 87 training eras without meta-model rows (dropped for loss B, logged per run; loss A keeps all 130). No `cv.py` exists in the repo, so the splitter is E186's purged walk-forward (22-date gaps for the 20-day target; test windows 2024-10-17..2026-10-14, identical for every model and loss). Grid: 36 targets 200..1M, but 700 and 800 both give w=8, so 35 distinct networks, P = 217..1,001,777, epochs 24..12,647, compute 1.4e-11..3.4e-5 PFLOP-days (6.4 decades), P/N 0.006..25.7 (P/N >= 1 from 40k; P/N ~ 10 at 400k). Batches of 4 whole eras -> 33 steps/epoch; fold-0 loss B uses 1 era/batch -> 43 steps.

Design: `experiments/e188_iara_scaling.py` (stages measure / smoke / worker / status). Architecture, initialisation, optimiser, epochs and checkpoint rule as the spec; target centred at 0.5 so validation R^2 is against predicting 0.5. Loss A alpha chosen from {0, 1e-6, 1e-4, 1e-2} by mean retained validation R^2 at P = 1k/10k/100k x 4 folds (seed 42), written to `alpha.json` automatically before any loss-A grid job starts. 5 seeds (42-46) x 4 folds x 35 sizes x 2 losses = 1,400 runs + 48 sweep runs. Per run: per-epoch parquet (losses, val CORR/MMC/CWMM/payout/era stats, grad-norm band, pred spread, churn, cumulative PFLOP-days), diagnostics parquet on a log + dense-tail epoch grid (train metrics, per-layer weight norm and update ratio), log-step snapshots, loss-B estimability (batch vs full-epoch MMC, two-batch gradient cosine), per-era test scores of the retained checkpoint.

Launch on `westeros` (the lab cluster is not reachable from this host): 8 GPU workers on the GTX 1080 Ti for P > 30k and 12 CPU workers for P <= 30k, each `systemd-run --user --unit nmr-e188-{gpu,cpu}N python3 experiments/e188_iara_scaling.py worker ...`, file-claim queue under `results/e188_iara_scaling/claims/`. Measured throughput: ~2.8 ms/step effective for the 1M network with 8 processes sharing the card; estimate 3-4 days wall time. The GPU is shared with E182. Logs `logs/e188/`.

### E189: normalisation of frozen Chronos-2-Synth hidden states, 5 axes x 7 transforms (registered 2026-11-19)

**Request (user, 2026-11-19).** How does normalisation change the predictive value of frozen
Chronos-2-Synth hidden states for Numerai Crypto? Use the walk-forward "Ridge KTZ" workflow, tuned
from scratch per feature set, over all 35 pairings of 5 normalisation axes and 7 transformations, with
raw-embedding, v2.0 full-universe and v2.0 matched-coverage ridge baselines. Register the full design
first, validate before the full run, select on development evidence only and confirm on a reserved
chronological period.

**"Ridge KTZ" resolved.** There is no `ridge_ktz` in `src/prediction/models.py`; KTZ priors exist only
for lgb/catboost/rf/mlp/tabnet, and every prior study (E131, E132, E145, E159 `MODEL_ALIASES`, E170)
maps "ridge_ktz" to the repo's `ridge` wrapper (`RidgeWrapper` = plain `sklearn.linear_model.Ridge`,
no scaler, `fit_intercept=True`, which centres but does not scale X). The tuning workflow is the
feature-set-evaluation protocol of E131-E146 (`experiments/e133_histfe_factorial.py`): `H.blocks`,
`H.pick`, `H.fit_predict`. E189 uses that estimator and that workflow; only the alpha grid is widened
(below), as the protocol's 6-value grid was set for 22 binned features, not 768 heterogeneous ones.

**Frozen extraction (reused cache, verified before use).**
- Model `autogluon/chronos-2-synth` at HF revision `3607918a9fd027d5c465d8213e46b98e2c041cea`
  (120M Chronos-2 architecture, d_model 768, pretrained on synthetic series only), `eval()`,
  `inference_mode`, fp16 autocast (E143 measured cosine >= 0.99999 against fp32). No fine-tuning.
  Venv `/scratch/danielco/chronos2/venv`: chronos-forecasting 2.3.2, torch 2.7.1+cu118,
  transformers 5.17.0, numpy 2.2.6, pandas 2.3.3.
- Extractor `src/data_mani/chronos2_syn_repr.py`, mode `cov`: per (ucid, D) the coin's
  `target_binned_return_20` series as target with its 22 v2.0 features as covariates, one group of 23
  variates per coin (`group_ids` = coin, so group attention never mixes coins). Context = last 256 grid
  steps up to the label wall; label wall = last grid date <= D - 35 days (E142: label resolves ~30d
  after its date, train served 32d behind live); features of dates in (wall, D] enter as known-future
  covariates (NaN -> 0.5 with mask 1); a row needs >= 32 resolved labels in its context, else no state
  (NaN). Horizon h = grid steps from wall to D, `num_output_patches = ceil(h/16)`, selected patch =
  `(h-1)//16`, i.e. the output patch containing D's step.
- Extraction point: `Chronos2Model.forward` slices `encoder_outputs[0][:, -num_output_patches:]` (after
  `final_layer_norm`) into `output_patch_embedding`; the extractor returns `hidden[target_rows,
  -n_patches:][:, patch]` -- that exact tensor, target variate only. Shape per row (768,), float16 on
  disk. One vector per covered (asset, date); no PCA, pooling or quantiles.
- Cache `/scratch/danielco/c2syn/part_000..034.npz` (`cov_rows`, `cov_reps`), keys
  `rows.parquet` (sha256 f346ba66...), written 2026-11-06 for E143 from the then-current v2.0 panel.
  Verified 2026-11-19: its 523,096 keys equal today's refreshed `data/crypto/v2.0` rows <= 2026-10-05
  in order; shipped features and label equal `v2.0-asof` (2026-10-24) on all 504,196 shared rows.
  Frozen panel for this study: `data/crypto/v2.0-asof-e189/` (copy of v2.0 on 2026-11-19, train
  sha256 778b47ce..., meta 5a047870...). Before the full run a GPU replay (validation step 3/4 below)
  must reproduce cached states from the frozen panel; if it does not, the cache is rebuilt and
  this section amended before any arm is fitted.
- Provenance vs causality. Pipeline causality is the wall/context rule above (cut-date replay
  tests it). Model provenance: synthetic-only pretraining, so no real crypto series can have been
  memorised; the architecture and synthetic generator were designed by the Chronos-2 authors after
  work on real data (incl. periods overlapping 2020-2026), a weak, unquantifiable channel recorded
  as a limitation, not a leak.

**Tensor and rows.** E[t,a,j], t = panel dates <= 2026-08-20 (the last resolved 20-day label),
a = ucid, j = 0..767; explicit date/ucid/coordinate maps; missing (a,t) stay NaN, never balanced.
One row per (date, asset) = that asset/date's 768 normalised coordinates, nothing else (no other
assets, adjacent dates, targets or identifiers). The 22 v2.0 features are Chronos covariates and the
B_* baselines only, never appended to an embedding arm.

**Common causal coverage mask.** M(a,t) = 1 iff the cov state exists (rule above; it depends only
on labels before the wall). In the study window 87.5% of rows, 244..296 covered of 300 per date
(>= 20 on every date, so the per-date minimum below never binds there). Every embedding arm and
B_COV fit, tune, normalise and predict on exactly M = 1 rows; uncovered rows enter no statistic.

**Normalisation axes** (reference set for coordinate j of row (t,a)):
1. `CS` across assets: covered rows of date t (min 20; else rows of that date excluded from all arms).
2. `TS` per-asset time: asset a's covered rows on the fold's TRAINING dates only (min 30; an asset
   with fewer uses the POOL reference of that fold -- preregistered fallback, share reported).
3. `POOL` dates x assets: all covered rows on the fold's training dates.
4. `DIM` embedding dimensions: the 768 coordinates of row (t,a) itself (per row, not per coordinate).
5. `GRP` across assets within size group: covered rows of date t in the same group. Group = the
   row's shipped `feature_market_cap_avg_20d` bin on date t (the causal size bucket of E148), merged
   to 3 groups: small {0, .25}, mid {.5}, large {.75, 1} (5/20/50/20/5 -> 25/50/25%, because the
   5-bucket edges hold ~13 covered coins). Min group size 20; a smaller (t, group) cell uses the CS
   reference of date t. Frozen before any arm is scored; never chosen on performance.
CS, DIM and GRP use only the current snapshot (no future dates, no labels), applied identically to
historical training rows (each on its own date's snapshot). TS and POOL references are fitted on the
fold's training partition and frozen for its validation/test rows, refitted separately for the inner
fold (fit dates only) and for the outer fold (all 130 training dates).

**Transformations** (x = value, R = reference values, n = |R|, float64, eps = 1e-6):
1. `Z`: (x - mean R) / sd R (ddof 0).
2. `ROB`: (x - median R) / (q75 - q25), linear-interpolated quantiles.
3. `MM`: (x - min R) / (max R - min R), clipped to [0, 1] (only train-referenced axes can exceed it).
4. `PCT`: p = (L + 0.5 E) / n with L = #{r < x}, E = #{r = x} (mid-rank; equals (avg rank - 0.5)/n
   inside the reference, and evaluates new rows against the frozen training distribution).
5. `RNK`: Phi^-1(clip(p, 0.5/n, 1 - 0.5/n)).
6. `CZ`: clip(Z, -3, 3).
7. `BIN`: q = clip((L + 0.5 E - 0.5)/(n - 1), 0, 1), bins 0/.25/.5/.75/1 at inclusive cuts
   .05/.25/.75/.95 in integer arithmetic -- for x inside R this is exactly
   `history_features.bin_ties_preserved` (Numerai's rule, 38,720/38,720 shipped cross-sections), so
   `CS_BIN` reproduces the repo's cross-sectional processing (what build_c2syn_c2ft did to these
   dims); BIN on TS/POOL/DIM/GRP is an experimental extension, labelled as such.
Constant reference (sd, IQR or range < eps, or n = 1): neutral value Z/ROB/CZ/RNK 0, MM/PCT/BIN 0.5.
Ties: mid-rank throughout. NaN: preserved (uncovered rows carry none; any NaN coordinate in a covered
vector is a blocking validation error). Downstream: none -- the arm's matrix goes straight to Ridge
(intercept only); no fillna, rank-binning or scaler runs after the transform.

**Arms (38).** `N_<AXIS>_<TF>` for the 35 pairings; `RAW` (float16 states as float32, no
normalisation); `B_ALL` (22 v2.0 features, NaN -> 0.5 as the shipped convention, every row);
`B_COV` (same features, M = 1 rows only). One pooled ridge per arm per block; every arm tuned from
scratch; no coefficients or penalties shared.

**Walk-forward and tuning (from the repo).** Label `target_binned_return_20`. Dev = E131's 11 blocks
over the 1,602 panel dates < 2026-02-20 (`H.blocks`: train 130 dates, purge 22, predict next 130;
predictions 2020-08-28..2026-02-19). Confirmation = block 11, train = last 130 dates before a 22-date
purge, predict 2026-02-20..2026-08-20 (130 dates, the keys of
`results/reports/crypto/e132_feed_fillind/preds/keys.parquet`); reserved now, before any arm is fitted
or inspected. Inner tuning (`H.pick` rule): fit on train[:-62], validate on the last 40 train dates,
criterion = mean per-date Spearman(pred, label) on the arm's eligible val rows; refit on all 130
dates with the pick. Alpha grid, common to all 38 arms: 25 log-spaced values 1e-3..1e9 (half-decade).
Expansion rule (inner data only): if a block's pick is a grid end, extend 6 points (3 decades) on that
side and re-pick, at most twice; every boundary pick and expansion is recorded. Scale audit: per arm
and block, trace(X'X)/(n p) is reported next to the chosen alpha. Ridge is deterministic; seed 0.

**Scoring.** Per date, numerai_corr and MMC (`correlation_contribution` vs the lagged-in-file meta
model, available 2024-05-07..2026-08-06), payout = 0.1 corr + mmc per date, via `e131_analysis`
(`corr_one`, `contrib_one`, `describe`: mean, sd, sharpe, max drawdown, NW lag-20 t, 20-date block
bootstrap). Two views: (a) routed full universe, E134/E135 `R_SLOT` -- B_ALL ranks every coin;
covered coins are reordered by the arm (or B_COV) inside the exact set of B_ALL's covered rank slots;
uncovered coins keep their exact B_ALL rank; (b) covered-only diagnostics on M = 1 rows. Payout/MMC
comparisons use the common meta window.

**Selection and inference (preregistered).** Primary metric: dev mean routed CORR (all dev dates).
Families of paired per-date NW(lag 20) tests, Holm-corrected within family: (i) 35 arms vs RAW,
(ii) 35 arms vs B_COV (routed), (iii) within each axis 7 transforms vs the axis's dev-best,
(iv) within each transform 5 axes vs its dev-best; plus the "indistinguishable set" = arms not
Holm-significantly below the dev-best (34 tests). Selected arm = argmax dev routed CORR, frozen, then
evaluated on confirmation against RAW, B_COV and B_ALL (Holm over these 3). Confirmation numbers for
the other arms are reported as exploratory only. Stability: per dev block, dev halves; coverage
cohorts: size group, coin tenure tercile (prior covered rows), date coverage-share tercile. Valid
conclusions include "indistinguishable", "raw is competitive" and "none beats v2.0".

**Validation before the full run** (`--validate`, JSON in the output dir): key alignment and one
vector per covered key; width 768 and finiteness; GPU re-extraction of 4 dates from the frozen panel
vs cache; cut-date replay (`--audit-cut` path) of 4 dates vs cache and of their transformed features;
future-perturbation invariance of earlier features on every axis; per-axis reference-set unit tests;
inner-fold statistics drawn only from inner fit dates; deterministic ties/constants/NaN/small groups;
identical scored keys and folds across the 35 arms + RAW + B_COV; uncovered rows unable to move any
statistic; routing preserves uncovered ranks and the covered slot set. Then the missing-values audit
(`.claude/skills/numerai-crypto-missing-values-audit`) before any interpretation.

**Commands** (host deepfour for GPU checks; ridge arms through `src.cluster`):
```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/danielco/hf HF_HUB_DISABLE_XET=1 /scratch/danielco/chronos2/venv/bin/python experiments/e189_extract_verify.py
python3 -m pytest -q src/tests/test_embedding_norm.py
python3 experiments/e189_c2norm.py --validate
python3 -m src.cluster run --jobs experiments/e189_c2norm.jobs
python3 experiments/e189_c2norm.py --score && python3 experiments/e189_c2norm.py --report
```
**Outputs.** `results/reports/crypto/e189_c2norm/` (preds/<arm>.parquet with all panel keys and
coverage, picks, scores, manifest.json with fingerprints, validation.json, report PDF);
`/scratch/danielco/e189/` (per-block transform parameters, replay states).

**Findings (2026-11-19, all 38 arms; report `results/reports/crypto/e189_c2norm/E189_report.pdf`).**
Verdict: **no reliable winner; no normalisation makes frozen Chronos-2-Synth states competitive with v2.0.**

| | dev routed CORR | dev payout | test routed CORR | test payout |
|---|---|---|---|---|
| B_ALL (v2.0, full universe) | 0.1182 | 0.1021 | 0.1218 | 0.0320 |
| B_COV (v2.0, covered rows, routed) | 0.1121 | 0.1004 | 0.1238 | 0.0330 |
| RAW (embedding, no normalisation) | 0.0960 | 0.0507 | 0.0990 | 0.0262 |
| N_POOL_MM (dev-best, selected) | 0.0990 | 0.0560 | 0.0995 | 0.0274 |
| N_CS_BIN (repo's cross-sectional bins) | 0.0973 | 0.0562 | 0.0995 | 0.0268 |
| N_TS_* range | 0.067-0.079 | 0.048-0.061 | 0.069-0.089 | 0.032-0.036 |

- **Across transformations and the CS/POOL/DIM/GRP axes, normalisation is a null.** 28 of those arms fall within 0.096 +- 0.004 dev routed CORR. 22 of 35 are indistinguishable from the dev-best after Holm over 34 tests. Ranking of arms between dev halves: Spearman 0.34.
- **vs RAW:** no arm is better after Holm. The best is N_POOL_MM, +0.0030 (t +2.27, Holm p 0.64). **The TS axis is the one robust effect, and it is harmful:** -0.017 to -0.029 CORR (Holm-significant for 6 of 7; TS_MM t -3.0, Holm p 0.09). Its payout is flat against RAW, so own-history scaling seems to strip the coin-level component the meta model already prices.
- **vs matched v2.0 (B_COV, routed):** every arm is lower, at -0.013 to -0.045 CORR. Only TS is Holm-significant; the others sit at t -1.6 to -2.4. On the meta window 2024-05-07..2026-08-06, payout is lower by -0.040 to -0.059, uncorrected t -2.4 to -4.7.
- **Confirmation** (2026-02-20..2026-08-20, frozen N_POOL_MM; Holm over 3): +0.0004 CORR vs RAW (t +0.34), -0.024 vs B_COV (t -1.54), -0.022 vs B_ALL (t -1.43). Payout deltas are inside their bootstrap intervals.
- **Cohorts (covered-only):** the states beat B_COV on small coins (dev 0.081 vs 0.063) and middle-tenure coins, and lose badly on large and old coins (0.058 vs 0.109). Young coins are worst for every model.
- **Penalty:** the grid was adequate. Boundary picks remaining after the registered expansion: N_TS_Z, N_TS_CZ, N_TS_BIN and B_ALL, one block each. Median alpha/(n var) is 0.3-8.6 across embedding arms.
- **Validation:**
  - extraction is bit-exact at 4 dates, from the frozen panel and from cut panels;
  - transformed features replay exactly from cut-panel states;
  - future perturbation leaves every arm's earlier features invariant;
  - the live-path prediction replay matches stored predictions to <= 2.2e-13;
  - routing assertions passed on every date.

  **Missing-values audit: PASS WITH CONDITIONS.** The conditions: a retrospectively served universe; coverage tied to tenure, with no tenure placebo (immaterial for a negative claim); expansion-unequal search budgets (<= 12 candidates); TS sends ~10% of rows per block to its POOL fallback.

**Limitations.** Ridge only; the zero-shot cov state only; the label-wall horizon is fixed at 25 weekday steps (output patch 1). fp16 states quantise coordinates, producing ties, which mid-ranks handle. The confirmation block was used by earlier studies for other feature sets. Chronos-2-Synth's synthetic-only pretraining rules out memorising real series, but its design postdates real-data work over these years (weak channel).

**How to apply:** do not normalise these states expecting to rescue them. Together with E143/E144/E146, this closes frozen Chronos-2 hidden states as ridge features. If they are ever used again, CS or POOL scaling are equivalent defaults; avoid per-coin own-history scaling.

### E190: v2.0 + normalised Chronos-2-Synth states, TS_MM / GRP_MM / POOL_MM (registered 2026-11-20)

**Request (user, 2026-11-20):** take N_TS_MM, N_GRP_MM and N_POOL_MM from E189 and rerun the same experiment, but with the downstream ridge's information set the merge of v2.0 and the embedding instead of the embedding alone.

**Design.** Identical to E189 in every respect except the feature matrix: on covered rows, X = [22 shipped v2.0 features (NaN -> 0.5, the B_* convention) | 768-d transformed state] = 790 columns. Transforms, references (TS/POOL fit on training rows per inner and outer fold, GRP per-date snapshot), schedule (E131 11 dev blocks + test block 11 held for confirmation), inner alpha pick, alpha grid logspace(-3, 9, 25) with the registered boundary expansion, model (repo `ridge`, intercept, no scaler), R_SLOT routing against B_ALL and covered-only scoring are unchanged.

- Arms: `V_N_TS_MM`, `V_N_GRP_MM`, `V_N_POOL_MM`, and control `V_RAW` (v2.0 + raw states), which was added so the transform's effect is separable from the effect of adding the states at all.
- Reused from E189 (same rows, schedule and code path): `B_ALL`, `B_COV`, `RAW` and the three embedding-only arms.
- Primary comparison: each V arm vs `B_COV` (does the embedding add to v2.0 on matched rows?); secondary: vs `V_RAW` and vs its E189 embedding-only twin. Paired Newey-West lag 20, Holm within each family, dev only for selection; test block reported as confirmation.
- Known design limitation, stated up front: one ridge alpha penalises both blocks. In E189 B_COV picked alpha 1e6-3e8 while the embedding arms picked 1e2-1e4, so a shared penalty cannot suit both blocks; the pipeline does not block-scale, and neither does this study.

**Launch:** `python3 experiments/e190_c2norm_v2.py --fit` then `--score`, `--report`; host deepfour (local, ~3 min per arm). Outputs: `results/reports/crypto/e190_c2norm_v2/`, `/scratch/danielco/e190/`.

**Findings (2026-11-20): NO IMPROVEMENT -- merging the states into v2.0 never beats v2.0 alone.** Routed R_SLOT; CORR over 1,430 dev dates, MMC/payout over the 468 with the meta model.

| arm | dev CORR | dev MMC | dev payout | p.sharpe | dd(pay) | test CORR | test payout |
|---|---|---|---|---|---|---|---|
| B_ALL | 0.1182 | 0.0860 | 0.1021 | 1.065 | 2.073 | 0.1218 | 0.0320 |
| B_COV | 0.1121 | 0.0846 | 0.1004 | 1.028 | 2.112 | 0.1238 | 0.0330 |
| V_N_POOL_MM | 0.1068 | 0.0593 | 0.0722 | 0.809 | 2.145 | 0.1203 | 0.0311 |
| V_N_GRP_MM | 0.0980 | 0.0519 | 0.0635 | 0.736 | 1.876 | 0.1185 | 0.0325 |
| V_N_TS_MM | 0.0842 | 0.0495 | 0.0594 | 0.773 | 0.732 | 0.1095 | 0.0333 |
| V_RAW | 0.1033 | 0.0479 | 0.0589 | 0.700 | 1.720 | 0.1124 | 0.0325 |

- Dev vs B_COV: every merged arm loses payout after Holm (-0.028 to -0.042, t -3.1 to -4.2); CORR deficits -0.005 (POOL_MM, n.s.) to -0.028 (TS_MM, t -3.3).
- Within merged arms: POOL_MM beats V_RAW on payout (+0.0133, t +3.84, Holm p 0.0004) at flat CORR; TS_MM costs CORR vs V_RAW (-0.019, t -3.07, Holm-significant), as in E189.
- Confirmation: all merged arms tie B_COV on payout (|t| <= 0.4) and sit below it on CORR (n.s.); merging adds +0.02-0.03 CORR over each E189 embedding-only twin.
- Mechanism: merged fits pick alpha ~1e3 like the embedding arms, while v2.0 alone wants 1e6-1e9 in most blocks, so one shared penalty under-shrinks the 22 shipped features and lets 768 state columns dominate. A per-block penalty (or block scaling) was not tested.
- Missing values: rows, coverage and routing are E189's; its audit (PASS WITH CONDITIONS) applies unchanged.

**How to apply:** do not add frozen Chronos-2 states to v2.0 in a single-penalty ridge. The only untested door is a two-penalty / block-scaled ridge; given E143-E190, low prior. Report: `results/reports/crypto/e190_c2norm_v2/E190_report.pdf`.

### E191: cross-asset variants of the Chronos-2-Synth state (registered 2026-11-20)

**Request (user, 2026-11-20):** rerun the E190 experiment with the own hidden state replaced by (2') own + average of other assets' states within the size group, (2'') own minus the other-asset average fed to a nonlinear model, (3') own + a compressed market state with interactions; same performance table. The literal versions (own + all-coin mean; flat concatenation of every coin's state) were declined because under a linear ridge scored per-date rank they reduce to the own state (all-coin LOO mean = date constant - own/(n-1)) or to a date constant (flat concatenation; also ~1M columns).

**Design.** Everything as E190 (frozen states, schedule with test block 2026-02-20..2026-08-20 held for confirmation, inner pick on last 40 train dates after 22 purge, v2.0 22 features always included, train-referenced transforms refit per fold, R_SLOT routing vs E189's B_ALL, covered-row scoring).

- 2' `G_<T>` (ridge, alpha grid + expansion as E189): X = [v2.0 | T(own) | T(m_grp)], m_grp = mean of the OTHER covered coins' raw states in the same date x size group (min cell 40). T in {TS_MM, GRP_MM, POOL_MM, RAW}, applied column-wise. Note GRP_MM(m_grp) is exactly 1 - GRP_MM(own), so G_GRP_MM is a collinear control.
- 2'' `L_*` (lgb_ktz: E133 FIXED, 6-candidate GRID num_leaves x colsample, same inner pick, one seed 0): L_B_COV = v2.0; L_OWN_RAW = v2.0 + own; L_REL_RAW / L_REL_TS_MM = v2.0 + T(own - m_all), m_all = mean of all OTHER covered coins on the date. POOL_MM is a global monotone map (identical to RAW for trees) and GRP_MM(own - m_all) = GRP_MM(own), so those are not run.
- 3' `M_<T>` (ridge): X = [v2.0 | T(own) | 16 market PCs | 32x16 products of own PCs and market PCs]; market state = per-date mean of all covered raw states (date-level, uses only that date); market PCA on the fit dates' means, own PCA on fit rows of T(own); all scores standardised on fit rows; refit per inner and outer fold.
- Comparisons: each arm vs B_COV and vs the matching E190 own-state arm (V_<T>; for L_* vs L_OWN_RAW and L_B_COV); paired Newey-West lag 20, Holm within family; dev for selection, test as confirmation.
- Limitation: lgb with one seed (house norm is 5) to keep cost ~1 h.

**Launch:** `python3 experiments/e191_c2cross.py --fit --arms ...` (ridge arms on deepfour; lgb arms via `python3 -m src.cluster run --jobs experiments/e191_c2cross.jobs`), then `--score`, `--report`. Outputs: `results/reports/crypto/e191_c2cross/`.

**Findings (2026-11-22): NO IMPROVEMENT.** Dev payout, sorted (B_ALL 0.1021, B_COV 0.1004): L_REL_N_TS_MM 0.0863, L_REL_RAW 0.0782, L_OWN_RAW 0.0775, L_B_COV 0.0744, G_N_POOL_MM 0.0729, V_N_POOL_MM 0.0722, ... M_RAW 0.0461 (full table `results/reports/crypto/e191_c2cross/table_by_payout.csv`, report `E191_report.pdf`).
- 2' (G_*) ties its own-state control on every transform (payout delta -0.003..+0.001, none Holm-significant), and all four lose to B_COV on payout after Holm. Peer-group context adds nothing.
- 3' (M_*) is strictly worse: CORR 0.015-0.038 below the own-state control (Holm p <= 0.003, all four), and payout lower (M_RAW -0.013, Holm 0.002). The 512 interaction columns add variance under one shared ridge alpha. Test shows M_N_POOL_MM/M_RAW +0.011 payout over control (t ~2.5, Holm 0.16), the opposite sign to dev, so it is not a lead.
- 2'' (L_*): trees are the best E191 family, but the gain comes from the model class. The own state adds nothing to LightGBM on v2.0 (L_OWN_RAW vs L_B_COV +0.003, t +0.51), and neither does the relative state (L_REL_RAW vs L_OWN_RAW +0.0007, t +0.79). L_REL_N_TS_MM is +0.012 over L_B_COV (t +1.68, Holm 0.75). Every L arm trails ridge B_COV on payout by 0.014-0.026. One seed only.
- Confirmation (test): null throughout.
- With E143/E146/E189/E190, this closes frozen Chronos-2-Synth states as features: own or cross-asset, linear or tree.

### E192: options 1-3 for the Chronos-2-Synth state, one ridge per coin (registered 2026-11-20)

**Request (user):** "What if instead of estimating ridge in the panel of assets I estimate it one asset at a time and then put them together? Does my original 2 and 3 work in that case?" then "Run my original 2 and 3 per coin then." The original options: (1) own hidden state, (2) own hidden state + average of the other assets' hidden states, (3) flat concatenation of all assets' hidden states for the time step.

**Why per coin changes the answer:** pooled (E189/E190), the date-common part of options 2 and 3 enters every coin with the same weight and cannot reorder a date; per coin, each coin has its own weights on it, so it can.

**Design:** E189's design and code path (frozen Chronos-2-Synth cov-mode 768-d states, cache `/scratch/danielco/c2syn`, panel `data/crypto/v2.0-asof-e189`, E131 11 dev blocks + test block 2026-02-20..2026-08-20 held for confirmation, R_SLOT routing vs B_ALL plus covered-only scores, E189 scorer).
- Blocks per coin i, date d (RAW states): OWN `x_i`; LOO `[x_i | (S_d - x_i)/(n_d - 1)]` over the covered coins on d; CAT every coin's state in a fixed slot, 0 when absent.
- Arms: `C_OWN`, `C_LOO`, `C_CAT` (state only); `CV_OWN`, `CV_LOO`, `CV_CAT` (22 v2.0 features + state); `CV_B` (v2.0 only, per coin) as the per-coin control.
- Model: ridge with an unpenalised intercept fitted separately for each coin and block on that coin's covered training rows (<= 130), solved in dual form (centred Gram, one eigendecomposition per coin, all alphas in one pass). CAT uses the date Gram `K[d,d'] = sum_c x_c,d . x_c,d'`, built once, so the ~1M-column matrix is never formed. `--check` confirms the dual solve equals sklearn `Ridge` on explicit features to 1.2e-12 (relative) for 3 coins x 7 arms, CAT included.
- Alpha: one per arm and block, shared by all coins, picked by E189's inner rule (fit on train[:-62], validate on the last 40 train dates, mean per-date Spearman over modelled coins, 25-pt grid 1e-3..1e9 + boundary expansion).
- Coins with < 30 covered training rows in the block (< 20 on the inner fold) take the pooled twin's prediction (RAW for C_*, V_RAW for CV_OWN/LOO/CAT, B_COV for CV_B); the share is logged per block.
- Comparisons: LOO and CAT vs OWN (does the cross-asset part add anything per coin), each per-coin arm vs its pooled twin, CV_* vs CV_B and B_COV; paired Newey-West lag 20, Holm over all pairs per window and metric.
- Caveats stated up front: per coin there are <= 130 rows with 768-300k columns and ~95% label overlap, so the shared alpha carries the regularisation; per-coin intercepts rank coins by their own recent mean target (the same in every arm, so the OWN/LOO/CAT contrast is clean); RAW scale only, so in CV_* one alpha shrinks v2.0 and the state together (E190's finding).

**Launch (hamilton, cluster manager):** `python3 -m src.cluster run --jobs experiments/e192_c2percoin.jobs` -> `python3 experiments/e192_c2percoin.py --fit --workers 7`; then `--score`, `--report`, PDF via `python3 -m src.analysis.html_pdf`. Outputs: `results/reports/crypto/e192_c2percoin/` (preds, picks, dual_check.json, scores, table, tests, report). Log `logs/cluster/e192_c2percoin.out`.

**Relaunch 2026-11-21:** the first run (2026-11-20 04:18, dostoievski) was stopped by systemd at 04:28 with no Python error, 1-3 of 12 blocks into each arm and nothing saved (arms wrote only at the end). `fit_arm` now checkpoints every block to `preds/<arm>.blocks/blockNN.{parquet,json}` (json written last marks completion) and resumes from them, so a killed run is relaunched with the same command.

**Findings (2026-11-21, relaunch ran 78 min on the cluster with no restarts; report `results/reports/crypto/e192_c2percoin/E192_report.pdf`): NO IMPROVEMENT.**
- Option 2 == option 1 per coin: C_LOO - C_OWN dev CORR -0.0004 (t -0.60), test +0.0000; same for CV_LOO - CV_OWN. With RAW states the leave-one-out mean of ~250 coins varies far less over dates than the own state, so under the shared alpha it barely enters the Gram.
- Option 3 (flat concatenation): dev CORR +0.0106 over C_OWN (t +2.08, Holm 0.19), test -0.0055 (t -0.99); CV_CAT vs CV_OWN the same pattern; payout null in both windows. Alpha picks swing 1e-9..3e8 across blocks.
- Per-coin fitting loses CORR against pooled: C_OWN vs RAW dev -0.034 (t -4.1), CV_B vs B_COV dev -0.039 (t -4.6). Per-coin dev payout (0.067-0.077) beats the pooled state arms (RAW 0.051, n.s.) through the per-coin intercept, but stays below B_COV 0.100. ~16% of covered rows fell back to the pooled twin.
- Untested: a standardised LOO block (the one variant that could change the option-2 null).

### E193: E189 rerun with Toto 2.0 states (registered 2026-11-20)

**Request (user, 2026-11-20).** "Register a new E189 with updated number and replace chronos-2 with toto,
TabPFN-TS, TempoPFN, t0, and timemistral. Run one full experiment for each and report one table per new
tsfm with corr, mmc, payout, sharpe.p, maxdd.p, and sort descending by payout." One study per model:
E193 Toto 2.0, E194 TabPFN-TS, E195 TempoPFN, E196 t0, E197 Timestral ("timemistral").

**Design shared by E193-E197 = E189 with only the frozen state swapped.** Everything downstream of the
state is E189's code path unchanged (`experiments/e193_tsfm_norm.py --tsfm <model>` imports
`experiments/e189_c2norm.py` and points it at the new cache): frozen panel `data/crypto/v2.0-asof-e189`,
rows <= 2026-08-20, E189's covered set (449,471 rows; the extractor recomputes the >= 32-resolved-labels
rule and asserts it equals the Chronos-2 cache's row set part by part), the 35 `N_<axis>_<tf>` arms
(axes CS/TS/POOL/DIM/GRP x transforms Z/ROB/MM/PCT/RNK/CZ/BIN) + `RAW`, repo `ridge` (sklearn Ridge,
intercept, no scaler), E131 blocks (11 dev blocks + held-out test block 2026-02-20..2026-08-20), inner
alpha pick on the last 40 train dates after a 22-date purge, alpha grid 1e-3..1e9 (25 pts) + boundary
expansion, train-referenced transforms refit per fold, R_SLOT routing against B_ALL. `B_ALL` and
`B_COV` are copied from E189 (identical rows, schedule, model and coverage -> identical predictions).

**State extraction** (`src/data_mani/tsfm_states.py`, adapters in `src/data_mani/tsfm_state_adapters/`):
per covered (ucid, D), the label series `target_binned_return_20` up to the wall (last grid date <=
D - 35d) in a 256-step context, the 22 shipped features over the context and, for the h steps from the
wall to D, as known-future covariates (NaN -> 0.5), exactly E189's inputs. Each adapter keeps ONE vector:
the model's last hidden state of the target series at the forecast position holding D -- the tensor its
forecasting head reads -- no pooling, PCA or quantiles; fp16 autocast on the RTX 2070s, frozen weights,
eval, no fine-tuning. Output `/scratch/danielco/tsfmstates/<model>/part_XXX.npz` (`cov_rows`,
`cov_reps`, float16) + `manifest.json` (checkpoint revision, dim, notes).

**Per-model choices.**
- E193 Toto 2.0, `Datadog/Toto-2.0-313m`, venv `/scratch/danielco/tsfm2/venv_toto`: target + 22
  `known_dynamic` covariates in one series; one forward pass (`decode_block_size=None`, leaderboard
  setting); state = `output_head` input (next-patch token) of the target variate at patch (h-1)//ps.
- E194 TabPFN-TS, TabPFN-3 time-series checkpoint (sha256 pinned), venv `/scratch/danielco/tsfm/envs/tabpfnts`:
  one TabPFN fit per row on the resolved context steps, X = [step index, 22 features], y = label; the
  test row is D (its step index and features); state = `get_embeddings` of that test row, n_estimators 1.
  Calendar/FFT features dropped (irregular early grid, 24/7 market).
- E195 TempoPFN 38M, venv `/scratch/danielco/tsfm/envs/tempopfn`: UNIVARIATE label series only -- TempoPFN
  merges batch x channel before its encoder, so covariates cannot reach the target; state = encoder
  output at prediction position h (input of `model.mlp`).
- E196 t0, `theforecastingcompany/t0-gamma`, venv `/scratch/danielco/tsfm2/venv_tt`: target + 22
  future covariates over context + horizon, `drop_constant_covariates=False`, no symmetric averaging;
  state = input of the decoder block at token ctx_patches - 1 + (h-1)//ps of the target row.
- E197 Timestral, `mistralai/Timestral-209M`, venv `/scratch/danielco/tsfm2/venv_tt`: target + 22 known
  covariates over context + horizon, rows assembled by the forecaster's `_assemble_rows`, one
  `predict_block`; state = `QuantileHead` input of the target row at the horizon patch holding D
  (un-mirrored pass only).

**Provenance.** TempoPFN and TabPFN are trained on synthetic priors (the TabPFN-3 time-series
checkpoint is a fine-tune whose data mix is not fully documented); Toto 2.0, t0 and Timestral saw real
series in (post-)training, so unlike Chronos-2-Synth a memorisation channel on 2020-2026
crypto/finance series cannot be ruled out. Recorded as a limitation; E189's cut-date replay (`--audit-cut`) is run per model.

**Launch.** States: `CUDA_VISIBLE_DEVICES=<g> <venv>/bin/python -m src.data_mani.tsfm_states --model <m>
--shard <k> --shards <n>` on deepfive/deepfour (4 RTX 2070s). Arms: `python3 -m src.cluster run --jobs
experiments/e193_tsfm_norm.jobs` (per model), then `--score` and `--report` per model, combined tables
in `results/reports/crypto/e193_e197_tsfm_tables.pdf`. Outputs per study:
`results/reports/crypto/<id>_<model>norm/` (preds, scores, tables). Host: lovelace (orchestration).

**Reporting.** One table per model: routed dev CORR, MMC, payout, payout Sharpe, payout max drawdown
(E189 conventions, payout rebuilt per period), all 38 arms sorted by dev payout descending; held-out
test payout alongside.

### E194: E189 rerun with TabPFN-TS states (registered 2026-11-20)

See E193 for the shared design and request; model-specific choices listed there. Output
`results/reports/crypto/e194_tabpfntsnorm/`, states `/scratch/danielco/tsfmstates/tabpfnts/`.

### E195: E189 rerun with TempoPFN states (registered 2026-11-20)

See E193. Univariate (label series only). Output `results/reports/crypto/e195_tempopfnnorm/`,
states `/scratch/danielco/tsfmstates/tempopfn/`.

### E196: E189 rerun with t0 states (registered 2026-11-20)

See E193. Output `results/reports/crypto/e196_t0norm/`, states `/scratch/danielco/tsfmstates/t0/`.

### E197: E189 rerun with Timestral states (registered 2026-11-20)

See E193. Output `results/reports/crypto/e197_timestralnorm/`, states `/scratch/danielco/tsfmstates/timestral/`.

### E198: cross-sectional statistics of raw features, crypto/v2.0-cs (registered 2026-11-20)

User request: build a dataset from v2.0 that adds cross-sectional features computed on the features' raw space
(before Numerai's bins), after first judging which v2.0 features deserve them.

Screen (dates < 2024-01-01, no fits; 2024+ left closed): raw values rebuilt from spot with the E157/E164-E166
definitions (rank agreement with the shipped bins 0.85-0.92). Survivors at Bonferroni |t| >= 3.6 over 176 tests:
the CS median/mean of the 20d Bollinger z moderates that feature's daily IC (t +5.5; Bowley skew t -5.1), and the raw
volatility carries information inside its bins (within-bin t -5.6 at 20d, -3.7 at 60d; Sharpe/RSI about -3). The
Bollinger median is 0.80-0.83 correlated with the median 20d return and Sharpe/RSI medians, so it is one
market-trend regime number (compare E68/E108). Close-price levels carry nothing; the shipped momentum is a dollar
change C[t]-C[t-w], so its CS moments are skipped; market cap is not in the spot store; kurtosis is never significant.

Dataset `data/crypto/v2.0-cs` (builder `python3 -m src.data_mani.build_cs_moments`): every v2.0 row and the 22
shipped features unchanged, plus 15 `feature_cs_*` columns. Raw signals attached as of D-1 (max 4 days stale),
wrong-coin series removed by build_vol_causal's identity gate, NaN kept where uncovered. Per-coin: 20-quantile
ranks of vol/sharpe/rsi 20+60 and 0.5+0.5*tanh(robust z/2) of log vol 20/60. Date-level (computed on the day's
gated coins, >= 30 of them, then causal trailing percentile): `cs_trend` (median Bollinger z), `cs_trend_skew`
(Bowley), `cs_vol_level`, `cs_vol_disp`, `cs_liq_disp` (std of log USD volume, Yahoo rows only since other venues
quote base units). Interactions 0.5+2*(f-0.5)*(m-0.5). Masks `cov_cs_coin`, `cov_cs_date`, `cov_cs_all`.

Evaluation (not launched): B_all / B_covered / F_covered on `cov_cs_all`, routed, ridge, mlp_ktz, lgb_ktz.
Ridge can use only the per-coin columns and the interactions; the date-level columns are for trees/MLP.
Host hamilton. Findings pending.

**Addition (user, 2026-11-20): one TabNet per coin, with a no-CS baseline.** "add to E193 experiment a baseline which
is tabnet on individual assets (same way as E193) but instead without any cs features" -- confirmed to mean E198,
one TabNet fitted per coin. Two arms on exactly the same rows:
- `F_TN_COIN`: per-coin TabNet on the 22 shipped + 15 `feature_cs_*` columns.
- `B_TN_COIN` (the requested baseline): per-coin TabNet on the 22 shipped v2.0 features only.

Both train only on `cov_cs_all == 1` rows, so the baseline sees the same coin/date keys as the candidate. Per coin, the
date-level CS columns are a time series the coin weights with its own parameters (pooled, they only enter via
interactions; compare E192). Matched across the two arms: walk-forward schedule and purge, one pinned TabNet config
(no per-coin search: a coin has <= 130 training rows; fixed in advance, never tuned on these data:
n_d=n_a=8, n_steps=3, lambda_sparse 1e-3, lr 0.02, batch 256, virtual batch 64, TV epochs),
epoch rule, seeds (3), and the fallback for coins with fewer than 30 training rows (they take the pooled B_covered
prediction in both arms). Per date, the per-coin predictions are ranked across coins and routed into B_all's
covered slots. Primary comparison: paired per-date `Route(F_TN_COIN) - Route(B_TN_COIN)`, Newey-West lag 20;
also each against pooled `B_covered`. Caveat: ~130 rows with ~95% label overlap per coin is far below what TabNet
normally needs (E18: default TabNet lost everywhere), so a null here says little about the CS columns themselves.

**Launch (2026-11-20, host hamilton).** Runner `experiments/e198_cs_eval.py`; writes OUTSIDE results/metrics/ to
`results/reports/crypto/e198_cs/` (preds/, per_date.parquet, table.csv, deltas.csv, E198_cs_report.html/.pdf).
Pooled configs are E131's fixed ones (`e133_histfe_factorial.FIXED`); seeds ridge 0, lgb_ktz/mlp_ktz/TabNet 0-2.
Schedule: E131 blocks 0-10 (development) + test block 11. Block 0 has no covered training rows (the identity gate
needs 40 rows of history), so the covered arms start at block 1 and block 0's dates score as B_ALL. Covered arms
route R_SLOT into their own model's B_ALL; TabNet routes into ridge's. Launch:
`python3 -m src.cluster run --jobs experiments/e198_cluster.jobs --exclude tolstoi dostoievski` (lanes: TabNet TN_B
and TN_F seeds run in sequence, beside lgb_ktz / mlp_ktz; `experiments/e198_score.sh` waits for all 23 prediction
files, scores, and prints the PDF on lovelace). Ridge fitted directly on hamilton before launch. Smoke check, ridge
only: F_COV - B_COV dev CORR -0.0075 (t -2.20), payout -0.0029 (t -1.28); test payout -0.0044 (t -2.30).

**Findings (2026-11-20): REJECT -- the CS columns do not help any model.** Report
`results/reports/crypto/e198_cs/E198_cs_report.pdf`; tables `table.csv`, `deltas.csv`. Paired F_COV - B_COV
(routed, Newey-West lag 20), dev n=1430 / test n=130:

| model | dev dCORR (t) | dev dpayout (t) | test dCORR (t) | test dpayout (t) |
|---|---|---|---|---|
| ridge | -0.0075 (-2.20) | -0.0029 (-1.28) | +0.0010 (+0.54) | -0.0044 (-2.30) |
| lgb_ktz | -0.0018 (-0.68) | +0.0002 (+0.06) | -0.0081 (-1.42) | -0.0152 (-3.82) |
| mlp_ktz | -0.0012 (-0.29) | +0.0039 (+1.04) | +0.0031 (+0.80) | -0.0020 (-0.57) |
| TabNet per coin, TN_F - TN_B | -0.0411 (-8.75) | -0.0472 (-5.91) | -0.0505 (-4.52) | -0.0239 (-2.26) |

No arm gains at |t| >= 2 anywhere; ridge loses dev CORR and test payout, lgb loses test MMC (t -4.16), mlp is null.
The trees and MLP -- the only models that can use the date-level regime columns -- get nothing from them, matching
E68/E108 (day-level momentum regime null). Training on the covered subset alone also costs every pooled model
against B_ALL (e.g. ridge dev payout 0.0977 vs 0.1007), as the routing standard expects.

Per-coin TabNet is unreliable as a model: the networks are undertrained (predictions mean -0.50 on a [0,1] label;
~190 single-batch steps per coin at 130 rows), so cross-coin ranks mostly rank per-coin offsets. TN_B's dev
payout 0.0581 trails ridge B_COV (t -3.24); its test payout 0.0507 (MMC 0.042) beats it (t +2.06) on one
130-date window with ~95% label overlap, while TN_F loses heavily to TN_B -- 15 more inputs on 130 rows make the
undertrained fit worse. Read the TabNet arms as "per-coin TabNet at this budget is noise", not as evidence about
the CS columns. A fair per-coin test would need far more optimisation steps per coin (smaller batch / epoch
multiplier), registered as an amendment. 20.6% of TabNet rows fell back to ridge B_COV (all of block 0 included).

**Correction (2026-11-20):** the TabNet arms above were UNTRAINED, not undertrained (batch 256 > rows, drop_last); see E199.

### E199: per-coin TabNet arms of E198, refitted so they train (registered 2026-11-20)

User request: "Yes, register and rerun" -- rerun E198's per-coin TabNet arms (`TN_B` without CS columns, `TN_F` with
them) with enough training to be a fair test.

**Correction to E198.** Its TabNet arms were not undertrained but UNTRAINED. `src/prediction/tabnet.py` fits with
`drop_last=True`; E198's batch 256 exceeds every coin's <= 130 training rows, so the only (partial) batch was dropped
every epoch and each "model" is its random initialisation (in-sample |mean pred - mean label| 0.99, train R2 -40 on
30 development coin-blocks). E198's TabNet rows are void. The wrapper now raises when rows < batch_size.

**Design.** E198 unchanged except TabNet `batch_size=32, virtual_batch_size=16`, TV epochs (epoch_mult 1). Chosen on
30 development coin-blocks (3 coins x blocks 1-10, seed 0) by TRAINING fit only -- no target score, no test block:
bias 0.012, median train R2 0.73, about 12 s per fit. Larger budgets were not pursued: R2 0.73 on 130 rows is
already memorising. Pooled arms (B_ALL/B_COV/F_COV), the ridge fallback and routing baselines are read from E198's
predictions. Runner `E198_VARIANT=e199 python3 experiments/e198_cs_eval.py`; jobs `experiments/e199_cluster.jobs`
(one per arm and seed, 6 workers), waiter `experiments/e199_score.sh`. Outputs
`results/reports/crypto/e199_tabnet_percoin/`. Comparisons: TN_F - TN_B, and each against ridge B_COV / F_COV.

**Launch (2026-11-20 ~08:10, from hamilton).** `loginctl enable-linger danielco` on tolstoi and dostoievski (user
approved), then `python3 -m src.cluster run --jobs experiments/e199_cluster.jobs`: four 4-worker lanes, two per box
(TN_B / TN_F seeds 0+2 on dostoievski, seed 1 on tolstoi), waiter `nmr-e199_score` on hamilton. A first launch was
stopped within minutes, before any output: coins with 30-31 training rows are under batch 32, which the new wrapper
guard rejects; the runner now caps batch at the coin's row count (identical in both arms). ETA ~7 h.

**Incident (2026-11-20 ~11:40).** TN_B seed 0 crashed at about task 3,000 of 4,068 (2.8 h in): one coin's fit went
NaN inside TabNet's sparsemax mask (`RuntimeError: index -1 is out of bounds for dimension 1 with size 22`), and the
exception killed the whole arm-seed, with nothing saved. The runner now catches a failed per-coin fit, gives that
coin-block the ridge B_COV fallback (the same rule as < 30 training rows, identical in both arms), logs it and
records `failed=True`; non-finite predictions are also sent to the fallback (6f0685d). Lane `e199_tnb_a` (TN_B seeds
0 and 2) was stopped and is relaunched with the guard as soon as a slot frees (the cluster was at its 30% share).
The other three lanes started before the guard; if one crashes it is relaunched the same way. Report the share of
failed fits per arm with the findings.

**Findings (2026-11-20).** Every per-coin TabNet fit now trains (batch 32). Fits that diverged and fell back to ridge B_COV: TN_B 0.03% / 0.07% of rows in seeds 0 / 2 (the guard was added after seed 1 ran), TN_F 0%. Report: `results/reports/crypto/e199_tabnet_percoin/E199_cs_report.pdf`. Paired Newey-West lag 20, dev n=1430 / test n=130:

| comparison | ΔCORR dev (t) | Δpayout dev (t) | ΔCORR test (t) | Δpayout test (t) |
|---|---|---|---|---|
| TN_F − TN_B (CS columns, per-coin) | −0.0074 (−3.46) | −0.0049 (−1.25) | +0.0079 (+1.14) | +0.0033 (+0.51) |
| TN_B − ridge B_COV | −0.0454 (−6.63) | −0.0234 (−2.65) | −0.0371 (−2.83) | +0.0073 (+0.78) |
| TN_F − ridge F_COV | −0.0453 (−6.53) | −0.0254 (−3.65) | −0.0302 (−3.10) | +0.0150 (+1.75) |

Verdict: REJECT. The CS columns hurt per-coin TabNet's dev CORR, as they hurt the pooled ridge in E198. Per-coin TabNet (CORR ~0.07-0.09) sits far below pooled ridge (~0.12). Its test MMC is higher (TN_F vs ridge F_COV, t +2.25), but that is one 130-date block after dev losses, so it is not a lead.

### E200: E198 with an equal-budget hyperparameter search (registered 2026-11-20)

User request: "Replace this experiment with the same version but hyperparamenter searched. Formulate it and run now."
E198 (fixed E131 configs, tuned on v2.0's 22 features) stays as the record; E200 is its searched replacement.

**Hypothesis.** Same as E198: adding the 15 `feature_cs_*` columns (F_COV) beats plain v2.0 on the same covered rows
(B_COV), now with each arm's hyperparameters searched rather than inherited from v2.0.

**Design.** Everything E198's (crypto/v2.0-cs, arms B_ALL / B_COV / F_COV, E131 blocks 0-10 + test block 11, seeds
ridge 0 / lgb_ktz and mlp_ktz 0-2, R_SLOT routing into each model's B_ALL, NW lag 20), except the config. Equal-budget
search, E133's `--tune` rule: per block and per seed, every candidate is fitted on the train window minus its last
62 dates, scored on the last 40 train dates (after a 22-date purge) by mean per-date Spearman with
target_binned_return_20 on that arm's own rows and columns, and the winner is refitted on all 130 train dates.
Selection reads train dates only, so the test block never enters the search. One candidate list per model, identical
across arms, built on top of the E131 fixed config:
- ridge `alpha` in {0.1, 1, 10, 65.8, 316, 1000, 3162, 1e4, 3.16e4, 1e5} (10). The first smoke run with a 1e4 top
  picked the edge in most blocks, so the grid was widened to 1e5 before launch. 1e5 is still picked often: for rank
  scoring, ridge as alpha -> infinity tends to the X'y direction, so the top of the grid is near that limit.
- lgb_ktz `num_leaves` {4, 8, 16} x `colsample_bytree` {0.1, 0.17, 0.3} x `min_child_samples` {100, 500} (18).
- mlp_ktz `alpha` {0.01, 1, 100} x `learning_rate_init` {0.01, 0.0501} x `activation` {identity, relu} (12).
Per-coin TabNet is not searched here (E199 holds the fixed-config rerun). Picks are written per arm and seed to
`preds/<model>/<arm>_s<seed>_picks.csv`. Primary comparison: F_COV - B_COV per model, dev and test separately.

**Launch (2026-11-20 ~09:45, from hamilton).** Runner `E198_VARIANT=e200 python3 experiments/e198_cs_eval.py`.
`python3 -m src.cluster run --jobs experiments/e200_cluster.jobs`: lgb_ktz one lane per arm (B_ALL 2 cores on hopper,
B_COV on dostoievski, F_COV on tolstoi, 1 core each, all 3 seeds in sequence); `nmr-e200_score` on hamilton fits ridge
(already fitted at registration, about 10 s per arm) and mlp_ktz, waits for the 9 lgb files, scores, and prints
`results/reports/crypto/e200_cs_tuned/E200_cs_report.pdf`. ETA about 3-4 h (the cluster is near its 30% share).

**Findings (2026-11-20). NO IMPROVEMENT -- the CS columns still add nothing once every arm is tuned.** F_COV - B_COV,
routed, NW lag 20 (dev n=1430 dates, test n=130):

| model | dev dCORR (t) | dev dpayout (t) | test dCORR (t) | test dMMC (t) | test dpayout (t) |
|---|---|---|---|---|---|
| ridge | +0.0033 (+1.09) | +0.0041 (+1.01) | +0.0081 (+1.28) | -0.0027 (-0.61) | -0.0017 (-0.35) |
| lgb_ktz | -0.0018 (-0.65) | -0.0007 (-0.14) | -0.0013 (-0.31) | -0.0045 (-1.66) | -0.0046 (-1.47) |
| mlp_ktz | -0.0151 (-2.28) | -0.0287 (-2.67) | +0.0041 (+0.42) | -0.0019 (-0.24) | -0.0016 (-0.19) |

- Tuning removes E198's significant losses (ridge dev CORR t -2.20, lgb test MMC t -4.16) but turns up no gain:
  ridge moves to small positive CORR, still under |t| 2 and with negative test MMC/payout; lgb is null.
- **The search lowers almost every level.** Dev payout tuned vs fixed: ridge about equal, lgb -0.002 to -0.004 per arm,
  mlp collapses (B_COV 0.088 -> 0.024, F_COV 0.092 -> -0.004). The 40-date inner validation is too noisy to beat
  the E131 configs, which were picked on far more data.
- **mlp_ktz is unreadable**, as in E137: the search often picks alpha 100 (14 of 33 F_COV picks are alpha 100 identity),
  where the MLP collapses. Its dev F_COV loss (t -2.67) measures collapse, not the features.
- Verdict for the E198 question is unchanged: do not add crypto/v2.0-cs. lgb picks lean to num_leaves 4 /
  colsample 0.1 on both arms; ridge picks span 0.1..1e5 across blocks (unstable).

**E193-E197 findings (2026-11-20, all 5 x 38 arms; tables `results/reports/crypto/e193_e197_tsfm_tables.pdf`,
per-model `results/reports/crypto/<id>_<model>norm/table_by_payout.csv`).** Routed dev payout (11 blocks,
468 periods), B_COV (v2.0 on the same rows) 0.1004, B_ALL 0.1021. Best state arm per model -- every one
is N_TS_MM, as in E189 (Chronos-2-Synth 0.0609):
Timestral 0.0683 (sharpe_p 0.77, test 0.0516), TabPFN-TS 0.0667 (0.82, test 0.0277), Toto 2.0 0.0626
(0.71, test 0.0251), t0 0.0586 (0.66, test 0.0352), TempoPFN 0.0574 (0.65, test 0.0464).
No state arm of any model comes within 0.03 payout of v2.0 on its own rows; RAW states run 0.031
(TempoPFN) to 0.050 (TabPFN-TS, Toto). GRP (within size group) is the worst axis for every model.
The ranking between models is not tested (no paired Newey-West tests in this pass); the 0.057-0.068
spread may be noise. Reading: swapping the foundation model does not change E189's answer -- frozen
TSFM states alone are far below the 22 shipped features. Marginal value IN ADDITION to v2.0
(E190-style merge) was not run.

### E201: E192 with an independent ridge per coin, own alpha per coin (registered 2026-11-22)

**Request (user):** "create a new experiment from E192 but instead of sharing alpha_b across coins estimate an independent ridge model per coin on the following dataset: own hidden state; own hidden state + v2.0; own hidden state + average of other assets hidden states; own hidden state + average of other assets hidden states + v2.0; concatenation of all hidden states for a given time step across all the assets in a flat way; concatenation ... + v2.0."

**Design:** E192 unchanged (frozen Chronos-2-Synth RAW cov-mode states, cache `/scratch/danielco/c2syn`, panel `data/crypto/v2.0-asof-e189`, E131 11 dev blocks + test block 2026-02-20..2026-08-20, per-coin dual-form ridge with intercept, R_SLOT routing vs B_ALL, E189 scorer) except the penalty:
- Each coin picks its own alpha per block: fit on the coin's rows in train[:-62], validate on its rows in the last 40 train dates (E189's purged split). One coin has one row per date, so there is no per-date ranking to score; the pick minimises the coin's validation MSE over a 49-point half-decade grid 1e-9..1e15 (the range E192's shared pick could reach with boundary expansion). Refit on all the coin's block training rows at that alpha.
- Arms: `A_OWN` own state; `AV_OWN` own + 22 v2.0; `A_LOO` own | leave-one-out mean of the other covered coins; `AV_LOO` + v2.0; `A_CAT` every coin's state in fixed slots (date Gram, as E192); `AV_CAT` + v2.0.
- Coins with >= 30 training rows but < 20 inner-fit or < 10 validation rows use the block's median coin alpha; coins with < 30 rows take the pooled twin's prediction, as in E192. Per-block alpha quantiles and edge-hit share are logged.
- `--check`: for 3 coins x 4 non-CAT arms the per-coin pick equals a brute-force sklearn `Ridge` sweep on explicit features (passed). Many coins choose the top of the grid, i.e. the coin's own training mean; the share is reported.
- Comparisons: each arm vs its E192 shared-alpha twin (`C_*`/`CV_*`), LOO and CAT vs OWN, +v2.0 vs state only, AV_* vs B_COV, vs pooled RAW/V_RAW; paired Newey-West lag 20, Holm over all pairs per window and metric.

**Launch (cluster manager):** `python3 -m src.cluster run --jobs experiments/e201_c2percoin_alpha.jobs` -> `python3 experiments/e201_c2percoin_alpha.py --fit --workers 6` (per-block checkpoints in `preds/<arm>.blocks/`, resumes on relaunch); then `--score`, `--report`, PDF via `python3 -m src.analysis.html_pdf`. Outputs `results/reports/crypto/e201_c2percoin_alpha/`; log `logs/cluster/e201_c2percoin_alpha.out`.

### E202: top-four TSFM states on the 60-day label, residualised, blended with live slots (registered 2026-11-22)

**Request (user):** "take those four models, regenerate the hidden states embedding features but using targe60d, then tune the downstream pridction model using target60d and then evaluate the same table but on targe20d. then redidualize the eight models, four trained on tareg20 and four trained on target60, then build test ensembles of those models and those models and the current models we have in place such as cu_sa_ia_bo_ma_r_01, cu_sa_ia_bo_ma_01, and cu_sa_ia_bo_02". "Those four" = the payout top four of E193-E197: N_TS_MM of Timestral (E197), TabPFN-TS (E194), Toto 2.0 (E193), t0 (E196).

**Design:**
- *60-day states.* `src/data_mani/tsfm_states.py --target 60`: same adapters, panel `data/crypto/v2.0-asof-e189`, 256-step context, but the label series fed to the model is `target_binned_return_60` and the wall is D-91 days. In the frozen panel the last resolved 60-day label (2026-08-24) is 56 days older than the last 20-day label (2026-10-19), so the 60-day wall = E189's 35 days + 56. Row set = the same >=32-resolved-labels rule on the 60-day label: 417,199 rows <= 2026-08-20 (vs 449,471). The rule is computed by an independent `expected_covered()` and asserted per part; on the 20-day label it reproduces E189's Chronos-2 row set exactly. Output `/scratch/danielco/tsfmstates60/<model>/`.
- *60-day ridge.* `experiments/e202_tsfm_t60.py`: E189's N_TS_MM transform and repo `ridge`, alpha tuned (E189 grid + boundary expansion, mean per-date Spearman) and fitted on the 60-day label. Outer train window = 130 dates ending 78 dates (E131's 22 + 56) before each block's first date, so no training label resolves after the block starts (block 0 is clipped at the panel start); inner split E189's (last 40 train dates, 22-date purge). Rows with no resolved 60-day label are dropped from fits. Scored exactly as E193 (routed into B_ALL's ranks on covered rows, payable 20-day label, E131 scorer). Reference `B_ALL60`: v2.0 22 features, same 60-day schedule.
- *Table.* Same columns as the E193-E197 top-four table (payout, sharpe.p, maxdd.p, corr, mmc, test payout/sharpe.p), 60-day arms next to their 20-day twins, B_ALL/B_COV/B_ALL60 as references; paired Newey-West (lag 20) 60d vs 20d per model.
- *Residualise.* The eight routed predictions (4 x 20d, 4 x 60d) exported as runs `tsfm_states/e202_<model>_t{20,60}` and residualised with the E107 transform (`src.prediction.meta_resid`: half the projection on the meta model lagged >= 5 days, frozen `historical_meta_models.parquet`) as `..._r`; 16 runs in `results/{forecast,per_period,metrics}/crypto/tsfm_states/`, scored with `per_period_table` (the scorer the slots' series use).
- *Ensembles.* Prediction-level per-date rank averages over the 16 new runs + `CU_SA_IA_BO_MA_R_01`, `CU_SA_IA_BO_MA_01`, `CU_SA_IA_BO_02` (forecasts of their registry backtest runs) on the common (symbol, date) rows: every equal-weight 2- and 3-member blend, plus slot x new model at weights 0.8/0.2 and 0.5/0.5. Picked on common dates < 2026-02-20, read on 2026-02-20..2026-05-13 (the slots' series end there); paired Newey-West vs the slot alone. The slots' own backtests were built with data through 2026-05, so the read window is out of sample for the blend choice only.
- Repo-standard report: `python3 -m src.analysis.report_pdf --model tsfm_states --runs 'e202_*'` (deployed slots included by default), plus the study PDF from the runner.

**Launch:** extraction as systemd user units on one GPU per box (deepfive card 1, deepsix card 0; the other three cards are held by another user's jobs), TabPFN-TS also on CPU claim workers through `src.cluster`; then `python3 experiments/e202_tsfm_t60.py --setup / --fit / --score / --export / --ensembles / --report`, chained by `experiments/e202_orchestrate.py` (unit `nmr-e202-orchestrate`). Logs `logs/tsfm/e202_*.log`.
