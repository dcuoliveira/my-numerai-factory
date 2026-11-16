# Pre-numbering legacy named arms

- **Scope:** forecast runs under `results/forecast/crypto/<model>/` whose names match NO
  numbered study (`edb.study_of` returns `None`). Families: `w6r6_*`, `xae_*`, `ens_*`,
  `eoe_*`, `ohl_*`, `t60_*`, `saci*`, the `lstm` investigation arms (`x3_*`, `a4_*`,
  `a2_*`, `a5_*`, `b2_*`, `d3win_*`, `ax1_*`, `lstm_*`, `embexp_*`), the `torch_scratch`
  / `torch_mlp` walk-forward replicas, and legacy `lgb` blocks (`e86xb_*`, `e46_*`,
  `e16_*`, `g3_*`, `z_vol*`, `x_lag*`, `e89book_*`).
- **Footprint:** 564 families, **784 forecast files, 3002.9 MB (2.93 GB)**
  (from `scratchpad/legacy_families.json`, the retention manifest for this batch).
- **Provenance:** FINDINGS-BACKED for the ensemble/weighting/shrinkage arms (they are
  the forecast artifacts behind EXPERIMENTS.md §E23/§E25/§E27); the LSTM verdict is
  CLAUDE.md + `results/reports/crypto/lstm_investigation.pdf` + `lstm_board.{html,pdf}`;
  the torch walk-forward is MEMORY-BACKED; the feature/encoding sweeps are NUMBERS-ONLY.
- **Sources read now:** `experiments/EXPERIMENTS.md` §E23/§E25/§E27 (lines ~272–457);
  `src/analysis/retention.py`; `CLAUDE.md` (The submitted models; The LSTM cannot run
  live); memory: `walkforward-noreg-mlp`, `e33-fenc-identity-calmar-fragile`,
  `torch-mlp-identity-support`, `e104-daybatch-vs-rowbatch`, `e105-lr-schedulers-day3`;
  metrics JSONs listed per theme below.

## How these files are protected, and what this KB unlocks

`src/analysis/retention.py` never deletes anything hand-listed. A forecast is **KEPT**
if any of these rules fire (first match wins), and is only ever a delete *candidate*
when it is unkept AND positively opted in via `--families <prefix>` on the command line.
The legacy arms are all `study = None`, so `--families <prefix>` is the only switch that
can reclaim any of them — but it reaches **only the unprotected (`other`) arms**, because
`candidates()` skips everything already in `keep` before it ever looks at `--families`. An
arm kept by K1, K2, K3 or K5 stays kept whatever is passed. This document records each
theme's findings and scored target so that deleting the `other` forecasts loses nothing.

Retention classes, with the estimated MB each holds in this batch
(uniform-file-size-within-family estimate; counts are exact file counts from the
manifest):

| class | rule (retention.py) | files | ~MB | this KB can lift it? |
|---|---|---:|---:|---|
| **K1** | deployed slot backtest or a member/leg of one | **51** | 164.8 | **NO — never delete** |
| K2 (`cited`) | run name cited in tracked code/docs (`src/`, `experiments/`, `.claude/`, CLAUDE.md) | 72 | 316.7 | No (still referenced in code) |
| K5 (`report`) | named by a published report under `results/reports/` | 24 | 116.7 | No (report must stay re-renderable) |
| K3 (`record`) | metrics config has NO `target`, so the forecast is the only record of the scored label | 268 | 851.8 | **NO — K3 is a KEEP rule; `--families` cannot reach it** |
| (`other`) | no keep rule fires; unprotected | 369 | 1552.9 | **YES — opt in with `--families`** |

**Reclaimable via `retention --families`: the `other` (unprotected) class ONLY —
369 files, ≈ 1552.9 MB apparent. On the shared quota that frees roughly 2.4× less,
≈ 650 MB, because the forecast CSVs compress on disk.** Kept regardless: K1 + K2 + K3 +
K5 = 415 files, ≈ 1450.0 MB apparent — and that total **includes all 268 `record` files**.
In `retention.py` the `record` rule is a KEEP rule: `build()` puts every target-silent
forecast into `keep`, and `candidates()` skips anything in `keep`, so a `record` forecast
is never a delete candidate even under `--families`. `retention.py` does not read this KB,
and writing this markdown changes nothing about what it keeps — so the firebreak is not
"removed." K3 exists because deleting a forecast used to erase the only record of which
label (20-day vs 60-day) the run was scored on; this document copies those targets into
durable prose per theme, but that does not lift the protection. (The quote "writing the
conclusion is what unlocks the disk" in `retention.py`'s header is about numbered studies
flagged `forecasts="reclaim"` in `STUDIES`, not about these `study = None` legacy arms,
which only `--families` can reach.) Only `per_period/` + `metrics/` siblings are required
to survive a delete, and they are out of scope here.

> ⚠️ **K1 must not be deleted.** These are the live CURUPIRA / SACI / IA / MA book
> backtests and the legs the slot blends rank-average. `member_closure` follows
> `config["members"]` transitively (e.g. `CURUPIRA_08` sits inside `CURUPIRA_SACI_03`
> inside `CU_SA_IA_BO_01`), so a deep member is as protected as a top slot. The K1 run
> names in this batch, grouped by what they feed:
> - **CURUPIRA ridge_glmnet 20-day:** `w6r6_glmnet_reg_tenc` (CURUPIRA_01's series, per
>   CLAUDE.md), `w6r6_glmnet_reg_onehot_rollsel`, `w6r6_glmnet_reg_allfeatpp`,
>   `w6r6_alone_{tenc,onehot,interact}`, `w6r6_grp_price_glmnet`, `w6r6_blend_quintet2`.
> - **CURUPIRA_07 & members:** `xae_c07_base` (per-period rank average of
>   `xae_c01_base`, `xae_c04_base`, `xae_c05_base`, `xae_c05_60_base`, 0.25 each),
>   `xae_m_{tenc,onehot,interact,price}_base`, `xae_t60_{tenc,interact,onehot,price}_base`.
> - **60-day CURUPIRA (t60):** `t60_c01_tenc`, `t60_c04_allfeatpp`,
>   `t60_m_{tenc,interact,onehot,price}`.
> - **Other screened/blend legs:** `eoe_c01_*` (2), `ens_scr_*` (2).
> - **ridge tree:** `w6r6_fixed_reg` (CURUPIRA_02 backtest = `submit.SIBLING`, the
>   fallback rung for every slot), `w6r6_grp_size_fixed`, `xae_m_size_base`,
>   `xae_t60_size_base`, `t60_m_size`.
> - **lgb tree:** `saci01_c03_c0560`, `saci04_c05_c08`, `x_lag3_*`, `fe_lgb_reg_interact`,
>   `fe_lgb_reg_raw`, `ohc_lgb_reg_allfeat_m50`, `x3_int_e60tenc_c05`, `boto01_slot_mlpfenc`.
> - **Neural / other components:** `mlp/iu_mlp_tenc_pin`, `mlp/iu_mlp_interact_pin`,
>   `mlp/a2_mlp_fenccausal` (IA components), `ktz_farm/iu_ia_01_deployable`,
>   `rf_ktz/e_cu02_rf6733`, `tabpfn/tu_cu_sa_ia_bo_ca_01`, `meta_resid/e107px*` (E107
>   residualised slot blends).
>
> Verify any single run before acting: `python3 -m src.analysis.retention --why <model>/<run>`.

---

## Theme 1 — Deployed-slot backtests (`w6r6_*`, `xae_*`, `t60_*`, `saci*`, `eoe_*`, `ens_scr_*`)

**What it was.** The walk-forward backtests behind the live CURUPIRA/SACI book and the
legs the slot blends rank-average. `w6r6` = "window 6, refit 6" cadence; `xae` relates to
CURUPIRA_07; `t60`/`*_60` fit the 60-day label; `saci*` are the SACI slot blends.

**Main question.** These are not an experiment — they are the production record. The
only "question" is reproducibility: does each deployed slot's live spec reproduce its
stored backtest series? (`src/analysis/slots.py` self-check: 34/37 reproduce
`backtest.payout` to 1e-6.)

**Trained parameters** (from `results/metrics/crypto/ridge_glmnet/w6r6_glmnet_reg_tenc.json`,
CURUPIRA_01's series): `model ridge_glmnet`, `data_version crypto/v2.0-tenc`,
`cv_scheme rolling`, `train_window 130`, `refit_every 130`, `n_splits 5`, `val_frac 0.3`,
`embargo 20`, `n_iter 50`, `n_features 23`, `proxy_alpha 1.0`, `seed 42`,
`task regression`, `metric corr`. Blend legs carry no fitted params — e.g.
`xae_c07_base` is `{"combination":"per-period rank average","weights":[0.25,0.25,0.25,0.25]}`.

**What was found.** FINDINGS-BACKED (CLAUDE.md "The submitted models"). This is the
production backbone; the slots are what the account actually submits. **All runs here with
a K1 verdict are PROTECTED and must not be deleted.** Any `w6r6_*`/`xae_*`/`t60_*` run in
this theme that is NOT a slot member falls through to K3 (target firebreak) or K2 (cited)
and is documented under the sweep themes below.

## Theme 2 — The LSTM investigation (`lstm/*`: `x3_*`, `a4_*`, `a2_*`, `a5_*`, `b2_*`, `d3win_*`, `ax1_*`, `lstm_*`, `embexp_*`)

- **Footprint:** 152 families, 187 files, **878.6 MB** — the single largest theme.
  Retention split: k1 0, cited 6, report 5, record 38, other 138. **Reclaimable
  (`other` only) = 138 files; the 38 `record` files are KEPT by K3, and 0 files are K1.**

**What it was.** The torch LSTM research programme on crypto. Families map to probes:
`x3_*` = architecture/HP sweep (hidden size, num layers, dropout, embed dim, batch size,
LR, epochs, layer-norm/clip ablations, seeds); `a4_*` = loss-function bake-off
(`corr`/`dcorr`/`dsharpe`/`ce`/`bins` at various batch sizes and seeds); `a2_*` = feature
encodings (fenccausal/fencglobal/tenc/interact/spread); `a5_*` = sequence-length /
snapshot; `b2_*` = tenc/cumlag; `d3win_*` = lookback window `L14`/`L21`; `ax1_hpsearch` =
HP search; `lstm_mixture_*`, `lstm_cv_*`, `lstm_loss_*` = CV-scheme and mixture-loss runs;
`embexp_scout` = embedding scout.

**Main question.** Can a sequence model (LSTM) beat the tabular book, and can it be
deployed?

**Trained parameters** (representative, from
`results/metrics/crypto/lstm/x3_nv1_emb0_h240.json`): `model lstm`, `data crypto/v2.0`,
`seq_length 21`, `hidden 240`, `num_layers 1`, `dropout 0.3`, `layer_norm true`,
`loss corr`, `lr 0.001`, `batch_size 32`, `optimizer adam`, `max_grad_norm 1.0`,
`n_classes 5`, `n_features 22`, `embed_dim 0`, `overlapping true`, `train_window 130`,
`refit_every 130`, `embargo 20`, `seeds [11,22]`, `target target_binned_return_20`
(CORR 0.12938, MMC 0.05246, n 513). A later config
(`lstm/lstm_mixture_1y_L21_seeds100.json`): `loss mixture`, `bmc_weight 1.0`,
`corr_weight 0.1`, `embed_dim 50`, `hidden 60`, `epochs 8`, `max_trials 100`,
`cv_scheme expanding`, `refit_every 260` (CORR 0.09749, MMC 0.04205, n 510).

**What was found.** FINDINGS-BACKED (CLAUDE.md + `lstm_investigation.pdf`). The MLP beats
the LSTM outright on identical raw-22 data — **CORR 0.15812 (MLP) vs 0.10495 (LSTM)** — so
`IA_01` is a feedforward MLP and the LSTM is deliberately absent from
`src.prediction.models.MODELS`. (CLAUDE.md caveats both: the `mlp` family has 4 runs in a
book of 500+ and its 0.16172 CORR is one unreplicated config.) **The LSTM also cannot run
live at all:** no torch live fit/predict path exists, and `live_history.parquet` must
reach `seq_length` dates (held 7 as of 2026-08-20). From EXPERIMENTS.md §E23, a 1/N blend
that includes the LSTM (`div3`, `all5`) is not better than the tabular-only blend and
*cannot be submitted* anyway. **None of these 187 files is a deployed-slot member (k1=0),
but 38 are kept by K3 (`record`): so apart from the 6 cited + 5 report + 38 record files,
the 138 `other` LSTM files are reclaimable.**

## Theme 3 — Dead ensemble / weighting / shrinkage screens (`ens_*`, `ens_ae_*`, `ens_scr_*`, `eoe_*`, `w6r6_swp_*`, `ridge/ens_w0.*`, `e32blends/*`, `g3_*`)

- **Footprint:** ensemble/blend screens 45 files / 124.6 MB (k1 4, record 40, cited 1);
  `w6r6_swp_*` shrinkage/weighting 47 files / 171.3 MB (all 47 `record`, k1 0).

**What it was.** The forecast artifacts behind the combiner studies. `ens_*` = fixed 1/N
rank averages over member pools; `w6r6_swp_*` = the walk-forward weighting/selection
schemes — `shrunk` (shrunk_tangency), `invvar` (inverse_variance), `equal`, `subset`
(complete_subset) — swept over update cadence, window, shrinkage and subset size.

**Main question.** Can any fitted combiner or weighting scheme beat a plain equal-weight
(1/N) rank average of the member slots?

**Trained parameters** (from `results/metrics/crypto/ridge_glmnet/w6r6_swp_shrunk_c65w260_l05.json`):
`scheme shrunk_tangency`, members `w6r6_alone_{onehot,tenc,interact}` +
`ridge/w6r6_grp_size_fixed` + `w6r6_grp_price_glmnet`, `update_every 65`, `lag 20`,
`min_history 60`, `window 260`, `objective sharpe`, `shrinkage 0.5`, `subset_size 3`,
`keep_fraction 0.5`, `ridge_alpha 1.0`, `seed 42` (CORR 0.12390, MMC 0.07048, n 510).

**What was found.** FINDINGS-BACKED (EXPERIMENTS.md §E23, §E25, §E27). **Nothing beats
1/N.** §E25: all **14 of 14** weighting/selection deltas are negative; the strongest
single signal is that a fitted combiner (`stacking_nnls`) is WORSE than equal weights at
**t −2.01**; shrunk_tangency −0.001184 (t −0.95), inverse_variance −0.002327 (t −1.33),
complete_subset −0.003671 (t −1.88). §E23: the diverse 1/N blend `all5` − best member
`ridge` = +0.003669 payout but only **t +0.47** (not established); adding the orthogonal
LSTM contributes nothing (`div3` − `tab2` = −0.000943, t −0.19). §E27: the FARM family
reproduces the paper on squared-error loss but no arm beats 1/N on `numerai_corr` at
|t|≥2. **Operational corollary (stated in §E25): the hand-set equal weights in the
deployed CURUPIRA slot blends are correct and leave nothing on the table.** The 4 k1 files
in `ens_scr_*`/`eoe_*` ARE slot legs (protected); every one of the ~87 remaining screen
outputs carries no `target` in its config, so K3 KEEPS it as `record` (0 fall through to
`other`). **None of this theme is reclaimable via `--families`** — writing the null here
preserves the knowledge but does not unprotect the files, since `retention.py` never reads
this KB.

## Theme 4 — Feature / encoding / cadence sweeps (ridge_glmnet `w6r6_rawloo_*`, `w6r6_*lag*`, `w6r6_*vol*cum*`; lgb `ohl_*`, `w6r6_lgb_grp_*`)

- **Footprint:** `w6r6_rawloo_*` 22 files / 89.7 MB (all record); glmnet lag/cumlag 30
  files / 134.5 MB (k1 2, cited 9, other 19); glmnet vol/cum 20 files / 74.8 MB (all
  other); `ohl_*` lgb 20 files / 89.6 MB (all other); `w6r6_lgb_grp_*` 15 files / 45.0 MB.

**What it was.** Ablation sweeps on the deployed cadence. `w6r6_rawloo_*` = leave-one-out
of each raw feature (market_cap_avg_20d/60d, market_cap_ewa_20d/60d, RSI 20/60,
sharpe_ratio 20/60, …); `*lag#`/`*cumlag#`/`*volcum#` = lag/cumulative-lag feature-block
sweeps; `ohl_*` = lgb one-hot/tenc/rollsel feature sets at `min_child_samples`
(`m20`…`m99`); `w6r6_lgb_grp_*` = grouped-feature (oscillator/volatility) reg-vs-clf.

**Main question.** Which raw features and encoding/cadence choices matter for the deployed
ridge/lgb, and do any beat the shipped config?

**Trained parameters** (from
`results/metrics/crypto/lgb/ohl_lgb_reg_tenc_onehot_rollsel_m80.json`): `model lgb`,
`data crypto/v2.0-tenc-onehot-rollsel`, `n_features 787`, `fixed_params`
`{colsample_bytree 0.070213, learning_rate 0.002572, max_depth 8, min_child_samples 80,
n_estimators 4327, num_leaves 255, reg_lambda 1.757511, subsample 0.816153,
subsample_freq 1}`, `train_window 130`, `refit_every 130`, `n_splits 2`, `n_iter 50`,
`target target_binned_return_20` (CORR 0.13599, MMC 0.06648, n 510).

**What was found.** NUMBERS-ONLY — no dedicated EXPERIMENTS.md section for these legacy
names. From the metrics the best `ohl` arm lands CORR ≈ 0.136 / MMC ≈ 0.066 over n=510,
in line with the deployed book rather than beating it; the `w6r6_rawloo_*` LOO arms exist
only to attribute each raw feature's marginal contribution — and all 22 of them are
`record` (no `target` in config), so K3 KEEPS them. Two `*lag*` files are K1 (slot
members) and 9 are K2 (cited); of the rest, only the `other` files are reclaimable — every
`record` file (the rawloo LOO set among them) stays kept. No written claim survives for
these beyond the configs themselves.

## Theme 5 — Legacy lgb blocks (`e86xb_*`, `e46_*`, `e16_*`, `e89book_*`, `z_vol*`, `g3_*`, `x3_int_*`) and lgb_ktz `e86ens_*`

- **Footprint:** `e86xb_*` vol/size blocks 46 files / 124.8 MB (all other); misc lgb
  legacy 25 files / 79.1 MB; `lgb_ktz/e86ens_*` 6 files / 21.0 MB.

**What it was.** `e86xb_*` = lgb blends crossing volatility/size blocks (`volC/volL/volF/
volR/volM`, `sizeR/sizeF`) and a basket of `bl_v*` blends, each at window `w10`/`w20`;
`e46_*` = raw lgb depth/LR/colsample grid; `e16_*`, `e89book_*`, `z_vol*`, `g3_*`,
`x3_int_*` = assorted legacy lgb probes; `lgb_ktz/e86ens_*` = KTZ-prior ensemble cells.

**Main question.** Do volatility/size-conditioned lgb blends or the KTZ-prior ensembles
improve on the single lgb book members?

**Trained parameters** (from `results/metrics/crypto/lgb/e86xb_bl_v3sr_w10.json`):
per-period rank average, members `lgb/e47_cusaiabo02` + `lgb_ktz/e86ens_v3_sr`,
`weights [0.9, 0.1]`, `target target_binned_return_20` (CORR 0.16558, MMC 0.08774,
n 442).

**What was found.** NUMBERS-ONLY. The best `e86xb` blend reaches CORR ≈ 0.166 / MMC ≈
0.088 (n 442) — the 0.9/0.1 weighting barely moves off the dominant member, consistent
with the combiner null in Theme 3. No dedicated written finding exists for these legacy
names. Only the `other` files are reclaimable via `--families` (the `e86xb_`/`e86ens_`
blends and most of the misc lgb probes); the `record` files here — misc lgb runs whose
config carries no `target` — are KEPT by K3, and any K1 slot member (e.g. `x3_int_*`) is
protected.

## Theme 6 — Walk-forward torch MLP replicas (`torch_scratch/*`, `torch_mlp/e33repl_*`, `ridge/wfdiag_*`, `torch_mlp_loss/*`)

- **Footprint:** torch_scratch 15 files / 83.1 MB (cited 5, report 2, other 8);
  torch_mlp 3 files / 9.0 MB (all other); torch_mlp_loss 2 / 5.6 MB (record).

**What it was.** The from-scratch / sklearn-replica torch MLP runs used to (a) reproduce
the fragile `e33_fenc` identity MLP, (b) characterise the unregularised 500-epoch net at
the procedure level, and (c) test day-vs-row batching and LR schedules. Tags:
`torchscratch_{corr,mse}_500ep_wf_s42/s43`, `torchmlp_noreg_500ep_wf_s42`,
`e33repl_torch_fenc_*`, `wfdiag_ridge_full_a1`.

**Main question.** Is the MLP's apparent edge (and the `e33_fenc` calmar 0.859 outlier)
real, and how should NN training be reported here?

**Trained parameters** (from
`results/metrics/crypto/torch_scratch/torchscratch_corr_500ep_wf_s42_ep500.json`):
`model torch_scratch`, `activation identity`, `hidden "64,32,16"`, `alpha 0.0`,
`batch days` (`days_per_batch 3`, `batch_size 1024`, `penalty_denominator 1024.0`),
`loss corr`, `lr 0.001`, `max_epochs 500`, `no_stop true`, `init sklearn`,
`data crypto/v2.0-fenccausal` (44 features), `n_refits 12`, `device cuda`
(RTX 2070, deepfive) (CORR 0.11454, MMC 0.07013, n 1531).

**What was found.** MEMORY-BACKED. (`walkforward-noreg-mlp`) the unregularised relu MLP
overfits every fold and loses to a same-window ridge **12/12** (median gap −0.057); the
from-scratch identity `-corr` net beats the ridge on CORR in 11/12 folds (**+0.0036,
t +2.9**) but the payout gain is not significant (t +1.0), and only ~¼ of the CORR edge
reaches MMC (+0.0054 → +0.0014, level with the ridge). (`e33-fenc-identity-calmar-fragile`
+ `torch-mlp-identity-support`) the `e33_fenc` identity calmar **0.859 is a float
artifact** of an over-regularized linear map; the faithful torch replica scores calmar
0.130 (one seed diverges, sign-flipped), so it must not be deployed for its calmar.
(`e104`/`e105`) day-vs-row batching is a tight null (t +0.09); batch SIZE drives collapse
via step count, a single-seed t +4.18 was false, and annealing the LR to ~0 (OneCycleLR
+0.0356 payout, t +9.2) rescues the degenerate arm — early-stopping-in-LR-time, not a
better fit. **5 cited + 2 report files stay protected; the other 8 torch_scratch files
are reclaimable, as are the 3 `e33repl` files (memory preserves the finding).**

## Theme 7 — Misc ridge legacy (`e30price_*`, `ridge/xae_m_size_ae`, …)

- **Footprint:** ~5 files / 13.9 MB (record). `e30price_pric_size_fenc_{inte,spre}` =
  price/size feature blocks under ridge_glmnet.

**Main question / found.** THIN — no dedicated EXPERIMENTS.md section and no memory
reference for these specific legacy names; all 5 are `record` (K3) — their config carries
no `target`, so the forecast is the only record of the scored label and retention KEEPS
them. Supplying the target in this document does NOT make them reclaimable: `retention.py`
does not read this KB, and `candidates()` never offers a kept file to `--families`.
