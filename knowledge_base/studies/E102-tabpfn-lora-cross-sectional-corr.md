# E102 — LoRA fine-tune of TabPFN with a per-date cross-sectional CORR loss

> ⚠️ **PRODUCTION — DO NOT DELETE:** e102_lora backs TUPA_02 (BACKTEST_ONLY slot; leg of TU_CU_SA_IA_BO_CA_01). These forecasts are the backtest references `src.analysis.slots` resolves; retention protects them (K1). Deleting them breaks the registry self-check.

- **State:** done (1/1 arm)
- **Runner:** /home/danielco/e102_chain.sh
- **Models / datasets:** tabpfn (LoRA fine-tuned) on crypto/v2.0-fenccausal (44f); evaluated as the ordinary E19 walk-forward via model_path
- **Artifacts on disk:** 5.5 MB across 1 forecast run, plus 1 metrics + 1 hyperparams + 1 per_period (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/tabpfn/e102_lora.json

## Main question
The one fine-tune cell E44/E45/E83 never hit on all three axes at once — rank-aware, cross-sectional and prior-preserving. (a) a differentiable cross-sectional CORR loss (1 − pearson, the numerai_corr stand-in); (b) each training episode is one date's full cross-section (context = that coin-set's fenccausal history up to the T−20 leak wall, query = the date-T cross-section), so the loss is over the date's ranking, not pointwise rows; (c) only LoRA adapters (r=8) on the attention q/k/v/out projections train — 294,912 of 11.4M params, backbone frozen — vs E44/E45 full FT that corrupted the prior and E83's pointwise head that was neutral. Prior: likely neutral.

## What was tried
LoRA adapters merged into the frozen TabPFN backbone and saved, so evaluation is the ordinary E19 walk-forward loaded via model_path. Leak-safe (≤2024-06-27). One arm (e102_lora), paired NW vs E19. src/analysis/tabpfn_lora.py.

## Trained parameters
From results/metrics/crypto/tabpfn/e102_lora.json config: model tabpfn; data_version crypto/v2.0-fenccausal; target target_binned_return_20; task regression; cv_scheme rolling; inner_cv purged; train_window 130; refit_every 130; embargo 20; era_step 1; n_splits 2; val_frac 0.3; n_iter 50; search_type random; metric corr; n_features 44; seed 42; fixed_params null; model_options {device cuda, max_context 4096, n_estimators 4, model_path .../models/tabpfn/e102_lora.ckpt}. The LoRA rank/target-modules (r=8 on q/k/v/out) are described in the study idea but are properties of the saved checkpoint, not recorded in this metrics config.

## What was found
NUMBERS-ONLY. No written finding, and the comparison baseline (E19 zero-shot) was not among the artifacts read, so no paired NW delta can be quoted. From the metrics (n=510): CORR mean 0.13593 (sharpe 1.348, dd 1.086), MMC mean 0.04977 (dd 1.664), stored pnl-series mean 0.03739 (sharpe 1.278). These are the LoRA arm's absolute scores; the study's hypothesis (neutral vs E19) cannot be confirmed or refuted from the sources read because E19's numbers are not here.
