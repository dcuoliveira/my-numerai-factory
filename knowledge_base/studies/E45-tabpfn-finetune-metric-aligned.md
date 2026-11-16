# E45 — Fine-tune TabPFN with a METRIC-ALIGNED objective (CORR, MMC)

- **State:** done (2/2 arms)
- **Runner:** src/analysis/tabpfn_finetune_metric.py
- **Models / datasets:** model `tabpfn` (TabPFN v2, custom differentiable metric-aligned fine-tune); crypto/v2.0-fenccausal (44 feat), `target_binned_return_20`
- **Artifacts on disk:** 11.0 MB across 2 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/tabpfn/e45_corr.json, results/metrics/crypto/tabpfn/e45_mmc.json

## Main question
E44 fine-tuned TabPFN with the library CRPS+MSE loss and it HURT (CORR 0.145 -> 0.118); part of the damage may be objective mismatch (MSE compresses the cross-sectional spread the ranking pays for). E45 tests whether fine-tuning toward the PAYABLE metric instead recovers or beats E19. If a correct objective still loses, prior-corruption (not just the loss) is the cause; if it wins, the E44 loss was the problem.

## What was tried
Two arms, both using a CUSTOM differentiable fine-tune loop (fit_from_preprocessed context -> differentiable bar-distribution mean on the query -> loss), since tabpfn 8.5.0's public API ships only CRPS/MSE. Arm `corr`: minimise negative per-date cross-sectional correlation of query preds vs target. Arm `mmc`: minimise negative MMC (contribution over the meta model), computed only on query rows where the meta model is available (~33% coverage), meta merged from historical_meta_models on (ucid,date). Same leak-safe pre-2024-06-27 pool as E44, same save->model_path round-trip, same E19 walk-forward eval. Paired NW vs E19 (zero-shot) and vs E44 (MSE fine-tune).

## Trained parameters
From results/metrics/crypto/tabpfn/e45_corr.json (representative): model `tabpfn`, data_version crypto/v2.0-fenccausal, n_features 44, target target_binned_return_20, cv_scheme rolling, train_window 130, refit_every 130, embargo 20, n_splits 2, n_iter 50, seed 42, val_frac 0.3, inner_cv purged, bmc_reference benchmark. model_options: device cuda, max_context 4096, n_estimators 4, model_path .../models/tabpfn/e45_corr.ckpt (e45_mmc uses .../e45_mmc.ckpt). The custom-loop epochs/learning rate are not recorded in the metrics JSON.

## What was found
No written finding; from the metrics JSONs, neither metric-aligned arm recovers the E19 zero-shot 0.145 — both stay below it, so a correct objective alone did not fix fine-tuning. e45_corr: CORR mean 0.111697 (std 0.101851, sharpe 1.096665, max_drawdown 1.1138), MMC mean 0.042251, n = 510. e45_mmc: CORR mean 0.084876 (std 0.131411, sharpe 0.645883, max_drawdown 2.8411), MMC mean 0.039053, n = 510. The corr arm (0.1117) is close to E44's MSE fine-tune (0.1179) and the mmc arm is worse (0.0849) with a much higher drawdown; the direction points to prior-corruption, not just the loss, but no paired t-stat is recorded in available artifacts.
