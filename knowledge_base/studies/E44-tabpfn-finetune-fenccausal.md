# E44 — Fine-tuned TabPFN on fenccausal: can adapting the weights beat E19?

- **State:** done (1/1 arms)
- **Runner:** src/analysis/tabpfn_finetune.py
- **Models / datasets:** model `tabpfn` (TabPFN v2 transformer, library CRPS+MSE fine-tune); crypto/v2.0-fenccausal (44 feat), `target_binned_return_20`
- **Artifacts on disk:** 5.5 MB across 1 forecast run (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/tabpfn/e44_fenccausal_ft.json

## Main question
E19 ran TabPFN zero-shot (in-context, frozen weights) on crypto/v2.0-fenccausal and scored CORR 0.145 — the strongest foundation-model arm on the board (E19 number per this study's recorded idea; E19 is not in this batch). E44 FINE-TUNES the TabPFN v2 transformer on the leak-safe pre-cutoff rows, then re-runs the EXACT E19 walk-forward. E19-vs-E44 isolates fine-tuning, holding data/cadence/context fixed.

## What was tried
One arm. Fine-tune the TabPFN v2 transformer (tabpfn 8.5.0 finetuning API: get_preprocessed_datasets + fit_from_preprocessed) on the 345,496 fenccausal rows on/before the leak cutoff 2024-06-27 (min scored date minus 20-period embargo), save the adapted checkpoint, then re-run the E19 walk-forward with model_path pointed at the fine-tuned .ckpt. Scored on tabpfn/e19_fenccausal's 510 dates; paired NW vs E19. Library loss is CRPS+MSE.

## Trained parameters
From results/metrics/crypto/tabpfn/e44_fenccausal_ft.json config: model `tabpfn`, data_version crypto/v2.0-fenccausal, n_features 44, target target_binned_return_20, cv_scheme rolling, train_window 130, refit_every 130, embargo 20, n_splits 2, n_iter 50, seed 42, val_frac 0.3, inner_cv purged, bmc_reference benchmark. model_options: device cuda, max_context 4096, n_estimators 4, model_path .../models/tabpfn/e44_fenccausal_ft.ckpt. (Matches the E19 walk-forward: rolling 130/130, embargo 20, seed 42, n_estimators 4, ctx 4096.) The fine-tuning epochs/learning rate are not recorded in the metrics JSON.

## What was found
No written finding; from the metrics JSON, fine-tuning did NOT beat zero-shot — it landed BELOW the E19 zero-shot 0.145 (0.145 per the recorded idea). e44_fenccausal_ft: CORR mean 0.117883 (std 0.095672, sharpe 1.232157, max_drawdown 1.8743), MMC mean 0.056220 (sharpe 0.773889), n = 510. The study idea records the paired result as CORR 0.145 -> 0.118, t -3.02 (that t-stat is in the study idea, not independently in EXPERIMENTS.md or the metrics JSON); the 0.118 matches the JSON's 0.117883.
