# E41 — Fine-tuned TEMPO: does adapting the weights lift E40's null?

- **State:** done (4/4 arms)
- **Runner:** src/analysis/tempo_finetune.py
- **Models / datasets:** model `tempo` (TEMPO-80M, parameter-efficient fine-tune — GPT-2 frozen, only LoRA + heads + RevIN affine trained); crypto/v2.0, `target_binned_return_20`, 1 input feature
- **Artifacts on disk:** 12.2 MB across 4 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/tempo/e41_wf_mse.json, results/metrics/crypto/tempo/e41_sg_corr.json

## Main question
E40 (TEMPO zero-shot) scored CORR -0.017, null. Does fine-tuning TEMPO's weights on the crypto panel lift that null? E40-vs-E41 isolates fine-tuning. Prior: fine-tuning cannot reach the cross-sectional edge (E39), so a null across arms was the likely, still-informative outcome.

## What was tried
A 2x2 matrix of 4 arms: scheme {single global (sg), walk-forward refit-every 130 (wf)} x model-selection criterion {validation MSE, validation cross-sectional CORR}. Parameter-efficient fine-tune (GPT-2 frozen; LoRA + heads + RevIN affine trained), warm-started from Melady/TEMPO (.pth loaded strict=False). Training windows are (336->20) slices over each symbol's resolved target series with horizon-end <= the leak cutoff 2024-06-27 (min scored date minus 20-period embargo); loss is MSE on the first 20 output steps. Eval half reuses E40's build_tasks/forecast_signal verbatim; scored by `timesfm_score.py --model tempo` on the same 510 dates. Runs: e41_wf_mse, e41_wf_corr, e41_sg_mse, e41_sg_corr.

## Trained parameters
From the metrics JSONs, the scored config is recorded only as the eval description (`model` = tempo, `data_version` = crypto/v2.0, `target` = target_binned_return_20, `n_features` = 1, `note` = "zero-shot univariate, one symbol at a time" — the eval note carried over from E40). The fine-tuning scheme/selection axes (sg/wf x mse/corr) and the leak cutoff 2024-06-27 are recorded in the study idea/run names, not in the metrics JSON config; LoRA rank, learning rate and epoch count are not recorded in available artifacts.

## What was found
No written finding; from the metrics JSONs all four arms remain null-to-negative — fine-tuning did not lift E40's null, consistent with the stated prior. e41_wf_mse: CORR mean -0.013440 (sharpe -0.160126), MMC mean -0.004502, n = 510. e41_sg_corr: CORR mean -0.015009 (sharpe -0.178194), MMC mean -0.007202, n = 510. (The other two arms, e41_wf_corr and e41_sg_mse, were not opened in this pass but all four runs exist.) Every arm's CORR sits near the E40 zero-shot -0.017; adapting the weights did not recover a cross-sectional edge.
