# E40 — TEMPO zero-shot univariate forecast of each symbol's target

- **State:** done (1/1 arms)
- **Runner:** src/analysis/tempo_zeroshot.py
- **Models / datasets:** model `tempo` (TEMPO-80M GPT-2-backbone time-series foundation model, zero-shot); scored on crypto/v2.0, `target_binned_return_20`, 1 input feature (the coin's own target history)
- **Artifacts on disk:** 2.9 MB across 1 forecast run (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/tempo/e40_target.json

## Main question
Can a pretrained generative time-series foundation model (TEMPO), applied zero-shot per-symbol and univariately on each coin's own `target_binned_return_20` history, produce any edge? It is the controlled model-swap sibling of E38 (TimesFM in place of TEMPO), designed so E38-vs-E40 isolates the MODEL. A weak signal was the stated prior — the edge on this panel is cross-sectional (E39), not autoregressive.

## What was tried
One arm. For each symbol, context = the coin's own `target_binned_return_20` up to the 20-period leak wall (dates <= t-20), seq_len 336, forecast horizon 20, signal = mean of the horizon. Pretrained Melady/TEMPO TEMPO-80M_v1 weights, no adaptation (frozen zero-shot). Bespoke inference via `src/analysis/tempo_zeroshot.py`, scored by `timesfm_score.py --model tempo` on tabpfn/e19_fenccausal's 510 scored dates.

## Trained parameters
From results/metrics/crypto/tempo/e40_target.json config: `model` = tempo, `data_version` = crypto/v2.0, `target` = target_binned_return_20, `n_features` = 1, `note` = "zero-shot univariate, one symbol at a time". No hyperparameter grid, cv_scheme, train_window or seed is recorded in the metrics JSON (it is a frozen zero-shot inference, not a fit).

## What was found
No written finding exists; from the metrics JSON the arm is null-to-negative, confirming the stated prior. CORR mean -0.017124 (std 0.083931, sharpe -0.204024, max_drawdown 8.9495), MMC mean -0.002688 (sharpe -0.037919), over n = 510 periods. The zero-shot univariate framing carries no usable edge on this panel.
