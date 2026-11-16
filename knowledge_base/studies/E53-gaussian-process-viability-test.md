# E53 — Gaussian Process viability test -- full book, reg + clf

- **State:** not started (0/18 scouted)
- **Runner:** experiments/e53_gp_test_crypto.sh
- **Models / datasets:** model `gp` (subsampled exact GP, src/prediction/gp.py); planned over all 9 book datasets (raw22, tenc23, interact36, fenccausal44, cumlag3-88, tenc_onehot763, onehot_rollsel786, allfeat788, allfeatpp812)
- **Artifacts on disk:** 0.0 MB across 0 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** THIN
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: none on disk

## Main question
Does the Gaussian-Process wrapper run end to end on both regression and classification, and is reg or clf the task worth carrying into the expensive E54/E55 GP sweeps? A viability probe (like E18 TabNet), not a tuned model, and the gate before the E54 KTZ sweep.

## What was tried
Planned 18 arms = one `gp` arm per (dataset x task) across all 9 book datasets x {regression (posterior mean on the 5-binned target), classification (GPC positive-class probability, target binarised at 0.5)}. A subsampled EXACT GP with training rows capped (reg 6000, clf 1500) via `--model-options` because a GP is O(n^3); kernel frozen, outer search tunes family + length-scale (+ alpha for reg); light `--n-iter 12` at the deployable cadence (rolling 130/130, embargo 20, n_splits 5, seed 42); serial search x arm-level pool. The study never launched: 0 of 18 scout arms ran.

## Trained parameters
config JSON not on disk; no runs were produced (0/18). Scored target and all configs would be in the forecast headers, which do not exist for this study.

## What was found
THIN — the study was never started (state "not started", scout 0/18) and produced no forecasts, metrics, or written finding. No EXPERIMENTS.md section, memory file, or metrics JSON exists for it in the artifacts read. There is nothing scientific to record and nothing to delete.
