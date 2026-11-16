# E103 — E102 LoRA fine-tune per Tier-1 (crypto-derived-structural) feature set

- **State:** done (7/7 arms)
- **Runner:** /home/danielco/e103_chain.sh
- **Models / datasets:** tabpfn (LoRA fine-tuned) on 7 Tier-1 feature sets — raw v2.0 (22f), lag3 (44f), cumdiff3 (69f), ar (25f), interact (36f), tenc (23f), aetuned (30f); fenccausal (44f) is e102_lora reused as the 8th screen member
- **Artifacts on disk:** 35.4 MB across 7 forecast runs, plus 7 metrics + 7 hyperparams + 7 per_period (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: all 7 arms (e103_lora_{raw,lag3,cumdiff3,ar,interact,tenc,aetuned}.json)

## Main question
Apply E102's LoRA scheme (per-date cross-sectional CORR loss, r=8 adapters on q/k/v/out, backbone frozen, merged then evaluated as the ordinary E19 walk-forward) to each Tier-1 feature set — the ones built by structural processing of the raw 22, no external data. Hypothesis: each single arm underperforms E19 zero-shot (as e102 did), but the LoRA objective may decorrelate them so a blend — and a blend with the book — adds a risk (drawdown/calmar) edge rather than a payout-level one.

## What was tried
Seven arms, each a LoRA fine-tune on one Tier-1 dataset. Dropped: allfeat 812f (TabPFN stall/OOM on 8 GB), e10stack 2f (degenerate), the cumulative-lags family (OOM on 8 GB). Trained on deepfive (2× idle GPU). Then intended to screen ensembles of the 8 LoRA arms, and of the winners with the submitted book + e102. Leak-safe (≤2024-06-27), paired NW.

## Trained parameters
From results/metrics/crypto/tabpfn/e103_lora_aetuned.json config (representative): model tabpfn; data_version crypto/v2.0-aetuned; target target_binned_return_20; task regression; cv_scheme rolling; inner_cv purged; train_window 130; refit_every 130; embargo 20; era_step 1; n_splits 2; val_frac 0.3; n_iter 50; metric corr; n_features 30; seed 42; fixed_params null; model_options {device cuda, max_context 4096, n_estimators 4, model_path .../models/tabpfn/e103_lora_aetuned.ckpt}. What swept across arms is only the data_version / n_features: raw 22, lag3 44, cumdiff3 69, ar 25, interact 36, tenc 23, aetuned 30.

## What was found
NUMBERS-ONLY. No written finding, and the E19 zero-shot baseline and any ensemble-screen output were not among the artifacts read, so neither the per-arm vs-E19 delta nor the blend result can be quoted. From the metrics, per-arm single-model scores (CORR mean / MMC mean / n): raw 0.11132 / 0.02436 / 524; lag3 0.10814 / 0.02440 / 510; ar 0.10555 / 0.02440 / 514; interact 0.10782 / 0.02800 / 510; tenc 0.10783 / 0.02852 / 510; aetuned 0.10520 / 0.02400 / 513; cumdiff3 collapses to 0.05018 / 0.01937 / 514. Best single arm: raw (CORR 0.11132, CORR dd 0.567). Every arm's MMC is low (~0.024-0.029) with a large MMC drawdown (~3.2-4.0). All seven sit below the fenccausal e102_lora arm (CORR 0.13593). Direction, from numbers only: the Tier-1 LoRA single arms are modest and below the fenccausal reference; whether a blend adds a risk edge is not answerable from the sources read.
