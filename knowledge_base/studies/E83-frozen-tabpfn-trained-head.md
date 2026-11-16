# E83 — Frozen TabPFN backbone + a trained 5-class head (E[y|x] readout)

- **State:** done (2/2 arms)
- **Runner:** src/analysis/tabpfn_head.py
- **Models / datasets:** tabpfn_head on `crypto/v2.0` (the head reads the frozen backbone's bar-distribution)
- **Artifacts on disk:** 4.9 MB across 2 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §E83 row (full finding); memory: e82-e85-combo-fenccausal.md (notes the E83/E84 id-collision only); metrics: results/metrics/crypto/tabpfn_head/e83_head_base.json

## Main question
The gentle adaptation that full fine-tuning (E44/E45) lacked: with TabPFN's backbone frozen, can a newly trained 5-class head improve on the zero-shot readout without corrupting the meta-learned prior?

## What was tried
Backbone frozen (vendored checkpoint untouched); only a new head trains — `Linear` over the softmax'd ~5000-bin bar-distribution → 5 classes {0,.25,.5,.75,1} → softmax → E[y|x]. The emitted signal is the continuous bounded expectation, not a hard label. Head trained with cross-entropy on the leak-safe pre-2024-06-27 pool (CE 1.54→1.33 over 4 epochs). Eval = per-date walk-forward (context = 130 dates before T−20, subsample 4096); one forward per date yields both `e83_head` (head E[y|x]) and `e83_head_base` (raw bar-dist mean = matched n_est=1 zero-shot, validated ≈ E19). Pairing head vs base isolates the head; also compared vs E19 (n_est=4 zero-shot).

## Trained parameters
From results/metrics/crypto/tabpfn_head/e83_head_base.json: model tabpfn_head; data_version `crypto/v2.0`; target target_binned_return_20; n_features 1 (the readout column). The standard search config fields (cv_scheme, train_window, refit_every, embargo, seed, n_iter, n_splits, val_frac) are all null — this is a bespoke walk-forward driver, not the random-search pipeline, so no hyperparameter search / fixed_params are recorded. Head training detail (epochs, CE) is in the EXPERIMENTS.md narrative, not the metrics JSON.

## What was found
FINDINGS-BACKED (EXPERIMENTS.md). **Neutral, not a win, but crucially not corrupting.** Head CORR 0.12207 vs matched base 0.11129, paired **+0.0108 (t +0.82)**; MMC −0.0014 (t −0.13); payout **−0.0003 (t −0.02)** — a tiny insignificant CORR nudge, flat payout (≈153k rows/arm). Head vs E19 (n_est=4 zero-shot) payout −0.028 (t −2.40), still well below. Freezing the backbone avoids the E44/E45 prior corruption (head ≈ neutral, not damaging) and emits a valid bounded E[y|x], but cannot manufacture a real gain: the frozen backbone already produces a near-optimal ranking, so the head just re-learns the natural re-binning (≈ the mean). **Zero-shot TabPFN (E19) remains the best.**
