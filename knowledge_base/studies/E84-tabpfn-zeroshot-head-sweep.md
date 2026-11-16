# E84 — TabPFN zero-shot AND frozen+head on each of CU_SA_IA_BO_01's 8 feature-sets

- **State:** done (21/21 arms)
- **Runner:** /home/danielco/e84_chain.sh (zero-shot via src.run_forecast; head via src/analysis/tabpfn_head.py)
- **Models / datasets:** tabpfn (zero-shot) + tabpfn_head on 8 feature-sets of CU_SA_IA_BO_01 — raw 22, fenccausal 44, allfeat_pp 812, and the ridge-quintet blocks onehot/tenc/interact/size/price
- **Artifacts on disk:** 51.9 MB across 21 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §E84 row (full finding); memory: e82-e85-combo-fenccausal.md (id-collision note only); metrics: results/metrics/crypto/tabpfn/e84_zs_tenc.json

## Main question
Which feature-set is best for TabPFN zero-shot, and does the trained 5-class head (E83) ever beat its matched zero-shot base on any of the 8 blocks that compose the deployed slot CU_SA_IA_BO_01 (E83 found it neutral on fenccausal)?

## What was tried
For each of the 8 feature-sets: (a) TabPFN zero-shot at E19's config (n_est 4, ctx 4096, rolling 130/130, embargo 20, seed 42) — raw/fenccausal/allfeat_pp reuse E19's runs, onehot run at reduced n_est 1 / ctx 1024 because 740-dim is pathologically slow; (b) the E83 frozen-backbone + trained head (which also emits its matched n_est=1 zero-shot base). 13 new runs (5 new zero-shot + 8 head), ~3h. Reordered after a full-config onehot zero-shot stalled the sequential chain.

## Trained parameters
From results/metrics/crypto/tabpfn/e84_zs_tenc.json (zero-shot arm): model tabpfn; data_version `crypto/v2.0-tenc`; feature_include `^feature_sym_tenc$` (n_features 1); cv_scheme rolling, inner_cv purged; train_window 130, refit_every 130, embargo 20; n_splits 2, val_frac 0.3; seed 42; target target_binned_return_20; refit-0 best_params max_context 4096. Feature-sets selected via `--feature-include` regexes; the head arms use the bespoke tabpfn_head driver (no search config).

## What was found
FINDINGS-BACKED (EXPERIMENTS.md). **(1) Best zero-shot = fenccausal 0.14504 ≈ raw 0.14299** — the causal per-symbol means barely help TabPFN, and MORE features HURT (allfeat_pp 812 → 0.124 < raw). Blocks weak-but-real (size/tenc/price/interact 0.06–0.08; a single tenc column reaches 0.082); onehot ~null 0.015. **(2) Head vs base: NEUTRAL on every strong set** (raw/fenccausal/allfeat_pp |t|<1.1, confirms E83 generalizes); significantly HELPS only onehot (+0.0046 t +3.48, payout t +2.64 — recovers the ticker fixed-effect the mean underuses, but stays 0.033); HURTS price (−0.0225 t −2.43). Under ~8 tests the bar is ~|t| 2.7, so onehot is the one robust head-win. **No feature-set where the head makes a good TabPFN better.** From the metrics (standalone): best zero-shot cells in this batch are tenc 0.06092 / CORR 0.08206 (n 510), size 0.04814, price 0.03989, interact 0.03568, onehot 0.00609.
