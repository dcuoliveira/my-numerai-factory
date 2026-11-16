# E70 — Engineered allfeat-style block on v2.0-momswap-macro

- **State:** done (10/10 arms)
- **Runner:** `experiments/e70_eng_momswap_macro_crypto.sh`
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz × {base `crypto/v2.0-momswap-macro` (49f), eng `crypto/v2.0-momswap-macro-eng` (925f)}
- **Artifacts on disk:** 60.2 MB across 20 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none (planning-table row, status "registered"); memory: e70-eng-momswap-macro.md, e71-xinteractions-momswap-macro.md (states E70's outcome); metrics: results/metrics/crypto/catboost_ktz/e70_catboost_ktz_eng.json (+ all 10 full arms)

## Main question
Does the full `allfeat_pp` engineering recipe help on top of momswap-macro (49f)? `build_momswap_eng.py` builds `crypto/v2.0-momswap-macro-eng` (925f) by adding, retargeted to the swapped set: tenc (1, causal lag-20 symbol target-encoding), onehot (752 symbol dummies), rollsel (87, rolling means {5,20,60} of 17 momentum + 10 macro-return + 2 volatility), spread (12, term-structure diffs), interact (24 = accel + adapted products + 12 momentum×macro products).

## What was tried
10 arms = 5 models × {49f base, 925f eng}, from scratch. Derived blocks stay continuous [0,1]; onehot 0/1; NaN→0.5. The hypothesis was that onehot/tenc (symbol identity) and momentum×macro (regime gating) help via interactions (trees/MLP), weakly for a linear model. Decision criterion: eng beats base at paired |t|≥2.

## Trained parameters
From the metrics JSONs: `cv_scheme=rolling`, `train_window=130`, `refit_every=130`, `n_splits=5`, `embargo=20`, `val_frac=0.3`, `metric=corr`, `search_type=random`, `n_iter=50`, `seed=42`, `target=target_binned_return_20`. `fixed_params` = null (ktz priors in the search grid). `n_features`: base 49, eng 925. n=520.

## What was found
MEMORY-BACKED (e71-xinteractions memory, describing E70's result: "dumped the full 925f allfeat block and it was null-to-negative — the 752 onehot dummies crushed ridge/MLP on CORR (−0.053/−0.040 at |t|>2), boosters only a slight non-sig +MMC (~0.006, t<1)"). The metrics reproduce this exactly: ΔCORR (eng − base) ridge −0.05327, mlp −0.04023 (the 752 onehot dummies crush the linear model and the MLP). The boosters are roughly flat on CORR (lgb +0.00132, catboost −0.00363, rf −0.00356) with payout deltas lgb +0.00608, catboost +0.00646, rf −0.00172 (0.1·ΔCORR + ΔMMC; the boosters' gain is MMC-driven), which the memory characterises as non-significant (t<1). Best eng arm by payout is `catboost_ktz_eng` (0.0761, then `lgb_ktz_eng` 0.0735); `rf_ktz_eng` pays 0.071428 at a punishing dd(corr) 2.25. Direction: null-to-negative — the full 925f block does not beat the 49f base; the onehot dummy noise is the culprit on the linear/MLP side. This motivated the E71 follow-up (strip to the interact block only).
