# E39 — The tabular reframing of E38: TabPFN zero-shot on ALL features

- **State:** done (1/1 arms)
- **Runner:** run_forecast (tabpfn on crypto/v2.0-tenc-onehot-rollsel-spread-interact) — recorded as a prose string, not a file path
- **Models / datasets:** `tabpfn` (TabPFN v2 regressor) zero-shot on `allfeat_pp` = `crypto/v2.0-tenc-onehot-rollsel-spread-interact` (812 columns)
- **Artifacts on disk:** 4.5 MB across 1 forecast run (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §E39 (line 41); memory: e28-e29-prefix-collision.md (notes E39's prose runner string breaks a test — bookkeeping only, no result); metrics: results/metrics/crypto/tabpfn/e39_tabpfn_allfeat.json

## Main question
E38's direct counterpart: same prediction problem and window, but reframed from a univariate time series to a TABULAR one and given to TabPFN zero-shot — each row is its full feature vector, and the foundation model predicts the target in-context from past resolved rows. Uses ALL features (allfeat_pp, the 812-column union of every engineered block) at context 2048 (better than E19's memory-constrained 1024). Isolates the FRAMING: TimesFM on the target's own history (E38) got CORR 0.012; what does the SAME kind of zero-shot foundation model extract once price history is presented as tabular features?

## What was tried
1 arm. Zero-shot TabPFN through the `run_forecast` path on the 812-column allfeat_pp dataset, context length 2048, GPU. Compared conceptually against E38 (0.012) and the earlier `e19_allfeat_pp` (~0.124 at context 1024).

## Trained parameters
From `e39_tabpfn_allfeat.json`: model `tabpfn`, data_version `crypto/v2.0-tenc-onehot-rollsel-spread-interact`, n_features 812, fixed_params {max_context 2048}, model_options {device cuda, max_context 2048, model_path models/tabpfn/tabpfn-v2-regressor.ckpt, n_estimators 4}, cv_scheme rolling, n_iter 1, n_splits 2, train_window 130, refit_every 130, embargo 20, seed 42, target `target_binned_return_20`. Zero-shot (no fitting/search); the only tuned knob is the context length (2048).

## What was found
FINDINGS-BACKED (EXPERIMENTS.md §E39, line 41), cross-checked against the metric JSON. **Reframing the same zero-shot foundation-model idea from univariate (E38, CORR 0.012) to tabular-on-all-features lifts it by an order of magnitude:**
- `e39_tabpfn_allfeat`: CORR mean 0.134525, MMC mean 0.058321, payout mean 0.071773 (= 0.1·CORR + MMC, per period; EXPERIMENTS.md line 41 quotes 0.0718), n_periods 510.
So the framing — not the model family — is what extracts the signal: TabPFN on the 812-column feature vector at context 2048 scores CORR ≈ 0.135, versus TimesFM's ≈ 0.012 on the target's own history and the ~0.124 that `e19_allfeat_pp` reached at context 1024. EXPERIMENTS.md §E39 (line 41) records the paired comparison against E38: CORR +0.123 (t +6.23), MMC +0.046 (t +3.36), payout +0.058 (t +3.77), c.sharpe 1.07.
