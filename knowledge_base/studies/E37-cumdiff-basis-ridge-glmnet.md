# E37 — Cumulative per-symbol differences to a year, on ridge_glmnet

- **State:** done (26/26 arms)
- **Runner:** experiments/e17_cumdiff_glmnet_crypto.sh
- **Models / datasets:** `ridge_glmnet`; per-symbol cumulative-difference / lag feature sets built on the ladder 1,2,3,5,10,21,63,130,261 over `crypto/v2.0` (data_version variants `crypto/v2.0-e17{diff,lag,featdiff,rawdiff}N`)
- **Artifacts on disk:** 130.5 MB across 26 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none (no dedicated section for E37); memory: e28-e29-prefix-collision.md (notes the E17/E37 id collision as bookkeeping only, no result); metrics: results/metrics/crypto/ridge_glmnet/e17_raw.json, e17_diff9.json, e17_lag9.json, e17_featdiff9.json, e17_rawdiff9.json, e17_rawdiff8.json

## Main question
E16's span question asked of the linear model and reframed as a BASIS question: build per-symbol cumulative differences on the ladder 1,2,3,5,10,21,63,130,261 (261 dates ≈ one calendar year). Span is already known null (E05/E06/E13/E14), but span{f_t, f_t−f_{t−k}} = span{f_t, f_{t−k}}, so an unpenalised fit cannot tell a difference set from a lag set while ridge can (||b||² does not commute with a change of basis; lag-1 autocorrelation 0.82–0.99 means the lagged-level basis hands near-copies and the difference basis near-orthogonal residuals). Levels 3/6/9 carry a featdiff/lag pair at identical span and feature count to isolate exactly that.

## What was tried
26 `ridge_glmnet` arms on a refitted current snapshot (control `e17_raw` is refit because `w6r6_glmnet_reg` is on an older date count). Arm families (by data_version): `diff1..9` (pure cumulative-difference sets), `lag3/6/9` (lagged levels), `featdiff3/6/9` (feature differences beside raw), `rawdiff1..9` (raw differences beside raw), plus a `tgtdiff6` arm and the `raw` baseline. Coverage caveat recorded in the idea: an order-261 difference is real for only ~61% of rows.

## Trained parameters
From the configs: model `ridge_glmnet`, inner_cv purged, n_splits 5, n_iter 50, search_jobs 8, cv_scheme rolling, train_window 130, refit_every 130, embargo 20, seed 42, target `target_binned_return_20`, fixed_params null, model_options {}. Representative `e17_rawdiff8.json`: data `crypto/v2.0-e17rawdiff8`, n_features 206; `e17_lag9` n_features 220, `e17_featdiff9` n_features 220, `e17_rawdiff9` n_features 229, `e17_diff9` n_features 207. Arms sweep the data_version (feature construction) and feature count; the ridge search protocol is held constant.

## What was found
NUMBERS-ONLY — no written finding; from the metrics (n=514 each). **No difference/lag/rawdiff arm beats the raw 22-feature baseline, and the pure-difference sets collapse:**
- `e17_raw` (22 feat): CORR 0.145497, MMC 0.063020, payout 0.077569.
- `e17_rawdiff8` (206 feat, best of the sampled ladder arms): CORR 0.136446, MMC 0.060724, payout 0.074369.
- `e17_rawdiff9` (229 feat): CORR 0.129757, payout 0.070187.
- `e17_featdiff9` (220 feat): CORR 0.132888, payout 0.072207.
- `e17_lag9` (220 feat): CORR 0.129747, payout 0.066452.
- `e17_diff9` (pure difference set, 207 feat): CORR 0.017981, MMC 0.001941, payout 0.003739 — a near-total collapse; the difference-only basis carries almost no rank signal on its own.
So the lagged-level and raw+difference sets all sit below raw (CORR ~0.13 vs 0.1455) and the pure-difference set is far worse. No paired t-stats are recorded in EXPERIMENTS.md; this is a metric-level summary, not a tested finding.
