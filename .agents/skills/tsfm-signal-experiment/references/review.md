# Review against the momentum workflow

The original attached E206 request was "Run an extensive forecasting and portfolio
experiment on crypto/v2.0-e205-ts-momentum3". Its six standardizations, A/B/C grid,
reserved test and actual-return portfolios were superseded by later user choices.
This reusable specification starts AFTER the user has selected and built a
signal dataset. It carries forward the later choices, with the latest top-three
and ensemble-residualization requirements.

| Original / shorthand request | Consistent reusable interpretation |
| --- | --- |
| Momentum dataset and 33 features | Any supplied dataset and its declared feature list; no momentum constants, masks or dates |
| Six H standardizations x A/B/C | Recommended standardization per selected model, design C; sweeps only when explicitly requested |
| One asset at a time | Per-asset foundation calls and supervised ridge heads; C intentionally uses other assets' contemporaneous H after extraction |
| "Add H" to ridge | Historical C-only H head plus matched feature-only baseline; raw-X-plus-H is a separately requested augmentation |
| "Train the foundation models" | Frozen pretrained weights; fresh per-asset ridge heads. TabPFN-TS uses native causal contextual fitting |
| Target20 and target60 | Extract horizon-specific contextual H and fit heads for each; score every stream on BOTH labels |
| Train/test portfolio summaries | One aggregated full-history chronological forecast backtest, retaining causal inner validation and warm-up diagnostics |
| Residualize predictions | Existing strength0.5 prediction transform on meta snapshots at least5 days old; retain raw controls; no residualized training targets |
| Ensemble residualization | Latest explicit choice: ALSO save and score a separately residualized version of every rank-average blend |
| Return Sharpe / drawdown | sharpe.p and maxdd.p of additive Numerai payout; no price-return metric substitution |
| "Cumulative returns" in the last bullet | Cumulative payout scores, consistent with the metric correction in this chat |
| Top five in E211 | Latest request: top THREE ensembles plus every standalone variant, separately by evaluation label |
| Same-history top ensemble | Descriptive/exploratory membership search, not independent confirmation |
| Original automatic CU doubling | Use current authorization/policy and actual capacity; do not carry an old study-specific exception into future studies |

Necessary details added: immutable source/feature/label identities, causal
coverage and full-panel routing, label walls and purges, independent per-asset
alpha selection, training-only normalization/vocabulary, matched feature-only
control, raw/residual comparison labels, full trial counts, both evaluation
labels, scalar and batched score checks, per-device extraction audits, cache
fingerprints, resumable scheduling, resource/ETA checks, source-vintage limits,
registry-before-launch, and all artifact/report completion checks.

## Reference implementation map

Read these as verified building blocks and study-specific examples, not as a
ready dataset-agnostic launcher:

- `src/data_mani/e206_tsfm_embeddings.py`: per-asset frozen extraction, origin
  batching, audits, atomic caches; root/scratch/target/wall environment overrides.
- `src/data_mani/tsfm_state_adapters/`: model-specific pre-head extraction.
- `src/analysis/tsfm_momentum_study.py`: H standardization, fixed-vocabulary C
  kernels, constant handling and per-asset dual ridge.
- `experiments/e206_momentum3_tsfm.py`: per-asset tuning/fitting and routing;
  default dataset/features/fixed split are momentum-specific.
- `experiments/e207_fullsample.py`: chronological block assembly; hardcoded
  dataset/caches and its original72-arm grid require adaptation.
- `experiments/e210_tsfm_ensembles.py`: exhaustive equal-rank blends, fresh scoring
  and standalone replay; its five-row pool and E206 dates are specific to that study.
- `experiments/e211_cross_target_tsfm.py`: 20/60 walls, fresh heads, causal meta
  lookup and raw/residual standalone pool. Its momentum paths, label cutoff,
  fixed16 pool, top-five plot, and absence of post-ensemble residualization do
  not implement this skill unchanged. It also does not produce the requested
  standalone payout heatmap: add that output when adapting the pipeline.
- `src/prediction/meta_resid.py`: shared residualization and causal meta snapshots.
- `experiments/e150_finish_scoring.py::matrix_scores`: batched CORR/proxy MMC.
- `src/live/submit.py::rank_normalise` and
  `src/analysis/asset_momentum.py::route_slots`: ranking/routing contracts.

Review source behavior before reusing a component: historical scripts are
examples and may still have dataset-specific assertions, path globals or
completion assumptions. No existing E206/E207/E211 job is changed by installing
this skill, and no new experiment is launched solely to create it.
