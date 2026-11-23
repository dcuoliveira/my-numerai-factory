# Execution protocol

## Dataset and frozen configuration

Resolve the supplied dataset to an actual artifact and read its manifest,
builder, feature definitions, model/prediction entrypoints and evaluation path.
Record dataset/version, selected feature names, stable asset IDs, date range,
label columns and availability, source vintage, coverage and hashes. Inputs
are the provided signals; there is no automatic repeat of momentum formula
selection, binning, price downloading, one-hot encoding or feature engineering.
Use the dataset's actual feature count, never a fixed 33-column assertion.

Keep every source asset/date key and genuine missing value. Coverage masks,
short-history eligibility and source staleness are metadata, not predictors.
Own-asset uncovered rows cannot contribute to candidate fitting or prediction.
Join absent official target20/target60 labels using validated asset/date identity;
retain unavailable labels as missing, never impute them or shrink the source
panel. Respect any existing fenced holdout and filter dates at read time. Full
sample means the permitted source history, not permission to open a holdout.

The default question is standalone representation performance versus the matched
feature-only ridge. Use the missing-values standard's B_all/B_covered/F_covered
controls if the requested claim is instead marginal value alongside native
Numerai features; do not describe this standalone study as that marginal test.

Freeze one manifest for the selected dataset, checkpoints/revisions, training
horizons, feature list, masks, context, extraction layer/token, standardization,
ridge grid, validation loss, forecast blocks, residualization, scoring and
ensemble pool. Use unique IDs containing dataset, model, training target,
standardization, design and run. Get the next free study number immediately
before registration; commit registry/notes before real runs, including smokes.

## Time and coverage

Defaults inherited from E211, subject to verified dataset label availability:

| Item | Default |
| --- | --- |
| Training / evaluation labels | target_binned_return_20 and target_binned_return_60 |
| Label walls for these v2.0 labels | 35 / 91 calendar days |
| Frozen-model context | 256 panel observations; at least 32 resolved labels |
| Head history | Up to 504 preceding panel dates |
| Inner validation | Last 63 resolved panel dates |
| Forecast block | 130 panel dates, first candidate origin after 200 panel dates |
| Head eligibility per asset | At least 60 training and 20 validation observations |
| Ridge grid | 33 alphas, logspace(-4, 4), independent per-asset MSE selection |
| Rolling H windows | 63, 126, 252 prior valid observations, minimum 32 |

At forecast origin D, for horizon h use labels with availability <= D. With the
calendar-wall convention, final fitting ends at D-wall(h); validation ends there
and training for parameter selection ends no later than validation_start-wall(h).
Never confuse 60 panel observations with the 91-calendar-day serving wall.
Verify walls against the actual dataset; use actual availability timestamps or
a more conservative delay when needed. Covariates stop at their permitted
cutoff, at most D, even when supplied as known-future relative to the label wall.

For each chronological block, choose each asset's alpha on past validation,
refit its head on the resolved trailing history and freeze parameters for that
forecast block. No pooled supervised head. For rolling H, choose the window per
foundation model/training target/block by equal-weight mean per-asset validation
MSE; each asset still gets its own alpha/intercept/coefficients. Do not optimize
this window on the final payout table. Robust training scalers and feature-only
baseline scalers are fitted on eligible training observations and frozen per
block. Rolling H statistics exclude the current observation.

Join disjoint forecast blocks; assert no duplicate asset/date predictions. Keep
warm-up and insufficient-history predictions missing in candidate artifacts and
report their dates separately. Never replace them with in-sample forecasts or
invent payout observations. Payout curves start when common scoring is possible;
a cash convention for an actual tradable portfolio is a separate metric.

## H and independent ridge heads

Reuse the standardization skill's precise scale and constant handling. Robust
uses pooled training medians and IQR/1.349. Native foundation-model normalization
is unchanged. H has logical shape [asset, date, width] and a validity mask; use
sparse row-indexed storage/memmaps where useful.

For each training horizon, feed that horizon's resolved historical target and
the selected dataset signals to the frozen model. Select its target-series state
immediately before the forecasting projection at the token corresponding to D.
Record missing-channel treatment, horizon/patch alignment and actual width.
TabPFN-TS uses the historically conditioned query representation; document its
architectural difference. Unsupported extraction/covariate paths get explicit
unsupported status, not fabricated H.

Foundation-model calls contain one asset only. Multiple independent origins of
that same asset can be batched when horizon-compatible. Design C is constructed
AFTER extraction: concatenate contemporaneous standardized H from the fixed
training asset vocabulary in a saved column order, including the predicted asset
once. Each predicted asset has a separate ridge, even though C includes other
assets' H. It does not imply pooled supervised training. Internal zero filling
of missing context blocks follows E206's C convention, without coverage flags;
never train on a missing own-asset H block. Freeze the vocabulary for the block,
record later entrants and remove constant coordinates, including implicit
removal by an exactly centered dual kernel. Use primal/dual solutions that are
numerically equivalent to the declared ridge objective.

The historical default head input is **C only**. Dataset signals already enter
the foundation context, and the feature-only ridge is a separate baseline.
Concatenating raw signals with C would be an additional, explicitly named arm;
do not quietly change the experiment to raw-X-plus-H.

Cache identities include dataset/feature/label/mask hashes, training horizon,
label wall, checkpoint, layer/token, context and extraction-code revision.
Changing to target60 requires new target60-context H, not just new ridge y on
unchanged target20 H. Share embeddings across downstream heads only when cache
identities match; save normalization and per-asset parameters separately by
block. Retained CPU/GPU asset caches must record their actual execution device.

## Residualization, ranking and matched evaluation

Use `src.prediction.meta_resid.residualise`, STRENGTH=0.5 and LAG_DAYS=5.
For origin D align symbols to the latest historical meta snapshot <= D-5
calendar days using `snapshot_for`/`lagged_meta`. Preserve native missing-meta
and no-snapshot pass-through behavior. Store snapshot, coverage, beta and
pass-through diagnostics. A cached lookup must match the shared helper exactly.
Never residualize with the same-day meta used by the evaluator.

Apply the transform to every standalone raw forecast once and keep both views.
For each ensemble, preserve the rank-average blend and transform that blend
once into a distinct post-residualized ensemble view, even if members contain
standalone residualized forecasts. Do not overwrite the original or recursively
residualize an already post-residualized ensemble. No residualized training labels.

Separate the causal predictor mask from the evaluation-only label/meta mask.
Build the common causal sample from all declared standalone predictor
requirements, using only cutoff-available information. Apply historical label
and scoring-meta availability only for evaluation. Keep identical scored
asset/date keys, >=10 assets per scored date, for all trials and both evaluation
targets. Do not use target availability to choose a transform's cross-section.

Use the feature-only ridge as the default routing baseline, with a per-asset
resolved-training-label mean fallback where it cannot predict and a declared
neutral prior before any resolved labels exist. These are output fallbacks, not
feature imputation or permission to train on uncovered candidate rows. Fit the
covered feature-only comparison on the same eligible training keys as the H
heads. Keep the fallback explicit in the forecast artifacts.

On the full panel, route covered candidates into exactly the baseline rank slots
and preserve uncovered baseline ranks. Verify all routed predictions finite and
both rank-slot invariants. Store raw candidates, masks and routed forecasts;
score the common-covered view requested here, with a separately labelled full
routed view if reported. Native-feature marginal claims require the matched
routed control comparison described in the missing-values skill.

Use a documented deterministic rank convention for ensemble blending and
routing; preserve the repository scorer's own tie treatment for CORR/MMC.
Compute member ranks, ensemble blends and residualization on the causal common
predictor universe before applying the evaluation-only label/meta mask. Freshly
score each prediction blend; never average member CORR, MMC or payout
series as a substitute for constructing the ensemble's predictions. Batched
ensemble scoring must match scalar scoring. Reproduce cached standalone scores
on identical keys before accepting cache-based comparisons.

For each date and evaluation horizon, use the repository's Numerai CORR and
historical-meta proxy MMC definitions. Explicitly identify proxy MMC when it
uses the existing historical meta for target60 rather than a separate target60
meta benchmark. Define p_t = 0.1*corr_t + mmc_t, then:

- payout = mean(p_t)
- sharpe.p = mean(p_t) / std(p_t, ddof=0), unannualized; undefined at zero variance
- maxdd.p = max_t(running_max(S)_t - S_t), S=[0, cumulative_sum(p_t)]

Payout is additive; do not compound it or replace payout Sharpe/drawdown with
actual asset-return measures. Treat existing `dd.p` fields as the same payout
quantity only after verifying the definition; render the requested `maxdd.p`
header. Keep CORR and MMC, as well as per-date p_t, in the output artifacts.

## Ensemble counts, tables and plots

With M selected foundation models plus one feature-only baseline, two training
horizons and raw/residual views produce K=4*(M+1) standalone streams. Enumerate
E=sum(comb(K,s), s=2..min(5,K)) subsets, with equal weights and no trained ensemble
weights. Both blend and post-residualized views are scored on two targets, so
the complete row count is 2*(K+2*E). Count from actual declared streams rather
than reusing E211's fixed16. Label failures/unsupported streams; do not silently
shrink the pool or replace failed forecasts with neutral values to hide failure.
Exact equivalent forecasts may share computation, but preserve all trial IDs.

Examples: three selected TSFMs -> K=16, E=6,868, 27,504 scored rows; all four ->
K=20, E=21,679, 86,756 rows. These totals include both ensemble views. Estimate
scoring time, memory and report size before choosing batching/storage. Large
exhaustive tables still need complete CSV/Parquet exports, not only a top-N list.

For each evaluation target, deliver:

1. Standalone table sorted by payout descending, including raw/residual views,
   training target, H method/design, dates and coverage. Required metric columns:
   corr, mmc, payout, sharpe.p, maxdd.p.
2. Heatmap of Pearson correlations between DAILY PAYOUT series of the standalone
   rows on the identical dates. Do not correlate cumulative curves. Label any
   separately requested prediction/return correlation differently.
3. Cumulative payout plot of all standalone rows.
4. Consolidated table of every standalone and original/post-residualized ensemble
   trial, sorted by payout within evaluation target. Membership labels include
   training horizon and member raw/residual view; a separate field identifies
   ensemble-level residualization.
5. A new cumulative payout plot with the top THREE ensemble rows by mean payout
   and all standalone variants, drawn on exactly the same score dates. Keep
   target20 and target60 plots separate. Plot values must reconstruct from the
   exported daily scores; ties use a declared deterministic trial-ID order.

Top-three membership selection uses the displayed history and is exploratory.
Do not call this untouched confirmation; uncertainty comparisons, if requested,
need paired dates and horizon-aware autocorrelation handling. A later deployment
selection needs its own causal membership protocol.

## Execution and completion

Probe actual GPU utilization, free memory, CPU/RAM capacity and disk headroom;
do not copy old host availability. Prefer GPU extraction, CPU ridge/scoring,
with measured per-origin timings and an updated ETA. Avoid silently starting
a multi-day CPU fallback when GPUs exist; explain the resource constraint and
resolve any required allocation decision while completing independent setup.
Honor current authorization and resource limits; an old E206/E211 GPU/CU
exception is not automatically authorization for a new study. Do not stop other
studies or occupy another user's GPU. Account for existing live GPU reservations
across scheduler iterations, not just placements within the current iteration.

Use the repository scheduler, pinned affinity/thread counts, atomic per-asset
caches and completed-fold checkpoints, safe owner-verified locks and bounded
retries. On out-of-memory reduce batch size without changing context or signal
semantics. Save commands, hosts/devices, resource snapshots and progress counts
by stage, assets, origins, blocks and trials. Stop only named study units and
restore any scoped allowance in cleanup. Preserve successful caches on restart.

Before launch, verify cutoff/future-label and prefix invariance at both horizons,
batch-origin independence, asset identity/missingness, causal scalers/vocabulary,
primal/dual correctness, native residualization parity, both ensemble views,
scalar/batch scorer equivalence, rank routing, metric formulas and report
reconstruction. Register real smoke runs under the study before execution.
Mark embedding readiness only after the final cross-model coverage mask is
atomically saved, and completion only after required metrics/reports succeed.

Persist config/code/checkpoint hashes, H and masks, scalers, per-asset selected
alphas/parameters, block cutoffs, raw/residual/routed forecasts, residualization
snapshots, ensemble definitions, daily scores, all tables, plots, HTML/PDF, logs,
resource usage and terminal status. Run the missing-values audit before claiming
results; distinguish calculation causality from uncertified source vintages,
pretraining overlap and prior signal/configuration selection. Write findings in
EXPERIMENTS.md; commit/push the requested code/docs changes, preserving unrelated
work. Report blockers and report-generation failures explicitly, not as success.
