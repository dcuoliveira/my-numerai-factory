# Volatility proxy experiments

This series uses the November 13, 2026 volatility synthesis as a design reference.
Each family is a separate development study. The panel is the frozen Numerai
Crypto v2.0 sample through February 19, 2026 with its 20-day binned target.
Published historical meta-model predictions are joined by date and permanent ID.

| Study | Estimator variants |
| --- | --- |
| EWMA | Recursive log and simple returns, normalized log-return weights, and the fixed RiskMetrics .94 decay |
| Semivariance | Downside, upside and joint inputs. Zero log or simple return, trailing mean, and side-count denominator |
| Idiosyncratic | Rolling residual variance relative to BTC, ETH, or BTC and ETH |
| Parkinson | Arithmetic and exponentially weighted daily range |
| Garman–Klass | Standard two-term, four-term, and squared gap added |
| Rogers–Satchell | Arithmetic mean, squared gap added, and exponentially weighted mean |
| Yang–Zhang | Standard, gap-free, and mean-square opening gap |

Each study crosses daily and weekday sampling, the reference's 11 horizons from
5 to 240, and raw, own-history percentile, and own-history bins. The fixed
RiskMetrics decay has one 20-return warm-up per calendar. Semivariance tests
each side separately and jointly in one study. See the report for equations and
source references.

Ridge tunes 13 log-spaced penalties from 1e-4 to 1e2 per setting. The main
marginal comparison adds one proxy to the 22 shipped features with a same-row
shipped-only comparator. Stand-alone and asset-fixed-effect fits have their
own controls. A nonlinear sensitivity uses four equally budgeted LightGBM
settings for each anchored 20/60 raw or own-history percentile add-on.
Calendar, estimator, horizon and transform can also be selected by inner
validation separately in each fold. Swept outer leaders are descriptive.

The coverage cohort is the intersection across the full within-family grid.
Stored features retain missing values and original panel keys. Only shipped
v2.0 feature gaps receive the repository's neutral bin. For historical scoring,
each covered prediction occupies a slot from the same model family's full v2.0
baseline ranking. The full universe has finite predictions and uncovered assets
retain their baseline ranks exactly. Price alignment stops at D−1. Backtests
preserve the same 22-date purge for inner and outer splits.

## Run

Keep the request cache for replay. A rate limit closes the current download
batch while preserving successful responses. Re-running resumes the missing
request windows. `--cache-only` assembles an existing cache without requesting
more bars. Examine the price-request manifest for any remaining source gaps.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m src.data_mani.fetch_volatility_prices --output results/volatility_proxies
for family in ewma semivariance idiosyncratic parkinson garman_klass rogers_satchell yang_zhang
do
    python -m src.analysis.volatility_experiment "$family"
done
python -m src.analysis.volatility_proxy_report
```

The PDFs, full sweep CSVs and paired comparison CSVs appear in
`results/volatility_proxies/reports`. Each study directory contains the feature
values and masks, fold schedule, parameter selections, date-level scores, routed
prediction audit files and input hashes. To reproduce the original run, use the
same hashed data, source cache and request manifest.

The download is retrospective. The cut-date replays check calculation timing.
Historical provider availability and live execution parity require independent
evidence. The causal panel-age control and retrospective source-availability
placebos are diagnostics and stay out of model selection.

## Consolidated window comparison

After the original sweeps complete, run
`python -m src.analysis.volatility_consolidated` with the same single-threaded
BLAS settings. It replays the recorded Ridge selections for raw 120,
own-history percentile 40, and own-history bins 40 and 120. The fold's
Ridge + v2 implementation also nominates the variant for the four-point
LightGBM add-on search. All 112 forecast series use the original test folds
and proxy-specific cohorts.

The consolidated PDF, score table and two labeled prediction-correlation
matrices appear in `results/volatility_proxies/consolidated`. The PDF leaderboard
and score CSV rank all 112 combinations by descending mean payout on the
468 dates with historical meta predictions. The matrices
average daily Spearman correlations, once for full-universe routed forecasts
and once for covered-model forecasts on the intersection of asset-date keys.
The companion count matrices give the number of dates used by each pair.

## Replace the shipped volatility columns

The replacement dataset retains the base parquet's asset-date keys, targets
and 20 other shipped inputs. Nine fixed volatility features replace the
shipped 20- and 60-day volatility columns. The estimator variants use earlier
fold nominations, with ties preferring the standard formula and daily bars.
The two semivariance columns use the most frequent joint variant. Only rows
with all nine values enter the joint cross-sectional binning. Partial
availability is recorded in the sidecar and the stored replacement block
is missing for those rows.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m src.data_mani.fetch_volatility_prices --extend-from results/volatility_proxies \
    --output results/volatility_replacement
python -m src.data_mani.build_volatility_replacement
python -m src.analysis.volatility_replacement
python -m src.analysis.volatility_replacement_report
```

The candle extension preserves the old snapshot, including its source gaps,
and downloads later bars through the latest training cutoff in the local v2.0
panel. The experiment reuses the original 11 development folds and stops at
February 19, 2026. Ridge, LightGBM and a ReLU MLP train from scratch on the
original full universe, original matched coverage and replacement matched
coverage. Covered model predictions use the original full model's rank slots.
The PDF and CSV tables report absolute full-universe performance, paired
matched comparisons, and the coverage diagnostics. Raw source manifests,
feature specifications, cut-date checks, parameter picks and forecasts are in
`results/volatility_replacement`.

The replacement runner accepts `--mlp-epochs` and `--mlp-device cuda`.
The epoch count applies to each MLP fit, including inner search candidates
and each final refit seed. The CUDA path uses a fixed PyTorch training loop
and runs folds serially on the selected GPU. The CPU path advances the
scikit-learn MLP by one epoch per call.
