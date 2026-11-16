# Strict volume-alpha dataset

The user requests a historical point-in-time study, not just timestamp-causal
features reconstructed today. This implementation keeps that distinction in
the source contract. Existing `fetch_volalpha_binance`, `fetch_volalpha_cex`
and `fetch_volalpha_onchain` are retained without modification.

The Binance cache is `/scratch/danielco/volalpha/binance/`; the older spot cache
is `data/crypto/spot/`. An archive economic date and a current retrieval date
do not establish what vintage was published historically. The old archive
reader also discards original close timestamps. The newer reader retains
retrieval time and archive object, but not vintage publication or permanent
asset identity. These caches are useful for a separately labeled reconstruction
experiment; they cannot silently satisfy this strict contract.

## Commands

Use the environment containing pandas, pyarrow, sklearn, lightgbm and
numerai-tools, normally `/home/danielco/miniconda3/bin/python3`.

```bash
python3 -m src.data_mani.build_volumealpha_causal --audit-local
python3 -m src.data_mani.build_volumealpha_causal --inputs /path/to/verified_bundle
python3 -m src.analysis.volumealpha_evaluation --phase dev
python3 -m src.analysis.volumealpha_evaluation --select
# Only after the registered forward labels have resolved:
python3 -m src.analysis.volumealpha_evaluation --phase test
```

Default output is `data/crypto/v2.0-volumealpha-causal`. Existing nonempty
outputs are never overwritten. The builder copies every v2.0 column and row
exactly, preserving the original index and dtypes; it copies historical meta
models and stores hashes of the frozen base and source inputs. The manifest
names unavailable feature families and reports coverage. A missing input bundle
or no admissible source rows is an explicit failure, not an all-NaN success.

Source sensitivity: use separate bundle allowlists for Binance-only and other
CEX venues. Run `--source-family dex --output data/crypto/v2.0-volumealpha-causal-dex`
for an admitted DEX archive, or `combined` with a separate output. Never sum
aggregator volume or derivatives volume into spot. A Binance-only result is
single-venue CEX evidence, not consolidated market-volume evidence. Quote/USD
conversion must be independently evidenced, even for stablecoins.

## Verified input bundle

`sources.json` supplies `{"trusted_venues": ["binance"]}` and human-readable
source descriptions / evidence locations. Admission is an assertion by the
source preparer backed by archived evidence, not proof conferred by a boolean.
The code cannot authenticate the contents of a vendor receipt on its own.

All tables carry:

| Column | Meaning |
|---|---|
| `available_at` | UTC publication timestamp of this exact value/schedule vintage |
| `retrieved_at` | Actual UTC retrieval timestamp; never backdated to the economic day |
| `evidence` | Nonempty reference to the archived receipt/publication/revision record |
| `vintage_verified` | Boolean; false rows are never admitted into strict snapshots |

Revisions use separate rows and their real publication timestamps. For an
economic key, the most recent verified vintage available by that prediction's
cutoff is selected. A modern revised value cannot use its original observation
date as its availability timestamp.

### identities.parquet — mandatory

`ucid` (Numerai/CMC ID), `asset_id`, `chain`, `contract_address`, `valid_from`,
`valid_to`, plus provenance. `asset_id = chain + ':' + contract_address`.
For native coins use e.g. `bitcoin:native`, `ethereum:native`. Use a canonical,
case-aware address representation appropriate to each chain. Validity is
half-open `[valid_from, valid_to)`; a native token migration or wrapper is a
distinct identity. Include evidence for the historical mapping, not a present-day
ticker lookup. Mapping intervals for one ucid must not overlap. An optional
`sector` is usable only under the same dated evidence contract. There is no
ticker fallback or price-correlation identity gate.

### market.parquet — mandatory

`asset_id`, `venue`, canonical native `pair` identifier, UTC midnight `date`,
raw `observed_at`, `source_family` (`cex` or `dex`), `market_type` (`spot`),
`base_volume`, `price_quote`, `quote`, `quote_usd`, `quote_fx_at`,
`quote_fx_available_at`, `quote_fx_evidence`, plus provenance.

One row per venue/pair/day/vintage; duplicate rows are rejected. The adapter
must canonicalize venue/pair aliases and deduplicate mirrors. CEX/DEX records
are disjoint source families. Dollar volume is computed as
`base_volume * price_quote * quote_usd`; stablecoins are not assumed to be $1.
Daily close × base volume is explicitly a daily-bar notional proxy, not the
sum of trade-level notionals. For exact traded USD notional supply sufficiently
fine observations upstream and document aggregation; do not claim exactness
from a daily close. Optional `high_usd` / `low_usd` support the range proxy.

Venue count is the number of unique venues, not the number of quote pairs.
Concentration is HHI of venue USD-volume shares. Distinct order books may be
summed once each. Global quantities use only observed then-known Numerai
membership; no complete-market-census claim is made.

### metrics.parquet — optional, independently covered

`asset_id`, economic UTC midnight `date`, `metric`, `value`, plus provenance.
Metric names supported:

`market_cap`, `fdv`, `fees`, `revenue`, `transactions`, `active_addresses`,
`dex_volume`, `open_interest`, `funding`, `liquidations`, `tvl`, `realized_cap`,
`launch_unix_seconds`.

USD notional values are required for market cap, FDV, daily fees/revenue,
DEX volume, open interest, liquidations, TVL and realized cap. Daily flows
must not be supplied as cumulative totals. Funding is the provider's
documented aggregate rate, not a sum over duplicated perpetual contracts.
FDV must be an actual FDV; a ten-year projected supply is not silently FDV.
Launch time needs historical evidence; exchange listing is not protocol launch.
Surprise = (latest known daily observation − previous 22-day mean) / previous
22-day std. Valuation annualizes 30 calendar days of daily fees/revenue by 365.
Missing optional inputs disable only their covered arms.

### events.parquet — optional

`event_id`, `asset_id` (or `global`), `kind`, `scheduled_for`, `announced_at`,
plus provenance. Each schedule revision must have its own `available_at`.
Upcoming events appear only after public announcement; first/last bars are not
announcements, and a present-day deterministic expiry calendar is not historical
evidence of every exception. Supported kinds are listed in
`volumealpha_features.EVENTS` and `GLOBAL_EVENTS`.

Coin-event distances are clipped to categories -4..5, meaning ≤−4 and ≥+5
at the ends. Global event dummies are never shipped unmodified: they enter
only through beta or open-interest interactions. Historical weekend-volume
sensitivity is another per-coin interaction. Missing catalogs stay missing.

## Output and evaluation boundaries

- `train.parquet`: all original base columns plus binned `feature_va_*` only;
  unavailable values remain NaN. Use the dedicated evaluator because the generic
  runner neutral-fills missing features and does not enforce covered-only fits.
- `raw_signals.parquet`, `continuous_panel.parquet`: continuous signals before
  and after causal historical winsorization; no target values as features.
- `coverage.parquet`, `identity_keys.parquet`: sidecars, never production features.
- `volume_targets.parquet`, `volume_oof_predictions.parquet`, `volume_fits.json`:
  auxiliary targets, genuine historical predictions, fit/publication boundaries.
- `volume_oof_metrics.csv`: R² vs trailing-five-day baseline, MAE, lower-tail
  coverage, Brier scores, asymmetric loss; splits by dated cap/liquidity and
  predeclared stress months. Missing OOF forecasts produce missing results.
- `manifest.json`: hashes, definitions, causal coverage and named unavailable arms.

The reference snapshot engine recomputes each cutoff's exact vintage, including
revision history. It is intentionally conservative and is not optimized for a
multi-million-row full daily rebuild. Compare truncated-source builds on actual
data before interpreting a result; the source audit currently blocks that step.

The evaluator uses all three requested model families under their actual repo
names, equal-budget searches, matched covered samples and full-universe
slot-preserving routing. The explicit forward block begins 2026-11-13 and needs
mature labels. Five seeds and date-paired HAC/bootstrap are required; infer
anything about return alpha only from that return evaluation, not from volume
R² or modeled transaction costs. Full definitions and the frozen dev-selection
rule are in `experiments/e152_registration.md`.

## Research sources

- [Trading Volume Alpha, NBER working paper 33037](https://www.nber.org/system/files/working_papers/w33037/w33037.pdf):
  volume forecasts inform portfolio execution and net-of-cost performance.
- [Binance public-data documentation](https://github.com/binance/binance-public-data/blob/master/README.md):
  archive formats, quote/base volume fields and spot timestamp units. Archive
  structure does not certify the historical vintage of a modern download.


## User-authorized inferred reconstruction (2026-11-13)

The user subsequently authorized alternate sources and inferred provenance. Run
`python3 -m src.data_mani.build_volumealpha_inferred` to materialize the main
`crypto/v2.0-volumealpha-causal` dataset and Binance-only / other-CEX / direct
Coinbase-USD-conversion source ablations. This path retains every base row and
column, exposes the admitted coverage in sidecars, and keeps the original
strict builder's vintage proof requirement intact.

The main manifest status is `BUILT_INFERRED_RECONSTRUCTION`, with
`point_in_time_verified=false`. Two calendar days of information lag implement a
conservative inference from Binance's next-day publication policy; daily versus
monthly archive publication is not confused. Modern revisions remain possible.
Coinbase direct measured USD candles are preferred over the existing measured
Yahoo USD series; a missing conversion is never filled with a $1 stablecoin peg.
31 reused ticker histories are excluded before canonical ID assignment. The
remaining candidate relations use permanent CMC IDs, native chains/current
contract metadata where possible, and a causal volatility-based identity
cross-check; this is explicitly inferred historical identity, not proof.

The real source evidence and exact retrieval timestamps reside in
`/scratch/danielco/volumealpha_e152/`, including unmodified-source references,
independent USD response JSONs, HTTP publication metadata samples, canonical
identity candidates, and `venue_observations.parquet`. These source histories are
reconstructed, so no historical-vintage claim is made for the broad arm.

OOF models fit original eta and direct one-day/five-day/twenty-calendar-day
log-dollar-volume targets. Direct level forecasts avoid using the unobserved D-1
volume mean when the information cutoff is D-2. Expanding refits are every 90
calendar days, at most 50,000 deterministic sampled training rows, seed 152, 50
iterations, ridge / GBM / 32-16-8 MLP; lower-tail models and publication-time
purges remain in place. These choices were registered before fitting real data.

An evaluation of these reconstructed datasets requires an explicit flag:

```bash
python3 -m src.analysis.volumealpha_evaluation --phase dev --allow-inferred
```

This carries the reconstruction limitations in the report. It does not silently
upgrade a reconstructed dataset to strict status. Missing event catalogs,
liquidations, FDV, launch dates and per-token DEX volume remain named gaps, not
fabricated columns. Revised fundamental families remain separable.

After the build, run
`python3 -m src.analysis.validate_volumealpha_dataset` to check the original
parquet exactly, allowed bins, every saved OOF publication boundary, and
independent technical recomputation on five real-source truncation dates. The
build report and validation JSON are under
`results/reports/crypto/e152_volumealpha_causal_build/`.
