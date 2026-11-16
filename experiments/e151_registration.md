# E151 -- volume alpha: does point-in-time information about expected trading volume and liquidity add marginal value on Numerai Crypto?

Registered 2026-11-12, before any feature was built, any arm fitted or any score read. The dataset is
`crypto/v2.0-volumealpha-causal`. It extends `crypto/v2.0` and leaves every shipped column and target
untouched.

## Hypothesis and what each test can say

The source is "Trading Volume Alpha", which predicts volume in order to EXECUTE better, not to predict
returns. Four things are kept apart:

1. **Raw predictors of future volume.** These are the technical block, the extensions, fundamentals,
   events and cross-asset volume (arms 2-6, 9).
2. **Out-of-sample volume forecasts.** These are built from the raw predictors (arm 7).
3. **Trading-cost proxies.** These are derived from the forecasts (arm 8).
4. **The paper's own use, execution.** Numerai's score never sees the submitter's trading cost, so this
   study does not test it.

A volume model that forecasts well is reported as a **volume / execution / capacity result**. It counts
as return alpha only if it also improves routed Numerai CORR or MMC on the final block. A null on CORR
and MMC does not reject the paper.

## Sources (feasibility measured 2026-11-12, free and no key only)

| family | source | identity | point-in-time status |
|---|---|---|---|
| Binance spot venue volume | data.binance.vision daily klines, every quote, **delisted pairs included** | ticker match, then per-row identity gate | timestamp-causal, archive retrospective |
| other CEX venue volume | gate, okx, kucoin, bybit, bitget, coinbase, mexc, hyperliquid REST | ticker match, then gate | **currently listed pairs only** (survivorship) |
| aggregator volume | Yahoo `<SYM>-USD` volume (existing spot store) | existing ticker/ucid join | retrospective reconstruction |
| perp volume, OI, funding, long/short | Binance USD-M archive klines, daily `metrics` (OI from 2021-12), fundingRate | ticker match, then gate | timestamp-causal, includes delisted contracts |
| size, supply, MVRV, active addresses, tx count | Coin Metrics community API (about 140 assets) | CM PriceUSD must track the coin's own price | **revised, not vintage** |
| TVL, fees, revenue, DEX volume of a protocol or chain | DefiLlama, joined by `cmcId` = panel `ucid` | permanent id | **revised, not vintage** |
| governance votes | Snapshot (`created`, `start`, `end`), spaces from DefiLlama `governanceID` | permanent id via DefiLlama | **point-in-time** (proposals are immutable and carry a creation time) |
| BTC/ETH options and CME futures expiries | deterministic calendar (last Friday; quarterly in Mar/Jun/Sep/Dec) | global | known in advance |
| listings and delistings | first and last archive kline per venue | per venue | post-event only (announcement times unavailable) |
| stablecoin USD rates | Yahoo USDT/USDC/BUSD/FDUSD/TUSD/DAI/USDP-USD | -- | retrospective |

**Named gaps.** These are not built, because no free source exists for them, or none with announcement
timestamps:

- per-token DEX trading volume (GeckoTerminal's free tier keeps 180 days);
- token unlocks and vesting releases, and scheduled emission changes (DefiLlama emissions is paid);
- protocol upgrades and hard forks, airdrop snapshots and claims, mainnet and product launches;
- crypto-index rebalances;
- liquidation volume (the archive's `liquidationSnapshot` prefix is empty);
- realized cap for most assets (Coin Metrics paywalls `CapRealUSD`, so MVRV is used where it is free);
- protocol or token launch dates (DefiLlama's `listedAt` is a listing date, not a launch date).

**Contract addresses.** Identity is by the panel's permanent `ucid`, which is the CoinMarketCap id. It
is joined by id to DefiLlama, which also records the contract `address`, kept for audit. Exchange series
are matched by ticker, so every exchange series must pass the causal identity gate on each row before
it counts (vol-causal rule: expanding mean gap between the per-date percentile of rebuilt 20/60-day
volatility and the shipped `feature_volatility_20d/60d`, at most 0.18). A venue series must also stay
within 25% of the cross-venue median close on that day, or that pair-day is dropped.

## Timing

- **Cutoff.** For Numerai date D, every input is dated D-1 or earlier: a daily bar for D-1 closes at
  00:00 UTC on D, and the Binance `metrics` file for D-1 is published by 08:19 UTC on D, before the
  12:00 UTC round open. Attachment is as-of D-1 with at most 4 days of staleness (vol-causal).
- **Rolling statistics.** Every rolling statistic is computed on a series dated D-1 or earlier. There is
  no backfill. A value that is not there stays NaN.
- **Revised series.** Coin Metrics and DefiLlama series are additionally lagged 2 days (D-2), because
  their daily points settle after the day ends.

## Dollar volume

```
dollar_volume(venue, pair, t) = base_volume(t) x close(t) x USD(quote, t)
```

- **Quote conversion.** `USD(quote)` is 1 for USD and the Yahoo rate for each stablecoin. For BTC, ETH
  and BNB quotes it is that coin's Binance USDT close times the USDT rate. The venues' own
  quote-volume fields are not used.
- **No double counting.** Different quote pairs on one venue are different order books, so they are
  summed. Binance appears only from the archive. The Yahoo aggregator figure is never added to the venue
  sum, and perp volume is never added to spot.
- **Source families.** `bin` (Binance spot), `cex` (the other eight venues), `ven` = `bin` + `cex`, and
  `agg` (Yahoo). DEX volume is a named gap.
- **Primary series.** The primary `logdv = log(1 + dollar_volume_ven)`. The other families are the
  source ablations.
- **Venue structure.** Venue count and venue concentration (HHI of venue shares) come from the `ven`
  family.

## Blocks (each built continuously, then binned per date)

All windows are in calendar days, because crypto trades 7 days a week.

### TECH (8 columns, the paper's technical block)

- lagged moving averages over 1, 5, 22 and 252 days of daily return;
- the same moving averages of logdv.

### EXT (extensions)

Never mixed into TECH.

- shock5 and shock22: logdv minus its trailing 5- or 22-day mean;
- logdv growth over 1, 5 and 22 days, and acceleration;
- volatility of the daily logdv change over 22 days;
- Amihud illiquidity: the 22-day mean of |r| / dollar volume;
- range / dollar volume;
- venue-count change and HHI change over 22 days;
- 22-day correlation of the return with the logdv change (volume-price divergence).

Turnover (dollar volume / market cap) goes in FUND_SIZE, because market cap is a revised Coin Metrics
series. Spot-volume market share is rank-identical to logdv within a date, so only its 22-day change is
built, in XASSET.

### FUND (crypto analogues of the small fundamental block)

Separable sub-families, each with its own coverage:

- **SIZE.** Coin Metrics market cap; FDV = price x `SplyExpFut10yr`; turnover.
- **SURPRISE.** For protocol fees, protocol revenue, transaction count, active addresses and DEX volume:
  `(log(1 + 7-day sum) - its trailing 90-day mean) / its trailing 90-day standard deviation`. Every
  input is lagged.
- **LEVERAGE.** OI value / market cap; OI value / spot dollar volume; the 7-day mean funding rate; the
  7-day change in OI. Liquidations are a named gap.
- **VALUE.** Market cap / TVL; market cap / (365 x 30-day mean fees); market cap / (365 x 30-day mean
  revenue); MVRV.
- **BETA.** Dimson beta (contemporaneous plus one lag) over 90 days to BTC, to ETH, and to a
  leave-one-out market index (lagged-dollar-volume-weighted mean return of every other covered coin).
- **AGE.** Days since the first gated price observation; days since the first gated venue-volume
  observation; days since the Binance spot listing. Age is tenure, and tenure was a confound in E131;
  it is reported as such.

### EVENT

- **Coin-specific, point-in-time.**
  - Snapshot vote end. Distance categories ≤-4, -3, ..., +4, ≥+5 in calendar days. An upcoming vote
    counts only once its proposal was created before D's cutoff.
  - Binance spot and perp listing, and delisting. Post-event only: 0..+4 and ≥+5. Upcoming listings are
    never encoded, because announcement times are unavailable.
  - Each category is a 0/1 column. A coin with no governance space has NaN, not 0.
- **Global events, only as interactions with a coin exposure.**
  - expiry window x BTC beta;
  - expiry window x OI / dollar volume;
  - expiry window x the coin's historical volume-event beta (its mean logdv shock on past expiry days);
  - D is Monday x the coin's weekend-volume sensitivity (trailing 252-day mean of weekend logdv minus
    weekday logdv).
  - No unmodified global column is shipped.

### XASSET (cross-asset volume)

Every market-wide scalar enters only multiplied by a coin sensitivity estimated over 90 days. The
scalars are:

- total market logdv shock;
- BTC and ETH shocks;
- the statistical-sector shock and its leave-one-out version;
- breadth (the fraction of the universe with a positive shock).

Coin-level columns:

- lead-lag sensitivity: correlation of the coin's logdv change with BTC's and ETH's lagged logdv change;
- 22-day change in the coin's share of market volume.

Statistical sectors are k-means (k = 8) on the trailing 180-day return correlation, re-estimated at each
walk-forward block start from data before it. There is no free sector taxonomy that covers the panel.

### OOF (volume forecasts)

Out-of-fold only. The primary auxiliary target is

```
eta(i,t) = logdv(i,t) - mean(logdv(i,t-1..t-5))
```

- **Separate direct targets.**
  - 5-day: mean logdv over t..t+4 minus the trailing mean;
  - challenge horizon: mean logdv over t+2..t+30 minus the trailing mean.
  - None is a sum of one-day forecasts.
- **Models.** Ridge, LightGBM, sklearn MLP, and the paper net (32/16/8).
- **Training.** Inputs are the TECH, EXT and XASSET raw predictors plus the day of week.
  - Refits are quarterly on an expanding window. Each refit uses only rows whose target is fully
    observed before the quarter starts: 1, 5 and 30 days of purge for the three targets.
  - Training rows are uniformly subsampled to at most 600k.
- **Features.**
  - `hat_eta_1d`, `hat_logdv_1d`, `hat_logdv_5d`, `hat_logdv_ch`;
  - P(logdv < causal 10th percentile) and P(logdv > causal 90th percentile), from LightGBM classifiers;
    the percentiles are the coin's own trailing 252 days;
  - uncertainty (q90 - q10 of LightGBM quantile models);
  - disagreement (standard deviation across the four models);
  - the previous realized error (eta at t-1 minus its forecast, known at D).
- No in-sample prediction ever enters the panel.

### COST (trading-cost proxies)

From `hat_dv = exp(hat_logdv_1d) - 1` and the lower-quantile forecast:

- inverse predicted dollar volume;
- participation for trade sizes of $10k, $100k and $1M, as three separate families, capped at 1;
- `lambda_hat = 0.2 / hat_dv`;
- low-liquidity risk (the P(< q10) above);
- capacity `hat_dv / mcap`;
- expected Amihud (trailing 22-day mean |r| / hat_dv);
- the conservative q10 forecast;
- a downside-adjusted forecast (`hat - (q50 - q10)`).

**Stated before any result:** inverse volume, lambda and uncapped participation are monotone in `hat_dv`,
so per-date rank-binning makes them identical to the rank of `hat_dv`. The build checks which binned
columns coincide, and the arm keeps one copy of each distinct column, with the duplicates named in the
report.

## Processing

1. Causal transform.
2. Winsorize at expanding-history 1% / 99% limits: quantiles of all values dated before the row's date,
   with at least 250 dates.
3. Rank across the dated universe.
4. Bin with Numerai's rule (`history_features.bin_ties_preserved`), which preserves ties.
5. Leave missing values as NaN.

No missingness or coverage indicator is a feature. Coverage masks are non-feature `cov_*` columns.

## Arms

| # | arm | features |
|---|---|---|
| 1 | `B_COV` | v2.0 22, trained on the arm's covered rows |
| 2 | `TECH` | +8 |
| 3 | `EXT` | + extensions |
| 4 | `FUND` | + BETA and AGE (core), with optional sub-families SIZE, SURPRISE, LEVERAGE and VALUE filled 0.5 inside the mask |
| 5 | `EVENT` | + event block |
| 6 | `XASSET` | + cross-asset block |
| 7 | `OOF` | + forecasts |
| 8 | `COST` | + distinct cost proxies |
| 9 | `RAW` | + 2-6 |
| 10 | `MODEL` | + 7-8 |
| 11 | `PARSIM` | dev-selected |
| -- | `TECH_BIN`, `TECH_CEX`, `TECH_AGG` | source ablations, each on its own mask with its own `B_COV` |
| -- | `F_SIZE`, `F_SURPRISE`, `F_LEVERAGE`, `F_VALUE` | sparse sub-families on their own masks with their own `B_COV` |
| -- | `P_HIST` | tenure control |
| -- | `P_WARM` | share of the block observed, binned |
| -- | `RAWvsMODEL` | `RAW` against `MODEL` routed, the two-stage question |

- **Main mask.** Arms 1-11 share one common mask M: TECH (ven) observed and OOF observed, as in E150's
  single matched mask. Sparse optional families are 0.5 inside M, the only value the loader uses.
- **Masks of the sub-family and source arms.** These are M_x = M_x-coverage, with that family's columns
  required. Each gets its own `B_COV` and its own routing.

## Evaluation (frozen)

### Auxiliary (volume models)

- out-of-sample R² against the trailing-5-day-mean baseline (eta = 0);
- MAE;
- lower-tail calibration (the share of outcomes below q10);
- by shipped market-cap bin and by dollar-volume tercile;
- through stress windows: 2020-03, 2021-05, 2022-05, 2022-11, 2024-08 and 2025-10.

### Numerai

- **Protocol (E133/E150).**
  - walk-forward blocks 130/22/130;
  - target `target_binned_return_20`;
  - fixed configs ridge (there is no `ridge_ktz`; ridge needs none), lgb_ktz (the "gbm_ktz") and
    mlp_ktz;
  - 5 seeds for lgb and mlp;
  - an equal-budget retune for `B_COV`, `RAW`, `MODEL` and `PARSIM`.
- **Deployment score.** R_SLOT routing through `B_ALL`, the reused v2.0 B0: covered coins are reordered
  inside their v2.0 rank slots, and uncovered coins keep their exact v2.0 ranks.
- **Comparisons.** Routed arm against routed `B_COV` on the same mask.
- **Statistics.** Newey-West lag 20, block bootstrap, proxy MMC.
- **Reported.** win rate, worst block, coverage, results by market-cap bin and liquidity tercile, and the
  final block.
- **Bar.** Per model:
  - **ADOPT:** dev routed CORR t ≥ 2, test the same sign, and P_HIST / P_WARM do not match it;
  - **REJECT:** dev CORR t ≤ -2, or a placebo explains the gain;
  - **PROMISING BUT UNCONFIRMED:** dev positive, not significant, with a positive test;
  - otherwise **NO IMPROVEMENT**.
- **PARSIM rule (dev only).** A block qualifies when its arm beats `B_COV` on routed dev CORR at t ≥ 2
  for at least 2 of the 3 models, with dev payout t > -2 on all three. PARSIM = every qualifying block.
  If none qualifies, there is no PARSIM.
- **Windows.** The final block (2026-02-20..08-20) was scored by earlier studies, so it is not virgin.
  Labelled dates after 2026-08-20 stay unread for a forward test.

### Frozen panel

`data/crypto/v2.0-asof-e151` (a copy of v2.0 taken at build time). `data/crypto/v2.0` is rewritten
daily.
