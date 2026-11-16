---
name: crypto-price-data
description: "Source external spot / OHLCV price data for the Numerai Crypto symbols over the dataset's span, keyed by ucid (which IS the CoinMarketCap coin id). Use when fetching prices, returns, volume, market cap or any market data for the coins in data/crypto/v2.0, reconstructing the raw signals the binned features hide, joining an external feed to the panel, or preparing the input to a derived-feature build. Triggers on 'get price data', 'download OHLCV', 'fetch spot prices', 'raw returns for the coins', 'map symbols to CoinMarketCap/CoinGecko', 'external market data', 'reconstruct momentum from prices'."
---

# External price data for Numerai Crypto symbols

## The join key: ucid = CoinMarketCap id

`data/crypto/v2.0/train.parquet` carries a `ucid` column that **is the CoinMarketCap coin
id** — BTC=1, LTC=2, XRP=52, DOGE=74, XMR=328, ETH=1027. It is the reliable identifier;
the `symbol` ticker is ambiguous (≈30 symbols map to more than one ucid over time, and
tickers get reused/renamed). Join external data on `ucid` where correctness matters; a
provider that only knows tickers (Yahoo, Binance) forces a symbol join — flag that and
prefer ucid downstream.

The base panel: **2020-01-01 … 2026-08-20**, 1,732 **weekday** dates (Mon–Fri; crypto
still trades weekends, but the panel does not), ≤300 coins/date, **1,331 unique symbols**,
22 features all pre-binned to `{0,.25,.5,.75,1}` (see [crypto-feature-binning]) plus targets
`target_binned_return_20` (payable) and `_60`. The raw prices are NOT in the Numerai data —
that is why you source them.

## Sourcing: a no-key cascade (as of 2026-09)

Full 2020→2026 daily history with no API key is only feasible via **yfinance → Binance**:

- **yfinance** `<SYM>-USD` (Yahoo). Strong on majors, full OHLCV, includes weekends. Strip a
  leading `$` and try both spellings (`$GALA` → `GALA`) or memecoin tickers all miss.
- **Binance** `<SYM>USDT` klines (`api.binance.com/api/v3/klines`, no key, paginate 1000/day)
  fills alts Yahoo lacks. Reachable here; the `data-api.binance.vision` mirror is not.
- **CryptoCompare** now returns **401** (key-gated). **CoinGecko** free caps history at the
  last **365 days** — both unusable for the full span.

Realistic coverage is **~81% of all 1,331 symbols, ~86% of the active 300-coin universe**;
the long tail (DEX memecoins like BRETT/APEPE/BabyDoge) has no free full-history feed and
needs a keyed provider (CoinGecko Pro, CoinMarketCap, a DEX aggregator). A single-threaded
retry recovers ~0 of the misses — they are genuinely absent, not throttled.

Fetch with a **threaded, checkpointed, resumable** script (one part-file + coverage row per
symbol, skip on resume); Yahoo's batch endpoint times out on bulk, so go per-symbol. The
reference implementation is `scratchpad/fetch_spot.py` from the session that built
`data/crypto/spot/`.

## Where it lands

`data/crypto/spot/` (gitignored, disposable, rebuildable):

- `ohlcv.parquet` — `symbol, ucid, date, open, high, low, close, volume, source,
  provider_ticker` (source = yahoo|binance). ~1.73M rows.
- `coverage.csv` — per symbol: matched?, source, row count, first/last date, `pct_data_dates`.
- `missing.csv` — the symbols with no free feed (active coins first).
- `README.md` — provenance + caveats.

## Using it — two caveats that matter

- **Truncation ≠ gap.** A series starting after 2020-01-01 usually means the coin listed
  later (SOL 2020-04-10, GALA 2021-09-13 on Binance), not missing data.
- **Point-in-time.** OHLCV includes weekends and the day-D bar is not final at round-D
  submission. When merging onto the panel, attach the freshest close dated ≤ **D−1** — do
  NOT exact-join on D. This is the whole point of `momentum_features.attach_signals_asof`;
  see [crypto-derived-features].

## Interpreter

Reading the parquet needs pyarrow. Bare `python3` may be the 3.8 sandbox without it — use
`/home/danielco/miniconda3/bin/python3` (3.12, pyarrow + pandas 2.x) and verify first.
