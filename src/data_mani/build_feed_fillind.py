"""
Rebuild a fillind dataset on the extended spot file, keeping only coins we have
prices for: <source> -> <source>-feed-fillind.

    python3 -m src.data_mani.build_feed_fillind --source-version crypto/v2.0-litmom
    python3 -m src.data_mani.build_feed_fillind --source-version crypto/v2.0-momswap
    python3 -m src.data_mani.build_feed_fillind --source-version crypto/v2.0-momswap-macro-carver

What changes against <source>-fillind:

  1. feature_mom_ / feature_tf_ are recomputed from the CURRENT spot file, so coins
     added since the source was built get real values. Each block goes through the
     same compute -> as-of attach (D-1, max 4 days stale) -> per-date bin as
     build_momentum / build_trend.
  2. Coins with no price on ANY of their panel rows are dropped.
  3. NaN -> 0.5 plus feature_isna_* indicators, as build_mommask --mode fillind.

The spine is the SOURCE dataset, not crypto/v2.0: that file is live-refreshed past
the 2026-08-20 end of most spot series, so re-running the builders on it would add
weeks of rows with an empty engineered block. Using the source also keeps the rows,
the shipped features and the macro block identical to what earlier studies saw.

"Missing" is judged per coin (symbol + ucid), on what the as-of join actually
attaches, not by whether the ticker is in the spot file: a reused ticker can be
present while one of its coins has no prices on its own dates (SHARE ucid 6868,
2020-21, against a 2026 series). A coin with prices but too little history for any
window is KEPT: that is warm-up, which is known at the time. Binning happens before
the drop, and a no-price coin is NaN so it never entered a rank; the shipped
features keep their bins over Numerai's full universe.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.binning import bin_cross_sectional
from src.data_mani.build_mommask import apply_fillind
from src.data_mani.momentum_features import (MOMENTUM_SPECS, attach_signals_asof,
                                             compute_momentum_signals)
from src.data_mani.trend_signals import TREND_SPECS, compute_trend_signals
from src.data_mani.utils import (download_if_missing, get_api, get_spec,
                                 local_path)

SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")
LAG, STALE = 1, 4
BLOCKS = (("feature_mom_", MOMENTUM_SPECS, compute_momentum_signals),
          ("feature_tf_", TREND_SPECS, compute_trend_signals))
PRICE = "_price"


def recompute_blocks(panel, spot, period="date", symbol="symbol"):
    """Recompute every engineered price block present in ``panel`` from ``spot``
    (symbol, date, close) and attach ``PRICE``, the as-of close each row sees.
    Returns the panel in its original row order and column order, plus PRICE."""
    order = list(panel.columns)
    out = panel.drop(columns=[c for p, _, _ in BLOCKS for c in panel if c.startswith(p)])
    for prefix, specs, compute in BLOCKS:
        cols = [c for c in order if c.startswith(prefix)]
        if not cols:
            continue
        unknown = sorted(set(cols) - set(specs))
        if unknown:
            raise SystemExit("no spec for %s" % unknown)
        raw = compute(spot, {c: specs[c] for c in cols}).drop_duplicates(["symbol", "date"])
        out = attach_signals_asof(out, raw, cols, lag_days=LAG, max_staleness_days=STALE,
                                  period_col=period, symbol_col=symbol)
        out = bin_cross_sectional(out, cols, period)
    px = spot.rename(columns={"close": PRICE})[["symbol", "date", PRICE]]
    out = attach_signals_asof(out, px.drop_duplicates(["symbol", "date"]), [PRICE],
                              lag_days=LAG, max_staleness_days=STALE,
                              period_col=period, symbol_col=symbol)
    if len(out) != len(panel):
        raise SystemExit("as-of merge changed the row count %d -> %d" % (len(panel), len(out)))
    return out[order + [PRICE]]


def coins_without_prices(df, coin=("symbol", "ucid")):
    """Boolean row mask: True on every row of a coin that has no PRICE on any row."""
    return ~df[PRICE].notna().groupby([df[c] for c in coin]).transform("any")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", required=True)
    p.add_argument("--out-suffix", default="-feed-fillind")
    p.add_argument("--spot", default=SPOT)
    return p.parse_args()


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")
    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    base["ucid"] = base["ucid"].astype(str)
    print("source: %d rows, %d symbols, %d features (%s)"
          % (len(base), base["symbol"].nunique(),
             sum(c.startswith("feature_") for c in base), args.source_version))

    spot = pd.read_parquet(args.spot, columns=["symbol", "date", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    out = recompute_blocks(base, spot, spec["period_col"], spec["id_col"])

    gone = coins_without_prices(out)
    has = out[PRICE].notna()
    kept = out.loc[~gone]
    print("dropping %d coins with no price on any panel row: %d rows (%.1f%%), "
          "%d symbols" % (out.loc[gone, ["symbol", "ucid"]].drop_duplicates().shape[0],
                          gone.sum(), 100 * gone.mean(), out.loc[gone, "symbol"].nunique()))
    print("kept coins: %d rows, of which %d (%.1f%%) have no price on their own date "
          "(before the feed starts / after it ends)"
          % (len(kept), (~has & ~gone).sum(), 100 * (~has & ~gone).sum() / len(kept)))

    feats = [c for c in kept.columns if c.startswith("feature_")]
    n_missing = int(kept[feats].isna().sum().sum())
    final, inds = apply_fillind(kept.drop(columns=[PRICE]), feats)
    print("filled %d missing values (%.2f%% of cells); %d feature_isna_* indicators"
          % (n_missing, 100 * n_missing / (len(kept) * len(feats)), len(inds)))

    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    final.set_index("symbol").to_parquet(dest)
    print("wrote %d rows, %d symbols, %d features -> %s (%.0f MB)"
          % (len(final), final["symbol"].nunique(), len(feats) + len(inds), dest,
             os.path.getsize(dest) / 1e6))
    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)
    print("run it with:  --data-version %s" % version)


if __name__ == "__main__":
    main()
