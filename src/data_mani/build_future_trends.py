"""
Build crypto/v2.0-future-trends: a LOOK-AHEAD sanity-check dataset. Never deploy it.

    python3 -m src.data_mani.build_future_trends

Adds one column, feature_future_ret_20d, each coin's NEXT 20-day spot return:

    close(D + 20 calendar days) / close(D) - 1

read from data/crypto/spot/ohlcv.parquet by symbol, then rank-binned within each date onto
Numerai's {0,.25,.5,.75,1} (history_features.bin_ties_preserved, the exact shipped rule). A
row whose close on D or D+20 is missing stays NaN, which the loader fills with 0.5. Nothing is
forward-filled. The 22 shipped features and the targets are copied unchanged.

The column reads prices up to 20 days AFTER the row's date, so it is the answer, not a feature.
It exists to check that the evaluation machinery flags a block that knows the future: it should
win by a margin no honest block has come near. It has no live path, and src/live must never
offer it.

The 20-day target is not this window. On the panel, the per-date Spearman correlation of
close(D+a)/close(D+b) with target_binned_return_20 peaks at a=2, b=30 (0.88), against 0.59 for
the D..D+20 window built here: the target looks like a 20-trading-day return starting two days
after the row's date.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.history_features import bin_ties_preserved
from src.data_mani.utils import download_if_missing, get_api, local_path

SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")
COLUMN = "feature_future_ret_20d"


def future_return(panel, spot, horizon=20, symbol="symbol", period="date"):
    """close(D + horizon days) / close(D) - 1 per panel row, NaN where either close is missing."""
    px = spot.loc[spot["close"] > 0].drop_duplicates([symbol, period]).set_index([symbol, period])["close"]
    now = px.reindex(pd.MultiIndex.from_arrays([panel[symbol], panel[period]])).to_numpy()
    later = px.reindex(pd.MultiIndex.from_arrays([panel[symbol], panel[period] + pd.Timedelta(days=horizon)])).to_numpy()
    return pd.Series(later / now - 1.0, index=panel.index)


def parse_args():
    p = argparse.ArgumentParser(description="Build the look-ahead sanity-check dataset.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0")
    p.add_argument("--out-suffix", default="-future-trends")
    p.add_argument("--spot", default=SPOT)
    p.add_argument("--horizon", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    api = get_api("crypto")
    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    df = pd.read_parquet(src).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    spot = pd.read_parquet(args.spot, columns=["symbol", "date", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    print("source: %d rows, %d symbols, %s .. %s" % (len(df), df["symbol"].nunique(),
                                                     df["date"].min().date(), df["date"].max().date()))

    raw = future_return(df, spot, args.horizon).to_frame(COLUMN)
    print("LOOK-AHEAD: %s reads closes up to %d days after each row" % (COLUMN, args.horizon))
    print("observed on %.1f%% of rows" % (100 * raw[COLUMN].notna().mean()))
    df[COLUMN] = bin_ties_preserved(raw, df["date"])[COLUMN].astype("float32")

    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    df.set_index("symbol").to_parquet(dest)
    n_feat = sum(c.startswith("feature_") for c in df.columns)
    print("features: %d total -> %s (%.0f MB)" % (n_feat, dest, os.path.getsize(dest) / 1e6))

    source = download_if_missing(api, "crypto", args.source_version, "historical_meta_models.parquet")
    target = local_path("crypto", version, "historical_meta_models.parquet")
    if not os.path.exists(target):
        shutil.copy(source, target)
    print("run it with:  --data-version %s" % version)


if __name__ == "__main__":
    main()
