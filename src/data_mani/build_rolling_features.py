"""
Build a Crypto dataset with rolling means of the features, computed per ticker.

    python3 -m src.data_mani.build_rolling_features --windows 5,20,60

Writes data/crypto/<version>-roll/train.parquet, used by passing

    --data-version crypto/v2.0-roll

For each selected feature and each window w, adds

    feature_<name>_rm<w> = mean of that feature over the symbol's last w periods

grouped by symbol so two tokens' histories are never mixed.

------------------------------------------------------------------------------
Why this is worth adding
------------------------------------------------------------------------------

The shipped features are rank-binned to five values within each date, so a
single observation carries very little: a token is in one of five buckets. A
20-period mean of that bucket turns a coarse snapshot into a measure of how
*persistently* a token has sat high or low. That is information the raw feature
cannot express, and it is different in kind from the ticker experiments -- it
is per-token but time-varying, rather than a fixed per-token effect.

Note the features are already 20d and 60d constructions (feature_momentum_20d
and so on), so a rolling mean of them is a second smoothing. Short windows are
therefore the interesting ones; a 60-period mean of a 60d indicator is very
slow-moving.

------------------------------------------------------------------------------
Causality
------------------------------------------------------------------------------

Simpler than the target encoding, and worth being explicit about why. A feature
at date t describes what was observable at t -- these are trailing indicators
computed from past prices. So a rolling mean *including* the current row uses
only information available at that moment, and needs no resolution lag.

The target is the opposite case: it is a forward return, unobservable until 20
periods later, which is why build_target_encoding shifts by the embargo. Mixing
the two rules up is the easiest way to leak, so: features may include t, targets
may not go past t - 20.

The rolling window is over a symbol's own consecutive observations. Where a
token has gaps -- delisted and relisted -- the window spans the gap rather than
resetting, which slightly ages the mean for those rows.
"""

import argparse
import os
import shutil

import pandas as pd

from src.data_mani.utils import download_if_missing, get_api, get_spec, local_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add per-ticker rolling means of the features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-version", default="crypto/v2.0")
    parser.add_argument("--out-suffix", default="-roll")
    parser.add_argument("--windows", default="5,20,60",
                        help="comma-separated lookbacks, in periods")
    parser.add_argument("--features", default=None,
                        help="comma-separated feature names to smooth; default "
                             "is every feature in the dataset")
    parser.add_argument("--drop-originals", action="store_true",
                        help="keep only the rolling means, so the comparison "
                             "isolates smoothing rather than adding to it")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    df = pd.read_parquet(src).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    every = [c for c in df.columns if c.startswith("feature_")]
    selected = every if not args.features else [
        f.strip() for f in args.features.split(",")
    ]
    missing = [f for f in selected if f not in every]
    if missing:
        raise SystemExit("unknown features: %s" % missing)

    windows = [int(w) for w in args.windows.split(",")]
    print("source: %d rows, %d symbols" % (len(df), df["symbol"].nunique()))
    print("smoothing %d of %d features over windows %s" % (
        len(selected), len(every), windows))

    grouped = df.groupby("symbol", sort=False)[selected]
    for window in windows:
        # min_periods=1 so early rows get the mean of what exists rather than
        # NaN -- a token's first day is still predictable, just with a shorter
        # lookback behind its smoothed features.
        rolled = grouped.rolling(window, min_periods=1).mean()
        rolled = rolled.reset_index(level=0, drop=True).sort_index()
        rolled.columns = ["%s_rm%d" % (c, window) for c in selected]
        df = pd.concat([df, rolled.astype("float32")], axis=1)
        print("   window %-3d -> %d new columns" % (window, len(rolled.columns)))

    if args.drop_originals:
        df = df.drop(columns=selected)
        print("dropped the %d original features" % len(selected))

    n_feat = sum(c.startswith("feature_") for c in df.columns)
    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    df.set_index("symbol").to_parquet(dest)
    print("features: %d total -> %s (%.0f MB)" % (
        n_feat, dest, os.path.getsize(dest) / 1e6))

    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % version)


if __name__ == "__main__":
    main()
