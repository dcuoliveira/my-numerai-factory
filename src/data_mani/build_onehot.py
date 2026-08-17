"""
Build a Crypto dataset with the tickers added as one-hot features.

    python3 -m src.data_mani.build_onehot --min-dates 130

Writes data/crypto/<version>-onehot/train.parquet alongside the reference files
the pipeline needs, so the new dataset is used simply by passing

    --data-version crypto/v2.0-onehot

to src.run_forecast. Nothing in the modelling code changes: the crypto loader
picks up features by the "feature_" prefix, so the new columns are found the
same way the original 22 are.

Why this can work at all: the target is rank-binned *within* each date, so its
cross-sectional mean is identical every day and a ticker one-hot cannot learn a
level. What it can learn is a symbol's persistent tendency to rank high or low.
That effect is real in this data -- per-symbol mean targets have 5.85x the
dispersion noise would produce -- and it persists, with first-half and
second-half symbol means correlating 0.309.

The min-dates threshold matters. A symbol seen on 6 dates gives a column that is
almost all zeros, which cannot be estimated and only adds noise; symbols below
the threshold simply get no column, which the model reads as "unknown ticker".
At 130 dates that keeps 740 of 1,309 symbols and still covers 95% of rows.
"""

import argparse
import os
import shutil

import pandas as pd

from src.data_mani.utils import get_api, local_path, download_if_missing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add ticker one-hot features to the Crypto dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-version", default="crypto/v2.0")
    parser.add_argument("--out-suffix", default="-onehot",
                        help="appended to the version to name the new dataset")
    parser.add_argument("--min-dates", type=int, default=130,
                        help="a symbol needs at least this many dates to get a "
                             "column; below it the one-hot cannot be estimated "
                             "and only adds noise")
    return parser.parse_args()


def main():
    args = parse_args()
    api = get_api("crypto")

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    df = pd.read_parquet(src).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    print("source: %s  (%d rows, %d symbols)" % (src, len(df), df["symbol"].nunique()))

    counts = df.groupby("symbol")["date"].nunique()
    keep = sorted(counts[counts >= args.min_dates].index)
    covered = df["symbol"].isin(keep).mean()
    print("keeping %d of %d symbols (>= %d dates), covering %.1f%% of rows" % (
        len(keep), len(counts), args.min_dates, 100 * covered))

    # uint8 keeps 740 extra columns cheap on disk and in memory; the modelling
    # code casts to float only for the training block it is fitting.
    dummies = pd.get_dummies(
        df["symbol"].where(df["symbol"].isin(keep)),
        prefix="feature_sym", dtype="uint8",
    )
    out = pd.concat([df, dummies], axis=1)

    n_feat = sum(c.startswith("feature_") for c in out.columns)
    print("features: %d original + %d ticker = %d total" % (
        n_feat - len(dummies.columns), len(dummies.columns), n_feat))

    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # Match the source layout: the shipped file carries symbol as the index.
    out.set_index("symbol").to_parquet(dest)
    print("wrote %s (%.0f MB)" % (dest, os.path.getsize(dest) / 1e6))

    # The reference and meta-model loaders resolve paths from the version, so the
    # new dataset directory needs its own copies. Same files, so BMC and MMC stay
    # identical between the two experiments and the comparison is like-for-like.
    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)
            print("copied %s" % os.path.basename(target))

    print("\nrun it with:  --data-version %s" % version)


if __name__ == "__main__":
    main()
