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

import numpy as np
import pandas as pd

from src.data_mani.utils import get_api, local_path, download_if_missing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add ticker one-hot features to the Crypto dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--key", default="symbol",
                        choices=("symbol", "ucid", "cap"),
                        help="what to one-hot. 'symbol' is the TICKER, which is "
                             "NOT a stable identity: 30 tickers in this panel "
                             "are reused by different coins (ACT covers three) "
                             "and 3 coins were renamed (GRAM->TON), so a ticker "
                             "dummy conflates 2.5%% of rows. 'ucid' is the "
                             "stable coin id and fixes both. 'cap' buckets by "
                             "market-cap rank instead of identity")
    parser.add_argument("--only", action="store_true",
                        help="drop the original feature_ columns, so the one-hot "
                             "REPLACES them rather than joining them")
    parser.add_argument("--cap-column", default="feature_market_cap_avg_20d",
                        help="for --key cap: the rank-binned column bucketed")
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

    if args.key == "cap":
        # Market-cap BUCKET rather than identity. The column is already rank
        # binned WITHIN each date to {0, .25, .5, .75, 1}, so the bucket is a
        # cross-sectional cap rank and a coin moves between buckets over time --
        # which is the point: it asks whether SIZE carries the persistence that
        # a ticker dummy is credited with, at three columns instead of 740.
        if args.cap_column not in df.columns:
            raise SystemExit("--cap-column %r is not in the source; have %s"
                             % (args.cap_column,
                                [c for c in df.columns if "market_cap" in c]))
        rank = df[args.cap_column]
        bucket = pd.Series(np.where(rank <= 0.25, "low",
                                    np.where(rank >= 0.75, "large", "mid")),
                           index=df.index)
        bucket = bucket.where(rank.notna())
        dummies = pd.get_dummies(bucket, prefix="feature_cap", dtype="uint8")
        print("cap buckets from %s: %s"
              % (args.cap_column,
                 ", ".join("%s=%d" % (b, int(dummies["feature_cap_" + b].sum()))
                           for b in ("low", "mid", "large")
                           if "feature_cap_" + b in dummies)))
    else:
        key = args.key
        counts = df.groupby(key)["date"].nunique()
        keep = sorted(counts[counts >= args.min_dates].index)
        covered = df[key].isin(keep).mean()
        print("keeping %d of %d %ss (>= %d dates), covering %.1f%% of rows" % (
            len(keep), len(counts), key, args.min_dates, 100 * covered))
        # Prefixed feature_sym_ whatever the key, so every downstream regex --
        # CURUPIRA_05's members, generalised_ridge's FEATURE_BLOCKS, the
        # feature_groups shell scripts -- keeps working unchanged and a ucid
        # build is a drop-in for a symbol build.
        # uint8 keeps 740 extra columns cheap on disk and in memory; the
        # modelling code casts to float only for the block it is fitting.
        dummies = pd.get_dummies(
            df[key].where(df[key].isin(keep)),
            prefix="feature_sym", dtype="uint8",
        )

    original = [c for c in df.columns if c.startswith("feature_")]
    if args.only:
        # The one-hot REPLACES the raw columns rather than joining them.
        out = pd.concat([df.drop(columns=original), dummies], axis=1)
        print("--only: dropped the %d original feature columns" % len(original))
    else:
        out = pd.concat([df, dummies], axis=1)

    n_feat = sum(c.startswith("feature_") for c in out.columns)
    print("features: %d original + %d one-hot = %d total" % (
        n_feat - len(dummies.columns), len(dummies.columns), n_feat))

    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # Match the source layout: the shipped file carries symbol as the index,
    # regardless of what was one-hotted. ucid stays a plain column.
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
