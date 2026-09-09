"""
Build a Crypto dataset with the raw features plus one lagged copy of them.

    python3 -m src.data_mani.build_lagged --lag 3

For every shipped feature f, adds  feature_<f>_lag<p>  =  that symbol's value of
f, p periods earlier. 22 features in, 22 new columns out, 44 total.

------------------------------------------------------------------------------
What a lag adds that the raw features do not
------------------------------------------------------------------------------

The shipped features are LEVELS observed today. A lag pairs each level with the
same quantity p days ago, which lets a linear model form a DIFFERENCE it cannot
otherwise express: ridge can fit b1*f_t + b2*f_{t-p}, and with b2 negative that
is a change over p days.

This is the same idea the `spread` block used across HORIZONS (20d minus 60d) and
the `interact` block used across SMOOTHINGS (ewa minus sma). Those two are the
only feature blocks in this project that ever cleared |t| > 2 -- `interact` at
t = 2.18 in the leave-one-block-out study -- so a difference across TIME is the
natural third axis and the one that has not been tried.

------------------------------------------------------------------------------
Causality
------------------------------------------------------------------------------

A lag reaches strictly backwards: row (symbol, t) receives that symbol's own
feature value from t - p, which was observable at t. No target is touched and no
resolution lag is needed. That is the feature rule; the target rule is stricter
and different (see build_target_encoding, where the encoding must additionally
wait 20 periods for the label to resolve).

The shift is taken WITHIN each symbol after sorting by date. Shifting the frame
as a whole would hand one token's history to another, which is not a lag but a
data error, and it would not announce itself in any score.

------------------------------------------------------------------------------
The first p rows of every symbol
------------------------------------------------------------------------------

They have no predecessor, so the lag is NaN. Filled with 0.5 -- the neutral
middle bin these rank-binned features use, and the same value the pipeline fills
missing features with. It reads as "no view", which is exactly right for a token
that has not existed long enough. Leaving the NaN would be worse: Ridge cannot
fit it at all.
"""

import argparse
import os
import shutil

import pandas as pd

from src.data_mani.utils import download_if_missing, get_api, local_path

NEUTRAL = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add a lagged copy of every raw feature.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lag", type=int, default=None,
                        help="a SINGLE lag: raw + that one shifted copy")
    parser.add_argument("--through", type=int, default=None,
                        help="CUMULATIVE: raw + every lag from 1 to this value. "
                             "raw + lag1 + lag2 + ... Answers a different "
                             "question from --lag: how much history helps, "
                             "rather than which single horizon does")
    parser.add_argument("--source-version", default="crypto/v2.0")
    parser.add_argument("--out-suffix", default=None,
                        help="defaults to -lag<p>")
    return parser.parse_args()


def add_lag(df, features, lag, symbol_col="symbol", period_col="date"):
    """
    That symbol's own feature values, `lag` periods earlier.

    Sorted by (symbol, date) first so the shift follows each token's own
    chronology rather than whatever order the parquet happened to store.
    """
    df = df.sort_values([symbol_col, period_col]).reset_index(drop=True)
    shifted = df.groupby(symbol_col, sort=False)[features].shift(lag)
    shifted.columns = ["%s_lag%d" % (c, lag) for c in features]
    return df, shifted


def main():
    args = parse_args()
    if (args.lag is None) == (args.through is None):
        raise SystemExit("pass exactly one of --lag or --through")
    if args.lag is not None and args.lag < 1:
        raise SystemExit("--lag must be >= 1")
    if args.through is not None and args.through < 1:
        raise SystemExit("--through must be >= 1")

    lags = [args.lag] if args.lag is not None else list(range(1, args.through + 1))
    api = get_api("crypto")
    suffix = args.out_suffix or (
        "-lag%d" % args.lag if args.lag is not None else "-cumlag%d" % args.through)

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    df = pd.read_parquet(src).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    features = [c for c in df.columns if c.startswith("feature_")]
    print("source: %d rows, %d symbols, %d features" % (
        len(df), df["symbol"].nunique(), len(features)))

    blocks = []
    missing = 0
    for lag in lags:
        df, shifted = add_lag(df, features, lag)
        missing += int(shifted.isna().to_numpy().sum())
        blocks.append(shifted.fillna(NEUTRAL).astype("float32"))
    added = pd.concat(blocks, axis=1)
    df = pd.concat([df, added], axis=1)

    print("lags %s -> %d new columns (%d NaN filled with the neutral %.1f, "
          "%.2f%% of the block -- the first rows of each symbol)" % (
              ",".join(str(l) for l in lags), len(added.columns), missing,
              NEUTRAL, 100.0 * missing / added.size))
    for name in list(added.columns)[:2] + list(added.columns)[-1:]:
        print("    %-42s mean %.4f  sd %.4f" % (
            name, df[name].mean(), df[name].std()))

    n_feat = sum(c.startswith("feature_") for c in df.columns)
    version = args.source_version + suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    df.set_index("symbol").to_parquet(dest)
    print("features: %d total -> %s (%.0f MB)" % (
        n_feat, dest, os.path.getsize(dest) / 1e6))

    # The reference file must be present in every version directory, or BMC and
    # MMC would be scored on a different denominator than the baseline's.
    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("run it with:  --data-version %s\n" % version)


if __name__ == "__main__":
    main()
