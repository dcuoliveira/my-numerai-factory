"""
Build a Crypto dataset with the ticker replaced by a mean-target encoding.

    python3 -m src.data_mani.build_target_encoding --smoothing 50

Writes data/crypto/<version>-tenc/train.parquet, used by passing

    --data-version crypto/v2.0-tenc

Same idea as the one-hot dataset -- capture a symbol's persistent tendency to
rank high or low -- but as ONE feature instead of 740. Where the one-hot asks
the model to estimate 740 coefficients from a 6-month window, this hands it the
per-symbol mean directly.

------------------------------------------------------------------------------
Avoiding leakage
------------------------------------------------------------------------------

Mean target encoding leaks trivially if done naively: compute each symbol's mean
target over the whole dataset, and every row's feature contains that row's own
target. The model then "predicts" something it was told, and the backtest looks
spectacular and means nothing.

Two constraints make it honest here, and both are enforced by construction:

1. STRICTLY PAST DATA. The encoding for (symbol s, date t) uses only rows of s
   at dates strictly before t. A row never contributes to its own feature.

2. THE TARGET RESOLUTION LAG. This is the one that is easy to miss. The target
   at date d is a *forward* 20-day return -- it is not observable until 20
   periods later. So at date t you only know targets for dates up to t - 20.
   Using dates up to t - 1 would be using returns that had not happened yet:
   leakage that no train/validation split can catch, because it is inside the
   feature itself.

   The lag is the same `embargo` the CV uses, and for the same reason.

So: encoding(s, t) = mean of target[s, d] over d <= t - horizon.

That is exactly what would be available live, which is the real test.

Symbols with little history get shrunk toward the global mean:

    encoding = (n * symbol_mean + m * global_mean) / (n + m)

with m = --smoothing. A symbol seen 3 times sits near the global mean; one seen
500 times is essentially its own average. Unseen symbols get the global mean,
which for this rank-binned target is 0.5 -- i.e. "no opinion".
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.utils import (
    default_embargo,
    download_if_missing,
    get_api,
    get_spec,
    local_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add a leakage-safe mean-target encoding of the ticker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-version", default="crypto/v2.0")
    parser.add_argument("--out-suffix", default="-tenc")
    parser.add_argument("--smoothing", type=float, default=50.0,
                        help="pseudo-count pulling thin symbols toward the "
                             "global mean; higher = more shrinkage")
    parser.add_argument("--window", type=int, default=0,
                        help="use only the last N periods of history per symbol "
                             "(0 = expanding, all available past)")
    parser.add_argument("--keep-onehot", action="store_true",
                        help="also emit the one-hot columns, for a combined dataset")
    return parser.parse_args()


def causal_target_encoding(df, symbol_col, period_col, target_col,
                           lag, smoothing, window=0):
    """
    Per-row mean-target encoding using only data available at that row's date.

    Works on integer period positions rather than dates so the lag is exact:
    position p may only use a symbol's observations at positions <= p - lag.
    For each row we binary-search the symbol's own sorted positions to find how
    many qualify, then read a prefix sum. No row can see itself or the future.

    :param df: full frame
    :type df: pd.DataFrame
    :param lag: periods the target takes to resolve (the embargo)
    :type lag: int
    :param smoothing: pseudo-count toward the global mean
    :type smoothing: float
    :param window: keep only the last N qualifying observations (0 = all)
    :type window: int
    :return: (encoding, n_observations_behind_it)
    :rtype: (np.ndarray, np.ndarray)
    """
    periods = np.sort(df[period_col].unique())
    pos = pd.Series(np.arange(len(periods)), index=periods)
    row_pos = df[period_col].map(pos).to_numpy()

    global_mean = float(df[target_col].mean())
    encoding = np.full(len(df), np.nan)
    counts = np.zeros(len(df))

    order = np.argsort(row_pos, kind="stable")
    for symbol, idx in pd.Series(np.arange(len(df)), index=df[symbol_col].to_numpy()).groupby(level=0):
        rows = idx.to_numpy()
        rows = rows[np.argsort(row_pos[rows], kind="stable")]   # chronological
        p = row_pos[rows]
        y = df[target_col].to_numpy()[rows]

        # Prefix sums over this symbol's own chronological observations. A NaN
        # target is a row whose label has not resolved yet -- the live panel's
        # history rows -- and is not an observation: it adds nothing to either
        # sum. Summing it raw poisoned every later prefix, and once Numerai's
        # train file lagged live by more than the lag (32 days from 2026-10-01)
        # every live row read a NaN. With no NaN target this is byte-identical
        # to the plain cumsum/arange it replaces.
        seen = ~np.isnan(y)
        csum = np.concatenate([[0.0], np.cumsum(np.where(seen, y, 0.0))])
        ccnt = np.concatenate([[0], np.cumsum(seen)])

        # How many of this symbol's observations are at position <= p - lag?
        hi = np.searchsorted(p, p - lag, side="right")
        lo = np.zeros_like(hi) if window <= 0 else np.maximum(0, hi - window)

        n = ccnt[hi] - ccnt[lo]
        total = csum[hi] - csum[lo]

        # Shrink toward the global mean; n = 0 gives exactly the global mean.
        encoding[rows] = (total + smoothing * global_mean) / (n + smoothing)
        counts[rows] = n

    return encoding, counts


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    df = pd.read_parquet(src).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[spec["target"]]).reset_index(drop=True)

    lag = default_embargo("crypto", spec["target"])
    print("source: %d rows, %d symbols, %d dates" % (
        len(df), df["symbol"].nunique(), df["date"].nunique()))
    print("resolution lag: %d periods (a %s target is not observable until then)" % (
        lag, spec["target"]))

    encoding, counts = causal_target_encoding(
        df, "symbol", "date", spec["target"],
        lag=lag, smoothing=args.smoothing, window=args.window,
    )
    df["feature_sym_tenc"] = encoding.astype("float32")

    print("encoding: mean %.4f  sd %.4f  (global target mean %.4f)" % (
        np.nanmean(encoding), np.nanstd(encoding), df[spec["target"]].mean()))
    print("rows with no usable history yet: %.1f%% (they get the global mean)" % (
        100 * (counts == 0).mean()))
    print("median observations behind an encoding: %d" % np.median(counts))

    if args.keep_onehot:
        counts_by_sym = df.groupby("symbol")["date"].nunique()
        keep = counts_by_sym[counts_by_sym >= 130].index
        dummies = pd.get_dummies(df["symbol"].where(df["symbol"].isin(keep)),
                                 prefix="feature_sym", dtype="uint8")
        df = pd.concat([df, dummies], axis=1)
        print("also kept %d one-hot columns" % len(dummies.columns))

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
