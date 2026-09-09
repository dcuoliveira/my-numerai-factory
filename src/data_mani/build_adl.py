"""
Build an autoregressive distributed-lag (ADL) Crypto dataset.

    python3 -m src.data_mani.build_adl --target-lags 20,40,60 --feature-lags 20,40,60

Writes data/crypto/<version>-adl/train.parquet, used by passing

    --data-version crypto/v2.0-adl

An ADL model regresses y_t on its own past AND on past values of the drivers:

    y_t  ~  y_{t-p} ...            the AUTOREGRESSIVE part
            x_{t-q} ...            the DISTRIBUTED-LAG part
            x_t                    the contemporaneous features already shipped

The DL half is what build_lagged.py already does. The AR half is new, and it is
the half that can silently destroy a study -- see below.

------------------------------------------------------------------------------
THE TARGET LAG MUST BE AT LEAST THE TARGET'S HORIZON, OR IT IS LOOKAHEAD
------------------------------------------------------------------------------

`target_binned_return_20` at row (symbol, t) is a FORWARD return: it summarises
[t, t+20] and is not observable until t+20. So a feature at row t may only use a
target from t-k where the whole window has already closed:

        t - k + 20  <=  t        i.e.        k >= 20

At k = 20 the lagged target covers exactly [t-20, t] and is known at t. At k = 19
it covers [t-19, t+1] and one day of it has not happened yet -- and that one day
is enough. The builder REFUSES a lag below the horizon rather than warning,
because the resulting run completes, scores extremely well, and looks like a
discovery.

This is the same rule build_target_encoding applies (its docstring: "the target
rule is stricter and different ... the encoding must additionally wait 20 periods
for the label to resolve"). It bears repeating because the FEATURE rule is
different and laxer: a feature is a level observed at t, so any lag >= 1 is fine.
Mixing the two rules up in one builder is exactly how this goes wrong.

The default lags 20, 40, 60 are chosen so the AR terms are NON-OVERLAPPING
20-day returns: [t-20, t], [t-40, t-20], [t-60, t-40]. Consecutive daily lags
would give 20 nearly-identical copies of the same window, which is 20 columns of
one fact.

------------------------------------------------------------------------------
What the repo already knows about lags here
------------------------------------------------------------------------------

The DL half has been tried and did not help. E05 swept cumulative feature lags
1..10 on LightGBM and the best depth by payout was ZERO -- no lags. E06 narrowed
it to the volatility group and found lags actively harmful, t = -2.18 to -2.30
across depths. The ridge cumlag arms exist for the same comparison.

So the interesting half here is the AR one, which nothing has tried, and the
honest prior for the DL columns is that they cost more than they add.
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
    target_horizon_days,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add autoregressive and distributed lags to the Crypto dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-lags", type=str, default="20,40,60",
                        help="comma-separated lags of the TARGET, in periods. "
                             "Every one must be >= the target's horizon or the "
                             "column would carry unresolved future; the builder "
                             "refuses rather than warns. Empty for none")
    parser.add_argument("--feature-lags", type=str, default="20,40,60",
                        help="comma-separated lags of every FEATURE, in periods. "
                             "A feature is a level observed at t, so any lag >= 1 "
                             "is causal. Empty for none")
    parser.add_argument("--source-version", type=str, default="crypto/v2.0")
    parser.add_argument("--out-suffix", type=str, default="-adl")
    return parser.parse_args()


def parse_lags(text, name):
    """
    :param text: comma-separated integers, possibly empty
    :return: sorted distinct positive lags
    :rtype: [int]
    """
    if not text or not text.strip():
        return []
    try:
        lags = sorted({int(part) for part in text.split(",") if part.strip()})
    except ValueError:
        raise SystemExit("--%s must be comma-separated integers, got %r"
                         % (name, text))
    if any(lag < 1 for lag in lags):
        raise SystemExit("--%s must all be >= 1, got %s" % (name, lags))
    return lags


def shift_within_symbol(frame, columns, lag, symbol_col="symbol",
                        period_col="date"):
    """
    Each symbol's own value `lag` periods earlier.

    The shift is taken WITHIN a symbol after sorting by period. Shifting the
    frame as a whole would hand one token's history to another, which is not a
    lag but a data error -- and it would not announce itself in any score.
    Copied in spirit from build_lagged.add_lag, which says the same.

    :param frame: the panel, any row order
    :param columns: columns to shift
    :param lag: periods to reach back
    :return: shifted columns, aligned to `frame`'s original index
    :rtype: pd.DataFrame
    """
    ordered = frame.sort_values([symbol_col, period_col])
    shifted = ordered.groupby(symbol_col, sort=False)[list(columns)].shift(lag)
    return shifted.reindex(frame.index)


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")
    target = spec["target"]

    target_lags = parse_lags(args.target_lags, "target-lags")
    feature_lags = parse_lags(args.feature_lags, "feature-lags")
    if not target_lags and not feature_lags:
        raise SystemExit("nothing to build: both lag lists are empty")

    horizon = target_horizon_days(target)
    embargo = default_embargo("crypto", target)
    bad = [lag for lag in target_lags if lag < horizon]
    if bad:
        # Refused, not warned. A run built on such a column completes, scores
        # extremely well, and looks like a discovery.
        raise SystemExit(
            "REFUSING to build: --target-lags %s are below the target's %d-period "
            "horizon.\n"
            "  %s at row t summarises [t, t+%d] and is not observable until "
            "t+%d.\n"
            "  A lag of k gives a window [t-k, t-k+%d], which has closed only "
            "when k >= %d.\n"
            "  Use --target-lags %d,%d,%d for non-overlapping resolved windows."
            % (bad, horizon, target, horizon, horizon, horizon, horizon,
               horizon, 2 * horizon, 3 * horizon)
        )

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    df = pd.read_parquet(src).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[target]).reset_index(drop=True)
    features = [c for c in df.columns if c.startswith("feature_")]

    print("source: %d rows, %d symbols, %d dates, %d features"
          % (len(df), df["symbol"].nunique(), df["date"].nunique(), len(features)))
    print("target %s: horizon %d periods, embargo %d"
          % (target, horizon, embargo))
    print("AR lags %s (all >= %d, so every window has closed)"
          % (target_lags or "none", horizon))
    print("DL lags %s (a feature is a level at t, so any lag >= 1 is causal)"
          % (feature_lags or "none"))

    added = []
    for lag in target_lags:
        column = "feature_target_lag%d" % lag
        df[column] = shift_within_symbol(df, [target], lag)[target].astype("float32")
        added.append(column)
    for lag in feature_lags:
        shifted = shift_within_symbol(df, features, lag)
        for feature in features:
            column = "%s_lag%d" % (feature, lag)
            df[column] = shifted[feature].astype("float32")
            added.append(column)

    # The first rows of every symbol have no history and come out NaN. They are
    # KEPT: fill_missing_features replaces them with the tournament's neutral
    # value at load time, and dropping them instead would bias the panel toward
    # long-lived symbols -- the same survivorship problem build_lagged names.
    missing = df[added].isna().mean().mean()
    print("added %d columns; %.1f%% of their cells are NaN (early history), left "
          "for fill_missing_features to neutralise at load" % (len(added), 100 * missing))

    # A cheap, loud check that the AR columns are what they claim. If a lag were
    # short by even one period this correlation would jump, because the column
    # would overlap the target it is predicting.
    for lag in target_lags:
        column = "feature_target_lag%d" % lag
        both = df[[column, target]].dropna()
        rho = np.corrcoef(both[column], both[target])[0, 1] if len(both) else float("nan")
        flag = "  <-- SUSPICIOUS, check the lag" if abs(rho) > 0.25 else ""
        print("  corr(%s, %s) = %+.4f%s" % (column, target, rho, flag))

    n_feat = sum(c.startswith("feature_") for c in df.columns)
    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    df.set_index("symbol").to_parquet(dest)
    print("features: %d total -> %s (%.0f MB)"
          % (n_feat, dest, os.path.getsize(dest) / 1e6))

    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        copy_to = local_path("crypto", version, filename)
        if not os.path.exists(copy_to):
            shutil.copy(source, copy_to)

    print("\nrun it with:  --data-version %s" % version)


if __name__ == "__main__":
    main()
