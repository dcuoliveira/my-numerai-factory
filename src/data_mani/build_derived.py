"""
Build Crypto datasets with derived features the raw set implies but does not contain.

    python3 -m src.data_mani.build_derived --kind spread
    python3 -m src.data_mani.build_derived --kind interact

Two kinds, built as separate datasets so each is tested in isolation:

  spread    the 20d-minus-60d difference of every indicator. The raw features
            are LEVELS; the spread is a CHANGE OF STATE, which is a different
            object. momentum_20d minus momentum_60d separates short-term
            reversal from longer-term momentum -- in equities these carry
            opposite signs, so a single momentum level conflates them.
            volatility_20d minus volatility_60d is the volatility term
            structure, the standard stress signal, and volatility_20d is
            already the strongest single feature in the set.

  interact  products and same-horizon differences a linear model cannot form:
              ewa minus sma      -- EWA leads SMA, so the gap is trend or flow
                                    ACCELERATION rather than level
              momentum x vol     -- momentum crashes in high-volatility regimes
              mcap x momentum    -- small-cap momentum is a distinct effect
              volume x vol       -- separates informed flow from noise
              rsi x vol          -- overbought means different things in calm
                                    and stressed markets
            Ridge cannot build a product; this is how you give it some of the
            expressiveness a tree gets for free.

------------------------------------------------------------------------------
Scale convention
------------------------------------------------------------------------------

The shipped features are rank-binned into [0, 1] with 0.5 as the neutral middle
bin, and the pipeline fills missing features with that same 0.5. Every derived
column here is mapped back onto [0, 1] with 0.5 neutral, so that convention keeps
holding:

    difference   (a - b + 1) / 2          a == b  ->  0.5
    product      (a-0.5)(b-0.5)*2 + 0.5   either neutral  ->  0.5

Getting this wrong matters. A raw difference sits in [-1, 1] with 0 neutral, so
the pipeline's 0.5 fill would assert "20d far above 60d" for exactly the young
tokens whose 60d indicator is missing -- turning an absence of data into a
strong spurious signal.

------------------------------------------------------------------------------
Causality
------------------------------------------------------------------------------

Both operands are observable at the row's own date -- these are trailing
indicators built from past prices -- so a same-row combination introduces no
lookahead and needs no resolution lag. That is the feature rule; the target rule
is different and stricter (see build_target_encoding).
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.utils import download_if_missing, get_api, local_path

HORIZONS = ("20d", "60d")

# (left, right) indicator pairs whose product is motivated by market structure.
INTERACTIONS = [
    ("momentum", "volatility"),                 # momentum crashes when vol spikes
    ("market_cap_avg", "momentum"),             # size-conditioned momentum
    ("volume_avg", "volatility"),               # informed flow vs noise
    ("relative_strength_index", "volatility"),  # overbought, conditioned on regime
]

# (fast, slow) pairs where the gap measures acceleration rather than level.
ACCELERATION = [
    ("close_ewa", "close_avg"),
    ("volume_ewa", "volume_avg"),
    ("market_cap_ewa", "market_cap_avg"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add horizon spreads or feature interactions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kind", required=True, choices=["spread", "interact"])
    parser.add_argument("--source-version", default="crypto/v2.0")
    parser.add_argument("--out-suffix", default=None,
                        help="defaults to -<kind>")
    parser.add_argument("--drop-originals", action="store_true",
                        help="keep only the derived columns, isolating their "
                             "contribution instead of adding to the raw set")
    return parser.parse_args()


def difference(a, b):
    """Map a - b from [-1, 1] onto [0, 1] with 0.5 neutral."""
    return ((a - b) + 1.0) / 2.0


def product(a, b):
    """Map the centred product onto [0, 1] with 0.5 neutral."""
    return (a - 0.5) * (b - 0.5) * 2.0 + 0.5


def build_spreads(df):
    """20d minus 60d, for every indicator present at both horizons."""
    added = {}
    for column in [c for c in df.columns if c.endswith("_20d")]:
        indicator = column[len("feature_"):-len("_20d")]
        long_column = "feature_%s_60d" % indicator
        if long_column not in df.columns:
            continue
        added["feature_%s_spread" % indicator] = difference(
            df[column], df[long_column]
        )
    return added


def build_interactions(df):
    """Acceleration gaps and the market-structure products, at both horizons."""
    added = {}
    for horizon in HORIZONS:
        for fast, slow in ACCELERATION:
            f, s = "feature_%s_%s" % (fast, horizon), "feature_%s_%s" % (slow, horizon)
            if f in df.columns and s in df.columns:
                name = fast.replace("_ewa", "")
                added["feature_%s_accel_%s" % (name, horizon)] = difference(df[f], df[s])

        for left, right in INTERACTIONS:
            l, r = "feature_%s_%s" % (left, horizon), "feature_%s_%s" % (right, horizon)
            if l in df.columns and r in df.columns:
                short_left = left.replace("_avg", "").replace("relative_strength_index", "rsi")
                short_right = right.replace("_avg", "")
                added["feature_%s_x_%s_%s" % (short_left, short_right, horizon)] = product(
                    df[l], df[r]
                )
    return added


def main():
    args = parse_args()
    api = get_api("crypto")
    suffix = args.out_suffix or ("-%s" % args.kind)

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    df = pd.read_parquet(src).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    original = [c for c in df.columns if c.startswith("feature_")]
    print("source: %d rows, %d features" % (len(df), len(original)))

    added = build_spreads(df) if args.kind == "spread" else build_interactions(df)
    derived = pd.DataFrame(added, index=df.index)

    # Missing operands propagate to NaN. Fill with the neutral 0.5 here rather
    # than leaving it to the pipeline, so the meaning is explicit: "no view",
    # not "extreme reading".
    n_missing = int(derived.isna().to_numpy().sum())
    derived = derived.fillna(0.5).astype("float32")
    df = pd.concat([df, derived], axis=1)

    print("kind=%s -> %d derived columns (%d NaN filled with the neutral 0.5)" % (
        args.kind, len(derived.columns), n_missing))
    for name in list(derived.columns)[:4]:
        print("    %-44s mean %.4f  sd %.4f" % (
            name, derived[name].mean(), derived[name].std()))
    if len(derived.columns) > 4:
        print("    ... and %d more" % (len(derived.columns) - 4))

    if args.drop_originals:
        df = df.drop(columns=original)
        print("dropped the %d original features" % len(original))

    n_feat = sum(c.startswith("feature_") for c in df.columns)
    version = args.source_version + suffix
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

    print("run it with:  --data-version %s\n" % version)


if __name__ == "__main__":
    main()
