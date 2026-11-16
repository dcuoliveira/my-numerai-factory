"""
Build a Crypto dataset encoding each ticker by its mean FEATURE values.

    python3 -m src.data_mani.build_feature_encoding --mode causal
    python3 -m src.data_mani.build_feature_encoding --mode global
    python3 -m src.data_mani.build_feature_encoding --mode causal --bin

For every feature f, adds  feature_<f>_symmean = that symbol's average of f.
22 features in, 22 new columns out -- one per feature, versus 740 for the
one-hot.

------------------------------------------------------------------------------
What this encodes, and how it differs from what is already built
------------------------------------------------------------------------------

    ticker one-hot   WHICH token it is        (identity, 740 columns)
    tenc             HOW IT HAS PERFORMED     (mean TARGET per symbol)
    rolling mean     WHERE IT SITS LATELY     (feature over the last 5/20/60)
    this             WHAT KIND OF TOKEN IT IS (typical feature profile)

Because the shipped features are already cross-sectional ranks within each date,
a symbol's mean feature value is its typical standing among peers: "this token
usually sits in the 80th percentile of market cap and the 30th of volatility".
That is a stable description of the asset rather than a statement about its
recent behaviour or its past returns.

------------------------------------------------------------------------------
Two modes, and why the distinction matters
------------------------------------------------------------------------------

causal   Expanding mean over the symbol's OWN PAST only. A row never sees its
         own feature value or any later one. Deployable, and directly
         comparable with every other arm.

         No resolution lag is needed here, unlike the target encoding: features
         describe what was observable at their date, whereas the target is a
         forward return that does not resolve for 20 periods. Feature rule and
         target rule are deliberately different -- conflating them is the
         easiest way to leak (see build_target_encoding).

global   Whole-sample mean. This is LOOKAHEAD. Not target leakage -- no target
         is touched -- but at prediction time you would not know a symbol's
         future feature values, and the full-sample mean encodes them. That
         matters more than it sounds: 62% of tokens do not survive three years,
         so a full-sample profile is partly a statement about whether the token
         survived, which correlates with the target.

         Build it as an UPPER BOUND, not a candidate. If causal and global score
         the same, the profile carries no lookahead advantage and the causal
         version is trustworthy. If global scores far higher, the gap is the
         part you can never have live -- the same argument the leaky
         target-encoding variants settled (0.254 leaky vs 0.056 causal).

Symbols with little history are shrunk toward the global feature mean with a
pseudo-count, so a token seen three times does not get a confident profile.

--bin rank-bins the encoded columns within each date onto Numerai's exact scale,
{0,.25,.5,.75,1}, under the same names, so -fenccausal-binned is a drop-in for
-fenccausal. The means themselves are unchanged. The live path does not offer it.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.history_features import bin_ties_preserved
from src.data_mani.utils import download_if_missing, get_api, local_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode each ticker by its mean feature values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", required=True, choices=["causal", "global"])
    parser.add_argument("--source-version", default="crypto/v2.0")
    parser.add_argument("--out-suffix", default=None, help="defaults to -fenc<mode>")
    parser.add_argument("--smoothing", type=float, default=50.0,
                        help="pseudo-count pulling thin symbols toward the "
                             "overall feature mean")
    parser.add_argument("--features", default=None,
                        help="comma-separated subset; default is every feature")
    parser.add_argument("--bin", action="store_true",
                        help="rank-bin the encodings per date to {0,.25,.5,.75,1}")
    return parser.parse_args()


def causal_feature_means(df, features, symbol_col, period_col, smoothing,
                         prior=None):
    """
    Expanding per-symbol mean of each feature, using only strictly earlier rows.

    Implemented as a prefix sum over each symbol's own chronological rows, then
    shifted by one so a row never contributes to its own encoding. Rows with no
    history yet fall back to the overall feature mean, which for these
    rank-binned features is ~0.5 -- "no view".

    `prior` overrides that fallback with a precomputed {feature: value} mapping.
    It exists for the LIVE path. The default computes the fallback from the frame
    it is handed, which offline is the training panel and is therefore fine -- but
    live the frame also holds today's rows, so today's features would enter the
    prior and shift the encoding of every HISTORICAL row. Measured on a 60-date
    live-shaped panel the shift is ~6e-6 (the features are rank-binned within each
    date, so the mean is 0.5 by construction and barely moves), but a past value
    that depends on the present is wrong regardless of size. src/live/features.py
    passes a prior computed from non-live rows only.
    """
    df = df.sort_values([symbol_col, period_col])
    order = df.index

    encoded = {}
    grouped = df.groupby(symbol_col, sort=False)
    counts = grouped.cumcount()                       # rows strictly before this one

    for feature in features:
        # cumsum then shift: the total over rows strictly earlier than this one.
        cumulative = grouped[feature].cumsum() - df[feature]
        overall = (float(df[feature].mean()) if prior is None
                   else float(prior[feature]))
        encoded["%s_symmean" % feature] = (
            (cumulative + smoothing * overall) / (counts + smoothing)
        )

    out = pd.DataFrame(encoded, index=order).sort_index()
    return out, int((counts == 0).sum())


def global_feature_means(df, features, symbol_col, smoothing):
    """Whole-sample per-symbol mean. Lookahead by construction -- see module docs."""
    means = df.groupby(symbol_col)[features].mean()
    sizes = df.groupby(symbol_col).size()
    overall = df[features].mean()

    # Same shrinkage, so the two modes differ only in which rows they average.
    shrunk = means.mul(sizes, axis=0).add(overall * smoothing, axis=1)
    shrunk = shrunk.div(sizes + smoothing, axis=0)
    shrunk.columns = ["%s_symmean" % c for c in shrunk.columns]

    return df[[symbol_col]].join(shrunk, on=symbol_col).drop(columns=[symbol_col]), 0


def main():
    args = parse_args()
    api = get_api("crypto")
    suffix = args.out_suffix or ("-fenc%s%s" % (args.mode, "-binned" if args.bin else ""))

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    df = pd.read_parquet(src).reset_index()
    df["date"] = pd.to_datetime(df["date"])

    every = [c for c in df.columns if c.startswith("feature_")]
    features = every if not args.features else [f.strip() for f in args.features.split(",")]
    missing = [f for f in features if f not in every]
    if missing:
        raise SystemExit("unknown features: %s" % missing)

    print("source: %d rows, %d symbols, %d features" % (
        len(df), df["symbol"].nunique(), len(every)))
    print("mode=%s, smoothing=%g" % (args.mode, args.smoothing))

    if args.mode == "causal":
        encoded, no_history = causal_feature_means(
            df, features, "symbol", "date", args.smoothing)
        print("rows with no prior history (fall back to the overall mean): %.1f%%" % (
            100 * no_history / len(df)))
    else:
        encoded, _ = global_feature_means(df, features, "symbol", args.smoothing)
        print("LOOKAHEAD: whole-sample means. Diagnostic upper bound, not a candidate.")

    if args.bin:
        encoded = bin_ties_preserved(encoded, df["date"])
        print("rank-binned per date to {0,.25,.5,.75,1}")

    df = pd.concat([df, encoded.astype("float32")], axis=1)
    print("added %d encoding columns" % len(encoded.columns))
    for name in list(encoded.columns)[:3]:
        print("    %-46s mean %.4f  sd %.4f" % (
            name, df[name].mean(), df[name].std()))

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
