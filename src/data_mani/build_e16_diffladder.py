"""
E16: cumulative per-symbol DIFFERENCES on a lag ladder reaching one year.

    python3 -m src.data_mani.build_e16_diffladder --through 22
    python3 -m src.data_mani.build_e16_diffladder --through 264 --drop-originals

NAMESPACED ON PURPOSE. A concurrently running study in this same checkout owns
`build_cumdiff.py` and the `v2.0-cumdiff*` dataset names, and emits columns
suffixed `_cumdiff<k>`. This module writes `v2.0-e16d*` and `_e16d<k>` so the
two cannot overwrite each other's datasets or silently mix two definitions of
"cumulative difference" into one table.

------------------------------------------------------------------------------
The ladder, and why it is not every integer
------------------------------------------------------------------------------

A month is 22 dates on Crypto, so a year is 264. Taking every difference order
from 1 to 264 would add 264 x 22 = 5,808 feature columns plus 264 target
columns -- more columns than the panel has rows per date, and a sampling rate
that would bury the 22 raw features completely.

The ladder is log-spaced instead, one rung per horizon a trader would name:

    1    one day
    5    one week
    22   one month
    66   one quarter
    132  half a year
    264  one year

"Cumulative through k" means every RUNG up to k, not every integer:

    through 22   ->  d1, d5, d22            (3 orders x 22 features + 3 target)
    through 264  ->  d1, d5, d22, d66, d132, d264

------------------------------------------------------------------------------
Where this leaks, and the anchor that stops it
------------------------------------------------------------------------------

target_binned_return_20 at date t is the return over [t, t+20] and is NOT
observable at t. Any target difference may therefore only use values resolved by
t, so every one is anchored at the horizon H read off the target name:

    feature_target_e16d<q>  =  y_{t-H} - y_{t-H-q}

y_t - y_{t-q} would overlap the label being predicted and is lookahead. Feature
differences need no anchor -- features describe what was observable at their own
date -- so f_t - f_{t-q} is fine. The two rules differ deliberately; conflating
them is the trap build_target_encoding and build_adl both document.

------------------------------------------------------------------------------
Scale
------------------------------------------------------------------------------

Features are rank-binned to [0, 1], so a difference lands in [-1, 1]. This reuses
build_derived.difference to map it back onto [0, 1] with 0.5 neutral, because
missing values are filled with the neutral 0.5 downstream and 0.5 must mean "no
change". On the raw scale the neutral fill would be 0 -- a strong negative
change -- turning every unavailable history into a confident false signal.

A long rung costs coverage: d264 is undefined for a symbol's first 264 dates,
and many tokens never live that long. Those rows fill neutral, which is the
honest reading, but it means the long rungs carry real signal on a shrinking
subset of the panel. The runner reports that coverage per rung.
"""

import argparse
import os
import shutil

import pandas as pd

from src.data_mani.build_derived import difference
from src.data_mani.utils import (default_embargo, download_if_missing, get_api,
                                 get_spec, local_path)

SYMBOL = "symbol"
PERIOD = "date"

# One rung per horizon a trader would name. 22 dates = one month on Crypto.
LADDER = (1, 5, 22, 66, 132, 264)


def parse_args():
    parser = argparse.ArgumentParser(
        description="E16 cumulative difference ladder, per symbol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--through", type=int, required=True,
                        help="highest ladder rung to include; must be one of %s"
                             % (LADDER,))
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--drop-originals", action="store_true",
                        help="keep ONLY the differences, testing the block alone")
    parser.add_argument("--source-version", type=str, default="crypto/v2.0")
    return parser.parse_args()


def rungs_through(k):
    if k not in LADDER:
        raise SystemExit("--through must be one of %s, got %d" % (LADDER, k))
    return [r for r in LADDER if r <= k]


def add_diffs(df, features, rungs, target, horizon):
    """d<q> for every feature and for the horizon-anchored target, within symbol."""
    df = df.sort_values([SYMBOL, PERIOD]).reset_index(drop=True)
    grouped = df.groupby(SYMBOL, sort=False)
    added, coverage = {}, {}
    for q in rungs:
        for f in features:
            added["%s_e16d%d" % (f, q)] = difference(df[f], grouped[f].shift(q))
        base = grouped[target].shift(horizon)
        added["feature_target_e16d%d" % q] = difference(base, grouped[target].shift(horizon + q))
        # How much of the panel actually HAS this much history?
        probe = grouped[features[0]].shift(q)
        coverage[q] = float(probe.notna().mean())
    block = pd.DataFrame(added, index=df.index).astype("float32")
    return pd.concat([df, block], axis=1), list(block.columns), coverage


def main():
    args = parse_args()
    spec = get_spec("crypto")
    target = args.target or spec["target"]
    horizon = default_embargo("crypto", target)
    rungs = rungs_through(args.through)
    api = get_api("crypto")

    path = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    df = pd.read_parquet(path).reset_index()
    features = sorted(c for c in df.columns if c.startswith("feature_"))

    print("ladder: %s   (a month is 22 dates, a year 264)" % (rungs,))
    print("target %s resolves in %d periods; every target difference is anchored at t-%d"
          % (target, horizon, horizon))
    df, added, coverage = add_diffs(df, features, rungs, target, horizon)
    print("rows with enough history, per rung: " +
          ", ".join("d%d %.0f%%" % (q, 100 * c) for q, c in coverage.items()))

    if args.drop_originals:
        df = df.drop(columns=features)
        print("dropped the %d originals: the difference block is tested alone" % len(features))

    version = "%s-e16d%d%s" % (args.source_version, args.through,
                              "only" if args.drop_originals else "")
    out = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    for filename in ("historical_meta_models.parquet",):
        src = local_path("crypto", args.source_version, filename)
        if os.path.exists(src):
            shutil.copy(src, local_path("crypto", version, filename))

    print("features: %d total -> %s" % (
        sum(c.startswith("feature_") for c in df.columns), out))
    print("run it with:  --data-version %s" % version)


if __name__ == "__main__":
    main()
