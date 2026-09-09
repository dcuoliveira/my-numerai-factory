"""
Build a Crypto dataset of CUMULATIVE per-symbol DIFFERENCES.

    python3 -m src.data_mani.build_cumdiff --through 3 --mode rawdiff

For depth p, adds every difference order 1..p of every shipped feature, plus the
same orders of the (resolved) target:

    feature_<f>_diff<k>      =  f_t  -  f_{t-k}          for k = 1..p
    feature_target_diff<k>   =  y_{t-h}  -  y_{t-h-k}    for k = 1..p, h = horizon

`--mode diff` keeps ONLY those columns; `--mode rawdiff` keeps the raw 22 beside
them. Both are written as their own dataset version so no regex has to select
them at run time.

------------------------------------------------------------------------------
Why this is NOT the cumulative-lag study again
------------------------------------------------------------------------------

It is the same SPAN and a different BASIS, and for a penalised estimator that is
a different model.

    span{ f_t, f_t - f_{t-1}, ..., f_t - f_{t-p} }  ==  span{ f_t, f_{t-1}, ..., f_{t-p} }

so `--mode rawdiff --through p` and `crypto/v2.0-cumlagp` describe exactly the
same linear subspace. An UNPENALISED least squares fit could not tell them apart:
same fitted values, same residuals, coefficients related by an invertible map.

Ridge can. The penalty ||b||^2 is not invariant under a change of basis, so
shrinkage is applied along the axes the columns happen to define. And these two
bases are conditioned very differently. Same-feature autocorrelation at lag 1 on
this panel runs 0.82 (momentum), 0.94 (volatility), 0.99 (close_avg), so the
LAGGED-LEVEL basis hands glmnet a design matrix whose columns are near-copies of
each other; the DIFFERENCE basis replaces each near-copy with a residual that is
close to orthogonal to the level it was formed from. Under one alpha shared
across the whole matrix -- which is what glmnet does -- the second basis should
shrink far less destructively.

That is the hypothesis, and it is testable for free: all ten
`w6r6_glmnet_reg_cumlag{1..10}` arms already exist, so every rawdiff arm has an
exact same-span partner to be paired against.

The prior on the LEVEL of the result is still bad. E05 swept cumulative lags on
LightGBM and the best depth was ZERO; E06 narrowed it to volatility and found
lags actively harmful (t = -2.18 .. -2.30). What is new here is that neither
study varied the basis, only the span.

------------------------------------------------------------------------------
THE TARGET DIFFERENCE MUST CLEAR THE HORIZON, OR IT IS LOOKAHEAD
------------------------------------------------------------------------------

`target_binned_return_20` at row (symbol, t) summarises [t, t+20] and is not
observable until t+20. So the NAIVE difference

    y_t - y_{t-k}                                    <-- LOOKAHEAD, never built

carries 20 days of unresolved future in its first term. Every target difference
here is anchored at the last resolved label instead:

    feature_target_diff<k>  =  y_{t-h}  -  y_{t-h-k}          h = horizon = 20

whose terms cover [t-20, t] and [t-20-k, t-k]. Both windows have closed by t.
This is the same rule build_adl.py enforces on its AR lags and the same one
build_target_encoding.py applies to its encoding; the builder REFUSES a smaller
anchor rather than warning, because a run built on the leaky column completes,
scores around CORR 0.80, and looks like a discovery. E13's leak control did
exactly that.

------------------------------------------------------------------------------
THE FILL VALUE FOR A DIFFERENCE IS 0.0, NOT THE NEUTRAL BIN
------------------------------------------------------------------------------

The first k rows of a symbol have no predecessor, so the difference is NaN.

build_lagged fills its columns with 0.5, and that is right THERE: a lag column
holds a LEVEL, and 0.5 is the middle bin of these rank-binned features -- the
literal "no information" value.

A difference is not a level. It lives on [-1, +1] with a mean near zero, and its
"no information" value is 0.0. Leaving the NaN for `fill_missing_features` to
handle at load would stamp 0.5 -- half the full range, and the same sign for every
young token -- onto exactly the rows that have no history. So the builder fills
the difference columns itself, before writing, and the loader then finds nothing
to do. Raw columns in `--mode rawdiff` are left alone and still get 0.5, which is
correct for them.

------------------------------------------------------------------------------
One property inherited from build_lagged, worth stating
------------------------------------------------------------------------------

`shift(k)` reaches back k OBSERVATIONS within a symbol, not k calendar days. A
symbol with a hole in its date series therefore gets a difference spanning more
than k days. build_lagged and build_adl both behave this way; the alternative
(reindexing every symbol onto the full date axis) would manufacture rows that
were never observed. Consistency with the existing lag datasets matters more
here, because the whole point is a paired comparison against them.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.utils import (
    download_if_missing,
    get_api,
    get_spec,
    local_path,
    target_horizon_days,
)

#: A difference means "no change" at zero. This is NOT the features' neutral
#: bin -- see the module docstring.
DIFF_NEUTRAL = 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cumulative per-symbol differences of features and target.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--through", type=int, default=None,
                        help="CUMULATIVE depth p: every difference order 1..p. "
                             "Consecutive orders are near-duplicate columns at "
                             "long horizons -- prefer --orders past a few weeks")
    parser.add_argument("--orders", type=str, default=None,
                        help="explicit comma-separated difference orders, e.g. "
                             "1,2,3,5,10,21,63,130,261. The right instrument for "
                             "reaching a year: 261 consecutive orders would be "
                             "261 near-identical copies of one fact, and 5,742 "
                             "columns of it")
    parser.add_argument("--mode", choices=("diff", "rawdiff"), default="rawdiff",
                        help="diff = the difference columns alone; "
                             "rawdiff = the raw features beside them")
    parser.add_argument("--no-feature-diffs", action="store_true",
                        help="omit the feature differences (attribution arm)")
    parser.add_argument("--no-target-diffs", action="store_true",
                        help="omit the target differences (attribution arm)")
    parser.add_argument("--source-version", default="crypto/v2.0")
    parser.add_argument("--out-suffix", default=None,
                        help="defaults to -cumdiff<p> / -rawcumdiff<p>")
    return parser.parse_args()


def diff_within_symbol(frame, columns, k, symbol_col="symbol", period_col="date"):
    """
    Each symbol's own change over k observations: v_t - v_{t-k}.

    Taken WITHIN a symbol after sorting by period. Differencing the frame as a
    whole would subtract one token's history from another's, which is not a
    difference but a data error -- and it would not announce itself in any score.

    :param frame: the panel, any row order
    :type frame: pd.DataFrame
    :param columns: columns to difference
    :type columns: [str]
    :param k: observations to reach back
    :type k: int
    :return: differenced columns, aligned to `frame`'s original index
    :rtype: pd.DataFrame
    """
    ordered = frame.sort_values([symbol_col, period_col])
    differenced = ordered.groupby(symbol_col, sort=False)[list(columns)].diff(k)
    return differenced.reindex(frame.index)


def shift_within_symbol(frame, columns, lag, symbol_col="symbol", period_col="date"):
    """Each symbol's own value `lag` observations earlier. See build_adl."""
    ordered = frame.sort_values([symbol_col, period_col])
    shifted = ordered.groupby(symbol_col, sort=False)[list(columns)].shift(lag)
    return shifted.reindex(frame.index)


def main():
    args = parse_args()
    if (args.through is None) == (args.orders is None):
        raise SystemExit("pass exactly one of --through or --orders")
    if args.through is not None and args.through < 1:
        raise SystemExit("--through must be >= 1")
    if args.no_feature_diffs and args.no_target_diffs:
        raise SystemExit("nothing to build: both difference blocks are disabled")
    if args.mode == "diff" and args.no_feature_diffs and args.no_target_diffs:
        raise SystemExit("--mode diff with no difference blocks leaves no features")

    spec = get_spec("crypto")
    api = get_api("crypto")
    target = spec["target"]
    horizon = target_horizon_days(target)
    if args.through is not None:
        orders = list(range(1, args.through + 1))
    else:
        try:
            orders = sorted({int(x) for x in args.orders.split(",") if x.strip()})
        except ValueError:
            raise SystemExit("--orders must be comma-separated integers, got %r"
                             % args.orders)
        if not orders or orders[0] < 1:
            raise SystemExit("--orders must all be >= 1, got %s" % orders)

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    df = pd.read_parquet(src).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    raw_features = [c for c in df.columns if c.startswith("feature_")]

    print("source: %d rows, %d symbols, %d dates, %d features"
          % (len(df), df["symbol"].nunique(), df["date"].nunique(), len(raw_features)))
    print("target %s: horizon %d periods" % (target, horizon))
    print("mode %s, difference orders %s"
          % (args.mode, ",".join(str(k) for k in orders)))

    # Blocks are collected and concatenated ONCE. Assigning 200+ columns one at
    # a time re-copies the frame on nearly every insert (pandas warns about the
    # fragmentation) and turns a 15-second build into minutes.
    blocks, added = [], []

    if not args.no_feature_diffs:
        for k in orders:
            differenced = diff_within_symbol(df, raw_features, k)
            differenced.columns = ["%s_diff%d" % (c, k) for c in raw_features]
            blocks.append(differenced.astype("float32"))
            added.extend(differenced.columns)

    if not args.no_target_diffs:
        # Anchor at the last RESOLVED label, never at y_t. See the docstring.
        resolved = shift_within_symbol(df, [target], horizon).rename(
            columns={target: "_resolved"})
        anchored = pd.concat([df[["symbol", "date"]], resolved], axis=1)
        columns = {}
        for k in orders:
            older = shift_within_symbol(anchored, ["_resolved"], k)["_resolved"]
            columns["feature_target_diff%d" % k] = (
                anchored["_resolved"] - older).astype("float32")
        block = pd.DataFrame(columns, index=df.index)
        blocks.append(block)
        added.extend(block.columns)

    df = pd.concat([df] + blocks, axis=1)

    # Fill the difference columns HERE, with 0.0. fill_missing_features would
    # stamp the features' neutral bin 0.5 on them at load, which is half the
    # range and the same sign for every young token.
    missing = int(df[added].isna().to_numpy().sum())
    cells = df[added].size
    print("added %d difference columns; %d cells (%.2f%% of the block) have no "
          "predecessor and become %.1f"
          % (len(added), missing, 100.0 * missing / cells, DIFF_NEUTRAL))
    # Coverage per order, because it collapses at long horizons and a mostly
    # filled column is a mostly constant column. 61% of rows have a real
    # 261-order predecessor; the rest are tokens younger than a year.
    print("  %8s %14s %10s" % ("order", "real cells", "coverage"))
    for k in orders:
        block = [c for c in added if c.endswith("_diff%d" % k)]
        if not block:
            continue
        real = int(df[block].notna().to_numpy().sum())
        print("  %8d %14d %9.1f%%" % (k, real, 100.0 * real / df[block].size))
    df[added] = df[added].fillna(DIFF_NEUTRAL)

    if args.mode == "diff":
        df = df.drop(columns=raw_features)
        print("mode diff: dropped the %d raw feature columns" % len(raw_features))

    # A cheap, loud check that the target differences are anchored where they
    # claim. If the anchor were short by even one period these would jump,
    # because the column would overlap the label it is predicting. E13's leak
    # control scored corr 0.80 on exactly this mistake.
    for k in orders:
        column = "feature_target_diff%d" % k
        if column not in df.columns:
            continue
        both = df[[column, target]].dropna()
        rho = np.corrcoef(both[column], both[target])[0, 1] if len(both) else np.nan
        flag = "  <-- SUSPICIOUS, check the anchor" if abs(rho) > 0.25 else ""
        print("  corr(%s, %s) = %+.4f%s" % (column, target, rho, flag))

    n_feat = sum(c.startswith("feature_") for c in df.columns)
    stem = "-cumdiff%s" if args.mode == "diff" else "-rawcumdiff%s"
    default_suffix = stem % (args.through if args.through is not None
                             else "x".join(str(k) for k in orders))
    suffix = args.out_suffix or default_suffix
    version = args.source_version + suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    df.set_index("symbol").to_parquet(dest)
    print("features: %d total -> %s (%.0f MB)"
          % (n_feat, dest, os.path.getsize(dest) / 1e6))

    # The reference file must exist in every version directory, or BMC and MMC
    # would be scored on a different denominator than the baseline's.
    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        copy_to = local_path("crypto", version, filename)
        if not os.path.exists(copy_to):
            shutil.copy(source, copy_to)

    print("\nrun it with:  --data-version %s" % version)


if __name__ == "__main__":
    main()
