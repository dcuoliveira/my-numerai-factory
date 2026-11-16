"""
Cross-sectional momentum-moment features for Numerai Crypto.

For each date t, summarise the RAW (un-binned) momentum cross-section into three
market-regime numbers -- the MEAN, VARIANCE and SKEW of the momentum distribution
across coins that day:

    raw 17 momentum signals (per symbol,date, from spot; momentum_features)
      -> per date, cross-sectional {mean,var,skew} of EACH of the 17   (51 series)
      -> causal trailing percentile of each (scale-free, in [0,1])     (51 series)
      -> collapse: average the 17 within each moment type              (3 series)
      -> causal trailing percentile again (restore a clean [0,1] range)(3 series)
      -> date-level as-of join onto v2.0-momswap, lagged to D-1        (+3 columns)
      -> data/crypto/v2.0-momswap-xsmom/train.parquet                  (31 -> 34)

WHY NOT rank-binned. These three numbers are DAY-LEVEL: identical for every symbol
on t. Numerai's rank-bin is WITHIN a date, so a constant-across-the-cross-section
column maps entirely to the neutral 0.5 -- it carries nothing. Like the macro
features, they are normalised across TIME instead (macro_features.
causal_trailing_percentile -> [0,1], 0.5 neutral, causal) and broadcast with a
DATE-level as-of join (attach_macro_asof, no `by` symbol).

WHY the raw signals, not the binned momentum columns. v2.0-momswap's momentum is
already rank-binned within each date, so its per-day cross-sectional mean is 0.5
and its variance/skew are fixed by the 5/20/50/20/5 mass -- aggregating them is a
constant. The cross-sectional distribution only MOVES on the raw (return-scale)
signals, so the moments are computed there.

WHY percentile-then-average-then-percentile. The 17 raw signals live on different
scales (a 7-day return vs a 112-day return vs a MAX), so averaging their raw
cross-sectional variances would just track the largest-scale signal. Converting
each of the 51 series to its own causal percentile first makes them comparable;
averaging the 17 percentiles within a moment type collapses to 3; a final causal
percentile restores a full, uniform [0,1] range (the first average concentrates
near 0.5).

LEAKAGE. Each raw signal at date d reads only closes <= d (momentum_features'
min_periods window). The cross-sectional moment at d is a function of date-d
signals; the causal percentile uses only dates <= d; the as-of join gives row D
the moment from the latest date <= D - lag_days. So row D's three features are a
function of spot closes <= D - lag_days, the last bar known when round D is
submitted. Nothing reads a future price.

HONEST CAVEAT (as with the macro block). A within-date-constant feature has ~zero
standalone within-era numerai_corr -- that metric is a within-date rank
correlation, and a constant column has no within-date rank. It can only help a
tree or a net via REGIME-GATING or an explicit interaction with the per-symbol
features; it will not show up as a standalone signal. Judge it in the paired
base-vs-block comparison, not on its own IC.

    python3 -m src.data_mani.build_xsmom                 # -> data/crypto/v2.0-momswap-xsmom/
    python3 -m src.data_mani.build_xsmom --drop-originals # only the 3, for isolation
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.momentum_features import MOMENTUM_SPECS, compute_momentum_signals
from src.data_mani.macro_features import (causal_trailing_percentile,
                                          attach_macro_asof)
from src.data_mani.utils import download_if_missing, get_api, get_spec, local_path

DEFAULT_SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")

# The three cross-sectional moments, and the emitted feature name for each.
MOMENTS = [
    ("mean", "feature_xsmom_mean"),
    ("var",  "feature_xsmom_var"),
    ("skew", "feature_xsmom_skew"),
]


def cross_sectional_moment_features(raw, mom_cols, min_symbols=20,
                                    window=None, min_periods=60):
    """
    Collapse the raw momentum cross-section to three causal-percentile day series.

    :param raw: per (symbol,date) raw momentum signals (from compute_momentum_signals)
    :type raw: pd.DataFrame
    :param mom_cols: the raw momentum signal columns (the 17)
    :type mom_cols: list[str]
    :param min_symbols: a date needs at least this many symbols with a value for a
        signal before its cross-sectional moment is trusted (skew needs >= 3; 20
        keeps the estimate stable); fewer -> NaN that hour
    :type min_symbols: int
    :param window: trailing window for the causal percentile (None = expanding)
    :param min_periods: history required before a percentile is emitted
    :return: frame keyed ``date`` with feature_xsmom_{mean,var,skew} in [0,1]
    :rtype: pd.DataFrame
    """
    g = raw.groupby("date")
    counts = g[mom_cols].count()                      # symbols per (date, signal)

    # For each moment type, one causal-percentile series per raw signal, then the
    # mean across the 17, then a final causal percentile.
    out = {}
    for moment, feat in MOMENTS:
        if moment == "mean":
            per_signal = g[mom_cols].mean()
        elif moment == "var":
            per_signal = g[mom_cols].var()            # ddof=1; sd would percentile identically
        elif moment == "skew":
            per_signal = g[mom_cols].skew()
        else:
            raise ValueError("unknown moment %r" % moment)

        # Mask thin days so a 2-symbol date does not inject a noise moment.
        per_signal = per_signal.where(counts >= min_symbols)

        # Scale-free: each signal's daily moment -> its own causal percentile.
        pct = pd.DataFrame(index=per_signal.index)
        for c in mom_cols:
            pct[c] = causal_trailing_percentile(per_signal[c], window=window,
                                                min_periods=min_periods)
        # Collapse the 17 to one, then restore a clean [0,1] range.
        collapsed = pct.mean(axis=1, skipna=True)
        out[feat] = causal_trailing_percentile(collapsed, window=window,
                                               min_periods=min_periods)

    daily = pd.DataFrame(out)
    daily.index.name = "date"
    return daily.reset_index()


def parse_args():
    p = argparse.ArgumentParser(
        description="Add cross-sectional momentum-moment (mean/var/skew) features.")
    p.add_argument("--source-version", default="crypto/v2.0-momswap",
                   help="base panel to extend (the 31-feature swap set)")
    p.add_argument("--out-suffix", default="-momswap-xsmom",
                   help="written to data/crypto/v2.0<suffix>/")
    p.add_argument("--spot", default=DEFAULT_SPOT,
                   help="spot OHLCV parquet with symbol,date,close")
    p.add_argument("--min-symbols", type=int, default=20,
                   help="min symbols on a date for a trusted cross-sectional moment")
    p.add_argument("--submit-lag-days", type=int, default=1,
                   help="row D may only use moment dates <= D - this (last bar known)")
    p.add_argument("--max-staleness-days", type=int, default=4,
                   help="attached moment date must be within this many days of D")
    p.add_argument("--drop-originals", action="store_true",
                   help="keep only the 3 new features, for isolation")
    return p.parse_args()


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")
    version = "crypto/v2.0" + args.out_suffix
    feat_cols = [f for _, f in MOMENTS]

    # ---- base panel (v2.0-momswap) --------------------------------------- #
    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    base_feats = [c for c in base.columns if c.startswith("feature_")]
    print("base: %d rows, %d symbols, %d features (%s)"
          % (len(base), base["symbol"].nunique(), len(base_feats), args.source_version))

    # ---- raw momentum signals from spot ---------------------------------- #
    if not os.path.exists(args.spot):
        raise SystemExit("spot file not found: %s" % args.spot)
    spot = pd.read_parquet(args.spot, columns=["symbol", "date", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    mom_cols = list(MOMENTUM_SPECS.keys())
    print("computing %d raw momentum signals from spot (%d rows) ..."
          % (len(mom_cols), len(spot)))
    raw = compute_momentum_signals(spot, MOMENTUM_SPECS).drop_duplicates(["symbol", "date"])

    # ---- cross-sectional moments -> 3 causal-percentile day series -------- #
    print("collapsing the cross-section to 3 moment features ...")
    daily = cross_sectional_moment_features(raw, mom_cols, min_symbols=args.min_symbols)
    cover = daily[feat_cols].notna().any(axis=1).mean()
    print("  moment series span %d dates (%.0f%% non-NaN); ranges %s"
          % (len(daily), 100 * cover,
             {f: (round(float(daily[f].min(skipna=True) or np.nan), 3),
                  round(float(daily[f].max(skipna=True) or np.nan), 3)) for f in feat_cols}))

    # ---- broadcast onto the panel POINT-IN-TIME (date-level as-of) -------- #
    print("attaching as-of (submit lag %d day, max staleness %d days) ..."
          % (args.submit_lag_days, args.max_staleness_days))
    merged = attach_macro_asof(base, daily, feat_cols,
                               lag_days=args.submit_lag_days,
                               max_staleness_days=args.max_staleness_days,
                               period_col=spec["period_col"])
    if len(merged) != len(base):
        raise SystemExit("row count changed in the as-of join: %d -> %d"
                         % (len(base), len(merged)))

    if args.drop_originals:
        keep = [c for c in merged.columns if not c.startswith("feature_")
                or c in feat_cols]
        merged = merged[keep]

    n_new = merged[feat_cols].notna().mean()
    print("attached: %s non-NaN fractions %s"
          % (feat_cols, {f: round(float(n_new[f]), 3) for f in feat_cols}))

    # ---- write the variant ------------------------------------------------ #
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    merged.set_index("symbol").to_parquet(dest)
    total_feats = len([c for c in merged.columns if c.startswith("feature_")])
    print("wrote %s (%d rows, %d features)" % (dest, len(merged), total_feats))

    # the meta-model file must travel with the variant, or reference/BMC breaks.
    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if os.path.abspath(source) != os.path.abspath(target):
            shutil.copy(source, target)
            print("copied %s" % filename)


if __name__ == "__main__":
    main()
