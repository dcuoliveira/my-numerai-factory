"""
Add the economically-meaningful SPREAD and RATIO pair-features to the combined
vol/infl/carver/ix/fenccausal dataset, then fillmask.

    python3 -m src.data_mani.build_pair_features
        # crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal -> ...-pairs (adds 20)

Run it with:  --data-version crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal-pairs

------------------------------------------------------------------------------
Why spreads one way and ratios another (the scale is the whole point)
------------------------------------------------------------------------------

The source features are rank-binned {0,.25,.5,.75,1} (base/vol/carver) or causal
percentiles [0,1] (inflation) or continuous per-symbol means (_symmean). So:

* SPREAD (A-B) of two such columns is a RANK difference -- "is this symbol ranked
  higher on A than on B" -- which is well defined and economically a relative-
  positioning / term-structure signal. We form it directly on the binned columns
  and rank-bin the difference per date, EXCEPT for the inflation spreads below.

* RATIO (A/B) is only meaningful on RAW MAGNITUDES; a ratio of rank-bins is
  degenerate (0/0, inf, 0.25/0.5 is not a magnitude ratio). So every ratio here is
  formed on the RAW, pre-binning signals (recomputed from spot via
  volatility_features / trend_signals, attached point-in-time <= D-1) and only THEN
  rank-binned -- the same discipline as build_vol_interactions.

* INFLATION spreads (bei - bei) are MACRO: identical across symbols on a date, so a
  cross-sectional rank-bin would map the whole column to 0.5. They are instead kept
  on a date-level [0,1] scale via the affine (diff+1)/2 (0.5 = no gap), the same
  reason macro features are normalised across time, not binned cross-sectionally.

Only pairs whose two legs actually DISAGREE were kept (empirical |corr| checked):
near-redundant pairs (Yang-Zhang vs Parkinson 0.94, close 20d vs 60d 0.98, ...) are
excluded because their spread is noise. A missing leg (no spot, short history)
makes the pair NaN, which fillmask then sets to the neutral 0.5.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.binning import BIN_MASS, bin_cross_sectional, bin_mass
from src.data_mani.build_mommask import apply_fillmask
from src.data_mani.momentum_features import attach_signals_asof
from src.data_mani.trend_signals import TREND_SPECS, compute_trend_signals
from src.data_mani.volatility_features import (VOL_SPECS,
                                               compute_volatility_signals)
from src.data_mani.utils import (download_if_missing, get_api, get_spec,
                                 local_path)

DEFAULT_SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")

# SPREADS on existing binned columns, rank-binned per date after the difference.
# (new_name, A, B) -> feature = A - B
XS_SPREADS = [
    ("feature_sprd_vol_sqret_5d_60d",     "feature_vol_sqret_5d",   "feature_vol_sqret_60d"),
    ("feature_sprd_vol_yz_5d_60d",        "feature_vol_yz_5d",      "feature_vol_yz_60d"),
    ("feature_sprd_vol_ewma_fast_slow",   "feature_vol_ewma_lam90", "feature_vol_ewma_lam99"),
    ("feature_sprd_vol_sqret_minus_yz_20d","feature_vol_sqret_20d", "feature_vol_yz_20d"),
    ("feature_sprd_volyz_minus_base_20d", "feature_vol_yz_20d",     "feature_volatility_20d"),
    ("feature_sprd_volsqret_minus_base_60d","feature_vol_sqret_60d","feature_volatility_60d"),
    ("feature_sprd_ewmac_fast_slow",      "feature_tf_ewmac_2_8",   "feature_tf_ewmac_32_128"),
    ("feature_sprd_breakout_short_long",  "feature_tf_breakout_20", "feature_tf_breakout_160"),
    ("feature_sprd_trend_minus_revert",   "feature_tf_ewmac_8_32",  "feature_tf_bbz_20"),
    ("feature_sprd_mom_minus_rsi_20d",    "feature_momentum_20d",   "feature_relative_strength_index_20d"),
    ("feature_sprd_cap_minus_vol_20d",    "feature_market_cap_avg_20d", "feature_volume_avg_20d"),
    ("feature_sprd_mom_vs_own",           "feature_momentum_20d",   "feature_momentum_20d_symmean"),
    ("feature_sprd_ewmac_vs_own",         "feature_tf_ewmac_8_32",  "feature_tf_ewmac_8_32_symmean"),
    ("feature_sprd_volyz_vs_own",         "feature_vol_yz_20d",     "feature_vol_yz_20d_symmean"),
]

# INFLATION spreads: macro (date-level), kept on a [0,1] scale via (diff+1)/2.
INFL_SPREADS = [
    ("feature_sprd_bei_spot_fwd", "feature_macro_bei_5y",       "feature_macro_bei_5y5y"),
    ("feature_sprd_bei_slope",    "feature_macro_bei_5y",       "feature_macro_bei_30y"),
    ("feature_sprd_bei_accel",    "feature_macro_bei_10y_chg20","feature_macro_bei_10y_chg60"),
]

# RATIOS formed on RAW signals (recomputed from spot), rank-binned after.
# (new_name, raw_numerator_key, raw_denominator_key)
RAW_RATIOS = [
    ("feature_ratio_vol_yz_5d_60d",    "feature_vol_yz_5d",    "feature_vol_yz_60d"),
    ("feature_ratio_vol_sqret_5d_60d", "feature_vol_sqret_5d", "feature_vol_sqret_60d"),
    ("feature_ratio_ewmac_per_vol",    "feature_tf_ewmac_8_32","feature_vol_yz_20d"),
]
RAW_VOL_KEYS = ["feature_vol_sqret_5d", "feature_vol_sqret_60d",
                "feature_vol_yz_5d", "feature_vol_yz_60d", "feature_vol_yz_20d"]
RAW_TREND_KEYS = ["feature_tf_ewmac_8_32"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Add economically-meaningful spread/ratio pair features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal")
    p.add_argument("--out-version", default="crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal-pairs")
    p.add_argument("--spot", default=DEFAULT_SPOT)
    p.add_argument("--submit-lag-days", type=int, default=1)
    p.add_argument("--max-staleness-days", type=int, default=4)
    p.add_argument("--fill-value", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")
    per, idc = spec["period_col"], spec["id_col"]

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    df = pd.read_parquet(src).reset_index()
    df[per] = pd.to_datetime(df[per])
    n_rows = len(df)
    src_feats = [c for c in df.columns if c.startswith("feature_")]
    need = {a for _, a, b in XS_SPREADS + INFL_SPREADS for a in (a, b)}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit("source lacks columns: %s" % missing)
    print("source: %d rows, %d features (%s)" % (n_rows, len(src_feats), args.source_version))

    # ---- raw ratios: recompute raw vol + carver from spot, attach PIT -------- #
    spot = pd.read_parquet(args.spot, columns=["symbol", "date", "open", "high", "low", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    print("computing raw vol/carver signals for the ratios ...")
    rawv = compute_volatility_signals(spot, {k: VOL_SPECS[k] for k in RAW_VOL_KEYS})
    rawv = rawv.drop_duplicates(["symbol", "date"]).rename(
        columns={k: "_rr_" + k for k in RAW_VOL_KEYS})
    rawc = compute_trend_signals(spot[["symbol", "date", "close"]],
                                 {k: TREND_SPECS[k] for k in RAW_TREND_KEYS})
    rawc = rawc.drop_duplicates(["symbol", "date"]).rename(
        columns={k: "_rr_" + k for k in RAW_TREND_KEYS})
    df = attach_signals_asof(df, rawv, ["_rr_" + k for k in RAW_VOL_KEYS],
                             lag_days=args.submit_lag_days, max_staleness_days=args.max_staleness_days,
                             period_col=per, symbol_col=idc)
    df = attach_signals_asof(df, rawc, ["_rr_" + k for k in RAW_TREND_KEYS],
                             lag_days=args.submit_lag_days, max_staleness_days=args.max_staleness_days,
                             period_col=per, symbol_col=idc)

    # ---- build the new columns into a dict, concat once ---------------------- #
    new = {}
    for name, a, b in XS_SPREADS:
        new[name] = (df[a] - df[b]).astype("float32")        # binned later (xs)
    for name, a, b in INFL_SPREADS:
        new[name] = ((df[a] - df[b] + 1.0) / 2.0).astype("float32")  # date-level [0,1]
    for name, a, b in RAW_RATIOS:
        denom = df["_rr_" + b].replace(0.0, np.nan)
        new[name] = (df["_rr_" + a] / denom).astype("float32")  # binned later (xs)
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    xs_cols = [n for n, _, _ in XS_SPREADS] + [n for n, _, _ in RAW_RATIOS]
    infl_cols = [n for n, _, _ in INFL_SPREADS]
    print("added %d spreads + %d inflation spreads + %d ratios = %d"
          % (len(XS_SPREADS), len(INFL_SPREADS), len(RAW_RATIOS),
             len(XS_SPREADS) + len(INFL_SPREADS) + len(RAW_RATIOS)))

    # ---- rank-bin the cross-sectional ones per date ------------------------- #
    print("rank-binning %d cross-sectional pair features per date ..." % len(xs_cols))
    df = bin_cross_sectional(df, xs_cols, per)
    for c in xs_cols[:4] + xs_cols[-2:]:
        print("   %-38s mass drift %.3f" % (c, float(np.nanmax(np.abs(bin_mass(df[c]) - BIN_MASS)))))

    # ---- drop raw temporaries, fillmask, write ------------------------------ #
    df = df.drop(columns=["_rr_" + k for k in RAW_VOL_KEYS + RAW_TREND_KEYS])
    new_cols = xs_cols + infl_cols
    feats = [c for c in df.columns if c.startswith("feature_")]
    n_missing = int(df[new_cols].isna().sum().sum())
    df = apply_fillmask(df, feats, fill_value=args.fill_value)
    print("fillmask: filled %d missing pair-feature cells with %s; present kept"
          % (n_missing, args.fill_value))

    dest = local_path("crypto", args.out_version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    df.set_index(idc).to_parquet(dest)
    print("features: %d total (%d source + %d pairs) -> %s (%.0f MB)"
          % (len([c for c in df.columns if c.startswith("feature_")]),
             len(src_feats), len(new_cols), dest, os.path.getsize(dest) / 1e6))

    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", args.out_version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % args.out_version)
    print("isolate the pair features with:  --feature-include '^feature_(sprd|ratio)_'")


if __name__ == "__main__":
    main()
