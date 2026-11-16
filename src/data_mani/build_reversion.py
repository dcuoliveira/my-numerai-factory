"""
Build a Crypto dataset that carries mean-reversion / contrarian features, on the
SAME rank-bin scale as the shipped features, merged onto the tournament panel.

    python3 -m src.data_mani.build_reversion                  # crypto/v2.0 -> ...-rev (adds 14 to the 22)
    python3 -m src.data_mani.build_reversion --drop-originals # reversion only, for isolation

Run it with:  --data-version crypto/v2.0-rev

The block is 14 reversion features (src/data_mani/reversion_features.py) from the
crypto reversal literature: short-term reversal (str_{1,7,14}d), the MAX lottery
effect (max1/max5_28d), idiosyncratic volatility (ivol_28d), return skewness
(skew_28d), Amihud illiquidity (amihud_28d), and the technical oscillators RSI(14),
Bollinger-z(20), stochastic %K(14), CCI(20), plus distance-from-MA(20,50).

The chain is the usual three modules (reversion_features = pure OHLCV math; binning
= Numerai's {0,.25,.5,.75,1} quantization; this file = offline wiring). The join is
POINT-IN-TIME, lagged by the submission constraint (``attach_signals_asof``, row D
uses the freshest spot signal dated <= D-1); binning happens AFTER the join, over
the panel's per-date cross-section. Base rows and base feature columns are never
touched -- the panel is a superset. Same discipline as build_momentum/build_vol.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.binning import BIN_MASS, bin_cross_sectional, bin_mass
from src.data_mani.momentum_features import attach_signals_asof
from src.data_mani.reversion_features import (REVERSION_SPECS,
                                              compute_reversion_signals)
from src.data_mani.utils import (download_if_missing, get_api, get_spec,
                                 local_path)

DEFAULT_SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")


def parse_args():
    p = argparse.ArgumentParser(
        description="Add rank-binned mean-reversion features from spot OHLCV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0")
    p.add_argument("--out-suffix", default="-rev")
    p.add_argument("--spot", default=DEFAULT_SPOT,
                   help="spot OHLCV parquet with symbol,date,open,high,low,close,volume")
    p.add_argument("--features", default=None,
                   help="comma-separated reversion feature names; default all")
    p.add_argument("--drop-originals", action="store_true",
                   help="keep only the reversion features, to isolate the block")
    p.add_argument("--submit-lag-days", type=int, default=1)
    p.add_argument("--max-staleness-days", type=int, default=4)
    return p.parse_args()


def _select(names):
    if not names:
        return REVERSION_SPECS
    want = [n.strip() for n in names.split(",")]
    bad = [n for n in want if n not in REVERSION_SPECS]
    if bad:
        raise SystemExit("unknown reversion features: %s\nknown: %s" % (bad, list(REVERSION_SPECS)))
    return {n: REVERSION_SPECS[n] for n in want}


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")
    specs = _select(args.features)
    rev_cols = list(specs.keys())

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    n_rows = len(base)
    base_feats = [c for c in base.columns if c.startswith("feature_")]
    print("base: %d rows, %d symbols, %d features (%s)"
          % (n_rows, base["symbol"].nunique(), len(base_feats), args.source_version))

    if not os.path.exists(args.spot):
        raise SystemExit("spot file not found: %s" % args.spot)
    spot = pd.read_parquet(
        args.spot, columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    spot["date"] = pd.to_datetime(spot["date"])
    print("spot: %d rows, %d symbols (%s)" % (len(spot), spot["symbol"].nunique(), args.spot))

    print("computing %d reversion signals ..." % len(rev_cols))
    raw = compute_reversion_signals(spot, specs).drop_duplicates(["symbol", "date"])
    print("attaching as-of (submit lag %d day, max staleness %d days) ..."
          % (args.submit_lag_days, args.max_staleness_days))
    merged = attach_signals_asof(base, raw, rev_cols,
                                 lag_days=args.submit_lag_days,
                                 max_staleness_days=args.max_staleness_days,
                                 period_col=spec["period_col"], symbol_col=spec["id_col"])
    if len(merged) != n_rows:
        raise SystemExit("as-of merge changed the row count %d -> %d" % (n_rows, len(merged)))

    for c in rev_cols:
        print("   %-26s coverage %5.1f%% of panel rows" % (c, 100 * float(merged[c].notna().mean())))

    print("rank-binning to {0,.25,.5,.75,1} per date ...")
    merged = bin_cross_sectional(merged, rev_cols, spec["period_col"])
    for c in rev_cols:
        drift = float(np.nanmax(np.abs(bin_mass(merged[c]) - BIN_MASS)))
        print("   %-26s mass drift %.3f%s" % (c, drift, "  <-- OFF" if drift > 0.02 else ""))

    if args.drop_originals:
        merged = merged.drop(columns=base_feats)
        print("dropped the %d original features" % len(base_feats))

    n_feat = sum(c.startswith("feature_") for c in merged.columns)
    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    merged.set_index("symbol").to_parquet(dest)
    print("features: %d total (%d reversion) -> %s (%.0f MB)"
          % (n_feat, len(rev_cols), dest, os.path.getsize(dest) / 1e6))

    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % version)
    print("isolate the reversion features with:  --feature-include '^feature_rev_'")


if __name__ == "__main__":
    main()
