"""
Add Carver/Clenow trend-following signals to a Crypto dataset, on Numerai's
rank-bin scale, merged POINT-IN-TIME.

    python3 -m src.data_mani.build_trend        # v2.0-momswap-macro -> ...-carver

Run it with:  --data-version crypto/v2.0-momswap-macro-carver

Adds EWMAC, Carver breakout, Bollinger z-score, the simple 20d-high-in-5d rule and
Clenow momentum (src/data_mani/trend_signals.py) -- all per-symbol price signals,
computed raw from spot, attached as-of <= D-1 per symbol, then rank-binned per date
(same discipline as build_momentum). Three of the requested signals are NOT
re-added because they are already present: normalised momentum (return/vol) is the
existing trend/vol-scaled block, and cross-sectional momentum (buy strong/sell
weak) is exactly what the per-date rank-bin does to every one of these features.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.binning import BIN_MASS, bin_cross_sectional, bin_mass
from src.data_mani.momentum_features import attach_signals_asof
from src.data_mani.trend_signals import TREND_SPECS, compute_trend_signals
from src.data_mani.utils import (download_if_missing, get_api, get_spec,
                                 local_path)

DEFAULT_SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")


def parse_args():
    p = argparse.ArgumentParser(
        description="Add Carver/Clenow trend-following signals from spot prices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0-momswap-macro",
                   help="dataset the trend block is added onto")
    p.add_argument("--out-suffix", default="-carver")
    p.add_argument("--spot", default=DEFAULT_SPOT)
    p.add_argument("--features", default=None,
                   help="comma list of trend specs; default all TREND_SPECS")
    p.add_argument("--submit-lag-days", type=int, default=1)
    p.add_argument("--max-staleness-days", type=int, default=4)
    return p.parse_args()


def _subset(names):
    if not names:
        return TREND_SPECS
    want = [n.strip() for n in names.split(",")]
    bad = [n for n in want if n not in TREND_SPECS]
    if bad:
        raise SystemExit("unknown trend features: %s\nknown: %s" % (bad, list(TREND_SPECS)))
    return {n: TREND_SPECS[n] for n in want}


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")
    specs = _subset(args.features)
    tf_cols = list(specs)

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    n_rows = len(base)
    src_feats = [c for c in base.columns if c.startswith("feature_")]
    print("source: %d rows, %d features (%s)" % (n_rows, len(src_feats), args.source_version))

    if not os.path.exists(args.spot):
        raise SystemExit("spot file not found: %s" % args.spot)
    spot = pd.read_parquet(args.spot, columns=["symbol", "date", "close"])
    spot["date"] = pd.to_datetime(spot["date"])

    print("computing %d trend-following signals ..." % len(tf_cols))
    raw = compute_trend_signals(spot, specs).drop_duplicates(["symbol", "date"])
    print("attaching as-of (submit lag %d day, max staleness %d days) ..."
          % (args.submit_lag_days, args.max_staleness_days))
    merged = attach_signals_asof(base, raw, tf_cols,
                                 lag_days=args.submit_lag_days,
                                 max_staleness_days=args.max_staleness_days,
                                 period_col=spec["period_col"], symbol_col=spec["id_col"])
    if len(merged) != n_rows:
        raise SystemExit("as-of merge changed the row count %d -> %d" % (n_rows, len(merged)))

    for c in tf_cols:
        print("   %-26s coverage %5.1f%% of panel rows" % (c, 100 * float(merged[c].notna().mean())))

    print("rank-binning to {0,.25,.5,.75,1} per date ...")
    merged = bin_cross_sectional(merged, tf_cols, spec["period_col"])
    for c in tf_cols:
        drift = float(np.nanmax(np.abs(bin_mass(merged[c]) - BIN_MASS)))
        print("   %-26s mass drift %.3f%s" % (c, drift, "  <-- OFF" if drift > 0.02 else ""))

    n_feat = sum(c.startswith("feature_") for c in merged.columns)
    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    merged.set_index("symbol").to_parquet(dest)
    print("features: %d total (%d source + %d trend) -> %s (%.0f MB)"
          % (n_feat, len(src_feats), len(tf_cols), dest, os.path.getsize(dest) / 1e6))

    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % version)
    print("isolate the trend block with:  --feature-include '^feature_tf_'")


if __name__ == "__main__":
    main()
