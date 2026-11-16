"""
Build a Crypto dataset carrying cointegration / stat-arb RESIDUAL signals, on the
SAME rank-bin scale as the shipped features, merged onto the tournament panel.

    python3 -m src.data_mani.build_cointegration                  # crypto/v2.0 -> ...-coint (adds 4)
    python3 -m src.data_mani.build_cointegration --drop-originals # residuals only

Run it with:  --data-version crypto/v2.0-coint

The block (src/data_mani/cointegration_features.py) regresses each coin's log-price
on the log-levels of BTC + the equal-weight crypto market + gold/DXY/S&P over
rolling windows, and emits the residual z-score (60d, 120d), the AR(1) half-life and
the Dickey-Fuller stationarity t-stat of the 120d residual. The residual is a
relative-value mean-reversion signal; the half-life / ADF say whether it is actually
stationary (cointegrated), guarding the spurious-regression trap.

Three modules as usual (compute = pure OLS math; binning = Numerai's scale; this =
offline wiring). The join is POINT-IN-TIME (attach_signals_asof, row D uses signals
dated <= D-1); binning is AFTER the join, per date. Base rows/cols untouched.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.binning import BIN_MASS, bin_cross_sectional, bin_mass
from src.data_mani.cointegration_features import (COINT_SPECS,
                                                  compute_cointegration_signals)
from src.data_mani.momentum_features import attach_signals_asof
from src.data_mani.utils import (download_if_missing, get_api, get_spec,
                                 local_path)

DEFAULT_SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")
DEFAULT_MACRO = os.path.join("data", "crypto", "macro", "macro_daily.parquet")


def parse_args():
    p = argparse.ArgumentParser(
        description="Add cointegration-residual (stat-arb) features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0")
    p.add_argument("--out-suffix", default="-coint")
    p.add_argument("--spot", default=DEFAULT_SPOT)
    p.add_argument("--macro", default=DEFAULT_MACRO)
    p.add_argument("--drop-originals", action="store_true")
    p.add_argument("--submit-lag-days", type=int, default=1)
    p.add_argument("--max-staleness-days", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")
    coint_cols = list(COINT_SPECS.keys())

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    n_rows = len(base)
    base_feats = [c for c in base.columns if c.startswith("feature_")]
    print("base: %d rows, %d symbols, %d features (%s)"
          % (n_rows, base["symbol"].nunique(), len(base_feats), args.source_version))

    if not os.path.exists(args.spot):
        raise SystemExit("spot not found: %s" % args.spot)
    if not os.path.exists(args.macro):
        raise SystemExit("macro store not found: %s" % args.macro)
    spot = pd.read_parquet(args.spot, columns=["symbol", "date", "close"])
    macro = pd.read_parquet(args.macro)
    print("spot: %d rows, %d symbols | macro: %d rows"
          % (len(spot), spot["symbol"].nunique(), len(macro)))

    print("computing cointegration residuals (rolling OLS per symbol) ...")
    raw = compute_cointegration_signals(spot, macro).drop_duplicates(["symbol", "date"])
    print("attaching as-of (submit lag %d day, max staleness %d days) ..."
          % (args.submit_lag_days, args.max_staleness_days))
    merged = attach_signals_asof(base, raw, coint_cols,
                                 lag_days=args.submit_lag_days,
                                 max_staleness_days=args.max_staleness_days,
                                 period_col=spec["period_col"], symbol_col=spec["id_col"])
    if len(merged) != n_rows:
        raise SystemExit("as-of merge changed the row count %d -> %d" % (n_rows, len(merged)))

    for c in coint_cols:
        print("   %-28s coverage %5.1f%% of panel rows" % (c, 100 * float(merged[c].notna().mean())))

    print("rank-binning to {0,.25,.5,.75,1} per date ...")
    merged = bin_cross_sectional(merged, coint_cols, spec["period_col"])
    for c in coint_cols:
        drift = float(np.nanmax(np.abs(bin_mass(merged[c]) - BIN_MASS)))
        print("   %-28s mass drift %.3f%s" % (c, drift, "  <-- OFF" if drift > 0.02 else ""))

    if args.drop_originals:
        merged = merged.drop(columns=base_feats)
        print("dropped the %d original features" % len(base_feats))

    n_feat = sum(c.startswith("feature_") for c in merged.columns)
    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    merged.set_index("symbol").to_parquet(dest)
    print("features: %d total (%d cointegration) -> %s (%.0f MB)"
          % (n_feat, len(coint_cols), dest, os.path.getsize(dest) / 1e6))

    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % version)
    print("isolate the cointegration features with:  --feature-include '^feature_coint_'")


if __name__ == "__main__":
    main()
