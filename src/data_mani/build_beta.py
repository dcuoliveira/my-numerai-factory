"""
Build a Crypto dataset carrying rolling market-beta (vs BTC) features, on the
SAME rank-bin scale as the shipped features, merged point-in-time onto the panel.

    python3 -m src.data_mani.build_beta                 # -> data/crypto/v2.0-beta (22 + 3)
    python3 -m src.data_mani.build_beta --drop-originals # beta only, for isolation

Run it with:  --data-version crypto/v2.0-beta

beta = OLS slope of a coin's daily returns on BTC's daily returns over a trailing
window (30/60/90d): the coin's market sensitivity. Per-symbol (cross-sectional),
so it rank-bins normally. The chain mirrors build_momentum/build_vol:

    spot OHLCV -> beta_features.compute_beta_signals -> attach_signals_asof
                 (lag 1 day, as every external-feed builder) -> bin_cross_sectional
                 (per panel date) -> data/crypto/v2.0-beta/train.parquet

The join is POINT-IN-TIME (row D uses returns through D-1, the last completed bar
known at submission); binning is AFTER the merge, over the panel's per-date
cross-section. A coin with too little history, or no spot, is NaN -> neutral 0.5
at load. INTERPRETER: /home/danielco/miniconda3/bin/python3 (needs pyarrow).
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.beta_features import BETA_SPECS, compute_beta_signals
from src.data_mani.binning import BIN_MASS, bin_cross_sectional, bin_mass
from src.data_mani.momentum_features import attach_signals_asof
from src.data_mani.utils import (download_if_missing, get_api, get_spec,
                                 local_path)

DEFAULT_SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")


def parse_args():
    p = argparse.ArgumentParser(
        description="Add rank-binned rolling market-beta (vs BTC) features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0")
    p.add_argument("--out-suffix", default="-beta")
    p.add_argument("--spot", default=DEFAULT_SPOT,
                   help="spot OHLCV parquet with symbol,date,close")
    p.add_argument("--market-symbol", default="BTC",
                   help="the market factor coins are projected on")
    p.add_argument("--features", default=None,
                   help="comma-separated beta feature names; default all BETA_SPECS")
    p.add_argument("--drop-originals", action="store_true",
                   help="keep only the beta features, to isolate them")
    p.add_argument("--submit-lag-days", type=int, default=1,
                   help="row date D may only use spot closes dated <= D - this")
    p.add_argument("--max-staleness-days", type=int, default=4,
                   help="attached signal must be within this many days of D")
    return p.parse_args()


def _select_specs(names):
    if not names:
        return BETA_SPECS
    wanted = [n.strip() for n in names.split(",")]
    unknown = [n for n in wanted if n not in BETA_SPECS]
    if unknown:
        raise SystemExit("unknown beta features: %s\nknown: %s"
                         % (unknown, list(BETA_SPECS)))
    return {n: BETA_SPECS[n] for n in wanted}


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")
    specs = _select_specs(args.features)
    beta_cols = list(specs.keys())

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    n_rows = len(base)
    base_feats = [c for c in base.columns if c.startswith("feature_")]
    print("base: %d rows, %d symbols, %d features (%s)"
          % (n_rows, base["symbol"].nunique(), len(base_feats), args.source_version))

    if not os.path.exists(args.spot):
        raise SystemExit("spot file not found: %s" % args.spot)
    spot = pd.read_parquet(args.spot, columns=["symbol", "date", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    print("spot: %d rows, %d symbols (%s)"
          % (len(spot), spot["symbol"].nunique(), args.spot))

    print("computing %d rolling-beta signals vs %s ..."
          % (len(beta_cols), args.market_symbol))
    raw = compute_beta_signals(spot, specs, market_symbol=args.market_symbol)
    raw = raw.drop_duplicates(["symbol", "date"])

    print("attaching as-of (submit lag %d day, max staleness %d days) ..."
          % (args.submit_lag_days, args.max_staleness_days))
    merged = attach_signals_asof(base, raw, beta_cols,
                                 lag_days=args.submit_lag_days,
                                 max_staleness_days=args.max_staleness_days,
                                 period_col=spec["period_col"],
                                 symbol_col=spec["id_col"])
    if len(merged) != n_rows:
        raise SystemExit("as-of merge changed the row count %d -> %d"
                         % (n_rows, len(merged)))

    for c in beta_cols:
        print("   %-26s coverage %5.1f%% of panel rows"
              % (c, 100 * float(merged[c].notna().mean())))

    print("rank-binning to {0,.25,.5,.75,1} per date ...")
    merged = bin_cross_sectional(merged, beta_cols, spec["period_col"])
    for c in beta_cols:
        mass = bin_mass(merged[c])
        drift = float(np.nanmax(np.abs(mass - BIN_MASS)))
        flag = "  <-- OFF" if drift > 0.02 else ""
        print("   %-26s mass %s (max drift %.3f)%s"
              % (c, np.array2string(mass, precision=3, floatmode="fixed"), drift, flag))

    if args.drop_originals:
        merged = merged.drop(columns=base_feats)
        print("dropped the %d original features" % len(base_feats))

    n_feat = sum(c.startswith("feature_") for c in merged.columns)
    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    merged.set_index("symbol").to_parquet(dest)
    print("features: %d total (%d beta) -> %s (%.0f MB)"
          % (n_feat, len(beta_cols), dest, os.path.getsize(dest) / 1e6))

    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % version)
    print("isolate the beta features with:  --feature-include '^feature_beta_'")


if __name__ == "__main__":
    main()
