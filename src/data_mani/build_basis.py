"""
Build a Crypto dataset carrying perp/spot BASIS and funding-carry features, on the
SAME rank-bin scale as the shipped features, merged onto the tournament panel.

    python3 -m src.data_mani.build_basis                  # crypto/v2.0 -> ...-basis (adds 10)
    python3 -m src.data_mani.build_basis --drop-originals # basis/funding only

Run it with:  --data-version crypto/v2.0-basis

Inputs: the spot store (data/crypto/spot/ohlcv.parquet) + the fetched Binance perp
store (data/crypto/perp/{perp_daily,funding_daily}.parquet). The basis is
(perp_close - spot_close)/spot_close; funding is Binance perpetual funding. Three
modules as usual (basis_features = pure math; binning = Numerai's scale; this =
wiring). The join is POINT-IN-TIME (attach_signals_asof, row D uses signals <= D-1);
binning is AFTER the join, per date. Only the Binance-perp subset of symbols has a
basis; the rest stay NaN -> neutral bin at load.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.basis_features import BASIS_FEATURES, compute_basis_signals
from src.data_mani.binning import BIN_MASS, bin_cross_sectional, bin_mass
from src.data_mani.momentum_features import attach_signals_asof
from src.data_mani.utils import (download_if_missing, get_api, get_spec,
                                 local_path)

DEFAULT_SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")
DEFAULT_PERP = os.path.join("data", "crypto", "perp", "perp_daily.parquet")
DEFAULT_FUND = os.path.join("data", "crypto", "perp", "funding_daily.parquet")


def parse_args():
    p = argparse.ArgumentParser(
        description="Add perp/spot basis + funding features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0")
    p.add_argument("--out-suffix", default="-basis")
    p.add_argument("--spot", default=DEFAULT_SPOT)
    p.add_argument("--perp", default=DEFAULT_PERP)
    p.add_argument("--funding", default=DEFAULT_FUND)
    p.add_argument("--drop-originals", action="store_true")
    p.add_argument("--submit-lag-days", type=int, default=1)
    p.add_argument("--max-staleness-days", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    n_rows = len(base)
    base_feats = [c for c in base.columns if c.startswith("feature_")]
    print("base: %d rows, %d symbols, %d features (%s)"
          % (n_rows, base["symbol"].nunique(), len(base_feats), args.source_version))

    for pth in (args.spot, args.perp, args.funding):
        if not os.path.exists(pth):
            raise SystemExit("required input not found: %s" % pth)
    spot = pd.read_parquet(args.spot, columns=["symbol", "date", "close"]).rename(columns={"close": "spot_close"})
    perp = pd.read_parquet(args.perp)[["symbol", "date", "perp_close"]]
    fund = pd.read_parquet(args.funding)[["symbol", "date", "funding_sum"]].rename(columns={"funding_sum": "funding"})
    for d in (spot, perp, fund):
        d["date"] = pd.to_datetime(d["date"])
    print("spot %d sym | perp %d sym | funding %d sym"
          % (spot["symbol"].nunique(), perp["symbol"].nunique(), fund["symbol"].nunique()))

    # assemble the per-(symbol,date) basis panel over the perp-covered rows
    panel = perp.merge(spot, on=["symbol", "date"], how="inner").merge(
        fund, on=["symbol", "date"], how="left")
    print("basis panel: %d rows, %d symbols (perp & spot both present)"
          % (len(panel), panel["symbol"].nunique()))

    print("computing %d basis/funding signals ..." % len(BASIS_FEATURES))
    raw = compute_basis_signals(panel).drop_duplicates(["symbol", "date"])
    print("attaching as-of (submit lag %d day, max staleness %d days) ..."
          % (args.submit_lag_days, args.max_staleness_days))
    merged = attach_signals_asof(base, raw, BASIS_FEATURES,
                                 lag_days=args.submit_lag_days,
                                 max_staleness_days=args.max_staleness_days,
                                 period_col=spec["period_col"], symbol_col=spec["id_col"])
    if len(merged) != n_rows:
        raise SystemExit("as-of merge changed the row count %d -> %d" % (n_rows, len(merged)))

    for c in BASIS_FEATURES:
        print("   %-26s coverage %5.1f%% of panel rows" % (c, 100 * float(merged[c].notna().mean())))

    print("rank-binning to {0,.25,.5,.75,1} per date ...")
    merged = bin_cross_sectional(merged, BASIS_FEATURES, spec["period_col"])
    for c in BASIS_FEATURES:
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
    print("features: %d total (%d basis/funding) -> %s (%.0f MB)"
          % (n_feat, len(BASIS_FEATURES), dest, os.path.getsize(dest) / 1e6))

    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % version)
    print("isolate the basis/funding features with:  --feature-include '^feature_(basis|funding)_'")


if __name__ == "__main__":
    main()
