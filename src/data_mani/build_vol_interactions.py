"""
Add vol x carver, vol x inflation, and vol x carver x inflation INTERACTIONS to the
combined vol/infl/carver dataset, computed on the RAW signals and rank-binned AFTER
the product (the xinteractions discipline), then fillmasked.

    python3 -m src.data_mani.build_vol_interactions
        # crypto/v2.0-vol-infl-carver-fillmask -> ...-ix-fillmask (adds 682)

Run it with:  --data-version crypto/v2.0-vol-infl-carver-ix-fillmask

------------------------------------------------------------------------------
The three interaction families, and why inflation is a SIGNED regime gate
------------------------------------------------------------------------------

  vol x carver          22 vol x 15 carver              = 330
  vol x inflation       22 vol x 1 inflation regime     =  22
  vol x carver x infl   22 vol x 15 carver x 1 regime   = 330
                                                    total 682

Every product is formed on the RAW, pre-binning signals and only THEN rank-binned
per date onto {0,.25,.5,.75,1} -- exactly build_xinteractions' rule, for the same
reason: binning is invariant to a positive per-date scale and reversed by a negative
one, so the order (product first, bin second) is what turns a within-date-constant
factor into a cross-sectionally active one.

* vol and carver both VARY across symbols within a date, so vol_raw x carver_raw is a
  genuine new cross-sectional ordering (high vol AND strong trend -> large).

* Inflation is a MACRO feature: the same value for every symbol on a date. A plain
  product with its [0,1] percentile (always positive) would collapse straight back to
  the other factor under per-date ranking (the degeneracy build_xinteractions warns
  about). So inflation enters as ONE SIGNED regime factor,
      infl_regime(date) = mean(bei_5y, bei_10y, bei_30y, bei_5y5y percentile) - 0.5,
  in [-0.5, 0.5]: positive when inflation expectations sit above their own history,
  negative below. Its 12 BEI columns are ~one axis, so a single composite carries the
  regime without the 12x blow-up. Multiplying by it flips the cross-sectional ranking
  in below-median-inflation regimes and keeps it above -- "vol (or vol x carver),
  sign-switched by the inflation regime."

Point-in-time: the raw vol and carver signals are attached as-of <= D-1 per symbol
(the submission lag), and the inflation regime is read from the already-PIT infl
columns of the source dataset, so the products use only information known at
submission -- backtest == live, same as every other block.

Fillmask: a product is NaN wherever either raw signal is missing (a young token, a
no-spot symbol); after binning it stays NaN and is set to the neutral 0.5, so the
interactions follow the same fillmask logic as the rest of the dataset.
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
# the break-even LEVEL columns whose centred mean is the inflation regime factor
INFL_LEVEL_COLS = ["feature_macro_bei_5y", "feature_macro_bei_10y",
                   "feature_macro_bei_30y", "feature_macro_bei_5y5y"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Add vol x carver x inflation interactions (raw product, binned).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0-vol-infl-carver-fillmask",
                   help="combined dataset the interactions are added onto")
    p.add_argument("--out-version", default="crypto/v2.0-vol-infl-carver-ix-fillmask")
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

    # ---- source combined dataset ------------------------------------------ #
    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base[per] = pd.to_datetime(base[per])
    n_rows = len(base)
    src_feats = [c for c in base.columns if c.startswith("feature_")]
    miss = [c for c in INFL_LEVEL_COLS if c not in base.columns]
    if miss:
        raise SystemExit("source lacks inflation level cols %s" % miss)
    print("source: %d rows, %d features (%s)" % (n_rows, len(src_feats), args.source_version))

    # ---- RAW vol and carver signals, attached point-in-time (temp cols) ---- #
    spot = pd.read_parquet(args.spot, columns=["symbol", "date", "open", "high", "low", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    print("computing raw vol (%d) and carver (%d) signals ..."
          % (len(VOL_SPECS), len(TREND_SPECS)))
    rawv = compute_volatility_signals(spot, VOL_SPECS).drop_duplicates(["symbol", "date"])
    rawc = compute_trend_signals(spot[["symbol", "date", "close"]], TREND_SPECS).drop_duplicates(["symbol", "date"])
    vol_names = list(VOL_SPECS)         # feature_vol_*
    car_names = list(TREND_SPECS)       # feature_tf_*
    rawv = rawv.rename(columns={c: "_rv_" + c for c in vol_names})
    rawc = rawc.rename(columns={c: "_rc_" + c for c in car_names})

    print("attaching raw signals as-of <= D-%d ..." % args.submit_lag_days)
    base = attach_signals_asof(base, rawv, ["_rv_" + c for c in vol_names],
                               lag_days=args.submit_lag_days,
                               max_staleness_days=args.max_staleness_days,
                               period_col=per, symbol_col=idc)
    base = attach_signals_asof(base, rawc, ["_rc_" + c for c in car_names],
                               lag_days=args.submit_lag_days,
                               max_staleness_days=args.max_staleness_days,
                               period_col=per, symbol_col=idc)

    # ---- inflation regime: signed composite, date-level, from the PIT infl -- #
    base["_infl"] = base[INFL_LEVEL_COLS].mean(axis=1) - 0.5

    # ---- raw products (bin AFTER) ----------------------------------------- #
    def sv(c): return c[len("feature_vol_"):]
    def sc(c): return c[len("feature_tf_"):]
    # Build every product into a dict and concat ONCE -- inserting 682 columns one
    # at a time fragments the frame (a flood of PerformanceWarnings and slow).
    print("forming raw products ...")
    infl = base["_infl"]
    ix = {}
    for v in vol_names:
        rv = base["_rv_" + v]
        ix["feature_ix_%s_x_infl" % sv(v)] = rv * infl          # vol x inflation regime
        for c in car_names:
            vc = rv * base["_rc_" + c]                            # vol x carver
            ix["feature_ix_%s_x_%s" % (sv(v), sc(c))] = vc
            ix["feature_ix_%s_x_%s_x_infl" % (sv(v), sc(c))] = vc * infl  # x inflation
    ix_cols = list(ix)
    base = pd.concat([base, pd.DataFrame(ix, index=base.index)], axis=1)
    print("formed %d interactions (vc=%d, vi=%d, vci=%d)"
          % (len(ix_cols), len(vol_names) * len(car_names), len(vol_names),
             len(vol_names) * len(car_names)))

    # ---- rank-bin each interaction per date ------------------------------- #
    print("rank-binning %d interactions per date (this is the slow step) ..." % len(ix_cols))
    base = bin_cross_sectional(base, ix_cols, per)
    drift = max(float(np.nanmax(np.abs(bin_mass(base[c]) - BIN_MASS))) for c in ix_cols[:50])
    print("   mass drift (first 50) max %.3f" % drift)

    # ---- drop temporaries, fillmask, write -------------------------------- #
    base = base.drop(columns=["_infl"] + ["_rv_" + c for c in vol_names]
                     + ["_rc_" + c for c in car_names])
    feats = [c for c in base.columns if c.startswith("feature_")]
    n_missing = int(base[ix_cols].isna().sum().sum())
    base = apply_fillmask(base, feats, fill_value=args.fill_value)
    print("fillmask: filled %d missing interaction cells with %s; present kept"
          % (n_missing, args.fill_value))

    dest = local_path("crypto", args.out_version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    base.set_index(idc).to_parquet(dest)
    print("features: %d total (%d source + %d interactions) -> %s (%.0f MB)"
          % (len(feats), len(src_feats), len(ix_cols), dest, os.path.getsize(dest) / 1e6))

    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", args.out_version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % args.out_version)
    print("isolate the interactions with:  --feature-include '^feature_ix_'")


if __name__ == "__main__":
    main()
