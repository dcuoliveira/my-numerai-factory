"""
Build a Crypto dataset that carries realised-volatility estimators, on the SAME
rank-bin scale as the shipped features, merged onto the tournament panel.

    python3 -m src.data_mani.build_vol                  # -> data/crypto/v2.0-vol/ (adds 22 to the 22)
    python3 -m src.data_mani.build_vol --drop-originals # volatility only, for isolation

Run it with:  --data-version crypto/v2.0-vol

The block is 22 volatility features (src/data_mani/volatility_features.py): four
estimators over a hyperparameter grid --
    squared-returns realised variance   feature_vol_sqret_{5,10,20,40,60,120}d
    EWMA / RiskMetrics variance         feature_vol_ewma_lam{90,94,97,99}
    Parkinson (high/low range)          feature_vol_park_{5,10,20,40,60,120}d
    Yang-Zhang (full OHLC + prev close) feature_vol_yz_{5,10,20,40,60,120}d

------------------------------------------------------------------------------
The chain, and why it is three modules  (same as build_momentum / build_trend)
------------------------------------------------------------------------------

    spot OHLCV ──▶ volatility_features.compute_volatility_signals ──▶ raw signals
       │                                                                │
    (data/crypto/spot/ohlcv.parquet)                        merge onto the panel
                                                                        │
    base panel ─────────────────────────────────────────────────────── ┤
    (crypto/v2.0/train.parquet)                                         ▼
                                          binning.bin_cross_sectional (per date)
                                                                        │
                                                                        ▼
                                        data/crypto/v2.0-vol/train.parquet (drop-in)

* ``volatility_features`` is pure OHLC math -- no file layout, no Numerai scale.
* ``binning`` is the ONE place Numerai's ``{0,.25,.5,.75,1}`` quantization lives.
* this file is the offline wiring: where the inputs are, where the output goes.

Unlike build_momentum/build_trend (which read close only), this builder reads the
full daily OHLC, because Parkinson needs the high/low and Yang-Zhang needs the
open/high/low/close plus the previous close.

------------------------------------------------------------------------------
The join is POINT-IN-TIME, lagged by the submission constraint
------------------------------------------------------------------------------

The volatility on round date D must use only prices known when we submit round D.
Our spot feed is an external daily source whose bar for day D does not close until
the end of day D, but the round is submitted DURING day D -- so the newest usable
close is the last completed bar, D-1. The join is therefore an as-of backward
merge with a one-day lag (``attach_signals_asof``); an exact same-date join would
attach day-D prices to row D and quietly leak information not final at submission.
This is identical to the momentum/trend builders and keeps the backtest honest.

------------------------------------------------------------------------------
Binning happens AFTER the join, on the panel's cross-section
------------------------------------------------------------------------------

The rank-bin of a symbol at date t is taken over the SAME universe the model scores
on -- the tournament panel's symbols on that date -- not over whatever set the spot
feed covered. Compute raw estimates per symbol, attach as-of, then bin per date. A
symbol with no spot price, or too little history for a window, is NaN after the
join and stays NaN through the bin; ``fill_missing_features`` puts the neutral 0.5
there at load. Base rows and base feature columns are never touched -- the panel is
a superset.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.binning import BIN_MASS, bin_cross_sectional, bin_mass
from src.data_mani.momentum_features import attach_signals_asof
from src.data_mani.volatility_features import (VOL_SPECS,
                                               compute_volatility_signals)
from src.data_mani.utils import (download_if_missing, get_api, get_spec,
                                 local_path)

DEFAULT_SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")


def parse_args():
    p = argparse.ArgumentParser(
        description="Add rank-binned realised-volatility features from spot OHLCV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0",
                   help="dataset the volatility block is added onto")
    p.add_argument("--out-suffix", default="-vol",
                   help="output dir suffix; the variant is <source-version><suffix>")
    p.add_argument("--spot", default=DEFAULT_SPOT,
                   help="spot OHLCV parquet with symbol,date,open,high,low,close")
    p.add_argument("--features", default=None,
                   help="comma-separated volatility feature names to build; "
                        "default is all of VOL_SPECS")
    p.add_argument("--drop-originals", action="store_true",
                   help="keep only the volatility features, so a comparison "
                        "isolates them rather than adding to the base")
    p.add_argument("--submit-lag-days", type=int, default=1,
                   help="row date D may only use spot bars dated <= D - this. "
                        "1 = the last completed daily bar, the newest price "
                        "guaranteed known when round D is submitted")
    p.add_argument("--max-staleness-days", type=int, default=4,
                   help="the attached spot signal must be within this many days "
                        "of D, so a delisted token cannot carry a months-stale "
                        "value forward; else the row is left neutral")
    return p.parse_args()


def _select_specs(names):
    if not names:
        return VOL_SPECS
    wanted = [n.strip() for n in names.split(",")]
    unknown = [n for n in wanted if n not in VOL_SPECS]
    if unknown:
        raise SystemExit("unknown volatility features: %s\nknown: %s"
                         % (unknown, list(VOL_SPECS)))
    return {n: VOL_SPECS[n] for n in wanted}


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")
    specs = _select_specs(args.features)
    vol_cols = list(specs.keys())

    # ---- base tournament panel -------------------------------------------- #
    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    n_rows = len(base)
    base_feats = [c for c in base.columns if c.startswith("feature_")]
    print("base: %d rows, %d symbols, %d features (%s)"
          % (n_rows, base["symbol"].nunique(), len(base_feats), args.source_version))

    # ---- spot OHLCV -------------------------------------------------------- #
    if not os.path.exists(args.spot):
        raise SystemExit("spot file not found: %s\nbuild it first (fetch spot "
                         "OHLCV into data/crypto/spot/ohlcv.parquet)" % args.spot)
    spot = pd.read_parquet(
        args.spot, columns=["symbol", "date", "open", "high", "low", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    print("spot: %d rows, %d symbols (%s)"
          % (len(spot), spot["symbol"].nunique(), args.spot))

    # ---- raw volatility, then POINT-IN-TIME join onto the panel ------------ #
    print("computing %d volatility estimators ..." % len(vol_cols))
    raw = compute_volatility_signals(spot, specs)
    raw = raw.drop_duplicates(["symbol", "date"])
    print("attaching as-of (submit lag %d day, max staleness %d days) ..."
          % (args.submit_lag_days, args.max_staleness_days))
    merged = attach_signals_asof(base, raw, vol_cols,
                                 lag_days=args.submit_lag_days,
                                 max_staleness_days=args.max_staleness_days,
                                 period_col=spec["period_col"],
                                 symbol_col=spec["id_col"])
    if len(merged) != n_rows:
        raise SystemExit("as-of merge changed the row count %d -> %d"
                         % (n_rows, len(merged)))

    for c in vol_cols:
        cov = float(merged[c].notna().mean())
        print("   %-26s coverage %5.1f%% of panel rows" % (c, 100 * cov))

    # ---- bin onto Numerai's scale, over the panel's per-date cross-section - #
    print("rank-binning to {0,.25,.5,.75,1} per date ...")
    merged = bin_cross_sectional(merged, vol_cols, spec["period_col"])
    for c in vol_cols:
        mass = bin_mass(merged[c])
        drift = float(np.nanmax(np.abs(mass - BIN_MASS)))
        flag = "  <-- OFF" if drift > 0.02 else ""
        print("   %-26s mass %s (max drift %.3f)%s"
              % (c, np.array2string(mass, precision=3, floatmode="fixed"),
                 drift, flag))

    if args.drop_originals:
        merged = merged.drop(columns=base_feats)
        print("dropped the %d original features" % len(base_feats))

    n_feat = sum(c.startswith("feature_") for c in merged.columns)
    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    merged.set_index("symbol").to_parquet(dest)
    print("features: %d total (%d volatility) -> %s (%.0f MB)"
          % (n_feat, len(vol_cols), dest, os.path.getsize(dest) / 1e6))

    # the reference / meta-model file must travel with the variant, same as every
    # other builder, or load_reference/attach_meta_model can't find it.
    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % version)
    print("isolate the volatility features with:  "
          "--feature-include '^feature_vol_'")


if __name__ == "__main__":
    main()
