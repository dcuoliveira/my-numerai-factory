"""
Build a Crypto dataset for the betting-against-beta-under-short-constraints idea.

    python3 -m src.data_mani.build_bab                  # -> data/crypto/v2.0-bab (22 + 9)
    python3 -m src.data_mani.build_bab --drop-originals  # BAB block only

Run it with:  --data-version crypto/v2.0-bab

The block (per window w in 30/60/90d), all rank-binned to {0,.25,.5,.75,1}:

    feature_beta_btc_{w}d          market sensitivity (OLS slope on BTC returns)
    feature_bab_beta_x_illiq_{w}d  pctl(beta) * pctl(Amihud illiquidity)
    feature_bab_beta_x_rvol_{w}d   pctl(beta) * pctl(realised vol)

The interactions are the short-constraint / disagreement CONDITIONING: a coin
that is high-beta AND illiquid (hard to short) AND high-vol (high disagreement)
is where Miller/Hong-Sraer overpricing concentrates -> expected LOW forward
return. Products are formed from per-date CENTERED PERCENTILES (rank_pct - 0.5),
so "both high" is extreme and the result is robust to the heavy skew of Amihud.

Chain mirrors build_momentum/build_vol/build_beta: spot (close+volume) ->
bab_features.compute_bab_inputs -> attach_signals_asof (lag 1 day, external feed)
-> per-date percentile products -> bin_cross_sectional (per panel date) -> write.
Point-in-time (row D uses data through D-1); binning AFTER the merge over the
panel cross-section. INTERPRETER: /home/danielco/miniconda3/bin/python3.
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.bab_features import WINDOWS, compute_bab_inputs
from src.data_mani.binning import BIN_MASS, bin_cross_sectional, bin_mass
from src.data_mani.momentum_features import attach_signals_asof
from src.data_mani.utils import (download_if_missing, get_api, get_spec,
                                 local_path)

DEFAULT_SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")


def parse_args():
    p = argparse.ArgumentParser(
        description="Add rank-binned betting-against-beta (short-constraint) features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0")
    p.add_argument("--out-suffix", default="-bab")
    p.add_argument("--spot", default=DEFAULT_SPOT)
    p.add_argument("--market-symbol", default="BTC")
    p.add_argument("--windows", default=",".join(str(w) for w in WINDOWS),
                   help="comma-separated trailing windows")
    p.add_argument("--drop-originals", action="store_true",
                   help="keep only the BAB block, to isolate it")
    p.add_argument("--submit-lag-days", type=int, default=1)
    p.add_argument("--max-staleness-days", type=int, default=4)
    return p.parse_args()


def _centered_pct(frame, col, period_col):
    """Per-date cross-sectional percentile centered to [-0.5, 0.5]; NaN kept NaN."""
    return frame.groupby(period_col)[col].rank(pct=True) - 0.5


def main():
    args = parse_args()
    windows = [int(w) for w in args.windows.split(",")]
    spec = get_spec("crypto")
    api = get_api("crypto")

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    n_rows = len(base)
    base_feats = [c for c in base.columns if c.startswith("feature_")]
    print("base: %d rows, %d symbols, %d features (%s)"
          % (n_rows, base["symbol"].nunique(), len(base_feats), args.source_version))

    if not os.path.exists(args.spot):
        raise SystemExit("spot file not found: %s" % args.spot)
    spot = pd.read_parquet(args.spot, columns=["symbol", "date", "close", "volume"])
    spot["date"] = pd.to_datetime(spot["date"])
    print("spot: %d rows, %d symbols" % (len(spot), spot["symbol"].nunique()))

    print("computing BAB inputs (beta / Amihud illiq / rvol) vs %s, windows %s ..."
          % (args.market_symbol, windows))
    raw = compute_bab_inputs(spot, windows=tuple(windows),
                             market_symbol=args.market_symbol)
    raw = raw.drop_duplicates(["symbol", "date"])
    raw_cols = [c for c in raw.columns if c.startswith("_")]

    print("attaching as-of (submit lag %d day, max staleness %d days) ..."
          % (args.submit_lag_days, args.max_staleness_days))
    merged = attach_signals_asof(base, raw, raw_cols,
                                 lag_days=args.submit_lag_days,
                                 max_staleness_days=args.max_staleness_days,
                                 period_col=spec["period_col"],
                                 symbol_col=spec["id_col"])
    if len(merged) != n_rows:
        raise SystemExit("as-of merge changed row count %d -> %d" % (n_rows, len(merged)))

    # build the final feature columns: beta, and the two interactions as products
    # of per-date centered percentiles (robust to Amihud's skew).
    period_col = spec["period_col"]
    feat_cols = []
    for w in windows:
        beta = "_beta_%d" % w
        p_beta = _centered_pct(merged, beta, period_col)
        p_illiq = _centered_pct(merged, "_illiq_%d" % w, period_col)
        p_rvol = _centered_pct(merged, "_rvol_%d" % w, period_col)
        fb = "feature_beta_btc_%dd" % w
        fi = "feature_bab_beta_x_illiq_%dd" % w
        fr = "feature_bab_beta_x_rvol_%dd" % w
        merged[fb] = merged[beta]          # continuous beta, binned below
        merged[fi] = p_beta * p_illiq
        merged[fr] = p_beta * p_rvol
        feat_cols += [fb, fi, fr]

    for c in feat_cols:
        print("   %-30s coverage %5.1f%%" % (c, 100 * float(merged[c].notna().mean())))

    print("rank-binning to {0,.25,.5,.75,1} per date ...")
    merged = bin_cross_sectional(merged, feat_cols, period_col)
    for c in feat_cols:
        mass = bin_mass(merged[c])
        drift = float(np.nanmax(np.abs(mass - BIN_MASS)))
        flag = "  <-- OFF" if drift > 0.02 else ""
        print("   %-30s mass %s (drift %.3f)%s"
              % (c, np.array2string(mass, precision=3, floatmode="fixed"), drift, flag))

    merged = merged.drop(columns=raw_cols)  # drop the internal continuous inputs
    if args.drop_originals:
        merged = merged.drop(columns=base_feats)
        print("dropped the %d original features" % len(base_feats))

    n_feat = sum(c.startswith("feature_") for c in merged.columns)
    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    merged.set_index("symbol").to_parquet(dest)
    print("features: %d total (%d BAB) -> %s (%.0f MB)"
          % (n_feat, len(feat_cols), dest, os.path.getsize(dest) / 1e6))

    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % version)
    print("isolate the BAB block with:  --feature-include '^feature_(beta_btc|bab_)'")


if __name__ == "__main__":
    main()
