"""
Add the SHARED inflation block -- break-even inflation LEVELS and the MOMENTUM of
those levels -- to a Crypto dataset, merged POINT-IN-TIME.

    python3 -m src.data_mani.build_inflation   # v2.0-momswap-macro -> ...-infl

Run it with:  --data-version crypto/v2.0-momswap-macro-infl

Break-even inflation = nominal Treasury yield - TIPS real yield, the daily
market-implied expected inflation; plus the 5y5y forward. Both sourced from the
US Treasury daily par + real yield curves (home.treasury.gov, public-domain,
keyless -- the same store the macro block reads, now carrying the real yields
r5/r10/r30 alongside the nominal y*). These are SHARED across every crypto, so
they use the same causal-trailing-percentile normalisation and date-level as-of
broadcast as the rest of the macro block (macro_features.py) -- NOT cross-sectional
binning, which would collapse a constant-per-date feature. Honest caveat, as for
the whole macro block: ~zero standalone within-era IC; helps via regime /
interactions. Point-in-time: row D uses values dated <= D-1.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.macro_features import (INFLATION_SPECS, attach_macro_asof,
                                          compute_macro_signals)
from src.data_mani.utils import (download_if_missing, get_api, get_spec,
                                 local_path)

DEFAULT_MACRO = os.path.join("data", "crypto", "macro", "macro_daily.parquet")


def parse_args():
    p = argparse.ArgumentParser(
        description="Add break-even inflation levels + their momentum (shared).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0-momswap-macro")
    p.add_argument("--out-suffix", default="-infl")
    p.add_argument("--macro", default=DEFAULT_MACRO)
    p.add_argument("--submit-lag-days", type=int, default=1)
    p.add_argument("--max-staleness-days", type=int, default=7)
    p.add_argument("--min-periods", type=int, default=252)
    p.add_argument("--drop-source-features", action="store_true",
                   help="keep ONLY the inflation features (drop every source "
                        "feature_ column) -- an inflation-only dataset")
    return p.parse_args()


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")
    infl_cols = list(INFLATION_SPECS.keys())

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    n_rows = len(base)
    src_feats = [c for c in base.columns if c.startswith("feature_")]
    print("source: %d rows, %d features (%s)" % (n_rows, len(src_feats), args.source_version))

    if not os.path.exists(args.macro):
        raise SystemExit("macro store not found: %s" % args.macro)
    macro_daily = pd.read_parquet(args.macro)
    need = {"y5", "y10", "y30", "r5", "r10", "r30"}
    missing = need - set(macro_daily.columns)
    if missing:
        raise SystemExit("macro store lacks the real yields %s -- augment it with "
                         "the Treasury daily_treasury_real_yield_curve first" % sorted(missing))
    print("macro store: %d rows, %s .. %s"
          % (len(macro_daily), pd.to_datetime(macro_daily["date"]).min().date(),
             pd.to_datetime(macro_daily["date"]).max().date()))

    print("computing %d inflation features (break-even levels + their momentum, "
          "causal percentile) ..." % len(infl_cols))
    feats = compute_macro_signals(macro_daily, INFLATION_SPECS, normalize=True,
                                  min_periods=args.min_periods)
    print("attaching as-of (submit lag %d day, max staleness %d days), broadcast ..."
          % (args.submit_lag_days, args.max_staleness_days))
    merged = attach_macro_asof(base, feats, infl_cols,
                               lag_days=args.submit_lag_days,
                               max_staleness_days=args.max_staleness_days,
                               period_col=spec["period_col"])
    if len(merged) != n_rows:
        raise SystemExit("attach changed the row count %d -> %d" % (n_rows, len(merged)))

    sample_date = merged[spec["period_col"]].dropna().iloc[len(merged) // 2]
    day = merged[merged[spec["period_col"]] == sample_date]
    for c in infl_cols:
        cov = float(merged[c].notna().mean())
        nu = int(day[c].nunique(dropna=True))
        flag = "" if nu <= 1 else "  <-- NOT shared!"
        print("   %-28s coverage %5.1f%%  (%d distinct/date)%s" % (c, 100 * cov, nu, flag))

    if args.drop_source_features:
        merged = merged.drop(columns=src_feats)
        print("dropped %d source features -> inflation-only (%d features)"
              % (len(src_feats), len(infl_cols)))

    n_feat = sum(c.startswith("feature_") for c in merged.columns)
    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    merged.set_index("symbol").to_parquet(dest)
    print("features: %d total (%d source + %d inflation) -> %s (%.0f MB)"
          % (n_feat, len(src_feats), len(infl_cols), dest, os.path.getsize(dest) / 1e6))

    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % version)
    print("isolate the inflation block with:  --feature-include '^feature_macro_bei_'")


if __name__ == "__main__":
    main()
