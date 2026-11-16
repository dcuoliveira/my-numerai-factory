"""
Add SHARED macro / cross-asset features (gold, silver, US yields 1/2/5/10/30,
yield slopes, S&P momentum) to a Crypto dataset, merged POINT-IN-TIME.

    python3 -m src.data_mani.build_macro                       # v2.0-momswap -> v2.0-momswap-macro
    python3 -m src.data_mani.build_macro --source-version crypto/v2.0  # onto raw

Run it with:  --data-version crypto/v2.0-momswap-macro

------------------------------------------------------------------------------
What this is, and how it differs from build_momentum
------------------------------------------------------------------------------

The momentum block is PER-SYMBOL (each coin's own price path). Macro features are
PER-DATE and SHARED across every coin that day -- gold, the Treasury curve and the
S&P are the same for all symbols. So the two builders differ in exactly two places
(everything else is the same point-in-time discipline):

  * NORMALISATION. A shared feature is constant across the cross-section, so it
    CANNOT be rank-binned within a date (every symbol would tie -> all 0.5). It is
    normalised across TIME instead: a causal expanding percentile in [0,1], 0.5 =
    running median. See src/data_mani/macro_features.py.
  * ATTACH. A DATE-level as-of join (attach_macro_asof) broadcasts the one macro
    row to all symbols on that date, with a wider staleness window (7d) because the
    macro series are business-day while crypto is 24/7.

Read macro_features.py for the honest caveat: a within-date-constant feature has
~zero standalone within-era IC and helps only via regime-conditioning /
interactions. These are added as inputs for the tree/NN models to gate on.

Source of the raw macro series: data/crypto/macro/macro_daily.parquet, built by
the macro fetcher (US Treasury daily par yield curve for yields; yfinance GC=F /
SI=F / ^GSPC for gold / silver / S&P). Same file the live path would read.

PRODUCTION: the live path reads macro_daily up to D-1, calls compute_macro_signals
then attach_macro_asof with the SAME lag -- offline and live share the arithmetic,
so backtest == live. The only precondition is a macro feed that delivers the last
business day's values before the round deadline.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.macro_features import (MACRO_SPECS, attach_macro_asof,
                                          compute_macro_signals)
from src.data_mani.utils import (download_if_missing, get_api, get_spec,
                                 local_path)

DEFAULT_MACRO = os.path.join("data", "crypto", "macro", "macro_daily.parquet")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add shared macro features (metals, yields, slopes, S&P).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-version", default="crypto/v2.0-momswap",
                        help="dataset to add the macro block onto")
    parser.add_argument("--out-suffix", default="-macro")
    parser.add_argument("--macro", default=DEFAULT_MACRO,
                        help="raw macro daily store (date + gold/silver/sp500/y*)")
    parser.add_argument("--submit-lag-days", type=int, default=1,
                        help="row date D may use macro dated <= D - this (1 = the "
                             "last completed business day, known at submission)")
    parser.add_argument("--max-staleness-days", type=int, default=7,
                        help="attached macro must be within this many days of D, "
                             "wide enough to bridge weekends/holidays (business-day "
                             "series vs 24/7 crypto); else the row is neutral")
    parser.add_argument("--min-periods", type=int, default=252,
                        help="history required before the causal percentile emits")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")
    macro_cols = list(MACRO_SPECS.keys())

    # ---- source dataset (momswap by default) ------------------------------ #
    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    n_rows = len(base)
    src_feats = [c for c in base.columns if c.startswith("feature_")]
    print("source: %d rows, %d features (%s)"
          % (n_rows, len(src_feats), args.source_version))

    # ---- raw macro store --------------------------------------------------- #
    if not os.path.exists(args.macro):
        raise SystemExit("macro store not found: %s\nfetch it first (gold/silver/"
                         "sp500 via yfinance, yields via US Treasury)" % args.macro)
    macro_daily = pd.read_parquet(args.macro)
    print("macro store: %d rows, %s .. %s, series %s"
          % (len(macro_daily), pd.to_datetime(macro_daily["date"]).min().date(),
             pd.to_datetime(macro_daily["date"]).max().date(),
             [c for c in macro_daily.columns if c != "date"]))

    # ---- compute + causal cross-time normalise ----------------------------- #
    print("computing %d macro features (causal trailing percentile) ..."
          % len(macro_cols))
    macro_feats = compute_macro_signals(macro_daily, MACRO_SPECS, normalize=True,
                                        min_periods=args.min_periods)

    # ---- broadcast onto the panel, point-in-time --------------------------- #
    print("attaching as-of (submit lag %d day, max staleness %d days), broadcast "
          "across symbols ..." % (args.submit_lag_days, args.max_staleness_days))
    merged = attach_macro_asof(base, macro_feats, macro_cols,
                               lag_days=args.submit_lag_days,
                               max_staleness_days=args.max_staleness_days,
                               period_col=spec["period_col"])
    if len(merged) != n_rows:
        raise SystemExit("attach changed the row count %d -> %d" % (n_rows, len(merged)))

    # report coverage + confirm each macro column is constant WITHIN a date
    sample_date = merged[spec["period_col"]].dropna().iloc[len(merged) // 2]
    day = merged[merged[spec["period_col"]] == sample_date]
    for c in macro_cols:
        cov = float(merged[c].notna().mean())
        n_unique = int(day[c].nunique(dropna=True))
        flag = "" if n_unique <= 1 else "  <-- NOT shared!"
        print("   %-28s coverage %5.1f%%  (%d distinct value on a date)%s"
              % (c, 100 * cov, n_unique, flag))

    n_feat = sum(c.startswith("feature_") for c in merged.columns)
    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    merged.set_index("symbol").to_parquet(dest)
    print("features: %d total (%d source + %d macro) -> %s (%.0f MB)"
          % (n_feat, len(src_feats), len(macro_cols), dest,
             os.path.getsize(dest) / 1e6))

    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % version)
    print("isolate the macro block with:  --feature-include '^feature_macro_'")


if __name__ == "__main__":
    main()
