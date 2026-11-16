"""
Build crypto/v2.0-carver-macro: every shipped v2.0 feature, plus the 15 Carver /
Clenow trend signals and the 18 shared macro features, under the missing-value
standard E134-E137 settled on.

    python3 -m src.data_mani.build_carver_macro
    python3 -m src.data_mani.build_carver_macro --source-version crypto/v2.0 --out-suffix=-carver-macro

Run it with:  --data-version crypto/v2.0-carver-macro

What it holds (55 features):

  feature_*        the 22 shipped features, untouched, on EVERY source row
  feature_tf_*     15 trend signals (trend_signals.TREND_SPECS), recomputed from the
                   CURRENT spot file with build_feed_fillind.recompute_blocks: compute
                   per symbol -> as-of attach (close of D-1, at most 4 days stale) ->
                   rank-bin per date over the panel's own cross-section
  feature_macro_*  18 shared macro features (macro_features.MACRO_SPECS): causal
                   trailing percentile, date-level as-of attach (D-1, 7 days stale)

The missing-value standard, and why it is not -fillind / -feed-fillind:

  * NO ROW IS DROPPED. E132 dropped coins with no feed and gave them a neutral 0.5;
    which coin has a feed is decided by today's coverage, so the row set was
    look-ahead, and the neutral rows alone cost -0.0175 CORR (t -8). Live, the v2.0
    model scores every coin anyway.
  * NO feature_isna_* INDICATORS. A coverage flag built from today's feed marks the
    coins that later died (E131: helpful on dev, sharply harmful on test).
  * A missing cell stays NaN in the file. The loader fills 0.5 (E79: identical to a
    0.5 fillmask), and the evaluation ROUTES: v2.0 ranks every coin and this
    dataset's model reorders only the coins whose trend block is observed that day,
    inside the rank slots they already hold (experiments/e134_route_feature_sets.py).
    So a coin's missing block can change its own score only through training rows.

The spine is crypto/v2.0 itself, not v2.0-momswap-macro-carver: that dataset dropped
the eight base momentum proxies, and this one keeps all 22. On the 468,000 keys E133-
E137 score (2020-08-28 .. 2026-08-20) v2.0's features, ucid and target equal E133's
panel cell for cell, so their B0 / tenure / retuned-B0 predictions apply unchanged.
v2.0 is live-refreshed: rows past the end of a coin's spot series have an empty trend
block, which is warm-down, known at the time, and routed like warm-up.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.build_feed_fillind import PRICE, SPOT, coins_without_prices, recompute_blocks
from src.data_mani.macro_features import MACRO_SPECS, attach_macro_asof, compute_macro_signals
from src.data_mani.trend_signals import TREND_SPECS
from src.data_mani.utils import download_if_missing, get_api, get_spec, local_path

MACRO = os.path.join("data", "crypto", "macro", "macro_daily.parquet")
MACRO_LAG, MACRO_STALE, MACRO_MIN = 1, 7, 252     # build_macro's defaults


def build(base, spot, macro_daily, period="date", symbol="symbol"):
    """Return (panel with the two blocks added, as-of close each row saw)."""
    order = list(base.columns)
    tf = list(TREND_SPECS)
    clash = [c for c in order if c.startswith(("feature_tf_", "feature_macro_", "feature_mom_"))]
    if clash:
        raise SystemExit("source already carries engineered columns: %s" % clash[:5])
    panel = base.assign(**{c: np.nan for c in tf})
    panel = recompute_blocks(panel, spot, period, symbol)
    price = panel.pop(PRICE)
    macro = list(MACRO_SPECS)
    feats = compute_macro_signals(macro_daily, MACRO_SPECS, normalize=True, min_periods=MACRO_MIN)
    out = attach_macro_asof(panel, feats, macro, lag_days=MACRO_LAG, max_staleness_days=MACRO_STALE,
                            period_col=period)
    if len(out) != len(base):
        raise SystemExit("macro attach changed the row count %d -> %d" % (len(base), len(out)))
    return out[order + tf + macro], price.values


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0")
    p.add_argument("--out-suffix", default="-carver-macro")
    p.add_argument("--spot", default=SPOT)
    p.add_argument("--macro", default=MACRO)
    return p.parse_args()


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")
    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    base["ucid"] = base["ucid"].astype(str)
    n_src = sum(c.startswith("feature_") for c in base)
    print("source: %d rows, %d symbols, %d features, %s .. %s (%s)"
          % (len(base), base["symbol"].nunique(), n_src, base["date"].min().date(),
             base["date"].max().date(), args.source_version))
    spot = pd.read_parquet(args.spot, columns=["symbol", "date", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    macro_daily = pd.read_parquet(args.macro)
    out, price = build(base, spot, macro_daily, spec["period_col"], spec["id_col"])

    tf, macro = list(TREND_SPECS), list(MACRO_SPECS)
    has = pd.Series(price).notna().values
    nofeed = coins_without_prices(out.assign(**{PRICE: price})).values
    obs = out[tf].notna().any(axis=1).values
    print("rows %d: as-of price %.1f%%, coin with no feed at all %.1f%% (kept, routed to v2.0), "
          "trend block observed %.1f%%" % (len(out), 100 * has.mean(), 100 * nofeed.mean(), 100 * obs.mean()))
    for c in tf + macro:
        print("   %-32s observed %5.1f%%" % (c, 100 * out[c].notna().mean()))

    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.set_index("symbol").to_parquet(dest)
    print("wrote %d rows, %d features (%d shipped + %d trend + %d macro) -> %s (%.0f MB)"
          % (len(out), n_src + len(tf) + len(macro), n_src, len(tf), len(macro), dest,
             os.path.getsize(dest) / 1e6))
    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)
    print("run it with:  --data-version %s" % version)


if __name__ == "__main__":
    main()
