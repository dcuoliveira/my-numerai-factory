"""
Add MACRO-momentum x CRYPTO-TREND interactions to a dataset that carries the Carver /
Clenow trend block: crypto/v2.0-carver-macro-infl -> ...-xinteractions.

    python3 -m src.data_mani.build_trend_xinteractions
    python3 -m src.data_mani.build_trend_xinteractions --source-version crypto/v2.0-carver-macro-infl --out-suffix=-xinteractions

Run it with:  --data-version crypto/v2.0-carver-macro-infl-xinteractions

Same construction as build_xinteractions (read its docstring): product of the RAW
signals, rank-binned per date AFTER the product, so the binned interaction is the
crypto signal's ranking, sign-flipped on dates when the macro signal is negative.
What differs is the inputs:

  crypto side   the 15 raw trend signals (trend_signals.TREND_SPECS), since this
                dataset has no litmom feature_mom_ block. Same as-of attach as the
                binned block: close of D-1, at most 4 days stale.
  macro side    every SIGNED macro momentum: the 10 return specs (gold / silver /
                S&P / DXY) and the 8 break-even inflation changes. Levels, yields and
                slopes are not momentum and are left out. Attached D-1, 7 days stale.

15 x 18 = 270 columns feature_ix_<trend>_X_<macro>.

Missing values follow the E134-E138 standard: no row is dropped and no indicator is
added. A product is NaN wherever the trend signal is (no feed, warm-up), so the
interaction block has exactly the trend block's coverage, the loader fills 0.5, and
the evaluation routes those coins to v2.0.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.binning import BIN_MASS, bin_cross_sectional, bin_mass
from src.data_mani.build_feed_fillind import LAG, SPOT, STALE
from src.data_mani.macro_features import (INFLATION_SPECS, MACRO_SPECS, attach_macro_asof,
                                          compute_macro_signals)
from src.data_mani.momentum_features import attach_signals_asof
from src.data_mani.trend_signals import TREND_SPECS, compute_trend_signals
from src.data_mani.utils import download_if_missing, get_api, get_spec, local_path

MACRO = os.path.join("data", "crypto", "macro", "macro_daily.parquet")
MACRO_STALE = 7
MACRO_MOM = dict([(k, v) for k, v in MACRO_SPECS.items() if v["kind"] == "ret"]
                 + [(k, v) for k, v in INFLATION_SPECS.items() if v["kind"].endswith("_change")])


def ix_name(t, m):
    return "feature_ix_" + t.replace("feature_tf_", "") + "_X_" + m.replace("feature_macro_", "")


def add_interactions(base, spot, macro_daily, period="date", symbol="symbol"):
    """Return ``base`` plus the 270 binned interactions, row order kept."""
    tnames, mnames = list(TREND_SPECS), list(MACRO_MOM)
    n = len(base)
    rawc = compute_trend_signals(spot, TREND_SPECS).drop_duplicates(["symbol", "date"])
    rawc = rawc.rename(columns={c: "_rawc_" + c for c in tnames})
    out = attach_signals_asof(base, rawc, ["_rawc_" + c for c in tnames], lag_days=LAG,
                              max_staleness_days=STALE, period_col=period, symbol_col=symbol)
    rawm = compute_macro_signals(macro_daily, MACRO_MOM, normalize=False)
    rawm = rawm.rename(columns={c: "_rawm_" + c for c in mnames})
    out = attach_macro_asof(out, rawm, ["_rawm_" + c for c in mnames], lag_days=LAG,
                            max_staleness_days=MACRO_STALE, period_col=period)
    if len(out) != n:
        raise SystemExit("as-of attach changed the row count %d -> %d" % (n, len(out)))
    ix = {ix_name(t, m): out["_rawc_" + t].values * out["_rawm_" + m].values for t in tnames for m in mnames}
    out = pd.concat([out[list(base.columns)], pd.DataFrame(ix, index=out.index)], axis=1)
    out = bin_cross_sectional(out, list(ix), period)
    return out, list(ix)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0-carver-macro-infl")
    p.add_argument("--out-suffix", default="-xinteractions")
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
    print("source: %d rows, %d features (%s)" % (len(base), n_src, args.source_version))
    spot = pd.read_parquet(args.spot, columns=["symbol", "date", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    out, ix = add_interactions(base, spot, pd.read_parquet(args.macro), spec["period_col"], spec["id_col"])
    tf = [c for c in out if c.startswith("feature_tf_")]
    if tf:
        same = (out[ix].notna().any(axis=1) == out[tf].notna().any(axis=1)).mean()
        print("rows where interaction coverage equals trend coverage: %.4f" % same)
    drift = max(float(np.nanmax(np.abs(bin_mass(out[c]) - BIN_MASS))) for c in ix)
    print("%d interactions, observed %.1f%% of cells, max bin-mass drift %.4f"
          % (len(ix), 100 * out[ix].notna().to_numpy().mean(), drift))
    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.set_index("symbol").to_parquet(dest)
    print("wrote %d rows, %d features (%d source + %d interactions) -> %s (%.0f MB)"
          % (len(out), n_src + len(ix), n_src, len(ix), dest, os.path.getsize(dest) / 1e6))
    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)
    print("run it with:  --data-version %s" % version)


if __name__ == "__main__":
    main()
