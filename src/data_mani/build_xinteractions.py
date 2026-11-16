"""
Build a Crypto dataset of MACRO-momentum x CRYPTO-momentum INTERACTIONS, computed
on the RAW signals and rank-binned AFTER the product.

    python3 -m src.data_mani.build_xinteractions   # v2.0-momswap-macro -> ...-xinteractions

Run it with:  --data-version crypto/v2.0-momswap-macro-xinteractions

------------------------------------------------------------------------------
What "interaction before binning" means, and what it actually encodes
------------------------------------------------------------------------------

For every (crypto momentum feature c, macro momentum feature m) we form

    interaction(symbol, date) = c_raw(symbol, date) * m_raw(date)

on the RAW, pre-binning signals -- c_raw is the continuous per-symbol crypto
momentum (return / MAX / vol-scaled / robust-trend, NOT its 5-bin form), m_raw is
the continuous SIGNED macro momentum RETURN (gold/silver/S&P/DXY, NOT its [0,1]
percentile) -- and only THEN rank-bin the product per date onto {0,.25,.5,.75,1}.

The order matters and has a precise consequence. m_raw(date) is the SAME for every
symbol on a date, so the product is the per-symbol crypto signal times a per-date
scalar. Cross-sectional ranking is invariant to a positive per-date scale and
reversed by a negative one, so:

    binned interaction  =  binned crypto momentum        when m_raw(date) > 0
                        =  REVERSED binned crypto momentum when m_raw(date) < 0

i.e. the macro momentum's MAGNITUDE washes out and only its SIGN survives -- the
feature is "crypto momentum, sign-flipped by the macro regime." That is the
regime-switch the macro-crypto evidence supports (macro moves the market beta, not
the cross-section), and it is the whole point of interacting BEFORE binning: it
turns a within-date-constant macro signal (which is inert cross-sectionally on its
own) into a cross-sectionally active one. Using the macro [0,1] percentile instead
of the signed return would be degenerate -- always positive, so every interaction
would collapse straight back to the plain crypto momentum.

------------------------------------------------------------------------------
Point-in-time
------------------------------------------------------------------------------

Both raw inputs are attached POINT-IN-TIME before the product: the crypto signal
as-of <= D-1 per symbol (momentum_features.attach_signals_asof), the macro signal
as-of <= D-1 broadcast across symbols (macro_features.attach_macro_asof). So the
product, and its binning, use only information known at submission -- backtest ==
live, same discipline as every other block. The raw temporaries are dropped; the
output carries the source features plus the binned interactions.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.binning import BIN_MASS, bin_cross_sectional, bin_mass
from src.data_mani.macro_features import (MACRO_SPECS, attach_macro_asof,
                                          compute_macro_signals)
from src.data_mani.momentum_features import (MOMENTUM_SPECS,
                                             attach_signals_asof,
                                             compute_momentum_signals)
from src.data_mani.utils import (download_if_missing, get_api, get_spec,
                                 local_path)

DEFAULT_SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")
DEFAULT_MACRO = os.path.join("data", "crypto", "macro", "macro_daily.parquet")

# The macro MOMENTUM features are the return-kind macro specs (gold/silver/S&P/
# DXY over their horizons); yield levels and slopes are not momentum.
MACRO_MOM_SPECS = {k: v for k, v in MACRO_SPECS.items() if v["kind"] == "ret"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Interactions of macro-momentum x crypto-momentum, binned "
                    "after the raw product.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0-momswap-macro",
                   help="dataset the binned interactions are added onto")
    p.add_argument("--out-suffix", default="-xinteractions")
    p.add_argument("--spot", default=DEFAULT_SPOT)
    p.add_argument("--macro", default=DEFAULT_MACRO)
    p.add_argument("--crypto-features", default=None,
                   help="comma list of crypto momentum specs; default all "
                        "(feature_mom_*)")
    p.add_argument("--macro-features", default=None,
                   help="comma list of macro momentum specs; default all "
                        "return-kind (gold/silver/snp/dxy)")
    p.add_argument("--submit-lag-days", type=int, default=1)
    p.add_argument("--crypto-staleness-days", type=int, default=4)
    p.add_argument("--macro-staleness-days", type=int, default=7)
    return p.parse_args()


def _subset(specs, names, what):
    if not names:
        return specs
    want = [n.strip() for n in names.split(",")]
    bad = [n for n in want if n not in specs]
    if bad:
        raise SystemExit("unknown %s features: %s\nknown: %s" % (what, bad, list(specs)))
    return {n: specs[n] for n in want}


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")
    crypto_specs = _subset(MOMENTUM_SPECS, args.crypto_features, "crypto")
    macro_specs = _subset(MACRO_MOM_SPECS, args.macro_features, "macro")
    crypto_names = list(crypto_specs)
    macro_names = list(macro_specs)

    # ---- source dataset --------------------------------------------------- #
    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    n_rows = len(base)
    src_feats = [c for c in base.columns if c.startswith("feature_")]
    print("source: %d rows, %d features (%s)" % (n_rows, len(src_feats), args.source_version))
    print("interacting %d crypto momentum x %d macro momentum = %d interactions"
          % (len(crypto_names), len(macro_names), len(crypto_names) * len(macro_names)))

    # ---- RAW crypto momentum, attached point-in-time (temp columns) -------- #
    spot = pd.read_parquet(args.spot, columns=["symbol", "date", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    print("computing raw crypto momentum ...")
    rawc = compute_momentum_signals(spot, crypto_specs).drop_duplicates(["symbol", "date"])
    rawc = rawc.rename(columns={c: "_rawc_" + c for c in crypto_names})
    base = attach_signals_asof(base, rawc, ["_rawc_" + c for c in crypto_names],
                               lag_days=args.submit_lag_days,
                               max_staleness_days=args.crypto_staleness_days,
                               period_col=spec["period_col"], symbol_col=spec["id_col"])

    # ---- RAW macro momentum (SIGNED return, un-normalised), broadcast PIT --- #
    macro_daily = pd.read_parquet(args.macro)
    print("computing raw macro momentum (signed returns) ...")
    rawm = compute_macro_signals(macro_daily, macro_specs, normalize=False)
    rawm = rawm.rename(columns={c: "_rawm_" + c for c in macro_names})
    base = attach_macro_asof(base, rawm, ["_rawm_" + c for c in macro_names],
                             lag_days=args.submit_lag_days,
                             max_staleness_days=args.macro_staleness_days,
                             period_col=spec["period_col"])

    # ---- the products, then bin AFTER --------------------------------------- #
    print("forming raw products ...")
    ix_cols = []
    for c in crypto_names:
        for m in macro_names:
            name = ("feature_ix_" + c.replace("feature_mom_", "")
                    + "_X_" + m.replace("feature_macro_", ""))
            base[name] = base["_rawc_" + c] * base["_rawm_" + m]
            ix_cols.append(name)

    print("rank-binning %d interactions to {0,.25,.5,.75,1} per date ..." % len(ix_cols))
    base = bin_cross_sectional(base, ix_cols, spec["period_col"])

    # mass sanity on a sample of the interactions
    for name in ix_cols[:3] + ix_cols[-3:]:
        drift = float(np.nanmax(np.abs(bin_mass(base[name]) - BIN_MASS)))
        print("   %-40s mass drift %.3f" % (name, drift))

    base = base.drop(columns=["_rawc_" + c for c in crypto_names]
                     + ["_rawm_" + m for m in macro_names])

    n_feat = sum(c.startswith("feature_") for c in base.columns)
    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    base.set_index("symbol").to_parquet(dest)
    print("features: %d total (%d source + %d interactions) -> %s (%.0f MB)"
          % (n_feat, len(src_feats), len(ix_cols), dest, os.path.getsize(dest) / 1e6))

    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % version)
    print("isolate the interactions with:  --feature-include '^feature_ix_'")


if __name__ == "__main__":
    main()
