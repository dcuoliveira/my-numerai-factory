"""
Build a Crypto dataset that carries literature crypto-momentum features, on the
SAME rank-bin scale as the shipped features, merged onto the tournament panel.

    python3 -m src.data_mani.build_momentum                  # -> data/crypto/v2.0-litmom/ (adds 17 to the 22)
    python3 -m src.data_mani.build_momentum --replace-proxies # -> data/crypto/v2.0-momswap/ (14 base + 17)
    python3 -m src.data_mani.build_momentum --drop-originals  # momentum only, for isolation

Run it with:  --data-version crypto/v2.0-litmom   (or crypto/v2.0-momswap for the swap)

--replace-proxies is the SWAP: it drops the base's own 8 momentum proxies
(feature_{momentum,relative_strength_index,sharpe_ratio,bollinger}_{20,60}d) and
keeps the 14 non-momentum base features, so the literature block REPLACES the
base's crude momentum rather than adding to it (22 -> 14 + 17 = 31 features).
This is E66's swap as a first-class dataset instead of a run-time --feature-exclude.

------------------------------------------------------------------------------
The chain, and why it is three modules
------------------------------------------------------------------------------

    spot OHLCV ──▶ momentum_features.compute_momentum_signals ──▶ raw signals
       │                                                             │
    (data/crypto/spot/ohlcv.parquet)                     merge onto the panel
                                                                     │
    base panel ────────────────────────────────────────────────────┤
    (crypto/v2.0/train.parquet)                                      ▼
                                          binning.bin_cross_sectional (per date)
                                                                     │
                                                                     ▼
                                     data/crypto/v2.0-litmom/train.parquet  (drop-in)

* ``momentum_features`` is pure price math -- no file layout, no Numerai scale.
* ``binning`` is the ONE place Numerai's ``{0,.25,.5,.75,1}`` quantization lives.
* this file is the offline wiring: where the inputs are, where the output goes.

The split is deliberate so the LIVE path (see PRODUCTION below) can reuse the
first two modules unchanged.

------------------------------------------------------------------------------
The join is POINT-IN-TIME, lagged by the submission constraint
------------------------------------------------------------------------------

The momentum on round date D must use only prices known when we submit round D.
Our spot feed is an external daily source whose bar for day D does not close
until the end of day D, but the round is submitted DURING day D -- so the newest
usable close is the last completed bar, D-1. The join is therefore an as-of
backward merge with a one-day lag (``attach_signals_asof``): row D receives the
FRESHEST spot signal dated <= D-1, capped at ``--max-staleness-days`` so a
delisted token cannot carry a stale value forward. An exact same-date join would
attach close(D) to row D and quietly leak a price that is not final at
submission; reproducing the live cutoff offline is what keeps the backtest from
being optimistic about information it would not have had.

------------------------------------------------------------------------------
Binning happens AFTER the join, on the panel's cross-section
------------------------------------------------------------------------------

The rank-bin of a symbol at date t must be taken over the SAME universe the model
scores on -- the tournament panel's symbols on that date -- not over whatever set
the spot feed happened to cover. So we compute raw signals per symbol, attach
them as-of, and only then bin per date. A symbol with no spot price, or too
little history for a window, is NaN after the join and stays NaN through the bin;
``fill_missing_features`` puts the neutral 0.5 there at load time. Base rows and
base feature columns are never touched -- the panel is a superset.

------------------------------------------------------------------------------
PRODUCTION (not wired here -- wire it once the features validate)
------------------------------------------------------------------------------

This builds the BACKTEST dataset the prediction team validates against. To put a
validated momentum feature live, ``src/live/features.py`` would, per round D:

    1. read the symbol close history from an external spot store (the live twin
       of data/crypto/spot) covering dates up to D-1 -- the last completed bar,
    2. call momentum_features.compute_momentum_signals on that (symbol,date,close)
       panel -- the SAME function used here,
    3. call momentum_features.attach_signals_asof with the SAME submit lag, so
       round D receives the freshest signal dated <= D-1 -- the exact cutoff this
       builder reproduces offline,
    4. call binning.bin_cross_sectional with df restricted to round D's
       cross-section -- the SAME quantization used here,
    5. append the resulting feature_mom_* columns to the live feature frame.

Two preconditions: (a) history depth -- the live store must reach the longest
window back from D-1 (feature_mom_ret_12_1 needs 360 dates); (b) a spot feed that
delivers D-1's close before the round's deadline. Nothing else changes, because
the offline and live paths share compute + attach + bin by construction -- which
is the whole point of keeping those three as importable functions.
"""

import argparse
import os
import re
import shutil

import numpy as np
import pandas as pd

from src.data_mani.binning import BIN_MASS, bin_cross_sectional, bin_mass
from src.data_mani.momentum_features import (MOMENTUM_SPECS,
                                             attach_signals_asof,
                                             compute_momentum_signals)
from src.data_mani.utils import (download_if_missing, get_api, get_spec,
                                 local_path)

DEFAULT_SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")

# The base v2.0 features that are themselves momentum proxies -- the ones the
# literature block REPLACES in --replace-proxies mode. Same alternation E66's
# swap regex uses, so a built dataset equals exactly what E66 tested. Drops 8:
# feature_{momentum,relative_strength_index,sharpe_ratio,bollinger}_{20,60}d.
# The 14 non-momentum base features (close/market_cap/volatility/volume) stay.
OLD_MOMENTUM_PROXIES_RX = \
    r"^feature_(momentum|relative_strength_index|sharpe_ratio|bollinger)_[0-9]+d$"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add rank-binned crypto momentum features from spot prices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-version", default="crypto/v2.0")
    parser.add_argument("--out-suffix", default=None,
                        help="output dir suffix; defaults to -litmom, or "
                             "-momswap when --replace-proxies is set")
    parser.add_argument("--spot", default=DEFAULT_SPOT,
                        help="spot OHLCV parquet with symbol,date,close")
    parser.add_argument("--features", default=None,
                        help="comma-separated momentum feature names to build; "
                             "default is all of MOMENTUM_SPECS")
    parser.add_argument("--drop-originals", action="store_true",
                        help="keep only the momentum features, so a comparison "
                             "isolates them rather than adding to the 22")
    parser.add_argument("--replace-proxies", action="store_true",
                        help="SWAP mode: drop the base momentum proxies "
                             "(feature_{momentum,relative_strength_index,"
                             "sharpe_ratio,bollinger}_*) and keep the rest, so the "
                             "literature block replaces the base's crude momentum "
                             "rather than adding to it (22 -> 14 base + momentum)")
    parser.add_argument("--proxies-regex", default=OLD_MOMENTUM_PROXIES_RX,
                        help="which base features --replace-proxies drops")
    parser.add_argument("--submit-lag-days", type=int, default=1,
                        help="row date D may only use spot closes dated <= D - "
                             "this. 1 = the last completed daily bar, the newest "
                             "price guaranteed known when round D is submitted")
    parser.add_argument("--max-staleness-days", type=int, default=4,
                        help="the attached spot signal must be within this many "
                             "days of D, so a delisted token cannot carry a "
                             "months-stale value forward; else the row is neutral")
    return parser.parse_args()


def _select_specs(names):
    if not names:
        return MOMENTUM_SPECS
    wanted = [n.strip() for n in names.split(",")]
    unknown = [n for n in wanted if n not in MOMENTUM_SPECS]
    if unknown:
        raise SystemExit("unknown momentum features: %s\nknown: %s"
                         % (unknown, list(MOMENTUM_SPECS)))
    return {n: MOMENTUM_SPECS[n] for n in wanted}


def main():
    args = parse_args()
    if args.replace_proxies and args.drop_originals:
        raise SystemExit("--replace-proxies and --drop-originals are mutually "
                         "exclusive: the first keeps the 14 non-momentum base "
                         "features, the second drops all base features")
    out_suffix = args.out_suffix or (
        "-momswap" if args.replace_proxies else "-litmom")
    spec = get_spec("crypto")
    api = get_api("crypto")
    specs = _select_specs(args.features)
    mom_cols = list(specs.keys())

    # ---- base tournament panel -------------------------------------------- #
    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    n_rows = len(base)
    base_feats = [c for c in base.columns if c.startswith("feature_")]
    print("base: %d rows, %d symbols, %d features (%s)"
          % (n_rows, base["symbol"].nunique(), len(base_feats), args.source_version))

    # ---- spot close series ------------------------------------------------- #
    if not os.path.exists(args.spot):
        raise SystemExit("spot file not found: %s\nbuild it first (fetch spot "
                         "OHLCV into data/crypto/spot/ohlcv.parquet)" % args.spot)
    spot = pd.read_parquet(args.spot, columns=["symbol", "date", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    print("spot: %d rows, %d symbols (%s)"
          % (len(spot), spot["symbol"].nunique(), args.spot))

    # ---- raw momentum, then POINT-IN-TIME join onto the panel ------------- #
    print("computing %d momentum signals ..." % len(mom_cols))
    raw = compute_momentum_signals(spot, specs)
    # collapse to one row per (symbol,date) in case the spot feed had dups
    raw = raw.drop_duplicates(["symbol", "date"])
    # Attach as-of, lagged by the submission constraint: row D uses the freshest
    # spot close dated <= D - submit_lag_days. This makes the offline dataset
    # carry exactly the information the live path has at submission, no more.
    print("attaching as-of (submit lag %d day, max staleness %d days) ..."
          % (args.submit_lag_days, args.max_staleness_days))
    merged = attach_signals_asof(base, raw, mom_cols,
                                 lag_days=args.submit_lag_days,
                                 max_staleness_days=args.max_staleness_days,
                                 period_col=spec["period_col"],
                                 symbol_col=spec["id_col"])
    if len(merged) != n_rows:
        raise SystemExit("as-of merge changed the row count %d -> %d"
                         % (n_rows, len(merged)))

    # per-feature coverage BEFORE binning (how many panel rows got a real signal)
    for c in mom_cols:
        cov = float(merged[c].notna().mean())
        print("   %-26s coverage %5.1f%% of panel rows" % (c, 100 * cov))

    # ---- bin onto Numerai's scale, over the panel's per-date cross-section - #
    print("rank-binning to {0,.25,.5,.75,1} per date ...")
    merged = bin_cross_sectional(merged, mom_cols, spec["period_col"])

    # sanity: every binned column must reproduce the 5/20/50/20/5 mass split
    for c in mom_cols:
        mass = bin_mass(merged[c])
        drift = float(np.nanmax(np.abs(mass - BIN_MASS)))
        flag = "  <-- OFF" if drift > 0.02 else ""
        print("   %-26s mass %s (max drift %.3f)%s"
              % (c, np.array2string(mass, precision=3, floatmode="fixed"),
                 drift, flag))

    if args.drop_originals:
        merged = merged.drop(columns=base_feats)
        print("dropped the %d original features" % len(base_feats))

    if args.replace_proxies:
        # SWAP: drop the base momentum proxies, keep the other base features and
        # the literature momentum block. Assert the count so a regex drift can
        # never silently score the wrong feature set (mirrors E66's preflight).
        proxies = [c for c in base_feats if re.match(args.proxies_regex, c)]
        if len(proxies) != 8:
            raise SystemExit("proxies regex %r matched %d base features, expected "
                             "exactly the 8 momentum proxies: %s"
                             % (args.proxies_regex, len(proxies), proxies))
        merged = merged.drop(columns=proxies)
        kept_base = [c for c in base_feats if c not in proxies]
        print("SWAP: dropped %d base momentum proxies %s"
              % (len(proxies), proxies))
        print("      kept %d non-momentum base + %d literature momentum = %d"
              % (len(kept_base), len(mom_cols), len(kept_base) + len(mom_cols)))

    n_feat = sum(c.startswith("feature_") for c in merged.columns)
    version = args.source_version + out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    merged.set_index("symbol").to_parquet(dest)
    print("features: %d total (%d momentum) -> %s (%.0f MB)"
          % (n_feat, len(mom_cols), dest, os.path.getsize(dest) / 1e6))

    # the reference / meta-model file must travel with the variant, same as
    # every other builder, or load_reference/attach_meta_model can't find it.
    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)

    print("\nrun it with:  --data-version %s" % version)
    print("isolate the momentum features with:  "
          "--feature-include '^feature_mom_'")


if __name__ == "__main__":
    main()
