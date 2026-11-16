"""
Engineered feature blocks on top of crypto/v2.0-momswap-macro (49f).

Replicates the allfeat_pp recipe (tenc / onehot / rollsel / spread / interact)
but retargeted to the momswap-macro feature set, which DROPPED the base momentum
proxies (feature_momentum_*, relative_strength_index_*, sharpe_*, bollinger_*) in
the swap and ADDED 17 literature momentum + 18 macro/cross-asset features. So the
blocks are adapted:

  tenc      feature_sym_tenc -- causal (lag-20) mean-target encoding of the symbol
            (build_target_encoding.causal_target_encoding), leakage-safe.
  onehot    one identity dummy per symbol with >= 130 dates (whole-panel vocab,
            the offline convention).
  rollsel   per-symbol rolling means {5,20,60} of the 17 momentum + 10 macro-return
            + 2 volatility features (RSI is ABSENT in momswap, so it is dropped
            from the user's list; volatility is kept). 29 x 3 = 87 columns.
  spread    term-structure differences WITHIN each momentum / macro family (the
            "different windows" spreads), mapped to [0,1] via build_derived.difference.
  interact  the accel + product blocks we built before, ADAPTED to the surviving
            features (feature_momentum is gone, so the canonical mid-horizon
            momentum feature_mom_ret_28d stands in; rsi products dropped), PLUS the
            new momentum x macro-momentum products -- crypto trend GATED by the
            macro regime, a per-symbol (not day-level) cross term. Curated, not
            all-pairs.

Scale: like the reference allfeat_pp, the derived blocks stay on CONTINUOUS [0,1]
(the difference/product maps and rolling means of already-[0,1] inputs), NOT
re-binned; onehot is 0/1; tenc is the [0,1]-ish encoding. NaN -> neutral 0.5.

Leakage: only tenc reads the target, causally (position <= p-20). Everything else
is a deterministic transform of contemporaneous features already in the panel;
the macro inputs were themselves attached point-in-time (D-1) upstream.

    python3 -m src.data_mani.build_momswap_eng   # -> data/crypto/v2.0-momswap-macro-eng/
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.build_derived import difference, product
from src.data_mani.build_target_encoding import causal_target_encoding
from src.data_mani.utils import download_if_missing, get_api, get_spec, local_path

SYMBOL, PERIOD, TARGET = "symbol", "date", "target_binned_return_20"
TENC_LAG, TENC_SMOOTHING = 20, 50.0
ONEHOT_MIN_DATES = 130
ROLL_WINDOWS = [5, 20, 60]

MOM = ["feature_mom_ret_7d", "feature_mom_ret_14d", "feature_mom_ret_21d",
       "feature_mom_ret_28d", "feature_mom_ret_56d", "feature_mom_ret_112d",
       "feature_mom_ret_12_1", "feature_mom_max1_28d", "feature_mom_max5_28d",
       "feature_mom_volscaled_28d", "feature_mom_trend_7d", "feature_mom_trend_14d",
       "feature_mom_trend_30d", "feature_mom_trend_60d", "feature_mom_trend_120d",
       "feature_mom_trend_240d", "feature_mom_trend_robust"]
MACRO_RET = ["feature_macro_gold_ret_20d", "feature_macro_gold_ret_60d",
             "feature_macro_silver_ret_20d", "feature_macro_silver_ret_60d",
             "feature_macro_snp_ret_5d", "feature_macro_snp_ret_20d",
             "feature_macro_snp_ret_60d", "feature_macro_dxy_ret_5d",
             "feature_macro_dxy_ret_20d", "feature_macro_dxy_ret_60d"]
VOL = ["feature_volatility_20d", "feature_volatility_60d"]
ROLLSEL_FEATURES = MOM + MACRO_RET + VOL                       # 29

# Term-structure spreads WITHIN each family: (short, long, name).
SPREADS = [
    ("feature_mom_ret_7d",  "feature_mom_ret_28d",  "mom_ret_7_28"),
    ("feature_mom_ret_7d",  "feature_mom_ret_112d", "mom_ret_7_112"),
    ("feature_mom_ret_28d", "feature_mom_ret_112d", "mom_ret_28_112"),
    ("feature_mom_trend_7d",  "feature_mom_trend_60d",  "mom_trend_7_60"),
    ("feature_mom_trend_7d",  "feature_mom_trend_240d", "mom_trend_7_240"),
    ("feature_mom_trend_60d", "feature_mom_trend_240d", "mom_trend_60_240"),
    ("feature_macro_gold_ret_20d",   "feature_macro_gold_ret_60d",   "macro_gold_20_60"),
    ("feature_macro_silver_ret_20d", "feature_macro_silver_ret_60d", "macro_silver_20_60"),
    ("feature_macro_snp_ret_5d",  "feature_macro_snp_ret_20d", "macro_snp_5_20"),
    ("feature_macro_snp_ret_20d", "feature_macro_snp_ret_60d", "macro_snp_20_60"),
    ("feature_macro_dxy_ret_5d",  "feature_macro_dxy_ret_20d", "macro_dxy_5_20"),
    ("feature_macro_dxy_ret_20d", "feature_macro_dxy_ret_60d", "macro_dxy_20_60"),
]

# accel = ewa - avg (present base features survive the swap).
ACCEL = [("close_ewa", "close_avg"), ("volume_ewa", "volume_avg"),
         ("market_cap_ewa", "market_cap_avg")]
HORIZONS = ("20d", "60d")

# "before" products, adapted: feature_momentum is gone, so mid-horizon
# feature_mom_ret_28d stands in; rsi products dropped (rsi absent). (left, right, name).
PRODUCTS_BEFORE = [
    ("feature_volume_avg_%s", "feature_volatility_%s", "volume_x_volatility_%s"),
    ("feature_mom_ret_28d",   "feature_volatility_%s", "momret28_x_volatility_%s"),
    ("feature_market_cap_avg_%s", "feature_mom_ret_28d", "marketcap_x_momret28_%s"),
]

# NEW: momentum x macro-momentum -- crypto momentum GATED by the macro regime.
# Curated crypto reps x macro reps (not all 17x10).
MOM_REPS = ["feature_mom_ret_28d", "feature_mom_trend_60d", "feature_mom_ret_112d"]
MACRO_REPS = ["feature_macro_dxy_ret_20d", "feature_macro_gold_ret_20d",
              "feature_macro_snp_ret_20d", "feature_macro_silver_ret_20d"]


def build_tenc(df):
    enc, _ = causal_target_encoding(df, SYMBOL, PERIOD, TARGET,
                                    lag=TENC_LAG, smoothing=TENC_SMOOTHING, window=0)
    return {"feature_sym_tenc": enc.astype("float32")}


def build_onehot(df):
    counts = df.groupby(SYMBOL)[PERIOD].nunique()
    vocab = sorted(counts[counts >= ONEHOT_MIN_DATES].index)
    codes = df[SYMBOL].where(df[SYMBOL].isin(vocab))
    dummies = pd.get_dummies(codes, prefix="feature_sym", dtype="uint8")
    cols = ["feature_sym_%s" % s for s in vocab]
    dummies = dummies.reindex(columns=cols, fill_value=0)
    dummies.index = df.index
    return dummies, cols, len(vocab)


def build_rollsel(df):
    present = [c for c in ROLLSEL_FEATURES if c in df.columns]
    missing = [c for c in ROLLSEL_FEATURES if c not in df.columns]
    if missing:
        print("  rollsel: SKIPPING %d absent inputs: %s" % (len(missing), missing))
    g = df.sort_values([SYMBOL, PERIOD]).groupby(SYMBOL, sort=False)[present]
    out = {}
    for w in ROLL_WINDOWS:
        rolled = g.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
        for c in present:
            out["%s_rm%d" % (c, w)] = rolled[c].astype("float32")
    return out


def build_spread(df):
    out = {}
    for short, long, name in SPREADS:
        if short in df.columns and long in df.columns:
            out["feature_spread_%s" % name] = difference(df[short], df[long])
    return out


def build_interact(df):
    out = {}
    for h in HORIZONS:
        for fast, slow in ACCEL:
            f, s = "feature_%s_%s" % (fast, h), "feature_%s_%s" % (slow, h)
            if f in df.columns and s in df.columns:
                base = fast.replace("_ewa", "")
                out["feature_%s_accel_%s" % (base, h)] = difference(df[f], df[s])
        for left, right, name in PRODUCTS_BEFORE:
            l = left % h if "%s" in left else left
            r = right % h if "%s" in right else right
            nm = name % h if "%s" in name else name
            if l in df.columns and r in df.columns:
                out["feature_%s" % nm] = product(df[l], df[r])
    # momentum x macro-momentum (curated)
    for cm in MOM_REPS:
        for mm in MACRO_REPS:
            if cm in df.columns and mm in df.columns:
                cmn = cm.replace("feature_mom_", "mom")
                mmn = mm.replace("feature_macro_", "").replace("_ret", "")
                out["feature_%s_x_%s" % (cmn, mmn)] = product(df[cm], df[mm])
    return out


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")
    version = "crypto/v2.0" + args.out_suffix

    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    df = pd.read_parquet(src).reset_index()
    df[PERIOD] = pd.to_datetime(df[PERIOD])
    base_feats = [c for c in df.columns if c.startswith("feature_")]
    print("base: %d rows, %d symbols, %d features (%s)"
          % (len(df), df[SYMBOL].nunique(), len(base_feats), args.source_version))

    want = set(b.strip() for b in args.blocks.split(",") if b.strip())
    print("  blocks: %s" % sorted(want))
    blocks = {}
    oh_cols, dummies = [], None
    if "tenc" in want:
        blocks.update(build_tenc(df));            print("  tenc:     1")
    if "onehot" in want:
        dummies, oh_cols, vocab = build_onehot(df); print("  onehot:   %d (vocab>=%d dates)" % (len(oh_cols), ONEHOT_MIN_DATES))
    if "rollsel" in want:
        roll = build_rollsel(df);                 print("  rollsel:  %d" % len(roll)); blocks.update(roll)
    if "spread" in want:
        sprd = build_spread(df);                  print("  spread:   %d" % len(sprd)); blocks.update(sprd)
    if "interact" in want:
        inter = build_interact(df);               print("  interact: %d" % len(inter)); blocks.update(inter)

    derived = pd.DataFrame(blocks, index=df.index)
    n_missing = int(derived.isna().to_numpy().sum())
    derived = derived.fillna(0.5).astype("float32")
    parts = [df, derived] + ([dummies] if dummies is not None else [])
    out = pd.concat(parts, axis=1)

    total = len([c for c in out.columns if c.startswith("feature_")])
    print("added %d derived + %d onehot = %d new; %d NaN filled to 0.5; total features %d"
          % (len(derived.columns), len(oh_cols), len(derived.columns) + len(oh_cols),
             n_missing, total))

    if args.drop_originals:
        keep = [c for c in out.columns if not c.startswith("feature_")
                or c in derived.columns or c in oh_cols]
        out = out[keep]

    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.set_index(SYMBOL).to_parquet(dest)
    print("wrote %s (%d rows, %d features)" % (dest, len(out),
          len([c for c in out.columns if c.startswith("feature_")])))

    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if os.path.abspath(source) != os.path.abspath(target):
            shutil.copy(source, target)
            print("copied %s" % filename)


def parse_args():
    p = argparse.ArgumentParser(description="Engineered blocks on v2.0-momswap-macro.")
    p.add_argument("--source-version", default="crypto/v2.0-momswap-macro")
    p.add_argument("--out-suffix", default="-momswap-macro-eng")
    p.add_argument("--blocks", default="tenc,onehot,rollsel,spread,interact",
                   help="comma-separated subset of blocks to build")
    p.add_argument("--drop-originals", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
