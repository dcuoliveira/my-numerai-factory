"""
A/B test: cross-block INTERACTIONS (products) computed on the RAW continuous signals
vs on the already-BINNED columns. Builds two datasets, identical except for that one
choice, so a paired comparison isolates it:

    crypto/v2.0-crbmc-xraw-fillmask  -- product of RAW signals, THEN rank-binned
    crypto/v2.0-crbmc-xbin-fillmask  -- product of BINNED columns, THEN re-binned

Both = the 82f crbmc base + the SAME 1387 cross-block product interactions (coint,
rev, basis/funding, mom, tf; cross-block pairs only), rank-binned per date and
fillmasked. SPREADS are excluded: a raw difference across incommensurate scales is
degenerate, so there is no meaningful "raw spread" to compare against the binned one.

STATIONARISATION OF THE RAW LEGS. A raw product would otherwise be dominated by the
heaviest-scale / heaviest-tailed leg (amihud, vols, cointegration half-life), and its
cross-sectional distribution would drift over time -- a non-stationary artifact, not
an interaction. So before multiplying, each raw signal is standardised CAUSALLY: a
pooled expanding z-score, z_t = (x - mean_{<=t}) / std_{<=t} over all observations
dated <= t (no lookahead), then winsorised to +/-5. This removes level/scale drift
and tames tails while preserving magnitude ordering (unlike a rank). The BINNED legs
need none of this -- a rank is already stationary and scale-free.

Resumable: the recomputed raw signal frame and the per-dataset column chunks are
cached on disk and skipped on a rerun.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.momentum_features import (MOMENTUM_SPECS, attach_signals_asof,
                                             compute_momentum_signals)
from src.data_mani.trend_signals import TREND_SPECS, compute_trend_signals
from src.data_mani.reversion_features import (REVERSION_SPECS,
                                              compute_reversion_signals)
from src.data_mani.cointegration_features import (COINT_SPECS,
                                                  compute_cointegration_signals)
from src.data_mani.basis_features import BASIS_FEATURES, compute_basis_signals
from src.data_mani.utils import (download_if_missing, get_api, get_spec,
                                 local_path)

SPINE = "crypto/v2.0-coint-rev-basis-mom-carver-fillmask"   # 82f base+blocks
SPOT = "data/crypto/spot/ohlcv.parquet"
MACRO = "data/crypto/macro/macro_daily.parquet"
PERP = "data/crypto/perp/perp_daily.parquet"
FUND = "data/crypto/perp/funding_daily.parquet"
# pre-fill BINNED source datasets (one per block), same prefixes as the blocks
BIN_SRC = [("crypto/v2.0-coint", ("feature_coint_",)),
           ("crypto/v2.0-rev", ("feature_rev_",)),
           ("crypto/v2.0-basis", ("feature_basis_", "feature_funding_")),
           ("crypto/v2.0-litmom", ("feature_mom_",)),
           ("crypto/v2.0-momswap-macro-carver", ("feature_tf_",))]
LAG, STALE = 1, 4


def _block(c):
    for p, b in [("feature_coint_", "coint"), ("feature_rev_", "rev"),
                 ("feature_basis_", "basis"), ("feature_funding_", "basis"),
                 ("feature_mom_", "mom"), ("feature_tf_", "tf")]:
        if c.startswith(p):
            return b
    return "?"


def causal_standardize(R, cols, period, clip=5.0):
    """Pooled expanding z-score per feature over dates <= t (causal), winsorised to
    +/-clip. Stationarises the raw legs so a product is not scale/outlier-dominated.
    NaN is preserved (a missing leg stays missing -> fillmask downstream)."""
    by = R[period]                                   # original row order preserved
    s1 = R.groupby(by)[cols].sum(min_count=1)        # per-date, sorted-date index
    cc = R.groupby(by)[cols].count()
    s2 = (R[cols] ** 2).groupby(by).sum(min_count=1)
    cumc = cc.cumsum().replace(0, np.nan)            # cumulative over dates <= t
    mean = s1.cumsum() / cumc
    std = np.sqrt((s2.cumsum() / cumc - mean ** 2).clip(lower=1e-12))
    meanr = mean.reindex(by).to_numpy()              # map each row's date -> its <=t stat
    stdr = std.reindex(by).to_numpy()
    z = np.clip((R[cols].to_numpy() - meanr) / stdr, -clip, clip)
    out = R.copy()
    out[cols] = z.astype("float32")
    return out


def fast_bin(frame, cols, period):
    pct = frame.groupby(period)[cols].rank(method="first", pct=True).to_numpy()
    b = np.select([pct <= 0.05, pct <= 0.25, pct <= 0.75, pct <= 0.95],
                  [0.0, 0.25, 0.5, 0.75], default=1.0).astype("float32")
    b[np.isnan(pct)] = np.nan
    return pd.DataFrame(b, index=frame.index, columns=cols)


def build_raw_frame(keys, spec):
    """Recompute all 60 raw signals and PIT-attach them onto the panel keys."""
    per, idc = spec["period_col"], spec["id_col"]
    spot = pd.read_parquet(SPOT, columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    spot["date"] = pd.to_datetime(spot["date"])
    sc = spot[["symbol", "date", "close"]]
    R = keys.copy()

    def att(raw, cols):
        return attach_signals_asof(R, raw.drop_duplicates(["symbol", "date"]), cols,
                                   lag_days=LAG, max_staleness_days=STALE,
                                   period_col=per, symbol_col=idc)
    print("  raw mom ..."); R = att(compute_momentum_signals(sc, MOMENTUM_SPECS), list(MOMENTUM_SPECS))
    print("  raw tf ...");  R = att(compute_trend_signals(sc, TREND_SPECS), list(TREND_SPECS))
    print("  raw rev ..."); R = att(compute_reversion_signals(spot, REVERSION_SPECS), list(REVERSION_SPECS))
    print("  raw coint (rolling OLS, slow) ...")
    R = att(compute_cointegration_signals(sc, pd.read_parquet(MACRO)), list(COINT_SPECS))
    print("  raw basis ...")
    perp = pd.read_parquet(PERP)[["symbol", "date", "perp_close"]]
    fund = pd.read_parquet(FUND)[["symbol", "date", "funding_sum"]].rename(columns={"funding_sum": "funding"})
    sp = spot[["symbol", "date", "close"]].rename(columns={"close": "spot_close"})
    for d in (perp, fund, sp):
        d["date"] = pd.to_datetime(d["date"])
    panel = perp.merge(sp, on=["symbol", "date"], how="inner").merge(fund, on=["symbol", "date"], how="left")
    R = att(compute_basis_signals(panel), BASIS_FEATURES)
    return R


def build_bin_frame(keys, spec):
    """Assemble the pre-fill BINNED block columns onto the panel keys."""
    per, idc = spec["period_col"], spec["id_col"]
    api = get_api("crypto")
    B = keys.copy()
    for ver, prefixes in BIN_SRC:
        d = pd.read_parquet(download_if_missing(api, "crypto", ver, "train.parquet")).reset_index()
        d[per] = pd.to_datetime(d[per])
        cols = [c for c in d.columns if c.startswith(prefixes)]
        B = B.merge(d[[idc, per] + cols].drop_duplicates([idc, per]), on=[idc, per], how="left")
    return B


def materialise(tag, frame, pairs, base, keys, per, idc, api, chunk, fill):
    out_version = "crypto/v2.0-crbmc-x%s-fillmask" % tag
    ck = local_path("crypto", out_version, "train.parquet").rsplit("/", 1)[0] + "/_chunks"
    os.makedirs(ck, exist_ok=True)
    defs = [("feature_x2_mul_%s__%s" % (a[len("feature_"):], b[len("feature_"):]), a, b) for a, b in pairs]
    n_chunks = (len(defs) + chunk - 1) // chunk
    for ci in range(n_chunks):
        cp = os.path.join(ck, "chunk_%03d.parquet" % ci)
        if os.path.exists(cp):
            print("    [%s] chunk %d/%d cached" % (tag, ci + 1, n_chunks)); continue
        sub = defs[ci * chunk:(ci + 1) * chunk]
        raw = {nm: frame[a] * frame[b] for nm, a, b in sub}
        names = [d[0] for d in sub]
        fr = pd.concat([keys, pd.DataFrame(raw, index=frame.index)], axis=1)
        pd.concat([keys, fast_bin(fr, names, per).fillna(fill)], axis=1).to_parquet(cp, index=False)
        print("    [%s] chunk %d/%d: %d cols" % (tag, ci + 1, n_chunks, len(names)))
    out = base.copy()
    for ci in range(n_chunks):
        out = out.merge(pd.read_parquet(os.path.join(ck, "chunk_%03d.parquet" % ci)), on=[idc, per], how="left")
    dest = local_path("crypto", out_version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.set_index(idc).to_parquet(dest)
    nf = sum(c.startswith("feature_") for c in out.columns)
    print("  [%s] features: %d (%d cross) -> %s (%.0f MB)" % (tag, nf, len(defs), dest, os.path.getsize(dest) / 1e6))
    tgt = local_path("crypto", out_version, "historical_meta_models.parquet")
    if not os.path.exists(tgt):
        shutil.copy(download_if_missing(api, "crypto", SPINE, "historical_meta_models.parquet"), tgt)
    shutil.rmtree(ck, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk", type=int, default=500)
    p.add_argument("--fill-value", type=float, default=0.5)
    args = p.parse_args()
    spec = get_spec("crypto"); api = get_api("crypto")
    per, idc = spec["period_col"], spec["id_col"]
    base = pd.read_parquet(download_if_missing(api, "crypto", SPINE, "train.parquet")).reset_index()
    base[per] = pd.to_datetime(base[per])
    keys = base[[idc, per]].copy()

    rawcache = local_path("crypto", "crypto/v2.0-crbmc-xraw-fillmask", "train.parquet").rsplit("/", 1)[0] + "/_rawframe.parquet"
    os.makedirs(os.path.dirname(rawcache), exist_ok=True)
    if os.path.exists(rawcache):
        print("raw signal frame cached"); Rraw = pd.read_parquet(rawcache)
    else:
        print("recomputing raw signals ..."); Rraw = build_raw_frame(keys, spec)
        Rraw.to_parquet(rawcache, index=False)
    Rbin = build_bin_frame(keys, spec)
    feats = [c for c in Rraw.columns if c.startswith("feature_")]
    pairs = [(feats[i], feats[j]) for i in range(len(feats)) for j in range(i + 1, len(feats))
             if _block(feats[i]) != _block(feats[j])]
    print("60 signals -> %d cross-block product pairs" % len(pairs))

    print("stationarising raw legs (causal pooled z-score + winsorise) ...")
    Rraw = causal_standardize(Rraw, feats, per)
    materialise("raw", Rraw, pairs, base, keys, per, idc, api, args.chunk, args.fill_value)
    materialise("bin", Rbin, pairs, base, keys, per, idc, api, args.chunk, args.fill_value)
    if os.path.exists(rawcache):
        os.remove(rawcache)
    print("\nrun:  --data-version crypto/v2.0-crbmc-xraw-fillmask   (raw-product interactions)")
    print("run:  --data-version crypto/v2.0-crbmc-xbin-fillmask   (binned-product interactions)")


if __name__ == "__main__":
    main()
