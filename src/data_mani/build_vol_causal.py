"""
Build crypto/v2.0-vol-causal: the 22 realised-volatility features of crypto/v2.0-vol plus their 132
own-history ranks, rebuilt for the
covered-subset evaluation (matched B_covered / V_covered models, slot-preserving routing).

    python3 -m src.data_mani.build_vol_causal

What changes against crypto/v2.0-vol
------------------------------------
1. Every v2.0 row is kept and the 22 shipped features are copied unchanged. A cell the block cannot
   fill honestly stays NaN. No indicator column is added, and nothing is forward- or back-filled.
2. Prices are assigned per COIN, not per ticker. Six coins whose Yahoo ticker carries another coin's
   prices take their own Binance spot series from the cached public archive (ARCHIVE below). Binance
   daily klines open at 00:00 UTC and close at 23:59:59.999 UTC, the same day boundary as Yahoo's crypto
   bars; USDT quotes are used as USD, which no estimator here can see (every one is scale-free).
3. A series must pass a CAUSAL identity gate to count. On each panel date t the statistic is the
   mean, over the series' panel rows up to and including t (at least GATE_MIN of them), of the
   absolute gap between the per-date percentile of our rebuilt 20- and 60-day close-to-close
   volatility and the shipped feature_volatility_20d / _60d. Both sides are known at submission on t.
   A row passes when the statistic is at or below GATE_THRESHOLD, a frozen constant. It was set once,
   on the 2026-11-12 build, to the 5% quantile of the same statistic under a wrong-coin null (each
   coin's shipped features against another coin's rebuilt series, a fixed derangement), so about 5% of
   wrong-coin rows get through. Freezing it keeps every row a function of data up to its own date; a
   threshold re-estimated on each build would be a whole-sample statistic. The manifest records the
   quantile on this build's full file and on dates up to CALIB_END, to show how far it moves.
   Volatility is used, not momentum, because Numerai's momentum_20d does not track the rebuilt return
   for sub-cent coins. An expanding mean was chosen over trailing windows (20 to 250 rows) because it
   separated right from wrong coins best; its cost is that a ticker that switches assets mid-history
   is caught only once the new asset's rows outweigh the old ones.
4. Estimates are computed per series, attached as-of D-1 (max 4 days stale), and rank-binned per date
   with Numerai's exact rule (history_features.bin_ties_preserved) over the coins that passed.
5. Own-history ranks: each of the 22 RAW estimates is also ranked against the same series' own daily
   history over TSR_WINDOWS calendar days, (d - w days, d], before the as-of join:
   rank = (values below + (values equal + 1) / 2) / values in the window, the convention of
   history_features. A window counts only with raw values on at least TSR_COVERAGE of its days, so a
   365-day rank needs about 219 days of estimates behind it. The rank stays continuous until the
   per-date binning (binning it first would only add ties). Columns feature_vol_<est>_tsr<w>; they
   go through the same D-1 join, identity gate and per-date binning as the 22.
6. Coverage masks are written as NON-feature columns, one per arm: cov_<family>_<short|long> is 1
   when every column of that arm is observed on that row, cov_all when all 22 are; likewise
   cov_tsr_<family>_<short|long> for the own-history ranks (short 7/14/30, long 90/182/365) and
   cov_tsr_all. The covered models
   train on exactly the rows of their arm's mask; routing reorders exactly those coins.

What this is NOT: point-in-time. The spot store was fetched in 2026, so which coins have prices at all,
and those prices, are a retrospective reconstruction (vendors backfill, drop dead coins, remap
tickers). The block is timestamp-causal on a retrospectively reconstructed universe. The manifest
records the hashes of every input so a later, dated store can be compared against it.
"""

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data_mani import fetch_binance_archive as fb
from src.data_mani.history_features import _window_stats, bin_ties_preserved
from src.data_mani.momentum_features import attach_signals_asof
from src.data_mani.utils import download_if_missing, get_api, local_path
from src.data_mani.volatility_features import VOL_SPECS, compute_volatility_signals

SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")
LAG, STALE = 1, 4
GATE_MIN, GATE_Q, GATE_THRESHOLD = 40, 0.05, 0.18
CALIB_END = pd.Timestamp("2022-12-31")
GATE = {"feature_vol_sqret_20d": "feature_volatility_20d",     # rebuilt -> shipped
        "feature_vol_sqret_60d": "feature_volatility_60d"}
# (symbol, ucid) -> Binance archive pair. Each passed the archive identity check
# (binance_archive_report.csv) while its Yahoo ticker carries another coin.
ARCHIVE = {("GAL", "11877"): "GALUSDT", ("ID", "21846"): "IDUSDT", ("STRK", "22691"): "STRKUSDT",
           ("SKY", "33038"): "SKYUSDT", ("MET", "38353"): "METUSDT", ("ONE", "3945"): "ONEUSDT"}
JUMP = np.log(5.0)                          # a one-day move this large is reported as a possible migration
ARMS = {"sqret_short": ["feature_vol_sqret_%dd" % w for w in (5, 10, 20)],
        "sqret_long": ["feature_vol_sqret_%dd" % w for w in (40, 60, 120)],
        "ewma_short": ["feature_vol_ewma_lam90", "feature_vol_ewma_lam94"],
        "ewma_long": ["feature_vol_ewma_lam97", "feature_vol_ewma_lam99"],
        "park_short": ["feature_vol_park_%dd" % w for w in (5, 10, 20)],
        "park_long": ["feature_vol_park_%dd" % w for w in (40, 60, 120)],
        "yz_short": ["feature_vol_yz_%dd" % w for w in (5, 10, 20)],
        "yz_long": ["feature_vol_yz_%dd" % w for w in (40, 60, 120)]}
COLS = list(VOL_SPECS)
assert sorted(c for a in ARMS.values() for c in a) == sorted(COLS)
TSR_WINDOWS, TSR_SHORT, TSR_COVERAGE = (7, 14, 30, 90, 182, 365), (7, 14, 30), 0.6
TSR = {c: ["%s_tsr%d" % (c, w) for w in TSR_WINDOWS] for c in COLS}
TSR_COLS = [t for c in COLS for t in TSR[c]]
TSR_ARMS = {"tsr_%s_%s" % (fam, h): [t for c in COLS if c.startswith("feature_vol_%s_" % fam) for w, t in zip(TSR_WINDOWS, TSR[c])
                                    if (w in TSR_SHORT) == (h == "short")]
            for fam in ("sqret", "ewma", "park", "yz") for h in ("short", "long")}
assert sorted(c for a in TSR_ARMS.values() for c in a) == sorted(TSR_COLS)


def own_history_ranks(raw):
    """Own-history rank of each raw estimate over TSR_WINDOWS calendar days, per series.

    raw: frame keyed symbol (the series key), date (daily), one column per COLS. Returns the same
    keys with one column per TSR_COLS. A row reads only its series' values dated <= its own date."""
    raw = raw.sort_values(["symbol", "date"]).reset_index(drop=True)
    grid = np.arange(raw["date"].min(), raw["date"].max() + pd.Timedelta(days=1), dtype="datetime64[D]")
    days = grid.astype("int64")
    rt = np.searchsorted(grid, raw["date"].to_numpy().astype("datetime64[D]"))
    keys, ru = np.unique(raw["symbol"].to_numpy(), return_inverse=True)
    ru = ru.astype("int64")
    out = {}
    for est in COLS:
        x = np.full((len(grid), len(keys)), np.nan)
        x[rt, ru] = raw[est].to_numpy("float64")
        for w, name in zip(TSR_WINDOWS, TSR[est]):
            start = np.maximum(np.arange(len(grid)) - w + 1, 0)
            # coverage counts the FULL window's days, so a window clipped by the store's start cannot pass early
            need = np.full(len(grid), int(np.ceil(TSR_COVERAGE * w)), dtype="int64")
            out[name] = _window_stats(x, days, start, need, rt, ru)[0]
    return pd.concat([raw[["symbol", "date"]], pd.DataFrame(out, index=raw.index)], axis=1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def series_key(panel):
    """Which price series a panel row reads: its ticker, or SYMBOL@ucid for an archive coin."""
    k = panel["symbol"].astype(str).copy()
    for (s, u), _ in ARCHIVE.items():
        m = (panel["symbol"] == s) & (panel["ucid"].astype(str) == u)
        k[m] = "%s@%s" % (s, u)
    return k


def archive_series(panel, asof=None):
    """One OHLC frame per archive coin, read from the cached monthly zips, with a jump report."""
    out, report = [], []
    for (s, u), pair in ARCHIVE.items():
        rows = panel[(panel["symbol"] == s) & (panel["ucid"].astype(str) == u)]
        if rows.empty:                  # not listed yet on a cut panel
            continue
        f = fb.fetch_pair(pair, rows["date"].min(), rows["date"].max())
        if f is not None and asof is not None:
            f = f[f["date"] <= asof]
        if f is None or f.empty:        # not on Binance yet: the coin stays uncovered
            report.append(dict(key="%s@%s" % (s, u), pair=pair, bars=0))
            continue
        r = np.log(f["close"]).diff().abs()
        report.append(dict(key="%s@%s" % (s, u), pair=pair, first=str(f["date"].min().date()), last=str(f["date"].max().date()),
                           bars=len(f), panel_first=str(rows["date"].min().date()), panel_last=str(rows["date"].max().date()),
                           jumps=[str(d.date()) for d in f.loc[r > JUMP, "date"]]))
        out.append(f.assign(symbol="%s@%s" % (s, u)))
    cols = ["symbol", "date", "open", "high", "low", "close"]
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)), report


def identity_gate(panel, key, att, seed=0):
    """Per-row gate statistic and the wrong-coin null it is calibrated against.

    The expanding mean runs over each series' own panel rows in date order. Returns
    (stat per row, null quantiles on the full file and up to CALIB_END, pass rates at GATE_THRESHOLD)."""
    df = pd.DataFrame({"key": key.values, "date": panel["date"].values})
    for i, (r, sh) in enumerate(GATE.items()):
        df["p%d" % i] = pd.Series(att[r].values).groupby(df["date"]).rank(pct=True).values
        df["s%d" % i] = panel[sh].values
    keys = df["key"].unique()
    rng = np.random.default_rng(seed)
    perm = keys.copy()
    while (perm == keys).any():
        rng.shuffle(perm)
    look = pd.MultiIndex.from_arrays([df["date"], df["key"].map(dict(zip(keys, perm)))])
    gap, null = [], []
    for i in range(len(GATE)):
        wide = df.pivot(index="date", columns="key", values="p%d" % i)
        other = wide.stack(future_stack=True).reindex(look).to_numpy()
        gap.append((df["p%d" % i] - df["s%d" % i]).abs())
        null.append((pd.Series(other) - df["s%d" % i]).abs())
    df["gap"] = pd.concat(gap, axis=1).mean(axis=1)
    df["null"] = pd.concat(null, axis=1).mean(axis=1)
    order = df.sort_values(["key", "date"]).index
    grp = df.loc[order].groupby("key")
    expand = lambda col: grp[col].transform(lambda x: x.expanding(min_periods=GATE_MIN).mean()).reindex(df.index)
    stat, nstat = expand("gap"), expand("null")
    early = (df["date"] <= CALIB_END).to_numpy()
    cal = {"null_q%02d_full" % round(100 * GATE_Q): float(np.nanquantile(nstat, GATE_Q)),
           "null_q%02d_to_%s" % (round(100 * GATE_Q), CALIB_END.date()): float(np.nanquantile(nstat[early], GATE_Q)),
           "wrong_coin_pass_rate": float((nstat[nstat.notna()] <= GATE_THRESHOLD).mean()),
           "right_coin_pass_rate": float((stat[stat.notna()] <= GATE_THRESHOLD).mean())}
    return stat.to_numpy(), cal


def build(panel, spot, asof=None):
    """Return (panel with the 22 gated, binned columns and the coverage masks, manifest facts).

    asof cuts the panel, the spot store and the archive bars at that date first: the audit's way of
    asking what the block would have been with nothing after it."""
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    if asof is not None:
        panel = panel[panel["date"] <= asof].reset_index(drop=True)
        spot = spot[spot["date"] <= asof]
    panel["ucid"] = panel["ucid"].astype(str)
    key = series_key(panel)
    arch, arch_report = archive_series(panel, asof)
    base = spot.loc[spot["close"] > 0, ["symbol", "date", "open", "high", "low", "close"]]
    ohlc = pd.concat([base, arch[["symbol", "date", "open", "high", "low", "close"]]], ignore_index=True)
    raw = compute_volatility_signals(ohlc).drop_duplicates(["symbol", "date"])
    raw = raw.merge(own_history_ranks(raw), on=["symbol", "date"], validate="1:1")
    left = panel[["date"]].assign(symbol=key.values)
    att = attach_signals_asof(left, raw, COLS + TSR_COLS, lag_days=LAG, max_staleness_days=STALE)
    att = att.set_index(panel.index)
    stat, cal = identity_gate(panel, key, att)
    passed = stat <= GATE_THRESHOLD
    vals = att[COLS + TSR_COLS].where(pd.Series(passed, index=panel.index), axis=0)
    binned = bin_ties_preserved(vals, panel["date"])
    binned = binned.astype("float32")
    masks = {"cov_" + a: binned[cols].notna().all(axis=1) for a, cols in list(ARMS.items()) + [("all", COLS)]}
    masks.update({"cov_" + a: binned[cols].notna().all(axis=1) for a, cols in list(TSR_ARMS.items()) + [("tsr_all", TSR_COLS)]})
    extra = pd.DataFrame(masks).astype("int8").assign(vol_gate_stat=stat.astype("float32"), vol_series=key.values)
    out = pd.concat([panel, binned, extra.set_index(panel.index)], axis=1)
    seen = att[list(GATE)].notna().any(axis=1).to_numpy()
    facts = dict(gate=dict(statistic="expanding mean |pct(rebuilt vol) - shipped vol|, 20d and 60d", min_obs=GATE_MIN,
                           threshold=GATE_THRESHOLD, **cal),
                 rows=len(out), attached=float(seen.mean()), gated_out=float((seen & ~passed).mean()), archive_series=arch_report)
    return out, facts


def parse_args():
    p = argparse.ArgumentParser(description="Build the causal, coverage-masked volatility dataset.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0")
    p.add_argument("--out-suffix", default="-vol-causal")
    p.add_argument("--spot", default=SPOT)
    return p.parse_args()


def main():
    args = parse_args()
    api = get_api("crypto")
    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    panel = pd.read_parquet(src).reset_index()
    spot = pd.read_parquet(args.spot, columns=["symbol", "date", "open", "high", "low", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    print("source: %d rows, %d symbols, %s .. %s" % (len(panel), panel["symbol"].nunique(),
                                                     pd.to_datetime(panel["date"]).min().date(), pd.to_datetime(panel["date"]).max().date()))
    out, facts = build(panel, spot)
    g = facts["gate"]
    print("identity gate: stat <= %.2f; passes %.1f%% of wrong-coin and %.1f%% of own-coin rows with a statistic"
          % (g["threshold"], 100 * g["wrong_coin_pass_rate"], 100 * g["right_coin_pass_rate"]))
    print("   null 5%% quantile: %s" % {k: round(v, 3) for k, v in g.items() if k.startswith("null_")})
    print("attached %.1f%% of rows, gated out %.1f%%" % (100 * facts["attached"], 100 * facts["gated_out"]))
    for a in list(ARMS) + ["all"] + list(TSR_ARMS) + ["tsr_all"]:
        print("   cov_%-16s %.1f%%" % (a, 100 * out["cov_" + a].mean()))
    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.set_index("symbol").to_parquet(dest)
    source = download_if_missing(api, "crypto", args.source_version, "historical_meta_models.parquet")
    target = local_path("crypto", version, "historical_meta_models.parquet")
    if not os.path.exists(target):
        shutil.copy(source, target)
    arch_files = {pair: sorted(os.listdir(os.path.join(fb.CACHE, pair))) for pair in ARCHIVE.values()}
    manifest = dict(built_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), version=version,
                    label="timestamp-causal on a retrospectively reconstructed universe; not point-in-time",
                    source_train=dict(path=src, sha256=sha256(src)), spot=dict(path=args.spot, sha256=sha256(args.spot)),
                    archive={pair: hashlib.sha256("".join(sha256(os.path.join(fb.CACHE, pair, f)) for f in fs).encode()).hexdigest()
                             for pair, fs in arch_files.items()},
                    lag_days=LAG, max_staleness_days=STALE, arms=ARMS, tsr_arms=TSR_ARMS, tsr_windows=TSR_WINDOWS,
                    tsr_coverage=TSR_COVERAGE, **facts)
    json.dump(manifest, open(os.path.join(os.path.dirname(dest), "manifest.json"), "w"), indent=1)
    print("features: %d total -> %s (%.0f MB)" % (sum(c.startswith("feature_") for c in out), dest, os.path.getsize(dest) / 1e6))
    print("run it with:  --data-version %s" % version)


if __name__ == "__main__":
    main()
