"""
Build crypto/v2.0-vol-causal-macrovol: crypto/v2.0-vol-causal plus interactions of each coin's
causal volatility with macro-market volatility regimes, and coin-level macro exposures.

    python3 -m src.data_mani.build_vol_macrovol                      # the dataset
    python3 -m src.data_mani.build_vol_macrovol --variant circular   # placebo: macro rotated ~2 years
    python3 -m src.data_mani.build_vol_macrovol --variant future     # LOOK-AHEAD placebo, never select on it
    python3 -m src.data_mani.build_vol_macrovol --variant scalar     # control: positive v x m products

Every vol-causal column, row, NaN and mask is copied unchanged; the new columns are prefixed
feature_mvx_. Placebo and control variants write ONLY their new columns plus symbol/date, to
data/crypto/v2.0-vol-causal-macrovol-<variant>/, never the deployable dataset.

Macro series (src/data_mani/fetch_macro_vol.py, data/crypto/macro/macro_vol_daily.parquet):
    gold   log returns of GLD            sp500  log returns of ^GSPC      dxy  log returns of DX-Y.NYB
    y10    daily change of the 10-year par yield in basis points (never a percentage return)
    vixlvl the VIX level itself, an implied-volatility regime series
    vixvov realised volatility of daily log changes of VIX ("VIX vol-of-vol")

Macro volatility m(k,t,w) uses the crypto block's conventions on each source's own trading days:
mean squared change over w = 5/10/20/40/60/120 observations (full window required), and EWMA of the
squared change with lambda 0.90/0.94/0.97/0.99 (min periods 1/(1-lambda)). Each crypto estimator is
paired with ONE macro estimator: sqret_w, park_w and yz_w with macro sqret_w, ewma_lam with macro
ewma_lam. vixlvl has one series, paired with every estimator.

Causal normalisation, strictly before the feature's own timestamp:
    zc(i,t,w)  crypto: log(v + 1e-12), percentile among the coin's own values in (t-365d, t-1d],
               at least 120 of them, mapped through the normal quantile: z = Phi^-1((below + equal/2
               + 1/2) / (n + 1)). The +1/2 and n+1 keep z finite.
    zm(k,t,w)  macro: the same on log m, over the previous 252 trading days, at least 120.
    pm(k,t,w)  the macro percentile itself, (below + equal/2) / n, in [0,1].
zc is computed on the coin's daily spot series and attached as-of D-1 exactly like the vol-causal
block (same series keys, archive coins, identity gate). zm and pm are attached as-of D-1 by date,
backward only, at most STALE days old, else missing.

Interaction families, per (crypto estimator, macro source):
    sgn   zc * zm                 sign/regime effect: zm is constant within a date, so after per-date
                                  binning this reverses the coin ORDER of zc when zm < 0 and preserves
                                  it when zm > 0; the size of the common multiplier washes out.
    dist  |zc - zm|               how far a coin's own vol regime sits from the macro regime.
    hi    zc where pm >= 0.80     crypto vol only inside a high-stress macro regime, else missing
    lo    zc where pm <= 0.20     crypto vol only inside a low-stress macro regime, else missing
Exposures, per coin and macro change series (gold/sp500/dxy log returns, y10 bp changes, VIX log
changes) over the previous 60 and 120 trading days (at least 40 / 80 aligned days), computed on
macro trading days from the coin's daily close on the same date:
    cov, corr, beta = cov/var_m, resvol = sqrt(var_c - beta^2 var_m), r2 = clip(beta^2 var_m/var_c, 0, 1)
A crypto close is the UTC day's last bar and a US close is about 20:00-21:00 UTC, so the two returns
are offset by a few hours; the exposure measures co-movement at a daily scale, not intraday.

Every new column is then rank-binned within each date (history_features.bin_ties_preserved, Numerai's
exact rule) over the coins that have it; missing stays missing; no indicator is added. Non-feature
coverage masks cov_mvx_<source> mark rows where that source's sgn, dist and exposure columns are all
observed (hi/lo are missing by design and do not enter a mask); cov_mvx_all for all six.

NOT point-in-time: macro histories are today's downloads (raw runs are kept with hashes so they can
be compared over time), and the crypto side inherits vol-causal's retrospectively reconstructed
universe.
"""

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

import numba
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data_mani import build_vol_causal as V
from src.data_mani.history_features import bin_ties_preserved
from src.data_mani.momentum_features import attach_signals_asof
from src.data_mani.volatility_features import VOL_SPECS, compute_volatility_signals
from src.data_mani.macrovol_contract import attach_available, validate_store

BASE = os.path.join("data", "crypto", "v2.0-vol-causal")
MACRO = os.path.join("data", "crypto", "macro", "macro_vol_daily.parquet")
LAG, STALE = 1, 4
EPS = 1e-12
ZC_DAYS, ZC_MIN = 365, 120              # crypto: calendar days of daily estimates
ZM_OBS, ZM_MIN = 252, 120               # macro: trading days
HI, LO = 0.80, 0.20
EXPO = ((60, 40), (120, 80))            # (window, minimum aligned days)
WINDOWS = (5, 10, 20, 40, 60, 120)
LAMBDAS = (0.90, 0.94, 0.97, 0.99)
SOURCES = ("gold", "sp500", "dxy", "y10", "vixlvl", "vixvov")
CHANGES = {"gold": "gold", "sp500": "sp500", "dxy": "dxy", "y10": "y10", "vix": "vix"}   # exposure series
CIRCULAR_SHIFT, FUTURE_SHIFT = 504, 20  # trading days
EST = list(VOL_SPECS)


def short(e):
    return e[len("feature_vol_"):]


def partner(e):
    """The macro estimator paired with crypto estimator e."""
    spec = VOL_SPECS[e]
    return "ewma_lam%d" % round(100 * spec["lam"]) if spec["kind"] == "ewma" else "sqret_%dd" % spec["window"]


def macro_estimators():
    return ["sqret_%dd" % w for w in WINDOWS] + ["ewma_lam%d" % round(100 * l) for l in LAMBDAS]


@numba.njit(cache=True)
def _prior_rank(x, start, need, rt, ru):
    """For each row, (below, equal, n) of x[t,u] among x[start[t]..t-1, u]; n = -1 if too few."""
    m = len(rt)
    below = np.zeros(m)
    equal = np.zeros(m)
    count = np.full(m, -1.0)
    for i in range(m):
        t, u = rt[i], ru[i]
        cur = x[t, u]
        if cur != cur:
            continue
        b, e, n = 0, 0, 0
        for k in range(start[t], t):
            v = x[k, u]
            if v != v:
                continue
            n += 1
            if v < cur:
                b += 1
            elif v == cur:
                e += 1
        if n >= need:
            below[i], equal[i], count[i] = b, e, n
    return below, equal, count


def prior_scores(frame, cols, key, window, min_obs, calendar):
    """Strictly-prior percentile and probit score of log(value + EPS) per column.

    calendar=True: window is calendar days on a daily grid; False: window is observations of the key."""
    frame = frame.sort_values([key, "date"]).reset_index(drop=True)
    keys, ru = np.unique(frame[key].to_numpy(), return_inverse=True)
    ru = ru.astype("int64")
    if calendar:
        grid = np.arange(frame["date"].min(), frame["date"].max() + pd.Timedelta(days=1), dtype="datetime64[D]")
        rt = np.searchsorted(grid, frame["date"].to_numpy().astype("datetime64[D]"))
        start = np.maximum(np.arange(len(grid)) - window, 0)
    else:                                   # one key per series: observation index is the row's position
        rt = frame.groupby(key).cumcount().to_numpy()
        grid = np.arange(rt.max() + 1)
        start = np.maximum(grid - window, 0)
    pct, z = {}, {}
    for c in cols:
        x = np.full((len(grid), len(keys)), np.nan)
        values = frame[c].to_numpy("float64")
        positive = np.isfinite(values) & (values > 0)
        x[rt, ru] = np.log(np.where(positive, values, np.nan) + EPS)
        b, e, n = _prior_rank(x, start, min_obs, rt, ru)
        ok = n > 0
        p = np.where(ok, (b + e / 2.0) / np.where(ok, n, 1), np.nan)
        q = np.where(ok, (b + e / 2.0 + 0.5) / (np.where(ok, n, 1) + 1.0), np.nan)
        pct[c], z[c] = p, norm.ppf(q)
    return frame[[key, "date"]], pd.DataFrame(pct, index=frame.index), pd.DataFrame(z, index=frame.index)


def macro_changes(store, variant):
    """Daily change series per source on its own trading days (wide: date x source)."""
    w = store.pivot(index="date", columns="series", values="value").sort_index()
    out = {}
    for s in ("gold", "sp500", "dxy", "vix"):
        x = w[s].dropna()
        out[s] = np.log(x).diff()
    y = w["y10"].dropna()
    out["y10"] = 100.0 * y.diff()            # basis points
    lvl = w["vix"].dropna()
    if variant in ("circular", "future"):
        shift = CIRCULAR_SHIFT if variant == "circular" else -FUTURE_SHIFT
        roll = lambda s: pd.Series(np.roll(s.to_numpy(), shift), index=s.index) if variant == "circular" else s.shift(shift)
        out = {k: roll(v.dropna()) for k, v in out.items()}
        lvl = roll(lvl)
    return out, lvl


def macro_vol(changes, lvl):
    """m(k,t,w) per source and paired estimator, on the source's trading days. Long frame."""
    rows = []
    for k in SOURCES:
        if k == "vixlvl":
            f = pd.DataFrame({e: lvl for e in macro_estimators()})
        else:
            r2 = changes["vix" if k == "vixvov" else k].dropna() ** 2
            f = pd.DataFrame({"sqret_%dd" % w: r2.rolling(w, min_periods=w).mean() for w in WINDOWS})
            for l in LAMBDAS:
                f["ewma_lam%d" % round(100 * l)] = r2.ewm(alpha=1 - l, adjust=False, min_periods=int(round(1 / (1 - l)))).mean()
        rows.append(f.dropna(how="all").rename_axis("date").reset_index().assign(source=k))
    return pd.concat(rows, ignore_index=True)


def exposures(spot_close, changes):
    """Rolling coin x macro covariance stats on macro trading days. Long frame keyed symbol(series key), date."""
    lc = np.log(spot_close.pivot_table(index="date", columns="symbol", values="close"))
    parts = []
    for k, r_m in changes.items():
        r_m = r_m.dropna()
        # Calculate a DAILY crypto return before selecting macro trading dates.
        # Differencing after reindexing incorrectly gives Friday-to-Monday coin
        # returns against a Monday-only macro change.
        daily = lc.reindex(pd.date_range(lc.index.min(), lc.index.max(), freq="D"))
        r_c = daily.diff().reindex(r_m.index)
        x = pd.DataFrame(np.repeat(r_m.to_numpy()[:, None], r_c.shape[1], axis=1), index=r_c.index, columns=r_c.columns)
        ok = r_c.notna() & x.notna()
        a, b = r_c.where(ok), x.where(ok)
        for w, mp in EXPO:
            roll = lambda f: f.rolling(w, min_periods=mp)
            n = roll(ok.astype(float).where(ok)).count()
            ma, mb = roll(a).mean(), roll(b).mean()
            cov = roll(a * b).mean() - ma * mb
            va, vb = roll(a * a).mean() - ma ** 2, roll(b * b).mean() - mb ** 2
            good = n >= mp
            beta = cov / vb
            stats = {"cov": cov, "corr": cov / np.sqrt(va * vb), "beta": beta,
                     "resvol": np.sqrt((va - beta ** 2 * vb).clip(lower=0)), "r2": (beta ** 2 * vb / va).clip(0, 1)}
            for s, f in stats.items():
                name = "feature_mvx_%s_%s_%d" % (k if k != "vix" else "vixchg", s, w)
                parts.append(f.where(good).stack(future_stack=True).rename(name))
    out = pd.concat(parts, axis=1).rename_axis(["date", "symbol"]).reset_index()
    return out.dropna(how="all", subset=[c for c in out if c.startswith("feature_")])


def columns():
    """Every new feature name with its formula, for the manifest."""
    doc = {}
    for k in SOURCES:
        for e in EST:
            m = "m(%s, %s)" % (k, "level" if k == "vixlvl" else partner(e))
            doc["feature_mvx_%s_sgn_%s" % (k, short(e))] = "zc(%s) * zm(%s)" % (short(e), m)
            doc["feature_mvx_%s_dist_%s" % (k, short(e))] = "abs(zc(%s) - zm(%s))" % (short(e), m)
            doc["feature_mvx_%s_hi_%s" % (k, short(e))] = "zc(%s) where pm(%s) >= %.2f, else missing" % (short(e), m, HI)
            doc["feature_mvx_%s_lo_%s" % (k, short(e))] = "zc(%s) where pm(%s) <= %.2f, else missing" % (short(e), m, LO)
    for k in ("gold", "sp500", "dxy", "y10", "vixchg"):
        for w, mp in EXPO:
            for s, f in (("cov", "cov(r_c, r_m)"), ("corr", "corr(r_c, r_m)"), ("beta", "cov / var(r_m)"),
                         ("resvol", "sqrt(var(r_c) - beta^2 var(r_m))"), ("r2", "clip(beta^2 var(r_m) / var(r_c), 0, 1)")):
                doc["feature_mvx_%s_%s_%d" % (k, s, w)] = "%s over the previous %d %s trading days (>= %d)" % (f, w, k, mp)
    return doc


def source_cols(k, cols):
    keep = [c for c in cols if c.startswith("feature_mvx_%s_" % k) and "_hi_" not in c and "_lo_" not in c]
    expo = {"vixlvl": None, "vixvov": "vixchg"}.get(k, k)
    if expo and expo != k:
        keep += [c for c in cols if c.startswith("feature_mvx_%s_" % expo)]
    return keep


def build(variant="main", asof=None, macro_path=MACRO, research_retrospective=False):
    """asof cuts the base panel, spot store, archive bars and macro store at that date first (the audit)."""
    store = pd.read_parquet(macro_path)
    if not research_retrospective:
        store = validate_store(store)
    base = pd.read_parquet(os.path.join(BASE, "train.parquet")).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    if asof is not None:
        base = base[base["date"] <= asof].reset_index(drop=True)
    panel = base[["symbol", "ucid", "date", "feature_volatility_20d", "feature_volatility_60d"]].copy()
    panel["ucid"] = panel["ucid"].astype(str)
    key = V.series_key(panel)
    if not (key.values == base["vol_series"].values).all():
        raise SystemExit("series keys differ from vol-causal's: rebuild vol-causal first")
    spot = pd.read_parquet(V.SPOT, columns=["symbol", "date", "open", "high", "low", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    if asof is not None:
        spot = spot[spot["date"] <= asof]
    arch, _ = V.archive_series(panel, asof)
    ohlc = pd.concat([spot.loc[spot["close"] > 0], arch[["symbol", "date", "open", "high", "low", "close"]]], ignore_index=True)
    raw = compute_volatility_signals(ohlc).drop_duplicates(["symbol", "date"])
    keys, _, zc = prior_scores(raw, EST, "symbol", ZC_DAYS, ZC_MIN, calendar=True)
    zc_frame = pd.concat([keys, zc.add_prefix("zc_")], axis=1)
    passed = base["vol_gate_stat"].to_numpy() <= V.GATE_THRESHOLD
    left = panel[["date"]].assign(symbol=key.values)
    zcp = attach_signals_asof(left, zc_frame, ["zc_" + e for e in EST], lag_days=LAG, max_staleness_days=STALE)
    zcp = zcp[["zc_" + e for e in EST]].set_axis(panel.index).where(pd.Series(passed, index=panel.index), axis=0)

    if asof is not None:
        store = store[pd.to_datetime(store["date"]) <= asof]
    changes, lvl = macro_changes(store, variant)
    mv = macro_vol(changes, lvl)
    est = macro_estimators()
    att = {}
    for k in SOURCES:
        f = mv[mv["source"] == k].drop(columns="source").assign(src=k)
        _, p, z = prior_scores(f, est, "src", ZM_OBS, ZM_MIN, calendar=False)
        g = pd.concat([f[["date"]].reset_index(drop=True), p.add_prefix("pm_").reset_index(drop=True), z.add_prefix("zm_").reset_index(drop=True)], axis=1)
        g = g.assign(symbol="__macro__")
        names = list(g.columns.drop(["date", "symbol"]))
        if research_retrospective:
            d = attach_signals_asof(panel[["date"]].assign(symbol="__macro__"), g, names,
                                    lag_days=LAG, max_staleness_days=STALE)
        else:
            obs = store[store["series"] == ("vix" if k in ("vixlvl", "vixvov") else k)]
            d = attach_available(panel[["date"]].assign(symbol="__macro__"), g, names, obs,
                                 lag_days=LAG, stale_days=STALE)
        att[k] = d.set_axis(panel.index)

    new = {}
    for k in SOURCES if variant != "scalar" else ():
        for e in EST:
            c, zm, pm = zcp["zc_" + e], att[k]["zm_" + partner(e)], att[k]["pm_" + partner(e)]
            s = short(e)
            new["feature_mvx_%s_sgn_%s" % (k, s)] = c * zm
            new["feature_mvx_%s_dist_%s" % (k, s)] = (c - zm).abs()
            new["feature_mvx_%s_hi_%s" % (k, s)] = c.where(pm >= HI)
            new["feature_mvx_%s_lo_%s" % (k, s)] = c.where(pm <= LO)
    if variant == "scalar":
        # control: raw positive crypto variance times the raw positive macro variance, per date a common
        # positive multiplier, so the per-date rank is the crypto estimator's own (rank-equivalent).
        rv = attach_signals_asof(left, raw, EST, lag_days=LAG, max_staleness_days=STALE)[EST].set_axis(panel.index)
        rv = rv.where(pd.Series(passed, index=panel.index), axis=0)
        mvw = {}
        for k in SOURCES:
            f = mv[mv["source"] == k].drop(columns="source").assign(symbol="__macro__")
            if research_retrospective:
                mvw[k] = attach_signals_asof(panel[["date"]].assign(symbol="__macro__"), f, est, lag_days=LAG,
                                             max_staleness_days=STALE).set_axis(panel.index)
            else:
                obs = store[store["series"] == ("vix" if k in ("vixlvl", "vixvov") else k)]
                mvw[k] = attach_available(panel[["date"]].assign(symbol="__macro__"), f, est, obs,
                                         lag_days=LAG, stale_days=STALE)
            for e in EST:
                new["feature_mvx_%s_mul_%s" % (k, short(e))] = rv[e] * mvw[k][partner(e)]
    else:
        sc = spot.loc[spot["close"] > 0, ["symbol", "date", "close"]]
        sc = pd.concat([sc, arch[["symbol", "date", "close"]]], ignore_index=True).drop_duplicates(["symbol", "date"])
        ex = exposures(sc, changes)
        ecols = [c for c in ex if c.startswith("feature_")]
        # Join each exposure on its own source calendar/availability. A fresh
        # observation from another market must not refresh a stale exposure.
        attached = []
        for source in CHANGES:
            prefix = "feature_mvx_%s_" % ("vixchg" if source == "vix" else source)
            names = [c for c in ecols if c.startswith(prefix)]
            dates = changes[source].dropna().index
            frame = ex[ex["date"].isin(dates)][["symbol", "date"] + names]
            if research_retrospective:
                part = attach_signals_asof(left, frame, names, lag_days=LAG, max_staleness_days=STALE)[names]
            else:
                part = attach_available(left, frame, names, store[store["series"] == source],
                                        lag_days=LAG, stale_days=STALE)
            attached.append(part.set_axis(panel.index))
        exp_att = pd.concat(attached, axis=1)
        exp_att = exp_att.where(pd.Series(passed, index=panel.index), axis=0)
        for c in ecols:
            new[c] = exp_att[c]
    raw_new = pd.DataFrame(new, index=panel.index)
    binned = bin_ties_preserved(raw_new.astype("float64"), panel["date"]).astype("float32")
    return base, panel, binned, raw_new


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Build the macro-volatility interaction dataset or a placebo.")
    ap.add_argument("--variant", default="main", choices=["main", "circular", "future", "scalar"])
    ap.add_argument("--macro", default=MACRO, help="Frozen-version macro store with availability evidence")
    ap.add_argument("--research-retrospective", action="store_true",
                    help="Unverified histories; writes a separate -retrospective dataset")
    a = ap.parse_args()
    base, panel, binned, raw_new = build(a.variant, macro_path=a.macro,
                                        research_retrospective=a.research_retrospective)
    cols = list(binned.columns)
    if a.variant == "main":
        masks = {"cov_mvx_" + k: binned[source_cols(k, cols)].notna().all(axis=1) for k in SOURCES}
        masks["cov_mvx_all"] = pd.concat(masks.values(), axis=1).all(axis=1)
        out = pd.concat([base, binned, pd.DataFrame(masks).astype("int8")], axis=1)
        dest_dir = BASE + "-macrovol"
    else:
        out = pd.concat([panel[["symbol", "date"]], binned], axis=1)
        dest_dir = BASE + "-macrovol-" + {"circular": "placebo-circular", "future": "placebo-future-LOOKAHEAD", "scalar": "control-scalar"}[a.variant]
    if a.research_retrospective:
        dest_dir += "-retrospective"
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "train.parquet")
    out.set_index("symbol").to_parquet(dest)
    if a.variant == "main":
        shutil.copy(os.path.join(BASE, "historical_meta_models.parquet"), os.path.join(dest_dir, "historical_meta_models.parquet"))
    macro_store = pd.read_parquet(a.macro)
    if "run" in macro_store and a.research_retrospective:
        run = macro_store["run"].iloc[0]
        raw_manifest = json.load(open(os.path.join(os.path.dirname(a.macro), "raw", run, "manifest.json")))
    else:
        raw_manifest = macro_store.drop(columns=["value"]).astype(str).to_dict("records")
    valid = binned.notna().groupby(panel["date"])
    cov_by_date = valid.mean()
    cov_by_date.to_parquet(os.path.join(dest_dir, "coverage_by_date.parquet"))
    span = {c: dict(first=str(cov_by_date.index[cov_by_date[c] > 0].min().date()) if (cov_by_date[c] > 0).any() else None,
                    last=str(cov_by_date.index[cov_by_date[c] > 0].max().date()) if (cov_by_date[c] > 0).any() else None,
                    coverage=round(float(binned[c].notna().mean()), 4)) for c in cols}
    doc = columns()
    manifest = dict(built_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), variant=a.variant, dataset=dest_dir,
                    macro_availability_verified=not a.research_retrospective,
                    label=("timestamp-causal on a retrospectively reconstructed universe; not point-in-time"
                           if a.research_retrospective else
                           "publication-aware frozen macro versions; crypto inherits the base dataset's reconstruction")
                          + ("" if a.variant == "main" else "; PLACEBO/CONTROL, never deployable or selectable"),
                    timing=dict(cutoff="D-%d" % LAG, max_staleness_days=STALE, join="backward as-of, never backfilled"),
                    normalisation=dict(crypto=dict(window_calendar_days=ZC_DAYS, min_obs=ZC_MIN), macro=dict(window_trading_days=ZM_OBS, min_obs=ZM_MIN),
                                       transform="log(value + %g), strictly-prior percentile, probit" % EPS),
                    macro_estimators=dict(windows=WINDOWS, ewma_lambdas=LAMBDAS, pairing="sqret/park/yz w -> macro sqret w; ewma lam -> macro ewma lam"),
                    regime_gates=dict(hi=HI, lo=LO), exposure_windows=[dict(window=w, min_obs=m) for w, m in EXPO],
                    placebo_shift=dict(circular_trading_days=CIRCULAR_SHIFT, future_trading_days=FUTURE_SHIFT),
                    macro_series=raw_manifest, columns={c: dict(formula=doc.get(c, "control: raw v * raw m"), **span[c]) for c in cols},
                    coverage_by_month=cov_by_date.groupby(cov_by_date.index.to_period("M")).mean().mean(axis=1).round(4).rename(str).to_dict(),
                    base_columns={c: "unchanged from crypto/v2.0-vol-causal:" + c for c in base},
                    coverage_by_date="coverage_by_date.parquet",
                    inputs={p: sha256(p) for p in (os.path.join(BASE, "train.parquet"),
                                                  os.path.join(BASE, "manifest.json"), V.SPOT, a.macro)})
    json.dump(manifest, open(os.path.join(dest_dir, "manifest.json"), "w"), indent=1, default=str)
    print("%s: %d new columns, %d rows -> %s (%.0f MB)" % (a.variant, len(cols), len(out), dest, os.path.getsize(dest) / 1e6))
    if a.variant == "main":
        for k in list(SOURCES) + ["all"]:
            print("   cov_mvx_%-7s %.1f%%" % (k, 100 * out["cov_mvx_" + k].mean()))


if __name__ == "__main__":
    main()
