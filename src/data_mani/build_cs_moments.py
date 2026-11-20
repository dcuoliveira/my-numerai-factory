"""
Build crypto/v2.0-cs (E198): cross-sectional statistics of the RAW (pre-bin) v2.0 features.

    python3 -m src.data_mani.build_cs_moments

Why raw. Every shipped v2.0 feature is rank-binned within its date (5/20/50/20/5), so any per-date
statistic of a shipped column is a constant: mean 0.5, spread and skew fixed by the bin masses. The
cross-section only moves on the raw values, so they are rebuilt from the spot store with the
definitions the reverse-engineering studies matched (rank agreement with the shipped bins 0.85-0.92):

    volatility_20d / _60d   population var of weekday simple returns, 19 / 59 returns      (E157)
    bollinger_20d           (C - SMA20) / population std20 of weekday closes                (E164)
    sharpe_ratio_20d / _60d mean / population std of weekday simple returns                 (E165)
    rsi_20d / _60d          SMA(gains) / (SMA(gains) + SMA(losses)) of weekday price changes (E166)
    log USD volume, 20d     log SMA20 of weekday volume, Yahoo rows only (USD); other venues quote
                            base-asset units, so mixing them would rank the unit, not the liquidity

Which features, and which statistics, came from a pre-2024 screen (EXPERIMENTS.md, E198): the
Bollinger CS median moderates that feature's daily IC (t +5.5), and raw volatility carries signal
inside its bins (t -5.6). Close-price levels, the dollar-change momentum and kurtosis were dropped.

Columns added (15, all on [0, 1], NaN where not honestly computable):

  per coin, over the day's gated coins
    feature_cs_{volatility,sharpe_ratio,rsi}_{20,60}d_q20   20-quantile rank (finer than the 5 bins)
    feature_cs_volatility_{20,60}d_rz                       0.5 + 0.5 tanh(robust z / 2) of log vol,
                                                            robust z = (x - median) / IQR
  per date (identical across coins), causal trailing percentile in time
    feature_cs_trend           median Bollinger z           (market above/below its 20d mean)
    feature_cs_trend_skew      Bowley skew of Bollinger z
    feature_cs_vol_level       median log vol 20d
    feature_cs_vol_disp        IQR of log vol 20d
    feature_cs_liq_disp        std of log USD volume 20d
  interactions, 0.5 + 2 (f - 0.5)(m - 0.5): a day-constant cannot move a linear model's within-date
  ranking, so ridge can only use the regime through an explicit product
    feature_cs_x_bollinger_trend   shipped bollinger_20d x cs_trend
    feature_cs_x_volatility_level  shipped volatility_20d x cs_vol_level

Coverage masks (non-feature, for B_covered / F_covered and routing): cov_cs_coin (every per-coin
column), cov_cs_date (every date-level column), cov_cs_all (all 15).

Causality. Signals at spot date d read closes <= d; a panel row D takes the newest one dated <= D-1
(max 4 days stale), the last completed bar at submission. Date-level statistics on D use only the
rows of D, so only prices <= D-1; the time percentile uses dates <= D. Series whose prices belong
to another coin are removed by build_vol_causal's causal identity gate (expanding mean of the gap
between our rebuilt volatility percentile and the shipped bin, frozen threshold), fed here with this
builder's own rebuilt 20d/60d volatility.

What this is NOT: point-in-time. The spot store was fetched in 2026 (retrospective universe), as in
v2.0-vol-causal. The manifest records the input hashes.
"""

import argparse
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data_mani.build_vol_causal import (GATE, GATE_THRESHOLD, archive_series, identity_gate,
                                            series_key, sha256)
from src.data_mani.macro_features import causal_trailing_percentile
from src.data_mani.momentum_features import attach_signals_asof
from src.data_mani.utils import download_if_missing, get_api, local_path

SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")
LAG, STALE, MIN_COINS = 1, 4, 30
RAW = ["vol20", "vol60", "boll20", "sharpe20", "sharpe60", "rsi20", "rsi60", "logusdvol20"]
GATE_SRC = {"vol20": "feature_vol_sqret_20d", "vol60": "feature_vol_sqret_60d"}   # names identity_gate reads
assert set(GATE_SRC.values()) == set(GATE)
Q20 = {"volatility_20d": "vol20", "volatility_60d": "vol60", "sharpe_ratio_20d": "sharpe20",
       "sharpe_ratio_60d": "sharpe60", "rsi_20d": "rsi20", "rsi_60d": "rsi60"}
COIN_COLS = ["feature_cs_%s_q20" % k for k in Q20] + ["feature_cs_volatility_20d_rz", "feature_cs_volatility_60d_rz"]
DATE_COLS = ["feature_cs_trend", "feature_cs_trend_skew", "feature_cs_vol_level", "feature_cs_vol_disp",
             "feature_cs_liq_disp"]
X_COLS = {"feature_cs_x_bollinger_trend": ("feature_bollinger_20d", "feature_cs_trend"),
          "feature_cs_x_volatility_level": ("feature_volatility_20d", "feature_cs_vol_level")}
ALL_COLS = COIN_COLS + DATE_COLS + list(X_COLS)


def raw_signals(ohlc):
    """Per (symbol, date) raw signals on the weekday calendar; a row dated d reads closes <= d."""
    out = []
    for sym, g in ohlc.sort_values("date").groupby("symbol", sort=False):
        g = g.set_index("date")
        g = g[g.index.dayofweek < 5]
        c, r, d = g["close"], g["close"].pct_change(fill_method=None), g["close"].diff()
        roll = lambda x, w: x.rolling(w, min_periods=w)
        gain, loss = d.clip(lower=0), (-d).clip(lower=0)
        f = pd.DataFrame(index=g.index)
        f["vol20"], f["vol60"] = roll(r, 19).var(ddof=0), roll(r, 59).var(ddof=0)
        f["boll20"] = (c - roll(c, 20).mean()) / roll(c, 20).std(ddof=0)
        for w in (20, 60):
            f["sharpe%d" % w] = roll(r, w).mean() / roll(r, w).std(ddof=0)
            gm, lm = roll(gain, w).mean(), roll(loss, w).mean()
            f["rsi%d" % w] = gm / (gm + lm)
        usd = g["volume"].where(g["usd_volume"])
        f["logusdvol20"] = np.log(roll(usd, 20).mean().where(lambda x: x > 0))
        out.append(f.replace([np.inf, -np.inf], np.nan).assign(symbol=sym).reset_index())
    return pd.concat(out, ignore_index=True)


def q20(x, dates):
    """Per-date rank on 20 levels {0, 1/19, .., 1}; ties take their average rank."""
    g = x.groupby(dates)
    r, n = g.rank(method="average") - 1, g.transform("count") - 1
    return (np.floor(20 * r / n.where(n > 0)).clip(upper=19) / 19).where(n > 0, 0.5).where(x.notna())


def date_stats(att, dates):
    """One row per date of the raw CS statistics, NaN on dates with fewer than MIN_COINS coins."""
    lv, lu = np.log(att["vol20"].where(att["vol20"] > 0)), att["logusdvol20"]
    df = pd.DataFrame({"d": dates, "b": att["boll20"], "lv": lv, "lu": lu})
    rows = {}
    for d, g in df.groupby("d"):
        b, v, u = g["b"].dropna(), g["lv"].dropna(), g["lu"].dropna()
        s = {}
        if len(b) >= MIN_COINS:
            q1, q2, q3 = np.quantile(b, [.25, .5, .75])
            s["trend"], s["trend_skew"] = q2, (q3 + q1 - 2 * q2) / (q3 - q1) if q3 > q1 else np.nan
        if len(v) >= MIN_COINS:
            q1, q2, q3 = np.quantile(v, [.25, .5, .75])
            s["vol_level"], s["vol_disp"] = q2, q3 - q1
        if len(u) >= MIN_COINS:
            s["liq_disp"] = u.std()
        rows[d] = s
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=[c[len("feature_cs_"):] for c in DATE_COLS]).sort_index()


def build(panel, spot, asof=None):
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    if asof is not None:
        panel = panel[panel["date"] <= asof].reset_index(drop=True)
        spot = spot[spot["date"] <= asof]
    panel["ucid"] = panel["ucid"].astype(str)
    key = series_key(panel)
    arch, arch_report = archive_series(panel, asof)
    base = spot.loc[spot["close"] > 0, ["symbol", "date", "close", "volume", "source"]].assign(
        usd_volume=lambda f: f["source"].eq("yahoo"))
    arch = arch.assign(volume=np.nan, usd_volume=False)[["symbol", "date", "close", "volume", "usd_volume"]]
    ohlc = pd.concat([base.drop(columns="source"), arch], ignore_index=True)
    raw = raw_signals(ohlc).drop_duplicates(["symbol", "date"])
    left = panel[["date"]].assign(symbol=key.values)
    att = attach_signals_asof(left, raw, RAW, lag_days=LAG, max_staleness_days=STALE).set_index(panel.index)
    stat, cal = identity_gate(panel, key, att.rename(columns=GATE_SRC))
    passed = pd.Series(stat <= GATE_THRESHOLD, index=panel.index)
    seen = att[RAW].notna().any(axis=1)
    att = att[RAW].where(passed, axis=0)
    dates = panel["date"]

    new = pd.DataFrame(index=panel.index)
    for name, src in Q20.items():
        new["feature_cs_%s_q20" % name] = q20(att[src], dates)
    for w in (20, 60):
        lv = np.log(att["vol%d" % w].where(att["vol%d" % w] > 0))
        g = lv.groupby(dates)
        med, iqr = g.transform("median"), g.transform(lambda x: x.quantile(.75) - x.quantile(.25))
        ok = g.transform("count") >= MIN_COINS
        new["feature_cs_volatility_%dd_rz" % w] = (0.5 + 0.5 * np.tanh((lv - med) / iqr.where(iqr > 0) / 2)).where(ok)

    ds = date_stats(att, dates)
    pct = pd.DataFrame({"feature_cs_" + c: causal_trailing_percentile(ds[c]) for c in ds.columns})
    new = new.join(pd.DataFrame(pct.reindex(dates.values).to_numpy(), columns=pct.columns, index=panel.index))
    for col, (f, m) in X_COLS.items():
        new[col] = 0.5 + 2 * (panel[f] - 0.5) * (new[m] - 0.5)
    new = new[ALL_COLS].astype("float32")
    masks = pd.DataFrame({"cov_cs_coin": new[COIN_COLS].notna().all(axis=1),
                          "cov_cs_date": new[DATE_COLS].notna().all(axis=1),
                          "cov_cs_all": new.notna().all(axis=1)}).astype("int8")
    out = pd.concat([panel, new, masks, pd.DataFrame({"cs_gate_stat": stat.astype("float32"), "cs_series": key.values},
                                                     index=panel.index)], axis=1)
    facts = dict(gate=dict(threshold=GATE_THRESHOLD, **cal), rows=len(out),
                 attached=float(seen.mean()), gated_out=float((seen & ~passed).mean()),
                 raw_attached={c: float(att[c].notna().mean()) for c in RAW},
                 coverage={c: float(masks[c].mean()) for c in masks}, archive_series=arch_report,
                 date_stats_dates=int(ds.notna().all(axis=1).sum()))
    return out, facts


def parse_args():
    p = argparse.ArgumentParser(description="Build crypto/v2.0-cs: CS statistics of raw v2.0 features (E198).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0")
    p.add_argument("--out-suffix", default="-cs")
    p.add_argument("--spot", default=SPOT)
    return p.parse_args()


def main():
    args = parse_args()
    api = get_api("crypto")
    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    panel = pd.read_parquet(src).reset_index()
    spot = pd.read_parquet(args.spot, columns=["symbol", "date", "close", "volume", "source"])
    spot["date"] = pd.to_datetime(spot["date"])
    out, facts = build(panel, spot)
    g = facts["gate"]
    print("identity gate <= %.2f: wrong-coin pass %.1f%%, own-coin pass %.1f%%"
          % (g["threshold"], 100 * g["wrong_coin_pass_rate"], 100 * g["right_coin_pass_rate"]))
    print("raw attached:", {k: round(v, 3) for k, v in facts["raw_attached"].items()})
    print("coverage:", {k: round(v, 3) for k, v in facts["coverage"].items()})
    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.set_index("symbol").to_parquet(dest)
    meta_src = download_if_missing(api, "crypto", args.source_version, "historical_meta_models.parquet")
    meta_dst = local_path("crypto", version, "historical_meta_models.parquet")
    if not os.path.exists(meta_dst):
        shutil.copy(meta_src, meta_dst)
    manifest = dict(built_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), version=version, study="E198",
                    label="timestamp-causal on a retrospectively reconstructed universe; not point-in-time",
                    source_train=dict(path=src, sha256=sha256(src)), spot=dict(path=args.spot, sha256=sha256(args.spot)),
                    lag_days=LAG, max_staleness_days=STALE, min_coins=MIN_COINS,
                    columns=dict(coin=COIN_COLS, date=DATE_COLS, interaction=X_COLS), **facts)
    with open(os.path.join(os.path.dirname(dest), "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print("features: %d -> %s (%.0f MB)" % (sum(c.startswith("feature_") for c in out), dest, os.path.getsize(dest) / 1e6))
    print("run it with:  --data-version %s" % version)


if __name__ == "__main__":
    main()
