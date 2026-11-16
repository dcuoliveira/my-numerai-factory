"""
Extend data/crypto/spot/ohlcv.parquet with daily OHLCV from free, no-key EXCHANGE
public APIs, for the panel coins the yfinance -> Binance cascade never matched.

    python3 -m src.data_mani.fetch_exchange_spot                 # the live-universe misses
    python3 -m src.data_mani.fetch_exchange_spot --symbols HYPE PI
    python3 -m src.data_mani.fetch_exchange_spot --write          # validate + append

Per coin, every venue is tried and the series with the most daily bars inside the
coin's panel span wins (ties -> venue order). Venues and their free history depth:

    gate        full history (2 requests from 2020)     api.gateio.ws
    kucoin      full history (1500/request)             api.kucoin.com
    okx         full history (100/request, paginated)   www.okx.com
    bybit       full history (1000/request)             api.bybit.com
    bitget      full history (200/request)              api.bitget.com
    coinbase    full history (300/request)              api.exchange.coinbase.com
    hyperliquid full history (spot pairs only)          api.hyperliquid.xyz
    mexc        only back to ~2025-06 on the free API   api.mexc.com

Exchanges list only coins still trading, so this does NOT fix survivorship for coins
that left the universe -- it fills coverage of the live universe.

IDENTITY CHECK. A venue knows a ticker, not a CoinMarketCap id, and tickers are
reused (SHARE in the 2020 panel is ucid 6868, today's SHARE is 39955). So each
fetched series is checked against the coin's own shipped Numerai features over its
panel dates: the rebuilt 20d return vs feature_momentum_20d and the rebuilt 20d
realised vol vs feature_volatility_20d (time-series Spearman). Both are taken
MARKET-RELATIVE (minus / over the cross-sectional median of the existing spot file)
where that file reaches, else raw. Thresholds come from a WRONG-TOKEN null: each of
--calib known-good coins' features scored against a different coin's prices; a
series passes when its mean score clears the null's 95th pct, or its vol score alone
clears the null's 99th pct. The vol-only route exists because Numerai's momentum
feature does not track the rebuilt return for sub-cent coins -- SHIB and PEPE, both
certainly the right token, score -0.03 / -0.38 on momentum and 0.41 / 0.73 on vol.
Coins with no panel dates to check against are written but flagged unverified.
"""

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

SPOT_DIR = "data/crypto/spot"
OHLCV = os.path.join(SPOT_DIR, "ohlcv.parquet")
PANEL = "data/crypto/v2.0/train.parquet"
LIVE = "data/crypto/live/live.parquet"
CACHE = os.path.join(SPOT_DIR, "_exchange_cache")
START = pd.Timestamp("2020-01-01")
DAY = 86400
COLS = ["date", "open", "high", "low", "close", "volume"]


def _get(url, method="GET", **kw):
    for k in range(4):
        try:
            r = requests.request(method, url, timeout=30, **kw)
            if r.status_code == 429:
                time.sleep(2 + 3 * k)
                continue
            if not r.ok:
                return None
            return r.json()
        except (requests.RequestException, ValueError):
            time.sleep(1 + k)
    return None


def _frame(rows):
    """rows: list of (unix_seconds, o, h, l, c, v) -> daily frame, de-duplicated."""
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"].astype("int64"), unit="s").dt.normalize()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["close"] > 0].drop_duplicates("date").sort_values("date")
    return df[COLS].reset_index(drop=True) if len(df) else None


def _paginate_forward(fetch, step_days, now):
    """Walk [START, now] forward in windows of step_days; fetch(t0, t1) -> rows."""
    rows, t0, end = [], int(START.timestamp()), int(now.timestamp())
    while t0 < end:
        t1 = min(t0 + step_days * DAY, end)
        part = fetch(t0, t1)
        if part is None:
            return rows or None
        rows += part
        t0 = t1
    return rows


def gate(sym, now):
    pair = sym.upper() + "_USDT"

    def f(t0, t1):
        j = _get("https://api.gateio.ws/api/v4/spot/candlesticks",
                 params={"currency_pair": pair, "interval": "1d", "from": t0, "to": t1})
        if j is None or isinstance(j, dict):
            return None
        return [(int(x[0]), x[5], x[3], x[4], x[2], x[6]) for x in j]
    return _frame(_paginate_forward(f, 999, now))


def kucoin(sym, now):
    pair = sym.upper() + "-USDT"

    def f(t0, t1):
        j = _get("https://api.kucoin.com/api/v1/market/candles",
                 params={"symbol": pair, "type": "1day", "startAt": t0, "endAt": t1})
        if not j or j.get("code") != "200000":
            return None
        return [(int(x[0]), x[1], x[3], x[4], x[2], x[5]) for x in j["data"]]
    return _frame(_paginate_forward(f, 1400, now))


def okx(sym, now):
    inst, rows, after = sym.upper() + "-USDT", [], None
    while True:
        p = {"instId": inst, "bar": "1Dutc", "limit": 100}
        if after:
            p["after"] = after
        j = _get("https://www.okx.com/api/v5/market/history-candles", params=p)
        if not j or j.get("code") != "0" or not j["data"]:
            break
        rows += [(int(x[0]) // 1000, x[1], x[2], x[3], x[4], x[5]) for x in j["data"]]
        after = j["data"][-1][0]
        if int(after) // 1000 < START.timestamp():
            break
    return _frame(rows)


def bybit(sym, now):
    pair = sym.upper() + "USDT"

    def f(t0, t1):
        j = _get("https://api.bybit.com/v5/market/kline",
                 params={"category": "spot", "symbol": pair, "interval": "D",
                         "start": t0 * 1000, "end": t1 * 1000, "limit": 1000})
        if not j or j.get("retCode") != 0:
            return None
        return [(int(x[0]) // 1000, x[1], x[2], x[3], x[4], x[5]) for x in j["result"]["list"]]
    return _frame(_paginate_forward(f, 999, now))


def bitget(sym, now):
    pair = sym.upper() + "USDT"

    def f(t0, t1):
        j = _get("https://api.bitget.com/api/v2/spot/market/history-candles",
                 params={"symbol": pair, "granularity": "1Dutc", "endTime": t1 * 1000,
                         "limit": 200})
        if not j or j.get("code") != "00000":
            return None
        return [(int(x[0]) // 1000, x[1], x[2], x[3], x[4], x[5]) for x in j["data"]
                if int(x[0]) // 1000 >= t0]
    return _frame(_paginate_forward(f, 199, now))


def coinbase(sym, now):
    pair = sym.upper() + "-USD"

    def f(t0, t1):
        j = _get("https://api.exchange.coinbase.com/products/%s/candles" % pair,
                 params={"granularity": DAY,
                         "start": pd.Timestamp(t0, unit="s").isoformat(),
                         "end": pd.Timestamp(t1, unit="s").isoformat()},
                 headers={"User-Agent": "numerai-research"})
        if j is None or isinstance(j, dict):
            return None
        return [(int(x[0]), x[3], x[2], x[1], x[4], x[5]) for x in j]
    return _frame(_paginate_forward(f, 299, now))


def mexc(sym, now):
    pair = sym.upper() + "USDT"

    def f(t0, t1):
        j = _get("https://api.mexc.com/api/v3/klines",
                 params={"symbol": pair, "interval": "1d", "startTime": t0 * 1000,
                         "endTime": t1 * 1000, "limit": 1000})
        if j is None or isinstance(j, dict):
            return None
        return [(int(x[0]) // 1000, x[1], x[2], x[3], x[4], x[5]) for x in j]
    return _frame(_paginate_forward(f, 999, now))


_HL_SPOT = {}


def hyperliquid(sym, now):
    if not _HL_SPOT:
        j = _get("https://api.hyperliquid.xyz/info", "POST", json={"type": "spotMeta"})
        if j:
            tok = {t["index"]: t["name"].upper() for t in j["tokens"]}
            for u in j["universe"]:
                base, quote = tok[u["tokens"][0]], tok[u["tokens"][1]]
                if quote in ("USDC", "USDT0", "USDH") and base not in _HL_SPOT:
                    _HL_SPOT[base] = u["name"]
    coin = _HL_SPOT.get(sym.upper())
    if coin is None:
        return None
    j = _get("https://api.hyperliquid.xyz/info", "POST",
             json={"type": "candleSnapshot",
                   "req": {"coin": coin, "interval": "1d",
                           "startTime": int(START.timestamp()) * 1000,
                           "endTime": int(now.timestamp()) * 1000}})
    if not j:
        return None
    return _frame([(int(x["t"]) // 1000, x["o"], x["h"], x["l"], x["c"], x["v"]) for x in j])


VENUES = {"gate": gate, "kucoin": kucoin, "okx": okx, "bybit": bybit,
          "bitget": bitget, "coinbase": coinbase, "hyperliquid": hyperliquid,
          "mexc": mexc}
TICKER_FMT = {"gate": "{}_USDT", "kucoin": "{}-USDT", "okx": "{}-USDT",
              "bybit": "{}USDT", "bitget": "{}USDT", "coinbase": "{}-USD",
              "hyperliquid": "{}/USDC", "mexc": "{}USDT"}


# (symbol, venue) -> first bar to trust. Gate's HYPE before 2024-12-23 is another
# asset (0.15 -> 28.69 overnight; KuCoin and Hyperliquid show 27-32 on those days).
BAD_BARS_BEFORE = {("HYPE", "gate"): pd.Timestamp("2024-12-23")}


def fetch_symbol(sym, now):
    """Every venue's series for one symbol, cached as one json per symbol."""
    path = os.path.join(CACHE, "%s.json" % sym.encode("unicode_escape").decode())
    if os.path.exists(path):
        with open(path) as fh:
            raw = json.load(fh)
        return {v: (pd.DataFrame(r).assign(date=lambda d: pd.to_datetime(d["date"]))
                    if r else None) for v, r in raw.items()}
    out = {}
    for name, fn in VENUES.items():
        try:
            out[name] = fn(sym, now)
        except Exception as e:      # one venue failing must not lose the others
            print("  %s/%s: %s" % (sym, name, e), file=sys.stderr)
            out[name] = None
    with open(path, "w") as fh:
        json.dump({v: (d.assign(date=d["date"].dt.strftime("%Y-%m-%d")).to_dict("list")
                       if d is not None else None) for v, d in out.items()}, fh)
    return out


MIN_CHECK = 30


def market_refs(spot):
    """Cross-sectional median 20d log return and 20d realised vol per date."""
    w = np.log(spot.pivot_table(index="date", columns="symbol", values="close"))
    vol = w.diff().rolling(20, min_periods=15).std()
    return (w - w.shift(20)).median(axis=1), vol.median(axis=1)


def identity_score(close, feat, refs, relative):
    """(mom_rho, vol_rho, n): time-series Spearman of the rebuilt 20d return / 20d vol
    against the coin's feature_momentum_20d / feature_volatility_20d on its panel
    dates. relative=True nets out the market median (only where refs exist)."""
    lc = np.log(close)
    r, v = lc - lc.shift(20), lc.diff().rolling(20, min_periods=15).std()
    if relative:
        r, v = r - refs[0].reindex(lc.index), v / refs[1].reindex(lc.index)
    x, y = r.reindex(feat.index), v.reindex(feat.index)
    ok = x.notna() & y.notna()
    if ok.sum() < MIN_CHECK:
        return np.nan, np.nan, int(ok.sum())
    rk = lambda s: s[ok].rank()
    return (rk(x).corr(rk(feat["feature_momentum_20d"])),
            rk(y).corr(rk(feat["feature_volatility_20d"])), int(ok.sum()))


def null_thresholds(spot, feats, refs, n, relative, seed=0):
    """Wrong-token null: score each sampled coin's features against ANOTHER coin's
    prices. Returns (95th pct of the mean score, 99th pct of the vol score) and the
    right-token pass rate at those thresholds."""
    rng = np.random.default_rng(seed)
    closes = {s: g.set_index("date")["close"] for s, g in spot.groupby("symbol")}
    pool = sorted(s for s in closes if s in feats and len(feats[s]) >= 120)
    good, wrong = [], []
    for s in rng.choice(pool, size=min(n, len(pool)), replace=False):
        o = rng.choice([q for q in pool if q != s])
        good.append(identity_score(closes[s], feats[s], refs, relative))
        wrong.append(identity_score(closes[o], feats[s], refs, relative))
    g = pd.DataFrame(good, columns=["mom", "vol", "n"]).dropna()
    w = pd.DataFrame(wrong, columns=["mom", "vol", "n"]).dropna()
    thr = (float(w[["mom", "vol"]].mean(axis=1).quantile(.95)), float(w["vol"].quantile(.99)))
    return thr, float(passes(g["mom"], g["vol"], thr).mean())


def passes(mom, vol, thr):
    """Mean of the AVAILABLE scores (a constant feature over a short span gives NaN)."""
    mean = pd.concat([mom, vol], axis=1).mean(axis=1, skipna=True)
    return (mean >= thr[0]) | (vol >= thr[1])


def load_context():
    """spot, panel (identity-check columns) and live, dates parsed, ucid as str."""
    spot = pd.read_parquet(OHLCV)
    spot["ucid"] = spot["ucid"].astype(str)
    panel = pd.read_parquet(PANEL, columns=["ucid", "date", "feature_momentum_20d",
                                            "feature_volatility_20d"]).reset_index()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ucid"] = panel["ucid"].astype(str)
    live = pd.read_parquet(LIVE).reset_index()
    live["ucid"] = live["ucid"].astype(str)
    return spot, panel, live


def calibrate(spot, panel, n):
    """Market refs + per-ucid panel features + wrong-token-null thresholds for the
    market-relative (True) and raw (False) checks."""
    feat = {u: g.set_index("date") for u, g in panel.groupby("ucid")}
    refs = market_refs(spot)
    by_sym = {s: feat[u] for s, u in spot.groupby("symbol")["ucid"].first().items()
              if u in feat}
    thr = {}
    for rel in (True, False):
        thr[rel], rate = null_thresholds(spot, by_sym, refs, n, rel)
        print("identity check (%s): pass if mean >= %.3f or vol >= %.3f "
              "[wrong-token null; right-token pass rate %.1f%%]"
              % ("market-relative" if rel else "raw", thr[rel][0], thr[rel][1], 100 * rate))
    return feat, refs, thr


def decide(cands, got, feat, refs, thr, now):
    """cands: [(symbol, ucid)]; got: {(symbol, ucid) or symbol: {venue: frame}}.
    Picks each coin's longest in-span venue series and runs the identity check
    against THAT ucid's panel rows. A coin too new to check is accepted unverified
    only if the series covers its panel dates; a series that misses the panel span
    is rejected. When several ucids of one ticker pass, only the best-checked is
    kept (one venue series cannot be two coins). Returns (report frame, accepted
    frames tagged symbol/ucid/source)."""
    rows, accepted = [], []
    for s, u in cands:
        span = feat.get(u)
        lo = span.index.min() if span is not None else now - pd.Timedelta(days=30)
        best = None
        for v, d in (got.get((s, u)) or got.get(s) or {}).items():
            if d is None or not len(d):
                continue
            if (s, v) in BAD_BARS_BEFORE:
                d = d[d["date"] >= BAD_BARS_BEFORE[(s, v)]]
            n_in = int((d["date"] >= lo).sum())
            if best is None or n_in > best[2]:
                best = (v, d, n_in)
        if best is None or best[2] == 0:
            rows.append(dict(symbol=s, ucid=u, venue=None, status="no venue"))
            continue
        v, d, _ = best
        c = d.set_index("date")["close"]
        mom = vol = np.nan
        n, check = 0, None
        if span is not None:
            for rel in (True, False):        # market-relative where refs reach, else raw
                mom, vol, n = identity_score(c, span, refs, rel)
                if n >= MIN_CHECK:
                    check = rel
                    break
        covered = 0 if span is None else int(((span.index >= d["date"].min())
                                              & (span.index <= d["date"].max())).sum())
        if span is not None and covered < min(MIN_CHECK, len(span)):
            status = "REJECTED (series misses panel span)"
        elif check is None:
            status = "accepted (unverified: <%d panel dates)" % MIN_CHECK
        elif passes(pd.Series([mom]), pd.Series([vol]), thr[check]).iloc[0]:
            status = "accepted"
        else:
            status = "REJECTED (identity)"
        rows.append(dict(symbol=s, ucid=u, venue=v,
                         first=d["date"].min().date(), last=d["date"].max().date(),
                         bars=len(d),
                         numerai_first=span.index.min().date() if span is not None else None,
                         numerai_last=span.index.max().date() if span is not None else None,
                         mom=round(mom, 2), vol=round(vol, 2), n_check=n,
                         check=None if check is None else ("relative" if check else "raw"),
                         status=status))
        if status.startswith("accepted"):
            accepted.append(d.assign(symbol=s, ucid=u, source=v,
                                     provider_ticker=TICKER_FMT[v].format(s.upper())))
    rep = pd.DataFrame(rows)
    ok = rep["status"].str.startswith("accepted")
    best = (rep[ok].assign(_n=rep["n_check"].fillna(0)).sort_values("_n", ascending=False)
            .drop_duplicates("symbol"))
    lose = ok & ~rep.index.isin(best.index)
    rep.loc[lose, "status"] = "REJECTED (ticker taken by a better-checked ucid)"
    keep = set(zip(best["symbol"], best["ucid"]))
    accepted = [a for a in accepted if (a["symbol"].iloc[0], a["ucid"].iloc[0]) in keep]
    return rep, accepted


def write_spot(spot, accepted, backup_tag):
    """Replace each accepted coin's rows (keyed by symbol AND ucid, so a reused
    ticker's other coin is kept) and append; backs the file up once per tag. A new
    bar on a (symbol, date) another ucid already holds is dropped, not overwritten."""
    backup = OHLCV.replace(".parquet", "_pre_%s.parquet" % backup_tag)
    if not os.path.exists(backup):
        shutil.copy(OHLCV, backup)
    new = pd.concat(accepted)[spot.columns]
    key = spot["symbol"] + "|" + spot["ucid"]
    keep = spot[~key.isin(set(new["symbol"] + "|" + new["ucid"]))]
    clash = new.set_index(["symbol", "date"]).index.isin(
        keep.set_index(["symbol", "date"]).index)
    if clash.any():
        print("dropping %d new bars on (symbol, date)s another ucid holds" % clash.sum())
    out = pd.concat([keep, new[~clash]]).sort_values(["symbol", "date"]).reset_index(drop=True)
    assert not out.duplicated(["symbol", "date"]).any()
    out.to_parquet(OHLCV, index=False)
    print("wrote %s: %d rows (+%d), %d symbols (backup %s)"
          % (OHLCV, len(out), len(out) - len(spot), out["symbol"].nunique(), backup))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbols", nargs="*", default=None,
                   help="default: live-universe coins absent from ohlcv.parquet")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--calib", type=int, default=150,
                   help="known-good matched coins used to calibrate the identity threshold")
    p.add_argument("--write", action="store_true",
                   help="append accepted series to ohlcv.parquet (backs it up first)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(CACHE, exist_ok=True)
    now = pd.Timestamp.utcnow().tz_localize(None).normalize()
    spot, panel, live = load_context()
    # live ucid first: these are CURRENT listings, and a ticker can be reused
    ucid = pd.concat([live.groupby("symbol")["ucid"].first(),
                      panel.groupby("symbol")["ucid"].last()])
    ucid = ucid[~ucid.index.duplicated()]
    syms = args.symbols or sorted(set(live["symbol"]) - set(spot["symbol"]))
    print("%d symbols to fetch from %d venues" % (len(syms), len(VENUES)))

    with ThreadPoolExecutor(args.workers) as ex:
        got = dict(zip(syms, ex.map(lambda s: fetch_symbol(s, now), syms)))

    feat, refs, thr = calibrate(spot, panel, args.calib)
    rep, accepted = decide([(s, ucid.get(s)) for s in syms], got, feat, refs, thr, now)
    rep.to_csv(os.path.join(SPOT_DIR, "exchange_fetch_report.csv"), index=False)
    pd.set_option("display.width", 200)
    print(rep.to_string(index=False))
    print("\naccepted %d / %d symbols" % (len(accepted), len(syms)))
    if args.write and accepted:
        write_spot(spot, accepted, "exchange")


if __name__ == "__main__":
    main()
