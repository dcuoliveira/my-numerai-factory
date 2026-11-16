"""
Fill spot-price gaps for coins Binance has DELISTED, from Binance's public data
archive (data.binance.vision: free, documented, keeps the daily klines of pairs that
no longer trade). The live API the original cascade used only knows pairs trading
today, so a coin delisted in 2022 came back empty even though Binance has its
2020-2022 prices.

    python3 -m src.data_mani.fetch_binance_archive            # probe + fetch + check
    python3 -m src.data_mani.fetch_binance_archive --write    # ... and append

Run it on a host that reaches data.binance.vision (lovelace does; deepfive does not).

Candidates are every panel coin (by ucid) with no rows in ohlcv.parquet. The S3
listing host is not reachable from the lab, so instead of listing the archive each
candidate is PROBED: <SYM>USDT and <SYM>BUSD monthly 1d files are HEAD-requested at
the first, middle and last month of the coin's panel span. A pair that answers is
downloaded for its panel span plus 13 months of lookback (ret_12_1 needs 360 days).
BTC/ETH-quoted pairs are not used -- their closes are not USD prices.

The fetched series then go through fetch_exchange_spot's identity check (the rebuilt
20d return / vol against the coin's own shipped Numerai features, thresholds from a
wrong-token null) and only accepted ones are written. Only coins that were ever on
Binance can be recovered here.
"""

import argparse
import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from src.data_mani import fetch_exchange_spot as fx

BASE = "https://data.binance.vision/data/spot/monthly/klines/{p}/1d/{p}-1d-{m}.zip"
CACHE = os.path.join(fx.SPOT_DIR, "_binance_archive")
QUOTES = ("USDT", "BUSD")
FIRST_MONTH = pd.Period("2017-07", "M")      # the archive starts 2017-08
LOOKBACK_MONTHS = 13


def _url(pair, month):
    return BASE.format(p=pair, m=str(month))


def _exists(pair, month):
    for _ in range(3):
        try:
            return requests.head(_url(pair, month), timeout=20).status_code == 200
        except requests.RequestException:
            continue
    return False


def probe(sym, first, last):
    """The first quote whose monthly file exists at the start, middle or end of the
    coin's panel span, or None."""
    lo, hi = pd.Period(first, "M"), pd.Period(last, "M")
    months = sorted({lo, lo + (hi - lo).n // 2, hi})
    for q in QUOTES:
        pair = sym.upper() + q
        if any(_exists(pair, m) for m in months):
            return pair
    return None


def _read_month(pair, month):
    path = os.path.join(CACHE, pair, "%s.csv" % month)
    if os.path.exists(path):
        return pd.read_csv(path) if os.path.getsize(path) else None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = None
    for _ in range(3):
        try:
            r = requests.get(_url(pair, month), timeout=60)
            break
        except requests.RequestException:
            continue
    if r is None or r.status_code != 200:
        open(path, "w").close()          # remember the miss
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = pd.read_csv(z.open(z.namelist()[0]), header=None)
    if not str(raw.iloc[0, 0]).isdigit():    # some files carry a header row
        raw = raw.iloc[1:]
    ts = pd.to_numeric(raw[0])
    ts = ts.where(ts < 1e14, ts // 1000)     # 2025+ spot files are in microseconds
    df = pd.DataFrame({"ts": (ts // 1000).astype("int64"), "open": raw[1], "high": raw[2],
                       "low": raw[3], "close": raw[4], "volume": raw[5]})
    df.to_csv(path, index=False)
    return df


def fetch_pair(pair, first, last):
    lo = max(pd.Period(first, "M") - LOOKBACK_MONTHS, FIRST_MONTH + 1)
    hi = pd.Period(last, "M") + 1
    parts = [_read_month(pair, m) for m in pd.period_range(lo, hi, freq="M")]
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return None
    return fx._frame(pd.concat(parts).itertuples(index=False, name=None))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--calib", type=int, default=150)
    p.add_argument("--write", action="store_true",
                   help="append accepted series to ohlcv.parquet (backs it up first)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(CACHE, exist_ok=True)
    now = pd.Timestamp.utcnow().tz_localize(None).normalize()
    spot, panel, _ = fx.load_context()
    span = panel.groupby("ucid").agg(symbol=("symbol", "last"), first=("date", "min"),
                                     last=("date", "max"))
    span = span[~span.index.isin(set(spot["ucid"]))]
    print("%d panel coins (by ucid) have no spot rows; probing the archive ..." % len(span))

    with ThreadPoolExecutor(args.workers) as ex:
        pairs = list(ex.map(lambda r: probe(r.symbol, r.first, r.last),
                            span.itertuples()))
    span["pair"] = pairs
    hit = span[span["pair"].notna()]
    print("on Binance during their panel span: %d of %d (%s)"
          % (len(hit), len(span), hit["pair"].str[-4:].value_counts().to_dict()))

    # one download per pair over the union of its ucids' spans (no cache races)
    win = hit.groupby("pair").agg(first=("first", "min"), last=("last", "max"))
    with ThreadPoolExecutor(args.workers) as ex:
        frames = dict(zip(win.index, ex.map(lambda r: fetch_pair(r.Index, r.first, r.last),
                                            win.itertuples())))
    got = {}
    for u, r in hit.iterrows():
        venue = "binance_archive_" + r.pair[-4:].lower()
        fx.TICKER_FMT.setdefault(venue, "{}" + r.pair[-4:])
        got[(r.symbol, u)] = {venue: frames[r.pair]}

    feat, refs, thr = fx.calibrate(spot, panel, args.calib)
    rep, accepted = fx.decide([(r.symbol, u) for u, r in hit.iterrows()], got,
                              feat, refs, thr, now)
    # keep only the coin's own window: panel span + lookback (a delisted ticker can
    # be reused by a later pair, which is not this ucid's price)
    accepted = [a[(a["date"] >= hit.at[a["ucid"].iloc[0], "first"]
                   - pd.DateOffset(months=LOOKBACK_MONTHS))
                  & (a["date"] <= hit.at[a["ucid"].iloc[0], "last"] + pd.DateOffset(months=1))]
                for a in accepted]
    # a ticker with a series keeps it: appending archive bars would stitch two
    # sources that can be different coins (ONE stepped -31% at the join)
    fresh = [a for a in accepted if a["symbol"].iloc[0] not in set(spot["symbol"])]
    rep["written"] = rep["symbol"].isin({a["symbol"].iloc[0] for a in fresh}) \
        & rep["ucid"].isin({a["ucid"].iloc[0] for a in fresh})
    accepted = fresh
    rep.to_csv(os.path.join(fx.SPOT_DIR, "binance_archive_report.csv"), index=False)
    pd.set_option("display.width", 220)
    print(rep.to_string(index=False))
    print("\nprobed %d | on Binance %d | accepted %d" % (len(span), len(hit), len(accepted)))
    if args.write and accepted:
        fx.write_spot(spot, accepted, "binance_archive")


if __name__ == "__main__":
    main()
