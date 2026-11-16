"""
E151 volume-alpha -- raw Binance venue data from the public archive (data.binance.vision).

    python3 -m src.data_mani.fetch_volalpha_binance list                 # S3 listing of every pair (cached)
    python3 -m src.data_mani.fetch_volalpha_binance klines --kind spot   # daily klines, every panel pair
    python3 -m src.data_mani.fetch_volalpha_binance klines --kind um     # USD-M perpetual daily klines
    python3 -m src.data_mani.fetch_volalpha_binance metrics              # USD-M daily metrics (open interest)
    python3 -m src.data_mani.fetch_volalpha_binance funding              # USD-M funding prints -> daily
    python3 -m src.data_mani.fetch_volalpha_binance consolidate          # one parquet per kind

Why the archive: it keeps DELISTED pairs (LUNA, FTT, SRM), which the live REST API forgets, so a series
built from it is not survivorship-filtered the way data/crypto/perp (417 still-listed perps) is.

Pair matching is by ticker: a pair is BASE+QUOTE with QUOTE in QUOTES and BASE a panel symbol, optionally
behind a 1000/1000000/1M multiplier prefix (perps). Ticker reuse is real (LUNA), so this stage keeps every
match with its pair name and the builder decides identity per (ucid, date) against the coin's own shipped
features. Nothing is dropped here.

Every stored row carries `src_file` (the archive object it came from) and `retrieved_at` (UTC download time):
the archive is a retrospective publication, so these are timestamp-causal, not vintage point-in-time.

Formats (verified 2026-11-12): spot timestamps switch from ms to MICROseconds on 2025-01-01; USD-M klines are
ms and carry a header row; spot files have no header. Metrics are 5-minute snapshots, one file per day.
"""
import argparse
import io
import json
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

ROOT = "/scratch/danielco/volalpha/binance"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DATA = "https://data.binance.vision/"
QUOTES = ("USDT", "BUSD", "USDC", "FDUSD", "TUSD", "USDP", "DAI", "USD", "BTC", "ETH", "BNB")
MULT = {"1000000": 1e6, "1M": 1e6, "1000": 1e3, "": 1.0}
PREFIX = {"spot": "data/spot/monthly/klines/", "um": "data/futures/um/monthly/klines/",
          "um_daily": "data/futures/um/daily/klines/", "spot_daily": "data/spot/daily/klines/",
          "metrics": "data/futures/um/daily/metrics/", "funding": "data/futures/um/monthly/fundingRate/"}
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
         "taker_buy_base", "taker_buy_quote", "ignore"]
FIRST = pd.Timestamp("2018-11-01")       # 252 trading days + slack before the panel's 2020-01-01


def session():
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=3))
    return s


S = session()


def get(url, **kw):
    for i in range(5):
        try:
            r = S.get(url, timeout=60, **kw)
            if r.status_code in (200, 404, 403):
                return r
        except requests.RequestException:
            pass
        time.sleep(1 + 2 * i)
    return None


def s3_list(prefix, delimiter=True):
    """Every CommonPrefix (delimiter) or Key under prefix, paging with marker."""
    out, marker = [], ""
    while True:
        q = {"prefix": prefix, "marker": marker}
        if delimiter:
            q["delimiter"] = "/"
        r = get(S3, params=q)
        txt = r.text
        tag = "Prefix" if delimiter else "Key"
        items = re.findall(r"<%s>([^<]*)</%s>" % (tag, tag), txt)
        items = [i for i in items if i != prefix]
        out += items
        if "<IsTruncated>true" not in txt:
            return out
        nm = re.findall(r"<NextMarker>([^<]*)</NextMarker>", txt)
        marker = nm[0] if nm else items[-1]


def panel_symbols():
    d = pd.read_parquet("data/crypto/v2.0/train.parquet", columns=["ucid", "date"]).reset_index()
    return set(d["symbol"].astype(str).str.upper().str.lstrip("$"))


def split_pair(pair, syms):
    """(base symbol, multiplier, quote) or None."""
    for q in QUOTES:
        if pair.endswith(q) and len(pair) > len(q):
            b = pair[: -len(q)]
            for m in ("1000000", "1M", "1000", ""):
                if b.startswith(m) and b[len(m):] in syms:
                    return b[len(m):], MULT[m], q
    return None


def cmd_list(args):
    os.makedirs(ROOT, exist_ok=True)
    syms = panel_symbols()
    out = {}
    for k in ("spot", "um", "metrics"):
        pairs = [p.rstrip("/").split("/")[-1] for p in s3_list(PREFIX[k])]
        m = {p: split_pair(p, syms) for p in pairs}
        out[k] = {p: v for p, v in m.items() if v}
        print("%-8s %5d pairs in archive, %5d match a panel symbol (%d symbols)"
              % (k, len(pairs), len(out[k]), len({v[0] for v in out[k].values()})))
    json.dump(out, open(os.path.join(ROOT, "pairs.json"), "w"), indent=0)


def pairs(kind):
    return json.load(open(os.path.join(ROOT, "pairs.json")))[kind]


def read_zip(content):
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        return pd.read_csv(z.open(z.namelist()[0]), header=None, dtype=str)


def parse_klines(raw, src):
    if not str(raw.iloc[0, 0]).isdigit():
        raw = raw.iloc[1:]
    raw = raw.iloc[:, :12]
    raw.columns = KCOLS
    df = raw.drop(columns="ignore").apply(pd.to_numeric, errors="coerce")
    ts = df["open_time"].where(df["open_time"] < 1e14, df["open_time"] // 1000)
    df["date"] = pd.to_datetime(ts, unit="ms").dt.normalize()
    df["src_file"] = src
    return df.drop(columns=["open_time", "close_time"])


def fetch_kline_pair(kind, pair):
    path = os.path.join(ROOT, kind, pair + ".parquet")
    if os.path.exists(path):
        return 0
    keys = [k for k in s3_list("%s%s/1d/" % (PREFIX[kind], pair), delimiter=False) if k.endswith(".zip")]
    months = {k for k in keys if pd.Timestamp(k[-11:-4] + "-01") >= FIRST}
    # the current month(s) exist only as daily files
    last = max([pd.Timestamp(k[-11:-4] + "-01") for k in keys], default=FIRST - pd.offsets.MonthBegin())
    dk = PREFIX["um_daily" if kind == "um" else "spot_daily"]
    daily = [k for k in s3_list("%s%s/1d/" % (dk, pair), delimiter=False)
             if k.endswith(".zip") and pd.Timestamp(k[-14:-4]) >= last + pd.offsets.MonthBegin()]
    parts = []
    for k in sorted(months) + sorted(daily):
        r = get(DATA + k)
        if r is None or r.status_code != 200:
            continue
        p = parse_klines(read_zip(r.content), k)
        p["retrieved_at"] = pd.Timestamp.utcnow().tz_localize(None)
        parts.append(p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.concat(parts) if parts else pd.DataFrame(columns=KCOLS[1:6] + ["date"])
    df = df.drop_duplicates("date", keep="last").sort_values("date")
    df.insert(0, "pair", pair)
    df.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return len(df)


def cmd_klines(args):
    ps = sorted(pairs(args.kind))
    print("%d %s pairs" % (len(ps), args.kind), flush=True)
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, n in enumerate(ex.map(lambda p: fetch_kline_pair(args.kind, p), ps)):
            done += n
            if i % 100 == 0:
                print("  %d/%d pairs, %d rows" % (i, len(ps), done), flush=True)


def metrics_day(pair, day):
    k = "%s%s/%s-metrics-%s.zip" % (PREFIX["metrics"], pair, pair, day)
    r = get(DATA + k)
    if r is None or r.status_code != 200:
        return None
    raw = read_zip(r.content)
    raw.columns = raw.iloc[0]
    raw = raw.iloc[1:]
    num = raw.drop(columns=["create_time", "symbol"]).apply(pd.to_numeric, errors="coerce")
    t = pd.to_datetime(raw["create_time"])
    # the day's last snapshot (state at the close) and its mean (average exposure over the day)
    last = num.iloc[int(t.values.argmax())] if len(num) else num.mean()
    row = {"date": pd.Timestamp(day), "n_snap": len(num), "src_file": k}
    for c in num.columns:
        row[c + "_last"] = last[c]
        row[c + "_mean"] = num[c].mean()
    return row


def fetch_metrics_pair(pair):
    path = os.path.join(ROOT, "metrics", pair + ".parquet")
    if os.path.exists(path):
        return 0
    keys = [k for k in s3_list("%s%s/" % (PREFIX["metrics"], pair), delimiter=False) if k.endswith(".zip")]
    days = sorted(k[-14:-4] for k in keys if pd.Timestamp(k[-14:-4]) >= FIRST)
    rows = [metrics_day(pair, d) for d in days]
    df = pd.DataFrame([r for r in rows if r])
    df["retrieved_at"] = pd.Timestamp.utcnow().tz_localize(None)
    df.insert(0, "pair", pair)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return len(df)


def cmd_metrics(args):
    ps = sorted(pairs("metrics"))
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        ps = ps[i::n]
    if args.reverse:          # a second process on another host meets the first in the middle
        ps = ps[::-1]
    print("%d metrics pairs" % len(ps), flush=True)
    tot = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, n in enumerate(ex.map(fetch_metrics_pair, ps)):
            tot += n
            if i % 10 == 0:
                print("  %d/%d pairs, %d pair-days" % (i, len(ps), tot), flush=True)


def fetch_funding_pair(pair):
    """Every funding print of one perp (monthly files), summed and averaged per UTC day."""
    path = os.path.join(ROOT, "funding", pair + ".parquet")
    if os.path.exists(path):
        return 0
    keys = sorted(k for k in s3_list("%s%s/" % (PREFIX["funding"], pair), delimiter=False) if k.endswith(".zip"))
    parts = []
    for k in keys:
        r = get(DATA + k)
        if r is None or r.status_code != 200:
            continue
        raw = read_zip(r.content)
        raw.columns = raw.iloc[0]
        raw = raw.iloc[1:]
        t = pd.to_datetime(pd.to_numeric(raw["calc_time"]), unit="ms")
        parts.append(pd.DataFrame({"date": t.dt.normalize(), "rate": pd.to_numeric(raw["last_funding_rate"]),
                                   "interval_h": pd.to_numeric(raw["funding_interval_hours"]), "src_file": k}))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if parts:
        d = pd.concat(parts)
        df = d.groupby("date").agg(funding_sum=("rate", "sum"), funding_mean=("rate", "mean"), n_prints=("rate", "size"),
                                   src_file=("src_file", "last")).reset_index()
    else:
        df = pd.DataFrame(columns=["date", "funding_sum", "funding_mean", "n_prints", "src_file"])
    df["retrieved_at"] = pd.Timestamp.utcnow().tz_localize(None)
    df.insert(0, "pair", pair)
    df.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return len(df)


def cmd_funding(args):
    ps = sorted(pairs("um"))
    with ThreadPoolExecutor(args.workers) as ex:
        for i, n in enumerate(ex.map(fetch_funding_pair, ps)):
            if i % 100 == 0:
                print("  %d/%d pairs" % (i, len(ps)), flush=True)


def cmd_consolidate(args):
    for kind in ("spot", "um", "metrics", "funding"):
        d = os.path.join(ROOT, kind)
        if not os.path.isdir(d):
            continue
        meta = pairs("um" if kind == "funding" else kind)
        parts = []
        for f in sorted(os.listdir(d)):
            if f.endswith(".parquet"):
                p = pd.read_parquet(os.path.join(d, f))
                if len(p):
                    b, m, q = meta[f[:-8]]
                    p["base"], p["mult"], p["quote"] = b, m, q
                    parts.append(p)
        out = pd.concat(parts, ignore_index=True)
        out.to_parquet(os.path.join(ROOT, kind + ".parquet"), index=False)
        print("%s: %d rows, %d pairs, %d bases, %s..%s" % (kind, len(out), out["pair"].nunique(),
              out["base"].nunique(), out["date"].min().date(), out["date"].max().date()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "klines", "metrics", "funding", "consolidate"])
    ap.add_argument("--kind", default="spot", choices=["spot", "um"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard", default="")
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    {"list": cmd_list, "klines": cmd_klines, "metrics": cmd_metrics, "funding": cmd_funding, "consolidate": cmd_consolidate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
