"""
E151 volume-alpha -- per-venue daily spot klines from the non-Binance CEX venues (no-key public APIs).

    python3 -m src.data_mani.fetch_volalpha_cex fetch [--workers 6]     # every panel symbol x every venue
    python3 -m src.data_mani.fetch_volalpha_cex consolidate             # -> /scratch/.../cex.parquet

Venues and their request code are fetch_exchange_spot.VENUES (gate, kucoin, okx, bybit, bitget, coinbase,
mexc, hyperliquid). Binance comes from the archive instead (fetch_volalpha_binance), so it is not here.
Every venue's volume column is BASE-asset volume; dollar volume is formed downstream as base volume x the
contemporaneous USD price, never from a venue's own quote-volume field.

SURVIVORSHIP: these REST endpoints only answer for pairs listed TODAY. A coin a venue delisted in 2022 has
no rows here even though it traded there. So this family is "currently-listed venues, history as published
now" -- timestamp-causal but not point-in-time, and the venue count built from it undercounts the past.
That is why it is kept as its own source family (and its own ablation) apart from the Binance archive.
Ticker matching only: identity is decided per (ucid, date) by the builder, not here.
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from src.data_mani import fetch_exchange_spot as fx

ROOT = "/scratch/danielco/volalpha/cex"
fx.START = pd.Timestamp("2018-11-01")


def fetch(sym, now):
    path = os.path.join(ROOT, "%s.parquet" % sym.encode("unicode_escape").decode().replace("/", "_"))
    if os.path.exists(path):
        return 0
    parts = []
    for name, fn in fx.VENUES.items():
        try:
            d = fn(sym, now)
        except Exception as e:  # one venue failing must not lose the others
            print("  %s/%s: %s" % (sym, name, e), file=sys.stderr)
            d = None
        if d is not None and len(d):
            parts.append(d.assign(venue=name, pair=fx.TICKER_FMT[name].format(sym.upper())))
    df = pd.concat(parts) if parts else pd.DataFrame(columns=fx.COLS + ["venue", "pair"])
    df["symbol"] = sym
    df["retrieved_at"] = pd.Timestamp.utcnow().tz_localize(None)
    df.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return len(df)


def cmd_fetch(a):
    os.makedirs(ROOT, exist_ok=True)
    panel = pd.read_parquet("data/crypto/v2.0/train.parquet", columns=["ucid"]).reset_index()
    syms = sorted(set(panel["symbol"].astype(str).str.lstrip("$")))
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        syms = syms[i::n]
    now = pd.Timestamp.utcnow().tz_localize(None).normalize()
    print("%d symbols x %d venues" % (len(syms), len(fx.VENUES)), flush=True)
    tot = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, n in enumerate(ex.map(lambda s: fetch(s, now), syms)):
            tot += n
            if i % 50 == 0:
                print("  %d/%d symbols, %d rows" % (i, len(syms), tot), flush=True)


def cmd_consolidate(a):
    parts = [pd.read_parquet(os.path.join(ROOT, f)) for f in sorted(os.listdir(ROOT)) if f.endswith(".parquet")]
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    for (s, v), t in fx.BAD_BARS_BEFORE.items():
        out = out[~((out["symbol"] == s) & (out["venue"] == v) & (out["date"] < t))]
    out.to_parquet(os.path.join(os.path.dirname(ROOT), "cex.parquet"), index=False)
    print(out.groupby("venue").agg(rows=("date", "size"), symbols=("symbol", "nunique"),
                                   first=("date", "min")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "consolidate"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--shard", default="")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    {"fetch": cmd_fetch, "consolidate": cmd_consolidate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
