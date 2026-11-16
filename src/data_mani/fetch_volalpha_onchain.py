"""
E151 volume-alpha -- fundamentals, valuation and governance sources (free, no key).

    python3 -m src.data_mani.fetch_volalpha_onchain coinmetrics   # market cap, supply, MVRV, active addresses, tx
    python3 -m src.data_mani.fetch_volalpha_onchain llama         # DefiLlama TVL / fees / revenue / DEX volume
    python3 -m src.data_mani.fetch_volalpha_onchain snapshot      # governance proposals (created, start, end)
    python3 -m src.data_mani.fetch_volalpha_onchain stables       # USDT/USDC/BUSD/FDUSD/TUSD/DAI vs USD (Yahoo)

Output under /scratch/danielco/volalpha/onchain/. Every file keeps `retrieved_at`.

Identity: DefiLlama carries `cmcId` on protocols and chains, and the panel's `ucid` IS the CoinMarketCap id,
so that join is by permanent id, not ticker. A cmcId sits on ONE child protocol (Aave V2) while the token
belongs to the parent (parent#aave), so a token's series is fetched under its parent's slug when it has one.
Coin Metrics asset ids are lowercase tickers; the builder accepts a CM asset for a ucid only where CM's
PriceUSD tracks that coin's own exchange price (identity check), never on the ticker alone.

REVISIONS. None of these is a vintage (as-first-published) archive. Coin Metrics and DefiLlama recompute
history (new adapters, re-indexed chains), so a 2022 value fetched today may differ from what was visible in
2022. The spec forbids revised observations unless revisions were available at the time; they were not, so
every family built from here is labelled "revised, not point-in-time" and kept out of the raw-volume arms.
Snapshot proposals are immutable once created and carry their creation timestamp, so the governance event
block IS point-in-time: a vote's end date is encoded only from the moment the proposal was created.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

ROOT = "/scratch/danielco/volalpha/onchain"
CM = "https://community-api.coinmetrics.io/v4"
CM_METRICS = ["PriceUSD", "CapMrktCurUSD", "CapMrktEstUSD", "CapMrktFFUSD", "SplyCur", "SplyFF", "SplyExpFut10yr",
              "CapRealUSD", "CapMVRVCur", "CapMVRVFF", "AdrActCnt", "TxCnt", "TxTfrCnt", "FeeTotNtv", "FeeTotUSD"]
LLAMA = "https://api.llama.fi"
START = "2018-11-01"


def get(url, **kw):
    for i in range(5):
        try:
            r = requests.get(url, timeout=120, **kw)
            if r.status_code == 429:
                time.sleep(5 + 10 * i)
                continue
            return r
        except requests.RequestException:
            time.sleep(2 + 3 * i)
    return None


def now():
    return pd.Timestamp.utcnow().tz_localize(None)


# ---------------------------------------------------------------- Coin Metrics
def cm_assets():
    out = set()
    # PriceUSD alone lists ~1,200 assets, almost all paywalled; the on-chain metrics list the ~140 free ones
    for m in ("CapMrktCurUSD", "AdrActCnt", "TxCnt", "CapMVRVCur", "SplyCur"):
        r = get(CM + "/catalog-v2/asset-metrics", params={"metrics": m, "page_size": 10000})
        out |= {a["asset"] for a in r.json().get("data", [])}
    return sorted(out)


def cm_free_metrics(asset):
    ok = []
    for m in CM_METRICS:
        r = get(CM + "/timeseries/asset-metrics", params={"assets": asset, "metrics": m, "start_time": "2024-01-01",
                                                           "end_time": "2024-01-03", "frequency": "1d"})
        if r is not None and r.status_code == 200 and r.json().get("data"):
            ok.append(m)
    return ok


def cm_series(asset):
    ms = cm_free_metrics(asset)
    if not ms:
        return None
    rows, url, p = [], CM + "/timeseries/asset-metrics", {"assets": asset, "metrics": ",".join(ms),
                                                          "start_time": START, "frequency": "1d", "page_size": 10000}
    while url:
        r = get(url, params=p)
        j = r.json()
        rows += j.get("data", [])
        url, p = j.get("next_page_url"), None
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    for m in ms:
        if m in df:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    df["retrieved_at"] = now()
    return df.drop(columns=["time"])


def cmd_coinmetrics(a):
    os.makedirs(ROOT, exist_ok=True)
    assets = cm_assets()
    print("%d Coin Metrics assets in the community catalog" % len(assets), flush=True)
    with ThreadPoolExecutor(4) as ex:
        parts = [p for p in ex.map(cm_series, assets) if p is not None]
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(os.path.join(ROOT, "coinmetrics.parquet"), index=False)
    print("coinmetrics: %d rows, %d assets; non-null per metric:\n%s" % (
        len(df), df["asset"].nunique(), df.groupby("asset").first().notna().sum().to_string()))


# ---------------------------------------------------------------- DefiLlama
def llama_tokens():
    """[(cmcId, slug, kind)] -- the series a token's fundamentals come from."""
    protos = get(LLAMA + "/protocols").json()
    chains = get(LLAMA + "/v2/chains").json()
    json.dump(protos, open(os.path.join(ROOT, "llama_protocols.json"), "w"))
    json.dump(chains, open(os.path.join(ROOT, "llama_chains.json"), "w"))
    out = {}
    for p in protos:
        if p.get("cmcId"):
            slug = p["parentProtocol"].split("#", 1)[1] if p.get("parentProtocol") else p["slug"]
            out.setdefault(str(p["cmcId"]), ("protocol", slug, p.get("address"), p.get("governanceID")))
    for c in chains:
        if c.get("cmcId"):
            out.setdefault(str(c["cmcId"]), ("chain", c["name"], None, None))
    return out


def _chart(j, key):
    """DefiLlama [[ts, value], ...] -> Series by date."""
    v = j.get(key) or []
    if not v:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(int(t), unit="s").normalize(): float(x) for t, x in v})
    return s[~s.index.duplicated(keep="last")]


def llama_one(item):
    cmc, (kind, slug, addr, gov) = item
    cols = {}
    if kind == "protocol":
        r = get(LLAMA + "/protocol/" + slug)
        if r is not None and r.ok:
            tvl = r.json().get("tvl") or []
            cols["tvl"] = pd.Series({pd.Timestamp(int(x["date"]), unit="s").normalize(): x["totalLiquidityUSD"]
                                     for x in tvl}, dtype=float)
        base = "/summary/fees/" + slug
        for dt, name in (("dailyFees", "fees"), ("dailyRevenue", "revenue"), ("dailyHoldersRevenue", "holders_rev")):
            r = get(LLAMA + base, params={"dataType": dt})
            if r is not None and r.ok:
                cols[name] = _chart(r.json(), "totalDataChart")
        r = get(LLAMA + "/summary/dexs/" + slug)
        if r is not None and r.ok:
            cols["dex_volume"] = _chart(r.json(), "totalDataChart")
    else:
        r = get(LLAMA + "/v2/historicalChainTvl/" + slug)
        if r is not None and r.ok:
            cols["tvl"] = pd.Series({pd.Timestamp(int(x["date"]), unit="s").normalize(): x["tvl"] for x in r.json()},
                                    dtype=float)
        for path, name in (("/overview/fees/", "fees"), ("/overview/dexs/", "dex_volume")):
            r = get(LLAMA + path + slug, params={"excludeTotalDataChartBreakdown": "true"})
            if r is not None and r.ok:
                cols[name] = _chart(r.json(), "totalDataChart")
    cols = {k: v for k, v in cols.items() if len(v)}
    if not cols:
        return None
    df = pd.DataFrame(cols).rename_axis("date").reset_index()
    df.insert(0, "cmcId", int(cmc))
    df["kind"], df["slug"], df["address"], df["governance"] = kind, slug, addr, json.dumps(gov) if gov else None
    df["retrieved_at"] = now()
    return df


def cmd_llama(a):
    os.makedirs(ROOT, exist_ok=True)
    toks = llama_tokens()
    panel = set(pd.read_parquet("data/crypto/v2.0/train.parquet", columns=["ucid"])["ucid"].astype(str))
    items = [(k, v) for k, v in toks.items() if k in panel]
    print("%d DefiLlama tokens with a cmcId, %d of them panel ucids" % (len(toks), len(items)), flush=True)
    with ThreadPoolExecutor(4) as ex:
        parts = [p for p in ex.map(llama_one, items) if p is not None]
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(os.path.join(ROOT, "llama.parquet"), index=False)
    print("llama: %d rows, %d ucids; ucids with each series: %s" % (
        len(df), df["cmcId"].nunique(),
        {c: int(df.groupby("cmcId")[c].count().gt(0).sum()) for c in ["tvl", "fees", "revenue", "dex_volume"] if c in df}))


# ---------------------------------------------------------------- Snapshot
def cmd_snapshot(a):
    protos = json.load(open(os.path.join(ROOT, "llama_protocols.json")))
    space_cmc = {}
    for p in protos:
        for g in p.get("governanceID") or []:
            if g.startswith("snapshot:"):
                cmc = p.get("cmcId")
                if not cmc and p.get("parentProtocol"):
                    cmc = next((q.get("cmcId") for q in protos
                                if q.get("parentProtocol") == p["parentProtocol"] and q.get("cmcId")), None)
                if cmc:
                    space_cmc[g.split(":", 1)[1]] = int(cmc)
    print("%d Snapshot spaces with a token cmcId" % len(space_cmc), flush=True)
    rows = []
    q = """{proposals(first:1000, skip:%d, where:{space_in:%s, created_gte:%d}, orderBy:"created", orderDirection:asc)
           {id space{id} created start end state type scores_total votes}}"""
    sp = list(space_cmc)
    for i in range(0, len(sp), 20):
        chunk, skip = json.dumps(sp[i:i + 20]), 0
        while True:
            r = requests.post("https://hub.snapshot.org/graphql",
                              json={"query": q % (skip, chunk, int(pd.Timestamp(START).timestamp()))}, timeout=120)
            if r.status_code == 429:
                time.sleep(20)
                continue
            got = (r.json().get("data") or {}).get("proposals") or []
            rows += got
            if len(got) < 1000 or skip >= 4000:
                break
            skip += 1000
        time.sleep(1)
    df = pd.DataFrame(rows)
    df["space"] = df["space"].map(lambda s: s["id"])
    df["cmcId"] = df["space"].map(space_cmc)
    for c in ("created", "start", "end"):
        df[c] = pd.to_datetime(df[c], unit="s")
    df["retrieved_at"] = now()
    df.to_parquet(os.path.join(ROOT, "snapshot.parquet"), index=False)
    print("snapshot: %d proposals, %d spaces, %d ucids" % (len(df), df["space"].nunique(), df["cmcId"].nunique()))


# ---------------------------------------------------------------- stablecoin USD rates
def cmd_stables(a):
    import yfinance as yf
    out = []
    for t in ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USDP"):
        h = yf.Ticker(t + "-USD").history(start=START, auto_adjust=True)
        if len(h):
            out.append(pd.DataFrame({"date": h.index.tz_localize(None).normalize(), "stable": t, "usd": h["Close"].values}))
    df = pd.concat(out, ignore_index=True)
    df["retrieved_at"] = now()
    df.to_parquet(os.path.join(ROOT, "stables.parquet"), index=False)
    print(df.groupby("stable")["date"].agg(["min", "max", "size"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["coinmetrics", "llama", "snapshot", "stables"])
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    os.makedirs(ROOT, exist_ok=True)
    {"coinmetrics": cmd_coinmetrics, "llama": cmd_llama, "snapshot": cmd_snapshot, "stables": cmd_stables}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
