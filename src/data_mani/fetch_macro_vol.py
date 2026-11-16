"""
Fetch the macro series behind crypto/v2.0-vol-causal-macrovol, keeping every raw download.

    python3 -m src.data_mani.fetch_macro_vol

Series (one value per trading day, the source's own calendar):

    gold    GLD close, Yahoo        spot-gold ETF; no futures roll, so log returns have no roll jumps
    sp500   ^GSPC close, Yahoo      S&P 500 index
    dxy     DX-Y.NYB close, Yahoo   ICE US Dollar Index
    vix     ^VIX close, Yahoo       CBOE VIX level (implied volatility, percent)
    y10     10-year par yield, US Treasury daily par yield curve (percent)

Every run writes the raw responses to data/crypto/macro/raw/<retrieved UTC>/ and a
manifest with the URL or ticker, retrieval time, library version and sha256 of each file.
That directory is append-only: a run never touches an earlier one. The combined store
data/crypto/macro/macro_vol_daily.parquet is rebuilt from the newest run and carries the
run id per row, so a value can always be traced to the download it came from.

What this does NOT prove: that a value was available on its date. Yahoo and Treasury
serve today's version of the history; only the dated raw runs, compared over time, can
show what was published when.
"""

import hashlib
import io
import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests

ROOT = os.path.join("data", "crypto", "macro")
RAW = os.path.join(ROOT, "raw")
STORE = os.path.join(ROOT, "macro_vol_daily.parquet")
START = "2018-01-01"
YAHOO = {"gold": "GLD", "sp500": "^GSPC", "dxy": "DX-Y.NYB", "vix": "^VIX"}
TREASURY = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
            "{y}/all?type=daily_treasury_yield_curve&field_tdr_date_value={y}&page&_format=csv")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def fetch_yahoo(ticker):
    import yfinance as yf
    d = yf.download(ticker, start=START, progress=False, auto_adjust=False, actions=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.reset_index().rename(columns={"Date": "date"})
    return d, d.to_csv(index=False).encode(), "yfinance %s" % yf.__version__


def fetch_treasury(year):
    """IPv4 curl first: requests on these hosts sometimes picks an unreachable IPv6 route."""
    import subprocess
    import time
    url = TREASURY.format(y=year)
    for attempt in range(4):
        r = subprocess.run(["curl", "-4", "-sfL", "--max-time", "60", url], capture_output=True)
        if r.returncode == 0 and r.stdout.startswith(b"Date"):
            return r.stdout
        try:
            q = requests.get(url, timeout=60)
            if q.ok and q.content.startswith(b"Date"):
                return q.content
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    raise SystemExit("Treasury %d: no response after 4 attempts" % year)


def main():
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final = os.path.join(RAW, run)
    out = final + ".partial"            # renamed only once every file is in, so a dated run is always complete
    os.makedirs(out, exist_ok=False)
    manifest, frames = {"run": run, "files": {}}, []
    for name, ticker in YAHOO.items():
        d, raw, lib = fetch_yahoo(ticker)
        path = os.path.join(out, "%s.csv" % name)
        open(path, "wb").write(raw)
        manifest["files"][name] = dict(source="yahoo", id=ticker, library=lib, field="Close", sha256=sha256_bytes(raw),
                                       rows=len(d), first=str(d["date"].min().date()), last=str(d["date"].max().date()))
        frames.append(pd.DataFrame({"date": pd.to_datetime(d["date"]), "series": name, "value": d["Close"].astype(float),
                                    "source": "yahoo", "source_id": ticker}))
        print("%-6s %-9s %d rows %s .. %s" % (name, ticker, len(d), d["date"].min().date(), d["date"].max().date()))
    parts = []
    for y in range(int(START[:4]), datetime.now(timezone.utc).year + 1):
        raw = fetch_treasury(y)
        path = os.path.join(out, "treasury_par_%d.csv" % y)
        open(path, "wb").write(raw)
        manifest["files"]["treasury_par_%d" % y] = dict(source="us_treasury", id="daily_treasury_yield_curve", url=TREASURY.format(y=y),
                                                       field="10 Yr", sha256=sha256_bytes(raw))
        parts.append(pd.read_csv(io.BytesIO(raw)))
    t = pd.concat(parts, ignore_index=True)
    t = pd.DataFrame({"date": pd.to_datetime(t["Date"], format="%m/%d/%Y"), "series": "y10", "value": pd.to_numeric(t["10 Yr"], errors="coerce"),
                      "source": "us_treasury", "source_id": "daily_treasury_yield_curve:10 Yr"})
    frames.append(t)
    print("y10    treasury  %d rows %s .. %s" % (len(t), t["date"].min().date(), t["date"].max().date()))
    store = pd.concat(frames, ignore_index=True).dropna(subset=["value"]).drop_duplicates(["series", "date"], keep="last")
    store = store.sort_values(["series", "date"]).assign(run=run).reset_index(drop=True)
    manifest["retrieved_utc"] = run
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    os.rename(out, final)
    store.to_parquet(STORE, index=False)
    print("store: %d rows -> %s (run %s)" % (len(store), STORE, run))


if __name__ == "__main__":
    main()
