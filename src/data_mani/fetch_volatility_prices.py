"""Cache UTC CoinMarketCap website candles by the panel's permanent UCID."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from threading import Event
from time import sleep

import pandas as pd
import requests

URL = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/historical"
CUTOFF = pd.Timestamp("2026-02-19")


def fetch(cache, task, cache_only, rate_limited):
    ucid, start, end = task
    path = cache / f"{ucid}_{start:%Y%m%d}_{end:%Y%m%d}.json"
    if not path.exists():
        if cache_only:
            return [], {"ucid": ucid, "start": str(start.date()),
                        "end": str(end.date()), "error": "uncached"}
        if rate_limited.is_set():
            return [], {"ucid": ucid, "start": str(start.date()),
                        "end": str(end.date()), "error": "rate limited"}
        params = {
            "id": ucid, "convertId": 2781, "interval": "1d",
            "timeStart": int(start.tz_localize("UTC").timestamp()),
            "timeEnd": int(end.tz_localize("UTC").timestamp()),
        }
        for attempt in range(4):
            try:
                response = requests.get(URL, params=params, timeout=20)
                if response.status_code == 429:
                    rate_limited.set()
                    return [], {"ucid": ucid, "start": str(start.date()),
                                "end": str(end.date()), "error": "HTTP 429"}
                response.raise_for_status()
                payload = response.json()
                if str(payload["status"]["error_code"]) != "0":
                    raise ValueError(payload["status"]["error_code"])
                entry = {
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "url": response.url,
                    "sha256": hashlib.sha256(response.content).hexdigest(),
                    "response": response.text,
                }
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(entry))
                temporary.replace(path)
                break
            except (requests.RequestException, ValueError, KeyError) as exc:
                if attempt == 3:
                    return [], {"ucid": ucid, "start": str(start.date()),
                                "end": str(end.date()), "error": str(exc)}
                sleep(2 ** attempt)
    entry = json.loads(path.read_text())
    payload = entry.pop("response")
    if hashlib.sha256(payload.encode()).hexdigest() != entry["sha256"]:
        raise ValueError(f"Corrupt response cache for {path.name}")
    data = json.loads(payload).get("data") or {}
    quotes = data.get("quotes") or []
    if quotes and int(data["id"]) != ucid:
        raise ValueError(f"Wrong UCID for {path.name}")
    records = []
    first, last = start.to_pydatetime(), end.to_pydatetime()
    for bar in quotes:
        opened = datetime.fromisoformat(bar["timeOpen"]).astimezone(timezone.utc).replace(tzinfo=None)
        closed = datetime.fromisoformat(bar["timeClose"]).astimezone(timezone.utc).replace(tzinfo=None)
        if opened != opened.replace(hour=0, minute=0, second=0, microsecond=0) or not first <= opened <= last:
            raise ValueError(f"Unexpected opening timestamp for {path.name}")
        if not opened + timedelta(hours=23) <= closed < opened + timedelta(days=1):
            raise ValueError(f"Unexpected closing timestamp for {path.name}")
        quote = bar["quote"]
        if int(quote["name"]) != 2781:
            raise ValueError(f"Unexpected quote currency for {path.name}")
        records.append((ucid, opened, closed, data["symbol"],
                        *(quote.get(col) for col in ("open", "high", "low", "close"))))
    return records, dict(entry, ucid=ucid, start=str(start.date()), end=str(end.date()),
                         n=len(records), symbol=data.get("symbol"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/crypto/v2.0/train.parquet"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    cache = args.output / "source_cache"
    cache.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.panel, columns=["ucid", "date"])
    panel = panel[panel.date <= CUTOFF]
    panel["ucid"] = panel["ucid"].astype(int)
    bounds = panel.groupby("ucid").date.agg(["min", "max"])
    # Fixed BTC and ETH benchmark series span the whole estimation period.
    bounds.loc[[1, 1027], "min"] = panel.date.min()
    bounds.loc[[1, 1027], "max"] = CUTOFF
    tasks = []
    for ucid, row in bounds.iterrows():
        first = row["min"] - pd.Timedelta(days=800)
        last = row["max"] - pd.Timedelta(days=1)
        for start in pd.date_range(first, last, freq="550D"):
            tasks.append((int(ucid), start, min(start + pd.Timedelta(days=549), last)))
    rows, manifest = [], []
    rate_limited = Event()
    with ThreadPoolExecutor(args.workers) as pool:
        for i, (records, metadata) in enumerate(pool.map(lambda task: fetch(cache, task, args.cache_only, rate_limited), tasks), 1):
            rows.extend(records)
            manifest.append(metadata)
            if i % 100 == 0:
                print(f"{i}/{len(tasks)} requests, {len(rows):,} bars", flush=True)
    candles = pd.DataFrame(rows, columns=[
        "ucid", "bar_date", "bar_close", "source_symbol", "open", "high", "low", "close",
    ])
    if candles.duplicated(["ucid", "bar_date"]).any():
        raise ValueError("Duplicate candles")
    bars_tmp = args.output / "ohlc.tmp"
    request_tmp = args.output / "price_requests.tmp"
    candles.sort_values(["ucid", "bar_date"]).to_parquet(bars_tmp, index=False)
    pd.DataFrame(manifest).to_json(request_tmp, orient="records", indent=2)
    bars_tmp.replace(args.output / "ohlc.parquet")
    request_tmp.replace(args.output / "price_requests.json")
    print(f"{len(rows):,} bars, {sum('error' in row for row in manifest)} failed requests", flush=True)


if __name__ == "__main__":
    main()
