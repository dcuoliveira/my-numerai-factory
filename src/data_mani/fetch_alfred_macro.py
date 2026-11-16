"""Archive ALFRED initial releases, without replacing the legacy macro store.

Requires FRED_API_KEY (environment or .env). Gold and ICE DXY deliberately have
no substitutes here. Supply their separately archived records before building.
Initial-release values are a frozen series definition; later revisions are not
silently substituted. Raw responses retain the API's real-time period fields.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API = "https://api.stlouisfed.org/fred/series/observations"
SERIES = {"vix": "VIXCLS", "sp500": "SP500", "y10": "DGS10"}
DEFINITIONS = {"vix": "vix_close", "sp500": "sp500_price_index", "y10": "10y_constant_maturity_percent"}
ROOT = Path("data/crypto/macro/alfred")


def vintage_bound(dates):
    """Day-granular vintage dates: usable only after that entire US day ended.

    ALFRED has no intraday publication timestamp. Chicago midnight following
    the vintage day is an upper bound, not a claimed market-close timestamp.
    Localize after adding a calendar day, so DST boundaries are handled.
    """
    days = pd.to_datetime(dates) + pd.Timedelta(days=1)
    return days.dt.tz_localize("America/Chicago").dt.tz_convert("UTC")


def initial_records(observations, name, retrieved, raw_hash):
    f = pd.DataFrame(observations)
    if f.empty:
        raise ValueError("ALFRED returned no initial releases for " + name)
    f["value"] = pd.to_numeric(f["value"], errors="coerce")
    f = f.dropna(subset=["value"]).copy()
    f["date"] = pd.to_datetime(f["date"])
    if f.duplicated("date").any():
        raise ValueError("Initial-release response has duplicate observation dates")
    f["vintage_date"] = pd.to_datetime(f["realtime_start"])
    if (f["vintage_date"] < f["date"]).any():
        raise ValueError("Vintage predates observation; cannot establish availability")
    f["available_at"] = vintage_bound(f["vintage_date"])
    f["retrieved_at"] = pd.Timestamp(retrieved)
    f["series"] = name
    f["source"] = "alfred"
    f["source_id"] = SERIES[name]
    f["series_definition"] = DEFINITIONS[name]
    f["series_version"] = "initial-release:" + f["realtime_start"]
    f["availability_basis"] = "alfred_vintage_day_upper_bound"
    # Preserve the actual precision; never fabricate a source publication time.
    f["source_timestamp"] = pd.NaT
    f["source_timestamp_precision"] = "observation_date_only"
    f["raw_sha256"] = raw_hash
    return f.sort_values("date").reset_index(drop=True)


def fetch_series(session, key, name, start, end, directory):
    params = dict(series_id=SERIES[name], file_type="json", output_type=4,
                  realtime_start="1776-07-04", realtime_end=end,
                  observation_start=start, observation_end=end, limit=100000,
                  offset=0, sort_order="asc", units="lin")
    observations, pages = [], []
    while True:
        # Never print response URLs or requests exceptions: they contain the key.
        try:
            response = session.get(API, params=dict(params, api_key=key), timeout=60)
        except requests.RequestException:
            raise RuntimeError("ALFRED request failed for " + SERIES[name]) from None
        if response.status_code != 200:
            raise RuntimeError("ALFRED %s returned HTTP %d" % (SERIES[name], response.status_code))
        payload = response.json()
        rows = payload.get("observations", [])
        path = directory / ("%s_%06d.json" % (SERIES[name], params["offset"]))
        path.write_bytes(response.content)
        pages.append(dict(path=path.name, sha256=hashlib.sha256(response.content).hexdigest(),
                          parameters=dict(params)))
        observations.extend(rows)
        params["offset"] += len(rows)
        if params["offset"] >= int(payload["count"]):
            break
        if not rows:
            raise ValueError("ALFRED pagination stopped before count was reached")
    return observations, pages


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise SystemExit("Set FRED_API_KEY in the environment or .env; do not paste it into chat.")
    now = datetime.now(timezone.utc)
    directory = ROOT / now.strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    frames, files = [], {}
    with requests.Session() as session:
        for name in SERIES:
            rows, pages = fetch_series(session, key, name, args.start, args.end, directory)
            digest = hashlib.sha256("".join(p["sha256"] for p in pages).encode()).hexdigest()
            frames.append(initial_records(rows, name, now, digest))
            files[name] = pages
    store = pd.concat(frames, ignore_index=True)
    store.to_parquet(directory / "initial_releases.parquet", index=False)
    manifest = dict(retrieved_at=now.isoformat(), endpoint=API, output_type=4,
                    revision_policy="frozen initial release per observation",
                    availability="end of ALFRED vintage day in America/Chicago",
                    missing_sources=["gold spot/futures", "ICE DXY"], files=files,
                    store_sha256=hashlib.sha256((directory / "initial_releases.parquet").read_bytes()).hexdigest())
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(directory / "initial_releases.parquet")


if __name__ == "__main__":
    main()
