"""Strict source contracts for volumealpha; no ticker inference or vintage reconstruction.

Dates are UTC. `available_at` is the evidenced publication time of THIS vintage,
not the bar's economic date or the time a modern download labels it with.
See docs/volumealpha-causal.md for the input contracts.
"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


PROVENANCE = ["available_at", "retrieved_at", "evidence", "vintage_verified"]
MARKET_KEYS = ["asset_id", "venue", "pair", "date"]


def require(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if frame[columns].isna().any().any():
        raise ValueError(f"{name}: required fields contain nulls")


def utc(values):
    return pd.to_datetime(values, utc=True, errors="raise")


def validate_provenance(frame, name):
    require(frame, PROVENANCE, name)
    out = frame.copy()
    for c in ("available_at", "retrieved_at"):
        out[c] = utc(out[c])
    if not out["vintage_verified"].isin([True, False]).all():
        raise ValueError(f"{name}: vintage_verified must be boolean")
    if out["evidence"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{name}: empty provenance evidence")
    if (out["retrieved_at"] < out["available_at"]).any():
        raise ValueError(f"{name}: retrieval precedes publication")
    return out


def validate_market(frame, trusted_venues):
    required = MARKET_KEYS + ["observed_at", "source_family", "market_type",
                             "base_volume", "price_quote", "quote", "quote_usd",
                             "quote_fx_at", "quote_fx_available_at", "quote_fx_evidence"]
    require(frame, required, "market")
    out = validate_provenance(frame, "market")
    for c in ("date", "observed_at", "quote_fx_at", "quote_fx_available_at"):
        out[c] = utc(out[c])
    if not out["date"].eq(out["date"].dt.normalize()).all():
        raise ValueError("market: date must be a UTC day")
    if not out["source_family"].isin(["cex", "dex"]).all():
        raise ValueError("market: source_family must be cex or dex")
    if not out["market_type"].eq("spot").all():
        raise ValueError("market: perpetual/futures volume is not spot volume")
    if not out["venue"].isin(trusted_venues).all():
        raise ValueError("market: venue missing from explicit trusted-venue allowlist")
    for c in ("base_volume", "price_quote", "quote_usd"):
        v = pd.to_numeric(out[c], errors="raise")
        if not np.isfinite(v).all() or (v < 0).any() or (c != "base_volume" and (v == 0).any()):
            raise ValueError(f"market: invalid {c}")
        out[c] = v
    # A daily close is admissible, but not a quote conversion from another day.
    if not out["quote_fx_at"].dt.normalize().eq(out["date"]).all():
        raise ValueError("market: FX conversion is not contemporaneous")
    if out["quote_fx_evidence"].astype(str).str.strip().eq("").any():
        raise ValueError("market: USD conversion needs evidence, including stablecoins")
    if ((out["observed_at"] < out["date"]) |
            (out["observed_at"] >= out["date"] + pd.Timedelta(days=1))).any():
        raise ValueError("market: source timestamp outside observation day")
    if (out["available_at"] < out["observed_at"]).any():
        raise ValueError("market: publication precedes source observation")
    if out.duplicated(MARKET_KEYS + ["available_at"]).any():
        raise ValueError("market: duplicate venue/pair/day/vintage (would double count)")
    return out


def validate_identities(frame):
    require(frame, ["ucid", "asset_id", "chain", "contract_address", "valid_from", "valid_to"], "identities")
    out = validate_provenance(frame, "identities")
    for c in ("valid_from", "valid_to"):
        out[c] = utc(out[c])
    out["ucid"] = out["ucid"].astype(str)
    if (out["valid_to"] <= out["valid_from"]).any():
        raise ValueError("identities: invalid half-open validity interval")
    # Native assets use contract_address='native'; chain namespaces avoid collisions.
    expected = out["chain"].astype(str) + ":" + out["contract_address"].astype(str)
    if not out["asset_id"].eq(expected).all():
        raise ValueError("identities: asset_id must equal chain:contract_address (or chain:native)")
    for _, g in out.groupby("ucid"):
        g = g.sort_values("valid_from")
        if (g["valid_from"].iloc[1:].to_numpy() < g["valid_to"].iloc[:-1].to_numpy()).any():
            raise ValueError("identities: overlapping identity intervals for a ucid")
    return out


def snapshot(frame, cutoff, keys, date_col="date"):
    """Latest VERIFIED vintage available at cutoff, never the latest downloaded one."""
    cutoff = pd.Timestamp(cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    mask = frame["vintage_verified"].eq(True) & (frame["available_at"] <= cutoff)
    if date_col is not None:
        mask &= utc(frame[date_col]) < cutoff.normalize()
    if "quote_fx_available_at" in frame:
        mask &= frame["quote_fx_available_at"] <= cutoff
    return frame.loc[mask].sort_values("available_at").drop_duplicates(keys, keep="last")


def identity_at(identities, date):
    d = pd.Timestamp(date)
    d = d.tz_localize("UTC") if d.tzinfo is None else d.tz_convert("UTC")
    m = (identities["vintage_verified"].eq(True) & (identities["available_at"] <= d) &
         (identities["valid_from"] <= d) & (d < identities["valid_to"]))
    out = identities.loc[m]
    if out["ucid"].duplicated().any() or out["asset_id"].duplicated().any():
        raise ValueError("identity mapping is not one-to-one at cutoff")
    return out


def consolidate(market, family="cex"):
    """One asset/day; quote pairs counted once and concentration measured by VENUE."""
    if family not in ("cex", "dex", "combined"):
        raise ValueError("source arm must be cex, dex or combined")
    d = market if family == "combined" else market[market["source_family"] == family]
    cols = ["asset_id", "date", "dollar_volume", "close", "venue_count", "venue_hhi"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    d = d.copy()
    d["dollar_volume"] = d["base_volume"] * d["price_quote"] * d["quote_usd"]
    d["usd_price"] = d["price_quote"] * d["quote_usd"]
    d["price_weighted"] = d["usd_price"] * d["dollar_volume"]
    key = ["asset_id", "date"]
    daily = d.groupby(key).agg(dollar_volume=("dollar_volume", "sum"),
                               price_weighted=("price_weighted", "sum"),
                               price_fallback=("usd_price", "median"),
                               available_at=("available_at", "max"),
                               quote_fx_available_at=("quote_fx_available_at", "max"))
    daily["available_at"] = daily[["available_at", "quote_fx_available_at"]].max(axis=1)
    daily["close"] = (daily["price_weighted"] / daily["dollar_volume"].replace(0, np.nan)).fillna(daily["price_fallback"])
    venues = d.groupby(key + ["venue"])["dollar_volume"].sum()
    total = venues.groupby(level=[0, 1]).transform("sum").replace(0, np.nan)
    daily["venue_hhi"] = (venues / total).pow(2).groupby(level=[0, 1]).sum(min_count=1)
    daily["venue_count"] = venues.groupby(level=[0, 1]).size()
    if {"high_usd", "low_usd"} <= set(d):
        daily["high"] = d.groupby(key)["high_usd"].max()
        daily["low"] = d.groupby(key)["low_usd"].min()
    return daily.drop(columns=["price_weighted", "price_fallback", "quote_fx_available_at"]).reset_index()


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_local(root="/scratch/danielco/volalpha"):
    """Metadata-only audit of existing reconstruction stores; never fabricates availability."""
    import pyarrow.parquet as pq
    found = []
    paths = list(Path(root).glob("binance/spot/*.parquet"))[:3]
    paths += list(Path(root).glob("onchain/*.parquet"))
    paths += [Path("data/crypto/spot/ohlcv.parquet")]
    for p in paths:
        if not p.exists():
            continue
        f = pq.ParquetFile(p)
        columns = set(f.schema.names)
        missing = sorted(set(MARKET_KEYS + PROVENANCE + ["quote_usd", "quote_fx_at"]) - columns)
        found.append({"path": str(p), "rows": f.metadata.num_rows,
                      "columns": sorted(columns), "missing_strict_market_fields": missing,
                      "admitted_as_strict_market": not missing})
    return {"status": "BLOCKED_SOURCE_PROVENANCE", "stores_sampled": found,
            "reason": "Existing retrospective archives do not establish dated permanent identity, "
                      "original-vintage publication, and contemporaneous USD conversion. "
                      "A price-feature correlation gate is not proof of permanent identity."}
