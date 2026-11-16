"""
E151 volume-alpha -- raw stores -> tidy daily series per (source family, symbol).

The four price/volume FAMILIES are kept apart until a feature asks for one of them:
    bin   Binance spot, every quote, from the public archive (delisted pairs included)
    cex   gate, okx, kucoin, bybit, bitget, coinbase, mexc, hyperliquid (REST; currently listed pairs only)
    ven   bin + cex: the consolidated venue sum (the primary series)
    agg   Yahoo <SYM>-USD (an aggregator's retrospective consolidation, already in USD)
Plus the derivatives family (Binance USD-M perps: volume, open interest, funding, long/short ratios) and the
revised fundamentals (Coin Metrics by asset, DefiLlama by cmcId).

dollar_volume = base_volume x close x USD(quote): stablecoin quotes through their own Yahoo USD rate,
BTC/ETH/BNB quotes through that coin's Binance USDT close x USDT. The venues' quote-volume fields are not
used. One venue's different quote pairs are different order books and are summed; the aggregator is never
added to the venue sum and perps never to spot.

Bad prints: a pair-day whose USD close is more than 25% from the median USD close of all venue pairs of
that symbol that day is dropped (a reused ticker, a dead pair, a stale quote). A symbol with one pair has
no cross-check here; the panel-level identity gate catches a wrong coin there.

Everything is calendar-daily (UTC bars), keyed by ticker `symbol`. Which panel coin (ucid) a series belongs
to is decided later, per row, by the causal identity gate in build_volalpha.
"""
import os

import numpy as np
import pandas as pd

ROOT = "/scratch/danielco/volalpha"
STABLES = ("USDT", "BUSD", "USDC", "FDUSD", "TUSD", "USDP", "DAI")
OUTLIER = np.log(1.25)
CEX_QUOTE = {"gate": "USDT", "kucoin": "USDT", "okx": "USDT", "bybit": "USDT", "bitget": "USDT",
             "coinbase": "USD", "mexc": "USDT", "hyperliquid": "USDC"}


def _read(path, **kw):
    return pd.read_parquet(os.path.join(ROOT, path), **kw)


def usd_rates(spot):
    """(quote, date) -> USD per unit of quote. spot: the Binance spot frame (for BTC/ETH/BNB)."""
    st = _read("onchain/stables.parquet")
    rates = [st.rename(columns={"stable": "quote"})[["quote", "date", "usd"]]]
    usdt = st[st["stable"] == "USDT"].set_index("date")["usd"]
    for c in ("BTC", "ETH", "BNB"):
        s = spot[(spot["pair"] == c + "USDT")].set_index("date")["close"]
        rates.append(pd.DataFrame({"quote": c, "usd": s * usdt.reindex(s.index).fillna(1.0)}).reset_index())
    days = pd.date_range("2018-01-01", pd.Timestamp.utcnow().tz_localize(None).normalize())
    rates.append(pd.DataFrame({"quote": "USD", "date": days, "usd": 1.0}))
    r = pd.concat(rates, ignore_index=True)
    r["date"] = pd.to_datetime(r["date"]).dt.normalize()
    return r.drop_duplicates(["quote", "date"])


def venue_pairs():
    """One row per (symbol, venue, pair, date) with USD close/high/low, base volume and dollar volume."""
    sp = _read("binance/spot.parquet")
    sp["date"] = pd.to_datetime(sp["date"])
    rates = usd_rates(sp)
    b = sp.rename(columns={"base": "symbol", "volume": "base_vol"})[
        ["symbol", "pair", "quote", "date", "open", "high", "low", "close", "base_vol", "trades", "taker_buy_base"]]
    b["venue"] = "binance"
    c = _read("cex.parquet")
    c["date"] = pd.to_datetime(c["date"])
    c = c.rename(columns={"volume": "base_vol"})
    c["quote"] = c["venue"].map(CEX_QUOTE)
    c["symbol"] = c["symbol"].astype(str).str.upper().str.lstrip("$")
    v = pd.concat([b, c[["symbol", "venue", "pair", "quote", "date", "open", "high", "low", "close", "base_vol"]]],
                  ignore_index=True)
    v = v.merge(rates, on=["quote", "date"], how="left")
    for k in ("open", "high", "low", "close"):
        v[k + "_usd"] = v[k] * v["usd"]
    v = v[(v["close_usd"] > 0) & v["base_vol"].ge(0)]
    v["dv"] = v["base_vol"] * v["close_usd"]
    med = v.groupby(["symbol", "date"])["close_usd"].transform("median")
    v["outlier"] = np.abs(np.log(v["close_usd"] / med)) > OUTLIER
    return v


def family_daily(v, fam):
    """Per (symbol, date): close/high/low (median over the family's clean pairs), dollar volume (sum), the
    number of venues with volume, venue concentration (HHI of venue dollar-volume shares), Binance trades."""
    if fam == "bin":
        v = v[v["venue"] == "binance"]
    elif fam == "cex":
        v = v[v["venue"] != "binance"]
    v = v[~v["outlier"]]
    g = v.groupby(["symbol", "date"])
    out = g.agg(close=("close_usd", "median"), high=("high_usd", "median"), low=("low_usd", "median"),
                open=("open_usd", "median"), dv=("dv", "sum"), n_pairs=("pair", "size")).reset_index()
    ven = v[v["dv"] > 0].groupby(["symbol", "date", "venue"])["dv"].sum().rename("vdv").reset_index()
    ven["share"] = ven["vdv"] / ven.groupby(["symbol", "date"])["vdv"].transform("sum")
    vs = ven.groupby(["symbol", "date"]).agg(n_venues=("venue", "size"),
                                              hhi=("share", lambda s: float((s * s).sum()))).reset_index()
    out = out.merge(vs, on=["symbol", "date"], how="left")
    out["n_venues"] = out["n_venues"].fillna(0)
    if fam in ("bin", "ven"):
        tr = v[v["venue"] == "binance"].groupby(["symbol", "date"])["trades"].sum().rename("trades").reset_index()
        out = out.merge(tr, on=["symbol", "date"], how="left")
    out["family"] = fam
    return out


def yahoo_daily():
    s = pd.read_parquet("data/crypto/spot/ohlcv.parquet")
    s = s[s["source"] == "yahoo"].copy()
    s["date"] = pd.to_datetime(s["date"]).dt.normalize()
    s["symbol"] = s["symbol"].astype(str).str.upper().str.lstrip("$")
    s = s[s["close"] > 0]
    out = s.rename(columns={"volume": "dv"})[["symbol", "date", "open", "high", "low", "close", "dv"]]
    out["n_venues"], out["hhi"], out["family"] = np.nan, np.nan, "agg"
    return out.drop_duplicates(["symbol", "date"])


def families():
    """{fam: daily frame} for bin, cex, ven, agg."""
    v = venue_pairs()
    out = {f: family_daily(v, f) for f in ("bin", "cex", "ven")}
    out["agg"] = yahoo_daily()
    return out, v


def perps():
    """Per (symbol, date): perp dollar volume (summed over USD-margined pairs), open-interest value (last
    snapshot of the day, summed over pairs), funding (the USDT perp's daily sum and mean), long/short ratios
    and taker buy/sell ratio of the most liquid pair."""
    um = _read("binance/um.parquet")
    um["date"] = pd.to_datetime(um["date"])
    sp = _read("binance/spot.parquet", columns=["pair", "date", "close"])
    sp["date"] = pd.to_datetime(sp["date"])
    rates = usd_rates(sp)
    um = um.merge(rates, on=["quote", "date"], how="left")
    um["pdv"] = um["volume"] * um["close"] * um["usd"].fillna(1.0)
    d = um.groupby(["base", "date"]).agg(perp_dv=("pdv", "sum"), perp_close=("close", "last"),
                                         perp_mult=("mult", "first")).reset_index()
    out = d.rename(columns={"base": "symbol"})
    if os.path.exists(os.path.join(ROOT, "binance/metrics.parquet")):
        m = _read("binance/metrics.parquet")
        m["date"] = pd.to_datetime(m["date"])
        mm = m.groupby(["base", "date"]).agg(
            oi_value=("sum_open_interest_value_last", "sum"), oi_value_mean=("sum_open_interest_value_mean", "sum"),
            ls_top_pos=("sum_toptrader_long_short_ratio_mean", "mean"), ls_all=("count_long_short_ratio_mean", "mean"),
            taker_ls=("sum_taker_long_short_vol_ratio_mean", "mean")).reset_index().rename(columns={"base": "symbol"})
        out = out.merge(mm, on=["symbol", "date"], how="outer")
    if os.path.exists(os.path.join(ROOT, "binance/funding.parquet")):
        f = _read("binance/funding.parquet")
        f["date"] = pd.to_datetime(f["date"])
        f = f[f["quote"] == "USDT"].groupby(["base", "date"]).agg(funding_sum=("funding_sum", "mean"),
                                                                  funding_mean=("funding_mean", "mean")).reset_index()
        out = out.merge(f.rename(columns={"base": "symbol"}), on=["symbol", "date"], how="outer")
    return out


def listings(v):
    """First and last bar per (symbol, venue) and of Binance perps: the listing / delisting dates. Post-event
    use only -- the announcement dates are not in any free source."""
    ev = []
    b = v[v["venue"] == "binance"].groupby("symbol")["date"].agg(["min", "max"]).reset_index()
    ev.append(b.assign(kind="bin_spot"))
    um = _read("binance/um.parquet", columns=["base", "date"])
    um["date"] = pd.to_datetime(um["date"])
    p = um.groupby("base")["date"].agg(["min", "max"]).reset_index().rename(columns={"base": "symbol"})
    ev.append(p.assign(kind="bin_perp"))
    return pd.concat(ev, ignore_index=True).rename(columns={"min": "first", "max": "last"})


def coinmetrics():
    p = os.path.join(ROOT, "onchain/coinmetrics.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else None


def llama():
    p = os.path.join(ROOT, "onchain/llama.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else None


def snapshot():
    p = os.path.join(ROOT, "onchain/snapshot.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else None
