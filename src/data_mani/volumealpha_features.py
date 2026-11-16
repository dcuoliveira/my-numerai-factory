"""Continuous, calendar-day volume-alpha signals on an admitted as-of snapshot.

This module never bins or fills missing values. A row dated D sees at most D-1.
The eight `technical` features are kept apart from every proposed extension.
"""
import numpy as np
import pandas as pd

PREFIX = "feature_va_"
WINDOWS = (1, 5, 22, 252)
EVENTS = ("unlock", "emission", "upgrade", "governance_close", "governance_execute",
          "airdrop_snapshot", "airdrop_claim", "launch", "listing", "delisting")
GLOBAL_EVENTS = ("btc_options_expiry", "eth_options_expiry", "futures_expiry", "index_rebalance")
FAMILIES = {
    "technical": [f"{kind}_ma{w}" for kind in ("return", "logdv") for w in WINDOWS],
    "liquidity": ["shock5", "shock22", "growth1", "growth5", "growth22", "acceleration",
                  "dv_vol22", "turnover", "amihud", "range_dv", "venue_count_change",
                  "venue_hhi_change", "volume_price_divergence"],
    "size": ["market_cap", "fdv"],
    "surprise": [f"{x}_surprise" for x in ("fees", "revenue", "transactions", "active_addresses", "dex_volume")],
    "leverage": ["oi_cap", "oi_spot", "funding", "liquidations_spot"],
    "valuation": ["cap_tvl", "cap_fees", "cap_revenue", "mvrv"],
    "beta": ["beta_btc", "beta_eth", "beta_market_loo"],
    "age": ["price_age", "volume_age", "launch_age"],
    "events": [f"distance_{x}" for x in EVENTS] + [f"{x}_beta_btc" for x in GLOBAL_EVENTS] +
              [f"{x}_oi" for x in GLOBAL_EVENTS] + ["weekend_sensitivity"],
    "cross_asset": ["market_shock_exposure", "btc_shock_exposure", "eth_shock_exposure",
                    "sector_shock_exposure", "sector_loo_share", "btc_leadlag", "eth_leadlag",
                    "market_share", "share_change", "breadth_exposure", "positive_fraction_exposure"],
}
FAMILIES = {k: [PREFIX + c for c in v] for k, v in FAMILIES.items()}
RAW_COLUMNS = sum(FAMILIES.values(), [])


def divide(a, b):
    return a / (b.replace(0, np.nan) if hasattr(b, "replace") else b if b != 0 else np.nan)


def dimson_beta(asset_return, market_return, window=120, minimum=60):
    """OLS on current + two lagged market returns, intercept included; no leads."""
    d = pd.concat([asset_return.rename("y"), market_return.rename("m0"),
                   market_return.shift(1).rename("m1"), market_return.shift(2).rename("m2")], axis=1)
    d = d.iloc[-window:].dropna()
    if len(d) < minimum:
        return np.nan
    X = np.column_stack([np.ones(len(d)), d[["m0", "m1", "m2"]]])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return np.nan
    return float(np.linalg.lstsq(X, d["y"], rcond=None)[0][1:].sum())


def _wide(daily, column, grid, assets):
    if column not in daily:
        return pd.DataFrame(np.nan, index=grid, columns=assets)
    return daily.pivot(index="date", columns="asset_id", values=column).reindex(index=grid, columns=assets)


def features_at(daily, date, assets, metrics=None, events=None, btc=None, eth=None,
                sectors=None, universe=None):
    """Features for one prediction date on the actual dated panel's mapped assets.

    daily / metrics / events must already be vintage-filtered at date's UTC
    midnight. `universe` is optional date-indexed *then-known* membership; market
    aggregates otherwise cover the supplied assets, never the entire crypto market.
    """
    D = pd.Timestamp(date)
    D = D.tz_localize("UTC") if D.tzinfo is None else D.tz_convert("UTC")
    assets = list(assets)
    out = pd.DataFrame(np.nan, index=pd.Index(assets, name="asset_id"), columns=RAW_COLUMNS)
    if daily.empty:
        return out
    daily = daily[daily["date"] < D.normalize()]
    if daily.empty:
        return out
    grid = pd.date_range(daily["date"].min(), D.normalize() - pd.Timedelta(days=1), freq="D")
    names = sorted(set(daily["asset_id"]) | set(assets))
    close, dv = (_wide(daily, c, grid, names).astype(float) for c in ("close", "dollar_volume"))
    ret, logdv = close.pct_change(fill_method=None), np.log1p(dv)
    shock = logdv - logdv.shift(1).rolling(5, min_periods=5).mean()
    ret_last = ret.iloc[-1]
    current_dv = dv.iloc[-1]

    def put(name, value):
        out[PREFIX + name] = value.reindex(assets) if isinstance(value, pd.Series) else value

    for w in WINDOWS:
        for kind, data in (("return", ret), ("logdv", logdv)):
            put(f"{kind}_ma{w}", data.rolling(w, min_periods=w).mean().iloc[-1])
    for w in (5, 22):
        put(f"shock{w}", (logdv - logdv.shift(1).rolling(w, min_periods=w).mean()).iloc[-1])
    for w in (1, 5, 22):
        put(f"growth{w}", (logdv - logdv.shift(w)).iloc[-1])
    put("acceleration", logdv.diff().diff().iloc[-1])
    put("dv_vol22", dv.rolling(22, min_periods=22).std().iloc[-1])
    put("amihud", divide(ret_last.abs(), current_dv))
    high, low = (_wide(daily, c, grid, names) for c in ("high", "low"))
    put("range_dv", divide(divide((high - low).iloc[-1], close.iloc[-1]), current_dv))
    for key, column in (("venue_count_change", "venue_count"), ("venue_hhi_change", "venue_hhi")):
        put(key, _wide(daily, column, grid, names).diff().iloc[-1])
    put("volume_price_divergence", logdv.diff(5).iloc[-1] - np.log(close).diff(5).iloc[-1])

    # Metric snapshots have one value per asset, economic day, metric. No as-of
    # filling: a daily flow/state absent on D-1 stays absent rather than stale.
    metric = {}
    if metrics is not None and not metrics.empty:
        for name, f in metrics.groupby("metric"):
            metric[name] = _wide(f.rename(columns={"value": name}), name, grid, names)
    empty = pd.DataFrame(np.nan, index=grid, columns=names)
    def series(name):
        return metric.get(name, empty)
    cap = series("market_cap").iloc[-1]
    put("market_cap", cap)
    put("fdv", series("fdv").iloc[-1])
    put("turnover", divide(current_dv, cap))
    for name in ("fees", "revenue", "transactions", "active_addresses", "dex_volume"):
        x = series(name)
        past = x.shift(1).rolling(22, min_periods=22)
        put(f"{name}_surprise", divide(x - past.mean(), past.std()).iloc[-1])
    oi = series("open_interest").iloc[-1]
    put("oi_cap", divide(oi, cap))
    put("oi_spot", divide(oi, current_dv))
    put("funding", series("funding").iloc[-1])
    put("liquidations_spot", divide(series("liquidations").iloc[-1], current_dv))
    put("cap_tvl", divide(cap, series("tvl").iloc[-1]))
    for name in ("fees", "revenue"):
        annual = 365 * series(name).rolling(30, min_periods=30).mean().iloc[-1]
        put(f"cap_{name}", divide(cap, annual))
    put("mvrv", divide(cap, series("realized_cap").iloc[-1]))

    for name, x in (("price_age", close), ("volume_age", dv)):
        first = x.apply(lambda s: s.first_valid_index())
        age = first.map(lambda t: (D - t).days if pd.notna(t) else np.nan)
        put(name, age)
    launch = series("launch_unix_seconds").iloc[-1]
    put("launch_age", (D.timestamp() - launch) / 86400)

    # Restrict market summaries to then-known membership, not surviving assets.
    members = pd.DataFrame(False, index=grid, columns=names)
    if universe is None:
        members.loc[:, assets] = True
    else:
        for day in grid:
            members.loc[day, list(set(universe.get(day, ())) & set(names))] = True
    mdv, mret = dv.where(members), ret.where(members)
    total = mdv.sum(axis=1, min_count=1)
    market_log = np.log1p(total)
    market_shock = market_log - market_log.shift(1).rolling(5, min_periods=5).mean()
    share = dv.div(total.replace(0, np.nan), axis=0)
    put("market_share", share.iloc[-1])
    put("share_change", share.diff().iloc[-1])
    breadth = shock.where(members).mean(axis=1)
    denominator = shock.where(members).notna().sum(axis=1).replace(0, np.nan)
    positive = ((shock > 0) & members).sum(axis=1) / denominator

    for a in assets:
        y = ret[a]
        loo = (mret.sum(axis=1, min_count=1) - mret[a].fillna(0)) / (
            mret.notna().sum(axis=1) - mret[a].notna().astype(int)).replace(0, np.nan)
        beta = dimson_beta(y, loo)
        out.loc[a, PREFIX + "beta_market_loo"] = beta
        out.loc[a, PREFIX + "market_shock_exposure"] = beta * market_shock.iloc[-1]
        out.loc[a, PREFIX + "breadth_exposure"] = beta * breadth.iloc[-1]
        out.loc[a, PREFIX + "positive_fraction_exposure"] = beta * positive.iloc[-1]
        for key, reference in (("btc", btc), ("eth", eth)):
            if reference not in ret:
                continue
            b = dimson_beta(y, ret[reference])
            out.loc[a, PREFIX + "beta_" + key] = b
            out.loc[a, PREFIX + key + "_shock_exposure"] = b * shock[reference].iloc[-1]
            out.loc[a, PREFIX + key + "_leadlag"] = shock[a].iloc[-60:].corr(shock[reference].shift(1).iloc[-60:], min_periods=40)
        if sectors and a in sectors:
            peers = [p for p in names if sectors.get(p) == sectors[a]]
            sdv = mdv[peers].sum(axis=1, min_count=1)
            slog = np.log1p(sdv)
            sshock = slog - slog.shift(1).rolling(5, min_periods=5).mean()
            out.loc[a, PREFIX + "sector_shock_exposure"] = beta * sshock.iloc[-1]
            out.loc[a, PREFIX + "sector_loo_share"] = divide(current_dv[a], sdv.iloc[-1] - current_dv[a])
        weekend = pd.Series(grid.dayofweek >= 5, index=grid)
        x = shock[a].iloc[-120:]
        sensitivity = x[weekend.iloc[-120:]].mean() - x[~weekend.iloc[-120:]].mean()
        if x.notna().sum() >= 60:
            out.loc[a, PREFIX + "weekend_sensitivity"] = float(D.dayofweek >= 5) * sensitivity

    if events is not None and not events.empty:
        e = events[(events["announced_at"] <= D) & (events["available_at"] <= D) & events["vintage_verified"].eq(True)]
        for a in assets:
            for kind in EVENTS:
                rows = e[(e["asset_id"] == a) & (e["kind"] == kind)]
                if rows.empty:
                    continue  # absence of a catalog row is not proof of no event
                distance = (D.normalize() - rows["scheduled_for"].dt.normalize()).dt.days
                nearest = distance.loc[distance.abs().sort_values().index[0]]
                out.loc[a, PREFIX + "distance_" + kind] = np.clip(nearest, -4, 5)
            for kind in GLOBAL_EVENTS:
                rows = e[(e["asset_id"] == "global") & (e["kind"] == kind)]
                if rows.empty:
                    continue
                active = float(rows["scheduled_for"].dt.normalize().eq(D.normalize()).any())
                out.loc[a, PREFIX + kind + "_beta_btc"] = active * out.loc[a, PREFIX + "beta_btc"]
                out.loc[a, PREFIX + kind + "_oi"] = active * oi[a]
    return out.replace([np.inf, -np.inf], np.nan)


def causal_winsorize(frame, columns, minimum=30):
    """Limits from strictly earlier DATE cross-sections, without order-dependent ties.

    Continuous features are pooled across prior panel dates. No limit is applied
    before minimum historical observations; future rows cannot change old limits.
    """
    out = frame.copy()
    ordered = frame.sort_values("date", kind="stable")
    dates = pd.to_datetime(ordered["date"])
    for col in columns:
        values = pd.to_numeric(ordered[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        past = values.shift(1).expanding(min_periods=minimum)
        first = ~dates.duplicated()
        lo = pd.Series(past.quantile(0.01).loc[first].to_numpy(), index=dates[first])
        hi = pd.Series(past.quantile(0.99).loc[first].to_numpy(), index=dates[first])
        clipped = values.clip(lower=dates.map(lo), upper=dates.map(hi))
        out.loc[ordered.index, col] = clipped
    return out
