"""
E151 volume-alpha -- continuous, causal, calendar-daily features per symbol (before attach, gate and binning).

Every function takes daily long frames from volalpha_sources and returns a long frame keyed (symbol, date)
whose row dated d uses ONLY observations dated <= d. The builder then attaches row d to Numerai date d+1
(attach_signals_asof, lag 1), so a Numerai date D reads data through D-1. Rolling windows are calendar days
(crypto trades every day); a window needs at least 60% of its days observed.

Names: feature_va_<block>_<name>[_<fam>]. Blocks: tech, ext, xa (cross-asset), beta, age, lev, size, surp,
val, ev (events), oof, cost.
"""
import numpy as np
import pandas as pd

TECH_WINDOWS = (1, 5, 22, 252)
COVER = 0.6
SECTOR_K, SECTOR_LOOKBACK, SECTOR_MIN = 8, 180, 120
SENS_WIN, SENS_MIN = 90, 60


def wide(df, col, days=None):
    w = df.pivot_table(index="date", columns="symbol", values=col, aggfunc="last")
    days = days if days is not None else pd.date_range(w.index.min(), w.index.max())
    return w.reindex(days)


def roll(w, k, fn="mean"):
    if k == 1:
        return w
    r = w.rolling(k, min_periods=max(1, int(np.ceil(COVER * k))))
    return getattr(r, fn)()


def long(frames):
    """{name: wide frame} -> long (symbol, date, name...) frame, rows with any value."""
    parts = []
    for n, w in frames.items():
        s = w.stack(future_stack=True)
        s.name = n
        parts.append(s)
    out = pd.concat(parts, axis=1)
    out.index.names = ["date", "symbol"]
    out = out.dropna(how="all").reset_index()
    return out


def basics(fam_df, days):
    close, dv = wide(fam_df, "close", days), wide(fam_df, "dv", days)
    ret = close / close.shift(1) - 1.0                      # calendar daily return, NaN across a gap
    logdv = np.log1p(dv.where(dv >= 0))
    return close, dv, ret, logdv


def tech(fam_df, fam, days):
    """The paper's technical block: lagged moving averages of daily return and of log dollar volume."""
    _, _, ret, logdv = basics(fam_df, days)
    f = {}
    for k in TECH_WINDOWS:
        f["feature_va_tech_ret_ma%d_%s" % (k, fam)] = roll(ret, k)
        f["feature_va_tech_logdv_ma%d_%s" % (k, fam)] = roll(logdv, k)
    return f


def shock(logdv, k):
    """logdv minus its trailing k-day mean over the PRIOR days (the eta construction)."""
    return logdv - roll(logdv.shift(1), k)


def ext(ven, days):
    close, dv, ret, logdv = basics(ven, days)
    high, low = wide(ven, "high", days), wide(ven, "low", days)
    nven, hhi = wide(ven, "n_venues", days), wide(ven, "hhi", days)
    g1 = logdv - logdv.shift(1)
    ma5, ma22 = roll(logdv, 5), roll(logdv, 22)
    g5 = ma5 - ma5.shift(5)
    dvp = dv.where(dv > 0)
    f = {"feature_va_ext_shock5": shock(logdv, 5), "feature_va_ext_shock22": shock(logdv, 22),
         "feature_va_ext_growth1": g1, "feature_va_ext_growth5": g5, "feature_va_ext_growth22": ma22 - ma22.shift(22),
         "feature_va_ext_accel": g5 - g5.shift(5), "feature_va_ext_dvvol22": roll(g1, 22, "std"),
         "feature_va_ext_amihud22": roll(ret.abs() / dvp, 22),
         "feature_va_ext_rangedv22": roll(((high - low) / close) / dvp, 22),
         "feature_va_ext_venchg22": nven - nven.shift(22), "feature_va_ext_hhichg22": hhi - hhi.shift(22),
         "feature_va_ext_vpcorr22": ret.rolling(22, min_periods=14).corr(g1)}
    return f


def rolling_beta(y, x, win=SENS_WIN, minp=SENS_MIN):
    """Per column beta of y on the series x (a Series on the same index), rolling."""
    X = pd.DataFrame(np.repeat(x.to_numpy()[:, None], y.shape[1], axis=1), index=y.index, columns=y.columns)
    X = X.where(y.notna())
    yy = y.where(X.notna())
    mx, my = X.rolling(win, min_periods=minp).mean(), yy.rolling(win, min_periods=minp).mean()
    cov = (X * yy).rolling(win, min_periods=minp).mean() - mx * my
    var = (X * X).rolling(win, min_periods=minp).mean() - mx * mx
    return cov / var.where(var > 0)


def dimson_beta(r, m, win=SENS_WIN, minp=SENS_MIN):
    """beta = b0 + b1 of r_t = a + b0 m_t + b1 m_{t-1} + e, rolling OLS per column. m: Series or frame."""
    M0 = m if isinstance(m, pd.DataFrame) else pd.DataFrame(np.repeat(m.to_numpy()[:, None], r.shape[1], 1),
                                                            index=r.index, columns=r.columns)
    M1 = M0.shift(1)
    ok = r.notna() & M0.notna() & M1.notna()
    y, a, b = r.where(ok), M0.where(ok), M1.where(ok)
    R = lambda z: z.rolling(win, min_periods=minp).mean()
    ma, mb, my = R(a), R(b), R(y)
    saa, sbb, sab = R(a * a) - ma * ma, R(b * b) - mb * mb, R(a * b) - ma * mb
    say, sby = R(a * y) - ma * my, R(b * y) - mb * my
    det = saa * sbb - sab * sab
    det = det.where(det.abs() > 1e-18)
    b0 = (sbb * say - sab * sby) / det
    b1 = (saa * sby - sab * say) / det
    return b0 + b1


def loo_market(ret, dv):
    """Leave-one-out market return: lagged-22d-mean-dollar-volume-weighted mean of every OTHER coin."""
    w = roll(dv, 22).shift(1).where(ret.notna())
    rw = (ret * w)
    tot_rw, tot_w = rw.sum(axis=1, min_count=1), w.sum(axis=1, min_count=1)
    num = -rw.sub(tot_rw, axis=0)
    den = -w.sub(tot_w, axis=0)
    return num / den.where(den > 0)


def beta_block(ven, days):
    close, dv, ret, _ = basics(ven, days)
    f = {}
    for c in ("BTC", "ETH"):
        if c in ret:
            f["feature_va_beta_dimson_%s" % c.lower()] = dimson_beta(ret, ret[c])
    f["feature_va_beta_dimson_loo"] = dimson_beta(ret, loo_market(ret, dv))
    return f


def age_block(ven, agg, listing, days):
    """Days since the first observation. CENSORED at the store's start (2018-11-01): every coin listed
    before it shares one age on a given date."""
    f = {}
    for name, df, col in (("price", pd.concat([ven, agg]), "close"), ("volume", ven[ven["dv"] > 0], "dv")):
        first = df.groupby("symbol")["date"].min()
        f["feature_va_age_%s" % name] = first
    lb = listing[listing["kind"] == "bin_spot"].set_index("symbol")["first"]
    f["feature_va_age_binlist"] = lb
    out = {}
    D = pd.DataFrame(index=days)
    for n, first in f.items():
        a = pd.DataFrame({s: (days - d).days.astype(float) for s, d in first.items()}, index=days)
        out[n] = a.where(a >= 0)
    return out


def sectors(ret, at):
    """Statistical sectors from returns strictly before `at`: k-means on the rows of the trailing 180-day
    return-correlation matrix of coins with >= SECTOR_MIN observations. {symbol: sector}."""
    from sklearn.cluster import KMeans
    w = ret.loc[(ret.index < at) & (ret.index >= at - pd.Timedelta(days=SECTOR_LOOKBACK))]
    w = w.loc[:, w.notna().sum() >= SECTOR_MIN]
    if w.shape[1] < 3 * SECTOR_K:
        return {}
    c = w.corr(min_periods=SECTOR_MIN // 2).fillna(0.0).to_numpy()
    lab = KMeans(SECTOR_K, n_init=10, random_state=0).fit_predict(c)
    return dict(zip(w.columns, lab))


def xasset(ven, days):
    close, dv, ret, logdv = basics(ven, days)
    s5 = shock(logdv, 5)
    g1 = logdv - logdv.shift(1)
    mkt = s5.median(axis=1)
    breadth = (s5 > 0).sum(axis=1) / s5.notna().sum(axis=1).replace(0, np.nan)
    f = {}
    sens_m = rolling_beta(s5, mkt)
    f["feature_va_xa_mktshock_x_sens"] = sens_m.mul(mkt, axis=0)
    f["feature_va_xa_breadth_x_sens"] = sens_m.mul(breadth - 0.5, axis=0)
    for c in ("BTC", "ETH"):
        if c in s5:
            sens = rolling_beta(s5, s5[c])
            f["feature_va_xa_%sshock_x_sens" % c.lower()] = sens.mul(s5[c], axis=0)
            gl = g1[c].shift(1)
            G = pd.DataFrame(np.repeat(gl.to_numpy()[:, None], g1.shape[1], 1), index=g1.index, columns=g1.columns)
            f["feature_va_xa_leadlag_%s" % c.lower()] = g1.rolling(SENS_WIN, min_periods=SENS_MIN).corr(G)
    share = roll(dv, 5).div(roll(dv, 5).sum(axis=1, min_count=1), axis=0)
    f["feature_va_xa_sharechg22"] = np.log(share.where(share > 0)) - np.log(share.where(share > 0)).shift(22)
    # statistical sectors, re-estimated each quarter from returns before the quarter
    sect = pd.DataFrame(np.nan, index=days, columns=s5.columns)
    for q in pd.date_range(days.min().to_period("Q").start_time, days.max(), freq="QS"):
        lab = sectors(ret, q)
        rows = (days >= q) & (days < q + pd.offsets.QuarterBegin())
        for s, k in lab.items():
            sect.loc[rows, s] = k
    loo = pd.DataFrame(np.nan, index=days, columns=s5.columns)
    for k in range(SECTOR_K):
        m = sect == k
        x = s5.where(m)
        tot, n = x.sum(axis=1, min_count=1), x.notna().sum(axis=1)
        val = (-x.sub(tot, axis=0)).div((n - 1).where(n > 1), axis=0)
        loo = loo.where(~(m & x.notna()), val)
    f["feature_va_xa_sectorloo_shock"] = loo
    sb = loo.where(s5.notna())
    # coin sensitivity to its own leave-one-out sector shock (90 days), times today's sector shock
    mx, my = sb.rolling(SENS_WIN, min_periods=SENS_MIN).mean(), s5.where(sb.notna()).rolling(SENS_WIN, min_periods=SENS_MIN).mean()
    cov = (sb * s5).rolling(SENS_WIN, min_periods=SENS_MIN).mean() - mx * my
    var = (sb * sb).rolling(SENS_WIN, min_periods=SENS_MIN).mean() - mx * mx
    f["feature_va_xa_sectorloo_x_sens"] = (cov / var.where(var > 0)) * loo
    return f, {"mkt": mkt, "breadth": breadth}


def leverage(perp, ven, days):
    _, dv, _, _ = basics(ven, days)
    oi = wide(perp, "oi_value", days) if "oi_value" in perp else None
    f = {}
    if oi is not None:
        oi = oi.where(oi > 0)
        f["feature_va_lev_oi_dv"] = oi / roll(dv, 7).where(roll(dv, 7) > 0)
        f["feature_va_lev_oichg7"] = np.log(oi) - np.log(oi.shift(7))
        f["_oi_value"] = oi
    if "funding_sum" in perp:
        f["feature_va_lev_funding7"] = roll(wide(perp, "funding_sum", days), 7)
    return f


def surprise(x, win=90):
    """(log1p(7-day sum) - its trailing 90-day mean) / its trailing 90-day std, the trailing stats taken on
    values at least 7 days old so the window never overlaps the current week."""
    s7 = np.log1p(x.where(x >= 0).rolling(7, min_periods=5).sum())
    past = s7.shift(7)
    mu, sd = past.rolling(win, min_periods=int(COVER * win)).mean(), past.rolling(win, min_periods=int(COVER * win)).std()
    return (s7 - mu) / sd.where(sd > 0)


def weekend_sensitivity(logdv):
    """Trailing 252-day mean logdv on Sat/Sun minus on weekdays (needs >= 30 weekend days)."""
    wk = pd.Series(logdv.index.dayofweek >= 5, index=logdv.index)
    we = logdv.where(wk, axis=0).rolling(252, min_periods=30 * 7 // 2).mean()
    wd = logdv.where(~wk, axis=0).rolling(252, min_periods=60).mean()
    return we - wd


def expiry_days(days):
    """Last Friday of every month (BTC/ETH monthly options on Deribit; CME monthly futures)."""
    m = pd.date_range(days.min() - pd.offsets.MonthBegin(2), days.max() + pd.offsets.MonthEnd(2), freq="ME")
    return pd.DatetimeIndex([d - pd.Timedelta(days=(d.dayofweek - 4) % 7) for d in m])


def expiry_window(dates, exp):
    """1 when D lies in [E-3, E+1] for some expiry E (scheduled, known in advance)."""
    d = pd.DatetimeIndex(dates)
    pos = np.searchsorted(exp.values, d.values)
    nxt = exp.values[np.minimum(pos, len(exp) - 1)]
    prv = exp.values[np.maximum(pos - 1, 0)]
    a = (nxt - d.values).astype("timedelta64[D]").astype(int)
    b = (d.values - prv).astype("timedelta64[D]").astype(int)
    return ((a >= 0) & (a <= 3)) | ((b >= 0) & (b <= 1))


def volume_event_beta(s5, exp):
    """Coin's mean shock5 on past expiry days over the trailing 365 days (>= 3 events)."""
    flag = pd.Series(s5.index.isin(exp), index=s5.index)
    x = s5.where(flag, axis=0)
    return x.rolling(365, min_periods=1).mean().where(x.notna().rolling(365, min_periods=1).sum() >= 3)
