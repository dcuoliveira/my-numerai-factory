"""E173: matched spot venue-pair volume comparisons; no fills or future rows."""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

SIGNALS = ['relative_return', 'surprise', 'persistent', 'volume_return', 'breadth',
           'venue_count', 'liquidity', 'liquidity_return']
CONTINUOUS = ['surprise', 'surprise_deviation', 'volume_return']
FOUR = ['binance', 'coinbase', 'okx', 'kucoin']


def matched_windows(values, width):
    """Sum the SAME complete pairs at every position of each calendar window."""
    n = len(values)
    result = np.full((n, width), np.nan)
    eligible = np.zeros((n, values.shape[1]), bool)
    if n < width:
        return result, eligible
    w = sliding_window_view(values, width, axis=0)
    ok = np.isfinite(w).all(axis=2)
    summed = np.where(ok[:, :, None], w, 0).sum(axis=1)
    summed[~ok.any(axis=1)] = np.nan
    result[width-1:] = summed
    eligible[width-1:] = ok
    return result, eligible


def asset_signals(g):
    grid = pd.date_range(g.date.min(), g.date.max(), freq='D')
    pairs = sorted(g[['venue', 'pair']].drop_duplicates().itertuples(index=False, name=None))
    def wide(c):
        return g.pivot(index='date', columns=['venue','pair'], values=c).reindex(index=grid, columns=pd.MultiIndex.from_tuples(pairs)).to_numpy(float)
    base, usd, price = [wide(c) for c in ['base_volume','dollar_volume','close_usd']]
    # A usable pair needs price, base volume and measured USD conversion on that day.
    valid = np.isfinite(base) & (base >= 0) & np.isfinite(usd) & (usd >= 0) & np.isfinite(price) & (price > 0)
    base, usd, price = [np.where(valid, x, np.nan) for x in (base, usd, price)]
    w21, ok21 = matched_windows(base, 21)
    w17, _ = matched_windows(base, 17)
    w20, _ = matched_windows(usd, 20)
    with np.errstate(divide='ignore', invalid='ignore'):
        surprise = np.log(w21[:, -1] / np.median(w21[:, :-1], axis=1))
        persistent = np.log(w17[:, -3:].mean(axis=1) / w17[:, :-3].mean(axis=1))
    liquidity = np.median(w20, axis=1)
    # Return: median USD prices over identical pairs at t and t-1.
    _, ok2 = matched_windows(price, 2)
    prev = np.vstack([np.full((1,price.shape[1]), np.nan), price[:-1]])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ret = np.nanmedian(np.where(ok2,price,np.nan),axis=1) / np.nanmedian(np.where(ok2,prev,np.nan),axis=1) - 1
    # Sum quote-pairs within each venue before its surge comparison. Each venue
    # uses a stable 21-day pair set; count distinct venues, never quote-pairs.
    reporting = np.zeros(len(grid), int); surges = np.zeros(len(grid), int)
    for venue in sorted(set(p[0] for p in pairs)):
        z, ok = matched_windows(base[:, [p[0] == venue for p in pairs]], 21)
        med = np.median(z[:,:-1], axis=1)
        good = ok.any(axis=1) & np.isfinite(med)
        reporting += good
        surges += good & (z[:,-1] > 1.5 * med)
    breadth = np.divide(surges, reporting, out=np.full(len(grid),np.nan), where=reporting >= 2)
    count = np.where(reporting > 0, reporting.astype(float), np.nan)
    out = pd.DataFrame(dict(date=grid + pd.Timedelta(days=2), usd_return=ret,
                           surprise=surprise, persistent=persistent, breadth=breadth,
                           venue_count=count, liquidity=liquidity))
    return out.replace([np.inf,-np.inf],np.nan)


def cross_section(panel):
    """Rank only observed assets in the actual prediction-date panel."""
    p = panel.copy()
    p['relative_return'] = p.usd_return - p.groupby('date').usd_return.transform('median')
    p['surprise_deviation'] = p.surprise - p.groupby('date').surprise.transform('median')
    p['volume_return'] = p.surprise * p.relative_return
    p['liquidity_return'] = p.liquidity * p.relative_return
    for c in SIGNALS:
        p['feature_sv_' + c] = p.groupby('date')[c].rank(method='average', pct=True)
    for c in CONTINUOUS:
        p['feature_svc_' + c] = p[c]
    return p


def arm_columns(base):
    f = lambda c: 'feature_sv_' + c
    c = lambda c: 'feature_svc_' + c
    a = {'A0':list(base)}
    a['A1'] = a['A0'] + [f('relative_return')]
    a['A2'] = a['A1'] + [f('surprise'),f('persistent'),c('surprise'),c('surprise_deviation')]
    a['A3'] = a['A2'] + [f('volume_return'),c('volume_return')]
    a['A4'] = a['A3'] + [f('breadth'),f('venue_count')]
    a['A5'] = a['A4'] + [f('liquidity'),f('liquidity_return')]
    return a


def prepare_fold(train, test, columns):
    """Training-only 1/99% winsorisation and standardisation of continuous inputs.

    Preserve NaNs in stored dataset. Estimators get neutral .5 rank values and
    zero for missing standardised continuous values, with no automatic indicators.
    """
    tr, te = train[columns].copy(), test[columns].copy(); bounds = {}
    for c in columns:
        if c.startswith('feature_svc_'):
            lo, hi = tr[c].quantile([.01,.99]).to_numpy()
            if not np.isfinite(lo): lo, hi = 0., 0.
            z = tr[c].clip(lo,hi); mean = float(z.mean()) if z.notna().any() else 0.
            sd = float(z.std(ddof=0)); sd = sd if np.isfinite(sd) and sd > 0 else 1.
            tr[c], te[c] = ((d[c].clip(lo,hi)-mean)/sd for d in (tr,te))
            tr[c], te[c] = tr[c].fillna(0), te[c].fillna(0)
            bounds[c] = dict(lower=float(lo),upper=float(hi),mean=mean,sd=sd)
        else:
            tr[c],te[c] = tr[c].fillna(.5),te[c].fillna(.5)
    return tr.to_numpy(np.float32), te.to_numpy(np.float32), bounds
