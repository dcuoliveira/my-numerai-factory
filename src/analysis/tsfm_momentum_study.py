"""E206 numerical protocol: chronological scalers, per-asset ridges and portfolios."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

MODELS = ['timestral', 't0', 'chronos2', 'tabpfnts']
NORMS = ['pooled', 'asset', 'cross_section', 'rolling', 'robust', 'l2']
DESIGNS = ['A', 'B', 'C']
ALPHAS = np.logspace(-4, 4, 33)
ROLLING = [63, 126, 252]
FLOOR = 1e-8
WALL = 35


def split_masks(dates):
    d = np.asarray(dates, dtype='datetime64[D]')
    val_start, test_start = np.datetime64('2024-01-01'), np.datetime64('2025-01-01')
    train = d <= val_start-np.timedelta64(WALL, 'D')
    val = (d >= val_start) & (d <= test_start-np.timedelta64(WALL, 'D'))
    final = d <= test_start-np.timedelta64(WALL, 'D')
    test = (d >= test_start) & (d < np.datetime64('2026-02-20'))
    return train, val, final, test


def train_standardization(h, assets, fit, kind):
    """Returns immutable fit references; unknown asset references are unsupported."""
    x = np.asarray(h[fit], np.float64)
    if not len(x):
        raise ValueError('No scaler training rows')
    if kind == 'robust':
        center = np.median(x, axis=0)
        scale = (np.percentile(x, 75, axis=0)-np.percentile(x, 25, axis=0))/1.349
    else:
        center, scale = x.mean(axis=0), x.std(axis=0)
    references = dict(center=center, scale=np.where(scale < FLOOR, 1., scale),
                      constant=scale < FLOOR)
    if kind == 'asset':
        ids = np.unique(assets[fit])
        refs = []
        for a in ids:
            z = np.asarray(h[fit & (assets == a)], np.float64)
            sd = z.std(axis=0)
            refs.append((z.mean(axis=0), np.where(sd < FLOOR, 1., sd), sd < FLOOR))
        references.update(asset_ids=ids, asset_center=np.stack([r[0] for r in refs]),
                          asset_scale=np.stack([r[1] for r in refs]),
                          asset_constant=np.stack([r[2] for r in refs]))
    return references


def normalize(h, assets, dates, valid, fit, kind, window=126):
    """Flat row-indexed H; only cutoff-available vectors can enter a snapshot/history."""
    out = np.full(h.shape, np.nan, np.float32)
    refs = {}
    if kind in ['pooled', 'asset', 'robust']:
        refs = train_standardization(h, assets, fit & valid, kind)
        if kind == 'asset':
            for j, a in enumerate(refs['asset_ids']):
                m = valid & (assets == a)
                z = (h[m]-refs['asset_center'][j])/refs['asset_scale'][j]
                z[:, refs['asset_constant'][j]] = 0
                out[m] = z
        else:
            z = (h[valid]-refs['center'])/refs['scale']
            z[:, refs['constant']] = 0
            out[valid] = z
    elif kind == 'cross_section':
        for d in np.unique(dates[valid]):
            m = valid & (dates == d)
            if m.sum() < 2:
                continue
            z = np.asarray(h[m], np.float64)
            sd = z.std(axis=0)
            z = (z-z.mean(axis=0))/np.where(sd < FLOOR, 1., sd)
            z[:, sd < FLOOR] = 0
            out[m] = z
    elif kind == 'rolling':
        for a in np.unique(assets):
            idx = np.flatnonzero(valid & (assets == a))
            # Stable, explicit chronological order, regardless of caller row order.
            idx = idx[np.argsort(dates[idx], kind='stable')]
            x = pd.DataFrame(np.asarray(h[idx], np.float64))
            mean = x.shift(1).rolling(window, min_periods=32).mean().to_numpy()
            sd = x.shift(1).rolling(window, min_periods=32).std(ddof=0).to_numpy()
            z = (x.to_numpy()-mean)/np.where(sd < FLOOR, 1., sd)
            z[sd < FLOOR] = 0
            out[idx] = z
        refs = dict(window=np.array(window), minimum=np.array(32), current_excluded=np.array(True))
    elif kind == 'l2':
        x = np.asarray(h[valid], np.float64)
        length = np.linalg.norm(x, axis=1)
        out[valid] = x/np.maximum(length[:, None], FLOOR)
    else:
        raise ValueError(kind)
    return out, refs


def common_history(assets, dates, valid, minimum=32):
    """Uniform eligible origins for every rolling window, no future history gate."""
    mask = np.zeros(len(valid), bool)
    for a in np.unique(assets):
        idx = np.flatnonzero(valid & (assets == a))
        idx = idx[np.argsort(dates[idx], kind='stable')]
        mask[idx[minimum:]] = True
    return mask


def distribution_design(h, dates, valid, kind):
    """A plus genuine leave-one-out distribution of the normalized tensor."""
    q = h.shape[1]
    # For CS Z, LOO mean/std are algebraic functions of own Z and date count;
    # remove those two blocks rather than present them as independent signals.
    blocks = ['p25', 'p50', 'p75'] if kind == 'cross_section' else ['mean', 'std', 'p25', 'p50', 'p75']
    out = np.full((len(h), q*(1+len(blocks))), np.nan, np.float32)
    for d in np.unique(dates[valid]):
        idx = np.flatnonzero(valid & (dates == d))
        if len(idx) < 2:
            continue
        x = np.asarray(h[idx], np.float64)
        n = len(x)
        mean = (x.sum(axis=0)[None, :]-x)/(n-1)
        variance = np.maximum((np.square(x).sum(axis=0)[None, :]-x*x)/(n-1)-mean*mean, 0)
        ordered = np.sort(x, axis=0)
        # For each excluded row, map a quantile of n-1 values back to the full
        # sorted list, with exact stable ranks even when values tie.
        ranks = rankdata(x, axis=0, method='ordinal').astype(int)-1
        quantiles = []
        for percentile in [.25, .5, .75]:
            position = (n-2)*percentile
            lo, hi = int(np.floor(position)), int(np.ceil(position))
            a = ordered[lo+(ranks <= lo), np.arange(q)[None, :]]
            b = ordered[hi+(ranks <= hi), np.arange(q)[None, :]]
            quantiles.append(a+(position-lo)*(b-a))
        values = [x] + ([] if kind == 'cross_section' else [mean, np.sqrt(variance)]) + quantiles
        out[idx] = np.concatenate(values, axis=1)
    return out, dict(blocks=['own']+blocks, removed=['LOO mean', 'LOO std'] if kind == 'cross_section' else [])


def drop_constants(x, fit):
    sd = np.asarray(x[fit], np.float64).std(axis=0)
    keep = sd > FLOOR
    return keep, dict(n_columns=len(sd), kept=int(keep.sum()), constant_columns=np.flatnonzero(~keep).tolist())


def dual_ridge(ktt, kqt, y, alphas=ALPHAS):
    """sum(residual^2)+alpha||beta||^2, unpenalized intercept; exact dual solve."""
    y = np.asarray(y, float)
    mean = ktt.mean(axis=0)
    grand = float(mean.mean())
    centered = ktt-mean[None, :]-mean[:, None]+grand
    query = kqt-mean[None, :]-kqt.mean(axis=1)[:, None]+grand
    eig, vectors = np.linalg.eigh((centered+centered.T)/2)
    weights = vectors @ ((vectors.T @ (y-y.mean()))[:, None]/
                         (np.maximum(eig, 0)[:, None]+np.asarray(alphas)[None, :]))
    predictions = query @ weights+y.mean()
    return predictions, dict(dual_weights=weights, kernel_train_mean=mean,
                             kernel_grand_mean=np.array(grand), intercept=np.array(y.mean()))


def concat_kernel(h, assets, steps, valid, vocabulary, n_dates):
    """Exactly one slot per training asset, absent blocks=0, no availability flags."""
    kernel = np.zeros((n_dates, n_dates), np.float64)
    for a in vocabulary:
        idx = np.flatnonzero(valid & (assets == a))
        x = np.asarray(h[idx], float)
        kernel[np.ix_(steps[idx], steps[idx])] += x @ x.T
    return kernel


def rank_ic(pred, target):
    if len(pred) < 2 or np.ptp(pred) == 0 or np.ptp(target) == 0:
        return 0.
    return float(np.corrcoef(rankdata(pred), rankdata(target))[0, 1])


def decile_weights(pred, side):
    """Stable ties, gross=1; long-only or 0.5 long / 0.5 short."""
    n = len(pred)
    if n < 10:
        return np.zeros(n)
    k = max(1, int(np.ceil(n*.1)))
    order = np.argsort(pred, kind='stable')
    w = np.zeros(n)
    w[order[-k:]] = (1. if side == 'long' else .5)/k
    if side == 'neutral':
        w[order[:k]] = -.5/k
    return w


def account_portfolio(target_weights, forward_returns, cost=.001):
    """Drift-aware rebalance turnover. Missing held returns are flagged, never zero.

    Each row is one execution-to-next-execution holding interval. After an
    unaccountable interval, restart a separately labeled accounting segment;
    no continuous wealth/Sharpe claim crosses the missing valuation.
    """
    previous = np.zeros(target_weights.shape[1])
    output = []
    segment = 0
    for w, r in zip(target_weights, forward_returns):
        traded = float(np.abs(w-previous).sum())
        fee = cost*traded
        accountable = np.isfinite(r[w != 0]).all()
        if accountable:
            gross = float(w[w != 0] @ r[w != 0])
            net = gross-fee
            growth = 1+net
            if growth <= 0:
                previous = np.zeros_like(w)
                segment += 1
            else:
                previous = np.where(w != 0, w*(1+np.nan_to_num(r, nan=0)), 0)/growth
        else:
            gross, net = np.nan, np.nan
            previous = np.zeros_like(w)
        output.append(dict(gross=gross, net=net, turnover=.5*traded, cost=fee,
                           accountable=bool(accountable), segment=segment))
        if not accountable:
            segment += 1
    if len(output):
        exit_fee = cost*float(np.abs(previous).sum())
        if output[-1]['accountable']:
            exit_fee *= max(0.,1+output[-1]['net'])
        output[-1]['cost'] += exit_fee
        if output[-1]['accountable']:
            output[-1]['net'] -= exit_fee
        output[-1]['turnover'] += .5*float(np.abs(previous).sum())
    return pd.DataFrame(output)
