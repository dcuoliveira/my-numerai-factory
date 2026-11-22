"""Causal momentum signals and independent, batched asset regressions for E205.

Batching is a computational device: every group has its own scaler, loss,
parameters, Adam moments, stopping time and query predictions. No gradient or
statistic is averaged across assets. Identity networks are evaluated through
their algebraically equivalent affine composition to avoid repeating the same
matrix products for each of the <=130 observations of an asset.
"""
import numpy as np
import pandas as pd
from numba import njit

FAMILIES = [
    'absolute_price_change', 'price_minus_sma', 'price_minus_ema',
    'absolute_price_slope', 'cumulative_return', 'mean_simple_return',
    'price_sma_ratio', 'price_ema_ratio', 'log_price_slope',
    'relative_price_slope', 'ewma_return', 'risk_adjusted_return',
]
WINDOWS = [5, 10, 15, 20, 30, 40, 50, 60, 90, 120, 240]


def signal(close, family, window):
    """E163 formula definitions; close already contains weekday observations."""
    roll = close.rolling(window, min_periods=window)
    mean = roll.mean()
    ret = close.pct_change(fill_method=None)
    if family == 'absolute_price_change':
        out = close - close.shift(window)
    elif family == 'price_minus_sma':
        out = close - mean
    elif family == 'price_minus_ema':
        out = close - close.ewm(span=window, adjust=False, min_periods=window).mean()
    elif family == 'cumulative_return':
        out = close / close.shift(window) - 1
    elif family == 'mean_simple_return':
        out = ret.rolling(window, min_periods=window).mean()
    elif family == 'price_sma_ratio':
        out = close / mean - 1
    elif family == 'price_ema_ratio':
        out = close / close.ewm(span=window, adjust=False, min_periods=window).mean() - 1
    elif family in ('absolute_price_slope', 'log_price_slope', 'relative_price_slope'):
        values = np.log(close) if family == 'log_price_slope' else close
        weights = np.arange(window) - (window - 1) / 2
        # Center absolute positions inside each window: the same E163 OLS
        # definition without millions of Python rolling-apply callbacks.
        position = np.arange(len(values), dtype=float)
        sums = values.rolling(window, min_periods=window).sum()
        weighted = values.mul(position, axis=0).rolling(window, min_periods=window).sum()
        out = (weighted-sums.mul(position-(window-1)/2, axis=0))/np.dot(weights, weights)
        if family == 'relative_price_slope':
            out = out / mean
    elif family == 'ewma_return':
        out = ret.ewm(span=window, adjust=False, min_periods=window).mean()
    elif family == 'risk_adjusted_return':
        r = ret.rolling(window, min_periods=window)
        out = r.mean() / r.std(ddof=0)
    else:
        raise ValueError(family)
    return out.replace([np.inf, -np.inf], np.nan).where(close.notna())


@njit(cache=True)
def prior_bins(values, history=252, minimum=126):
    """E163 strictly-prior own-history percentiles, then five bins."""
    out = np.full(values.shape, np.nan)
    for c in range(values.shape[1]):
        for t in range(values.shape[0]):
            v = values[t, c]
            if not np.isfinite(v):
                continue
            n, below, equal = 0, 0, 0
            for j in range(max(0, t-history), t):
                other = values[j, c]
                if np.isfinite(other):
                    n += 1
                    below += other < v
                    equal += other == v
            if n >= minimum:
                p = (below + .5 * equal) / n
                out[t, c] = .25 * (int(p > .05) + int(p > .25)
                                      + int(p > .75) + int(p > .95))
    return out


def attach(panel, raw, columns, lag=1):
    """Completed source bar <= D-lag, at most four calendar days stale."""
    dates = raw.index.to_numpy().astype('datetime64[D]')
    cutoff = panel.date.to_numpy().astype('datetime64[D]') - np.timedelta64(lag, 'D')
    row = np.searchsorted(dates, cutoff, side='right') - 1
    values = raw.to_numpy()
    stamps = np.where(np.isfinite(values), dates.astype('int64')[:, None], np.nan)
    stamps = pd.DataFrame(stamps).ffill(limit=4).to_numpy()
    filled = raw.ffill(limit=4).to_numpy()
    ok = (row >= 0) & (columns >= 0)
    out, age = np.full(len(panel), np.nan), np.full(len(panel), np.nan)
    out[ok] = filled[row[ok], columns[ok]]
    age[ok] = (panel.date.to_numpy().astype('datetime64[D]').astype('int64')[ok]
               - stamps[row[ok], columns[ok]])
    out[(age > 4) | (age < lag) | ~np.isfinite(age)] = np.nan
    return out


def price_context(panel, prices):
    """E157/E163 identity and OHLC quality cohort; no feature/target selection."""
    if prices.duplicated(['symbol', 'date']).any():
        raise ValueError('Duplicate source bars')
    identity = prices.groupby('symbol').agg(
        ucid=('ucid', 'first'), ids=('ucid', 'nunique'),
        sources=('source', 'nunique'), tickers=('provider_ticker', 'nunique'))
    supplied = panel.symbol.map(identity.ucid).astype(str)
    unique = (panel.symbol.map(panel.groupby('symbol').ucid.nunique()).eq(1)
              & panel.symbol.map(identity.ids).eq(1)
              & panel.symbol.map(identity.sources).eq(1)
              & panel.symbol.map(identity.tickers).eq(1)
              & supplied.eq(panel.ucid.astype(str))
              & ~supplied.str.contains(',', regex=False))
    days = pd.date_range(prices.date.min(), prices.date.max(), freq='D')
    wide = {c: prices.pivot(index='date', columns='symbol', values=c)
            .reindex(days).astype(float).where(lambda x: x > 0)
            for c in ['open', 'high', 'low', 'close']}
    good = (wide['open'].notna() & wide['high'].notna() & wide['low'].notna()
            & wide['close'].notna() & wide['high'].ge(wide['open'])
            & wide['high'].ge(wide['close']) & wide['low'].le(wide['open'])
            & wide['low'].le(wide['close']))
    cols = wide['close'].columns.get_indexer(panel.symbol)
    complete = good.rolling(130, min_periods=130).sum().eq(130).to_numpy()
    common = unique.to_numpy().copy()
    for lag in [3, 2, 1]:
        rows = days.get_indexer(panel.date - pd.Timedelta(days=lag))
        ok = (rows >= 0) & (cols >= 0)
        present = np.zeros(len(panel), bool)
        values = complete if lag == 3 else good.to_numpy()
        present[ok] = values[rows[ok], cols[ok]]
        common &= present
    close = wide['close'].where(good)
    return close.loc[close.index.dayofweek < 5], cols, common


def feature_array(x):
    x = np.asarray(x, float)
    return x[:, :, None] if x.ndim == 2 else x


def scale_groups(x, y):
    """Independent population-standardisation on finite training observations."""
    x = feature_array(x)
    mask = np.isfinite(x).all(axis=2) & np.isfinite(y)
    n = mask.sum(axis=1)
    if (n == 0).any():
        raise ValueError('Every fitted asset must have training observations')
    mean = np.where(mask[:, :, None], x, 0).sum(axis=1) / n[:, None]
    var = np.where(mask[:, :, None], (x-mean[:, None, :])**2, 0).sum(axis=1) / n[:, None]
    scale = np.sqrt(var)
    scale[scale < 1e-14] = 1
    z = np.where(mask[:, :, None], (x-mean[:, None, :])/scale[:, None, :], 0)
    return z, np.where(mask, y, 0), mask, n, mean, scale


def ridge_candidates(x, y, query, alphas):
    z, yy, mask, n, mean, scale = scale_groups(x, y)
    ym = yy.sum(axis=1) / n
    xy = np.einsum('gtf,gt->gf', z, np.where(mask, yy-ym[:, None], 0))
    gram = np.einsum('gtf,gth->gfh', z, z)
    eig, vec = np.linalg.eigh(gram)
    projected = np.einsum('gfi,gf->gi', vec, xy)
    coef = np.einsum('gfi,gic->gfc', vec, projected[:, :, None] /
                     (np.maximum(eig, 0)[:, :, None] + np.asarray(alphas)[None, None, :]))
    q = (feature_array(query)-mean[:, None, :])/scale[:, None, :]
    return np.einsum('gqf,gfc->gqc', q, coef) + ym[:, None, None]


def sklearn_weights(architecture, seed, n_features=1):
    """Exact sklearn MLPRegressor RandomState draw order, including biases."""
    rng = np.random.RandomState(seed)
    widths = [n_features] + list(architecture) + [1]
    result = []
    for a, b in zip(widths[:-1], widths[1:]):
        bound = np.sqrt(6/(a+b))
        result.append((rng.uniform(-bound, bound, (a, b)),
                       rng.uniform(-bound, bound, b)))
    return result


def torch_candidates(x, y, query, candidates, seed=0, max_iter=200,
                     tol=1e-4, patience=10):
    """Independent asset x candidate sklearn-initialised Torch identity MLPs.

    Returns asset x query-time x candidate predictions and per-model epochs.
    Inputs contain NaN only for genuinely excluded rows, never neutral fill.
    All training rows fit in the registered batch_size=1024. Full-batch
    shuffling is algebraically immaterial. The custom vectorised Adam update
    is reference-tested against torch.optim.Adam and the scalar repository
    TorchMLPRegressor loop. Stopped groups' outputs are captured immediately.
    """
    import torch
    torch.set_num_threads(1)
    z, yy, mask, n, mean, scale = scale_groups(x, y)
    x, query = feature_array(x), feature_array(query)
    q = (query-mean[:, None, :])/scale[:, None, :]
    result = np.full((len(x), query.shape[1], len(candidates)), np.nan)
    epochs = np.zeros((len(x), len(candidates)), int)
    if x.shape[1] > 1024:
        raise ValueError('Full-batch equivalence requires at most 1024 dates')
    for arch in sorted({tuple(p['hidden_layer_sizes']) for p in candidates}):
        indices = [i for i, p in enumerate(candidates) if tuple(p['hidden_layer_sizes']) == arch]
        if any(candidates[i].get('activation', 'identity') != 'identity' for i in indices):
            raise ValueError('Only the registered E163 identity activation is supported')
        ng, nc = len(x), len(indices)
        # Candidate-major order; each asset/candidate has distinct parameter storage.
        xx = torch.tensor(np.tile(z, (nc, 1, 1)), dtype=torch.float64)
        target = torch.tensor(np.tile(yy, (nc, 1)), dtype=torch.float64)
        valid = torch.tensor(np.tile(mask, (nc, 1)))
        counts = torch.tensor(np.tile(n, nc), dtype=torch.float64)
        qq = torch.tensor(np.tile(np.nan_to_num(q, nan=0), (nc, 1, 1)), dtype=torch.float64)
        lr = torch.tensor(np.repeat([candidates[i]['learning_rate_init'] for i in indices], ng))
        alpha = torch.tensor(np.repeat([candidates[i]['alpha'] for i in indices], ng))
        params, weights = [], []
        for w, b in sklearn_weights(arch, seed, x.shape[2]):
            wt = torch.tensor(np.broadcast_to(w, (nc*ng, *w.shape)).copy(), requires_grad=True)
            bt = torch.tensor(np.broadcast_to(b, (nc*ng, *b.shape)).copy(), requires_grad=True)
            params.extend([wt, bt]); weights.append(wt)
        moments = [torch.zeros_like(p) for p in params]
        variances = [torch.zeros_like(p) for p in params]
        active = torch.ones(nc*ng, dtype=torch.bool)
        best = torch.full((nc*ng,), float('inf'))
        stale = torch.zeros(nc*ng, dtype=torch.int64)
        captured = np.full((nc*ng, query.shape[1]), np.nan)
        group_epochs = np.zeros(nc*ng, int)

        def affine():
            slope, intercept = params[0], params[1][:, None, :]
            for j in range(2, len(params), 2):
                slope = torch.bmm(slope, params[j])
                intercept = torch.bmm(intercept, params[j]) + params[j+1][:, None, :]
            return slope[:, :, 0], intercept[:, 0, 0]

        for epoch in range(1, max_iter+1):
            slope, intercept = affine()
            prediction = (xx*slope[:, None, :]).sum(dim=2) + intercept[:, None]
            loss = (.5*torch.where(valid, (prediction-target)**2, 0).sum(dim=1)/counts
                    + .5*alpha*sum(w.square().sum(dim=(1, 2)) for w in weights)/counts)
            if not torch.isfinite(loss[active]).all():
                raise FloatingPointError('A registered Torch candidate diverged')
            (loss*active).sum().backward()
            # Adam with independent per-group learning rates and moments.
            with torch.no_grad():
                for p, m, v in zip(params, moments, variances):
                    grad = p.grad
                    m.mul_(.9).add_(grad, alpha=.1)
                    v.mul_(.999).addcmul_(grad, grad, value=.001)
                    rate = (lr*active).reshape((-1,)+(1,)*(p.ndim-1))
                    p.add_(-rate * (m/(1-.9**epoch)) / (torch.sqrt(v/(1-.999**epoch))+1e-8))
                    p.grad = None
                stale = torch.where(loss > best-tol, stale+1, 0)
                best = torch.minimum(best, loss)
                stop = active & ((stale > patience) | (epoch == max_iter))
                if stop.any():
                    slope, intercept = affine()
                    out = (qq*slope[:, None, :]).sum(dim=2) + intercept[:, None]
                    selected = stop.numpy()
                    captured[selected] = out.numpy()[selected]
                    group_epochs[selected] = epoch
                    active &= ~stop
            if not active.any():
                break
        for j, i in enumerate(indices):
            result[:, :, i] = captured[j*ng:(j+1)*ng]
            epochs[:, i] = group_epochs[j*ng:(j+1)*ng]
    result[~np.isfinite(query).all(axis=2), :] = np.nan
    return result, epochs


def route_slots(baseline_ranks, candidate):
    """Use live deterministic first-tie ordering; preserve uncovered ranks."""
    baseline = np.asarray(baseline_ranks, float)
    if not np.isfinite(baseline).all():
        raise ValueError('Missing full-universe baseline prediction')
    covered = np.isfinite(candidate)
    result = baseline.copy()
    rows = np.flatnonzero(covered)
    order = np.argsort(np.asarray(candidate)[rows], kind='stable')
    result[rows[order]] = np.sort(baseline[rows])
    assert np.array_equal(result[~covered], baseline[~covered])
    assert np.array_equal(np.sort(result[covered]), np.sort(baseline[covered]))
    return result
