"""Causal reversal / mean-reversion metrics for E219, one asset's own history.

Every function takes WIDE weekday frames (index = weekday bar date, one column
per symbol, invalid bars already NaN, as `asset_momentum.price_context`
returns them) and gives a frame of the same shape whose row t uses bars <= t
only. Swing pivots use a centred window but are shifted by their confirmation
lag, so a pivot at bar s enters at s+k. The only inputs shared across assets
are the BTC and BTC+ETH factor returns of the residual families (the E209
idiosyncratic convention); nothing is computed over the cross-section.

Signals keep their natural sign: the per-asset model learns the direction.
"""
import numpy as np
import pandas as pd

# Shorter than the momentum/vol grid (5..240): reversal lives at days-to-weeks.
WINDOWS = (2, 3, 4, 5, 7, 10, 14, 20, 30, 60, 120)
# Swing-pivot half-widths k (k bars each side), one per window slot.
SWING_K = (1, 2, 3, 4, 5, 7, 10, 14, 20, 30, 40)
BETA_WINDOW = 60      # factor-beta and volatility-scale lookback (weekday bars)
VOLUME_BASE = 60      # abnormal-volume baseline, strictly prior bars
TREND_WINDOW = 120    # drift used by trend_deviation, ending at t-w
SWING_SEARCH = None   # last confirmed pivot is carried until a newer one (no age cap)

FAMILIES = {
    # A raw short-term return
    'short_return': 'A: C_t/C_(t-w) - 1',
    # B residual reversal on the BTC / BTC+ETH factors
    'residual_btc': 'B: sum_w (r - beta_60 r_BTC)',
    'residual_btc_eth': 'B: sum_w residual of r on r_BTC, r_ETH (rolling 60 OLS)',
    # C volatility-scaled move
    'vol_scaled_return': 'C: log(C_t/C_(t-w)) / (sd_60(log r) sqrt(w))',
    # D anchors
    'stochastic_k': 'D: (C - min L_w) / (max H_w - min L_w)',
    'distance_high': 'D: log(C / max H_w)',
    'distance_low': 'D: log(C / min L_w)',
    # E volume-conditioned
    'low_volume_return': 'E: sum_w log r * 1[abnormal log volume <= 0]',
    'high_volume_return': 'E: sum_w log r * 1[abnormal log volume > 0]',
    # F extremes
    'max_return': 'F: max daily simple return over w',
    'min_return': 'F: min daily simple return over w',
    # G pullback within trend
    'trend_deviation': 'G: log(C_t/C_(t-w)) - w * mean_120 dlog C ending t-w',
    # RSI (E166 family, short windows)
    'rsi_wilder': 'RSI: Wilder RMA(1/w) of price changes',
    'rsi_sma': 'RSI: SMA_w of price changes (Numerai shipped form, E166)',
    # Pivots
    'pivot_classic': 'P1: (C_t - P)/ATR_w, P=(H+L+C_(t-1))/3 over bars t-w..t-1',
    'swing_high_distance': 'P2: log(C / last confirmed swing high), half-width k',
    'swing_low_distance': 'P2: log(C / last confirmed swing low), half-width k',
}


def _clean(out, close):
    return out.replace([np.inf, -np.inf], np.nan).where(close.notna())


def _factor_returns(factor, index):
    return factor.reindex(index).pct_change(fill_method=None)


def _residual(ret, factors, window, w):
    """Cumulative w-bar residual of each column on 1 or 2 factor returns.

    Rolling OLS with intercept over `window` bars ending t (pairwise-complete:
    a bar enters only where the asset and every factor are finite).
    """
    ok = ret.notna()
    for f in factors:
        ok &= f.notna().to_numpy()[:, None]
    y = ret.where(ok)
    xs = [pd.DataFrame(np.repeat(f.to_numpy()[:, None], ret.shape[1], 1),
                       index=ret.index, columns=ret.columns).where(ok) for f in factors]
    m = lambda frame: frame.rolling(window, min_periods=window).mean()
    my = m(y)
    mx = [m(x) for x in xs]
    if len(xs) == 1:
        cov = m(xs[0]*y) - mx[0]*my
        var = m(xs[0]*xs[0]) - mx[0]**2
        betas = [cov/var]
    else:
        s11 = m(xs[0]*xs[0]) - mx[0]**2
        s22 = m(xs[1]*xs[1]) - mx[1]**2
        s12 = m(xs[0]*xs[1]) - mx[0]*mx[1]
        s1y = m(xs[0]*y) - mx[0]*my
        s2y = m(xs[1]*y) - mx[1]*my
        det = s11*s22 - s12**2
        betas = [(s22*s1y - s12*s2y)/det, (s11*s2y - s12*s1y)/det]
    # Beta at t is applied to the w returns ending t (all inside the estimation window).
    total = y.rolling(w, min_periods=w).sum()
    for b, x in zip(betas, xs):
        total = total - b * x.rolling(w, min_periods=w).sum()
    return total


def _swing_distance(close, extreme, k, high):
    """log(C_t / most recent pivot confirmed by t). Pivot at s needs the full
    2k+1 window finite and is usable from s+k on."""
    span = 2*k + 1
    roll = extreme.rolling(span, center=True, min_periods=span)
    ref = roll.max() if high else roll.min()
    pivot = extreme.where(extreme.eq(ref) & ref.notna())
    known = pivot.shift(k).ffill(limit=SWING_SEARCH)
    return np.log(close/known)


def signal(bars, family, slot, factors):
    """bars: dict of wide weekday frames open/high/low/close/volume.
    slot: index into WINDOWS (and SWING_K). factors: {'btc': Series, 'eth': Series}
    of weekday closes."""
    close, high, low = bars['close'], bars['high'], bars['low']
    w = WINDOWS[slot]
    ret = close.pct_change(fill_method=None)
    logret = np.log(close).diff()
    roll = lambda frame, n=w: frame.rolling(n, min_periods=n)
    if family == 'short_return':
        out = close/close.shift(w) - 1
    elif family == 'residual_btc':
        out = _residual(ret, [_factor_returns(factors['btc'], close.index)], BETA_WINDOW, w)
    elif family == 'residual_btc_eth':
        out = _residual(ret, [_factor_returns(factors['btc'], close.index),
                              _factor_returns(factors['eth'], close.index)], BETA_WINDOW, w)
    elif family == 'vol_scaled_return':
        out = np.log(close/close.shift(w)) / (roll(logret, BETA_WINDOW).std(ddof=0)*np.sqrt(w))
    elif family == 'stochastic_k':
        lo, hi = roll(low).min(), roll(high).max()
        out = (close - lo)/(hi - lo)
    elif family == 'distance_high':
        out = np.log(close/roll(high).max())
    elif family == 'distance_low':
        out = np.log(close/roll(low).min())
    elif family in ('low_volume_return', 'high_volume_return'):
        lv = np.log(bars['volume'])
        abnormal = lv - roll(lv, VOLUME_BASE).mean().shift(1)
        part = abnormal.le(0) if family == 'low_volume_return' else abnormal.gt(0)
        out = roll(logret.where(abnormal.notna()) * part).sum()
    elif family == 'max_return':
        out = roll(ret).max()
    elif family == 'min_return':
        out = roll(ret).min()
    elif family == 'trend_deviation':
        drift = roll(logret, TREND_WINDOW).mean().shift(w)
        out = np.log(close/close.shift(w)) - w*drift
    elif family in ('rsi_wilder', 'rsi_sma'):
        change = close.diff()
        gain, loss = change.clip(lower=0), (-change).clip(lower=0)
        if family == 'rsi_wilder':
            g = gain.ewm(alpha=1/w, adjust=False, min_periods=w).mean()
            l = loss.ewm(alpha=1/w, adjust=False, min_periods=w).mean()
        else:
            g, l = roll(gain).mean(), roll(loss).mean()
        out = 100*g/(g + l)
        out = out.where((g + l) > 0, 50.0).where(g.notna() & l.notna())
    elif family == 'pivot_classic':
        prev_close = close.shift(1)
        hi, lo = roll(high.shift(1)).max(), roll(low.shift(1)).min()
        pivot = (hi + lo + prev_close)/3
        true_range = np.fmax(np.fmax(high - low, (high - prev_close).abs()),
                             (low - prev_close).abs()).where(high.notna() & low.notna() & prev_close.notna())
        atr = roll(true_range.shift(1)).mean()
        out = (close - pivot)/atr
    elif family == 'swing_high_distance':
        out = _swing_distance(close, high, SWING_K[slot], True)
    elif family == 'swing_low_distance':
        out = _swing_distance(close, low, SWING_K[slot], False)
    else:
        raise ValueError(family)
    return _clean(out, close)


def wide_bars(prices):
    """Weekday wide OHLCV with the E163 bar-validity rule (asset_momentum.price_context)."""
    days = pd.date_range(prices.date.min(), prices.date.max(), freq='D')
    wide = {c: prices.pivot(index='date', columns='symbol', values=c)
            .reindex(days).astype(float).where(lambda x: x > 0)
            for c in ['open', 'high', 'low', 'close', 'volume']}
    good = (wide['open'].notna() & wide['high'].notna() & wide['low'].notna()
            & wide['close'].notna() & wide['high'].ge(wide['open'])
            & wide['high'].ge(wide['close']) & wide['low'].le(wide['open'])
            & wide['low'].le(wide['close']))
    weekday = days.dayofweek < 5
    return {c: v.where(good).loc[weekday] for c, v in wide.items()}
