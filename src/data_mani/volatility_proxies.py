"""Causal daily-bar volatility estimates and own-history transformations."""

import numpy as np
import pandas as pd
from numba import njit

WINDOWS = (5, 10, 15, 20, 30, 40, 50, 60, 90, 120, 240)
FAMILIES = ("ewma", "semivariance", "idiosyncratic", "parkinson",
            "garman_klass", "rogers_satchell", "yang_zhang")
VIEWS = ("raw", "history_percentile", "history_bins")
BIN_CUTS = (.05, .25, .75, .95)


def bins(percentile):
    values = np.asarray(percentile)
    return np.where(np.isfinite(values), np.searchsorted(BIN_CUTS, values) / 4, np.nan)


@njit
def history_percentile(values):
    """Mid-percentile against 126+ finite values in the previous 252 bars."""
    result = np.full(values.shape, np.nan, dtype=np.float32)
    for i in range(126, len(values)):
        for j in range(values.shape[1]):
            current = values[i, j]
            if not np.isfinite(current):
                continue
            valid = 0
            below = 0
            equal = 0
            for k in range(max(0, i - 252), i):
                prior = values[k, j]
                if np.isfinite(prior):
                    valid += 1
                    below += prior < current
                    equal += prior == current
            if valid >= 126:
                result[i, j] = (below + .5 * equal) / valid
    return result


@njit
def _exp_average(values, decay, warmup, recursive):
    result = np.full(len(values), np.nan)
    count = 0
    numerator = 0.0
    denominator = 0.0
    for i, value in enumerate(values):
        if not np.isfinite(value):
            count = 0
            numerator = 0.0
            denominator = 0.0
            continue
        if recursive:
            numerator = value if count == 0 else decay * numerator + (1 - decay) * value
            denominator = 1.0
        else:
            numerator = decay * numerator + value
            denominator = decay * denominator + 1
        count += 1
        if count >= warmup:
            result[i] = numerator / denominator
    return result


def exp_average(values, decay, warmup, recursive):
    return pd.Series(_exp_average(values.to_numpy(), decay, warmup, recursive),
                     index=values.index)


def residual_variance(returns, factors, window):
    """Rolling OLS variance with intercept and residual degrees of freedom."""
    values = np.column_stack([returns, *factors])
    values[~np.isfinite(values).all(axis=1)] = np.nan
    keys = np.triu_indices(values.shape[1])
    means = pd.DataFrame(values).rolling(window).mean().to_numpy()
    products = values[:, keys[0]] * values[:, keys[1]]
    moments = pd.DataFrame(products).rolling(window).mean().to_numpy()
    covariance = moments - means[:, keys[0]] * means[:, keys[1]]
    centered = np.empty((len(values), values.shape[1], values.shape[1]))
    centered[:, keys[0], keys[1]] = covariance
    centered[:, keys[1], keys[0]] = covariance
    factor_cov = centered[:, 1:, 1:]
    finite = np.isfinite(factor_cov).all(axis=(1, 2))
    usable = finite.copy()
    usable[finite] = np.linalg.eigvalsh(factor_cov[finite])[:, 0] > 1e-12
    factor_target = centered[usable, 0, 1:]
    beta = np.linalg.solve(factor_cov[usable], factor_target[..., None])[..., 0]
    result = np.full(len(returns), np.nan)
    result[usable] = np.maximum(
        centered[usable, 0, 0] - np.sum(beta * factor_target, axis=1), 0
    ) * window / (window - values.shape[1])
    return result


def catalog(family):
    if family == "ewma":
        methods = {
            "recursive_log": ("ewma",), "normalized_log": ("ewma",),
            "recursive_simple": ("ewma",), "riskmetrics_094": ("ewma",),
        }
    elif family == "semivariance":
        methods = {
            "zero_log": ("down", "up", "joint"),
            "zero_simple": ("down", "up", "joint"),
            "window_mean_log": ("down", "up", "joint"),
            "side_count_log": ("down", "up", "joint"),
        }
    elif family == "idiosyncratic":
        methods = {method: ("residual",) for method in ("btc", "btc_eth", "eth")}
    elif family == "parkinson":
        methods = {method: ("range",) for method in ("standard", "ewma")}
    elif family == "garman_klass":
        methods = {method: ("range",) for method in ("standard", "four_term", "gap_added")}
    elif family == "rogers_satchell":
        methods = {method: ("range",) for method in ("standard", "gap_added", "ewma")}
    else:
        methods = {method: ("range",) for method in ("standard", "no_gap", "gap_rms")}
    return [
        (calendar, method, window, side)
        for calendar in ("daily", "weekday")
        for method, sides in methods.items()
        for window in ((20,) if method == "riskmetrics_094" else WINDOWS)
        for side in sides
    ]


@njit
def semivar_mean(values, window):
    down = np.full(len(values), np.nan)
    up = down.copy()
    for i in range(window - 1, len(values)):
        trailing = values[i - window + 1:i + 1]
        if np.isfinite(trailing).all():
            deviations = trailing - trailing.mean()
            down[i] = np.minimum(deviations, 0.0).dot(np.minimum(deviations, 0.0)) / window
            up[i] = np.maximum(deviations, 0.0).dot(np.maximum(deviations, 0.0)) / window
    return down, up


def estimate(bars, benchmark, family, method, window):
    """Return one or two variance series on a complete selected bar grid."""
    p = np.log(bars)
    ret = p.close.diff()
    if family == "ewma":
        if method == "riskmetrics_094":
            decay = .94
        else:
            decay = .5 ** (1 / window)
        if method == "recursive_simple":
            ret = bars.close.pct_change(fill_method=None)
        return (exp_average(ret ** 2, decay, window, method != "normalized_log"),)
    if family == "semivariance":
        if method == "window_mean_log":
            return semivar_mean(ret.to_numpy(), window)
        if method == "zero_simple":
            ret = bars.close.pct_change(fill_method=None)
        down = ret.clip(upper=0) ** 2
        up = ret.clip(lower=0) ** 2
        if method == "side_count_log":
            num_down = (ret < 0).where(ret.notna()).rolling(window).sum()
            num_up = (ret > 0).where(ret.notna()).rolling(window).sum()
            return tuple(
                (value.rolling(window).sum() / count.where(count > 0)).where(count.ne(0), 0)
                for value, count in ((down, num_down), (up, num_up))
            )
        return down.rolling(window).mean(), up.rolling(window).mean()
    if family == "idiosyncratic":
        factors = {"btc": [1], "eth": [1027], "btc_eth": [1, 1027]}[method]
        return (residual_variance(ret, [benchmark[ucid] for ucid in factors], window),)

    high_low = p.high - p.low
    high_open = p.high - p.open
    low_open = p.low - p.open
    intraday = p.close - p.open
    gap = p.open - p.close.shift()
    rs = high_open * (p.high - p.close) + low_open * (p.low - p.close)
    if family == "yang_zhang":
        k = .34 / (1.34 + (window + 1) / (window - 1))
        variance = k * intraday.rolling(window).var() + (1 - k) * rs.rolling(window).mean()
        if method == "standard":
            variance += gap.rolling(window).var()
        elif method == "gap_rms":
            variance += (gap ** 2).rolling(window).mean()
        return (variance,)
    if family == "parkinson":
        component = high_low ** 2 / (4 * np.log(2))
    elif family == "garman_klass":
        if method == "four_term":
            component = (.511 * high_low ** 2
                         - .019 * (intraday * (high_open + low_open) - 2 * high_open * low_open)
                         - .383 * intraday ** 2)
        else:
            component = .5 * high_low ** 2 - (2 * np.log(2) - 1) * intraday ** 2
    else:
        component = rs
    if method == "gap_added":
        component += gap ** 2
    if method == "ewma":
        return (exp_average(component, .5 ** (1 / window), window, False),)
    return (component.rolling(window).mean(),)


def clean_bars(candles):
    bars = candles.set_index("bar_date")[["open", "high", "low", "close"]].sort_index()
    valid = np.isfinite(bars) & (bars > 0)
    bars = bars.where(valid)
    valid_range = (bars.low <= bars.high) & (bars.low <= bars.open) & (
        bars.low <= bars.close) & (bars.high >= bars.open) & (bars.high >= bars.close)
    bars[["open", "high", "low"]] = bars[["open", "high", "low"]].where(valid_range)
    return bars


def build_asset(panel_dates, source, benchmark, family, configs):
    """Align the most recent completed bar at or before D-1 without filling gaps."""
    output = {}
    for calendar in ("daily", "weekday"):
        freq = "D" if calendar == "daily" else "B"
        cutoff = panel_dates - pd.Timedelta(days=1)
        if calendar == "weekday":
            cutoff = pd.DatetimeIndex([pd.offsets.BDay().rollback(day) for day in cutoff])
        grid = pd.date_range(panel_dates.min() - pd.Timedelta(days=800),
                             cutoff.max(), freq=freq)
        bars = source.reindex(grid)
        market = {ucid: np.log(series.reindex(grid)).diff() for ucid, series in benchmark.items()}
        for _, method, window in dict.fromkeys(c[:3] for c in configs if c[0] == calendar):
            values = np.column_stack(estimate(bars, market, family, method, window))
            values = np.maximum(values, 0)
            histories = history_percentile(values)
            aligned = grid.get_indexer(cutoff)
            output[calendar, method, window] = (values[aligned], histories[aligned])
    return output
