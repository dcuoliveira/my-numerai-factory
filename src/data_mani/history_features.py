"""
Numerai-style own-history features for the Crypto panel, computed directly on the
supplied rank-binned columns and put back on their scale.

For every shipped feature and token, using only dates <= t:

    feature_hfe_<name>_rank<w>     own-history rank of today's value, last w days
    feature_hfe_<name>_ma<s>_<l>   mean over s days minus mean over l days
    feature_hfe_<name>_tsmax<w>    calendar days since the latest maximum, last w days
    feature_hfe_<name>_tschg       calendar days since the value last changed

Each is then rank-binned within its date, ties preserved, onto {0,.25,.5,.75,1}.

The inputs are ALREADY cross-sectional bins, not prices. "Maximum" and "change" mean
the supplied value: a token whose momentum bin stays 0.75 has not changed, however
much its price moved. No smoothing precedes the own-history rank.

Windows are CALENDAR windows ending at t: w days is (t-w days, t]. The panel has
weekdays only, so 7 days holds 5 panel dates and 3 days holds 1 to 3 (1 on a Monday).
The token keys off ucid, because tickers are reused: 31 symbols map to several coins.

Coverage: a window counts only if the token has a value on at least MIN_COVERAGE of
the weekdays in it (7d: 3 of 5, 14d: 6 of 10, 30d: 13-14 of 22, 90d: 39 of 64, 364d:
156 of 260), otherwise the output stays NaN. Nothing is backfilled.
tschg has no window: it is NaN until the token's first observed change, because the
duration before that is unknown. A change across a gap in the token's history is
dated to the first observation after the gap.

Own-history rank is (values below today + (values equal today + 1) / 2) / values in
the window, i.e. pandas' rank(method="average", pct=True) of today within the window.
"""

import numba
import numpy as np
import pandas as pd

from src.data_mani.binning import BIN_LABELS

RANK_WINDOWS = (7, 14, 30, 90, 364)
MA_PAIRS = ((3, 7), (7, 14), (14, 30), (30, 90), (91, 364))
TSMAX_WINDOWS = (7, 14, 30, 90, 364)
SHORT_WINDOWS = (7, 14, 30)                  # rank / tsmax split for the short arm
SHORT_PAIRS = ((3, 7), (7, 14), (14, 30))    # MA split for the short arm
MIN_COVERAGE = 0.6
PREFIX = "feature_hfe_"


def history_columns(feats):
    """Output names, grouped: {"rank": [...], "ma": [...], "tsmax": [...], "tschg": [...]}."""
    names = [f[len("feature_"):] for f in feats]
    return {"rank": ["%s%s_rank%d" % (PREFIX, n, w) for n in names for w in RANK_WINDOWS],
            "ma": ["%s%s_ma%d_%d" % (PREFIX, n, s, l) for n in names for s, l in MA_PAIRS],
            "tsmax": ["%s%s_tsmax%d" % (PREFIX, n, w) for n in names for w in TSMAX_WINDOWS],
            "tschg": ["%s%s_tschg" % (PREFIX, n) for n in names]}


def is_short(col):
    """True for a short-horizon column, False for a long one, None for tschg."""
    tail = col.rsplit("_", 1)[-1]
    if tail == "tschg":
        return None
    if tail.startswith("rank") or tail.startswith("tsmax"):
        return int(tail.replace("rank", "").replace("tsmax", "")) in SHORT_WINDOWS
    return (int(col.rsplit("_", 2)[-2].replace("ma", "")), int(tail)) in SHORT_PAIRS


@numba.njit(cache=True)
def _window_stats(x, days, start, need, rt, ru):
    """Own-history rank, mean and days-since-max for each row (t=rt[i], token=ru[i])
    over dates start[t]..t. x is dates x tokens; NaN where the token has no value."""
    n = len(rt)
    rank = np.full(n, np.nan)
    mean = np.full(n, np.nan)
    tsmax = np.full(n, np.nan)
    for i in range(n):
        t, u = rt[i], ru[i]
        cur = x[t, u]
        count, below, equal = 0, 0, 0
        total, top, top_day = 0.0, -np.inf, 0
        for k in range(start[t], t + 1):
            v = x[k, u]
            if v != v:
                continue
            count += 1
            total += v
            if v >= top:                     # >= keeps the MOST RECENT maximum
                top, top_day = v, days[k]
            if v < cur:
                below += 1
            elif v == cur:
                equal += 1
        if count < need[t]:
            continue
        mean[i] = total / count
        if cur == cur:
            rank[i] = (below + (equal + 1) / 2.0) / count
            tsmax[i] = days[t] - top_day
    return rank, mean, tsmax


@numba.njit(cache=True)
def _time_since_change(x, days, rt, ru):
    """Days since the token's last observed change. rows must be sorted by (token, t)."""
    n = len(rt)
    out = np.full(n, np.nan)
    token, prev, last = -1, np.nan, -1
    for i in range(n):
        t, u = rt[i], ru[i]
        if u != token:
            token, prev, last = u, np.nan, -1
        v = x[t, u]
        if v != v:
            continue
        if prev == prev and v != prev:
            last = days[t]
        prev = v
        if last >= 0:
            out[i] = days[t] - last
    return out


def compute_history_signals(panel, feats, id_col="ucid", period_col="date"):
    """
    The raw (un-binned) derived signals, one column per history_columns name.

    Only rows present in ``panel`` are scored, and a window reads only that token's
    rows dated <= t, so a row depends on nothing after its own date.

    :param panel: long panel, one row per (id_col, period_col)
    :param feats: the shipped feature columns
    :return: frame aligned to ``panel.index``
    :rtype: pd.DataFrame
    """
    if panel.duplicated([id_col, period_col]).any():
        raise ValueError("panel has duplicate (%s, %s) rows" % (id_col, period_col))
    dates = pd.DatetimeIndex(pd.to_datetime(panel[period_col]))
    grid = np.array(sorted(dates.unique()), dtype="datetime64[D]")
    days = grid.astype("int64")
    rt = np.searchsorted(grid, dates.to_numpy().astype("datetime64[D]"))
    tokens, ru = np.unique(panel[id_col].astype(str).to_numpy(), return_inverse=True)
    ru = ru.astype("int64")

    windows = sorted(set(RANK_WINDOWS) | set(TSMAX_WINDOWS) | {w for p in MA_PAIRS for w in p})
    start, need = {}, {}
    for w in windows:
        start[w] = np.searchsorted(days, days - w + 1, side="left")
        # weekdays in the FULL calendar window, not panel dates so far: a window
        # clipped by the start of the panel must not pass on a handful of values
        need[w] = np.maximum(1, np.ceil(MIN_COVERAGE * np.busday_count(
            grid - np.timedelta64(w - 1, "D"), grid + np.timedelta64(1, "D")))).astype("int64")

    by_token = np.lexsort((rt, ru))
    out = {}
    for f in feats:
        name = PREFIX + f[len("feature_"):]
        x = np.full((len(grid), len(tokens)), np.nan)
        x[rt, ru] = panel[f].to_numpy(dtype="float64")
        ranks, means, sinces = {}, {}, {}
        for w in windows:
            ranks[w], means[w], sinces[w] = _window_stats(x, days, start[w], need[w], rt, ru)
        for w in RANK_WINDOWS:
            out["%s_rank%d" % (name, w)] = ranks[w]
        for s, l in MA_PAIRS:
            # rounded so two equal differences cannot rank apart on float noise
            out["%s_ma%d_%d" % (name, s, l)] = np.round(means[s] - means[l], 10)
        for w in TSMAX_WINDOWS:
            out["%s_tsmax%d" % (name, w)] = sinces[w]
        since = np.full(len(rt), np.nan)
        since[by_token] = _time_since_change(x, days, rt[by_token], ru[by_token])
        out["%s_tschg" % name] = since
    return pd.DataFrame(out, index=panel.index)


def bin_ties_preserved(signals, periods):
    """
    Numerai's bin rule with ties preserved. A row's position is (rank-1)/(n-1) and the
    cuts are inclusive at .05/.25/.75/.95. That reproduces every shipped feature on
    every date exactly (38,720 of 38,720 cross-sections), including dates below 300
    tokens, where binning.rank_to_bins is a few rows off: it is only exact at n=300.
    Ties take their average rank, so a tie group shares one bin and heavily tied
    columns drift from 5/20/50/20/5. The cuts stay in integers so no float can move
    a row across one. A lone value has no rank, and takes the neutral 0.5.

    :param signals: frame of raw signals
    :param periods: the date of each row, aligned to ``signals``
    :return: float32 frame of bins, NaN where the signal was NaN
    :rtype: pd.DataFrame
    """
    grouped = signals.groupby(np.asarray(periods))
    ranks = grouped.rank(method="average").to_numpy()
    counts = signals.notna().groupby(np.asarray(periods)).transform("sum").to_numpy()
    r, n = ranks - 1.0, counts - 1.0
    idx = ((20 * r > n).astype("int8") + (4 * r > n).astype("int8")
           + (4 * r > 3 * n).astype("int8") + (20 * r > 19 * n).astype("int8"))
    binned = BIN_LABELS[idx].astype("float32")
    binned[counts == 1] = 0.5
    binned[np.isnan(ranks)] = np.nan
    return pd.DataFrame(binned, index=signals.index, columns=signals.columns)


def add_history_features(panel, feats, id_col="ucid", period_col="date"):
    """The one entry point both the builder and src/live/features.py call."""
    signals = compute_history_signals(panel, feats, id_col, period_col)
    return bin_ties_preserved(signals, panel[period_col])
