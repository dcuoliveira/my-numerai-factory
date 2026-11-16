"""
Reproduce Numerai's cross-sectional rank-binning, so a feature WE build lands on
the very same scale as the 22 features Numerai ships.

------------------------------------------------------------------------------
Why this module exists at all
------------------------------------------------------------------------------

Numerai does not ship raw signals. Every Crypto feature arrives already
rank-binned WITHIN each date to five values ``{0, .25, .5, .75, 1}``, with the
mass split fixed at **5 / 20 / 50 / 20 / 5 percent** -- the tails hold 5% each,
the middle bin holds half the field. This is verifiable on the shipped data:
``feature_momentum_20d`` over the whole panel is 5.01 / 20.0 / 50.0 / 19.99 /
5.01 percent, and on any single date with n symbols the counts are
``round(n * [.05,.20,.50,.20,.05])``.

The binning is a PROPERTY OF THE DOWNLOAD, computed on Numerai's servers -- the
repo never had to reproduce it because it never built a feature of its own from
raw prices. The moment we do (crypto momentum from spot OHLCV, and anything
after it), a new feature that is NOT on this scale would be a second, silently
different feature language mixed into the same row: a booster would split on the
raw magnitude, a ridge would see an un-normalised column, and the neutral-fill
convention (``fill_missing_features`` puts 0.5 in the gaps) would no longer mean
"middle of the distribution". So every derived feature we hand to a predictor
must pass through here first.

------------------------------------------------------------------------------
The rule
------------------------------------------------------------------------------

Within one date, rank the symbols by the raw signal, then hand the two tails
exactly ``round(0.05 n)`` symbols each and the two shoulders ``round(0.20 n)``
each, the rest to the middle bin:

    lowest  round(0.05 n)  -> 0.00
    next    round(0.20 n)  -> 0.25
    middle  the remainder  -> 0.50
    next    round(0.20 n)  -> 0.75
    top     round(0.05 n)  -> 1.00

For n = 300 that is exactly 15 / 60 / 150 / 60 / 15 -- the counts the shipped
``feature_momentum_20d`` shows on a 300-symbol date. Symmetric COUNTS (rather than
a one-sided fraction cut, which rounds the two tails differently, or ``qcut`` on
the raw values, which breaks on ties) keep the two 5% tails equal exactly as
Numerai's do (its pooled tails are 5.01% / 5.01%), and reproduce the masses
regardless of n.

A NaN raw signal -- a symbol with no price history yet, or no spot data at all --
stays NaN here and is filled with the neutral middle bin (0.5) downstream by
``fill_missing_features`` at load time, exactly as the shipped 60-day indicators
are NaN for tokens younger than 60 days.

------------------------------------------------------------------------------
Causality
------------------------------------------------------------------------------

The binning is CROSS-SECTIONAL and contemporaneous: the bin of symbol s at date
t depends only on the other symbols' signals AT t, never on any future date.
That is the same information set the shipped features and the scorer use (rank
within an era). It is the caller's job to ensure the RAW signal fed in is itself
backward-looking; this module adds no look-ahead of its own.
"""

import numpy as np
import pandas as pd

# The five bin values, low to high. This is the Crypto scale; Classic uses the
# integers 0..4, which is the same five buckets on a different label set.
BIN_LABELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

# The mass each bin must carry: 5 / 20 / 50 / 20 / 5 percent. Used to size the
# per-date bins by count, and to validate the result.
BIN_MASS = np.array([0.05, 0.20, 0.50, 0.20, 0.05])


def rank_to_bins(values, ties="first"):
    """
    Rank-bin one cross-section (the symbols on a single date) onto Numerai's
    ``{0,.25,.5,.75,1}`` scale.

    ``ties="first"`` (the default, and what every existing builder uses) breaks
    ties by order of appearance, so the masses are exact but two symbols with
    the SAME value can land in different bins by row order. ``ties="average"``
    gives every member of a tie group its average rank, so a tie group always
    shares one bin -- the one its centre falls in -- at the cost of masses that
    drift from 5/20/50/20/5 by the size of the groups straddling a boundary.
    Use it for signals that tie heavily (E130's tree-leaf quantiles: up to 74%
    of rows tied). Without ties the two are identical.

    :param values: raw signal for every symbol in one date; NaN allowed
    :type values: array-like or pd.Series
    :param ties: "first" or "average"
    :type ties: str
    :return: the binned signal, same length/index, NaN where the input was NaN
    :rtype: pd.Series
    """
    v = pd.Series(values).astype("float64")
    out = pd.Series(np.nan, index=v.index, dtype="float64")

    present = v.dropna()
    n = len(present)
    if n == 0:
        return out

    # Symmetric COUNTS: both tails get round(5% n), both shoulders round(20% n),
    # the middle gets the rest. This keeps the two tails exactly equal (a
    # one-sided fraction cut rounds them differently) and matches Numerai's own
    # symmetric split.
    n_tail = int(round(0.05 * n))
    n_side = int(round(0.20 * n))
    n_mid = n - 2 * n_tail - 2 * n_side  # >= 0: 2*.05+2*.20 = 0.5 < 1
    # cumulative boundaries on the 0-based rank position, one per gap between bins
    cum = np.array([n_tail, n_tail + n_side,
                    n_tail + n_side + n_mid,
                    n_tail + n_side + n_mid + n_side])

    # rank 1..n; method="first" breaks the (rare, for continuous signals) ties
    # deterministically by order of appearance so the result is reproducible.
    if ties not in ("first", "average"):
        raise ValueError("ties must be 'first' or 'average', got %r" % (ties,))
    order = present.rank(method=ties).to_numpy()
    idx = np.searchsorted(cum, order - 1.0, side="right")
    out.loc[present.index] = BIN_LABELS[idx]
    return out


def bin_cross_sectional(df, cols, period_col="date", ties="first"):
    """
    Rank-bin each named column WITHIN each period, in place on a copy.

    This is the production-shaping seam: the offline builder calls it over the
    whole panel at once, and the live path calls it with ``df`` restricted to a
    single round's cross-section -- same function, same scale, so a feature
    validated offline bins identically live.

    :param df: panel carrying ``period_col`` and every column in ``cols``
    :type df: pd.DataFrame
    :param cols: raw-signal columns to bin (overwritten with their binned form)
    :type cols: list[str]
    :param period_col: the column that identifies a cross-section (a date/era)
    :type period_col: str
    :param ties: passed to rank_to_bins -- "first" (default) or "average"
    :type ties: str
    :return: a copy of ``df`` with each ``cols`` column rank-binned per period
    :rtype: pd.DataFrame
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError("columns not in frame: %s" % missing)
    if period_col not in df.columns:
        raise KeyError("period column %r not in frame" % period_col)

    out = df.copy()
    grouped = out.groupby(period_col, sort=False)
    for col in cols:
        out[col] = grouped[col].transform(rank_to_bins, ties=ties).astype("float32")
    return out


def bin_mass(series):
    """
    The realised mass in each of the five bins -- a validation helper.

    :param series: a binned column (values drawn from ``BIN_LABELS``, NaN ok)
    :type series: pd.Series
    :return: fraction of non-NaN rows in each of the five bins, low to high
    :rtype: np.ndarray
    """
    v = pd.Series(series).dropna()
    if len(v) == 0:
        return np.full(5, np.nan)
    counts = np.array([(np.isclose(v, lab)).sum() for lab in BIN_LABELS], dtype="float64")
    return counts / counts.sum()
