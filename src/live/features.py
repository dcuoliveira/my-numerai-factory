"""
Build the live feature sets, using the same code paths as the backtest.

    python3 -m src.live.features --feature-set onehot_roll   # inspect a build

The single largest source of live/backtest skew is having two implementations of
one feature. So this module does not reimplement anything: it imports
`causal_target_encoding` from src.data_mani.build_target_encoding and mirrors the
rolling/one-hot construction column-for-column, then builds TRAIN AND LIVE ROWS
IN ONE PASS over a concatenated panel. Whatever transformation the training rows
get, the live row got from the same call on the same frame.

------------------------------------------------------------------------------
The three sets
------------------------------------------------------------------------------

  raw           the 22 shipped features, straight from live.parquet.
                Nothing to build, nothing to break.

  tenc          raw + feature_sym_tenc -- each symbol's mean past target,
                shrunk toward the global mean with a pseudo-count of 50, and
                lagged 20 periods because a 20-day forward target is not
                observable until then. That lag is the whole ballgame: without
                it the encoding scores 0.2397 in-sample against 0.0557 causal.

  onehot_roll   raw + 740 feature_sym_<SYMBOL> identity columns + 24 rolling
                means (4 indicators x 2 horizons x windows 5/20/60).

------------------------------------------------------------------------------
The history gap, and why there is a local store
------------------------------------------------------------------------------

train.parquet only contains rows whose 20-day target has resolved, so it lags
the live file by roughly 20-40 days (as of writing: train ends 2026-07-08, live
is 2026-08-12). For `raw` and `tenc` that is harmless -- neither looks at recent
features, and the encoding is lagged 20 periods anyway.

For the rolling means it is not harmless. A 5-day mean computed on a panel that
jumps from 2026-07-08 straight to 2026-08-12 is not a 5-day mean; it is a stale
level with a hole in it. So every round's live snapshot is appended to

    data/crypto/live/live_history.parquet

and the rolling windows are computed over train + that store + today's rows. The
hole closes from both sides: the store fills forward from today, and Numerai's
train.parquet advances as targets resolve. Until they meet, CURUPIRA_03's
rolling columns are computed over a panel with a gap, which is a real (small,
shrinking) degradation and is recorded in the round report rather than hidden.
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.data_mani.build_derived import build_interactions, build_spreads
from src.data_mani.build_target_encoding import causal_target_encoding
from src.data_mani.utils import fill_missing_features

TRAIN_PARQUET = "data/crypto/v2.0/train.parquet"
LIVE_DIR = "data/crypto/live"
LIVE_PARQUET = os.path.join(LIVE_DIR, "live.parquet")
LIVE_HISTORY = os.path.join(LIVE_DIR, "live_history.parquet")

TARGET = "target_binned_return_20"
SYMBOL = "symbol"
PERIOD = "date"
NEUTRAL = 0.5                 # middle bin of {0, .25, .5, .75, 1}

# Matches build_target_encoding's defaults and the 20-day target horizon.
TENC_SMOOTHING = 50.0
TENC_LAG = 20

# Matches the rollsel arm: only the four indicators that are not already heavily
# smoothed by construction. Averaging an EWA is a second smoothing and adds
# nothing -- close_ewa's 20d and 60d versions already correlate 0.981.
ROLL_FEATURES = ["feature_sharpe_ratio", "feature_relative_strength_index",
                 "feature_volatility", "feature_momentum"]
ROLL_HORIZONS = ["20d", "60d"]
ROLL_WINDOWS = [5, 20, 60]

# A symbol needs this many dates before it earns an identity column, matching
# build_onehot --min-dates. Below it there is not enough history to fit a
# coefficient, and the column would be noise with a name.
ONEHOT_MIN_DATES = 130

FEATURE_SETS = ("raw", "tenc", "interact", "onehot_roll", "allfeat", "allfeat_pp")


def raw_features(frame):
    """The shipped feature columns, in a stable order."""
    return sorted(c for c in frame.columns
                  if c.startswith("feature_") and not c.startswith("feature_sym_")
                  and "_rm" not in c)


def append_live_history(live, path=LIVE_HISTORY):
    """
    Persist this round's live rows so future rounds can compute rolling windows
    across the gap between train.parquet and today.

    Idempotent: re-running a round replaces that date's rows rather than
    duplicating them, so a retry never double-counts a date.
    """
    live = live.reset_index()[[SYMBOL, PERIOD] + raw_features(live)]
    live[PERIOD] = pd.to_datetime(live[PERIOD])

    if os.path.exists(path):
        history = pd.read_parquet(path)
        history[PERIOD] = pd.to_datetime(history[PERIOD])
        history = history[~history[PERIOD].isin(live[PERIOD].unique())]
        history = pd.concat([history, live], ignore_index=True)
    else:
        history = live

    history = history.sort_values([SYMBOL, PERIOD]).reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    history.to_parquet(path, index=False)
    return history


def load_panel(train_path=TRAIN_PARQUET, live_path=LIVE_PARQUET,
               history_path=LIVE_HISTORY, verbose=True):
    """
    Concatenate train, the accumulated live history, and today's live rows.

    Returns (panel, live_date). Training rows carry a target; live-history and
    live rows do not, and are marked with is_live=True so the refit can exclude
    them without guessing from NaN.
    """
    train = pd.read_parquet(train_path).reset_index()
    train[PERIOD] = pd.to_datetime(train[PERIOD])
    train["is_live"] = False

    live = pd.read_parquet(live_path).reset_index()
    live[PERIOD] = pd.to_datetime(live[PERIOD])
    live_date = live[PERIOD].max()

    pieces = [train]
    if os.path.exists(history_path):
        history = pd.read_parquet(history_path)
        history[PERIOD] = pd.to_datetime(history[PERIOD])
        # Anything already covered by train (targets have since resolved) is
        # dropped -- train is authoritative because it carries the target.
        history = history[history[PERIOD] > train[PERIOD].max()]
        history = history[history[PERIOD] < live_date]
        if len(history):
            history["is_live"] = True
            pieces.append(history)

    live["is_live"] = True
    pieces.append(live)

    panel = pd.concat(pieces, ignore_index=True, sort=False)
    panel = panel.sort_values([SYMBOL, PERIOD]).reset_index(drop=True)

    gap_days = int((live_date - train[PERIOD].max()).days)
    if verbose:
        print("panel: %d rows | train through %s | live %s | gap %d days | "
              "%d live rows" % (len(panel), train[PERIOD].max().date(),
                                live_date.date(), gap_days, int(panel["is_live"].sum())))
    return panel, live_date, gap_days


def add_tenc(panel):
    """
    Add feature_sym_tenc, the lagged causal mean-target encoding.

    Delegates to the exact function the offline dataset was built with. Live
    rows have a NaN target; because the encoding at position p may only use
    observations at position <= p - 20 and live rows are the latest positions,
    their own NaN never enters any sum. The assertion below makes that a
    guarantee rather than an argument.
    """
    encoding, counts = causal_target_encoding(
        panel, SYMBOL, PERIOD, TARGET,
        lag=TENC_LAG, smoothing=TENC_SMOOTHING, window=0)

    if not np.isfinite(encoding).all():
        raise ValueError(
            "target encoding produced %d non-finite values -- a live row's NaN "
            "target leaked into a prefix sum" % int((~np.isfinite(encoding)).sum()))

    panel = panel.copy()
    panel["feature_sym_tenc"] = encoding.astype("float32")
    panel["_tenc_support"] = counts
    return panel, ["feature_sym_tenc"]


def add_rolling(panel):
    """
    Add the 24 rolling means, mirroring build_rolling_features exactly.

    min_periods=1 so a token's first day gets the mean of what exists rather
    than NaN: a new listing is still predictable, just with a shorter lookback.
    """
    selected = ["%s_%s" % (f, h) for f in ROLL_FEATURES for h in ROLL_HORIZONS]
    missing = [c for c in selected if c not in panel.columns]
    if missing:
        raise ValueError("rolling inputs missing from the panel: %s" % missing)

    panel = panel.sort_values([SYMBOL, PERIOD]).reset_index(drop=True)
    grouped = panel.groupby(SYMBOL, sort=False)[selected]

    added = []
    for window in ROLL_WINDOWS:
        rolled = grouped.rolling(window, min_periods=1).mean()
        rolled = rolled.reset_index(level=0, drop=True).sort_index()
        rolled.columns = ["%s_rm%d" % (c, window) for c in selected]
        panel = pd.concat([panel, rolled.astype("float32")], axis=1)
        added.extend(rolled.columns)
    return panel, added


def add_onehot(panel, train_mask):
    """
    Add one identity column per symbol with enough history.

    The vocabulary is derived from the TRAINING rows only. That differs from the
    offline builder, which took it from the whole panel -- and the difference is
    a bug fix, not a discrepancy: offline, the column set already knew about
    tokens that had not listed yet. Deriving it from the training block is what
    the live model can actually do, and it makes the backtest's cold-start rate
    (21-26% of symbols per refit) an honest estimate rather than an optimistic
    one.

    Symbols outside the vocabulary get an all-zero block and are predicted from
    their raw features alone. That is a graceful degradation, and the count is
    returned so the round report can show it.
    """
    counts = panel.loc[train_mask].groupby(SYMBOL)[PERIOD].nunique()
    vocabulary = sorted(counts[counts >= ONEHOT_MIN_DATES].index)

    codes = panel[SYMBOL].where(panel[SYMBOL].isin(vocabulary))
    dummies = pd.get_dummies(codes, prefix="feature_sym", dtype="uint8")

    # Reindex onto the full vocabulary so the column set is fixed by the
    # training block, not by which symbols happen to appear today.
    columns = ["feature_sym_%s" % s for s in vocabulary]
    dummies = dummies.reindex(columns=columns, fill_value=0)
    dummies.index = panel.index

    panel = pd.concat([panel, dummies], axis=1)
    cold = int((~panel[SYMBOL].isin(vocabulary) & panel["is_live"]).sum())
    return panel, columns, cold, len(vocabulary)


def add_derived(panel):
    """
    Add the 11 horizon spreads and 14 products / acceleration gaps.

    Delegates to the exact functions the offline datasets were built with, for
    the same reason the target encoding does: two implementations of one feature
    is the largest source of live/backtest skew there is.

    Both are pure same-row transforms of the current row's raw features -- a
    20d-minus-60d difference and a centred product -- so unlike the rolling
    means they need no history at all and are immune to the train/live gap.

    Missing operands propagate to NaN and are filled with the neutral 0.5 by
    build_features, which is the correct reading: a 60-day indicator on a
    30-day-old token has nothing to say, and the raw-difference alternative
    (0 neutral on [-1, 1]) would turn that absence into a strong false signal.
    """
    added = {}
    added.update(build_spreads(panel))
    added.update(build_interactions(panel))
    derived = pd.DataFrame(added, index=panel.index).astype("float32")
    return pd.concat([panel, derived], axis=1), list(derived.columns)


def add_interact(panel):
    """
    Add only the 14 products and acceleration gaps -- no spreads.

    add_derived() builds spreads AND interactions together, which is what
    allfeat_pp wants. The interact arm is raw + interactions only (36 columns),
    so it needs the narrower call; sharing add_derived() here would silently
    ship 11 extra spread columns and stop reproducing the backtest.

    Same delegation argument as add_derived(): build_interactions() is the exact
    function the offline dataset was built with, and these are pure same-row
    transforms, so there is no history to align and no live/train skew to get
    wrong.
    """
    added = build_interactions(panel)
    derived = pd.DataFrame(added, index=panel.index).astype("float32")
    return pd.concat([panel, derived], axis=1), list(derived.columns)


def build_features(panel, feature_set, verbose=True):
    """
    Return (panel, feature_columns, diagnostics) for one feature set.

    :param panel: train + live rows, as returned by load_panel
    :param feature_set: one of FEATURE_SETS
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError("unknown feature set '%s', expected one of %s" % (
            feature_set, list(FEATURE_SETS)))

    base = raw_features(panel)
    diagnostics = {"feature_set": feature_set, "n_base": len(base)}
    columns = list(base)

    if feature_set == "tenc":
        panel, added = add_tenc(panel)
        columns += added
        live_support = panel.loc[panel["is_live"], "_tenc_support"]
        diagnostics["tenc_median_support"] = float(live_support.median())
        diagnostics["tenc_live_no_history"] = int((live_support == 0).sum())

    elif feature_set == "interact":
        panel, added = add_interact(panel)
        columns += added
        diagnostics["n_interact"] = len(added)

    elif feature_set == "onehot_roll":
        panel, rolled = add_rolling(panel)
        panel, dummies, cold, vocab = add_onehot(panel, ~panel["is_live"])
        columns += rolled + dummies
        diagnostics.update(n_rolling=len(rolled), onehot_vocabulary=vocab,
                           live_cold_start=cold)

    elif feature_set == "allfeat":
        # allfeat_pp MINUS the 24 rolling means: 788 columns, not 812. Same
        # ordering argument as allfeat_pp -- the spreads and interactions are
        # built from the RAW features, so they go in before the one-hot block
        # widens the frame.
        panel, added = add_tenc(panel)
        columns += added
        panel, derived = add_derived(panel)
        columns += derived
        panel, dummies, cold, vocab = add_onehot(panel, ~panel["is_live"])
        columns += dummies
        live_support = panel.loc[panel["is_live"], "_tenc_support"]
        diagnostics.update(n_derived=len(derived), onehot_vocabulary=vocab,
                           live_cold_start=cold,
                           tenc_median_support=float(live_support.median()))

    elif feature_set == "allfeat_pp":
        # Every block at once: the union of the other three plus the derived
        # columns. Order matters -- the spreads and interactions are built from
        # the RAW features, so they must be added before the one-hot block
        # widens the frame, and never from the rolling columns (which carry
        # _rm suffixes and so do not match the 20d/60d patterns anyway).
        panel, added = add_tenc(panel)
        columns += added
        panel, derived = add_derived(panel)
        columns += derived
        panel, rolled = add_rolling(panel)
        panel, dummies, cold, vocab = add_onehot(panel, ~panel["is_live"])
        columns += rolled + dummies
        live_support = panel.loc[panel["is_live"], "_tenc_support"]
        diagnostics.update(n_derived=len(derived), n_rolling=len(rolled),
                           onehot_vocabulary=vocab, live_cold_start=cold,
                           tenc_median_support=float(live_support.median()))

    # Neutral fill last, so every derived column is covered by the same rule.
    # 0.5 means "this feature has nothing to say about this row" -- the correct
    # reading for a 60-day indicator on a token that is 30 days old.
    panel = fill_missing_features(panel, columns, NEUTRAL, verbose=False)

    diagnostics["n_features"] = len(columns)
    if verbose:
        print("%-12s -> %d features%s" % (
            feature_set, len(columns),
            "" if feature_set == "raw" else "  " + str(
                {k: v for k, v in diagnostics.items()
                 if k not in ("feature_set", "n_base", "n_features")})))
    return panel, columns, diagnostics


def main():
    parser = argparse.ArgumentParser(description="Inspect a live feature build.")
    parser.add_argument("--feature-set", default="onehot_roll", choices=FEATURE_SETS)
    args = parser.parse_args()

    panel, live_date, gap = load_panel()
    panel, columns, diagnostics = build_features(panel, args.feature_set)
    live = panel[panel["is_live"] & (panel[PERIOD] == live_date)]
    print("\nlive rows: %d | features: %d | NaN in live block: %d" % (
        len(live), len(columns), int(live[columns].isna().to_numpy().sum())))
    print("diagnostics:", diagnostics)


if __name__ == "__main__":
    main()
