"""
Random-forest hyperparameter sensitivity, after Kalfa, Timmermann & van der
Zwan (2025) -- the framework for experiment E48
(experiments/e48_rf_hyperparam_sensitivity_crypto.sh).

    python3 -m src.analysis.hyperparam_sensitivity_rf --prefix e48

The RF sibling of src.analysis.hyperparam_sensitivity (the MLP/E17 tool). It
reuses that module's statistical core -- box_stats, marginals, fanova, corners,
robustness -- passing the RANDOM FOREST axes instead of the MLP ones, and reads
each cell's OOS payout from its per-period series the same way. Only three
things differ, and all three follow from a forest not being a boosted net:

  * THE AXES. A forest has no learning rate and no architecture. Its four knobs
    are n_estimators, max_depth, max_features and min_samples_leaf.

  * THE TWO-WAY CLAIM. The paper's headline NN interaction is learning-rate x
    alpha (a fast net needs strong shrinkage). A forest has neither, so the
    analogue printed here is max_features x min_samples_leaf -- the two axes that
    carry the Numerai idioms (very low column sampling for diversity; heavy leaf
    regularisation for the noisy target). The open question E48 answers is
    whether those two idioms interact -- must a low column sample be paired with
    a heavy leaf, or do they help independently.

  * NO COLLAPSE NARRATIVE. Deep ReLU nets at a high learning rate go constant
    (E17's 15 collapsed cells); a forest does not. The one degenerate mode here
    is a min_samples_leaf so large no split is valid on a refit block, which
    also yields a constant prediction (corr NaN -> payout 0). On the ~38k-row
    crypto block the grid's ceiling of 1000 is ~2.6% of the rows and does not
    trigger it, but the reading is defensive: any payout ~= 0 cell is flagged,
    not silently averaged in, exactly as the MLP tool flags a collapsed net.

The metric is this repo's OOS payout, 0.1 * corr + mmc, the mean over the
walk-forward periods, read from each cell's per-period series so the half-sample
robustness split is exact. Every cell in the sweep shares one common window
(same data, cadence and seed), so the cells are directly comparable and no
window caveat applies within the grid.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from src.analysis.hyperparam_sensitivity import (
    box_stats, corners, fanova, fmt, marginals, order_values, robustness,
)

METRICS = "results/metrics/crypto/rf"
PER_PERIOD = "results/per_period/crypto/rf"

#: The forest's four axes, in the order the report walks them.
AXES = ["n_estimators", "max_depth", "max_features", "min_samples_leaf"]

#: Natural (small -> large) print order per axis. max_depth is carried as a
#: STRING so None ("grow the tree out fully") sits at the deep end of the axis
#: rather than being unorderable against the integer caps; the numeric axes sort
#: themselves. These must list every value the runner emits.
AXIS_ORDER = {
    "n_estimators": [100, 300, 1000],
    "max_depth": ["3", "6", "12", "None"],
    "max_features": [0.1, 0.3, 0.6, 1.0],
    "min_samples_leaf": [1, 10, 100, 1000],
}

#: The two axes that carry the Numerai idioms -- the forest analogue of the
#: paper's learning-rate x alpha interaction.
IDIOM_PAIR = ("max_features", "min_samples_leaf")

EXPECTED = 192  # 3 n_estimators x 4 max_depth x 4 max_features x 4 min_samples_leaf


def parse_config(metrics_json):
    """
    Recover the four RF axes from a cell's recorded config.

    Read from config["fixed_params"], which run_forecast stores verbatim -- not
    from the run name, which is a lossy tag. max_depth is normalised to a string
    ("None" for the grow-fully value passed as JSON null) so it orders and groups
    cleanly against the integer caps.

    :param metrics_json: the loaded metrics dict
    :return: {n_estimators, max_depth, max_features, min_samples_leaf} or None
    :rtype: dict or None
    """
    fixed = (metrics_json.get("config") or {}).get("fixed_params")
    if not fixed:
        return None
    try:
        depth = fixed["max_depth"]
        depth = "None" if depth is None else str(int(depth))
        return {
            "n_estimators": int(fixed["n_estimators"]),
            "max_depth": depth,
            "max_features": float(fixed["max_features"]),
            "min_samples_leaf": int(fixed["min_samples_leaf"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def load_grid(prefix):
    """
    Every E48 cell on disk, with its axes and its per-period payout series.

    Mirrors the MLP tool's loader (fillna(0) so an over-regularised, constant
    cell enters with an honest deflated payout rather than a NaN that silently
    drops it), but reads the `rf` model tree and the RF axes.

    :param prefix: e.g. "e48"
    :return: DataFrame, one row per cell
    :rtype: pd.DataFrame
    """
    rows = []
    for path in sorted(glob.glob("%s/%s_*.json" % (METRICS, prefix))):
        if path.endswith(".seeds.json"):
            continue
        run = os.path.basename(path)[:-5]
        data = json.load(open(path))
        axes = parse_config(data)
        if axes is None:
            continue
        pp_path = "%s/%s.csv" % (PER_PERIOD, run)
        if not os.path.exists(pp_path):
            continue
        frame = pd.read_csv(pp_path)
        series = (0.1 * frame["corr"] + frame["mmc"]).fillna(0.0).to_numpy()
        row = dict(axes)
        row["run"] = run
        row["payout"] = float(np.mean(series))
        row["corr"] = float(frame["corr"].mean())
        row["n"] = len(series)
        row["payout_series"] = series
        rows.append(row)
    return pd.DataFrame(rows)


def two_way(grid, a, b, score="payout"):
    """
    Median `score` for every (a, b) cell, ordered on both axes -- the paper's
    two-way table, here over the two Numerai-idiom axes.
    """
    piv = grid.pivot_table(index=a, columns=b, values=score, aggfunc="median")
    piv = piv.reindex(index=order_values(a, grid[a].unique(), AXIS_ORDER),
                      columns=order_values(b, grid[b].unique(), AXIS_ORDER))
    return piv


def top_quartile_support(grid, score="payout"):
    """
    Where the good mass sits: among the top quartile of cells by OOS payout, how
    many carry each value of each axis. A value absent from the good set is one
    the prior can drop.

    :return: (cutoff, n_good, {axis: {value: count}})
    """
    cut = grid[score].quantile(0.75)
    good = grid[grid[score] >= cut]
    support = {}
    for axis in AXES:
        counts = good[axis].value_counts()
        support[axis] = {v: int(counts.get(v, 0))
                         for v in order_values(axis, grid[axis].unique(), AXIS_ORDER)}
    return float(cut), int(len(good)), support


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--prefix", default="e48")
    parser.add_argument("--score", default="payout", choices=["payout", "corr"])
    parser.add_argument("--expected", type=int, default=EXPECTED)
    args = parser.parse_args(argv)

    grid = load_grid(args.prefix)
    print("=" * 78)
    print("E48 random-forest hyperparameter sensitivity "
          "(Kalfa-Timmermann-van der Zwan), metric = OOS %s" % args.score)
    print("=" * 78)
    print("cells on disk: %d of %d" % (len(grid), args.expected))
    if len(grid) == 0:
        print("nothing to analyse yet."); return 0

    # Degenerate cells: a min_samples_leaf too large for a refit block yields a
    # constant prediction (corr NaN -> payout 0). The forest analogue of the
    # MLP collapse, flagged rather than silently averaged in.
    degen = grid[grid["payout"].abs() < 1e-9]
    if len(degen):
        print("DEGENERATE cells (payout ~= 0 -- a leaf floor too large for the "
              "block\n  leaves no valid split, so the forest returns a constant): "
              "%d of %d" % (len(degen), len(grid)))
        for _, r in degen.iterrows():
            print("    %-40s min_samples_leaf=%d max_depth=%s"
                  % (r["run"], r["min_samples_leaf"], r["max_depth"]))
    if len(grid) < args.expected:
        print("WARNING: the grid is INCOMPLETE. Marginals and corners are read "
              "on what is present; the fANOVA decomposition assumes a balanced "
              "factorial and is only exact once all %d cells exist." % args.expected)
    print()

    for label, s in (("payout", grid["payout"]), ("corr", grid["corr"])):
        skew = float(((s - s.mean()) ** 3).mean() / (s.std() ** 3 + 1e-12))
        print("DISPERSION of OOS %s across all %d cells" % (label, len(grid)))
        print("  min %+.6f  q25 %+.6f  median %+.6f  q75 %+.6f  max %+.6f" % (
            s.min(), s.quantile(.25), s.median(), s.quantile(.75), s.max()))
        print("  range %.6f  |  skew %+.2f" % (s.max() - s.min(), skew))
    print()

    print("1. MARGINAL DISTRIBUTIONS -- fix one axis, vary the other three")
    for axis, table in marginals(grid, args.score, axes=AXES,
                                 axis_order=AXIS_ORDER).items():
        print("  [%s]" % axis)
        print("    " + fmt(table).replace("\n", "\n    "))
        spread = table["median"].max() - table["median"].min()
        print("    range of conditional medians: %.6f" % spread)
        print()

    print("2. IMPORTANCE -- functional-ANOVA variance shares")
    frame, total, unexplained = fanova(grid, args.score, axes=AXES)
    print("    " + fmt(frame[["kind", "term", "share"]]).replace("\n", "\n    "))
    print("    higher-order / unexplained: %.3f" % unexplained)
    print("    -> an axis with a near-zero main share is not worth tuning.")
    print()

    print("3. CORNER CASES")
    good, bad = corners(grid, args.score, axes=AXES, axis_order=AXIS_ORDER)
    print("  GOOD corners (high worst-case -- good regardless of the rest):")
    print("    " + fmt(good).replace("\n", "\n    "))
    print("  BAD corners (low best-case -- failure locked in):")
    print("    " + fmt(bad).replace("\n", "\n    "))
    print()
    a, b = IDIOM_PAIR
    print("  %s x %s, median OOS %s (the two Numerai-idiom axes -- the forest's"
          % (a, b, args.score))
    print("  analogue of the paper's learning-rate x alpha interaction):")
    print("    " + fmt(two_way(grid, a, b, args.score)).replace("\n", "\n    "))
    print()

    print("4. PRIORS -- read off the top-quartile cells")
    cut, n_good, support = top_quartile_support(grid, args.score)
    print("    good = payout >= %.6f  (%d cells)" % (cut, n_good))
    for axis in AXES:
        print("    %-16s support among good: %s" % (axis, support[axis]))
    print("    a value with zero support among the good cells is one to drop; a")
    print("    value present across the whole good set is a robust keep.")
    print()

    print("ROBUSTNESS -- main-effect shares on each half of the periods")
    print("    " + fmt(robustness(args.prefix, grid, axes=AXES)).replace("\n", "\n    "))
    print("    an importance ranking that survives both halves is a property of")
    print("    the hyperparameter, not of the sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
