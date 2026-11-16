"""
Gaussian-process hyperparameter sensitivity, after Kalfa, Timmermann & van der
Zwan (2025) -- the framework for experiment E54
(experiments/e54_gp_hyperparam_sensitivity_crypto.sh).

    python3 -m src.analysis.hyperparam_sensitivity_gp --prefix e54

The GP sibling of src.analysis.hyperparam_sensitivity_rf (E48) and
src.analysis.hyperparam_sensitivity (the MLP/E17 tool). It reuses that module's
statistical core -- box_stats, marginals, fanova, corners, robustness -- passing
the GAUSSIAN-PROCESS axes instead of the forest ones, and reads each cell's OOS
payout from its per-period series the same way. Only three things differ, and
all three follow from a GP being a kernel model rather than a tree ensemble:

  * THE AXES. A GP has no trees and no learning rate. Its knobs are the kernel
    FAMILY (Matern-1/2, -3/2, -5/2, RBF -- roughest to smoothest), whether the
    length-scale is shared (ISO) or per-feature (ARD), the LENGTH-SCALE itself
    (how fast the return surface varies), and ALPHA (the observation-noise
    variance on the kernel diagonal -- the GP's regularisation knob).

  * THE TWO-WAY CLAIM. The paper's headline NN interaction is learning-rate x
    alpha; E48's forest analogue is max_features x min_samples_leaf. The GP
    analogue printed here is length_scale x alpha -- FUNCTION SMOOTHNESS vs
    OBSERVATION NOISE, the two routes to controlling overfit on a noisy target.
    A long length-scale makes the fitted function smoother (fewer effective
    degrees of freedom); a large alpha shrinks the posterior toward the mean.
    The open question E54 answers is whether they substitute or must be paired.

  * THE DEGENERACY MODE. A deep ReLU net collapses (E17); a leaf floor too large
    leaves no split (E48). A GP's degeneracy is a near-singular kernel: a small
    alpha paired with a short length-scale makes K + alpha*I ill-conditioned, so
    the Cholesky either fails (the estimator falls back to a constant) or fits
    the training noise and predicts near-constant out of sample (corr NaN ->
    payout 0). It concentrates at alpha's grid floor; those cells are flagged,
    not silently averaged into the fANOVA.

The metric is this repo's OOS payout, 0.1 * corr + mmc, the mean over the
walk-forward periods, read from each cell's per-period series so the half-sample
robustness split is exact. Every cell in the sweep shares one common window
(same data, cadence and seed), so the cells are directly comparable and no
window caveat applies within the grid.

The priors this reads off do double duty: they set the gp_ktz SEARCH grid (the
KTZ sense -- the good region becomes the search space, bad corners dropped) AND
the GP's own kernel PRIOR (a kernel IS a prior: the length-scale support becomes
gp_ktz's length_scale / length_scale_bounds, the alpha support the assumed
observation noise, the kernel-family support the smoothness prior).
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

METRICS = "results/metrics/crypto/gp"
PER_PERIOD = "results/per_period/crypto/gp"

#: The GP's four axes, in the order the report walks them.
AXES = ["kernel", "ard", "length_scale", "alpha"]

#: Natural (rough -> smooth, iso -> ard, small -> large) print order per axis.
#: kernel and ard are carried as STRINGS so they order and group cleanly; the
#: numeric axes sort themselves. These must list every value the runner emits.
AXIS_ORDER = {
    "kernel": ["matern12", "matern32", "matern52", "rbf"],
    "ard": ["iso", "ard"],
    "length_scale": [0.25, 1.0, 4.0],
    "alpha": [1e-3, 1e-2, 1e-1, 1.0],
}

#: The two axes that carry the overfit-control idiom -- the GP analogue of the
#: paper's learning-rate x alpha and of E48's max_features x min_samples_leaf.
IDIOM_PAIR = ("length_scale", "alpha")

EXPECTED = 96  # 4 kernel x 2 ard x 3 length_scale x 4 alpha


def parse_config(metrics_json):
    """
    Recover the four GP axes from a cell's recorded config.

    Read from config["fixed_params"], which run_forecast stores verbatim -- not
    from the run name, which is a lossy tag. `ard` is normalised to the strings
    "iso"/"ard" (from the JSON bool) so it orders and groups cleanly, the same
    way E48 carries max_depth=None as a string.

    :param metrics_json: the loaded metrics dict
    :return: {kernel, ard, length_scale, alpha} or None
    :rtype: dict or None
    """
    fixed = (metrics_json.get("config") or {}).get("fixed_params")
    if not fixed:
        return None
    try:
        return {
            "kernel": str(fixed["kernel"]),
            "ard": "ard" if fixed["ard"] else "iso",
            "length_scale": float(fixed["length_scale"]),
            "alpha": float(fixed["alpha"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def load_grid(prefix):
    """
    Every E54 cell on disk, with its axes and its per-period payout series.

    Mirrors the RF tool's loader (fillna(0) so a degenerate, constant cell enters
    with an honest deflated payout rather than a NaN that silently drops it), but
    reads the `gp` model tree and the GP axes.

    :param prefix: e.g. "e54"
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
    two-way table, here over length_scale x alpha (smoothness vs noise).
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
    parser.add_argument("--prefix", default="e54")
    parser.add_argument("--score", default="payout", choices=["payout", "corr"])
    parser.add_argument("--expected", type=int, default=EXPECTED)
    args = parser.parse_args(argv)

    grid = load_grid(args.prefix)
    print("=" * 78)
    print("E54 Gaussian-process hyperparameter sensitivity "
          "(Kalfa-Timmermann-van der Zwan), metric = OOS %s" % args.score)
    print("=" * 78)
    print("cells on disk: %d of %d" % (len(grid), args.expected))
    if len(grid) == 0:
        print("nothing to analyse yet."); return 0

    # Degenerate cells: a near-singular kernel (small alpha + short length-scale)
    # predicts a constant (corr NaN -> payout 0). Flagged, not averaged in.
    degen = grid[grid["payout"].abs() < 1e-9]
    if len(degen):
        print("DEGENERATE cells (payout ~= 0 -- a small alpha with a short "
              "length-scale\n  makes K + alpha*I near-singular, so the GP returns "
              "a constant): %d of %d" % (len(degen), len(grid)))
        for _, r in degen.iterrows():
            print("    %-44s alpha=%g length_scale=%g kernel=%s"
                  % (r["run"], r["alpha"], r["length_scale"], r["kernel"]))
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
    print("  %s x %s, median OOS %s (smoothness vs observation noise -- the GP's"
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
    print("    value present across the whole good set is a robust keep. These")
    print("    become BOTH the gp_ktz search grid and the GP's kernel prior.")
    print()

    print("ROBUSTNESS -- main-effect shares on each half of the periods")
    print("    " + fmt(robustness(args.prefix, grid, axes=AXES)).replace("\n", "\n    "))
    print("    an importance ranking that survives both halves is a property of")
    print("    the hyperparameter, not of the sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
