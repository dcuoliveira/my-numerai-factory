"""
Hyperparameter sensitivity, after Kalfa, Timmermann & van der Zwan (2025),
"Overhyped? Can ML Models Reliably Predict Stock Returns?" -- the framework for
experiment E17 (experiments/e17_hyperparam_sensitivity_crypto.sh).

    python3 -m src.analysis.hyperparam_sensitivity --prefix e17

Reads one metrics json per cell of the NN hyperparameter factorial and produces
the paper's four readings, adapted to this repo's OOS payout:

  1. MARGINAL DISTRIBUTIONS.  Fix one hyperparameter at a value, look at the
     distribution of OOS payout over the full grid of all the OTHERS (paper's
     box plots, Figs 4-7). Reported as min / q25 / median / q75 / max per value.

  2. IMPORTANCE -- which hyperparameters are worth tuning.  Two lenses:
       (a) the range of conditional medians across a hyperparameter's values;
       (b) a functional-ANOVA variance decomposition -- the fraction of total
           variance in OOS payout explained by each hyperparameter's main effect
           and by each pair's interaction. A hyperparameter explaining ~0% is not
           worth tuning; this is the quantitative form of the paper's eyeballed
           "learning rate matters, architecture does not".

  3. CORNER CASES.  A "good corner" is a (hyperparameter, value) slice whose
     WORST config still performs well -- good regardless of the rest. A "bad
     corner" is a slice whose BEST config is still poor -- failure locked in.
     Reported one-way and for the paper's specific two-way NN claim, the
     learning-rate x alpha interaction.

  4. PRIORS.  From the good corners: which ranges to keep, which to drop, and the
     joint structure. The paper predicts a POSITIVE learning-rate/alpha
     correlation for NNs (a high learning rate must be paired with strong
     shrinkage). We test it directly among the top-quartile configs.

  ROBUSTNESS.  Every finding is recomputed on the first and second half of the
     walk-forward periods. A ranking that flips between halves is a property of
     the sample, not the hyperparameter -- the failure mode E12 and E14 both hit,
     where a |t| > 2 traced to one refit block.

The metric is OOS payout, 0.1 * corr + mmc, the mean over the walk-forward
periods -- the same axis the whole board is scored on. Read from each config's
per-period series, not from the stored summary, so the half-sample split is
exact.
"""

import argparse
import glob
import itertools
import json
import os
import re

import numpy as np
import pandas as pd

METRICS = "results/metrics/crypto/mlp"
PER_PERIOD = "results/per_period/crypto/mlp"

AXES = ["arch", "activation", "lr", "alpha"]
#: The order each axis's values should print in (small -> large), so a monotone
#: trend reads down the column rather than being scrambled by string sort.
AXIS_ORDER = {
    "arch": ["32", "32,16", "32,16,8", "32,16,8,4", "32,16,8,4,2"],
    "activation": ["identity", "relu"],
    "lr": [1e-4, 1e-3, 1e-2, 1e-1],
    "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
}


def parse_config(metrics_json):
    """
    Recover the four factorial axes from a run's recorded config.

    Read from config["fixed_params"], which run_forecast stores verbatim, NOT
    from the run name -- a name is a lossy tag and the config is the record.

    :param metrics_json: the loaded metrics dict
    :return: {arch, activation, lr, alpha} or None if not an E17 cell
    :rtype: dict or None
    """
    fixed = (metrics_json.get("config") or {}).get("fixed_params")
    if not fixed:
        return None
    try:
        arch = ",".join(str(int(x)) for x in fixed["mlp__hidden_layer_sizes"])
        return {
            "arch": arch,
            "activation": fixed["mlp__activation"],
            "lr": float(fixed["mlp__learning_rate_init"]),
            "alpha": float(fixed["mlp__alpha"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def load_grid(prefix):
    """
    Every E17 cell on disk, with its axes and its per-period payout series.

    :param prefix: e.g. "e17"
    :return: DataFrame, one row per config, columns = axes + payout summaries +
             a 'payout_series' object column
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
        # A period where the net output a constant prediction has undefined corr
        # (NaN in the csv). Economically that period carries no rank information,
        # so its payout is 0 -- fillna(0), not nanmean, which would over-weight
        # the periods that did work. This is what lets a PARTIALLY collapsed deep
        # net (constant in some refit blocks, not others) enter the grid with an
        # honest, deflated payout rather than a NaN that silently drops it.
        series = (0.1 * frame["corr"] + frame["mmc"]).fillna(0.0).to_numpy()
        row = dict(axes)
        row["run"] = run
        row["collapsed"] = bool(data.get("collapsed") or
                                (data.get("config") or {}).get("collapsed"))
        row["payout"] = float(np.mean(series))
        row["corr"] = float(frame["corr"].mean())
        row["n"] = len(series)
        row["payout_series"] = series
        rows.append(row)
    return pd.DataFrame(rows)


def order_values(axis, values, axis_order=None):
    """Sort an axis's observed values into the natural (small->large) order."""
    axis_order = AXIS_ORDER if axis_order is None else axis_order
    canonical = axis_order.get(axis, sorted(values))
    present = [v for v in canonical if v in set(values)]
    # anything unexpected goes last, sorted, so nothing is silently dropped
    present += sorted(v for v in set(values) if v not in set(canonical))
    return present


def box_stats(values):
    """min / q25 / median / q75 / max of a 1-D array, as the paper's box plot."""
    v = np.asarray(values, float)
    return {
        "n": len(v), "min": v.min(), "q25": np.percentile(v, 25),
        "median": np.median(v), "q75": np.percentile(v, 75), "max": v.max(),
    }


def marginals(grid, score="payout", axes=None, axis_order=None):
    """
    Paper's marginal box plots: fix each axis at each value, describe the
    distribution of `score` over the full grid of the other axes.

    :param axes: the factorial axes to describe (defaults to this module's MLP
                 AXES; the RF sibling passes its own)
    :param axis_order: value ordering per axis (defaults to AXIS_ORDER)
    :return: {axis: DataFrame indexed by value, columns box_stats}
    """
    axes = AXES if axes is None else axes
    out = {}
    for axis in axes:
        table = {}
        for value in order_values(axis, grid[axis].unique(), axis_order):
            table[value] = box_stats(grid.loc[grid[axis] == value, score])
        out[axis] = pd.DataFrame(table).T
    return out


def fanova(grid, score="payout", axes=None):
    """
    Functional-ANOVA variance decomposition over the full factorial.

    The fraction of the total variance of `score` explained by each axis's main
    effect and by each pair's interaction, computed the classical balanced-design
    way: a main effect is the variance of that axis's cell means about the grand
    mean; a two-way interaction is the residual variance of the two-axis cell
    means after removing both main effects. Reported as a share of total.

    A balanced full factorial makes this exact and additive; E17 is balanced by
    construction (the runner emits every cell), and a warning is printed if a
    cell is missing so a partial grid cannot masquerade as a clean decomposition.

    :param axes: the factorial axes to decompose (defaults to the MLP AXES)
    :return: (DataFrame of effects sorted by share, total_variance, unexplained)
    """
    axes = AXES if axes is None else axes
    g = grid.dropna(subset=[score]).copy()
    grand = g[score].mean()
    total = float(np.mean((g[score] - grand) ** 2))
    if total == 0:
        return pd.DataFrame(), 0.0, 0.0

    # main effects
    effects = {}
    main_pred = {}
    for axis in axes:
        means = g.groupby(axis)[score].transform("mean")
        main_pred[axis] = means - grand
        effects[("main", axis)] = float(np.mean(main_pred[axis] ** 2)) / total

    # two-way interactions: cell mean minus both main effects minus grand
    for a, b in itertools.combinations(axes, 2):
        cell = g.groupby([a, b])[score].transform("mean")
        inter = cell - grand - main_pred[a] - main_pred[b]
        effects[("pair", "%s x %s" % (a, b))] = float(np.mean(inter ** 2)) / total

    frame = pd.DataFrame(
        [{"kind": k, "term": t, "share": s} for (k, t), s in effects.items()]
    ).sort_values("share", ascending=False).reset_index(drop=True)
    explained = frame["share"].sum()
    return frame, total, max(0.0, 1.0 - explained)


def corners(grid, score="payout", top=6, axes=None, axis_order=None):
    """
    Good and bad corners, one-way.

    Good corner: high MINIMUM within the slice -- good no matter the rest.
    Bad corner:  low MAXIMUM within the slice  -- bad no matter the rest.

    :param axes: the factorial axes to slice (defaults to the MLP AXES)
    :param axis_order: value ordering per axis (defaults to AXIS_ORDER)
    :return: (good DataFrame, bad DataFrame), each sorted most-extreme first
    """
    axes = AXES if axes is None else axes
    good, bad = [], []
    for axis in axes:
        for value in order_values(axis, grid[axis].unique(), axis_order):
            s = grid.loc[grid[axis] == value, score]
            good.append({"axis": axis, "value": value, "worst": s.min(),
                         "median": s.median(), "best": s.max(), "n": len(s)})
            bad.append({"axis": axis, "value": value, "best": s.max(),
                        "median": s.median(), "worst": s.min(), "n": len(s)})
    good = pd.DataFrame(good).sort_values("worst", ascending=False).reset_index(drop=True)
    bad = pd.DataFrame(bad).sort_values("best").reset_index(drop=True)
    return good.head(top), bad.head(top)


def lr_alpha_interaction(grid, score="payout"):
    """
    The paper's specific two-way NN claim, as a table: median OOS payout for
    every (learning rate, alpha) cell. The paper predicts high lr needs high
    alpha -- so the good region should run along a positive lr/alpha diagonal.

    :return: DataFrame, lr as rows, alpha as columns, median payout in cells
    """
    piv = grid.pivot_table(index="lr", columns="alpha", values=score, aggfunc="median")
    piv = piv.reindex(index=AXIS_ORDER["lr"], columns=AXIS_ORDER["alpha"])
    return piv


def prior_from_corners(grid, score="payout"):
    """
    A prior on (lr, alpha) read off the good configs, in the paper's spirit.

    Restrict to the top quartile of configs by OOS payout and report, for lr and
    alpha, where that good mass sits (which values survive) and the correlation
    of (log10 lr, log10 alpha) among them -- the paper's positive-correlation
    prediction, measured.

    :return: dict with the surviving values and the log-log correlation
    """
    cut = grid[score].quantile(0.75)
    good = grid[grid[score] >= cut]
    result = {"cutoff_payout": float(cut), "n_good": int(len(good))}
    for axis in ("lr", "alpha"):
        counts = good[axis].value_counts()
        share = {v: int(counts.get(v, 0)) for v in AXIS_ORDER[axis]}
        result["%s_support" % axis] = share
    if len(good) > 2 and good["lr"].nunique() > 1 and good["alpha"].nunique() > 1:
        rho = np.corrcoef(np.log10(good["lr"]), np.log10(good["alpha"]))[0, 1]
        result["log_lr_alpha_corr_in_good"] = float(rho)
    else:
        result["log_lr_alpha_corr_in_good"] = float("nan")
    return result


def robustness(prefix, grid, axes=None):
    """
    Recompute the importance ranking on the first and second half of the
    walk-forward periods, from the per-period payout series.

    :param axes: the factorial axes to rank (defaults to the MLP AXES)
    :return: DataFrame, axis vs (share_full, share_first, share_second)
    """
    axes = AXES if axes is None else axes

    def halves(row):
        s = row["payout_series"]; h = len(s) // 2
        return float(np.mean(s[:h])), float(np.mean(s[h:]))

    parts = grid.copy()
    parts[["pay_first", "pay_second"]] = parts.apply(
        lambda r: pd.Series(halves(r)), axis=1)

    rows = []
    for score in ("payout", "pay_first", "pay_second"):
        frame, _, _ = fanova(parts, score=score, axes=axes)
        main = frame[frame["kind"] == "main"].set_index("term")["share"]
        rows.append(main.rename({"payout": "full"}.get(score, score)))
    out = pd.DataFrame({
        "full": rows[0], "first_half": rows[1], "second_half": rows[2]})
    return out.reindex(axes)


def fmt(frame, floatfmt="%.6f"):
    with pd.option_context("display.float_format", lambda x: floatfmt % x,
                           "display.width", 200, "display.max_columns", 20):
        return frame.to_string()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--prefix", default="e17")
    parser.add_argument("--score", default="payout", choices=["payout", "corr"])
    args = parser.parse_args(argv)

    grid = load_grid(args.prefix)
    expected = 160
    print("=" * 78)
    print("E17 hyperparameter sensitivity (Kalfa-Timmermann-van der Zwan), "
          "metric = OOS %s" % args.score)
    print("=" * 78)
    print("configs on disk: %d of %d" % (len(grid), expected))
    if len(grid) == 0:
        print("nothing to analyse yet."); return 0
    n_coll = int(grid.get("collapsed", pd.Series(dtype=bool)).sum()) if "collapsed" in grid else 0
    if n_coll:
        coll = grid[grid["collapsed"]]
        print("COLLAPSED cells: %d of %d -- deep ReLU nets at a high learning rate "
              "that\n  output a CONSTANT prediction (verified pred std ~5e-17). A "
              "constant ranks\n  all assets equally, so payout = 0; recorded, not "
              "measured. This IS the\n  paper's predicted NN bad corner (footnote "
              "25). Present at:" % (n_coll, len(grid)))
        for arch in sorted(coll["arch"].unique(), key=len):
            sub = coll[coll["arch"] == arch]
            print("    arch %-12s lr in {%s}" % (
                arch, ", ".join(sorted({"%g" % v for v in sub["lr"]}))))
        surv = grid[(grid["arch"].isin(["32,16,8,4", "32,16,8,4,2"])) &
                    (grid["activation"] == "relu") & (grid["lr"] >= 1e-2) &
                    (~grid["collapsed"])]
        if len(surv):
            print("  NOT every deep/high-lr cell collapsed -- %d survived (e.g. %s), "
                  "so\n  collapse is a boundary, not a guarantee." %
                  (len(surv), surv.iloc[0]["run"]))
        print()
    if len(grid) < expected:
        # Name it: a partial grid unbalances the fANOVA and hides corners.
        print("WARNING: the grid is INCOMPLETE. Marginals and corners are read on "
              "what is present; the fANOVA decomposition assumes a balanced "
              "factorial and is only exact once all %d cells exist." % expected)
    print()

    # How wide and how skewed is OOS performance across the grid. The paper's
    # central warning is about R^2_OoS specifically -- a wide, left-skewed
    # distribution with catastrophic downside. OOS PAYOUT (0.1*corr+mmc) is NOT
    # R^2_OoS: a collapsed net scores ~0 payout, not -30% R^2, so payout
    # structurally COMPRESSES exactly the disaster tail the paper's claim is
    # about. CORR is the closer analogue -- unbounded below, it can go clearly
    # negative -- so both are printed, and the skew is read off CORR, not payout.
    for label, s in (("payout", grid["payout"]), ("corr", grid["corr"])):
        skew = float(((s - s.mean()) ** 3).mean() / (s.std() ** 3 + 1e-12))
        print("DISPERSION of OOS %s across all %d configs" % (label, len(grid)))
        print("  min %+.6f  q25 %+.6f  median %+.6f  q75 %+.6f  max %+.6f" % (
            s.min(), s.quantile(.25), s.median(), s.quantile(.75), s.max()))
        print("  range %.6f  |  skew %+.2f" % (s.max() - s.min(), skew))
    print("  NOTE payout bounds the downside (a dead net -> ~0, not a large")
    print("  negative), so the paper's 'extremely poor / left-skewed R^2_OoS'")
    print("  warning shows up in CORR here, not payout. For scale, e15_mlp_raw")
    print("  (a full 50-draw search) pays ~0.0803.")
    print()

    # Per-architecture mean payout, so the paper's collapse of the 4- and 5-layer
    # nets (footnote 25: large biases dominate, activations saturate, predictions
    # go constant) is VISIBLE rather than silently inflating the architecture
    # effect below.
    print("0. PER-ARCHITECTURE mean OOS %s (watch the deep nets collapse)" % args.score)
    per_arch = grid.groupby("arch")[args.score].agg(["mean", "median", "min", "max", "count"])
    per_arch = per_arch.reindex([a for a in AXIS_ORDER["arch"] if a in set(grid["arch"])])
    print("    " + fmt(per_arch).replace("\n", "\n    "))
    print()

    # The paper reports its activation/lr/alpha sensitivities at a FIXED
    # architecture (NN2 = [32,16]) because deep nets collapse and pooling over
    # them contaminates the marginal. So both are shown: pooled over all archs,
    # and restricted to NN2, which is the paper-comparable one.
    nn2 = grid[grid["arch"] == "32,16"]

    print("1. MARGINAL DISTRIBUTIONS -- fix one axis, vary the rest (POOLED over archs)")
    for axis, table in marginals(grid, args.score).items():
        print("  [%s]" % axis)
        print("    " + fmt(table).replace("\n", "\n    "))
        spread = table["median"].max() - table["median"].min()
        print("    range of conditional medians: %.6f" % spread)
        print()

    if len(nn2):
        print("1b. MARGINALS at the FIXED paper architecture NN2 = [32,16] "
              "(activation/lr/alpha only)")
        for axis in ("activation", "lr", "alpha"):
            table = pd.DataFrame({v: box_stats(nn2.loc[nn2[axis] == v, args.score])
                                  for v in order_values(axis, nn2[axis].unique())}).T
            print("  [%s @ NN2]" % axis)
            print("    " + fmt(table).replace("\n", "\n    "))
        print()

    print("2. IMPORTANCE -- functional-ANOVA variance shares (POOLED over all archs)")
    print("   NOTE the 'arch' share is inflated by the deep-net collapse above; the")
    print("   activation/lr/alpha shares are the paper-comparable ones.")
    frame, total, unexplained = fanova(grid, args.score)
    print("    " + fmt(frame[["kind", "term", "share"]]).replace("\n", "\n    "))
    print("    higher-order / unexplained: %.3f" % unexplained)
    print("    -> an axis with a near-zero main share is not worth tuning.")
    print()

    print("3. CORNER CASES")
    good, bad = corners(grid, args.score)
    print("  GOOD corners (high worst-case -- good regardless of the rest):")
    print("    " + fmt(good).replace("\n", "\n    "))
    print("  BAD corners (low best-case -- failure locked in):")
    print("    " + fmt(bad).replace("\n", "\n    "))
    print()
    print("  learning-rate x alpha, median OOS %s (the paper's two-way NN claim):"
          % args.score)
    print("    " + fmt(lr_alpha_interaction(grid, args.score)).replace("\n", "\n    "))
    print()

    print("4. PRIORS -- read off the top-quartile configs")
    prior = prior_from_corners(grid, args.score)
    print("    good = payout >= %.6f  (%d configs)" %
          (prior["cutoff_payout"], prior["n_good"]))
    for axis in ("lr", "alpha"):
        print("    %-6s support among good: %s" % (axis, prior["%s_support" % axis]))
    print("    corr(log lr, log alpha) among good configs: %+.3f" %
          prior["log_lr_alpha_corr_in_good"])
    print("    CAUTION: the paper's rho = +0.9 is an IMPOSED sampling prior on a")
    print("    truncated bivariate normal (for Optuna), not an empirical claim")
    print("    about the good region. And if a slow learning rate is good")
    print("    REGARDLESS of alpha (a one-way corner), the good set spreads across")
    print("    all alphas at low lr, pushing this correlation toward zero -- so a")
    print("    near-zero rho here does NOT refute the paper. Read it with the")
    print("    lr x alpha table above, not on its own.")
    print()

    print("ROBUSTNESS -- main-effect shares on each half of the periods")
    print("    " + fmt(robustness(args.prefix, grid)).replace("\n", "\n    "))
    print("    an importance ranking that survives both halves is a property of")
    print("    the hyperparameter, not of the sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
