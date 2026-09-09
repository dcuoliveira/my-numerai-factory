"""
Hyperparameter sensitivity, in the style of Kalfa, Timmermann & van der Zwan (2025).

    python3 -m src.analysis.hyper_sensitivity --prefix e15_
    python3 -m src.analysis.hyper_sensitivity --prefix e15_ --metric corr

The paper's method, adapted to LightGBM on crypto: over a FACTORIAL grid of
hyperparameters, fix one hyperparameter to each of its values and marginalize
over all the others. The resulting DISTRIBUTION of out-of-sample performance
answers three questions the mean of a single run cannot:

  1. CORNER CASES -- a value that "locks in" good or bad performance regardless
     of how the other hyperparameters are set. In the paper, XGBoost depth in
     {1,2} virtually guarantees a positive R2; depth 6 or a high learning rate
     virtually guarantees a bad one. The tell is a value whose whole quartile
     box sits high (or low) and NARROW.

  2. THE PRIOR -- the sub-range where the median is good and the spread is small
     is the range worth searching; the rest can be excluded up front. Narrow,
     well-chosen ranges beat wide ones out of sample.

  3. WHICH HYPERPARAMETERS MATTER -- rank them by how far moving the value shifts
     the median AND widens the range. A hyperparameter whose marginals are all
     the same box does not matter; one whose boxes march up or down, or fan out,
     does.

Everything is read from each run's recorded `config`: the swept dimensions are
discovered as the fixed_params keys that take more than one value across the
runs, so this script does not need to be told the grid. The objective is the
per-period payout series (0.1*corr + mmc), summarised the same way the rest of
the repo summarises series.

Two adaptations from the paper are deliberate and are reported, not hidden:

  * We have no early stopping, so n_estimators is a real capacity dial rather
    than a truncated ceiling. A learning_rate marginal is therefore confounded
    with total learning. --interact learning_rate n_estimators prints the joint,
    and the derived quantity lr*n_estimators is reported so the confound is
    visible rather than assumed away.

  * In LightGBM num_leaves (with max_depth=-1) is the complexity control the
    paper calls "depth". The num_leaves x learning_rate plane is what tests the
    paper's negative depth/learning-rate prior here.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

METRICS = "results/metrics/crypto/lgb"
PER_PERIOD = "results/per_period/crypto/lgb"


def payout(frame):
    """Crypto pays 0.1 * corr + mmc, per period."""
    return 0.1 * frame["corr"] + frame["mmc"]


def series_stats(series):
    """Summarise a per-period payout series the way the repo does elsewhere."""
    x = np.asarray(series, dtype="float64")
    if x.size == 0:
        return dict(payout=np.nan, sharpe=np.nan, maxdd=np.nan)
    cum = x.cumsum()
    sd = x.std(ddof=0)
    return dict(
        payout=float(x.mean()),
        sharpe=float(x.mean() / sd) if sd > 0 else np.nan,
        maxdd=float((np.maximum.accumulate(cum) - cum).max()),
    )


def load(prefix, objective):
    """Every run under the prefix, one row per run, with its config unpacked."""
    rows = []
    for path in sorted(glob.glob(os.path.join(METRICS, prefix + "*.json"))):
        run = os.path.basename(path)[:-5]
        pp = os.path.join(PER_PERIOD, run + ".csv")
        if not os.path.exists(pp):
            continue
        d = json.load(open(path))
        config = d.get("config", {})
        fixed = config.get("fixed_params") or {}
        frame = pd.read_csv(pp)
        st = series_stats(payout(frame))
        row = dict(run=run, n_periods=len(frame),
                   corr=d["corr"]["mean"], mmc=d["mmc"]["mean"], **st)
        row["objective"] = row.get(objective, row["payout"])
        for k, v in fixed.items():
            row["hp_" + k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def swept_dimensions(table):
    """The fixed_params keys that take more than one value = the grid axes."""
    dims = []
    for col in [c for c in table.columns if c.startswith("hp_")]:
        vals = table[col].dropna().unique()
        if len(vals) > 1:
            dims.append(col)
    return dims


def _box(values):
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return dict(n=v.size, lo=v.min(), q1=np.percentile(v, 25), med=np.median(v),
                q3=np.percentile(v, 75), hi=v.max())


def marginals(table, dim, objective_col="objective"):
    """Distribution of the objective for each value of one hyperparameter."""
    out = []
    for value, grp in table.groupby(dim):
        b = _box(grp[objective_col])
        if b:
            out.append((value, b))
    return sorted(out, key=lambda kv: kv[0])


def importance(table, dims, objective_col="objective"):
    """
    Rank hyperparameters two ways, matching the paper's twin criteria.

    range_of_medians  -- how far the value shifts the MEDIAN outcome. A big
                         number means the parameter matters for the typical model.
    max_box_spread    -- the widest inter-quartile range any single value
                         produces. A big number means some setting of this
                         parameter makes the outcome UNSTABLE -- the paper's
                         "widens the range, bottom quartile goes negative".
    """
    rows = []
    for dim in dims:
        medians, spreads = [], []
        for _, b in marginals(table, dim, objective_col):
            medians.append(b["med"])
            spreads.append(b["q3"] - b["q1"])
        rows.append((dim[3:], max(medians) - min(medians), max(spreads),
                     min(medians), max(medians)))
    return sorted(rows, key=lambda r: -r[1])


def fmt_box(value, b):
    return ("  %-10s  n=%-3d  [min %8.5f | q1 %8.5f | MED %8.5f | q3 %8.5f | max %8.5f]"
            % (str(value), b["n"], b["lo"], b["q1"], b["med"], b["q3"], b["hi"]))


def print_marginals(table, dims, objective, objective_col="objective"):
    print("=" * 92)
    print("MARGINAL DISTRIBUTIONS of %s -- fix one hyperparameter, vary all others"
          % objective)
    print("(the paper's Figure 6: a box that sits high AND narrow is a corner "
          "that locks in a good model)")
    print("=" * 92)
    for dim in dims:
        m = marginals(table, dim, objective_col)
        print("\n%s  (%d values, %d runs each on average)"
              % (dim[3:], len(m), int(np.mean([b["n"] for _, b in m]))))
        for value, b in m:
            print(fmt_box(value, b))


def print_importance(table, dims, objective, objective_col="objective"):
    print("\n" + "=" * 92)
    print("WHICH HYPERPARAMETERS MATTER  (ranked by how far the value shifts the median %s)"
          % objective)
    print("=" * 92)
    print("  %-18s %14s %14s   %s" % ("hyperparameter", "median-shift",
                                      "worst-spread", "median range"))
    print("  " + "-" * 74)
    for name, mshift, spread, mlo, mhi in importance(table, dims, objective_col):
        print("  %-18s %14.6f %14.6f   [%.5f .. %.5f]"
              % (name, mshift, spread, mlo, mhi))
    print("\n  median-shift: a big value = the parameter moves the TYPICAL model.")
    print("  worst-spread: a big value = some setting makes the outcome UNSTABLE.")


def print_corners(table, dims, objective, objective_col="objective", k=3):
    print("\n" + "=" * 92)
    print("CORNER CASES  (values whose whole quartile box locks the outcome high or low)")
    print("=" * 92)
    good, bad = [], []
    overall_med = table[objective_col].median()
    for dim in dims:
        for value, b in marginals(table, dim, objective_col):
            # locks GOOD: even the worst quartile beats the overall median.
            if b["q1"] > overall_med:
                good.append((b["q1"], dim[3:], value, b))
            # locks BAD: even the best quartile is below it.
            if b["q3"] < overall_med:
                bad.append((b["q3"], dim[3:], value, b))
    print("\noverall median %s across all %d runs: %.6f\n"
          % (objective, len(table), overall_med))
    print("LOCKS IN GOOD -- worst quartile still above the overall median:")
    if good:
        for q1, name, value, b in sorted(good, reverse=True):
            print("  %-14s = %-10s  q1 %.6f  med %.6f  (never worse than the median)"
                  % (name, value, b["q1"], b["med"]))
    else:
        print("  none -- no single value guarantees an above-median model here")
    print("\nLOCKS IN BAD -- best quartile still below the overall median:")
    if bad:
        for q3, name, value, b in sorted(bad):
            print("  %-14s = %-10s  q3 %.6f  med %.6f  (never better than the median)"
                  % (name, value, b["q3"], b["med"]))
    else:
        print("  none -- no single value guarantees a below-median model here")


def print_interaction(table, a, b, objective, objective_col="objective"):
    ca, cb = "hp_" + a, "hp_" + b
    if ca not in table.columns or cb not in table.columns:
        print("\n(interaction %s x %s: one of them was not swept)" % (a, b))
        return
    print("\n" + "=" * 92)
    print("INTERACTION  %s (rows) x %s (cols)  --  median %s in each cell"
          % (a, b, objective))
    print("=" * 92)
    pivot = table.pivot_table(index=ca, columns=cb, values=objective_col,
                              aggfunc="median")
    print(pivot.round(5).to_string())
    # The paper's total-learning confound, made explicit.
    if {a, b} == {"learning_rate", "n_estimators"}:
        t = table.copy()
        t["lr_x_ntrees"] = t[ca] * t[cb] if a == "learning_rate" else t[cb] * t[ca]
        t["lr_x_ntrees"] = t["hp_learning_rate"] * t["hp_n_estimators"]
        print("\nby total learning (learning_rate * n_estimators) -- decouples the "
              "no-early-stopping confound:")
        t["bucket"] = pd.qcut(t["lr_x_ntrees"], min(6, t["lr_x_ntrees"].nunique()),
                              duplicates="drop")
        for bucket, grp in t.groupby("bucket", observed=True):
            bx = _box(grp[objective_col])
            print("  lr*ntrees in %-22s med %.6f  (n=%d)"
                  % (str(bucket), bx["med"], bx["n"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="e15_",
                    help="run-name prefix identifying the sweep")
    ap.add_argument("--metric", default="payout",
                    choices=["payout", "sharpe", "corr", "mmc", "maxdd"],
                    help="the objective to slice by")
    ap.add_argument("--interact", nargs=2, metavar=("A", "B"),
                    default=["num_leaves", "learning_rate"],
                    help="two hyperparameters whose joint median surface to print")
    args = ap.parse_args()

    table = load(args.prefix, args.metric)
    if table.empty:
        print("no runs found under prefix %r in %s" % (args.prefix, METRICS))
        return
    dims = swept_dimensions(table)
    if not dims:
        print("no hyperparameter varies across these runs -- nothing to slice")
        return

    npd = sorted(table["n_periods"].unique())
    print("sweep %r: %d runs, objective = %s, %d swept dimensions"
          % (args.prefix, len(table), args.metric, len(dims)))
    print("dimensions: %s" % ", ".join(d[3:] for d in dims))
    if len(npd) > 1:
        print("WARNING: runs span %s period counts %s -- not one common window; "
              "marginals mix panels" % (len(npd), npd))
    else:
        print("all runs scored on %d periods (one common window)" % npd[0])

    print_importance(table, dims, args.metric)
    print_corners(table, dims, args.metric)
    print_marginals(table, dims, args.metric)
    print_interaction(table, args.interact[0], args.interact[1], args.metric)
    # The learning-rate x trees confound is always worth showing for boosters.
    if args.interact != ["learning_rate", "n_estimators"]:
        print_interaction(table, "learning_rate", "n_estimators", args.metric)


if __name__ == "__main__":
    main()
