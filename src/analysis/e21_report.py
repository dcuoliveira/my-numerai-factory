"""
E21's report: cluster regularisation on allfeat_pp.

    python3 -m src.analysis.e21_report

Every arm shares e21_flat's estimator, alpha grid, folds and seed and differs
only in the SHAPE of Delta, so each delta is paired and attributable. Grouped by
MECHANISM rather than by score, because the three do different things and a
ranked list would hide that:

  (a) level up      penalise a derived block harder than raw
  (b) level down    penalise raw less -- which changes the lambda -> inf limit
  (c) dispersion    fuse within clusters instead of shrinking them
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from src.analysis.ablation_report import newey_west

PER_PERIOD = os.path.join("results", "per_period", "crypto", "gen_ridge")
METRICS = os.path.join("results", "metrics", "crypto", "gen_ridge")
LAG = 20

CONTROL = "flat"

#: (mechanism, arm suffix, what it does)
ARMS = [
    ("(a) penalise derived MORE", "int_x10", "interactions 10x"),
    ("(a) penalise derived MORE", "int_x100", "interactions 100x"),
    ("(a) penalise derived MORE", "onehot_x10", "740 dummies 10x"),
    ("(a) penalise derived MORE", "onehot_x100", "740 dummies 100x"),
    ("(a) penalise derived MORE", "derived_x10", "everything non-raw 10x"),
    ("(b) penalise raw LESS", "rawfree", "raw x1e-3"),
    ("(b) penalise raw LESS", "rawfree_hard", "raw x1e-6"),
    ("(b) penalise raw LESS", "rawfree_oh10", "raw x1e-3 + dummies 10x"),
    ("(c) fuse, do not shrink", "fuse", "within-stem fusion, rho searched"),
    ("searched", "searched", "the search picks the shape"),
]


def load(run):
    per = os.path.join(PER_PERIOD, run + ".csv")
    met = os.path.join(METRICS, run + ".json")
    if not (os.path.exists(per) and os.path.exists(met)):
        return None
    return pd.read_csv(per).set_index("date"), json.load(open(met))


def payout(frame):
    """Per-period payout, formed here rather than from the stored summaries."""
    return 0.1 * frame["corr"] + frame["mmc"]


def summarise(series):
    cumulative = series.cumsum()
    return (float(series.mean()),
            float(series.mean() / series.std(ddof=0)),
            float((cumulative.expanding(min_periods=1).max() - cumulative).max()))


def chosen(metrics, key):
    """What the search picked for `key` at each refit, as a compact string."""
    picked = []
    for refit in metrics.get("refits", []):
        value = (refit.get("best_params") or {}).get(key)
        if value is not None:
            picked.append(value if isinstance(value, str)
                          else "%.3g" % float(value))
    return "/".join(picked) if picked else "-"


def main():
    scout = len(sys.argv) > 1 and sys.argv[1].startswith("scout")
    prefix = "e21scout_" if scout else "e21_"

    control = load(prefix + CONTROL)
    if control is None:
        print("control %s%s has not finished" % (prefix, CONTROL))
        return
    control_frame, control_metrics = control
    control_pay = payout(control_frame)
    level, sharpe, drawdown = summarise(control_pay)

    print("\n" + "=" * 118)
    print("E21 -- cluster regularisation on allfeat_pp (812 cols: raw 22, spread 11,")
    print("       interact 14, rollsel 24, tenc 1, onehot 740)")
    print("  every arm shares the control's estimator, alpha grid, folds and seed.")
    print("  only the SHAPE of Delta differs.")
    print("=" * 118)
    print("  %-22s %-30s %9s %8s %8s | %11s %7s | %s"
          % ("arm", "what it does", "payout", "p.shp", "dd(pay)", "d payout", "t",
             "alpha chosen"))
    print("  %-22s %-30s %9.6f %8.2f %8.2f | %11s %7s | %s"
          % ("e21_flat", "CONTROL: one alpha on all 812", level, sharpe, drawdown,
             "-", "-", chosen(control_metrics, "alpha")))

    rows = []
    last_mechanism = None
    for mechanism, suffix, description in ARMS:
        if mechanism != last_mechanism:
            print("  " + "-" * 114)
            print("  %s" % mechanism)
            last_mechanism = mechanism
        loaded = load(prefix + suffix)
        if loaded is None:
            print("  %-22s %-30s   (not finished)" % ("e21_" + suffix, description))
            continue
        frame, metrics = loaded
        pay = payout(frame)
        arm_level, arm_sharpe, arm_dd = summarise(pay)
        joined = pd.concat([control_pay, pay], axis=1, join="inner",
                           keys=["base", "arm"]).dropna()
        difference = joined["arm"] - joined["base"]
        mean, _, t = newey_west(difference.to_numpy(), lag=LAG)
        extra = chosen(metrics, "alpha")
        if suffix == "fuse":
            extra += "  rho " + chosen(metrics, "group_penalty")
        if suffix == "searched":
            extra = chosen(metrics, "group_spec")
        print("  %-22s %-30s %9.6f %8.2f %8.2f | %+11.6f %7.2f | %s"
              % ("e21_" + suffix, description, arm_level, arm_sharpe, arm_dd,
                 mean, t, extra))
        rows.append((mechanism, suffix, mean, t))

    if not rows:
        print("\n  nothing to compare yet")
        return

    print("\n" + "=" * 118)
    strong = [r for r in rows if abs(r[3]) > 2]
    print("  %d of %d arms differ from one-alpha-on-everything at |t| > 2"
          % (len(strong), len(rows)))
    for mechanism, suffix, mean, t in strong:
        print("    %-22s %+.6f  t %+.2f   [%s]" % ("e21_" + suffix, mean, t, mechanism))
    if not strong:
        low, high = min(r[2] for r in rows), max(r[2] for r in rows)
        print("    None does. Payout deltas span %+.6f to %+.6f, inside the noise of"
              % (low, high))
        print("    ~26 effective independent windows. Shrinking the 740 ticker dummies")
        print("    harder than the raw 22 -- the thing one alpha across the whole")
        print("    design matrix cannot do -- does not recover what allfeat_pp loses")
        print("    against the raw 22. That is the finding.")

    print("\n  by mechanism, mean payout delta:")
    for mechanism in ("(a) penalise derived MORE", "(b) penalise raw LESS",
                      "(c) fuse, do not shrink", "searched"):
        block = [r[2] for r in rows if r[0] == mechanism]
        if block:
            print("    %-26s %+.6f  over %d arm(s), %d positive"
                  % (mechanism, float(np.mean(block)), len(block),
                     sum(1 for v in block if v > 0)))


if __name__ == "__main__":
    main()
