"""
Read the ablation arms and rank features by what removing them costs.

    python3 -m src.analysis.ablation_report                  # every family
    python3 -m src.analysis.ablation_report --family lobo
    python3 -m src.analysis.ablation_report --plot

------------------------------------------------------------------------------
Why the deltas are PAIRED, and why that is the whole point
------------------------------------------------------------------------------

Every arm is a near-copy of its reference: same data, same seed, same folds,
differing by one block of columns. Their per-period score series therefore move
together, and the DIFFERENCE series has a far smaller standard error than either
series alone.

The size of that effect in this study: comparing two arms 0.0085 apart on mean
CORR gave t = 0.75, while a paired test on an effect worth 0.0003 gave t = 3.14.
Two orders of magnitude of sensitivity, from the same 510 periods. Ranking
ablations on unpaired means would find nothing and conclude, wrongly, that no
feature matters.

Standard errors use Newey-West at lag 20 because a 20-day forward target on
daily dates makes consecutive periods ~95% overlapping. Ignoring that inflates
every t-statistic by roughly sqrt(20).

------------------------------------------------------------------------------
Sign convention
------------------------------------------------------------------------------

IMPORTANCE is reported as reference minus arm, so it reads as "what the block
was worth":

    importance > 0    removing it HURT   -> the block was contributing
    importance < 0    removing it HELPED -> the block was noise the model paid
                                            for, and dropping it is a free win

The second case is not a curiosity. glmnet applies one alpha across every
column, so a block of pure noise raises the shrinkage the useful columns get.
Removing it can improve the model without adding any information.

------------------------------------------------------------------------------
Reference per family
------------------------------------------------------------------------------

    lobo, full_loo   allfeat++ (812 features)   w6r6_glmnet_reg_allfeatpp
    raw_loo          the 22 raw baseline        w6r6_glmnet_reg
    alone            no reference -- these are absolute scores, ranked among
                     themselves and against allfeat++ for context
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

METRICS = "results/metrics/crypto/ridge_glmnet"
PER_PERIOD = "results/per_period/crypto/ridge_glmnet"

FAMILIES = {
    "lobo":     {"prefix": "w6r6_lobo_",     "reference": "w6r6_glmnet_reg_allfeatpp"},
    "alone":    {"prefix": "w6r6_alone_",    "reference": None},
    "raw_loo":  {"prefix": "w6r6_rawloo_",   "reference": "w6r6_glmnet_reg"},
    "full_loo": {"prefix": "w6r6_fullloo_",  "reference": "w6r6_glmnet_reg_allfeatpp"},
}

# The 20-day forward target makes consecutive daily periods ~95% overlapping.
NEWEY_WEST_LAG = 20


def payout(frame):
    """Crypto pays 0.1*CORR + 1*MMC. Rank on what pays, not on CORR."""
    return 0.1 * frame["corr"] + frame["mmc"]


def load(run):
    """(per-period frame, metrics dict) for one run, or (None, None)."""
    per_period = os.path.join(PER_PERIOD, run + ".csv")
    metrics = os.path.join(METRICS, run + ".json")
    if not (os.path.exists(per_period) and os.path.exists(metrics)):
        return None, None
    frame = pd.read_csv(per_period).set_index("date")
    frame["payout"] = payout(frame)
    with open(metrics) as handle:
        return frame, json.load(handle)


def newey_west(series, lag=NEWEY_WEST_LAG):
    """(mean, standard error, t) for the mean of an autocorrelated series."""
    x = np.asarray(series, dtype="float64")
    n = len(x)
    mean = x.mean()
    e = x - mean
    variance = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        variance += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(variance, 0.0) / n)
    return mean, se, (mean / se if se > 0 else np.nan)


def summarize(series):
    """mean, sharpe and max drawdown of a per-period score series."""
    x = np.asarray(series, dtype="float64")
    mean, sd = x.mean(), x.std(ddof=0)
    cumulative = x.cumsum()
    drawdown = (np.maximum.accumulate(cumulative) - cumulative).max()
    return mean, (mean / sd if sd > 0 else np.nan), drawdown


def collect(family):
    """Every completed arm in a family, as a list of records."""
    spec = FAMILIES[family]
    rows = []
    for path in sorted(glob.glob(os.path.join(METRICS, spec["prefix"] + "*.json"))):
        run = os.path.basename(path)[:-5]
        frame, metrics = load(run)
        if frame is None:
            continue
        mean, sharpe, drawdown = summarize(frame["payout"])
        rows.append({
            "run": run,
            "block": run[len(spec["prefix"]):],
            "n_features": metrics.get("config", {}).get("n_features"),
            "frame": frame, "payout": mean, "sharpe": sharpe, "maxdd": drawdown,
            "corr": frame["corr"].mean(),
        })
    return rows, spec["reference"]


def report(family, verbose=True):
    rows, reference_run = collect(family)
    if not rows:
        print("%-9s no completed arms yet" % family)
        return None

    reference = None
    if reference_run:
        reference_frame, _ = load(reference_run)
        if reference_frame is None:
            print("%-9s %d arms done, but the reference run '%s' has not been "
                  "run yet -- absolute scores only" % (family, len(rows), reference_run))
        else:
            reference = reference_frame

    print("\n" + "=" * 100)
    print("FAMILY %s   (%d arms)%s" % (
        family.upper(), len(rows),
        "   reference: %s" % reference_run if reference is not None else ""))
    print("=" * 100)

    if reference is None:
        # `alone` has no reference: report absolute scores, ranked.
        rows.sort(key=lambda r: -r["payout"])
        print("%-16s %6s %10s %9s %9s %9s" % (
            "BLOCK ALONE", "feat", "payout", "sharpe", "maxdd", "CORR"))
        for row in rows:
            print("%-16s %6d %10.6f %9.4f %9.4f %9.6f" % (
                row["block"], row["n_features"], row["payout"],
                row["sharpe"], row["maxdd"], row["corr"]))
        return rows

    reference_payout = reference["payout"]
    for row in rows:
        # Align on date: an arm and its reference always cover the same periods,
        # but an inner join makes a mismatch impossible to miss instead of
        # silently comparing different denominators.
        joined = pd.concat([reference_payout, row["frame"]["payout"]],
                           axis=1, join="inner", keys=["ref", "arm"])
        delta = joined["arm"] - joined["ref"]
        mean, se, t = newey_west(delta)
        row["d_payout"] = mean
        row["se"] = se
        # Everything downstream reports IMPORTANCE (= reference minus arm), so
        # carry its t, not the delta's. Keeping both signs in play is how the
        # verdict column ended up inverted: it tested the delta's t while the
        # table printed the importance's, and labelled the one block that
        # actually contributes as one to drop.
        row["importance"] = -mean          # what removal COST
        row["t"] = -t                      # t of the importance, same sign
        row["n_periods"] = len(joined)

    rows.sort(key=lambda r: -r["importance"])
    reference_mean, reference_sharpe, reference_dd = summarize(reference_payout)
    print("reference payout %.6f   sharpe %.4f   maxdd %.4f   n=%d\n" % (
        reference_mean, reference_sharpe, reference_dd, len(reference_payout)))
    print("%-22s %6s %10s %10s %8s %9s %9s  %s" % (
        "REMOVED", "feat", "payout", "importance", "t", "sharpe", "maxdd", "verdict"))
    for row in rows:
        # importance > 0 means removing the block HURT, i.e. it was contributing.
        if row["t"] > 2:
            verdict = "CONTRIBUTES -- keep it"
        elif row["t"] < -2:
            verdict = "COSTS -- drop it"
        else:
            verdict = "indistinguishable"
        print("%-22s %6d %10.6f %+10.6f %8.2f %9.4f %9.4f  %s" % (
            row["block"], row["n_features"], row["payout"], row["importance"],
            row["t"], row["sharpe"], row["maxdd"], verdict))

    strong = [r for r in rows if abs(r["t"]) > 2]
    print("\n%d of %d arms differ from the reference at |t| > 2" % (len(strong), len(rows)))
    if not strong:
        print("Nothing is distinguishable. On ~26 effective independent windows "
              "that is the expected result for a saturated feature set, and it "
              "is itself the finding.")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Rank features by ablation cost.")
    parser.add_argument("--family", default=None, choices=sorted(FAMILIES))
    parser.add_argument("--plot", action="store_true",
                        help="write results/plots/crypto/ablation_importance.png")
    args = parser.parse_args()

    families = [args.family] if args.family else list(FAMILIES)
    collected = {}
    for family in families:
        result = report(family)
        if result:
            collected[family] = result

    if args.plot and collected:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plottable = {k: v for k, v in collected.items() if "importance" in v[0]}
        if not plottable:
            print("\nnothing to plot yet")
            return
        figure, axes = plt.subplots(1, len(plottable),
                                    figsize=(7 * len(plottable), 6), squeeze=False)
        for ax, (family, rows) in zip(axes[0], plottable.items()):
            rows = sorted(rows, key=lambda r: r["importance"])
            labels = [r["block"] for r in rows]
            values = [r["importance"] for r in rows]
            errors = [1.96 * r["se"] for r in rows]
            colours = ["#2c7fb8" if abs(r["t"]) > 2 else "#bdbdbd" for r in rows]
            ax.barh(range(len(rows)), values, xerr=errors, color=colours,
                    error_kw={"ecolor": "#555", "lw": 1})
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.axvline(0, color="black", lw=1)
            ax.set_xlabel("payout lost by removing it  (95% CI)")
            ax.set_title("%s -- importance by ablation" % family)
        os.makedirs("results/plots/crypto", exist_ok=True)
        out = "results/plots/crypto/ablation_importance.png"
        figure.savefig(out, dpi=140, bbox_inches="tight")
        print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
