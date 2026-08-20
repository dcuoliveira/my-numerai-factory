"""
Paired per-period comparison of experiment arms against a baseline.

    python3 .claude/skills/crypto-experiment-reporting/scripts/paired_delta.py \
        --model ridge --baseline c02_raw --arms c02_raw_ae

    # members living in another model's tree use model/run
    python3 .../paired_delta.py --model ridge_glmnet --baseline w6r6_glmnet_reg \
        --arms w6r6_alone_tenc ridge/w6r6_grp_size_fixed

Why this exists rather than a one-liner per experiment: every arm study in this
repo needs the same comparison, and the two ways of getting it wrong are quiet.
Comparing the arms' MEANS throws away the pairing -- the arms share their periods,
folds and seed, so almost all of their variance is common, and the unpaired
standard error is several times too wide. And treating the per-period series as
independent ignores that a 20-day forward target on daily dates leaves
consecutive periods ~95% overlapping, which inflates every t by roughly sqrt(20).

The same numbers on the same data: t = 0.75 unpaired, t = 3.14 paired
(src/analysis/ablation_report.py). Two orders of magnitude of sensitivity.

Reads results/per_period/<tournament>/<model>/<run>.csv, joins on the period
index so a delta is only ever taken between two scores of the same date, and
reports Newey-West standard errors at lag 20 via the repo's own implementation.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.getcwd())
from src.analysis.ablation_report import newey_west  # noqa: E402

#: Numerai Crypto's payout weighting. Kept here so every report uses one formula.
PAYOUT_CORR_WEIGHT = 0.1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Paired per-period deltas against a baseline run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tournament", default="crypto")
    parser.add_argument("--model", default="ridge_glmnet",
                        help="model tree the runs live in; a member may override "
                             "it with model/run")
    parser.add_argument("--baseline", required=True,
                        help="the run every arm is measured against")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--metrics", nargs="+",
                        default=["corr", "mmc", "payout"])
    parser.add_argument("--lag", type=int, default=20,
                        help="Newey-West lag; 20 matches the target horizon")
    return parser.parse_args(argv)


def load(tournament, default_model, name):
    """
    One run's per-period scores, indexed by period.

    :param name: "run" in the default model tree, or "model/run" to cross trees
    :type name: str
    :return: the frame, with a payout column added
    :rtype: pd.DataFrame
    """
    model, _, run = name.rpartition("/")
    path = os.path.join("results", "per_period", tournament,
                        model or default_model, "%s.csv" % run)
    if not os.path.exists(path):
        raise SystemExit(
            "no per-period file at %s -- has that arm been run?" % path)

    frame = pd.read_csv(path, index_col=0)
    frame["payout"] = PAYOUT_CORR_WEIGHT * frame["corr"] + frame["mmc"]
    return frame


def compare(base, arm, metrics, lag):
    """
    Paired deltas for one arm.

    The join is INNER on the period index, so a period either arm failed to
    score drops out of both rather than silently biasing one side.

    :return: (rows, n_periods)
    :rtype: ([dict], int)
    """
    joined = base.join(arm, lsuffix="_base", rsuffix="_arm", how="inner")

    rows = []
    for metric in metrics:
        delta = (joined[metric + "_arm"] - joined[metric + "_base"]).dropna()
        mean, se, t = newey_west(delta, lag=lag)
        rows.append({"metric": metric,
                     "base": joined[metric + "_base"].mean(),
                     "arm": joined[metric + "_arm"].mean(),
                     "delta": mean, "se": se, "t": t})
    return rows, len(joined)


def main(argv=None):
    args = parse_args(argv)
    base = load(args.tournament, args.model, args.baseline)

    print("baseline: %s (%d periods)\n" % (args.baseline, len(base)))

    for name in args.arms:
        arm = load(args.tournament, args.model, name)
        rows, n = compare(base, arm, args.metrics, args.lag)

        if n != len(base) or n != len(arm):
            print("NOTE %s: baseline has %d periods, arm has %d, %d shared"
                  % (name, len(base), len(arm), n))

        print("%s  vs  %s   (%d paired periods)" % (name, args.baseline, n))
        print("  %-8s %11s %11s %13s %10s %8s" % (
            "metric", "baseline", "arm", "paired delta", "NW se", "t"))
        for row in rows:
            print("  %-8s %11.6f %11.6f %+13.6f %10.6f %+8.2f" % (
                row["metric"], row["base"], row["arm"], row["delta"],
                row["se"], row["t"]))

        payout = next((r for r in rows if r["metric"] == "payout"), rows[-1])
        if payout["t"] > 2:
            verdict = "HELPS at |t| > 2"
        elif payout["t"] < -2:
            verdict = "HURTS at |t| > 2"
        else:
            verdict = ("no distinguishable difference -- on ~26 effective "
                       "independent windows this is the expected result, and "
                       "is itself the finding")
        print("  verdict: %s\n" % verdict)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
