"""
E22's report: the one-hot block -- the penalty, the key, the granularity.

    python3 -m src.analysis.e22_report

Grouped by the QUESTION each contrast answers rather than ranked, because the
arms differ along three axes and a single ordering would hide which axis moved.
Every delta is paired per period, Newey-West at lag 20, on an inner join.
"""

import json
import os

import numpy as np
import pandas as pd

from src.analysis.ablation_report import newey_west

PER_PERIOD = os.path.join("results", "per_period", "crypto", "gen_ridge")
METRICS = os.path.join("results", "metrics", "crypto", "gen_ridge")
LAG = 20

ARMS = [
    ("e22_raw", 22, "raw 22, nothing added"),
    ("e22_sym_flat", 768, "raw + ticker dummies, one alpha"),
    ("e22_sym_p10", 768, "ticker dummies 10x"),
    ("e22_sym_p0", 768, "ticker dummies UNPENALISED"),
    ("e22_ucid_flat", 772, "raw + ucid dummies, one alpha"),
    ("e22_ucid_p10", 772, "ucid dummies 10x"),
    ("e22_ucid_p0", 772, "ucid dummies UNPENALISED"),
    ("e22_cap_flat", 25, "raw + low/mid/large, one alpha"),
    ("e22_cap_p0", 25, "cap buckets UNPENALISED"),
    ("e22_only_sym_p0", 746, "ticker dummies ALONE, unpenalised"),
    ("e22_only_sym_flat", 746, "ticker dummies ALONE, one alpha"),
    ("e22_only_ucid_p0", 750, "ucid dummies ALONE, unpenalised"),
    ("e22_only_cap_p0", 3, "cap buckets ALONE, unpenalised"),
]

#: (question, treatment, control) -- each isolates ONE change.
CONTRASTS = [
    ("does the ticker block help at all",      "e22_sym_flat", "e22_raw"),
    ("penalising it 10x",                      "e22_sym_p10", "e22_sym_flat"),
    ("NOT penalising it",                      "e22_sym_p0", "e22_sym_flat"),
    ("the STABLE id, one alpha",               "e22_ucid_flat", "e22_sym_flat"),
    ("the STABLE id, 10x",                     "e22_ucid_p10", "e22_sym_p10"),
    ("the STABLE id, unpenalised",             "e22_ucid_p0", "e22_sym_p0"),
    ("three cap buckets instead of 746 ids",   "e22_cap_flat", "e22_sym_flat"),
    ("cap buckets unpenalised",                "e22_cap_p0", "e22_cap_flat"),
    ("ticker dummies ALONE vs raw 22",         "e22_only_sym_p0", "e22_raw"),
    ("ucid dummies ALONE vs ticker alone",     "e22_only_ucid_p0", "e22_only_sym_p0"),
    ("penalising the alone-block",             "e22_only_sym_flat", "e22_only_sym_p0"),
    ("cap ALONE vs raw 22",                    "e22_only_cap_p0", "e22_raw"),
]


def load(run):
    per = os.path.join(PER_PERIOD, run + ".csv")
    met = os.path.join(METRICS, run + ".json")
    if not (os.path.exists(per) and os.path.exists(met)):
        return None
    return pd.read_csv(per).set_index("date"), json.load(open(met))


def payout(frame):
    return 0.1 * frame["corr"] + frame["mmc"]


def summarise(series):
    cumulative = series.cumsum()
    return (float(series.mean()),
            float(series.mean() / series.std(ddof=0)),
            float((cumulative.expanding(min_periods=1).max() - cumulative).max()))


def delta(control, arm):
    joined = pd.concat([control, arm], axis=1, join="inner",
                       keys=["base", "arm"]).dropna()
    difference = joined["arm"] - joined["base"]
    mean, _, t = newey_west(difference.to_numpy(), lag=LAG)
    return mean, t, len(joined)


def main():
    loaded = {}
    for run, _, _ in ARMS:
        got = load(run)
        if got is not None:
            loaded[run] = got

    print("\n" + "=" * 112)
    print("E22 -- the one-hot block: the penalty, the key, the granularity")
    print("  ucid is the STABLE coin id. 30 tickers in this panel are reused by")
    print("  different coins and 3 coins were renamed, so 2.48% of rows sit on a")
    print("  ticker whose identity changed. Nothing in src/ had ever used ucid.")
    print("=" * 112)
    print("  %-20s %5s %-34s %9s %7s %8s" %
          ("arm", "feat", "what it is", "payout", "p.shp", "dd(pay)"))
    for run, n_features, description in ARMS:
        if run not in loaded:
            print("  %-20s %5d %-34s   (not finished)" % (run, n_features, description))
            continue
        frame, _ = loaded[run]
        level, sharpe, drawdown = summarise(payout(frame))
        print("  %-20s %5d %-34s %9.6f %7.2f %8.2f"
              % (run, n_features, description, level, sharpe, drawdown))

    print("\n" + "=" * 112)
    print("ONE CHANGE AT A TIME -- each row is a paired delta against its own control")
    print("=" * 112)
    print("  %-38s %-20s %11s %7s %5s" %
          ("question", "vs", "d payout", "t", "n"))
    rows = []
    for question, treatment, control in CONTRASTS:
        if treatment not in loaded or control not in loaded:
            print("  %-38s %-20s   (incomplete)" % (question, control))
            continue
        mean, t, n = delta(payout(loaded[control][0]), payout(loaded[treatment][0]))
        flag = "  <--" if abs(t) > 2 else ""
        print("  %-38s %-20s %+11.6f %+7.2f %5d%s"
              % (question, control, mean, t, n, flag))
        rows.append((question, treatment, mean, t))

    if not rows:
        return
    print("\n" + "=" * 112)
    strong = [r for r in rows if abs(r[3]) > 2]
    print("  %d of %d contrasts move at |t| > 2" % (len(strong), len(rows)))
    for question, treatment, mean, t in strong:
        print("    %-40s %+.6f  t %+.2f  (%s)" % (question, mean, t, treatment))
    if not strong:
        low, high = min(r[2] for r in rows), max(r[2] for r in rows)
        print("    None does. Deltas span %+.6f to %+.6f, inside the noise of ~26"
              % (low, high))
        print("    effective independent windows. Neither the penalty on the block,")
        print("    nor fixing the identifier, nor collapsing 746 identities to three")
        print("    cap buckets, changes what the one-hot block is worth.")


if __name__ == "__main__":
    main()
