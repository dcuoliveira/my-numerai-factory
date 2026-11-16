"""
E117's report: the training window of the best single ridge, 1 week to 6 months.

    python3 -m src.analysis.e117_report

Every arm SEARCHES lambda from scratch at every refit. Windows shorter than 130
periods are tuned on the 130 periods ending where the window ends, with inner
folds that each train on the window's own length, then refitted on the window
alone -- so all seven arms pick lambda on the same validation slices and differ
only in how much history the model sees.

Each arm is paired against the 130-period arm, which is production's own search
and reproduces w6r6_lobo_tenc. All rows share one window, so calmar compares.

The lambda block matters as much as the table: a chosen alpha sitting on the
TOP of the glmnet path means the search wanted more shrinkage than the path
offers -- for a short window, "predict as little as possible" -- and that arm's
score is then partly a statement about where the grid stops.
"""

import os

import numpy as np
import pandas as pd

from src.analysis.ablation_report import newey_west

ROOT = os.path.join("results", "%s", "crypto", "ridge_glmnet")
LAG = 20
ARMS = [("e117_w5", 5, "1 week"), ("e117_w10", 10, "2 weeks"),
        ("e117_w15", 15, "3 weeks"), ("e117_w20", 20, "4 weeks"),
        ("e117_w22", 22, "1 month"), ("e117_w65", 65, "3 months"),
        ("e117_w130", 130, "6 months")]
CONTROL = "e117_w130"


def per_period(run):
    path = os.path.join(ROOT % "per_period", run + ".csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).set_index("date")


def hyperparams(run):
    path = os.path.join(ROOT % "hyperparams", run + ".csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def main():
    frames = {r: per_period(r) for r, _, _ in ARMS + [("w6r6_lobo_tenc", 0, "")]}
    frames = {r: f for r, f in frames.items() if f is not None}
    if CONTROL not in frames:
        print("control %s has not finished" % CONTROL)
        return
    pay = pd.DataFrame({r: 0.1 * f["corr"] + f["mmc"] for r, f in frames.items()}).dropna()
    corr = pd.DataFrame({r: f["corr"] for r, f in frames.items()}).loc[pay.index]

    print("\n" + "=" * 106)
    print("E117 -- training window of ridge_glmnet on allfeat_pp minus tenc (811 features)")
    print("  lambda SEARCHED at every refit. %d common dates, %s .. %s, refit every 130."
          % (len(pay), pay.index.min(), pay.index.max()))
    print("=" * 106)
    print("  %-11s %5s %-9s %8s %8s %6s %6s %7s | %11s %7s"
          % ("arm", "win", "", "CORR", "payout", "p.shp", "dd", "calmar", "d vs 6mo", "t"))
    rows = []
    for run, window, label in ARMS:
        if run not in pay:
            print("  %-11s %5d %-9s   (not finished)" % (run, window, label))
            continue
        s = pay[run]; cum = s.cumsum(); dd = float((cum.cummax() - cum).max())
        if run == CONTROL:
            cells = ("-", "-")
        else:
            d, _, t = newey_west((s - pay[CONTROL]).to_numpy(), lag=LAG)
            cells = ("%+.6f" % d, "%+.2f" % t)
            rows.append((run, d, t))
        print("  %-11s %5d %-9s %8.5f %8.5f %6.2f %6.2f %7.3f | %11s %7s"
              % (run, window, label, corr[run].mean(), s.mean(),
                 s.mean() / s.std(ddof=0), dd, s.mean() / dd, cells[0], cells[1]))

    print("\n  lambda chosen per refit (glmnet scale, alpha / training rows)")
    print("  ^ = on the TOP of the path (wanted more shrinkage than offered)"
          "   v = on the bottom")
    print("  %-11s %-56s %s" % ("arm", "lambda at refit 0 / 1 / 2 / 3", "spread"))
    for run, _, _ in ARMS:
        h = hyperparams(run)
        if h is None:
            continue
        cells, lams = [], []
        for _, r in h.iterrows():
            lam = r["alpha"] / r["train_rows"]
            lams.append(lam)
            mark = ""
            if "alpha_grid_max" in h and np.isclose(r["alpha"], r["alpha_grid_max"], rtol=1e-9):
                mark = "^"
            elif "alpha_grid_min" in h and np.isclose(r["alpha"], r["alpha_grid_min"], rtol=1e-9):
                mark = "v"
            cells.append("%.3g%s" % (lam, mark))
        print("  %-11s %-56s %.0fx" % (run, "  ".join(cells), max(lams) / min(lams)))

    if "w6r6_lobo_tenc" in pay:
        d, _, t = newey_west((pay[CONTROL] - pay["w6r6_lobo_tenc"]).to_numpy(), lag=LAG)
        print("\n  reproduction: e117_w130 vs the original w6r6_lobo_tenc  d %+.6f  t %+.2f  "
              "max|diff| per period %.2e"
              % (d, t, float(np.abs(pay[CONTROL] - pay["w6r6_lobo_tenc"]).max())))

    if rows:
        print("\n" + "=" * 106)
        best = max([r for r, _, _ in ARMS if r in pay], key=lambda r: pay[r].mean())
        print("  best window by payout: %s" % best)
        worse = [r for r, d, t in rows if t < -2]
        better = [r for r, d, t in rows if t > 2]
        print("  significantly WORSE than 6 months (t < -2): %s" % (", ".join(worse) or "none"))
        print("  significantly BETTER than 6 months (t > +2): %s" % (", ".join(better) or "none"))


if __name__ == "__main__":
    main()
