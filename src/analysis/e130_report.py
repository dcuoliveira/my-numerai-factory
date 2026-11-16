"""
E130's report: tree-leaf blocks on the best ridge and the best MLP, each arm paired
against its same-seed in-study control.

    python -m src.analysis.e130_report

Newey-West at lag 20 on the per-period differences of CORR, MMC and payout, the
payout rebuilt per period (0.1 * corr + mmc) before anything is summarised. Every
row is on the arms' common window, so calmar and burn compare. Two checks come
first: r_ctl must reproduce e118_ridge_x29 (same data, same protocol), and the two
MLP seeds give the noise floor a single-seed MLP delta has to clear. Arms that have
not landed are named, not dropped silently.
"""

import os

import numpy as np
import pandas as pd

from src.analysis.cross_target import newey_west
from src.analysis.results_table import burn_days
from src.prediction.metrics import summarize

ARMS = [("ridge_glmnet", a) for a in ("r_ctl", "r_cat", "r_rf", "r_both")] + \
       [("mlp", a) for a in ("m_ctl", "m_cat", "m_rf", "m_both", "m_ctl_s43", "m_both_s43")]
PAIRS = [("r_cat", "r_ctl"), ("r_rf", "r_ctl"), ("r_both", "r_ctl"),
         ("m_cat", "m_ctl"), ("m_rf", "m_ctl"), ("m_both", "m_ctl"),
         ("m_both_s43", "m_ctl_s43"), ("m_ctl_s43", "m_ctl")]
STORED = [("r_ctl", "ridge_glmnet/e118_ridge_x29"), ("m_ctl", "mlp/e15_mlp_fenccausal")]


def series(model, run):
    path = os.path.join("results", "per_period", "crypto", model, run + ".csv")
    if not os.path.exists(path):
        return None
    f = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    f["payout"] = 0.1 * f["corr"] + f["mmc"]
    return f


def stats(f):
    s, cs = summarize(f["payout"]), summarize(f["corr"])
    dd = s.get("max_drawdown") or np.nan
    return dict(CORR=f["corr"].mean(), MMC=f["mmc"].mean(), payout=s["mean"], c_sharpe=cs.get("sharpe"),
                p_sharpe=s.get("sharpe"), dd_pay=dd, calmar=s["mean"] / dd,
                burn=burn_days(f.reset_index()), n=len(f))


def paired(a, b):
    row = {}
    for col in ("corr", "mmc", "payout"):
        mean, t = newey_west((a[col] - b[col]).to_numpy())
        row["d" + col], row["t_" + col] = mean, t
    return row


def main():
    frames = {arm: series(model, "e130_" + arm) for model, arm in ARMS}
    missing = [arm for arm, f in frames.items() if f is None]
    frames = {arm: f for arm, f in frames.items() if f is not None}
    if not frames:
        print("E130: no arm has landed yet")
        return
    common = sorted(set.intersection(*[set(f.index) for f in frames.values()]))
    pd.set_option("display.width", 200)
    print("E130 tree-leaf blocks -- %d arms on %d common periods (%s -> %s)"
          % (len(frames), len(common), common[0].date(), common[-1].date()))
    if missing:
        print("not landed: " + " ".join("e130_" + m for m in missing))
    table = pd.DataFrame({arm: stats(f.loc[common]) for arm, f in frames.items()}).T
    print("\n" + table.round(5).to_string())

    rows = []
    for a, b in PAIRS:
        if a in frames and b in frames:
            rows.append(dict(arm=a, vs=b, **paired(frames[a].loc[common], frames[b].loc[common])))
    for a, ref in STORED:
        model, run = ref.split("/")
        f = series(model, run)
        if a in frames and f is not None:
            dates = sorted(set(frames[a].index) & set(f.index))
            rows.append(dict(arm=a, vs=ref, **paired(frames[a].loc[dates], f.loc[dates])))
    if rows:
        print("\npaired Newey-West lag 20 (arm minus vs):")
        print(pd.DataFrame(rows).round(5).to_string(index=False))
    print("\nA result is |t_payout| >= 2 against the same-seed control; the m_ctl_s43 - m_ctl row "
          "is the seed noise a single MLP delta has to clear.")


if __name__ == "__main__":
    main()
