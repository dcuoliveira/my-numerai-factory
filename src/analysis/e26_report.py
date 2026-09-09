"""
E26's report: sign-constrained Generalised Ridge on allfeat_pp.

    python3 -m src.analysis.e26_report

Two constrained arms against one free control, everything else held: the same
60-point alpha grid (floored at 1e-2 in all three), the same folds, the same
seed. Only the FEASIBLE SET differs.

`n_binding` is the arm's own honesty check. It counts LIVE coordinates sitting
on a bound, excluding the constant columns -- roughly half of allfeat_pp's 812
are constant on any 130-date block, and counting those would make a constraint
that never bit look like one that did. An arm reporting 0 is bit-identical to
its control whatever its t-statistic says.
"""

import json
import os

import numpy as np
import pandas as pd

from src.analysis.ablation_report import newey_west

PER_PERIOD = os.path.join("results", "per_period", "crypto", "gen_ridge")
METRICS = os.path.join("results", "metrics", "crypto", "gen_ridge")
LAG = 20
ARMS = [("e26_free", "betas unconstrained"),
        ("e26_pos", "ALL betas >= 0"),
        ("e26_neg", "ALL betas <= 0"),
        ("e26_ens_posneg", "1/2 pos + 1/2 neg"),
        ("e26_ens_all", "1/3 free + pos + neg")]

#: Blends carry no best_params_ -- run_ensemble combines finished forecasts and
#: fits nothing -- so the binding/alpha diagnostics are printed for the fitted
#: arms only rather than as empty cells that read like a measurement.
FITTED = ("e26_free", "e26_pos", "e26_neg")


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


def per_refit(metrics, key):
    out = []
    for refit in metrics.get("refits", []):
        value = refit.get(key)
        if value is None:
            value = (refit.get("best_params") or {}).get(key)
        if value is not None:
            out.append(value if isinstance(value, str) else "%.4g" % float(value))
    return "/".join(out) if out else "-"


def main():
    loaded = {r: load(r) for r, _ in ARMS}
    if loaded.get("e26_free") is None:
        print("control e26_free has not finished")
        return
    control_pay = payout(loaded["e26_free"][0])

    print("\n" + "=" * 108)
    print("E26 -- sign-constrained Generalised Ridge on allfeat_pp (812 columns)")
    print("  same alpha grid (floored at 1e-2), same folds, same seed.")
    print("  only the FEASIBLE SET differs.")
    print("=" * 108)
    print("  %-16s %-22s %9s %8s %7s %8s | %11s %7s"
          % ("arm", "what it is", "CORR", "payout", "p.shp", "dd(pay)", "d payout", "t"))
    rows = []
    for run, description in ARMS:
        if loaded[run] is None:
            print("  %-16s %-22s   (not finished)" % (run, description))
            continue
        frame, metrics = loaded[run]
        pay = payout(frame)
        level, sharpe, drawdown = summarise(pay)
        if run == "e26_free":
            cells = ("-", "-")
        else:
            joined = pd.concat([control_pay, pay], axis=1, join="inner",
                               keys=["b", "a"]).dropna()
            mean, _, t = newey_west((joined["a"] - joined["b"]).to_numpy(), lag=LAG)
            cells = ("%+.6f" % mean, "%+.2f" % t)
            rows.append((run, mean, t))
        corr_mean = metrics.get("corr", {}).get("mean")
        print("  %-16s %-22s %9s %8.6f %7.2f %8.2f | %11s %7s"
              % (run, description,
                 "-" if corr_mean is None else "%.6f" % corr_mean,
                 level, sharpe, drawdown, cells[0], cells[1]))

    print("\n  did the constraint actually bite?  (fitted arms only)")
    print("  %-16s %-34s %s" % ("arm", "n_binding per refit (LIVE only)", "alpha chosen"))
    for run in FITTED:
        if loaded.get(run) is None:
            continue
        metrics = loaded[run][1]
        print("  %-16s %-34s %s" % (run, per_refit(metrics, "n_binding"),
                                      per_refit(metrics, "alpha")))
    print("  n_constrained per refit: %s"
          % (per_refit(loaded["e26_pos"][1], "n_constrained")
             if loaded.get("e26_pos") else "-"))

    if not rows:
        return
    print("\n" + "=" * 108)
    strong = [r for r in rows if abs(r[2]) > 2]
    print("  %d of %d constrained arms differ from the free fit at |t| > 2"
          % (len(strong), len(rows)))
    for run, mean, t in strong:
        print("    %-16s %+.6f  t %+.2f" % (run, mean, t))
    if not strong:
        print("    Neither does. Removing half the coefficient space -- in either")
        print("    direction -- does not change the payout by a margin ~26 effective")
        print("    independent windows can resolve. Read it with n_binding: the")
        print("    constraint was applied, it just did not matter.")


if __name__ == "__main__":
    main()
