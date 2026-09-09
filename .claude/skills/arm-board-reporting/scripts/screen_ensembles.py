"""
Screen equal-weight ensembles of finished runs, on the payout series.

    python3 screen_ensembles.py --model lgb --runs 'fe_lgb_reg_*' --with-registry

Costs seconds: run_ensemble rank-averages FINISHED forecasts, so nothing is
refitted.

THIS IS A SCREEN, NOT A PRICE. It averages the members' payout SERIES, which is
a cheap proxy for what run_ensemble actually does -- a per-period rank average of
PREDICTIONS. The two can differ a lot: SACI_01 + CURUPIRA_03 + CURUPIRA_05_60
screened at payout 0.0903 / drawdown 0.23, and the real blend came out at
0.0977 / 1.31. Higher payout, and drawdown five times worse. Use this to RANK
candidates, then run the shortlist through run_ensemble before believing any
number or deploying anything. Two rules this encodes, both learned the hard way:

  * An ensemble is compared to its OWN BEST MEMBER, never to a global baseline.
    "Does combining beat holding the best of these" is the decision; equal-weight
    averaging dilutes the mean by construction, so a global comparison flatters
    every blend.
  * Rank by sharpe and drawdown, not payout. Across 3,276 combinations on this
    panel, zero beat their best member on payout at |t| > 2 and every point
    estimate was negative -- while the best pairing cut max drawdown from 1.00
    to 0.23. Blending here is a risk decision.

This is now a CLI over src/analysis/ensemble_screen.py -- the same code the PDF
report calls, so a screened number in a report and a screened number on the
command line cannot drift apart. Run it from the repo root.
"""
import argparse, glob, os, sys

import pandas as pd

sys.path.insert(0, os.getcwd())
from src.analysis.ensemble_screen import LAG, beat_their_best, rank, screen
from src.analysis.results_table import payout_series

def load(tournament, model, run):
    path = "results/per_period/%s/%s/%s.csv" % (tournament, model, run)
    if not os.path.exists(path): return None
    f = pd.read_csv(path)
    return payout_series(f.set_index("date") if "date" in f.columns else f)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tournament", default="crypto")
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--with-registry", action="store_true",
                    help="also include the deployed slots from src/live/registry.py")
    ap.add_argument("--size", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--lag", type=int, default=LAG)
    args = ap.parse_args()

    base = "results/metrics/%s/%s" % (args.tournament, args.model)
    names = []
    for pattern in args.runs:
        names.extend(sorted(os.path.basename(p)[:-5]
                            for p in glob.glob(os.path.join(base, pattern + ".json"))))
    members = {}
    for run in dict.fromkeys(names):
        s = load(args.tournament, args.model, run)
        if s is not None: members[run] = s

    if args.with_registry:
        from src.analysis import slots
        for slot in slots.slot_rows("all", args.tournament)[0]:
            # A deployed slot's backtest_run may already be in --runs (SACI_01's
            # IS fe_lgb_reg_interact). Same series under two names would
            # double-count it in every combination.
            if slot.run not in members:
                members["deployed:" + slot.name] = slot.payout

    rows, solo, n_periods = screen(members, tuple(args.size), args.lag)
    if not rows:
        raise SystemExit("need at least 2 members with overlapping periods")
    print("%d members over %d shared periods\n" % (len(solo), n_periods))

    def show(title, key):
        print("=== %s ===" % title)
        print("%-52s %9s %8s %8s %11s %6s %5s"
              % ("ensemble", "payout", "sharpe", "maxdd", "d vs best", "t", "rho"))
        for c in rank(rows, key)[:args.top]:
            print("%-52s %9.6f %8.4f %8.4f %+11.6f %6.2f %5.2f"
                  % (" + ".join(m.replace("deployed:", "") for m in c["members"]),
                     c["payout"], c["sharpe"], c["maxdd"], c["delta"], c["t"], c["rho"]))
        print()
    show("by sharpe", "sharpe")
    show("by max drawdown", "maxdd")

    beat = beat_their_best(rows)
    print("%d of %d ensembles beat their own best member on payout at |t| > 2"
          % (len(beat), len(rows)))
    if not beat:
        print("  none -- equal-weight averaging dilutes the mean by construction, so read")
        print("  these as a RISK decision (sharpe, drawdown, correlation), not a level one")

if __name__ == "__main__":
    main()
