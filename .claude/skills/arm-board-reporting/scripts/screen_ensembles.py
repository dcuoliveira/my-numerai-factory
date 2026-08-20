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
"""
import argparse, glob, itertools, json, os, sys
import numpy as np
import pandas as pd

LAG = 20

def newey_west(series, lag=LAG):
    x = np.asarray(series, dtype="float64"); n = len(x)
    mean = x.mean(); e = x - mean; var = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(var, 0.0) / n)
    return mean, (mean / se if se > 0 else np.nan)

def payout(frame): return 0.1 * frame["corr"] + frame["mmc"]

def stats(series):
    cum = series.cumsum()
    return dict(payout=float(series.mean()),
                sharpe=float(series.mean() / series.std(ddof=0)),
                maxdd=float((np.maximum.accumulate(cum) - cum).max()))

def load(tournament, model, run):
    path = "results/per_period/%s/%s/%s.csv" % (tournament, model, run)
    if not os.path.exists(path): return None
    f = pd.read_csv(path)
    return payout(f.set_index("date") if "date" in f.columns else f)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tournament", default="crypto")
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--with-registry", action="store_true",
                    help="also include the deployed slots from src/live/registry.py")
    ap.add_argument("--size", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--top", type=int, default=8)
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
        sys.path.insert(0, ".")
        from src.live import registry
        for name in registry.ORDER:
            run = registry.get(name).get("backtest_run")
            if not run: continue
            for m in ("ridge_glmnet", "ridge", "lgb", "lstm"):
                s = load(args.tournament, m, run)
                if s is not None:
                    # A deployed slot's backtest_run may already be in --runs
                    # (SACI_01's IS fe_lgb_reg_interact). Same series under two
                    # names would double-count it in every combination.
                    if run not in members:
                        members["deployed:" + name] = s
                    break

    frame = pd.DataFrame(members).dropna()
    if frame.shape[1] < 2:
        raise SystemExit("need at least 2 members with overlapping periods")
    solo = {c: stats(frame[c]) for c in frame}
    print("%d members over %d shared periods\n" % (frame.shape[1], len(frame)))

    rows = []
    for r in args.size:
        for combo in itertools.combinations(frame.columns, r):
            blend = frame[list(combo)].mean(axis=1)
            st = stats(blend)
            best = max(combo, key=lambda c: solo[c]["payout"])
            d, t = newey_west(blend - frame[best])
            rho = np.mean([frame[a].corr(frame[b])
                           for a, b in itertools.combinations(combo, 2)])
            st.update(members=combo, best=best, delta=float(d), t=float(t), rho=float(rho))
            rows.append(st)

    def show(title, key):
        print("=== %s ===" % title)
        print("%-52s %9s %8s %8s %11s %6s %5s"
              % ("ensemble", "payout", "sharpe", "maxdd", "d vs best", "t", "rho"))
        for c in sorted(rows, key=key)[:args.top]:
            print("%-52s %9.6f %8.4f %8.4f %+11.6f %6.2f %5.2f"
                  % (" + ".join(m.replace("deployed:", "") for m in c["members"]),
                     c["payout"], c["sharpe"], c["maxdd"], c["delta"], c["t"], c["rho"]))
        print()
    show("by sharpe", lambda c: -c["sharpe"])
    show("by max drawdown", lambda c: c["maxdd"])

    beat = [c for c in rows if c["t"] > 2]
    print("%d of %d ensembles beat their own best member on payout at |t| > 2"
          % (len(beat), len(rows)))
    if not beat:
        print("  none -- equal-weight averaging dilutes the mean by construction, so read")
        print("  these as a RISK decision (sharpe, drawdown, correlation), not a level one")

if __name__ == "__main__":
    main()
