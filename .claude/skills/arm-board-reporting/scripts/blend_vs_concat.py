"""
Report a group BLEND vs CONCAT study.

    python3 blend_vs_concat.py --prefix e02 [--baseline e02_raw]

Handles the two run shapes such a study mixes, which is where the naive version
of this table crashes: run_forecast records config.n_features, run_ensemble does
NOT -- a blend has members, not a feature matrix. Read every field with a
fallback, or the report dies after the arms have already cost their hours.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

def pp(model, run):
    p = "results/per_period/crypto/%s/%s.csv" % (model, run)
    if not os.path.exists(p): return None
    f = pd.read_csv(p)
    return f.set_index("date") if "date" in f.columns else f

def payout(f): return 0.1 * f["corr"] + f["mmc"]

def newey_west(series, lag=20):
    x = np.asarray(series, dtype="float64"); n = len(x)
    mean = x.mean(); e = x - mean; var = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(var, 0.0) / n)
    return mean, (mean / se if se > 0 else float("nan"))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--model", default="lgb")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--lag", type=int, default=20)
    args = ap.parse_args()
    base_run = args.baseline or (args.prefix + "_raw")

    rows = []
    for p in sorted(glob.glob("results/metrics/crypto/%s/%s_*.json" % (args.model, args.prefix))):
        d = json.load(open(p)); c = d.get("config", {}); run = os.path.basename(p)[:-5]
        members = c.get("members")
        rows.append(dict(
            run=run,
            # a blend has members and no feature matrix; a fitted arm has both
            feat=c.get("n_features"), members=len(members) if members else None,
            corr=d["corr"]["mean"], mmc=d["mmc"]["mean"],
            sharpe=d["corr"]["sharpe"], maxdd=d["corr"]["max_drawdown"],
            payout=0.1 * d["corr"]["mean"] + d["mmc"]["mean"]))
    if not rows: raise SystemExit("no runs matching %s_*" % args.prefix)
    t = pd.DataFrame(rows)
    base = pp(args.model, base_run)

    print("%-26s %7s %10s %8s %8s %10s %11s %7s" % (
        "run", "feat", "CORR", "sharpe", "maxdd", "payout", "d vs raw", "t"))
    print("-" * 96)
    for _, r in t.sort_values("payout", ascending=False).iterrows():
        shape = ("%d" % r["feat"]) if pd.notna(r["feat"]) else ("blend/%d" % r["members"])
        if r["run"] == base_run or base is None:
            d = tt = None
        else:
            a = pp(args.model, r["run"])
            j = pd.concat([payout(base), payout(a)], axis=1, join="inner", keys=["b", "a"])
            d, tt = newey_west(j["a"] - j["b"], args.lag)
        print("%-26s %7s %10.6f %8.4f %8.4f %10.6f %11s %7s" % (
            r["run"], shape, r["corr"], r["sharpe"], r["maxdd"], r["payout"],
            "baseline" if d is None else "%+.6f" % d,
            "-" if tt is None else "%+.2f" % tt))
    print("-" * 96)

    print("\nblend vs concat, paired (Newey-West lag %d):" % args.lag)
    for kind in ("trio", "quintet"):
        b = pp(args.model, "%s_blend_%s" % (args.prefix, kind))
        c = pp(args.model, "%s_concat_%s" % (args.prefix, kind))
        if b is None or c is None:
            print("  %-8s missing an arm" % kind); continue
        j = pd.concat([payout(c), payout(b)], axis=1, join="inner", keys=["c", "b"])
        d, tt = newey_west(j["b"] - j["c"], args.lag)
        print("  %-8s BLEND %.6f  CONCAT %.6f  ->  blend %+.6f, t %+.2f%s" % (
            kind, j["b"].mean(), j["c"].mean(), d, tt,
            "" if abs(tt) > 2 else "   (not distinguishable)"))

if __name__ == "__main__":
    main()
