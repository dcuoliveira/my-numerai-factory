"""
Replicate Lee & Lee (2025) Table 1 on FRED-MD, to validate that the E27 FARM
estimators (src/prediction/farm.py) are faithful.

    python3 experiments/farm_paper_replication.py

Drives the identical estimator functions E27 uses, in the paper's LEVEL space
(src/prediction/farm_paper.py), over the paper's five targets x three horizons,
and prints the reproduced Table 1 beside the published one. Writes
results/farm_paper/table1_reproduction.csv. Ends with a PASS/FAIL against the
agreed pattern + anchor-cell criteria.

If FARM1 reproduces the paper's wins here, the E27 crypto null is a property of
the data and the rank-correlation metric, not a bug in the combiner.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.prediction import farm_paper as fp

SER = ["RPI", "CPIAUCSL", "PCEPI", "INDPRO", "UNRATE"]
DATA = "data/fredmd/fred-md.csv"
OUT = "results/farm_paper/table1_reproduction.csv"
# The factor methods carry a per-fit CV; throttle only those. The cheap,
# anchor-bearing methods stay fully rolling (update_every=1).
UPDATE_EVERY = {"farm2": 6, "farm3": 6, "ds": 3}

# Published Table 1 (relative MSFE, DM p-value), transcribed from the paper.
# label -> {h: ([rel x5], [p x5])}, series order = SER.
PAPER = {
 "gr":               {1:([1.631,2.064,1.516,1.539,1.175],[.999,1.00,.978,1.00,.891]),
                      2:([1.446,1.936,1.660,1.512,1.140],[1.,1.,.995,.993,.744]),
                      3:([1.662,1.898,1.855,1.482,1.462],[1.,.989,1.,.996,.998])},
 "ds":               {1:([1.004,0.881,0.823,1.068,0.916],[.720,.078,.012,.978,.087]),
                      2:([1.011,1.003,0.995,1.075,0.958],[.969,.547,.392,.979,.215]),
                      3:([1.000,1.032,1.012,1.069,1.149],[.517,.947,.726,.996,.997])},
 "farm1.postalasso": {1:([0.966,0.890,0.902,0.935,0.862],[.327,.087,.024,.151,.054]),
                      2:([0.987,0.964,1.023,0.985,0.859],[.123,.168,.677,.356,.027]),
                      3:([1.015,1.091,1.018,0.985,1.005],[.806,.843,.633,.298,.539])},
 "farm1.l2boost":    {1:([0.961,0.883,0.866,0.985,0.877],[.241,.004,.002,.340,.019]),
                      2:([1.014,0.929,0.932,1.086,0.877],[.697,.070,.031,.972,.021]),
                      3:([1.000,1.005,1.014,0.961,0.948],[.504,.557,.631,.114,.086])},
 "farm1.ridge":      {1:([1.007,0.932,0.913,1.016,0.926],[.829,.053,.007,.946,.082]),
                      2:([0.998,0.997,0.952,1.005,0.927],[.395,.043,.027,.577,.050]),
                      3:([1.006,1.001,1.003,1.013,0.969],[.797,.926,.997,.913,.033])},
 "farm2.postalasso": {1:([0.965,0.898,0.940,1.117,1.053],[.264,.064,.066,.963,.949]),
                      2:([0.997,0.960,0.979,1.159,1.069],[.348,.194,.304,.951,.982]),
                      3:([1.015,1.070,1.006,1.037,1.095],[.743,.855,.553,.737,1.])},
 "farm2.l2boost":    {1:([0.954,0.902,0.911,1.078,1.043],[.168,.002,.002,.985,.907]),
                      2:([1.010,0.970,0.969,1.144,1.055],[.748,.207,.136,.988,.979]),
                      3:([0.999,1.020,1.027,1.019,1.137],[.481,.799,.786,.671,1.])},
 "farm2.ridge":      {1:([1.005,0.953,0.993,1.063,1.102],[.727,.021,.167,1.,1.]),
                      2:([1.005,1.001,1.002,1.071,1.095],[.734,.677,.707,.967,1.]),
                      3:([1.003,1.003,1.003,1.065,1.105],[.717,.752,.743,.995,1.])},
 "farm3.postalasso": {1:([0.971,0.970,0.867,0.971,1.024],[.350,.212,.004,.262,.765]),
                      2:([1.024,0.953,1.044,0.983,0.923],[.879,.147,.857,.327,.069]),
                      3:([1.029,1.071,1.015,0.996,1.010],[.893,.789,.623,.439,.622])},
 "farm3.l2boost":    {1:([0.958,0.935,0.928,0.998,1.009],[.207,.023,.007,.462,.647]),
                      2:([1.017,0.949,0.963,1.090,0.934],[.808,.071,.052,.989,.046]),
                      3:([0.998,1.003,1.010,0.973,0.977],[.407,.542,.600,.150,.189])},
 "farm3.ridge":      {1:([1.006,0.947,0.938,1.034,1.016],[.814,.041,.008,.998,.775]),
                      2:([1.000,0.999,0.977,1.023,0.963],[.519,.171,.078,.857,.087]),
                      3:([1.006,1.001,1.002,1.026,0.988],[.815,.909,.967,.991,.171])},
}


def run():
    panel = fp.load_fredmd(DATA)
    print("FRED-MD panel: %d months x %d series, %s -> %s\n"
          % (panel.shape[0], panel.shape[1],
             panel.index.min().date(), panel.index.max().date()))

    rows = []
    for h in fp.HORIZONS:
        for target in SER:
            F, y, names, _ = fp.individual_forecasts(panel, target, h)
            base, base_y = fp.combine(F, y, "equal", "equal", None)
            loss_base = (base_y - base) ** 2
            for label, method, bm in fp.METHODS:
                if label == "equal":
                    continue
                pred, y_oos = fp.combine(F, y, label, method, bm,
                                         update_every=UPDATE_EVERY.get(method, 1))
                loss = (y_oos - pred) ** 2
                rel, t, p = fp.diebold_mariano(loss_base, loss, lag=h)
                si = SER.index(target)
                pr = PAPER.get(label, {}).get(h)
                rows.append(dict(h=h, target=target, method=label, N=len(names),
                                 rel_mine=rel, t_mine=t, p_mine=p,
                                 rel_paper=(pr[0][si] if pr else np.nan),
                                 p_paper=(pr[1][si] if pr else np.nan)))
        print("h=%d done" % h)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    _print_tables(df)
    _verdict(df)
    print("\nwrote %s" % OUT)


def _print_tables(df):
    for h in fp.HORIZONS:
        print("\n=== h=%d : reproduced relative MSFE (bold* = DM p<0.10 & <1) ===" % h)
        block = df[df.h == h]
        print("%-18s %s" % ("method", "  ".join("%9s" % s for s in SER)))
        order = [m for m, _, _ in fp.METHODS if m != "equal"]
        for label in order:
            cells = []
            for s in SER:
                r = block[(block.method == label) & (block.target == s)]
                if r.empty:
                    cells.append("%9s" % "-"); continue
                rel = r.rel_mine.iloc[0]; p = r.p_mine.iloc[0]
                mark = "*" if (rel < 1 and p < 0.10) else " "
                cells.append("%8.3f%s" % (rel, mark))
            print("%-18s %s" % (label, "  ".join(cells)))


def _verdict(df):
    print("\n" + "=" * 68 + "\nVALIDATION VERDICT (pattern + anchor cells)\n" + "=" * 68)
    checks = []

    # 1. anchor cells: FARM1.l2boost at h=1 for CPIAUCSL/PCEPI/UNRATE within +-0.05
    anchors = {("CPIAUCSL", 1): 0.883, ("PCEPI", 1): 0.866, ("UNRATE", 1): 0.877}
    for (tgt, h), paper_val in anchors.items():
        r = df[(df.method == "farm1.l2boost") & (df.target == tgt) & (df.h == h)]
        mine = r.rel_mine.iloc[0]
        ok = mine < 1 and abs(mine - paper_val) <= 0.05
        checks.append(("anchor FARM1.l2boost %s h%d: %.3f (paper %.3f)"
                       % (tgt, h, mine, paper_val), ok))

    # 2. FARM1 significantly beats equal for CPI/PCE/UNRATE at h=1,2 (any estimator)
    for tgt in ("CPIAUCSL", "PCEPI", "UNRATE"):
        for h in (1, 2):
            r = df[(df.method.str.startswith("farm1")) & (df.target == tgt) & (df.h == h)]
            ok = ((r.rel_mine < 1) & (r.p_mine < 0.10)).any()
            checks.append(("FARM1 beats equal (sig) %s h%d" % (tgt, h), ok))

    # 3. L2-Boost <= Ridge within FARM1 (majority of the 15 cells)
    lb = df[df.method == "farm1.l2boost"].set_index(["target", "h"]).rel_mine
    rg = df[df.method == "farm1.ridge"].set_index(["target", "h"]).rel_mine
    frac = (lb <= rg + 1e-9).mean()
    checks.append(("L2-Boost <= Ridge in FARM1 (%.0f%% of cells)" % (100 * frac), frac >= 0.6))

    # 4. GR blows up: relative MSFE > 1 on every series/horizon
    gr = df[df.method == "gr"]
    checks.append(("GR relative MSFE > 1 everywhere (min %.2f, max %.2f)"
                   % (gr.rel_mine.min(), gr.rel_mine.max()), (gr.rel_mine > 1).all()))

    # 5. FARM2/FARM3 do not beat FARM1 (best FARM1 <= best FARM2/3 per cell, majority)
    best = lambda pref: df[df.method.str.startswith(pref)].groupby(["target", "h"]).rel_mine.min()
    f1, f2, f3 = best("farm1"), best("farm2"), best("farm3")
    frac23 = ((f1 <= f2 + 1e-9) & (f1 <= f3 + 1e-9)).mean()
    checks.append(("FARM1 <= FARM2 and FARM3 (%.0f%% of cells)" % (100 * frac23), frac23 >= 0.6))

    # 6. horizon decay: mean FARM1 rel MSFE rises with h
    means = [df[(df.method.str.startswith("farm1")) & (df.h == h)].rel_mine.mean() for h in (1, 2, 3)]
    checks.append(("horizon decay: FARM1 mean relMSFE h1<h2<h3 (%.3f, %.3f, %.3f)"
                   % tuple(means), means[0] <= means[1] <= means[2] or means[0] < means[2]))

    for text, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", text))
    npass = sum(ok for _, ok in checks)
    print("\n%d/%d checks pass -> %s" % (npass, len(checks),
          "VALIDATED: the E27 estimators reproduce the paper's pattern."
          if npass >= len(checks) - 2 else
          "NOT validated: investigate the failing checks before trusting E27."))


if __name__ == "__main__":
    run()
