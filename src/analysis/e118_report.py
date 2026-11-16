"""
E118's report: 34 representation blocks x 3 families, each paired against its base.

    python -m src.analysis.e118_report [--no-html]

For every family it first checks that the in-study base (e118_<family>_x00)
reproduces the reused baseline per period, then pairs every block arm against
x00 -- Newey-West at lag 20 on the per-period differences of CORR, MMC and payout,
payout rebuilt per period -- and prints calmar and burn beside each (equal n, so
both compare). With 102 tests, about five cells clear |t| >= 2 by chance alone, so
the summary counts hits per family against that floor and lists the blocks that
move in the same direction on all three. Writes logs/e118_report.txt and
results/reports/crypto/e118_summary.csv, then one HTML report per family.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from src.analysis.cross_target import newey_west
from src.analysis.results_table import burn_days
from src.prediction.metrics import summarize

FAMILIES = [("ridge", "ridge_glmnet", "w6r6_lobo_tenc"), ("lgb", "lgb_ktz", "e29_tenc_onehot"),
            ("catboost", "catboost_ktz", "e109_catboost_ktz_fenc")]
BLOCKS = {
    "01": "TabPFN point predictions", "02": "TabPFN distribution + bin probs", "03": "TabPFN estimator disagreement",
    "04": "TabPFN embedding PCs", "05": "TabPFN block heads + contexts", "06": "TimesFM/Chronos-2 on spot",
    "07": "attention familiarity", "08": "rank uncertainty", "09": "forecast drift", "10": "embedding neighbours",
    "11": "residual learner", "12": "crowd forecast", "13": "60-day term structure", "14": "predicted extremeness",
    "15": "feature nowcasts", "16": "self-context", "17": "regime context", "18": "pairwise ranking",
    "19": "local drivers", "20": "evidence age", "22": "RF prediction", "23": "RF tree disagreement",
    "24": "RF leaf PCs", "25": "RF leaf support", "26": "CatBoost prediction", "27": "CatBoost virtual-ensemble std",
    "28": "LightGBM prediction", "29": "LightGBM stage profile", "30": "LightGBM SHAP summary",
    "31": "MLP prediction", "32": "MLP hidden PCs", "33": "MLP seed disagreement", "34": "block-ridge predictions",
    "35": "book member spread",
}
OUT = os.path.join("logs", "e118_report.txt")
CSV = os.path.join("results", "reports", "crypto", "e118_summary.csv")


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
                calmar=s["mean"] / dd, burn=burn_days(f.reset_index()), n=len(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()
    lines, rows = [], []

    def say(text=""):
        lines.append(text)
        print(text)

    say("E118 -- 34 causal representation blocks, one at a time, on the best clean ridge / LightGBM / CatBoost")
    say("each arm paired vs its family's in-study base e118_<family>_x00 (NW lag 20, payout rebuilt per period)")
    for fam, model, reused in FAMILIES:
        base = series(model, "e118_%s_x00" % fam)
        say("\n======== %s (%s) ========" % (fam, model))
        if base is None:
            say("  base e118_%s_x00 missing -- family skipped")
            continue
        old = series(model, reused)
        if old is not None:
            common = base.index.intersection(old.index)
            gap = float(np.abs(base.loc[common, "corr"] - old.loc[common, "corr"]).max())
            say("  base reproduces %s: max per-period CORR gap %.2e over %d periods" % (reused, gap, len(common)))
        b = stats(base)
        say("  base   CORR %.5f  MMC %.5f  payout %.5f  calmar %.3f  burn %d  n %d"
            % (b["CORR"], b["MMC"], b["payout"], b["calmar"], b["burn"], b["n"]))
        say("  %-3s %-30s %8s %16s %16s %8s %6s %4s" % ("x", "block", "payout", "dpay (t)", "dCORR (t)", "dMMC t", "calmar", "burn"))
        for nn, label in BLOCKS.items():
            arm = series(model, "e118_%s_x%s" % (fam, nn))
            if arm is None:
                say("  x%s %-30s (missing)" % (nn, label))
                continue
            common = base.index.intersection(arm.index)
            d = {k: newey_west((arm.loc[common, k] - base.loc[common, k]).to_numpy()) for k in ("payout", "corr", "mmc")}
            a = stats(arm.loc[common])
            rows.append(dict(family=fam, block="x" + nn, label=label, **{k: a[k] for k in ("CORR", "MMC", "payout", "calmar", "burn", "n")},
                             dpay=d["payout"][0], t_pay=d["payout"][1], dcorr=d["corr"][0], t_corr=d["corr"][1],
                             dmmc=d["mmc"][0], t_mmc=d["mmc"][1], base_calmar=b["calmar"], base_burn=b["burn"]))
            flag = " *" if abs(d["payout"][1]) >= 2 else ""
            say("  x%s %-30s %8.5f %+8.5f (%+5.2f) %+8.5f (%+5.2f) %+6.2f %6.3f %4d%s"
                % (nn, label, a["payout"], d["payout"][0], d["payout"][1], d["corr"][0], d["corr"][1], d["mmc"][1],
                   a["calmar"], a["burn"], flag))

    if rows:
        t = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(CSV), exist_ok=True)
        t.to_csv(CSV, index=False)
        say("\n======== summary ========")
        say("  %d arms on disk of 102; at |t| >= 2 chance alone gives ~%.1f hits" % (len(t), 0.0455 * len(t)))
        for fam, g in t.groupby("family", sort=False):
            say("  %-9s payout |t|>=2: %d up, %d down   CORR |t|>=2: %d up, %d down   best block %s (%+.5f, t %+.2f)"
                % (fam, (g.t_pay >= 2).sum(), (g.t_pay <= -2).sum(), (g.t_corr >= 2).sum(), (g.t_corr <= -2).sum(),
                   g.loc[g.dpay.idxmax(), "block"], g.dpay.max(), g.loc[g.dpay.idxmax(), "t_pay"]))
        wide = t.pivot(index="block", columns="family", values="t_pay")
        same_up = [b for b, r in wide.iterrows() if r.notna().all() and (r > 0).all()]
        say("  blocks with payout up on all three families: %s" % (", ".join(same_up) or "none"))
        say("  wrote %s" % CSV)

    if not args.no_html:
        for fam, model, _ in FAMILIES:
            runs = ["%s/e118_%s_x%s" % (model, fam, nn) for nn in ["00"] + list(BLOCKS)
                    if series(model, "e118_%s_x%s" % (fam, nn)) is not None]
            if len(runs) < 2:
                continue
            res = subprocess.run([sys.executable, "-m", "src.analysis.report_html", "--model", model, "--runs", *runs],
                                 capture_output=True, text=True)
            path = [w for w in res.stdout.split() if w.endswith(".html")]
            say("  html %s: %s" % (fam, path[-1] if path else "FAILED (see stderr)"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
