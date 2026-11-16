"""
E124's report: E117's training-window ladder, fitted on 20 and on 60, RAW and
META-RESIDUALISED, all scored on the PAYABLE 20-day label, in one table.

    python3 -m src.analysis.e124_report

Twenty-eight rows: seven windows x {fit 20 (E117), fit 60 (E124)} x {raw,
residualised}. Every row goes through ONE scorer -- cross_target's
score_on_payable_target, against the same target, meta model and reference as
every deployed slot -- so rows differ only by model and transform.

Residualised means the deployed transform, src.prediction.meta_resid at
STRENGTH 0.5 on a meta model at least 5 days old: what a live "resid": 0.5 slot
submits. It is applied to the finished forecasts; nothing is refitted.

Two check lines head the report: the re-score reproduces E117's own numbers,
and the residualiser reproduces E107's stored residualised CURUPIRA_04.

All rows share ONE window -- the 60-day arms cannot predict the last ~40 dates
-- so calmar and burn compare. Paired t is at Newey-West lag 20 (the scored
label's overlap), plus lag 60 for the fit-60 comparison as E60 did.
"""

import os

import numpy as np
import pandas as pd

from src.analysis.ablation_report import newey_west
from src.analysis.meta_resid_rescore import Residualiser

MODEL = "ridge_glmnet"
WINDOWS = [(5, "1 week"), (10, "2 weeks"), (15, "3 weeks"), (20, "4 weeks"),
           (22, "1 month"), (65, "3 months"), (130, "6 months")]
FITS = [("20", "e117_w%d"), ("60", "e124_w%d")]
REFERENCE = ("e117_w130", "raw")          # 6 months, fit 20, raw: the single-model config
CACHE = os.path.join("results", "cache", "e124_rescore_t20.csv")


def key(run, variant):
    return "%s|%s" % (run, variant)


def stats(series):
    cumulative = series.cumsum()
    dd = float((cumulative.cummax() - cumulative).max())
    return (float(series.mean()), float(series.mean() / series.std(ddof=0)), dd,
            float(series.mean()) / dd if dd > 0 else np.nan, int((series < 0).sum()))


def paired(a, b, lag=20):
    mean, _, t = newey_west((a - b).to_numpy(), lag=lag)
    return float((a - b).mean()), float(t)


def lambdas(run):
    p = os.path.join("results", "hyperparams", "crypto", MODEL, run + ".csv")
    if not os.path.exists(p):
        return "-"
    h = pd.read_csv(p)
    return " ".join("%.3g%s" % (r["alpha"] / r["train_rows"],
                                "^" if np.isclose(r["alpha"], r.get("alpha_grid_max", np.nan),
                                                  rtol=1e-9) else "")
                    for _, r in h.iterrows())


def main():
    res = Residualiser()

    # ---- check 1: the re-score reproduces run_forecast's own numbers
    print("\ncheck 1 -- raw re-score vs run_forecast's native scoring (E117):")
    worst = 0.0
    tables = {}
    for _, pattern in FITS:
        for window, _ in WINDOWS:
            run = pattern % window
            if not os.path.exists(os.path.join("results", "metrics", "crypto", MODEL, run + ".json")):
                continue
            tables[key(run, "raw")] = res.rescore(MODEL, run, residualise=False)
            tables[key(run, "resid")] = res.rescore(MODEL, run, residualise=True)
            if run.startswith("e117_"):
                native = pd.read_csv(os.path.join("results", "per_period", "crypto", MODEL,
                                                  run + ".csv")).set_index("date")
                j = tables[key(run, "raw")].join(native[["corr", "mmc"]], rsuffix="_n", how="inner")
                worst = max(worst, (j["corr"] - j["corr_n"]).abs().max(),
                            (j["mmc"] - j["mmc_n"]).abs().max())
    print("  max |difference| over all 7 arms and both metrics: %.1e" % worst)

    # ---- check 2: the residualiser reproduces E107
    mine = res.rescore(MODEL, "w6r6_glmnet_reg_allfeatpp", residualise=True)
    e107 = pd.read_csv("results/per_period/crypto/meta_resid/e107_curupira_04_s050.csv").set_index("date")
    j = mine.join(e107[["corr", "mmc"]], rsuffix="_e", how="inner")
    print("check 2 -- residualised CURUPIRA_04 vs E107's stored series, %d dates: max |difference| %.1e"
          % (len(j), max((j["corr"] - j["corr_e"]).abs().max(), (j["mmc"] - j["mmc_e"]).abs().max())))

    if key(*REFERENCE) not in tables:
        print("reference has not finished")
        return
    pay = pd.DataFrame({k: 0.1 * t["corr"] + t["mmc"] for k, t in tables.items()}).dropna()
    corr = pd.DataFrame({k: t["corr"] for k, t in tables.items()}).loc[pay.index]
    mmc = pd.DataFrame({k: t["mmc"] for k, t in tables.items()}).loc[pay.index]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pd.concat({"corr": corr, "mmc": mmc}, axis=1).to_csv(CACHE)
    ref = key(*REFERENCE)

    print("\n" + "=" * 150)
    print("E117 + E124 -- window x fitted label x meta-residualisation, ALL SCORED ON target_binned_return_20")
    print("  ridge_glmnet, allfeat_pp minus tenc (811 feat), refit 130, lambda searched every refit;")
    print("  resid = src.prediction.meta_resid, strength 0.5, meta model >= 5 days old (the live transform)")
    print("  %d common dates, %s .. %s.   d resid: against the same arm unresidualised.   "
          "d fit60: against fit 20 with the same transform.   vs ref: against %s raw."
          % (len(pay), pay.index.min(), pay.index.max(), REFERENCE[0]))
    print("=" * 150)
    print("  %-8s %-3s %-5s %8s %8s %8s %6s %6s %7s %5s | %-17s | %-24s | %s"
          % ("window", "fit", "", "CORR20", "MMC20", "payout", "p.shp", "dd", "calmar", "burn",
             "d resid       t", "d fit60       t20   t60", "vs ref        t"))
    for window, label in WINDOWS:
        first = True
        for fit, pattern in FITS:
            run = pattern % window
            for variant in ("raw", "resid"):
                k = key(run, variant)
                if k not in pay:
                    print("  %-8s %-3s %-5s   (not finished)" % (label if first else "", fit, variant))
                    first = False
                    continue
                m, sh, dd, cal, burn = stats(pay[k])
                d_resid = d_fit = ""
                if variant == "resid":
                    d, t = paired(pay[k], pay[key(run, "raw")])
                    d_resid = "%+.5f %+5.2f" % (d, t)
                twin = key("e117_w%d" % window, variant)
                if fit == "60" and twin in pay:
                    d, t20 = paired(pay[k], pay[twin], 20)
                    _, t60 = paired(pay[k], pay[twin], 60)
                    d_fit = "%+.5f %+5.2f %+5.2f" % (d, t20, t60)
                if k == ref:
                    vs = "-"
                else:
                    d, t = paired(pay[k], pay[ref])
                    vs = "%+.5f %+5.2f" % (d, t)
                print("  %-8s %-3s %-5s %8.5f %8.5f %8.5f %6.2f %6.2f %7.3f %5d | %-17s | %-24s | %s"
                      % (label if first else "", fit, variant, corr[k].mean(), mmc[k].mean(),
                         m, sh, dd, cal, burn, d_resid, d_fit, vs))
                first = False

    print("\n  lambda per refit (alpha / training rows; ^ = top of the path) -- residualising does not refit")
    for window, label in WINDOWS:
        print("  %-8s fit20 %-40s fit60 %s"
              % (label, lambdas("e117_w%d" % window), lambdas("e124_w%d" % window)))

    print("\n" + "=" * 150)
    resid_rows = [(k, *paired(pay[k], pay[k.replace("|resid", "|raw")]))
                  for k in pay.columns if k.endswith("|resid")]
    print("  residualising raises payout in %d of %d arms; %d at t > +2"
          % (sum(1 for r in resid_rows if r[1] > 0), len(resid_rows),
             sum(1 for r in resid_rows if r[2] > 2)))
    cal_rows = [(k, stats(pay[k])[3], stats(pay[k.replace("|resid", "|raw")])[3])
                for k, _, _ in resid_rows]
    print("  ...and lowers calmar in %d of %d" % (sum(1 for _, a, b in cal_rows if a < b), len(cal_rows)))
    best = max(pay.columns, key=lambda k: pay[k].mean())
    best_cal = max(pay.columns, key=lambda k: stats(pay[k])[3])
    print("  best payout: %s (%.5f)    best calmar: %s (%.3f)"
          % (best.replace("|", " "), pay[best].mean(), best_cal.replace("|", " "),
             stats(pay[best_cal])[3]))
    print("\n  re-scored series cached at %s" % CACHE)


if __name__ == "__main__":
    main()
