"""
E125's report: E119's MLP (IA_01) training-window ladder, fitted on 20 and on 60,
RAW and META-RESIDUALISED, all scored on the PAYABLE 20-day label, in one table.

    python3 -m src.analysis.e125_report

Twenty-eight rows: seven windows x {fit 20 (E119), fit 60 (E125)} x {raw,
residualised}. Every row goes through ONE scorer -- cross_target's
score_on_payable_target, against the same target, meta model and reference as
every deployed slot -- so rows differ only by model and transform. The
MLP mirror of e124_report (ridge) and e127_report (catboost).

Residualised means the deployed transform, src.prediction.meta_resid at
STRENGTH 0.5 on a meta model at least 5 days old: what a live "resid": 0.5 slot
submits. It is applied to the finished forecasts; nothing is refitted.

Two check lines head the report: the re-score reproduces E119's own numbers,
and the residualiser reproduces E107's stored residualised CURUPIRA_04.

All rows share ONE window -- the 60-day arms cannot predict the last ~40 dates
-- so calmar and burn compare. Paired t is at Newey-West lag 20 (the scored
label's overlap), plus lag 60 for the fit-60 comparison as E60 did.

The pairwise correlations of all 28 series (378 pairs) are written to
results/cache/e125_pair_corr.csv, each pair tagged "across" (fit-20 vs fit-60),
"window" (same label, different arm or transform) or "twin" (raw vs resid of
one arm): the within-date rank correlation of the PREDICTIONS averaged over the
common dates, and the correlation of the per-period PAYOUT series.
"""

import itertools
import os

import numpy as np
import pandas as pd

from src.analysis.ablation_report import newey_west
from src.analysis.cross_target import score_on_payable_target, within_period_rank_corr
from src.analysis.meta_resid_rescore import Residualiser

MODEL = "mlp"
WINDOWS = [(5, "1 week"), (10, "2 weeks"), (15, "3 weeks"), (20, "4 weeks"),
           (22, "1 month"), (65, "3 months"), (130, "6 months")]
FITS = [("20", "e119_w%d"), ("60", "e125_w%d")]
REFERENCE = ("e119_w130", "raw")          # 6 months, fit 20, raw: IA_01
CACHE = os.path.join("results", "cache", "e125_rescore_t20.csv")
PAIRS = os.path.join("results", "cache", "e125_pair_corr.csv")


def key(run, variant):
    return "%s|%s" % (run, variant)


def stats(series):
    cumulative = series.cumsum()
    dd = float((cumulative.cummax() - cumulative).max())
    return (float(series.mean()), float(series.mean() / series.std(ddof=0)), dd,
            float(series.mean()) / dd if dd > 0 else np.nan, int((series < 0).sum()))


def paired(a, b, lag=20):
    _, _, t = newey_west((a - b).to_numpy(), lag=lag)
    return float((a - b).mean()), float(t)


def cells(run):
    """Architecture / alpha / learning rate / batch chosen per refit."""
    p = os.path.join("results", "hyperparams", "crypto", MODEL, run + ".csv")
    if not os.path.exists(p):
        return "-"
    return "  ".join("%s/a%.3g/lr%.2g/b%d" % (str(r["mlp__hidden_layer_sizes"]).replace(" ", ""), r["mlp__alpha"],
                                              r["mlp__learning_rate_init"], r["mlp__batch_size"])
                     for _, r in pd.read_csv(p).iterrows())


def category(a, b):
    run_a, var_a = a.split("|")
    run_b, var_b = b.split("|")
    if run_a == run_b:
        return "twin"
    return "across" if run_a[:4] != run_b[:4] else "window"


def main():
    res = Residualiser()

    # Every series is a within-date pct rank, so it is both what gets scored and
    # what the prediction correlation is taken over.
    predictions, tables = {}, {}
    for _, pattern in FITS:
        for window, _ in WINDOWS:
            run = pattern % window
            if not os.path.exists(os.path.join("results", "metrics", "crypto", MODEL, run + ".json")):
                continue
            forecast = pd.read_csv(os.path.join("results", "forecast", "crypto", MODEL, run + ".csv"),
                                   usecols=["symbol", "date", "prediction"],
                                   parse_dates=["date"]).set_index(["symbol", "date"])
            predictions[key(run, "raw")] = forecast.groupby(level=1)["prediction"].rank(pct=True)
            predictions[key(run, "resid")] = res.residualised(forecast)
    for k, prediction in predictions.items():
        table, _ = score_on_payable_target(prediction, res.extras)
        table = table[["corr", "mmc"]].copy()
        table.index = pd.to_datetime(table.index).strftime("%Y-%m-%d")
        tables[k] = table

    # ---- check 1: the re-score reproduces run_forecast's own numbers
    worst = 0.0
    for window, _ in WINDOWS:
        k = key("e119_w%d" % window, "raw")
        if k not in tables:
            continue
        native = pd.read_csv(os.path.join("results", "per_period", "crypto", MODEL,
                                          "e119_w%d.csv" % window)).set_index("date")
        j = tables[k].join(native[["corr", "mmc"]], rsuffix="_n", how="inner")
        worst = max(worst, (j["corr"] - j["corr_n"]).abs().max(), (j["mmc"] - j["mmc_n"]).abs().max())
    print("\ncheck 1 -- raw re-score vs run_forecast's native scoring (E119): "
          "max |difference| %.1e" % worst)

    # ---- check 2: the residualiser reproduces E107
    mine = res.rescore("ridge_glmnet", "w6r6_glmnet_reg_allfeatpp", residualise=True)
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
    print("E119 + E125 -- window x fitted label x meta-residualisation, ALL SCORED ON target_binned_return_20")
    print("  mlp, fenccausal (44 feat), refit 130, 50-draw MLP search every refit;")
    print("  resid = src.prediction.meta_resid, strength 0.5, meta model >= 5 days old (the live transform)")
    print("  %d common dates, %s .. %s.   d resid: against the same arm unresidualised.   "
          "d fit60: against fit 20 with the same transform.   vs ref: against %s raw (IA_01)."
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
                twin = key("e119_w%d" % window, variant)
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

    print("\n  hyperparameters chosen per refit (arch / alpha / lr / batch) -- residualising does not refit")
    for window, label in WINDOWS:
        print("  %-8s fit20 %s" % (label, cells("e119_w%d" % window)))
        print("  %-8s fit60 %s" % ("", cells("e125_w%d" % window)))

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

    # ---- pairwise correlation, on the same common dates as the table
    frame = pd.DataFrame(predictions)
    frame = frame[frame.index.get_level_values("date").strftime("%Y-%m-%d").isin(pay.index)]
    frame = frame.dropna(how="any")
    pred_corr = within_period_rank_corr(frame)
    pay_corr = pay.corr()
    rows = [{"a": a, "b": b, "category": category(a, b),
             "pred_corr": float(pred_corr.loc[a, b]), "pay_corr": float(pay_corr.loc[a, b])}
            for a, b in itertools.combinations(list(pay.columns), 2)]
    pairs = pd.DataFrame(rows)
    pairs.to_csv(PAIRS, index=False)
    print("\n  pairwise correlation, %d series -> %d pairs over %d dates (%s)"
          % (len(pay.columns), len(pairs), frame.index.get_level_values("date").nunique(), PAIRS))
    for cat in ("across", "window", "twin"):
        p = pairs[pairs["category"] == cat]
        if len(p):
            print("    %-7s %4d pairs   prediction rho median %.2f [%.2f, %.2f]   "
                  "payout rho median %.2f [%.2f, %.2f]"
                  % (cat, len(p), p["pred_corr"].median(), p["pred_corr"].min(), p["pred_corr"].max(),
                     p["pay_corr"].median(), p["pay_corr"].min(), p["pay_corr"].max()))
    print("\n  re-scored series cached at %s" % CACHE)


if __name__ == "__main__":
    main()
