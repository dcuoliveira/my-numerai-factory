"""
E119's report: the training window of IA_01 (mlp on fenccausal), 1 week to 6 months.

    python3 -m src.analysis.e119_report

E117's design with the best standalone MLP in place of the best ridge. Every arm
SEARCHES its hyperparameters from scratch at every refit; windows shorter than 130
periods are tuned on the 130 periods ending where the window ends, with inner folds
that each train on the window's own length, then refitted on the window alone.

Each arm is paired against the 130-period arm, which is IA_01's own search and must
reproduce mlp/a2_mlp_fenccausal. All rows share one window, so calmar and burn(d)
compare. Payout is formed per period (0.1 * CORR + MMC) before it is summarised.

The second section is E125: the same seven arms FITTED on target_binned_return_60
(runs e125_w<N>), consolidated with E119's. Every one of the fourteen is RE-SCORED
on target_binned_return_20 by src.analysis.cross_target.score_on_payable_target --
the scorer E60 and CURUPIRA_05_60 were graded with -- so a 60-day fit is judged on
the label that pays, and all fourteen sit on the 470 dates they share (the 60-day
label is NaN over the last 40). The fit-20 re-scores are first checked against
their own stored series, so the two halves provably go through one scorer. Paired
t is printed at Newey-West lag 20 (the scored label's overlap) and lag 60 (E60's
conservative choice for a 60-day fit), as E124 does for the ridge.

The hyperparameter block matters as much as the table: an alpha on the TOP of the
grid means the search wanted more shrinkage than the grid offers -- for a short
window, "predict as little as possible" -- and that arm's score is then partly a
statement about where the grid stops.
"""

import os

import numpy as np
import pandas as pd

from src.analysis.ablation_report import newey_west

ROOT = os.path.join("results", "%s", "crypto", "mlp")
LAG = 20
ARMS = [("e119_w5", 5, "1 week"), ("e119_w10", 10, "2 weeks"),
        ("e119_w15", 15, "3 weeks"), ("e119_w20", 20, "4 weeks"),
        ("e119_w22", 22, "1 month"), ("e119_w65", 65, "3 months"),
        ("e119_w130", 130, "6 months")]
CONTROL = "e119_w130"
ORIGINAL = "a2_mlp_fenccausal"

FIT60 = [("e125_w%d" % w, w, label) for _, w, label in ARMS]
FIT60_CONTROL = "e125_w130"
FIT60_ORIGINAL = "e60_mlp_fenccausal"      # IA_01's setup fitted on 60 days, E60
FORECAST = os.path.join("results", "forecast", "crypto", "mlp", "%s.csv")
SCORE20_CACHE = os.path.join("results", "cache", "e119_score20")
LAGS = (20, 60)


def per_period(run):
    path = os.path.join(ROOT % "per_period", run + ".csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).set_index("date")


def hyperparams(run):
    path = os.path.join(ROOT % "hyperparams", run + ".csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def _edge(row, h, key):
    """'^' on the top of the key's grid, 'v' on the bottom, '' inside."""
    top, bottom = "%s_grid_max" % key, "%s_grid_min" % key
    if top in h and np.isclose(row[key], row[top], rtol=1e-9):
        return "^"
    if bottom in h and np.isclose(row[key], row[bottom], rtol=1e-9):
        return "v"
    return ""


def main():
    frames = {r: per_period(r) for r, _, _ in ARMS + [(ORIGINAL, 0, "")]}
    frames = {r: f for r, f in frames.items() if f is not None}
    if CONTROL not in frames:
        print("control %s has not finished" % CONTROL)
        return
    pay = pd.DataFrame({r: 0.1 * f["corr"] + f["mmc"] for r, f in frames.items()}).dropna()
    corr = pd.DataFrame({r: f["corr"] for r, f in frames.items()}).loc[pay.index]
    mmc = pd.DataFrame({r: f["mmc"] for r, f in frames.items()}).loc[pay.index]

    print("\n" + "=" * 118)
    print("E119 -- training window of IA_01: mlp on crypto/v2.0-fenccausal (44 features)")
    print("  hyperparameters SEARCHED at every refit. %d common dates, %s .. %s, refit every 130."
          % (len(pay), pay.index.min(), pay.index.max()))
    print("=" * 118)
    print("  %-11s %5s %-9s %8s %8s %8s %6s %6s %7s %5s | %11s %7s"
          % ("arm", "win", "", "CORR", "MMC", "payout", "p.shp", "dd", "calmar", "burn", "d vs 6mo", "t"))
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
        print("  %-11s %5d %-9s %8.5f %8.5f %8.5f %6.2f %6.2f %7.3f %5d | %11s %7s"
              % (run, window, label, corr[run].mean(), mmc[run].mean(), s.mean(),
                 s.mean() / s.std(ddof=0), dd, s.mean() / dd if dd else np.inf, int((s < 0).sum()),
                 cells[0], cells[1]))

    print("\n  hyperparameters chosen per refit: architecture / alpha / learning rate / batch")
    print("  ^ = on the TOP of that grid (alpha: wanted more shrinkage than offered)   v = on the bottom")
    for run, window, _ in ARMS:
        h = hyperparams(run)
        if h is None:
            continue
        cells = []
        for _, r in h.iterrows():
            cells.append("%s a%.3g%s lr%.2g%s b%d" % (
                str(r["mlp__hidden_layer_sizes"]).replace(" ", ""), r["mlp__alpha"], _edge(r, h, "mlp__alpha"),
                r["mlp__learning_rate_init"], _edge(r, h, "mlp__learning_rate_init"), int(r["mlp__batch_size"])))
        rows_note = ("  (%d training rows: batch clips to the window)" % (window * 300)
                     if window * 300 < 8192 else "")
        print("  %-11s %s%s" % (run, " | ".join(cells), rows_note))

    if ORIGINAL in pay:
        diff = pay[CONTROL] - pay[ORIGINAL]
        d, _, t = newey_west(diff.to_numpy(), lag=LAG)
        print("\n  reproduction: %s vs the original %s (IA_01)  d %+.6f  t %+.2f  max|diff| per period %.2e"
              % (CONTROL, ORIGINAL, d, t, float(np.abs(diff).max())))
        ho, hc = hyperparams(ORIGINAL), hyperparams(CONTROL)
        if ho is not None and hc is not None:
            keys = ["mlp__hidden_layer_sizes", "mlp__alpha", "mlp__learning_rate_init", "mlp__batch_size"]
            same = all((ho[k].astype(str).values == hc[k].astype(str).values).all() for k in keys)
            print("  same hyperparameters chosen at every refit: %s" % ("yes" if same else "NO"))

    if rows:
        print("\n" + "=" * 118)
        best = max([r for r, _, _ in ARMS if r in pay], key=lambda r: pay[r].mean())
        print("  best window by payout: %s" % best)
        worse = [r for r, d, t in rows if t < -2]
        better = [r for r, d, t in rows if t > 2]
        print("  significantly WORSE than 6 months (t < -2): %s" % (", ".join(worse) or "none"))
        print("  significantly BETTER than 6 months (t > +2): %s" % (", ".join(better) or "none"))

    consolidated()


def score20(run, extras):
    """
    A forecast's per-period table on target_binned_return_20, or None if the run
    has not finished. Cached per run and recomputed whenever the forecast is newer,
    because one re-score costs ~20s and the report re-scores sixteen runs.
    """
    path = FORECAST % run
    if not os.path.exists(path):
        return None
    cache = os.path.join(SCORE20_CACHE, run + ".csv")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        return pd.read_csv(cache, index_col="date")
    from src.analysis.cross_target import score_on_payable_target
    forecast = pd.read_csv(path, usecols=["symbol", "date", "prediction"], comment="#",
                           parse_dates=["date"]).set_index(["symbol", "date"])
    ranked = forecast.groupby(level=1)["prediction"].rank(pct=True)
    table, _ = score_on_payable_target(ranked, extras)
    table.index = pd.to_datetime(table.index).strftime("%Y-%m-%d")
    table.index.name = "date"
    os.makedirs(SCORE20_CACHE, exist_ok=True)
    table.to_csv(cache)
    return table


def _row(s, base):
    """CORR/MMC are carried by the caller; this is the payout summary + pairing cells."""
    cum = s.cumsum()
    dd = float((cum.cummax() - cum).max())
    cells = []
    if base is not None:
        for lag in LAGS:
            d, _, t = newey_west((s - base).to_numpy(), lag=lag)
            cells.append(t)
        cells = [d] + cells
    return s.mean(), s.mean() / s.std(ddof=0), dd, (s.mean() / dd if dd else np.inf), int((s < 0).sum()), cells


def consolidated():
    from src.analysis.cross_target import load_scoring_frame

    runs = [r for r, _, _ in ARMS] + [r for r, _, _ in FIT60] + [ORIGINAL, FIT60_ORIGINAL]
    print("\n" + "=" * 118)
    print("E125 -- the same ladder FITTED on target_binned_return_60, consolidated with E119's fit-20 arms")
    print("  every arm RE-SCORED on target_binned_return_20 (cross_target.score_on_payable_target)")
    print("=" * 118)
    extras = load_scoring_frame()
    tables = {r: score20(r, extras) for r in runs}
    missing = [r for r in runs if tables[r] is None]
    tables = {r: t for r, t in tables.items() if t is not None}
    if FIT60_CONTROL not in tables or CONTROL not in tables:
        print("  waiting on: %s" % ", ".join(missing))
        return

    # Gate: the fit-20 arms' re-scores must be their own stored series.
    worst = 0.0
    for run, _, _ in ARMS + [(ORIGINAL, 0, "")]:
        stored = per_period(run)
        if stored is None or run not in tables:
            continue
        both = tables[run][["corr", "mmc"]].join(stored[["corr", "mmc"]], rsuffix="_s", how="inner")
        worst = max(worst, float((both["corr"] - both["corr_s"]).abs().max()),
                    float((both["mmc"] - both["mmc_s"]).abs().max()))
    print("  scorer gate: fit-20 re-scores vs their stored series, max |diff| per period %.1e  (%s)"
          % (worst, "OK" if worst < 1e-6 else "NOT the same scorer -- read with care"))

    rows = [(r, w, l, "20") for r, w, l in ARMS] + [(r, w, l, "60") for r, w, l in FIT60]
    present = [r for r, _, _, _ in rows if r in tables]
    pay = pd.DataFrame({r: 0.1 * tables[r]["corr"] + tables[r]["mmc"] for r in present}).dropna()
    corr = pd.DataFrame({r: tables[r]["corr"] for r in present}).loc[pay.index]
    mmc = pd.DataFrame({r: tables[r]["mmc"] for r in present}).loc[pay.index]
    print("  %d common dates, %s .. %s. dd and calmar are on this window only (comparable across rows, not with the"
          % (len(pay), pay.index.min(), pay.index.max()))
    print("  510-date table above). t is paired Newey-West at lag 20 / lag 60.")
    if missing:
        print("  not finished, left out of the table: %s" % ", ".join(missing))
    print()
    print("  %-10s %4s %-9s %3s %8s %8s %8s %6s %6s %7s %5s | %-26s | %-26s"
          % ("arm", "win", "", "fit", "CORR", "MMC", "payout", "p.shp", "dd", "calmar", "burn",
             "vs fit-20 6 months: d  t20  t60", "fit-60 - fit-20: d  t20  t60"))
    for (r20, w, label), (r60, _, _) in zip(ARMS, FIT60):
        for run, fit in ((r20, "20"), (r60, "60")):
            if run not in pay:
                print("  %-10s %4d %-9s %3s   (not finished)" % (run, w, label, fit))
                continue
            base = None if run == CONTROL else pay[CONTROL]
            mean, shp, dd, calmar, burn, cells = _row(pay[run], base)
            vs6 = "%+.5f %+6.2f %+6.2f" % tuple(cells) if cells else "-"
            same = "-"
            if fit == "60" and r20 in pay:
                d, _, t20 = newey_west((pay[run] - pay[r20]).to_numpy(), lag=20)
                _, _, t60 = newey_west((pay[run] - pay[r20]).to_numpy(), lag=60)
                same = "%+.5f %+6.2f %+6.2f" % (d, t20, t60)
            print("  %-10s %4d %-9s %3s %8.5f %8.5f %8.5f %6.2f %6.2f %7.3f %5d | %26s | %26s"
                  % (run, w, label, fit, corr[run].mean(), mmc[run].mean(), mean, shp, dd, calmar,
                     burn, vs6, same))

    print("\n  fit-60 hyperparameters chosen per refit (same notation as above)")
    for run, window, _ in FIT60:
        h = hyperparams(run)
        if h is None:
            continue
        cells = ["%s a%.3g%s lr%.2g%s b%d" % (
            str(r["mlp__hidden_layer_sizes"]).replace(" ", ""), r["mlp__alpha"], _edge(r, h, "mlp__alpha"),
            r["mlp__learning_rate_init"], _edge(r, h, "mlp__learning_rate_init"), int(r["mlp__batch_size"]))
            for _, r in h.iterrows()]
        print("  %-10s %s" % (run, " | ".join(cells)))

    if FIT60_ORIGINAL in tables:
        a = pd.read_csv(FORECAST % FIT60_CONTROL, usecols=["symbol", "date", "prediction"], comment="#")
        b = pd.read_csv(FORECAST % FIT60_ORIGINAL, usecols=["symbol", "date", "prediction"], comment="#")
        m = a.merge(b, on=["symbol", "date"], suffixes=("", "_o"))
        both = pd.concat([tables[FIT60_CONTROL], tables[FIT60_ORIGINAL]], axis=1, keys=["c", "o"]).dropna()
        dpay = (0.1 * both["c"]["corr"] + both["c"]["mmc"]) - (0.1 * both["o"]["corr"] + both["o"]["mmc"])
        d, _, t = newey_west(dpay.to_numpy(), lag=20)
        print("\n  reproduction: %s vs %s (IA_01's setup on 60d, E60)  predictions max|diff| %.2e over %d rows;"
              "  20d payout d %+.6f t %+.2f" % (FIT60_CONTROL, FIT60_ORIGINAL,
                                                float((m["prediction"] - m["prediction_o"]).abs().max()), len(m), d, t))
        ho, hc = hyperparams(FIT60_ORIGINAL), hyperparams(FIT60_CONTROL)
        if ho is not None and hc is not None:
            keys = ["mlp__hidden_layer_sizes", "mlp__alpha", "mlp__learning_rate_init", "mlp__batch_size"]
            same = all((ho[k].astype(str).values == hc[k].astype(str).values).all() for k in keys)
            print("  same hyperparameters chosen at every refit: %s" % ("yes" if same else "NO"))


if __name__ == "__main__":
    main()
