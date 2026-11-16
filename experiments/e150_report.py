#!/usr/bin/env python3
"""E150 report: one self-contained HTML file, then printed to PDF.

    python3 experiments/e150_report.py
    python3 -m src.analysis.html_pdf <the html it prints>      # on lovelace (Chrome)

Every number comes from the CSVs experiments/e150_macrovol_evaluation.py --score wrote. Verdicts are
computed, never typed, with the house bar applied to ROUTED arm vs ROUTED VC_COV on the matched covered
sample: REJECT dev CORR t <= -2; ADOPT dev CORR t >= +2, test delta > 0 and payout t > -2 on both windows;
PROMISING test CORR t >= +2; else NO IMPROVEMENT. The dataset-level rejection rules of the request are
checked separately and listed with the verdict.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e133_report as W  # noqa: E402

os.chdir(REPO)
ROOT = "results/reports/crypto/e150_macrovol"
DATA = "data/crypto/v2.0-vol-causal-macrovol"
MODELS = ["ridge", "lgb_ktz", "mlp_ktz"]
NAME = {"ridge": "ridge", "lgb_ktz": "LightGBM", "mlp_ktz": "MLP (linear)"}
SOURCES = ["GOLD", "SP500", "DXY", "Y10", "VIXLVL", "VIXVOV"]
LABEL = {"GOLD": "gold", "SP500": "S&amp;P 500", "DXY": "DXY", "Y10": "10-year yield", "VIXLVL": "VIX level", "VIXVOV": "VIX vol-of-vol"}

D = pd.read_csv(os.path.join(ROOT, "paired_deltas.csv"))
L = pd.read_csv(os.path.join(ROOT, "levels.csv"))
RG = pd.read_csv(os.path.join(ROOT, "regime_deltas.csv"))
SD = pd.read_csv(os.path.join(ROOT, "seed_deltas.csv"))
CV = pd.read_csv(os.path.join(ROOT, "coverage_by_year.csv"))
PS = json.load(open(os.path.join(ROOT, "parsim.json")))
AU = json.load(open(os.path.join(DATA, "audit.json")))
S = pd.read_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"), columns=["model", "arm", "date", "kind", "corr", "payout"])


def row(m, a, b, metric, sc):
    q = D[(D["model"] == m) & (D["arm"] == a) & (D["vs"] == b) & (D["metric"] == metric) & (D["scope"] == sc)]
    return q.iloc[0] if len(q) else None


def cell(m, a, b, metric, sc):
    r = row(m, a, b, metric, sc)
    return W.tint(r["delta"], r["t"]) if r is not None else "&ndash;"


def verdict(m, a, b="VC_COV"):
    dc, tc, dp, tp = (row(m, a, b, x, sc) for x, sc in (("corr", "dev"), ("corr", "test"), ("payout", "dev"), ("payout", "test")))
    if dc is None:
        return "&ndash;"
    if dc["t"] <= -2:
        return "REJECT"
    if dc["t"] >= 2 and tc["delta"] > 0 and dp["t"] > -2 and tp["t"] > -2:
        return "ADOPT"
    if tc["t"] >= 2:
        return "PROMISING"
    return "NO IMPROVEMENT"


def table_rows(pairs, verdicts=True):
    out = []
    for a, b in pairs:
        for m in MODELS:
            r = row(m, a, b, "corr", "dev")
            if r is None:
                continue
            x = [a + ("" if b == "VC_COV" else " vs " + b), NAME[m]] + [cell(m, a, b, x, sc) for x in ("corr", "mmc", "payout") for sc in ("dev", "test")]
            x += ["%.0f%%" % (100 * r["win"]), W.num(r["worst_block"], 4, True)]
            out.append(x + [verdict(m, a, b)] if verdicts else x)
    return out


HDR = ["arm", "model", "ΔCORR dev", "ΔCORR test", "ΔMMC dev", "ΔMMC test", "Δpayout dev", "Δpayout test", "win dev", "worst block dev"]


def curves(arms_, base):
    A = S[S["arm"].isin(arms_ + [base])].groupby(["model", "arm", "date"], as_index=False)[["corr"]].mean().pivot_table(index=["model", "date"], columns="arm", values="corr")
    out = {}
    for m in MODELS:
        x = A.xs(m, level="model").dropna(subset=[base])
        out[m] = pd.DataFrame({a: (x[a] - x[base]).fillna(0) for a in arms_ if a in x}).cumsum()
    return out


def main():
    fam = {m: {k: verdict(m, "F_" + k) for k in SOURCES} for m in MODELS}
    allv = {m: verdict(m, "ALL") for m in MODELS}
    psv = {m: verdict(m, "PARSIM") for m in MODELS} if PS["sources"] else {}
    order = ["REJECT", "NO IMPROVEMENT", "PROMISING", "ADOPT"]
    best = max((v for m in MODELS for v in list(fam[m].values()) + [allv[m]] + list(psv.values()) if v in order), key=order.index, default="NO IMPROVEMENT")
    adopt_any = [(m, k) for m in MODELS for k, v in fam[m].items() if v == "ADOPT"]
    # dataset-level rejection checks of the request
    chk = []
    fut = [row(m, "PLC_FUTURE", "VC_COV", "corr", "dev") for m in MODELS]
    al = [row(m, "ALL", "VC_COV", "corr", "dev") for m in MODELS]
    chk.append(("Look-ahead placebo (future macro)", "dev ΔCORR %s" % ", ".join("%s %s (t %s)" % (NAME[m], W.num(r["delta"], 4, True), W.num(r["t"], 1, True)) for m, r in zip(MODELS, fut))))
    chk.append(("Circular placebo", "dev ΔCORR %s" % ", ".join("%s %s (t %s)" % (NAME[m], W.num(r["delta"], 4, True), W.num(r["t"], 1, True)) for m, r in zip(MODELS, [row(m, "PLC_CIRC", "VC_COV", "corr", "dev") for m in MODELS]))))
    chk.append(("Scalar control (rank-equivalent to v)", "bins equal the vol-causal bins in %.0f%% of cells; dev ΔCORR %s" % (100 * AU["scalar_control_equals_vol_bin"]["vixlvl_sqret20"],
               ", ".join("%s %s" % (NAME[m], W.num(row(m, "CTL_SCALAR", "VC_COV", "corr", "dev")["delta"], 4, True)) for m in MODELS))))
    chk.append(("Covered training sample alone", "B_COV vs B_ALL dev ΔCORR %s" % ", ".join("%s %s (t %s)" % (NAME[m], W.num(row(m, "B_COV", "B_ALL", "corr", "dev")["delta"], 4, True), W.num(row(m, "B_COV", "B_ALL", "corr", "dev")["t"], 1, True)) for m in MODELS)))
    head = best if best != "ADOPT" else "ADOPT (see checks)"
    tag = best.replace(" ", "_")
    out = os.path.join(ROOT, "E150_does_macro_vol_regime_help_%s.html" % tag)

    vt = [[NAME[m]] + [fam[m][k] for k in SOURCES] + [allv[m]] + ([psv[m]] if psv else []) for m in MODELS]
    H = ['<h1>Does a coin&#39;s volatility work differently across macro-volatility regimes? %s</h1>' % head,
         '<p class="sub"><code>crypto/v2.0-vol-causal-macrovol</code> (754 features: v2.0&#39;s 22, vol-causal&#39;s 154, 578 macro-volatility interactions and exposures) against <code>crypto/v2.0-vol-causal</code> (176), each family on the same covered rows. Registered before launch as E150.</p>',
         '<div class="verdict"><b>Best verdict on any family or combination: %s.</b> %s</div>' % (head,
             ("Families that reach ADOPT: %s." % ", ".join("%s on %s" % (LABEL[k], NAME[m]) for m, k in adopt_any)) if adopt_any else
             "No macro family reaches ADOPT on any model. Verdicts are routed arm against routed VC_COV on the matched covered sample, the deployment score."),
         '<h3>Verdict per macro family, against vol-causal on the same covered rows</h3>',
         W.table(["model"] + [LABEL[k] for k in SOURCES] + ["all six"] + (["PARSIM"] if psv else []), vt),
         '<p>PARSIM, picked on dev alone by the frozen rule (F_&lt;SRC&gt; beats VC_COV on dev CORR at t &ge; 2 on at least 2 models, payout t &gt; &minus;2 on all): %s.</p>' % (", ".join(LABEL[k.upper()] for k in PS["sources"]) or "<b>no family qualified</b>, so there is no PARSIM arm"),
         '<h3>Rejection checks from the request</h3>', W.table(["check", "result"], [[a, b] for a, b in chk])]

    H += ['<h2>Design</h2><ul>',
          '<li><b>Matched covered sample:</b> B_COV (22 shipped), VC_COV (22 + 154 vol-causal) and every macro arm are trained and predicted only on rows where the vol levels, the own-history ranks and every macro source&#39;s ungated columns are observed (one common mask). The regime-gated hi/lo columns are missing by design and are not in the mask; after binning the model sees 0.5 there.</li>',
          '<li><b>Routing:</b> v2.0&#39;s full-universe model (B_ALL, reused from E133/E135) ranks every coin; an arm reorders only the covered coins inside the rank positions B_ALL gave them; every other coin keeps B_ALL&#39;s exact rank. All comparisons are routed arm vs routed arm on the same dates.</li>',
          '<li><b>Fits:</b> 12 walk-forward blocks (train 130, purge 22, predict 130), fixed ridge / lgb_ktz / mlp_ktz configs (there is no ridge_ktz or gbm_ktz; ridge and lgb_ktz stand in), 5 seeds for LightGBM and the MLP. Block 0 (2020) has 2 covered training rows, because the mask needs a year of history, and is excluded from every arm and comparison.</li>',
          '<li><b>Retune:</b> the same small grid on B_COV, VC_COV and ALL, one seed, picked per block on the last 40 train dates after a 22-date purge; the MLP alpha grid is capped at 1.</li>',
          '<li><b>Final block</b> (2026-02-20 to 08-20) selects nothing here, but earlier studies scored those dates, so it is not virgin.</li></ul>']
    cv = CV[CV["kind"].isin(["dev", "test"])]
    tr = AU["truncation"]
    H += ['<h2>Dataset audit</h2><ul>',
          '<li>vol-causal columns, rows and missing values preserved exactly: %s.</li>' % ("yes" if AU["vol_causal_preserved"] else "<b>NO</b>"),
          '<li>Cut-date rebuild (panel, spot, archive and macro store cut at %s): %s of %s new cells differ.</li>' % (", ".join(t["date"] for t in tr), "{:,}".format(sum(t["differ"] for t in tr)), "{:,}".format(sum(t["cells"] for t in tr))),
          '<li>This proves the arithmetic reads nothing after its date. It does not prove the macro or spot histories were published as they are now: the study is timestamp-causal on a retrospectively reconstructed universe. Raw macro downloads are kept with hashes for later comparison.</li>',
          '<li>Regime gates: the high-stress side is open on %s of dates and the low-stress side on %s (VIX level shown; every source in the dataset audit).</li>' % ("%.0f%%" % (100 * AU["regime_dates_share"]["vixlvl"]["hi"]), "%.0f%%" % (100 * AU["regime_dates_share"]["vixlvl"]["lo"])),
          '<li>sgn interactions are not the vol bin in disguise: per-date |Spearman| with the coin&#39;s 20-day vol bin has median %.2f and is never above 0.9; the scalar control is identical to the vol bin, as it must be.</li>' % AU["sgn_vs_vol_per_date"]["median_abs"],
          '<li>Coin-level exposures: BTC&#39;s S&amp;P-correlation bin averaged %.2f in 2022 and %.2f in 2024.</li></ul>' % (AU["btc_bins_example"]["feature_mvx_sp500_corr_120"]["2022"], AU["btc_bins_example"]["feature_mvx_sp500_corr_120"]["2024"]),
          W.table(["year", "scope", "vol levels", "own-history ranks", "macro (all six)", "covered"],
                  [[int(r.date), r.kind] + ["%.1f%%" % (100 * getattr(r, c)) for c in ("cov_all", "cov_tsr_all", "cov_mvx_all", "covered")] for r in cv.itertuples()])]

    lv = L[L["arm"].isin(["B_ALL", "B_COV", "VC_COV", "ALL", "PARSIM"])].sort_values(["model", "arm", "scope"])
    H += ['<h2>Levels, routed</h2>', W.table(["model", "arm", "scope", "CORR", "MMC", "c.sharpe", "dd(corr)", "payout", "p.sharpe", "dd(pay)", "calmar", "burn(d)", "covered", "n", "n(pay)"],
                                              [[r.model, r.arm, r.scope, W.num(r.corr), W.num(r.mmc), W.num(r.c_sharpe, 2), W.num(r.dd_corr, 2), W.num(r.payout), W.num(r.p_sharpe, 2),
                                                W.num(r.dd_pay, 2), W.num(r.calmar, 3), int(r.burn), "%.0f%%" % (100 * r.covered), int(r.n), int(r.n_pay)] for r in lv.itertuples()], "l3"),
          '<p><small>Every row is routed through B_ALL, so B_ALL itself is v2.0. MMC is a proxy against the historical meta model, payout 0.1&times;CORR + MMC per period. Compare calmar and burn(d) within one scope only.</small></p>']
    H += ['<h2>Paired deltas, routed</h2><h3>The baselines</h3>', W.table(HDR, table_rows([("VC_COV", "B_COV"), ("B_COV", "B_ALL")], False), "l2"),
          '<h3>Each macro family against vol-causal</h3>', W.table(HDR + ["verdict"], table_rows([("F_" + k, "VC_COV") for k in SOURCES]), "l2"),
          '<h3>All six, the dev-selected combination, and the exposure split</h3>', W.table(HDR + ["verdict"], table_rows([("ALL", "VC_COV"), ("PARSIM", "VC_COV"), ("NO_EXPO", "VC_COV"), ("EXPO_ONLY", "VC_COV")]), "l2"),
          '<h3>Leave one source out: each source&#39;s margin given the other five</h3>', W.table(HDR, table_rows([("ALL", "LOO_" + k) for k in SOURCES], False), "l2"),
          '<h3>Controls and placebos</h3>', W.table(HDR, table_rows([("CTL_SCALAR", "VC_COV"), ("PLC_CIRC", "VC_COV"), ("PLC_FUTURE", "VC_COV"), ("ALL", "PLC_CIRC")], False), "l2"),
          '<p>PLC_FUTURE reads macro values 20 trading days ahead: look-ahead, excluded from every selection. PLC_CIRC rotates each macro series by 504 trading days, keeping its autocorrelation and breaking its timing. CTL_SCALAR multiplies each raw crypto variance by the raw positive macro variance, a common positive factor per date, so its bins equal the vol bins.</p>',
          '<h3>Equal-budget retune</h3>', W.table(HDR, table_rows([("tuneVC_COV", "tuneB_COV"), ("tuneALL", "tuneVC_COV"), ("tunePARSIM", "tuneVC_COV")], False), "l2")]
    rg = []
    for a in ["F_" + k for k in SOURCES] + ["ALL", "VC_COV", "PLC_FUTURE"]:
        for m in MODELS:
            q = RG[(RG["arm"] == a) & (RG["model"] == m) & (RG["scope"] == "dev")]
            if len(q):
                rg.append([a + ("" if a != "VC_COV" else " vs B_COV"), NAME[m]] + [W.tint(*q[q["regime"] == r][["delta", "t"]].iloc[0]) if (q["regime"] == r).any() else "&ndash;" for r in ("high", "mid", "low")])
    H += ['<h2>High versus low macro-volatility regimes</h2><p>Dates split by the VIX-level percentile attached as-of D&minus;1 (the pm the gates read): high &ge; 0.80, low &le; 0.20. Dev, routed ΔCORR against VC_COV.</p>',
          W.table(["arm", "model", "high", "mid", "low"], rg, "l2")]
    sd = []
    for a in ["ALL"] + ["F_" + k for k in SOURCES]:
        for m in ("lgb_ktz", "mlp_ktz"):
            for sc in ("dev", "test"):
                q = SD[(SD["arm"] == a) & (SD["model"] == m) & (SD["scope"] == sc)].sort_values("seed")
                if len(q):
                    sd.append([a, NAME[m], sc] + [W.num(x, 4, True) for x in q["delta"]])
    H += ['<h2>Stability</h2><h3>ΔCORR against VC_COV by seed</h3>', W.table(["arm", "model", "scope"] + ["seed %d" % i for i in range(5)], sd, "l2")]
    keys = ["ALL", "F_VIXLVL", "F_SP500", "PLC_FUTURE"]
    W.SERIES.clear()
    W.SERIES.update({k: i + 1 for i, k in enumerate(keys)})
    W.ARM_NAME.update({"ALL": "all six", "F_VIXLVL": "VIX level", "F_SP500": "S&P 500", "PLC_FUTURE": "future placebo"})
    H += [W.line_panels(curves(keys, "VC_COV"), keys, lambda m: m + ": cumulative routed ΔCORR vs vol-causal"),
          '<h2>Gaps</h2><ul><li>Macro histories are today&#39;s downloads; the dated raw runs only start now. The crypto side inherits vol-causal&#39;s reconstructed universe and its thin coverage after 2026-08-20.</li>',
          '<li>A crypto close is the UTC day&#39;s last bar, a US close about 20:00&ndash;21:00 UTC, so exposures are daily co-movement with a few hours&#39; offset.</li>',
          '<li>The squared-distance family was not built: for non-negative values it is a monotone transform of the distance, so its per-date bins would be identical.</li>',
          '<li>The retune used one seed and a small grid; MMC is a proxy; the final block is not virgin.</li></ul>']
    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E150 macro-vol regimes</title><style>%s\nh3{font-size:1rem;margin:1.4em 0 .3em}</style></head><body><main>%s</main><script>%s</script></body></html>'
           % (W.CSS, "\n".join(H), W.JS))
    open(out, "w").write(doc)
    print(out)


if __name__ == "__main__":
    main()
