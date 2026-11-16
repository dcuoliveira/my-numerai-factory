#!/usr/bin/env python3
"""
E145 report: one self-contained HTML file, then printed to PDF.

    python3 experiments/e145_report.py
    python3 -m src.analysis.html_pdf <the html it prints>      # run on lovelace

Every number is read from the CSVs experiments/e145_c2head_evaluation.py wrote (rank dataset in
results/reports/crypto/e145_c2head/, the five-bin copy in .../bin/). Every verdict is computed from
the rule frozen in experiments/e145_registration.md, never typed. The reading in READING is written
after the numbers were seen, and says so.
"""
import base64
import io
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
os.chdir(REPO)
import e133_report as W  # noqa: E402  CSS, table, tint, num

ROOT = "results/reports/crypto/e145_c2head"
BIN = os.path.join(ROOT, "bin")
MODELS = ["ridge", "lgb_ktz", "mlp_ktz"]
NAME = {"ridge": "ridge", "lgb_ktz": "LightGBM", "mlp_ktz": "MLP (linear)"}
CAND = ["ZS_COV_HEAD", "FT_COV_HEAD", "ZS_DELTA_HEAD", "FT_DELTA_HEAD", "ZS_FULL_HEAD", "FT_FULL_HEAD", "ALL_HEAD"]
TUNED = ["ZS_COV_HEAD", "FT_COV_HEAD", "ALL_HEAD"]
FAM = ["ZS_COV", "ZS_DELTA", "FT_COV", "FT_DELTA"]
LABEL = {"ZS_COV_HEAD": "zero-shot, with covariates", "FT_COV_HEAD": "fine-tuned, with covariates",
         "ZS_DELTA_HEAD": "zero-shot, covariates minus AR", "FT_DELTA_HEAD": "fine-tuned, covariates minus AR",
         "ZS_FULL_HEAD": "zero-shot cov + delta", "FT_FULL_HEAD": "fine-tuned cov + delta", "ALL_HEAD": "all four families",
         "HEAD_ONLY": "all four, no v2.0", "ZS_AR_HEAD": "zero-shot AR control", "FT_AR_HEAD": "fine-tuned AR control",
         "P_HIST": "tenure", "P_WARM": "warm-up share", "P_OBS": "context share", "P_NM": "never-forecast (LOOK-AHEAD)", "ALL_H": "all + tenure"}
READING = {}   # filled after the numbers are read


def load():
    D = {"rank": pd.read_csv(os.path.join(ROOT, "paired_deltas.csv")), "bin": pd.read_csv(os.path.join(BIN, "paired_deltas.csv"))}
    return dict(D=D, LV=pd.read_csv(os.path.join(ROOT, "summary_levels.csv")), LVB=pd.read_csv(os.path.join(BIN, "summary_levels.csv")),
                SD=pd.read_csv(os.path.join(ROOT, "paired_deltas_by_seed.csv")), BK=pd.read_csv(os.path.join(ROOT, "paired_deltas_by_block.csv")),
                LG=pd.read_csv(os.path.join(ROOT, "legs.csv")), S=pd.read_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet")),
                AU=json.load(open(os.path.join(ROOT, "audit.json"))), SP=pd.read_csv(os.path.join(ROOT, "spreads.csv")))


def cell(D, m, a, vs="B0", metric="corr", win="dev2", amode="route", bmode="plain"):
    r = D[(D["model"] == m) & (D["arm"] == a) & (D["vs"] == vs) & (D["metric"] == metric) & (D["window"] == win) & (D["amode"] == amode) & (D["bmode"] == bmode)]
    return (float(r["delta"].iloc[0]), float(r["t"].iloc[0])) if len(r) else (np.nan, np.nan)


def criteria(X, m, a):
    """The registered rule, criterion by criterion: {name: (pass, evidence)}."""
    D, Db = X["D"]["rank"], X["D"]["bin"]
    c1 = cell(D, m, a)
    pay = cell(D, m, a, metric="payout")
    sd = X["SD"][(X["SD"]["model"] == m) & (X["SD"]["arm"] == a) & (X["SD"]["vs"] == "B0") & (X["SD"]["amode"] == "route") & (X["SD"]["window"] == "dev2")]
    bk = X["BK"][(X["BK"]["model"] == m) & (X["BK"]["arm"] == a) & (X["BK"]["vs"] == "B0") & (X["BK"]["amode"] == "route") & X["BK"]["block"].between(2, 10)]
    ctl = {v: cell(D, m, a, vs=v, amode="plain") for v in ("P_HIST", "P_WARM", "P_OBS")}
    test = cell(D, m, a, win="test")
    tune = cell(D, m, "tune" + a, vs="tuneB0") if a in TUNED else (np.nan, np.nan)
    binv = cell(Db, m, a)
    seeds_ok = (sd["mean"] > 0).sum() >= 4 if m != "ridge" else True
    out = {
        "1 dev CORR t ≥ 2": (c1[1] >= 2, "%+.4f, t %+.1f" % c1),
        "2 payout not harmful": (pay[1] > -2 and pay[0] > -0.003, "%+.4f, t %+.1f" % pay),
        "3 seeds and blocks": (bool(seeds_ok and (bk["mean"] > 0).sum() >= 6),
                               "%s seeds, %d/9 blocks" % ("%d/5" % (sd["mean"] > 0).sum() if m != "ridge" else "1 seed", (bk["mean"] > 0).sum())),
        "4 beyond controls": (all(v[0] > 0 for v in ctl.values()), ", ".join("%s %+.4f" % (k, v[0]) for k, v in ctl.items())),
        "5 block 11 sign": (test[0] > 0, "%+.4f, t %+.1f" % test),
        "6 retune": (tune[0] > 0 if a in TUNED else None, "%+.4f, t %+.1f" % tune if a in TUNED else "not retuned"),
        "7 five-bin": (binv[0] > 0, "%+.4f, t %+.1f" % binv),
    }
    reject = c1[1] <= -2 or pay[1] <= -2 or test[1] <= -2
    cand = all(v[0] is not False for v in out.values())
    return out, ("REJECT" if reject else "FORWARD TEST ONLY" if cand else "NO IMPROVEMENT")


def leg_verdict(X, m, leg):
    L = X["LG"]
    z = L[(L["model"] == m) & (L["leg"] == leg) & L["chosen"]]
    dv = z[(z["metric"] == "corr") & (z["window"] == "dev2")].iloc[0]
    if dv["w"] == 0:
        return "NO IMPROVEMENT", dv
    pay = z[(z["metric"] == "payout") & (z["window"] == "dev2")].iloc[0]
    te = z[(z["metric"] == "corr") & (z["window"] == "test")].iloc[0]
    if dv["t"] >= 2 and pay["t"] > -2 and pay["delta"] > -0.003 and te["delta"] > 0:
        return "FORWARD TEST ONLY", dv
    if dv["t"] <= -2 or te["t"] <= -2:
        return "REJECT", dv
    return "NO IMPROVEMENT", dv


def fig(f):
    b = io.BytesIO()
    f.savefig(b, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(f)
    return '<img src="data:image/png;base64,%s" style="max-width:100%%">' % base64.b64encode(b.getvalue()).decode()


def cum_figure(X):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = X["S"]
    A = S.groupby(["model", "arm", "mode", "date"], as_index=False)["corr"].mean()
    b0 = A[(A["arm"] == "B0") & (A["mode"] == "plain")].set_index(["model", "date"])["corr"]
    f, ax = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for i, m in enumerate(MODELS):
        for a in ["ZS_COV_HEAD", "FT_COV_HEAD", "ALL_HEAD", "ZS_AR_HEAD"]:
            z = A[(A["model"] == m) & (A["arm"] == a) & (A["mode"] == "route")].set_index("date")["corr"]
            d = (z - b0.xs(m)).dropna().sort_index()
            ax[i].plot(d.index, d.cumsum(), label=LABEL[a], lw=1.2)
        ax[i].axhline(0, color="#888", lw=.6)
        ax[i].axvline(pd.Timestamp("2021-08-27"), color="#bbb", ls=":", lw=.8)
        ax[i].axvline(pd.Timestamp("2026-02-20"), color="#bbb", ls="--", lw=.8)
        ax[i].set_title(NAME[m], fontsize=10)
    ax[0].set_ylabel("cumulative routed ΔCORR vs v2.0")
    ax[2].legend(fontsize=7, loc="best")
    return fig(f)


def block_figure(X):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BK = X["BK"]
    f, ax = plt.subplots(1, 3, figsize=(13, 2.9))
    for i, m in enumerate(MODELS):
        z = BK[(BK["model"] == m) & (BK["vs"] == "B0") & (BK["amode"] == "route") & BK["arm"].isin(CAND)].pivot(index="arm", columns="block", values="mean").reindex(CAND)
        v = np.nanmax(np.abs(z.to_numpy())) or 1
        ax[i].imshow(z.to_numpy(), cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
        ax[i].set_xticks(range(z.shape[1]), z.columns, fontsize=7)
        ax[i].set_yticks(range(len(CAND)), CAND if i == 0 else [""] * len(CAND), fontsize=7)
        ax[i].set_title("%s (±%.3f)" % (NAME[m], v), fontsize=9)
    return fig(f)


def legs_figure(X):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = X["LG"]
    f, ax = plt.subplots(1, 3, figsize=(13, 3), sharey=True)
    for i, m in enumerate(MODELS):
        for leg in sorted(L["leg"].unique()):
            z = L[(L["model"] == m) & (L["leg"] == leg) & (L["metric"] == "corr") & (L["window"] == "dev2") & (L["w"] > 0)].sort_values("w")
            ax[i].plot([0] + list(z["w"]), [0] + list(z["delta"]), marker="o", ms=3, label=leg)
        ax[i].axhline(0, color="#888", lw=.6)
        ax[i].set_title(NAME[m], fontsize=10)
        ax[i].set_xlabel("blend weight w (dev, blocks 2-10)")
    ax[0].set_ylabel("routed ΔCORR vs v2.0")
    ax[2].legend(fontsize=7)
    return fig(f)


def lv_rows(LV, arms, win, mode_of):
    rows = []
    for m in MODELS:
        for a in arms:
            r = LV[(LV["model"] == m) & (LV["arm"] == a) & (LV["window"] == win) & (LV["mode"] == mode_of(a))]
            if not len(r):
                continue
            r = r.iloc[0]
            rows.append([NAME[m], a, W.num(r["corr"]), W.num(r["mmc"]), W.num(r["c_sharpe"], 3), W.num(r["dd_corr"], 3), W.num(r["payout"]),
                         W.num(r["p_sharpe"], 3), W.num(r["dd_pay"], 3), W.num(r["calmar"], 3), "%d" % r["burn"], "%d" % r["n"],
                         W.num(r["predcorr_b0"], 3), "%.1f%%" % (100 * r["routed"])])
    return rows


def delta_rows(D, items, metrics=("corr", "payout"), wins=("dev2", "dev", "test")):
    rows = []
    for m in MODELS:
        for a, vs, am, bm in items:
            row = [NAME[m], "%s vs %s%s" % (a, vs, "" if am == "route" else " (unrouted)")]
            ok = False
            for metric in metrics:
                for w in wins:
                    d, t = cell(D, m, a, vs, metric, w, am, bm)
                    ok |= d == d
                    row.append(W.tint(d, t))
            if ok:
                rows.append(row)
    return rows


def build():
    X = load()
    D = X["D"]["rank"]
    V = {(m, a): criteria(X, m, a) for m in MODELS for a in CAND}
    LV_ = {(m, l): leg_verdict(X, m, l) for m in MODELS for l in sorted(X["LG"]["leg"].unique())}
    statuses = [v[1] for v in V.values()] + [v[0] for v in LV_.values()]
    overall = "FORWARD TEST ONLY" if "FORWARD TEST ONLY" in statuses else \
        "REJECT" if all(s == "REJECT" for s in [v[1] for v in V.values()]) else "NO IMPROVEMENT"
    rejects = sorted({"%s %s" % (NAME[m], a) for (m, a), v in V.items() if v[1] == "REJECT"})
    slug = overall.replace(" ", "_")
    out = os.path.join(ROOT, "E145_do_chronos2_forecasts_help_%s.html" % slug)
    H = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>E145 Chronos-2 forecasts</title><style>%s</style></head><body><main>' % W.CSS,
         '<h1>Do Chronos-2-Synth&#39;s forecasts help on Numerai Crypto? %s</h1>' % overall,
         '<p class="lede"><code>crypto/v2.0-c2head</code>: the 22 v2.0 features plus five summaries of Chronos-2-Synth&#39;s own 20-step quantile forecast '
         'of each coin&#39;s label, zero-shot and LoRA fine-tuned per walk-forward block, with and without the 22 features as past-only covariates. '
         'Rules registered before any fit (<code>experiments/e145_registration.md</code>).</p>']
    rd = READING.get("verdict", "")
    H += ['<div class="verdict"><b>Recommendation: %s.</b> %s</div>' % (overall, rd)]
    if rejects:
        H += ['<p>Cells that meet REJECT: %s.</p>' % ", ".join(rejects)]
    # decision table
    crit = list(next(iter(V.values()))[0])
    rows = []
    for m in MODELS:
        for a in CAND:
            c, v = V[(m, a)]
            rows.append([NAME[m], a] + ['<td class="%s">%s</td>' % ("up" if c[k][0] else "down" if c[k][0] is False else "", c[k][1]) for k in crit] + ["<b>%s</b>" % v])
    H += ['<h2 id="rule">The registered rule, cell by cell</h2><p>Green: the criterion holds; red: it fails; blank: not applicable. '
          'Primary window: dev blocks 2-10, routed (a coin without the arm&#39;s forecasts keeps v2.0&#39;s rank). About 21 family × model cells, so a t between 2 and 2.9 is within multiplicity.</p>',
          W.table(["model", "arm"] + crit + ["verdict"], rows)]
    lrows = [[NAME[m], l, "%.2f" % v[1]["w"], W.tint(v[1]["delta"], v[1]["t"]), "<b>%s</b>" % v[0]] for (m, l), v in LV_.items()]
    H += ['<h3>Direct legs, blended with each model&#39;s v2.0 prediction (weight chosen on dev CORR)</h3>', W.table(["model", "leg", "dev-chosen w", "dev ΔCORR at w", "verdict"], lrows)]
    if READING.get("distinctions"):
        H += ['<h2 id="reading">What the evidence says</h2><ul>%s</ul>' % "".join("<li>%s</li>" % x for x in READING["distinctions"]),
              '<p><small>Written after the numbers were seen. The verdicts above are computed from the registered rule, not from this text.</small></p>']
    # design
    au = X["AU"]
    H += ['<h2 id="design">Design and what deviated</h2><ul>',
          '<li><b>Input per row (coin, D):</b> 256 grid steps ending at D; <code>target_binned_return_20</code> blanked after D − 35 days (a label resolves about 30 days after its date); '
          'the 22 v2.0 features through D as past-only covariates, no future values; NaN where the coin has no row, masked by Chronos and never filled from later dates. '
          'Forecast the next 20 steps; q10/q50/q90 kept. A row needs 32 resolved labels in its context, else no forecast (0.5 at load, routed to v2.0).</li>',
          '<li><b>Model:</b> <code>autogluon/chronos-2-synth</code> at revision 3607918, fp32 inference with TF32 off: batch-invariant to about 2e-6 (fp16 autocast moved forecasts by up to 0.01 with batch composition, so it was not used).</li>',
          '<li><b>Fine-tuning:</b> one LoRA model per mode and block 2-11 (r 8, alpha 16, the library&#39;s default targets, lr 1e-5, 1000 steps, seed 145), on the panel cut at the block&#39;s first train date. '
          'The loop is written here rather than <code>Chronos2Pipeline.fit</code>, so every training window is built exactly like an inference row (label wall 35 days before the origin, past-only covariates, 20-step target); '
          'early stopping on windows whose targets are the last 35 resolved steps, also inside the cut. Blocks 0 and 1 have too few resolved labels before their cut and get no fine-tuned model: their FT columns are NaN and routed.</li>',
          '<li><b>Summaries:</b> mean and terminal q50, OLS slope of q50 on the horizon, mean q90 − q10, mean asymmetry (q90 − q50) − (q50 − q10); deltas COV − AR on raw summaries; '
          'within-date continuous percentile rank (primary) and Numerai&#39;s five bins (robustness, <code>v2.0-c2head-bin</code>). No missingness indicator.</li>',
          '<li><b>Frozen panel:</b> <code>data/crypto/v2.0</code> is rewritten by the live round every morning; every stage read <code>data/crypto/v2.0-asof-e145</code>, byte-identical to the file the first stage loaded.</li>',
          '<li><b>Models:</b> ridge (there is no <code>ridge_ktz</code>; ridge has no prior to compile), <code>lgb_ktz</code>, <code>mlp_ktz</code> (identity activation, i.e. linear), fixed configs, 5 seeds for LightGBM and the MLP. '
          'B0, P_HIST and tuned B0 (ridge, LightGBM) are E133/E135&#39;s on the same 468,000 keys.</li>',
          '<li><b>Windows:</b> walk-forward 130/22/130. Primary dev = blocks 2-10 (every family exists); blocks 0-10 shown too. Block 11 (2026-02-20 to 2026-08-20) is reused evidence, scored by earlier studies. The 28 labelled dates after it and all live dates are unread.</li></ul>']
    # audit
    tr = pd.DataFrame(au["truncation"])
    q = tr[tr["mode"].isin(["cov", "ar"])]
    fin = tr[tr["mode"].str.startswith("final_")]
    arows = [[r.date, r.family, r.mode, "%d" % r.block, "%d" % r.forecasts, "{:,}".format(int(r.q_differ_cut_vs_stored)) + " / {:,}".format(int(r.q_cells)), W.num(r.q_maxdiff_cut_vs_stored, 6),
              ("{:,}".format(int(r.q_differ_cut_vs_full)) if r.family == "ft" else "&ndash;")] for r in q.itertuples()]
    frows = [[r.date, r.family, r.mode.replace("final_", ""), "{:,}".format(int(r.differ)) + " / {:,}".format(int(r.cells)), W.num(r.max_abs_diff, 6)] for r in fin.itertuples()]
    H += ['<h2 id="audit">Point-in-time and reproducibility audit</h2>',
          '<p>For each date T the forecasts were recomputed from a panel physically cut at T (rows after T dropped, labels after T − 35 days blanked). Zero-shot: cut against the stored forecast, same frozen model. '
          'Fine-tuned: the block&#39;s fp16 adapter reloaded and run on both the cut and the uncut panel (cut vs full must agree exactly), and that reload compared with the stored forecasts the in-memory model produced.</p>',
          W.table(["date", "family", "mode", "block", "forecasts", "cells differing, cut vs stored", "max |Δq|", "cut vs uncut (same adapter)"], arows),
          '<h3>After the final transformation (within-date rank / five bins of the 30 summaries)</h3>',
          W.table(["date", "family", "scale", "cells differing", "max |Δ|"], frows),
          '<ul><li><b>Missingness is causal:</b> a forecast exists exactly when the row&#39;s own context holds ≥ 32 resolved labels on %.4f%% of the %s keys.</li>' % (100 * au["missing_rule"]["agree"], "{:,}".format(au["missing_rule"]["rows"])),
          '<li><b>Same keys for every arm:</b> every arm is scored on the same %s keys; no row is dropped for lacking a forecast. Coverage: zero-shot %.1f%%, fine-tuned %.1f%%, both %.1f%%; %.1f%% of keys belong to a coin that never has a forecast (P_NM&#39;s look-ahead flag).</li></ul>'
          % ("{:,}".format(au["keys"]), 100 * au["coverage"]["zs"], 100 * au["coverage"]["ft"], 100 * au["coverage"]["both"], 100 * au["coverage"]["no_feed"])]
    ic = sorted(au["date_ic"].items(), key=lambda kv: -abs(kv[1]["t_dev2"]))
    H += ['<h3>Standalone signal: per-date IC of each summary with the label (dev blocks 2-10, NW lag 20)</h3>',
          W.table(["feature", "mean IC dev", "t", "mean IC block 11"], [[k.replace("feature_", ""), W.num(v["mean_dev2"], 4, True), W.num(v["t_dev2"], 1, True), W.num(v["mean_test"], 4, True)] for k, v in ic])]
    # levels
    HDR = ["model", "arm", "CORR", "MMC", "c.sharpe", "dd(corr)", "payout", "p.sharpe", "dd(pay)", "calmar", "burn", "n", "ρ(pred, B0)", "routed"]
    mode_of = lambda a: "plain" if a in ("B0", "P_HIST", "HEAD_ONLY", "P_WARM", "P_OBS", "P_NM") or a.endswith("_ONLY") else "route"
    H += ['<h2 id="levels">Levels</h2><p>Routed for arms holding v2.0, plain otherwise; per-period payout = 0.1 × CORR + proxy MMC; calmar = mean payout / max drawdown of cumulative payout, comparable only at equal n.</p>',
          '<h3>Dev, blocks 2-10</h3>', W.table(HDR, lv_rows(X["LV"], ["B0"] + CAND + ["ZS_AR_HEAD", "FT_AR_HEAD", "HEAD_ONLY"], "dev2", mode_of)),
          '<h3>Block 11 (reused)</h3>', W.table(HDR, lv_rows(X["LV"], ["B0"] + CAND, "test", mode_of))]
    DH = ["model", "comparison"] + ["%s %s" % (m, w) for m in ("ΔCORR", "Δpayout") for w in ("dev 2-10", "dev 0-10", "block 11")]
    H += ['<h2 id="deltas">Against v2.0</h2><h3>Routed (the deployment score)</h3>', W.table(DH, delta_rows(D, [(a, "B0", "route", "plain") for a in CAND])),
          '<h3>Unrouted</h3>', W.table(DH, delta_rows(D, [(a, "B0", "plain", "plain") for a in CAND])),
          '<h3>Added-only: the forecast summaries without v2.0</h3>', W.table(DH, delta_rows(D, [(a, "B0", "plain", "plain") for a in ["HEAD_ONLY"] + [f + "_ONLY" for f in FAM]])),
          cum_figure(X), '<p><small>Dotted line: first date with fine-tuned features (2021-08-27). Dashed: block 11.</small></p>']
    H += ['<h2 id="families">Each family&#39;s own contribution</h2><h3>Inside ALL_HEAD (ALL_HEAD minus the family)</h3>',
          W.table(DH, delta_rows(D, [("ALL_HEAD", "ALL_NO_" + f, "route", "route") for f in FAM])),
          '<h3>Covariates over AR, fine-tuned over zero-shot, and the AR controls against v2.0</h3><p>A gain over an AR arm counts for nothing unless the arm also beats v2.0.</p>',
          W.table(DH, delta_rows(D, [("ZS_COV_HEAD", "ZS_AR_HEAD", "route", "route"), ("FT_COV_HEAD", "FT_AR_HEAD", "route", "route"),
                                     ("FT_COV_HEAD", "ZS_COV_HEAD", "route", "route"), ("FT_FULL_HEAD", "ZS_FULL_HEAD", "route", "route"),
                                     ("ZS_AR_HEAD", "B0", "route", "plain"), ("FT_AR_HEAD", "B0", "route", "plain")]))]
    H += ['<h2 id="controls">Tenure, warm-up, context-share and look-ahead controls</h2>',
          W.table(DH, delta_rows(D, [(a, "B0", "plain", "plain") for a in ("P_HIST", "P_WARM", "P_OBS", "P_NM")] + [("ALL_H", "P_HIST", "plain", "plain"), ("ALL_H", "P_HIST", "route", "plain")])),
          '<h3>Each candidate beyond each control (unrouted, matched seeds)</h3>',
          W.table(DH, delta_rows(D, [(a, c, "plain", "plain") for a in CAND for c in ("P_HIST", "P_WARM", "P_OBS")])),
          '<p>P_NM is a look-ahead placebo (whether the coin ever has a forecast in the completed file) and is never part of a candidate.</p>']
    sp = X["SP"]
    H += ['<h2 id="retune">Equal-budget retune (one seed)</h2>', W.table(DH, delta_rows(D, [("tune" + a, "tuneB0", md, "plain") for a in TUNED for md in ("route", "plain")])),
          '<p>Collapsed blocks (within-date prediction spread &lt; 1e-8) in the tuned files: %s.</p>' % (", ".join("%s %s: %d" % (r.model, r.file, r.collapsed_blocks) for r in sp[sp["file"].str.startswith("tune")].itertuples() if r.collapsed_blocks) or "none")]
    Db = X["D"]["bin"]
    brows = []
    for m in MODELS:
        for a in CAND:
            r1, b1 = cell(D, m, a), cell(Db, m, a)
            r2, b2 = cell(D, m, a, win="test"), cell(Db, m, a, win="test")
            brows.append([NAME[m], a, W.tint(*r1), W.tint(*b1), W.tint(*r2), W.tint(*b2)])
    H += ['<h2 id="bin">Continuous rank against five bins (routed ΔCORR vs v2.0)</h2>', W.table(["model", "arm", "rank dev 2-10", "bin dev 2-10", "rank block 11", "bin block 11"], brows)]
    L = X["LG"]
    lrow = []
    for m in MODELS:
        for leg in sorted(L["leg"].unique()):
            z = L[(L["model"] == m) & (L["leg"] == leg)]
            al = {w: z[(z["w"] == -1) & (z["metric"] == "corr") & (z["window"] == w)].iloc[0] for w in ("dev2", "test")}
            ch = {w: z[z["chosen"] & (z["metric"] == mt) & (z["window"] == w)].iloc[0] for w in ("dev2", "test") for mt in ("corr",)}
            pc = z[z["chosen"] & (z["metric"] == "payout") & (z["window"] == "dev2")].iloc[0]
            lrow.append([NAME[m], leg, "%s <small>t %s</small>" % (W.num(al["dev2"]["level"]), W.num(al["dev2"]["t"], 1, True)), W.num(al["test"]["level"]),
                         "%.2f" % ch["dev2"]["w"], W.tint(ch["dev2"]["delta"], ch["dev2"]["t"]), W.tint(pc["delta"], pc["t"]), W.tint(ch["test"]["delta"], ch["test"]["t"])])
    H += ['<h2 id="legs">Direct prediction legs</h2><p>The leg alone is the within-date rank of one summary (missing → 0.5), scored as a prediction. The blend is (1 − w) · rank(B0) + w · rank(leg) on covered coins, slotted into v2.0&#39;s ranks; w picked on dev CORR alone.</p>',
          W.table(["model", "leg", "leg alone CORR dev", "leg alone block 11", "chosen w", "blend ΔCORR dev", "blend Δpayout dev", "blend ΔCORR block 11"], lrow), legs_figure(X)]
    SD = X["SD"]
    srows = []
    for m in ("lgb_ktz", "mlp_ktz"):
        for a in CAND:
            z = SD[(SD["model"] == m) & (SD["arm"] == a) & (SD["vs"] == "B0") & (SD["amode"] == "route")]
            srows.append([NAME[m], a] + [W.num(z[(z["seed"] == s) & (z["window"] == "dev2")]["mean"].iloc[0], 4, True) if len(z[(z["seed"] == s) & (z["window"] == "dev2")]) else "&ndash;" for s in range(5)]
                         + [W.num(z[(z["seed"] == s) & (z["window"] == "test")]["mean"].iloc[0], 4, True) if len(z[(z["seed"] == s) & (z["window"] == "test")]) else "&ndash;" for s in range(5)])
    H += ['<h2 id="stability">Stability</h2><h3>By seed (routed ΔCORR vs v2.0)</h3>', W.table(["model", "arm"] + ["dev s%d" % s for s in range(5)] + ["b11 s%d" % s for s in range(5)], srows),
          '<h3>By block (routed ΔCORR vs v2.0, seed mean)</h3>', block_figure(X)]
    cy = pd.DataFrame(au["coverage_by_year"])
    H += ['<h2 id="coverage">Coverage over time</h2>', W.table(["year", "scope", "zero-shot", "fine-tuned"], [[int(r.date), r.kind, "%.1f%%" % (100 * r.zs), "%.1f%%" % (100 * r.ft)] for r in cy.itertuples()])]
    H += ['<h2 id="next">Forward test and next steps</h2><ul>%s</ul>' % "".join("<li>%s</li>" % x for x in READING.get("next", [])),
          '<h2 id="gaps">Gaps</h2><ul><li>No ADOPT is reachable from this data: the rule requires a frozen forward test on ≥ 120 resolved dates after 2026-08-20, none of which has been read.</li>'
          '<li>The live path does not build these features: it needs the chronos2 venv and a GPU at every round, and a fine-tune per refit, none wired into <code>src/live</code>.</li>'
          '<li>Retuning covers B0, ZS_COV_HEAD, FT_COV_HEAD and ALL_HEAD on one seed; the other arms run their fixed configs only.</li></ul>',
          '</main></body></html>']
    open(out, "w").write("\n".join(H))
    print(out, os.path.getsize(out))
    return V, LV_, overall


if __name__ == "__main__":
    build()
