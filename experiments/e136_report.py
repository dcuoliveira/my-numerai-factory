#!/usr/bin/env python3
"""E136 report: the feature-set evaluation of crypto/v2.0-histfe as one self-contained HTML file.

    python3 experiments/e136_report.py
"""
import json
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e133_report as W  # noqa: E402  CSS, crosshair script, table, tint and line_panels

os.chdir(REPO)
ROOT = "results/reports/crypto/e136_histfe_evaluation"
OUT = os.path.join(ROOT, "E136_does_histfe_help_REJECT.html")
MODELS = ["ridge", "lgb_ktz", "mlp_ktz"]
D = pd.read_csv(os.path.join(ROOT, "paired_deltas.csv"))
L = pd.read_csv(os.path.join(ROOT, "summary_levels.csv"))
PW = pd.read_csv(os.path.join(ROOT, "payout_window.csv"))
BL = pd.read_csv(os.path.join(ROOT, "blends.csv"))
BY = pd.read_csv(os.path.join(ROOT, "paired_deltas_by_block.csv"))
SD = pd.read_csv(os.path.join(ROOT, "paired_deltas_by_seed.csv"))
LS = pd.read_csv(os.path.join(ROOT, "lgb_payout_by_seed.csv"))
AU = json.load(open(os.path.join(ROOT, "audit.json")))
R5 = pd.read_csv("results/reports/crypto/e135_route_histfe/paired_deltas.csv")
S = pd.read_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"), columns=["model", "arm", "date", "corr", "payout"])
Z = pd.read_parquet(os.path.join(ROOT, "blend_by_date.parquet"))
OBS = pd.read_parquet("results/reports/crypto/e135_route_histfe/observed.parquet", columns=["date", "obs_FULL"])
W.SERIES.clear()
W.SERIES.update({"FULL": 1, "F_RANK": 2, "TSC": 3, "P_HIST": 4, "F_RANK_H": 5})
W.ARM_NAME.update({"FULL": "FULL", "F_RANK": "F_RANK", "TSC": "TSC", "P_HIST": "tenure control", "F_RANK_H": "F_RANK + tenure"})
_table = W.table
W.table = lambda head, rows, cls="": _table(head, [[str(c) for c in r] for r in rows], cls)


def d(m, a, v, metric, sc):
    return D[(D["model"] == m) & (D["arm"] == a) & (D["vs"] == v) & (D["metric"] == metric) & (D["scope"] == sc)].iloc[0]


def verdict(m, a, v="B0"):
    """The numeric reading of the skill's vocabulary, the same bar as E134."""
    dv, tt = d(m, a, v, "corr", "dev"), d(m, a, v, "corr", "test")
    pd_, pt = d(m, a, v, "payout", "dev"), d(m, a, v, "payout", "test")
    if dv["t"] <= -2:
        return "REJECT"
    if dv["t"] >= 2 and tt["delta"] > 0 and pd_["t"] > -2 and pt["t"] > -2:
        return "ADOPT"
    if tt["t"] >= 2:
        return "PROMISING"
    return "NO IMPROVEMENT"


def t4(m, a, v, metric, sc):
    r = d(m, a, v, metric, sc)
    return W.tint(r["delta"], r["t"])


def pw(m, a, v, sc):
    return PW[(PW["model"] == m) & (PW["arm"] == a) & (PW["vs"] == v) & (PW["scope"] == sc)].iloc[0]


def arm_rows(pairs, verdicts=True):
    rows = []
    for a, v in pairs:
        for m in MODELS:
            r = [a if v == "B0" else "%s vs %s" % (a, v), m, t4(m, a, v, "corr", "dev"), t4(m, a, v, "corr", "test"), t4(m, a, v, "payout", "dev"), t4(m, a, v, "payout", "test")]
            rows.append(r + [verdict(m, a, v)] if verdicts else r)
    return rows


def curves(col, keys):
    A = S.groupby(["model", "arm", "date"], as_index=False)[[col]].mean().pivot_table(index=["model", "date"], columns="arm", values=col)
    out = {}
    for m in MODELS:
        x = A.xs(m, level="model").dropna(subset=["B0"])
        out[m] = x[keys].sub(x["B0"], axis=0).cumsum()
    return out


def level_rows(arms):
    rows = []
    for m in MODELS:
        for a in arms:
            for sc in ("dev", "test"):
                r = L[(L["model"] == m) & (L["arm"] == a) & (L["scope"] == sc)].iloc[0]
                rows.append([m, a, sc, W.num(r["corr"]), W.num(r["mmc"]), W.num(r["c_sharpe"], 2), W.num(r["dd_corr"], 2), W.num(r["payout"]), W.num(r["p_sharpe"], 2), W.num(r["dd_pay"], 2),
                             W.num(r["calmar"], 3), int(r["burn"]), "t20", int(r["n"]), int(r["n_pay"]), W.num(r["predcorr_b0"], 2)])
    return rows


def main():
    assert all(verdict(m, "FULL") == "REJECT" for m in MODELS)
    assert not any(verdict(m, a) == "ADOPT" for m in MODELS for a in ("SHORT", "LONG", "COMBO", "TSC", "FULL", "F_RANK", "F_MA", "F_TSMAX", "ADDED_ONLY"))
    assert verdict("lgb_ktz", "F_RANK") == "REJECT" and verdict("lgb_ktz", "TSC") == "PROMISING" and verdict("lgb_ktz", "SHORT") == "PROMISING"
    assert all(t["differ"] == 0 for t in AU["truncation"])
    H = [W.table(["model", "FULL ΔCORR dev", "FULL ΔCORR test", "FULL Δpayout dev", "FULL Δpayout test", "verdict"],
                 [[{"ridge": "ridge", "lgb_ktz": "LightGBM", "mlp_ktz": "MLP (linear)"}[m], t4(m, "FULL", "B0", "corr", "dev"), t4(m, "FULL", "B0", "corr", "test"),
                   t4(m, "FULL", "B0", "payout", "dev"), t4(m, "FULL", "B0", "payout", "test"), "REJECT"] for m in MODELS])]
    x1, x2, x3 = pw("lgb_ktz", "F_RANK_H", "P_HIST", "dev"), pw("lgb_ktz", "F_RANK_H", "P_HIST", "test"), pw("lgb_ktz", "F_RANK", "B0", "dev")
    s = ['<h1>Do own-history features help on Numerai Crypto? REJECT</h1>',
         '<p class="sub"><code>crypto/v2.0-histfe</code> (374 features) against <code>crypto/v2.0</code> (22). The full evaluation procedure: E133 (dev factorial), E135 (final block, routing), E136 (the remaining controls).</p>',
         '<nav><a href="#design">Design</a><a href="#audit">Audit</a><a href="#levels">Levels</a><a href="#deltas">Deltas</a><a href="#ablation">Ablations</a><a href="#controls">Controls</a><a href="#lead">The lead</a><a href="#blends">Blends</a><a href="#stability">Stability</a><a href="#gaps">Gaps</a></nav>',
         '<div class="verdict"><b>Verdict: REJECT.</b> The dataset does not beat v2.0, and nothing reaches ADOPT. Ridge and the MLP are harmed on both windows. LightGBM is REJECT on full-dev CORR and flat on test. '
         'By the registered bar, lgb TSC and lgb SHORT reach PROMISING BUT UNCONFIRMED on test alone. One lead is worth tracking: LightGBM with own-history rank plus tenure. The bar marks that arm REJECT on full-dev CORR, so it is a reading made after the fact, not a verdict. Do not deploy the dataset.</div>'] + H
    s += ['<p>Cells are the change against v2.0 with the Newey-West t at lag 20, tinted only at |t| &ge; 2. Payout uses proxy MMC, never official MMC, and is built per period. Dev is 1,430 CORR dates and 468 payout dates. Test is 130 and 120.</p>']

    s += ['<h2 id="design">Design and what deviated</h2><ul>',
          '<li><b>Fits:</b> 12 walk-forward blocks, train 130, purge 22, predict 130. Target <code>target_binned_return_20</code>. Fixed configs from raw v2.0: ridge, lgb_ktz, mlp_ktz (linear). Five seeds for LightGBM and the MLP.</li>',
          '<li><b>Rows:</b> every arm on the same 468,000 keys. No row was dropped, because every coin has the base features.</li>',
          '<li><b>Reused:</b> E133 dev predictions and E135 final-block predictions. B0 on the final block matches E132 to 1.2e-7.</li>',
          '<li><b>New in E136:</b> 99 fits for four ablations, three block + tenure arms, a warm-up placebo, and ADDED_ONLY on the final block.</li>',
          '<li><b>Test is not virgin.</b> E65, E131, E132 and E134 scored those dates on other features, and E135 read them for histfe. No arm or weight was chosen from them.</li>',
          '<li><b>No P_NM arm.</b> histfe has no look-ahead flag to add, so P_WARM replaces it (see Controls).</li>',
          '<li><b>Retune:</b> E133&#39;s small grid and one seed, not 50 draws.</li></ul>']

    tr = AU["truncation"]
    ov = AU["overlap"]
    OBS["date"] = pd.to_datetime(OBS["date"])
    yr = OBS.assign(y=OBS["date"].dt.year, warm=OBS["obs_FULL"] < .5).groupby("y").agg(obs=("obs_FULL", "mean"), warm=("warm", "mean"), none=("obs_FULL", lambda v: (v == 0).mean()))
    s += ['<h2 id="audit">Point-in-time audit</h2>',
          '<p><b>Clean.</b> The block was rebuilt on the real panel cut at four dates, and %s of %s cells differ from the dataset. So our build uses no later rows. This assumes Numerai&#39;s shipped bins are themselves point-in-time.</p>' % ("{:,}".format(sum(t["differ"] for t in tr)), "{:,}".format(sum(t["cells"] for t in tr))),
          '<p>Missing history is warm-up, not a look-ahead coverage flag. Warm-up keys, under half the block observed, are %.1f%% of all keys. They overlap weakly with E132&#39;s no-feed flag: correlation %.2f, %.1f%% of no-feed keys against %.1f%% of others.</p>' % (ov["warm"] * 100, ov["corr"], ov["warm_given_no_feed"] * 100, ov["warm_given_feed"] * 100),
          W.table(["year"] + [str(y) for y in yr.index], [["share of block observed"] + [W.num(v, 3) for v in yr["obs"]], ["keys in warm-up"] + ["%.1f%%" % (v * 100) for v in yr["warm"]], ["keys with no history at all"] + ["%.1f%%" % (v * 100) for v in yr["none"]]])]

    s += ['<h2 id="levels">Levels</h2>',
          W.table(["model", "arm", "scope", "CORR", "MMC", "c.sharpe", "dd(corr)", "payout", "p.sharpe", "dd(pay)", "calmar", "burn(d)", "tgt", "n", "n(pay)", "corr B0"], level_rows(["B0", "FULL", "F_RANK", "TSC", "P_HIST", "P_WARM", "F_RANK_H", "ADDED_ONLY"]), "l3"),
          '<p><small>MMC, payout, p.sharpe, dd(pay), calmar and burn(d) cover n(pay) dates. Compare calmar and burn(d) only within a scope. corr B0 is the mean per-date rank correlation of predictions with B0.</small></p>']

    hdr = ["arm", "model", "ΔCORR dev", "ΔCORR test", "Δpayout dev", "Δpayout test", "verdict"]
    s += ['<h2 id="deltas">Paired deltas</h2><h3>Added feature groups, against v2.0</h3>',
          W.table(hdr, arm_rows([(a, "B0") for a in ("SHORT", "LONG", "COMBO", "TSC", "FULL", "F_RANK", "F_MA", "F_TSMAX", "ADDED_ONLY")]), "l2"),
          '<p>Across these 27 cells the label depends on the window. lgb F_RANK is REJECT here at dev t %s, and PROMISING when routed at t &minus;1.96, so that label carries no information. The next sections read it properly.</p>' % W.num(d("lgb_ktz", "F_RANK", "B0", "corr", "dev")["t"], 2, True),
          '<h3>Effect of the live routing rule (E135)</h3>']
    rt = []
    for m in MODELS:
        g = R5[(R5["model"] == m) & (R5["arm"] == "FULL")]
        c = lambda rule, sc: W.tint(*g[(g["rule"] == rule) & (g["scope"] == sc)][["mean", "nw_t"]].iloc[0])  # noqa: E731
        rt.append([m, t4(m, "FULL", "B0", "corr", "dev"), c("R_SLOT", "dev"), c("R_HALF", "dev"), t4(m, "FULL", "B0", "corr", "test"), c("R_SLOT", "test"), c("R_HALF", "test")])
    s += [W.table(["model", "plain dev", "R_SLOT dev", "R_HALF dev", "plain test", "R_SLOT test", "R_HALF test"], rt), '<p>Routing FULL changes little, because only 0.3% of keys fall back, or 5.1% under R_HALF.</p>',
          '<h3>Equal-budget retune</h3>', W.table(hdr[:-1], arm_rows([("tuneFULL", "tuneB0")], False), "l2"), '<p>Retuning does not rescue FULL. Every ΔCORR is negative.</p>']

    ab = [[a[3:] if a.startswith("NO_") else a, m] + [t4(m, "FULL", a, x, sc) for x in ("corr", "payout") for sc in ("dev", "test")] for a in ("NO_RANK", "NO_MA", "NO_TSMAX", "NO_TSC") for m in MODELS]
    s += ['<h2 id="ablation">Group ablations</h2><p>Each row is FULL minus FULL without that family, so a negative cell means the family hurts when added to the others.</p>',
          W.table(["family", "model", "ΔCORR dev", "ΔCORR test", "Δpayout dev", "Δpayout test"], ab, "l2"),
          '<p>No single family explains the harm. On ridge and the MLP, rank and days-since-max both hurt on dev, and days-since-max also hurts on test. MA differences hurt ridge on test. On LightGBM no family reaches |t| 2 in either direction, so removing any one of them changes little.</p>']

    s += ['<h2 id="controls">Tenure and placebo controls</h2>',
          '<p><b>P_WARM</b> adds one causal feature to v2.0: the binned share of the block observed. It carries only depth of history, no history signal. It is the histfe placebo: a gain it can match is not evidence for the block.</p>',
          W.table(hdr, arm_rows([("P_HIST", "B0"), ("P_WARM", "B0"), ("P_HIST", "P_WARM"), ("F_RANK", "P_WARM"), ("FULL_H", "P_HIST"), ("F_RANK_H", "P_HIST"), ("TSC_H", "P_HIST")]), "l2"),
          '<ul><li><b>Tenure&#39;s gain does not hold.</b> It clears the numeric ADOPT bar on dev, but its test deltas are at most +0.0019, |t| below 0.5, so treat it as unconfirmed.</li>',
          '<li><b>The placebo absorbs most of F_RANK&#39;s dev payout gain.</b> One warm-up feature earns %s (t %s) on LightGBM, and F_RANK beats it by only %s (t %s).</li>' % (W.num(d("lgb_ktz", "P_WARM", "B0", "payout", "dev")["delta"], 4, True), W.num(d("lgb_ktz", "P_WARM", "B0", "payout", "dev")["t"], 1, True), W.num(d("lgb_ktz", "F_RANK", "P_WARM", "payout", "dev")["delta"], 4, True), W.num(d("lgb_ktz", "F_RANK", "P_WARM", "payout", "dev")["t"], 1, True)),
          '<li><b>So the dev case for the block is weak.</b> Depth of history alone does most of what history features appear to do on dev.</li></ul>']

    pl = [[m, a + (" vs " + v if v != "B0" else ""), sc] + [W.tint(r["corr_delta"], r["corr_t"]), W.tint(r["payout_delta"], r["payout_t"]), "%s, %s" % (W.num(r["payout_lo"], 4, True), W.num(r["payout_hi"], 4, True)), int(r["n"])]
          for m in ("lgb_ktz",) for a, v in (("F_RANK", "B0"), ("F_RANK", "P_WARM"), ("P_HIST", "B0"), ("F_RANK_H", "B0"), ("F_RANK_H", "P_HIST"), ("TSC", "B0")) for sc in ("dev", "test") for r in [pw(m, a, v, sc)]]
    z = LS[(LS["arm"] == "F_RANK_H") & (LS["vs"] == "P_HIST")]
    s += ['<h2 id="lead">The one lead: LightGBM, own-history rank + tenure</h2>',
          '<p>Dev CORR covers 2020 onwards, and dev payout only starts in May 2024, when the meta model does. Reading both on payout&#39;s dates removes the apparent conflict between negative CORR and positive payout:</p>',
          W.table(["model", "comparison", "scope", "ΔCORR", "Δpayout", "payout 95% CI", "n"], pl, "l3"),
          '<ul><li><b>For it:</b> F_RANK + tenure beats tenure alone on both recent windows: dev payout %s (t %s), test %s (t %s). All 5 seeds are positive on both, and the test CI excludes zero. lgb F_RANK alone has dev CORR of %s (t %s) on these dates.</li>' % (W.num(x1["payout_delta"], 4, True), W.num(x1["payout_t"], 1, True), W.num(x2["payout_delta"], 4, True), W.num(x2["payout_t"], 1, True), W.num(x3["corr_delta"], 4, True), W.num(x3["corr_t"], 1, True)),
          '<li><b>Against it:</b> full-dev CORR is negative, %s (t %s) against tenure, driven by blocks 0, 3 and 5. Against v2.0 the test payout gain is only %s (t %s). The dev t against tenure is below 2. It is LightGBM only: ridge and the MLP are harmed by the same arm.</li>' % (W.num(d("lgb_ktz", "F_RANK_H", "P_HIST", "corr", "dev")["delta"], 4, True), W.num(d("lgb_ktz", "F_RANK_H", "P_HIST", "corr", "dev")["t"], 1, True), W.num(pw("lgb_ktz", "F_RANK_H", "B0", "test")["payout_delta"], 4, True), W.num(pw("lgb_ktz", "F_RANK_H", "B0", "test")["payout_t"], 1, True)),
          '<li><b>This is a reading after the fact.</b> The payout window is natural, but it was opened because of what these numbers showed, and this arm is one of many. F_RANK + tenure was also not compared with the placebo. That is why it stays a lead. The final block is now used up, so only dates after 2026-08-20 can confirm it. The dataset already holds 28 labelled dates after that, unread here. They were left unread on purpose: they span about one independent 20-day period and have no meta model, so they cannot give a payout reading.</li></ul>',
          W.table(["seed", "dev Δpayout", "dev t", "test Δpayout", "test t"], [[str(k)] + [W.num(z[(z["seed"] == k) & (z["scope"] == sc)][c].iloc[0], n, True) for sc in ("dev", "test") for c, n in (("payout_delta", 4), ("t", 1))] for k in sorted(z["seed"].unique())]),
          '<p><small>lgb F_RANK + tenure against tenure, by seed.</small></p>',
          W.line_panels(curves("corr", ["FULL", "F_RANK", "TSC", "P_HIST", "F_RANK_H"]), ["FULL", "F_RANK", "TSC", "P_HIST", "F_RANK_H"], lambda m: m + ": cumulative ΔCORR vs v2.0"),
          W.line_panels(curves("payout", ["FULL", "F_RANK", "TSC", "P_HIST", "F_RANK_H"]), ["FULL", "F_RANK", "TSC", "P_HIST", "F_RANK_H"], lambda m: m + ": cumulative Δpayout vs v2.0, proxy MMC")]

    A = Z.groupby(["model", "date", "kind", "w"], as_index=False)[["corr", "payout"]].mean()
    bt = []
    for m in MODELS:
        for sc in ("dev", "test"):
            q = A[(A["model"] == m) & (A["kind"] == sc)].groupby("w")[["corr", "payout"]].mean()
            bt.append([m, sc, "CORR"] + [W.num(v) for v in q["corr"]])
            bt.append([m, sc, "payout"] + [W.num(v) for v in q["payout"]])
    br = [[r.model, r.metric, W.num(r.w, 2), r.scope, W.num(r.b0), W.num(r.level)] + ([W.tint(r.delta, r.t), "%s, %s" % (W.num(r.lo, 4, True), W.num(r.hi, 4, True))] if r.w > 0 else ["no blend", "&ndash;"]) for r in BL.itertuples()]
    s += ['<h2 id="blends">Blends</h2><p>Rank-blend of B0 and ADDED_ONLY. The weight comes from dev alone, and test then scores it once.</p>',
          W.table(["model", "picked on", "w", "scope", "B0", "blend", "Δ", "95% CI"], br, "l3"),
          '<p><b>No blend earns a place.</b> Dev picks little or no weight, and no gain reaches t 2 on either window. The best is LightGBM at w = 0.10, CORR %s on test, t %s.</p>' % (W.num(BL[(BL["model"] == "lgb_ktz") & (BL["metric"] == "corr") & (BL["scope"] == "test")]["delta"].iloc[0], 4, True), W.num(BL[(BL["model"] == "lgb_ktz") & (BL["metric"] == "corr") & (BL["scope"] == "test")]["t"].iloc[0], 1, True)),
          W.table(["model", "scope", "metric"] + ["w = %.2f" % w for w in sorted(Z["w"].unique())], bt, "l3")]

    blk = []
    for a, v in (("FULL", "B0"), ("F_RANK", "B0"), ("TSC", "B0"), ("F_RANK_H", "P_HIST")):
        for m in MODELS:
            g = BY[(BY["model"] == m) & (BY["arm"] == a) & (BY["vs"] == v)].sort_values("block")
            blk.append([a + (" vs tenure" if v != "B0" else ""), m] + [W.num(x, 3, True) for x in g["mean"]])
    sd = []
    for a in ("FULL", "F_RANK", "TSC", "F_RANK_H"):
        for m in MODELS[1:]:
            for sc in ("dev", "test"):
                g = SD[(SD["model"] == m) & (SD["arm"] == a) & (SD["scope"] == sc)]
                sd.append([a, m, sc, W.num(g["mean"].min(), 4, True), W.num(g["mean"].max(), 4, True), "%d of %d" % ((g["mean"] < 0).sum(), len(g)), "%d of %d" % ((g["t"].abs() >= 2).sum(), len(g))])
    s += ['<h2 id="stability">Stability</h2><h3>By block, mean ΔCORR (block 11 is test)</h3>',
          W.table(["arm", "model"] + [str(i) for i in range(12)], blk, "l2"),
          '<h3>By seed, ΔCORR against v2.0</h3>', W.table(["arm", "model", "scope", "lowest", "highest", "below 0", "|t| &ge; 2"], sd, "l3")]

    s += ['<h2 id="gaps">Gaps and next steps</h2><ul>',
          '<li>No catboost or forest arm was tested, and mlp_ktz is linear. MMC is a proxy.</li>',
          '<li>The final block is spent for histfe, so nothing here can be confirmed on it again.</li>',
          '<li>P_WARM and tenure were not routed. Their keys have nothing to fall back from.</li>',
          '<li>The arms in the block tables were scored plain. E135 is the routed reading.</li>',
          '<li>E136 ran no search over tenure or windows, and it has not tested the lead with meta-residualisation.</li></ul>',
          '<p><b>Recommended:</b> keep live slots on the existing book and do not deploy <code>crypto/v2.0-histfe</code>. Register lgb F_RANK + tenure as the single forward candidate, unstaked, and judge it only on dates after 2026-08-20, once a full block of about 130 dates with meta-model coverage exists. Stop further feature work on this dataset until then.</p>']

    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Histfe evaluation</title><style>%s\nnav a{margin-right:1em}h3{font-size:1rem;margin:1.4em 0 .3em}</style></head><body><main>%s</main><script>%s</script></body></html>'
           % (W.CSS, "\n".join(s), W.JS))
    open(OUT, "w").write(doc)
    print(OUT, os.path.getsize(OUT))


if __name__ == "__main__":
    main()
