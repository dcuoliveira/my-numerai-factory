#!/usr/bin/env python3
"""
E189 report: the 5 x 7 table and heatmaps, paired tests with Holm corrections, the preregistered
selection and its confirmation, stability, coverage cohorts, penalty audit, validation and audit.

    python3 experiments/e189_c2norm.py --report      # writes ROOT/E189_report.html + tables
    python3 -m src.analysis.html_pdf results/reports/crypto/e189_c2norm/E189_report.html
"""
import base64
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
sys.path.insert(0, REPO)
import e131_analysis as E  # noqa: E402
import e189_c2norm as M  # noqa: E402
from src.prediction import embedding_norm as N  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = M.ROOT
AX_LABEL = {"CS": "across assets", "TS": "per-asset time", "POOL": "dates x assets", "DIM": "embedding dims",
            "GRP": "assets within size group"}
TF_LABEL = {"Z": "z-score", "ROB": "robust", "MM": "min-max", "PCT": "pct rank", "RNK": "rank-normal",
            "CZ": "clipped z", "BIN": "Numerai bins"}


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def paired(S, a, b, col, kind):
    x = S[(S["arm"] == a) & (S["kind"] == kind)].set_index("date")[col]
    y = S[(S["arm"] == b) & (S["kind"] == kind)].set_index("date")[col]
    d = (x - y).dropna()
    m, se, t = E.newey_west(d.values)
    lo, hi = E.block_bootstrap_ci(d.values)
    return dict(delta=m, se=se, t=t, p=2 * norm.sf(abs(t)) if t == t else np.nan, boot_lo=lo, boot_hi=hi, n=len(d))


def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def heat(mat, title, fmt, ann=None, center=None, cmap="RdBu_r"):
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    v = mat.to_numpy(float)
    if center is not None:
        lim = np.nanmax(np.abs(v - center))
        im = ax.imshow(v, cmap=cmap, vmin=center - lim, vmax=center + lim)
    else:
        im = ax.imshow(v, cmap="viridis")
    ax.set_xticks(range(mat.shape[1]), [TF_LABEL[c] for c in mat.columns], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]), [AX_LABEL[r] for r in mat.index], fontsize=8)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            s = fmt % v[i, j]
            if ann is not None:
                s += "\n" + ann.iloc[i, j]
            ax.text(j, i, s, ha="center", va="center", fontsize=6.5, color="black",
                    bbox=dict(facecolor="white", alpha=0.6, lw=0, pad=0.5))
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03)
    return fig_b64(fig)


def grid(series):
    return pd.DataFrame([[series["N_%s_%s" % (a, t)] for t in N.TRANSFORMS] for a in N.AXES], index=N.AXES, columns=N.TRANSFORMS)


def cohorts(arms):
    """Covered-only per-date CORR within size group, tenure tercile and date-coverage tercile."""
    k = pd.read_parquet(os.path.join(ROOT, "preds", "RAW.parquet"), columns=["symbol", "ucid", "date", "covered", "group", "kind", M.T20])
    df = pd.read_parquet(M.DATA, columns=["ucid", "date"]).reset_index(drop=True) if False else None  # noqa: F841
    full = pd.read_parquet(os.path.join(ROOT, "preds", "B_ALL.parquet"), columns=["ucid", "date", "covered"])
    # tenure: number of the coin's EARLIER covered rows in the cache up to that date (causal)
    allk = pd.read_parquet(M.DATA).reset_index()[["ucid", "date"]]
    allk["date"] = pd.to_datetime(allk["date"])
    allk = allk.sort_values(["date", "ucid"]).reset_index(drop=True)
    allk = allk[allk["date"] <= M.LAST]
    ids = np.concatenate([np.load(os.path.join(M.CACHE, "part_%03d.npz" % i))["cov_rows"] for i in range(35)])
    allk["cov"] = False
    allk.loc[allk.index.intersection(ids), "cov"] = True
    allk["tenure"] = allk.groupby("ucid")["cov"].cumsum() - allk["cov"]
    k = k.merge(allk[["ucid", "date", "tenure"]], on=["ucid", "date"], how="left", validate="1:1")
    share = full.groupby("date")["covered"].mean()
    k = k[k["covered"]].copy()
    k["ten_q"] = k.groupby("date")["tenure"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
    sq = pd.qcut(share.rank(method="first"), 3, labels=False)
    k["date_cov_q"] = k["date"].map(sq)
    P = {a: pd.read_parquet(os.path.join(ROOT, "preds", a + ".parquet"), columns=["symbol", "date", "prediction"]) for a in arms}
    rows = []
    for a, p in P.items():
        g = k.merge(p, on=["symbol", "date"], how="left", validate="1:1")
        for name, col, vals in (("size group", "group", {0: "small", 1: "mid", 2: "large"}),
                                ("tenure tercile", "ten_q", {0: "young", 1: "middle", 2: "old"})):
            for v, lab in vals.items():
                h = g[g[col] == v]
                s = h.groupby("date").apply(lambda x: E.corr_one(x.set_index("symbol")) if len(x) >= 10 else np.nan)
                kk = h.groupby("date")["kind"].first()
                for kind in ("dev", "test"):
                    rows.append(dict(arm=a, cohort=name, level=lab, kind=kind, corr=s[kk == kind].mean()))
        s = g.groupby("date").apply(lambda x: E.corr_one(x.set_index("symbol")))
        kk = g.groupby("date")["kind"].first()
        for v, lab in {0: "low", 1: "mid", 2: "high"}.items():
            m = (g.groupby("date")["date_cov_q"].first() == v)
            for kind in ("dev", "test"):
                rows.append(dict(arm=a, cohort="date coverage share", level=lab, kind=kind, corr=s[m & (kk == kind)].mean()))
    return pd.DataFrame(rows)


def main():
    S = pd.read_parquet(os.path.join(ROOT, "scores.parquet"))
    S["date"] = pd.to_datetime(S["date"])
    W = S.loc[(S["arm"] == "RAW") & S["mmc"].notna(), "date"]
    assert (S.groupby("arm")["mmc"].apply(lambda s: s.notna().sum()).nunique() == 1), "meta window differs across arms"
    arms = M.ARMS
    T = []
    for a in arms:
        r = {"arm": a}
        for kind in ("dev", "test"):
            x = S[(S["arm"] == a) & (S["kind"] == kind)]
            dc, dp = E.describe(x["corr"]), E.describe(x["payout"])
            r.update({"%s_corr" % kind: dc["mean"], "%s_corr_sd" % kind: dc["std"], "%s_corr_sharpe" % kind: dc["sharpe"],
                      "%s_corr_dd" % kind: dc["max_dd"], "%s_mmc" % kind: x["mmc"].mean(), "%s_payout" % kind: dp["mean"],
                      "%s_pay_sharpe" % kind: dp["sharpe"], "%s_pay_dd" % kind: dp["max_dd"], "%s_burn" % kind: int((x["payout"] < 0).sum()),
                      "%s_n" % kind: dc["n"], "%s_n_pay" % kind: dp["n"], "%s_cov_corr" % kind: x["cov_corr"].mean(),
                      "%s_cov_payout" % kind: x["cov_payout"].mean()})
        T.append(r)
    T = pd.DataFrame(T).set_index("arm")
    # ---- families of paired tests (dev, routed CORR primary; payout reported alongside)
    fam = {}
    for ref in ("RAW", "B_COV"):
        rr = pd.DataFrame({a: paired(S, a, ref, "corr", "dev") for a in M.EMB_ARMS}).T
        rr["p_holm"] = holm(rr["p"])
        pp = pd.DataFrame({a: paired(S, a, ref, "payout", "dev") for a in M.EMB_ARMS}).T
        rr["pay_delta"], rr["pay_t"] = pp["delta"], pp["t"]
        fam[ref] = rr
    best = T.loc[M.EMB_ARMS, "dev_corr"].idxmax()
    ind = pd.DataFrame({a: paired(S, best, a, "corr", "dev") for a in M.EMB_ARMS if a != best}).T
    ind["p_holm"] = holm(ind["p"])
    indist = [best] + [a for a in ind.index if ind.loc[a, "p_holm"] >= 0.05]
    within_axis, within_tf = [], []
    for ax in N.AXES:
        names = ["N_%s_%s" % (ax, t) for t in N.TRANSFORMS]
        b = T.loc[names, "dev_corr"].idxmax()
        r = pd.DataFrame({a: paired(S, b, a, "corr", "dev") for a in names if a != b}).T
        r["p_holm"] = holm(r["p"])
        within_axis.append((ax, b, r))
    for tf in N.TRANSFORMS:
        names = ["N_%s_%s" % (ax, tf) for ax in N.AXES]
        b = T.loc[names, "dev_corr"].idxmax()
        r = pd.DataFrame({a: paired(S, b, a, "corr", "dev") for a in names if a != b}).T
        r["p_holm"] = holm(r["p"])
        within_tf.append((tf, b, r))
    # ---- confirmation of the frozen selection
    conf = pd.DataFrame({ref: paired(S, best, ref, "corr", "test") for ref in ("RAW", "B_COV", "B_ALL")}).T
    conf["p_holm"] = holm(conf["p"])
    confp = pd.DataFrame({ref: paired(S, best, ref, "payout", "test") for ref in ("RAW", "B_COV", "B_ALL")}).T
    conf["pay_delta"], conf["pay_t"], conf["pay_boot_lo"], conf["pay_boot_hi"] = confp["delta"], confp["t"], confp["boot_lo"], confp["boot_hi"]
    base = pd.DataFrame({a: paired(S, a, b, c, k) for a, b, c, k in
                         [("RAW", "B_COV", "corr", "dev"), ("RAW", "B_ALL", "corr", "dev"), ("B_COV", "B_ALL", "corr", "dev"),
                          ("RAW", "B_COV", "corr", "test"), ("B_COV", "B_ALL", "corr", "test")]}.items() if False} or {}).T
    base = pd.DataFrame([dict(comparison="%s - %s (%s %s)" % (a, b, k, c), **paired(S, a, b, c, k)) for a, b, c, k in
                         [("RAW", "B_COV", "corr", "dev"), ("RAW", "B_ALL", "corr", "dev"), ("B_COV", "B_ALL", "corr", "dev"),
                          ("RAW", "B_COV", "payout", "dev"), ("B_COV", "B_ALL", "payout", "dev"),
                          ("RAW", "B_COV", "corr", "test"), ("B_COV", "B_ALL", "corr", "test"), ("RAW", "B_COV", "payout", "test")]])
    # ---- stability: per dev block and dev halves, selected and RAW vs B_COV
    blocks = []
    for a in [best, "RAW"] + [w[1] for w in within_axis]:
        for ref in ("RAW", "B_COV"):
            if a == ref:
                continue
            x = S[(S["arm"] == a)].set_index("date")
            y = S[(S["arm"] == ref)].set_index("date")
            d = (x["corr"] - y["corr"]).to_frame("d").join(x["block"])
            blocks.append(dict(arm=a, ref=ref, **{"b%02d" % b: v for b, v in d.groupby("block")["d"].mean().items()}))
    blocks = pd.DataFrame(blocks).drop_duplicates(subset=["arm", "ref"])
    dev_dates = np.sort(S.loc[S["kind"] == "dev", "date"].unique())
    half = dev_dates[len(dev_dates) // 2]
    halves = []
    for a in M.EMB_ARMS + ["RAW"]:
        for ref in ("RAW", "B_COV"):
            if a == ref:
                continue
            x = S[(S["arm"] == a) & (S["kind"] == "dev")].set_index("date")["corr"]
            y = S[(S["arm"] == ref) & (S["kind"] == "dev")].set_index("date")["corr"]
            d = x - y
            halves.append(dict(arm=a, ref=ref, h1=d[d.index < half].mean(), h2=d[d.index >= half].mean()))
    halves = pd.DataFrame(halves)
    rank_h = halves[halves["ref"] == "B_COV"].set_index("arm").loc[M.EMB_ARMS]
    rho_halves = rank_h["h1"].rank().corr(rank_h["h2"].rank())
    coh = cohorts([best, "RAW", "B_COV"])
    # ---- penalty audit
    pa = []
    for a in arms:
        j = json.load(open(os.path.join(ROOT, "preds", a + ".picks.json")))
        P = pd.DataFrame(j["picks"])
        pa.append(dict(arm=a, median_alpha=P["alpha"].median(), min_alpha=P["alpha"].min(), max_alpha=P["alpha"].max(),
                       boundary_blocks=int(P["boundary"].sum()), expansions=int(P["expansions"].sum()),
                       median_alpha_over_n_var=P["alpha_over_n_var"].median(), mean_col_var=P["mean_col_var"].median(),
                       ts_fallback_rows=P["ts_fallback_rows"].dropna().sum() if "ts_fallback_rows" in P else np.nan,
                       minutes=j["seconds"] / 60, host=j["host"]))
    pa = pd.DataFrame(pa).set_index("arm")
    # ---- save tables
    T.to_csv(os.path.join(ROOT, "table_arms.csv"))
    for ref, rr in fam.items():
        rr.to_csv(os.path.join(ROOT, "tests_vs_%s.csv" % ref))
    conf.to_csv(os.path.join(ROOT, "confirmation.csv"))
    pa.to_csv(os.path.join(ROOT, "penalty_audit.csv"))
    coh.to_csv(os.path.join(ROOT, "cohorts.csv"), index=False)
    blocks.to_csv(os.path.join(ROOT, "stability_blocks.csv"), index=False)
    summary = dict(selected=best, indistinguishable=indist, n_indist=len(indist), rho_halves=rho_halves,
                   holm_sig_vs_raw_pos=[a for a in M.EMB_ARMS if fam["RAW"].loc[a, "p_holm"] < .05 and fam["RAW"].loc[a, "delta"] > 0],
                   holm_sig_vs_raw_neg=[a for a in M.EMB_ARMS if fam["RAW"].loc[a, "p_holm"] < .05 and fam["RAW"].loc[a, "delta"] < 0],
                   holm_sig_vs_bcov_pos=[a for a in M.EMB_ARMS if fam["B_COV"].loc[a, "p_holm"] < .05 and fam["B_COV"].loc[a, "delta"] > 0],
                   holm_sig_vs_bcov_neg=[a for a in M.EMB_ARMS if fam["B_COV"].loc[a, "p_holm"] < .05 and fam["B_COV"].loc[a, "delta"] < 0],
                   confirmation=conf.to_dict("index"), meta_window=[str(W.min())[:10], str(W.max())[:10]])
    json.dump(summary, open(os.path.join(ROOT, "summary.json"), "w"), indent=1, default=float)
    write_html(S, T, fam, ind, indist, within_axis, within_tf, conf, base, blocks, halves, rho_halves, coh, pa, best, W)
    print(json.dumps(summary, indent=1, default=float))
    print(T[["dev_corr", "dev_payout", "test_corr", "test_payout", "dev_cov_corr"]].round(4).to_string())


def tbl(df, fmt=None, index=True):
    return df.to_html(float_format=(fmt or (lambda v: "%.4f" % v)), index=index, border=0, classes="t", na_rep="")


def write_html(S, T, fam, ind, indist, within_axis, within_tf, conf, base, blocks, halves, rho_halves, coh, pa, best, W):
    val = json.load(open(os.path.join(ROOT, "validation.json")))
    rep = json.load(open(os.path.join(ROOT, "replay.json"))) if os.path.exists(os.path.join(ROOT, "replay.json")) else {}
    audit = open(os.path.join(ROOT, "audit.md")).read() if os.path.exists(os.path.join(ROOT, "audit.md")) else "(audit pending)"
    reading = open(os.path.join(ROOT, "reading.md")).read() if os.path.exists(os.path.join(ROOT, "reading.md")) else ""
    dev = grid(T["dev_corr"])
    tr = fam["RAW"]
    tb = fam["B_COV"]
    ann_raw = grid(tr.apply(lambda r: "t %+.1f%s" % (r["t"], "*" if r["p_holm"] < .05 else ""), axis=1))
    ann_b = grid(tb.apply(lambda r: "t %+.1f%s" % (r["t"], "*" if r["p_holm"] < .05 else ""), axis=1))
    h1 = heat(dev, "Dev mean routed CORR (blocks 0-10, %d dates)" % T.loc["RAW", "dev_n"], "%.4f")
    h2 = heat(grid(tr["delta"]), "Dev routed CORR minus RAW (NW lag 20; * = Holm p<.05 over 35)", "%+.4f", ann_raw, center=0)
    h3 = heat(grid(tb["delta"]), "Dev routed CORR minus matched v2.0 B_COV (* = Holm p<.05 over 35)", "%+.4f", ann_b, center=0)
    h4 = heat(grid(T["dev_payout"]), "Dev routed payout, meta window %s..%s (secondary)" % (str(W.min())[:10], str(W.max())[:10]), "%.4f")
    h5 = heat(grid(T["test_corr"]), "Confirmation-period routed CORR (EXPLORATORY for all but the selected arm)", "%.4f")
    h6 = heat(grid(T["dev_cov_corr"]), "Dev covered-only CORR (diagnostic: covered rows, unrouted)", "%.4f")
    # cumulative curves
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    for a, c in ((best, "C3"), ("RAW", "C0"), ("B_COV", "C2"), ("B_ALL", "k")):
        for ax, col in zip(axs, ("corr", "payout")):
            x = S[S["arm"] == a].set_index("date")[col].dropna()
            ax.plot(x.index, x.cumsum(), c, lw=1, label=a)
    for ax, t in zip(axs, ("cumulative routed CORR", "cumulative routed payout")):
        ax.axvline(M.TEST_START, color="grey", ls="--", lw=0.8)
        ax.set_title(t, fontsize=9)
        ax.legend(fontsize=7)
    cum = fig_b64(fig)
    main_tbl = []
    for a in M.EMB_ARMS + ["RAW", "B_COV", "B_ALL"]:
        r = T.loc[a]
        d = dict(arm=a, dev_corr=r["dev_corr"], dev_sharpe=r["dev_corr_sharpe"], dev_dd=r["dev_corr_dd"],
                 dev_mmc=r["dev_mmc"], dev_payout=r["dev_payout"], dev_pay_dd=r["dev_pay_dd"], dev_burn=r["dev_burn"],
                 test_corr=r["test_corr"], test_payout=r["test_payout"], cov_dev_corr=r["dev_cov_corr"])
        if a in tr.index:
            d.update(d_raw=tr.loc[a, "delta"], t_raw=tr.loc[a, "t"], holm_raw=tr.loc[a, "p_holm"],
                     d_bcov=tb.loc[a, "delta"], t_bcov=tb.loc[a, "t"], holm_bcov=tb.loc[a, "p_holm"])
        main_tbl.append(d)
    main_tbl = pd.DataFrame(main_tbl).set_index("arm")
    wa = "".join("<h4>%s: dev-best %s</h4>%s" % (AX_LABEL[ax], b, tbl(r[["delta", "t", "p_holm"]])) for ax, b, r in within_axis)
    wt = "".join("<h4>%s: dev-best %s</h4>%s" % (TF_LABEL[tf], b, tbl(r[["delta", "t", "p_holm"]])) for tf, b, r in within_tf)
    cohp = coh.pivot_table(index=["cohort", "level"], columns=["kind", "arm"], values="corr")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>E189 Chronos normalisation</title>
<style>body{{font-family:Helvetica,Arial,sans-serif;font-size:10.5px;margin:18px;max-width:1100px}}
h1{{font-size:18px}} h2{{font-size:14px;border-bottom:1px solid #aaa;margin-top:22px}} h4{{margin:8px 0 2px}}
table.t{{border-collapse:collapse;font-size:9px}} table.t td,table.t th{{padding:1px 5px;border-bottom:1px solid #ddd;text-align:right}}
img{{max-width:100%}} pre{{white-space:pre-wrap;font-size:9.5px}} .k{{background:#f4f4f4;padding:6px}}</style></head><body>
<h1>E189 &mdash; does normalisation change the value of frozen Chronos-2-Synth hidden states?</h1>
<p>Frozen <code>autogluon/chronos-2-synth</code> (rev 3607918a, synthetic-only pretraining), cov-mode 768-d target-patch state at the
input to <code>output_patch_embedding</code>. Embedding-only pooled ridge (repo <code>ridge</code>; no ridge_ktz exists), each arm tuned
from scratch per block on a common 25-point alpha grid 1e-3..1e9 (H.pick rule). 11 dev blocks (2020-08-28..2026-02-19) for selection,
block 11 (2026-02-20..2026-08-20) reserved for confirmation. Routed = R_SLOT into B_ALL's covered rank slots; uncovered coins keep B_ALL
ranks. Primary metric: dev mean routed CORR; payout/MMC on the meta window {str(W.min())[:10]}..{str(W.max())[:10]}.</p>
<div class="k">{reading}</div>
<h2>1. Preregistered selection and confirmation</h2>
<p>Selected (argmax dev routed CORR over 35): <b>{best}</b>. Statistically indistinguishable from it on dev after Holm over 34
paired tests: <b>{len(indist)}</b> arms ({', '.join(indist)}).</p>
<p>Confirmation (block 11, frozen selection; Holm over the 3 comparisons, CORR primary):</p>{tbl(conf[['delta','t','p','p_holm','boot_lo','boot_hi','n','pay_delta','pay_t','pay_boot_lo','pay_boot_hi']])}
<h2>2. The 5 x 7 matrix</h2>
<img src="data:image/png;base64,{h1}"><img src="data:image/png;base64,{h2}"><img src="data:image/png;base64,{h3}">
<img src="data:image/png;base64,{h4}"><img src="data:image/png;base64,{h6}"><img src="data:image/png;base64,{h5}">
<h2>3. Full table (routed unless marked cov_)</h2>{tbl(main_tbl)}
<p>dd = max drawdown of the cumulative series (positive); burn = negative-payout dates; dev payout n = {int(T.loc['RAW','dev_n_pay'])} dates,
test payout n = {int(T.loc['RAW','test_n_pay'])}; CORR n = {int(T.loc['RAW','dev_n'])} dev / {int(T.loc['RAW','test_n'])} test.
d_raw / d_bcov are paired dev routed-CORR deltas, t from Newey-West lag 20, holm_* are Holm-adjusted over the 35 arms.</p>
<h2>4. Baselines</h2>{tbl(base.set_index('comparison')[['delta','t','p','boot_lo','boot_hi','n']])}
<img src="data:image/png;base64,{cum}">
<h2>5. Across axes and within transformations</h2><h3>Within each axis (7 transforms vs the axis's dev-best, Holm over 6)</h3>{wa}
<h3>Within each transformation (5 axes vs its dev-best, Holm over 4)</h3>{wt}
<h3>Indistinguishability from the dev-best (Holm over 34)</h3>{tbl(ind[['delta','t','p','p_holm']].sort_values('delta'))}
<h2>6. Stability</h2><p>Mean dev routed-CORR delta per block (b00..b10 dev, b11 confirmation):</p>{tbl(blocks.set_index(['arm','ref']))}
<p>Dev halves (split {str(np.sort(S.loc[S['kind']=='dev','date'].unique())[len(S.loc[S['kind']=='dev','date'].unique())//2])[:10]}): Spearman of
the 35 arms' delta-vs-B_COV ranking between halves = {rho_halves:.2f}.</p>{tbl(halves[halves['ref']=='B_COV'].set_index('arm')[['h1','h2']])}
<h2>7. Coverage cohorts (covered-only CORR, per-date mean)</h2>{tbl(cohp)}
<h2>8. Penalty audit</h2><p>Common grid 1e-3..1e9 (25 log points); boundary rule: extend 3 decades, at most twice, inner data only.
alpha/(n var) puts the pick on the scale of the transformed features.</p>{tbl(pa)}
<h2>9. Validation</h2><pre>{json.dumps({k: val[k] for k in ('extraction_ok','keys','groups','uncovered_refused','ok')}, indent=1, default=str)}
cut-date replay of transformed features, max |diff| per date: {json.dumps({k: v['max_abs_diff'] for k, v in val['cut_replay_features'].items()})}
future-perturbation invariance (all 35 arms): {val['future_perturbation_invariant']['all']}
live-path prediction replay, max |diff| per date: {json.dumps({k: v['max_abs_diff'] for k, v in rep.items()})}
extraction (re-extract vs cache, cut-panel vs cache): {json.dumps({d: dict(full=r['full']['max_abs'], cut=r['cut']['max_abs'], rerun=r['rerun_identical'], h=r['horizon_steps'], patch=r['patch']) for d, r in val['extraction']['dates'].items()})}</pre>
<h2>10. Missing-values audit</h2><div>{audit}</div>
</body></html>"""
    open(os.path.join(ROOT, "E189_report.html"), "w").write(html)


if __name__ == "__main__":
    main()
