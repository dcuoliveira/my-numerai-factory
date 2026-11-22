"""Three summary tables for every Chronos-2-Synth hidden-state model tried (E189, E190, E191, E192, E201).

One table per hidden-state group: own state; own + average of other coins; all coins' states.
Rows are models (estimator x state transform), with the window (dev / test) as an outer row level.
Columns: state only | state + v2.0, each with CORR, MMC, payout, payout Sharpe, payout max drawdown.
Baselines with no state (B_ALL, B_COV, L_B_COV, CV_B) are left out.

    python3 experiments/c2_state_tables.py
Writes results/reports/crypto/c2_state_tables/{c2_state_tables.html,.csv}; print the PDF with
python3 -m src.analysis.html_pdf on the html.
"""
import os

import pandas as pd

R = "results/reports/crypto"
OUT = os.path.join(R, "c2_state_tables")
METRICS = [("corr", "CORR"), ("mmc", "MMC"), ("payout", "payout"), ("pay_sharpe", "sharpe.p"), ("pay_dd", "maxdd.p")]
TFS = ["RAW", "N_TS_MM", "N_GRP_MM", "N_POOL_MM"]


def long_table(path, study):
    t = pd.read_csv(path)
    if "kind" in t.columns:
        t = t[["kind", "arm"] + [m for m, _ in METRICS]]
    else:  # E189 wide layout
        rows = []
        for kind in ("dev", "test"):
            s = t[["arm"] + ["%s_%s" % (kind, m) for m, _ in METRICS]].copy()
            s.columns = ["arm"] + [m for m, _ in METRICS]
            s.insert(0, "kind", kind)
            rows.append(s)
        t = pd.concat(rows)
    t["study"] = study
    return t


def load():
    parts = [long_table(os.path.join(R, "e189_c2norm/table_arms.csv"), "E189"),
             long_table(os.path.join(R, "e190_c2norm_v2/table.csv"), "E190"),
             long_table(os.path.join(R, "e191_c2cross/table.csv"), "E191"),
             long_table(os.path.join(R, "e192_c2percoin/table.csv"), "E192"),
             long_table(os.path.join(R, "e201_c2percoin_alpha/table.csv"), "E201")]
    S = {}
    for p in parts:  # first study that produced an arm owns it (later ones reuse it by symlink)
        for r in p.itertuples(index=False):
            S.setdefault((r.arm, r.kind), (r.study, r))
    return S


def tf_name(t):
    return "untransformed" if t == "RAW" else t[2:]


def groups():
    e189 = ["RAW"] + ["N_%s_%s" % (a, b) for a in ("TS", "CS", "GRP", "POOL", "DIM")
                      for b in ("Z", "ROB", "MM", "RNK", "PCT", "BIN", "CZ")]
    own = [("pooled ridge, %s" % tf_name(t), t, "V_" + t if t in TFS else None) for t in e189]
    own += [("pooled LightGBM, untransformed", None, "L_OWN_RAW"),
            ("per-coin ridge, shared alpha", "C_OWN", "CV_OWN"),
            ("per-coin ridge, per-coin alpha", "A_OWN", "AV_OWN")]
    loo = [("pooled ridge, + size-group mean, %s" % tf_name(t), None, "G_" + t) for t in TFS]
    loo += [("pooled LightGBM, own - all-coin mean, untransformed", None, "L_REL_RAW"),
            ("pooled LightGBM, own - all-coin mean, TS_MM", None, "L_REL_N_TS_MM"),
            ("per-coin ridge, shared alpha", "C_LOO", "CV_LOO"),
            ("per-coin ridge, per-coin alpha", "A_LOO", "AV_LOO")]
    cat = [("pooled ridge, + 16 market PCs + PC products, %s" % tf_name(t), None, "M_" + t) for t in TFS]
    cat += [("per-coin ridge, shared alpha, flat concat", "C_CAT", "CV_CAT"),
            ("per-coin ridge, per-coin alpha, flat concat", "A_CAT", "AV_CAT")]
    return [("Own hidden state", own),
            ("Own hidden state + average of the other coins' states", loo),
            ("All coins' hidden states", cat)]


def build(S, rows, windows=("dev", "test")):
    cols = pd.MultiIndex.from_tuples([(side, lab) for side in ("hidden state", "hidden state + v2.0")
                                      for _, lab in METRICS])
    idx, data = [], []
    for kind in windows:
        for name, a, b in rows:
            vals, studies = [], []
            for arm in (a, b):
                hit = S.get((arm, kind)) if arm else None
                if hit:
                    studies.append(hit[0])
                vals += [getattr(hit[1], m) if hit else float("nan") for m, _ in METRICS]
            label = "%s [%s]" % (name, "/".join(dict.fromkeys(studies)))
            idx.append((kind, label))
            data.append(vals)
    return pd.DataFrame(data, index=pd.MultiIndex.from_tuples(idx, names=["window", "model"]), columns=cols)


CSS = ("body{font-family:Helvetica,Arial,sans-serif;font-size:9px;margin:18px;color:#111}"
       "h1{font-size:15px}h2{font-size:12px;margin-top:20px}table{border-collapse:collapse;margin-bottom:6px}"
       "th,td{border:1px solid #bbb;padding:2px 5px;text-align:right}th{background:#eee}"
       "tbody th{text-align:left;font-weight:normal}td.best{font-weight:bold;background:#e6f2e6}"
       "@page{size:A4 landscape;margin:10mm}")


DELTA = [("payout", "payout"), ("sharpe.p", "sharpe.p"), ("maxdd.p", "maxdd.p")]


def deltas(T):
    """(state + v2.0) - (state only) for every row that has both, per window and metric."""
    out = []
    for lab, _ in DELTA:
        d = (T[("hidden state + v2.0", lab)] - T[("hidden state", lab)]).dropna()
        out.append(d.rename(lab))
    return pd.concat(out, axis=1)


def plot_deltas(D, path, windows=("dev", "test"), suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    titles = [g for g, _ in groups()]
    short = ["own state", "own + others' average", "all coins"]
    fig, ax = plt.subplots(len(DELTA), len(titles), figsize=(11, 8.2), sharey="row")
    rng = np.random.default_rng(0)
    for i, (lab, _) in enumerate(DELTA):
        for j, g in enumerate(titles):
            a = ax[i, j]
            sub = D[D.group == g]
            data = [sub[sub.window == w][lab].values for w in windows]
            a.boxplot(data, widths=.45, showfliers=False, medianprops=dict(color="#333"))
            for k, (v, c) in enumerate(zip(data, ("#3b6ea8", "#c8702a"))):
                a.scatter(k + 1 + rng.uniform(-.12, .12, len(v)), v, s=22, color=c, zorder=3, alpha=.85)
            a.axhline(0, color="#888", lw=.8, ls="--")
            a.set_xticks(range(1, len(windows) + 1), ["%s (n=%d)" % (w, len(v)) for w, v in zip(windows, data)])
            if i == 0:
                a.set_title(short[j], fontsize=10)
            if j == 0:
                a.set_ylabel("delta " + lab + ("  (+ = worse)" if lab == "maxdd.p" else ""))
            a.grid(axis="y", alpha=.3)
    fig.suptitle("(hidden state + v2.0) - (hidden state only), one point per model row" + suffix, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


STUDY_DIRS = [("E189", "e189_c2norm"), ("E190", "e190_c2norm_v2"), ("E191", "e191_c2cross"),
              ("E192", "e192_c2percoin"), ("E201", "e201_c2percoin_alpha")]


def full_sample(S):
    """Add (arm, "full") rows: every metric recomputed from the per-period series over dev + test together.

    Payout, Sharpe and drawdown come from one payout series per arm, never from the window summaries.
    The dev rows are recomputed the same way first and checked against the stored tables."""
    from types import SimpleNamespace
    owner = {}
    for study, d in STUDY_DIRS:
        sc = pd.read_parquet(os.path.join(R, d, "scores.parquet"))
        for arm, g in sc.groupby("arm"):
            owner.setdefault(arm, (study, g))
    worst = 0.0
    for arm, (study, g) in owner.items():
        for kind, h in (("dev", g[g.kind == "dev"]), ("test", g[g.kind == "test"]), ("full", g)):
            h = h.sort_values("date")
            pay = h["payout"].dropna()
            cum = pay.cumsum()
            m = dict(corr=h["corr"].mean(), mmc=h.loc[pay.index, "mmc"].mean(), payout=pay.mean(),
                     pay_sharpe=pay.mean() / pay.std(ddof=0), pay_dd=float((cum.cummax() - cum).max()),
                     n_corr=int(h["corr"].notna().sum()), n_pay=len(pay))
            if kind == "full":
                S[(arm, "full")] = (study, SimpleNamespace(**m))
            elif (arm, kind) in S:
                ref = S[(arm, kind)][1]
                worst = max(worst, max(abs(m[k] - getattr(ref, k)) for k, _ in METRICS))
    print("recomputed dev/test vs stored tables, max abs diff %.2e" % worst)
    assert worst < 1e-9, worst
    n = S[("RAW", "full")][1]
    return n.n_corr, n.n_pay


def main():
    S = load()
    os.makedirs(OUT, exist_ok=True)
    body = ["<html><head><meta charset='utf-8'><title>Chronos-2 state tables</title><style>%s</style></head><body>" % CSS,
            "<h1>Chronos-2-Synth hidden-state models: every arm tried (E189, E190, E191, E192, E201)</h1>",
            "<p>Same panel (data/crypto/v2.0-asof-e189), same E131 dev blocks (1430 CORR dates / 468 payout dates) and "
            "test block 2026-02-20..2026-08-20 (130 / 120), same routed scorer. Payout = 0.1 CORR + MMC per period; "
            "sharpe.p = payout mean / std; maxdd.p = max drawdown of cumulative payout (positive, bigger is worse). "
            "Study ids in brackets. Blank = not run. Baselines with no hidden state are left out: B_ALL, B_COV "
            "(v2.0 ridge), L_B_COV (v2.0 LightGBM), CV_B (v2.0 per-coin ridge). For reference, B_COV dev payout 0.1004, "
            "test 0.0330.</p>"]
    allt = []
    for title, rows in groups():
        T = build(S, rows)
        allt.append(T.assign(group=title).set_index("group", append=True))
        body += ["<h2>%s</h2>" % title, T.to_html(float_format=lambda x: "%.4f" % x, na_rep="", border=0)]
    D = pd.concat([deltas(build(S, rows)).assign(group=t).reset_index() for t, rows in groups()])
    D.to_csv(os.path.join(OUT, "c2_state_deltas.csv"), index=False)
    png = os.path.join(OUT, "c2_state_deltas.png")
    plot_deltas(D, png)
    import base64
    summ = D.groupby(["group", "window"])[[l for l, _ in DELTA]].agg(["count", "mean", "median", "min", "max"])
    summ = summ.reindex([g for g, _ in groups()], level=0)
    body += ["<h2 style='page-break-before:always'>What v2.0 adds: deltas (hidden state + v2.0) - (hidden state only)</h2>",
             "<p>One delta per model row that was run both ways (pooled ridge on untransformed/TS_MM/GRP_MM/POOL_MM, "
             "per-coin ridge with shared and per-coin alpha). Positive payout and sharpe.p deltas mean v2.0 helps; "
             "a positive maxdd.p delta means a deeper drawdown (worse). Rows run only with v2.0 (E191, LightGBM) have "
             "no delta. The own + average and all-coins groups have only the two per-coin rows each, so their "
             "distributions are two points.</p>",
             "<img style='width:100%%' src='data:image/png;base64,%s'>" % base64.b64encode(open(png, "rb").read()).decode(),
             "<h3>Summary</h3>", summ.to_html(float_format=lambda x: "%.4f" % x, border=0),
             "<h3>Every delta</h3>",
             D.set_index(["group", "window", "model"]).to_html(float_format=lambda x: "%.4f" % x, border=0)]
    nc, npay = full_sample(S)
    bf = S[("B_COV", "full")][1]
    body += ["<h1 style='page-break-before:always'>Full sample: dev and test together</h1>",
             "<p>The same tables and deltas over the whole evaluated period, %d CORR dates and %d payout dates. Every metric "
             "is recomputed from each arm's per-period series over both windows, not averaged from the dev and test "
             "rows; the script recomputes the dev and test rows the same way and checks they match the stored tables "
             "to 1e-9. For reference, B_COV over the full sample: CORR %.4f, payout %.4f, sharpe.p %.4f, maxdd.p %.4f.</p>"
             % (nc, npay, bf.corr, bf.payout, bf.pay_sharpe, bf.pay_dd)]
    for title, rows in groups():
        T = build(S, rows, windows=("full",))
        allt.append(T.assign(group=title).set_index("group", append=True))
        body += ["<h2>%s (full sample)</h2>" % title,
                 T.droplevel("window").to_html(float_format=lambda x: "%.4f" % x, na_rep="", border=0)]
    DF = pd.concat([deltas(build(S, rows, windows=("full",))).assign(group=t).reset_index() for t, rows in groups()])
    DF.to_csv(os.path.join(OUT, "c2_state_deltas_full.csv"), index=False)
    pngf = os.path.join(OUT, "c2_state_deltas_full.png")
    plot_deltas(DF, pngf, windows=("full",), suffix=", full sample")
    summf = DF.groupby("group")[[l for l, _ in DELTA]].agg(["count", "mean", "median", "min", "max"])
    summf = summf.reindex([g for g, _ in groups()])
    body += ["<h2 style='page-break-before:always'>What v2.0 adds over the full sample</h2>",
             "<img style='width:100%%' src='data:image/png;base64,%s'>" % base64.b64encode(open(pngf, "rb").read()).decode(),
             "<h3>Summary (full sample)</h3>", summf.to_html(float_format=lambda x: "%.4f" % x, border=0),
             "<h3>Every delta (full sample)</h3>",
             DF.drop(columns="window").set_index(["group", "model"]).to_html(float_format=lambda x: "%.4f" % x, border=0)]
    body.append("<p>Notes. Pooled ridge rows of the own-state table are E189 (state only, 36 transforms) and "
                "E190 (+ v2.0, 4 transforms). The pooled + average row uses the leave-one-out mean of the coin's "
                "size group (E191 2'), not of all coins; the per-coin rows use all covered coins. The pooled "
                "all-coins rows summarise the cross-section with 16 market PCs (E191 3'), not the flat concatenation "
                "used per coin. E191 ran only with v2.0. LightGBM rows are one seed.</p></body></html>")
    pd.concat(allt).to_csv(os.path.join(OUT, "c2_state_tables.csv"))
    p = os.path.join(OUT, "c2_state_tables.html")
    open(p + ".tmp", "w").write("\n".join(body))
    os.replace(p + ".tmp", p)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_rows", 500):
        for title, rows in groups():
            print("==", title)
            print(build(S, rows).round(4).to_string())
        print(summ.round(4).to_string())
        for title, rows in groups():
            print("== full:", title)
            print(build(S, rows, windows=("full",)).round(4).to_string())
        print(summf.round(4).to_string())


if __name__ == "__main__":
    main()
