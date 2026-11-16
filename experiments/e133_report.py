#!/usr/bin/env python3
"""
E133 report: one self-contained HTML file from the tables e133_analysis.py and
e133_followup.py wrote. Reads dev rows only; nothing here is scored afresh except the
encoding and tie-mass descriptions, which read no target.

    python3 experiments/e133_report.py
"""
import html
import json
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
from src.data_mani.binning import BIN_LABELS  # noqa: E402
from src.data_mani.build_history_features import verify_shipped_encoding  # noqa: E402
from src.data_mani.history_features import history_columns  # noqa: E402

ROOT = "results/reports/crypto/e133_histfe"
OUT = os.path.join(ROOT, "E133_do_own_history_features_help_REJECT.html")
DATA = "data/crypto/v2.0-histfe/train.parquet"
TEST0 = pd.Timestamp("2026-02-20")
MODELS = ["ridge", "lgb_ktz", "mlp_ktz"]
SERIES = {"SHORT": 1, "LONG": 2, "COMBO": 3, "F_RANK": 4, "P_HIST": 5}
ARM_NAME = {"SHORT": "short (3-30d)", "LONG": "long (30-364d)", "COMBO": "short + long",
            "F_RANK": "own-history rank", "P_HIST": "tenure control"}


def num(x, d=4, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "&ndash;"
    return ("%+.*f" if sign else "%.*f") % (d, x)


def table(headers, rows, cls=""):
    head = "".join("<th>%s</th>" % h for h in headers)
    body = "".join("<tr>%s</tr>" % "".join(c if c.startswith("<td") else "<td>%s</td>" % c for c in r) for r in rows)
    return '<div class="scroll"><table class="%s"><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>' % (cls, head, body)


def tint(delta, t, d=4):
    """Diverging cell: tinted only when |t| >= 2, so colour never marks noise."""
    cls = "up" if (t >= 2) else "down" if (t <= -2) else ""
    return '<td class="%s">%s <small>t %s</small></td>' % (cls, num(delta, d, True), num(t, 1, True))


def nice_ticks(lo, hi, n=5):
    step = (hi - lo) / n
    mag = 10 ** np.floor(np.log10(step))
    step = min(m * mag for m in (1, 2, 5, 10) if m * mag >= step)
    return np.arange(np.ceil(lo / step), np.floor(hi / step) + 1) * step


def scale(lo, hi, a, b):
    return lambda v: a + (v - lo) * (b - a) / (hi - lo)


def line_panels(frames, keys, title_of, height=230):
    """Small multiples on one y axis. Series are written into data attributes for the crosshair."""
    W, pad = 300, dict(l=44, r=10, t=20, b=24)
    lo = min(f[keys].min().min() for f in frames.values())
    hi = max(f[keys].max().max() for f in frames.values())
    span = (hi - lo) * 0.06 or 1
    lo, hi = lo - span, hi + span
    panels = []
    for name, f in frames.items():
        x = scale(0, len(f) - 1, pad["l"], W - pad["r"])
        y = scale(lo, hi, height - pad["b"], pad["t"])
        ticks = nice_ticks(lo, hi)
        grid = "".join('<line class="grid" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/><text class="tick" x="%d" y="%.1f" text-anchor="end">%+.1f</text>'
                       % (pad["l"], W - pad["r"], y(t), y(t), pad["l"] - 4, y(t) + 3, t) for t in ticks if abs(t) > 1e-9)
        zero = '<line class="zero" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (pad["l"], W - pad["r"], y(0), y(0))
        years = pd.Series(range(len(f)), index=f.index).groupby(f.index.year).min()
        xt = "".join('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%d</text>' % (x(i), height - 8, yr)
                     for yr, i in years.items() if i < len(f) - 12 and (yr % 2 == 0 or len(years) < 4))
        lines = "".join('<polyline class="s%d" points="%s"/>' % (SERIES[k], " ".join("%.1f,%.1f" % (x(i), y(v)) for i, v in enumerate(f[k])))
                        for k in keys)
        data = json.dumps(dict(dates=[str(d)[:10] for d in f.index], names=[ARM_NAME[k] for k in keys],
                               slots=[SERIES[k] for k in keys], values=[[round(float(v), 3) for v in f[k]] for k in keys],
                               x0=pad["l"], x1=W - pad["r"]))
        panels.append('<figure class="panel" data-series=\'%s\'><figcaption>%s</figcaption><div class="plot"><svg viewBox="0 0 %d %d" role="img" '
                      'aria-label="%s">%s%s%s%s<line class="cross" y1="%d" y2="%d" x1="0" x2="0" visibility="hidden"/></svg><div class="tip" hidden></div></div></figure>'
                      % (data, html.escape(title_of(name)), W, height, html.escape(title_of(name)), grid, zero, xt, lines, pad["t"], height - pad["b"]))
    legend = "".join('<span><i class="k%d"></i>%s</span>' % (SERIES[k], ARM_NAME[k]) for k in keys)
    return '<div class="legend">%s</div><div class="panels">%s</div>' % (legend, "".join(panels))


def forest(D, metric, title):
    arms = ["SHORT", "LONG", "COMBO"]
    d = D[(D["metric"] == metric) & (D["vs"] == "B0") & D["arm"].isin(arms)]
    lo, hi = min(d["boot_lo"].min(), 0) * 1.1, max(d["boot_hi"].max(), 0.002) * 1.15
    W, rowh, left = 460, 20, 86
    H = 24 + (len(arms) * rowh + 12) * 3
    x = scale(lo, hi, left, W - 14)
    out = ['<line class="zero" x1="%.1f" x2="%.1f" y1="16" y2="%d"/>' % (x(0), x(0), H - 6)]
    for t in nice_ticks(lo, hi):
        out.append('<text class="tick" x="%.1f" y="11" text-anchor="middle">%+.2f</text>' % (x(t), t))
    y = 24
    for m in MODELS:
        out.append('<text class="rowlab" x="0" y="%d">%s</text>' % (y + rowh + 3, m))
        for a in arms:
            r = d[(d["model"] == m) & (d["arm"] == a)].iloc[0]
            yy = y + rowh / 2
            tip = "%s, %s: delta %+.4f, t %+.1f, 95%% CI %+.4f to %+.4f" % (m, ARM_NAME[a], r["delta"], r["t"], r["boot_lo"], r["boot_hi"])
            out.append('<g><title>%s</title><rect class="hit" x="%d" y="%d" width="%d" height="%d"/>'
                       '<line class="ci s%d" x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f"/><circle class="d%d" cx="%.1f" cy="%.1f" r="4.5"/></g>'
                       % (html.escape(tip), left, y, W - left, rowh, SERIES[a], x(r["boot_lo"]), x(r["boot_hi"]), yy, yy, SERIES[a], x(r["delta"]), yy))
            y += rowh
        y += 12
    legend = "".join('<span><i class="k%d"></i>%s</span>' % (SERIES[k], ARM_NAME[k]) for k in arms)
    return '<figure class="wide"><figcaption>%s</figcaption><div class="legend">%s</div><svg viewBox="0 0 %d %d" role="img" aria-label="%s">%s</svg></figure>' % (title, legend, W, H, title, "".join(out))


CSS = """
:root{color-scheme:light;--bg:#fcfcfb;--card:#f4f3ef;--ink:#0b0b0b;--ink2:#52514e;--line:#dddbd3;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;
--up:#d6ecdf;--down:#f7dcd8}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--bg:#1a1a19;--card:#242422;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--up:#1f3f30;--down:#4a2522}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--card:#242422;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--up:#1f3f30;--down:#4a2522}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:1.75rem;line-height:1.25;margin:0 0 .4em}h2{font-size:1.25rem;margin:2.2em 0 .5em;padding-top:.6em;border-top:1px solid var(--line)}h3{font-size:1rem;margin:1.5em 0 .3em}
p,li{max-width:74ch}.sub{color:var(--ink2)}small{color:var(--ink2)}
.verdict{background:var(--card);border-left:4px solid var(--s2);padding:14px 18px;border-radius:4px;margin:1em 0}
.scroll{overflow-x:auto;margin:.8em 0}table{border-collapse:collapse;font-size:.875rem;font-variant-numeric:tabular-nums;white-space:nowrap}
th{text-align:right;color:var(--ink2);font-weight:600;border-bottom:1px solid var(--line);padding:5px 10px}td{text-align:right;padding:4px 10px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}table.l2 th:nth-child(2),table.l2 td:nth-child(2){text-align:left}table.l3 th,table.l3 td{text-align:left;white-space:normal}td.up{background:var(--up)}td.down{background:var(--down)}
code{font:.875em ui-monospace,Menlo,monospace}nav{font-size:.875rem;color:var(--ink2)}nav a{color:inherit;margin-right:12px;white-space:nowrap}
figure{margin:1em 0}figcaption{font-size:.875rem;font-weight:600;margin-bottom:4px}figure.wide{max-width:560px}svg{width:100%;height:auto;display:block}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}.plot{position:relative}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:.875rem;color:var(--ink2);margin:.6em 0}.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.grid{stroke:var(--line);stroke-width:.6}.zero{stroke:var(--ink2);stroke-width:1}.tick{fill:var(--ink2);font-size:9px}.rowlab{fill:var(--ink);font-size:11px;font-weight:600}
polyline{fill:none;stroke-width:1.6;stroke-linejoin:round}.ci{stroke-width:2;stroke-linecap:round}.hit{fill:transparent}.cross{stroke:var(--ink2);stroke-width:.7;stroke-dasharray:3 2}
.s1{stroke:var(--s1)}.s2{stroke:var(--s2)}.s3{stroke:var(--s3)}.s4{stroke:var(--s4)}.s5{stroke:var(--s5)}
.d1{fill:var(--s1)}.d2{fill:var(--s2)}.d3{fill:var(--s3)}.d4{fill:var(--s4)}.d5{fill:var(--s5)}circle{stroke:var(--bg);stroke-width:2}
.k1{background:var(--s1)}.k2{background:var(--s2)}.k3{background:var(--s3)}.k4{background:var(--s4)}.k5{background:var(--s5)}
.tip{position:absolute;top:18px;background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:5px 8px;font-size:.75rem;pointer-events:none;white-space:nowrap;z-index:2}
"""

JS = """
document.querySelectorAll('.panel').forEach(function(p){
  var d=JSON.parse(p.dataset.series),svg=p.querySelector('svg'),c=p.querySelector('.cross'),tip=p.querySelector('.tip');
  function move(e){
    var r=svg.getBoundingClientRect(),sx=(e.clientX-r.left)/r.width*svg.viewBox.baseVal.width;
    var i=Math.round((sx-d.x0)/(d.x1-d.x0)*(d.dates.length-1));i=Math.max(0,Math.min(d.dates.length-1,i));
    var x=d.x0+i/(d.dates.length-1)*(d.x1-d.x0);c.setAttribute('x1',x);c.setAttribute('x2',x);c.setAttribute('visibility','visible');
    tip.hidden=false;tip.textContent='';var h=document.createElement('b');h.textContent=d.dates[i];tip.appendChild(h);
    d.names.forEach(function(n,k){var s=document.createElement('div');s.textContent=n+': '+(d.values[k][i]>=0?'+':'')+d.values[k][i].toFixed(2);tip.appendChild(s)});
    var px=x/svg.viewBox.baseVal.width*r.width;tip.style.left=(px>r.width/2?px-tip.offsetWidth-8:px+8)+'px';
  }
  svg.addEventListener('pointermove',move);
  svg.addEventListener('pointerleave',function(){c.setAttribute('visibility','hidden');tip.hidden=true});
});
"""


def main():
    L = pd.read_csv(os.path.join(ROOT, "summary_levels.csv"))
    D = pd.read_csv(os.path.join(ROOT, "paired_deltas.csv"))
    F = pd.read_csv(os.path.join(ROOT, "followup_pairs.csv"))
    PW = pd.read_csv(os.path.join(ROOT, "levels_payout_window.csv")).set_index(["model", "arm"])
    BC = pd.read_csv(os.path.join(ROOT, "blend_summary.csv")).set_index("model")
    BP = pd.read_csv(os.path.join(ROOT, "blend_payout.csv")).set_index("model")
    T = pd.read_csv(os.path.join(ROOT, "tuned_corr_by_block.csv"))
    SEED = pd.read_csv(os.path.join(ROOT, "lgb_frank_by_seed.csv"))
    BLOCK = pd.read_csv(os.path.join(ROOT, "lgb_frank_by_block.csv"))
    ST = pd.read_csv(os.path.join(ROOT, "stability.csv"))
    PER = pd.read_csv(os.path.join(ROOT, "feature_summary.csv")).set_index("feature")
    RED = json.load(open(os.path.join(ROOT, "redundancy.json")))
    COV = pd.read_csv(os.path.join(ROOT, "coverage_by_year.csv"), index_col=0)
    PICKS = json.load(open(os.path.join(ROOT, "tuned_picks.json")))
    S = pd.read_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"))
    assert S["date"].max() < TEST0

    def lv(model, arm, col):
        return L[(L["model"] == model) & (L["arm"] == arm)].iloc[0][col]

    def d(model, metric, arm, vs="B0"):
        return D[(D["model"] == model) & (D["metric"] == metric) & (D["arm"] == arm) & (D["vs"] == vs)].iloc[0]

    def f(model, metric, arm, vs):
        return F[(F["model"] == model) & (F["metric"] == metric) & (F["arm"] == arm) & (F["vs"] == vs)].iloc[0]

    # descriptions of the dataset: no target column is read
    df = pd.read_parquet(DATA, columns=None).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    base = [c for c in df if c.startswith("feature_") and not c.startswith("feature_hfe_")]
    hits, total = verify_shipped_encoding(df, base, "date")
    dev = df[df["date"] < TEST0]
    fam = history_columns(base)
    masses = []
    for k, cols in fam.items():
        v = dev[cols].to_numpy().ravel()
        v = v[~np.isnan(v)]
        masses.append([k, str(len(cols))] + ["%.3f" % (v == b).mean() for b in BIN_LABELS] + ["%.1f%%" % (100 * dev[cols].notna().to_numpy().mean())])
    feat_nan = df[base].isna().mean()
    nan_list = ", ".join("<code>%s</code> %.2f%%" % (c.replace("feature_", ""), 100 * v) for c, v in feat_nan[feat_nan > 0].items())
    test_rows = int((df["date"] >= TEST0).sum())

    def window(x):
        m = re.search(r"_(rank|tsmax)(\d+)$", x)
        if m:
            return m[1], m[2]
        m = re.search(r"_ma(\d+)_(\d+)$", x)
        return ("ma", m[1] + "/" + m[2]) if m else ("tschg", "&ndash;")
    PER[["fam", "win"]] = [window(x) for x in PER.index]
    by = PER.groupby(["fam", "win"], sort=False)
    byw = by[["present", "mid_bin", "ic", "max_abs_corr_base", "max_abs_corr_block"]].mean()
    byw["sig"] = by["t"].apply(lambda t: int((t.abs() >= 3.8).sum()))
    wrows = [[a, b, "%.1f%%" % (100 * r["present"]), "%.1f%%" % (100 * r["mid_bin"]), num(r["ic"], 3, True), str(int(r["sig"])),
              num(r["max_abs_corr_base"], 2), num(r["max_abs_corr_block"], 2)] for (a, b), r in byw.iterrows()]

    # headline rows
    head = []
    for m in MODELS:
        row = [m]
        for a in ("SHORT", "LONG", "COMBO"):
            r, p = d(m, "corr", a), d(m, "payout", a)
            row += [tint(r["delta"], r["t"]), tint(p["delta"], p["t"])]
        head.append(row)
    verdict_rows = [
        ["ridge", "<b>REJECT</b>", "Harmful on CORR at every horizon, in 10 of 11 blocks for COMBO. Retuning does not rescue it."],
        ["lgb_ktz", "<b>NO IMPROVEMENT</b>", "CORR falls on the long horizons; payout gains are small and none beats the tenure control."],
        ["mlp_ktz", "<b>REJECT</b>", "The largest harm of the three, in every block and seed. Retuning collapses."],
    ]
    r_block = ST[(ST["cut"] == "block") & (ST["arm"] == "COMBO")].groupby("model")["delta"].apply(lambda x: "%d of %d" % ((x < 0).sum(), len(x)))
    verdict_rows[0][2] = "Harmful on CORR at every horizon; COMBO is below B0 in %s blocks. Retuning does not rescue it." % r_block["ridge"]
    verdict_rows[2][2] = "The largest harm of the three; COMBO is below B0 in %s blocks. The retune collapses." % r_block["mlp_ktz"]

    lev_rows = []
    order = ["B0", "SHORT", "LONG", "COMBO", "TSC", "FULL", "ADDED_ONLY", "F_RANK", "F_MA", "F_TSMAX", "P_HIST", "tuneB0", "tuneFULL"]
    for m in MODELS:
        g = L[L["model"] == m].set_index("arm")
        for a in order:
            r = g.loc[a]
            lev_rows.append([m, a, num(r["CORR"]), num(r["MMC"]), num(r["c_sharpe"], 2), num(r["dd_corr"], 2), num(r["payout"]), num(r["p_sharpe"], 2),
                             num(r["dd_pay"], 2), num(r["calmar"], 3), str(int(r["burn"])), str(int(r["n"])), str(int(r["n_pay"])), num(r["predcorr_b0"], 2), str(int(r["seeds"]))])

    delta_rows = []
    labels = D.drop_duplicates(["arm", "vs"])[["arm", "vs", "label"]]
    for _, x in labels.iterrows():
        row = ["%s vs %s" % (x["arm"], x["vs"]), x["label"]]
        for m in MODELS:
            for mt in ("corr", "payout"):
                r = D[(D["model"] == m) & (D["metric"] == mt) & (D["arm"] == x["arm"]) & (D["vs"] == x["vs"])].iloc[0]
                row.append(tint(r["delta"], r["t"]))
        delta_rows.append(row)

    win = []
    for m in MODELS:
        win.append([m, num(PW.loc[(m, "B0"), "corr"]), num(PW.loc[(m, "F_RANK"), "corr"]), num(PW.loc[(m, "P_HIST"), "corr"]), num(PW.loc[(m, "COMBO"), "corr"]),
                    num(PW.loc[(m, "B0"), "payout"]), num(PW.loc[(m, "F_RANK"), "payout"]), num(PW.loc[(m, "P_HIST"), "payout"]), num(PW.loc[(m, "COMBO"), "payout"])])

    ten = []
    for m in MODELS:
        ten.append([m, tint(d(m, "corr", "P_HIST")["delta"], d(m, "corr", "P_HIST")["t"]), tint(d(m, "payout", "P_HIST")["delta"], d(m, "payout", "P_HIST")["t"]),
                    tint(f(m, "payout", "F_RANK", "P_HIST")["delta"], f(m, "payout", "F_RANK", "P_HIST")["t"]),
                    tint(f(m, "payout", "SHORT", "P_HIST")["delta"], f(m, "payout", "SHORT", "P_HIST")["t"]),
                    tint(f(m, "payout", "FULL", "P_HIST")["delta"], f(m, "payout", "FULL", "P_HIST")["t"])])

    tune_rows = []
    for m in MODELS:
        tc, tp = d(m, "corr", "tuneFULL", "tuneB0"), d(m, "payout", "tuneFULL", "tuneB0")
        g = T[T["model"] == m]
        tune_rows.append([m, num(g["B0"].mean()), num(g["FULL"].mean()), tint(tc["delta"], tc["t"]), tint(tp["delta"], tp["t"]), "%d of 11" % (g["FULL"] < g["B0"]).sum()])
    mlp = T[T["model"] == "mlp_ktz"].set_index("block")
    alpha = [float(x["mlp__alpha"]) for x in PICKS["mlp_ktz FULL"]]
    hi_alpha = [i for i, a in enumerate(alpha) if a == 100.0]
    lo_alpha = [i for i, a in enumerate(alpha) if a != 100.0]
    alpha_b0 = [float(x["mlp__alpha"]) for x in PICKS["mlp_ktz B0"]]
    hi_b0 = [i for i, a in enumerate(alpha_b0) if a == 100.0]

    blend_rows = []
    for m in MODELS:
        blend_rows.append([m, "CORR", "%.2f" % BC.loc[m, "w_full"], num(BC.loc[m, "w0.00"]), num(BC.loc[m, [c for c in BC if c.startswith("w0.")]].max()), num(BC.loc[m, "delta_cv"], 4, True), num(BC.loc[m, "t"], 1, True), str(int(BC.loc[m, "n"]))])
        blend_rows.append([m, "payout", "%.2f" % BP.loc[m, "w_full"], num(BP.loc[m, "w0.00"]), num(BP.loc[m, [c for c in BP if c.startswith("w0.")]].max()), num(BP.loc[m, "delta_cv"], 4, True), num(BP.loc[m, "t"], 1, True), str(int(BP.loc[m, "n"]))])

    stab = []
    bb = ST[ST["cut"] == "block"].pivot_table(index=["model", "arm"], columns="key", values="delta")
    for m in MODELS:
        for a in ("SHORT", "LONG", "COMBO", "FULL"):
            stab.append([m, a] + [num(v, 3, True) for v in bb.loc[(m, a)]])
    seed_rows = []
    ss = ST[ST["cut"] == "seed"].pivot_table(index=["model", "arm"], columns="key", values="delta")
    for m in ("lgb_ktz", "mlp_ktz"):
        for a in ("SHORT", "LONG", "COMBO", "FULL"):
            seed_rows.append([m, a] + [num(v, 4, True) for v in ss.loc[(m, a)]])

    cov_keys = ["rank7", "rank14", "rank30", "rank90", "rank364", "ma3_7", "ma7_14", "ma14_30", "ma30_90", "ma91_364", "tschg"]
    cov_rows = [[k] + ["%.0f%%" % (100 * v) for v in COV.loc[k]] for k in cov_keys]

    # figures: fixed arms, seed-averaged, every 5th date
    fixed = S[S["tune"] == ""]
    frames = {}
    for m in MODELS:
        p = fixed[fixed["model"] == m].pivot_table(index="date", columns="arm", values="corr")
        frames[m] = pd.DataFrame({a: (p[a] - p["B0"]).cumsum() for a in ("SHORT", "LONG", "COMBO")}).iloc[::5]
    pay = {}
    for m in MODELS:
        p = fixed[fixed["model"] == m].pivot_table(index="date", columns="arm", values="payout").dropna()
        pay[m] = pd.DataFrame({a: (p[a] - p["B0"]).cumsum() for a in ("F_RANK", "P_HIST")}).iloc[::2]

    s = lambda a: d("lgb_ktz", "payout", a)   # noqa: E731
    fr, frt = s("F_RANK"), f("lgb_ktz", "payout", "F_RANK", "P_HIST")
    lc, sc = d("lgb_ktz", "corr", "LONG"), d("lgb_ktz", "corr", "SHORT")
    nn = len(cov_keys)
    body = f"""
<h1>Do own-history features help on Numerai Crypto? REJECT</h1>
<p class="sub">E133 &middot; <code>crypto/v2.0-histfe</code> against the 22 shipped features &middot; 11 chronological dev blocks, {S['date'].min():%Y-%m-%d} to {S['date'].max():%Y-%m-%d} &middot; target <code>target_binned_return_20</code></p>
<nav><a href="#answer">Answer</a><a href="#encoding">Encoding</a><a href="#ties">Ties</a><a href="#coverage">Coverage</a><a href="#counts">Counts</a><a href="#design">Design</a><a href="#levels">Levels</a><a href="#deltas">Deltas</a><a href="#tenure">Tenure</a><a href="#retune">Retune</a><a href="#blends">Blends</a><a href="#stability">Stability</a><a href="#redundancy">Redundancy</a><a href="#gaps">Gaps</a></nav>

<div class="verdict"><b>Verdict: do not add this block.</b> Short horizons, long horizons and both together all lower CORR for ridge and the MLP, by {-d('ridge','corr','SHORT')['delta']:.3f} to {-d('mlp_ktz','corr','COMBO')['delta']:.3f}, and lower payout too. For LightGBM, short horizons are neutral on CORR and long horizons hurt it. The one strong-looking result, LightGBM with own-history ranks at payout {fr['delta']:+.4f} (t {fr['t']:+.1f}), is no better than adding a single tenure column ({frt['delta']:+.4f}, t {frt['t']:+.1f}). The final test period was not read, so no positive result here could rise above "promising but unconfirmed", and none reaches that.</div>

<h2 id="answer">Answer by model and horizon</h2>
{table(['model','verdict','why'], verdict_rows, 'l3')}
<p>Change against the 22-feature baseline, paired by date. Shaded cells have |t| &ge; 2 on Newey-West at lag 20. Red is worse, green is better. CORR covers 1,430 dates; payout covers the 468 dates that have a meta model.</p>
{table(['model','SHORT CORR','SHORT payout','LONG CORR','LONG payout','COMBO CORR','COMBO payout'], head)}
{forest(D,'corr','CORR change against baseline, with 95% block-bootstrap interval')}
<h3>Cumulative CORR change against baseline</h3>
{line_panels(frames,['SHORT','LONG','COMBO'],str)}
<p>The two horizons do not combine into anything. For LightGBM, COMBO is about as low as LONG alone (CORR {d('lgb_ktz','corr','COMBO')['delta']:+.4f} against {lc['delta']:+.4f}), while SHORT alone is {sc['delta']:+.4f}. For ridge and the MLP, COMBO is worse than either horizon alone.</p>

<h2 id="encoding">Detected encoding</h2>
<ul>
<li><b>Columns.</b> {len(base)} shipped <code>feature_*</code> columns. Identifiers (<code>symbol</code>, <code>ucid</code>, <code>date</code>), targets and predictions are excluded.</li>
<li><b>Frequency.</b> Weekdays only: {df['date'].nunique():,} dates, {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}, 241 to 300 tokens per date.</li>
<li><b>Values.</b> Five levels {{0, .25, .5, .75, 1}}, cross-sectional per date. This is not Unity's 0&ndash;4 encoding.</li>
<li><b>Missing values.</b> NaN, in {nan_list}. The training loader fills NaN with the neutral 0.5.</li>
<li><b>Identifier.</b> <code>ucid</code> has no nulls and no duplicate (ucid, date), so it is the token key. 31 symbols map to several coins, so <code>symbol</code> would have spliced different coins together.</li>
<li><b>Exact bin rule, verified from the data.</b> Position is (rank&minus;1)/(n&minus;1), with inclusive cuts at .05, .25, .75 and .95. It reproduces <b>{hits:,} of {total:,}</b> shipped date-feature cross-sections. No approximation is needed.</li>
<li><b>Repo mismatch.</b> <code>binning.rank_to_bins</code> uses a different rule and matches only 91.13% of cross-sections, being exact only at n=300. E133 uses the verified rule in its own function. <code>binning.py</code> is unchanged, because every existing dataset depends on it.</li>
</ul>

<h2 id="ties">Tie handling</h2>
<ul>
<li><b>Own-history rank:</b> average rank of today within the window, (below + (equal + 1) / 2) / count.</li>
<li><b>Time since maximum:</b> the most recent occurrence of the maximum.</li>
<li><b>Cross-sectional bins:</b> ties take their average rank, so a tie group always shares one bin. A lone value on a date takes 0.5.</li>
</ul>
<p>Inputs have only five levels, so ties are heavy and the bins cannot hold Numerai's 5/20/50/20/5 mass. Dev rows below; {(PER['mid_bin']>.9).sum()} of 352 columns have more than 90% of their values in the middle bin, and {(PER['mid_bin']>.8).sum()} have more than 80%.</p>
{table(['family','columns','bin 0','bin .25','bin .5','bin .75','bin 1','present'], masses)}

<h2 id="coverage">Coverage requirements</h2>
<p>A window counts only if the token has a value on at least 60% of the weekdays in the full calendar window, otherwise the feature stays NaN. A window cut short by the start of the panel does not pass on the few values it has. Nothing is backfilled. <code>tschg</code> is NaN until the token's first observed change, because the duration before that is unknown.</p>
{table(['window (days)','weekdays','needed'], [['3','1-3','1-2'],['7','5','3'],['14','10','6'],['30','21-22','13-14'],['90','64-65','39'],['91','65','39'],['364','260','156']])}
<p>The panel has weekdays only, so a 7-day window holds 5 dates and a 3-day window holds 1 to 3 (just 1 on a Monday). Share of rows with a value, by year of the dev period:</p>
{table(['feature group'] + list(COV.columns), cov_rows)}
<p><code>tsmax</code> has the same coverage as <code>rank</code>. The 2026 column stops at 2026-02-19. Tokens that later left the universe keep all their rows.</p>

<h2 id="counts">Feature counts</h2>
{table(['family','formula','columns'], [['own-history rank','22 &times; 5 windows','110'],['MA difference','22 &times; 5 pairs','110'],['time since max','22 &times; 5 windows','110'],['time since change','22 &times; 1','22'],['<b>new</b>','','<b>352</b>'],['original','','22'],['<b>total</b>','','<b>374</b>']])}
<p>Short arm: 220 columns with the baseline (rank and tsmax at 7, 14, 30; MA 3/7, 7/14, 14/30). Long arm: 154 (rank and tsmax at 90, 364; MA 30/90, 91/364). <code>tschg</code> has no horizon, so it is tested separately as TSC.</p>

<h2 id="design">Design and what deviated</h2>
<ul>
<li><b>Validation.</b> Each block trains on 130 dates, purges 22 and predicts the next 130. The target spans 20 days, so 22 clears it.</li>
<li><b>Models.</b> Ridge, <code>lgb_ktz</code> and <code>mlp_ktz</code> at E131's fixed configurations, searched on raw v2.0 and not on these features. <code>mlp_ktz</code> has identity activation, so it is a linear model. LightGBM and the MLP ran 5 seeds; ridge is deterministic.</li>
<li><b>Final test.</b> All {test_rows:,} rows dated 2026-02-20 or later were dropped at load and never fitted, scored or used to pick anything. Earlier studies scored those dates on other features, so that period is no longer virgin for the project, though it is for this block.</li>
<li><b>Point-in-time.</b> Every window ends at the row's date. The truncation test confirms that deleting later rows changes no earlier value. Only the 22 supplied columns are used, so no external feed or coverage snapshot can leak survivorship.</li>
<li><b>Live.</b> Feature set <code>histfe</code> in <code>src/live/features.py</code> calls the same function on train + live history + today's rows, and the parity test passes. It needs the ucid filled from the symbol's latest value, because live history has none. Train plus live history covers 22 of 23 weekdays after the train file ends. No slot uses it.</li>
<li><b>Deviations.</b> None in the features. The MMC is a proxy against the historical meta model and exists only from 2024-05-07.</li>
</ul>

<h2 id="levels">Levels</h2>
<p>Seed-averaged per date. <code>calmar</code> and <code>burn</code> use the {int(L['n_pay'].iloc[0])} payout dates, so they compare within this table only. <code>tune</code> arms are one seed.</p>
{table(['model','arm','CORR','MMC proxy','c.sharpe','dd(corr)','payout','p.sharpe','dd(pay)','calmar','burn(d)','n','n(pay)','corr w/ B0','seeds'], lev_rows, 'l2')}
<p><b>Drawdown moves the other way.</b> Ridge and MLP arms mostly have a smaller <code>dd(pay)</code> than the baseline, so calmar rises even as mean payout falls: ridge FULL has calmar {lv('ridge','FULL','calmar'):.3f} against {lv('ridge','B0','calmar'):.3f}, on payout {lv('ridge','FULL','payout'):.4f} against {lv('ridge','B0','payout'):.4f}. LightGBM F_RANK likewise has a smaller drawdown and higher calmar than tenure ({lv('lgb_ktz','F_RANK','calmar'):.3f} against {lv('lgb_ktz','P_HIST','calmar'):.3f}). A max drawdown is one episode in 468 overlapping dates, so this is a weak basis, and the verdict rests on the paired means. It is the one respect in which the block could still be worth a look.</p>
<h3>CORR on the payout window</h3>
<p>The headline CORR covers 2020 to 2026. Inside the 468 payout dates, LightGBM's picture changes: own-history rank and tenure both sit above the baseline on CORR too, so LightGBM's CORR loss comes from before May 2024.</p>
{table(['model','B0 CORR','F_RANK','P_HIST','COMBO','B0 payout','F_RANK','P_HIST','COMBO'], win)}

<h2 id="deltas">Paired deltas</h2>
<p>Each cell is the mean per-date difference with its Newey-West t. This table holds 15 comparisons &times; 3 models &times; 3 metrics, so about 135 tests with MMC included; a Bonferroni bar sits near |t| 3.6. Read isolated cells near 2 as noise.</p>
{table(['comparison','what it asks','ridge CORR','ridge payout','lgb CORR','lgb payout','mlp CORR','mlp payout'], delta_rows, 'l2')}

<h2 id="tenure">Tenure control</h2>
<p>P_HIST adds one column, the token's count of earlier panel dates, rank-binned. It lifts every model, and on mean CORR and mean payout it matches or beats every arm of the 352-column block. The block gives no evidence of value beyond tenure.</p>
{table(['model','tenure vs B0 CORR','tenure vs B0 payout','F_RANK vs tenure payout','SHORT vs tenure payout','FULL vs tenure payout'], ten)}
<h3>Cumulative payout change against baseline, May 2024 on</h3>
{line_panels(pay,['F_RANK','P_HIST'],str)}
<p>LightGBM F_RANK against the baseline is positive in all 5 seeds ({SEED['F_RANK-B0'].min():+.4f} to {SEED['F_RANK-B0'].max():+.4f}) and all 4 payout blocks ({BLOCK['F_RANK-B0'].min():+.4f} to {BLOCK['F_RANK-B0'].max():+.4f}). Against tenure it splits 3 to 2 across seeds and 2 to 2 across blocks. Equal gains do not prove both are the same signal. F_RANK + tenure against tenure was not run, so that remains open. The t of {fr['t']:+.1f} also falls short of the multiplicity bar.</p>
<p>Tenure is a lead rather than an adoption. It also came up in E131, and it has not been through a test block in either study.</p>

<h2 id="retune">Equal-budget retune</h2>
<p>Same 6 candidates on B0 and FULL, picked per block on the last 40 train dates after a 22-date purge. One seed each. This rules out "the fixed config suited the baseline" for ridge and LightGBM on CORR, though LightGBM's payout change stays positive and not significant.</p>
{table(['model','tuned B0 CORR','tuned FULL CORR','CORR change','payout change','blocks FULL below B0'], tune_rows)}
<p>The MLP retune's size is not a fair estimate of the harm. FULL picked <code>alpha=100</code> on blocks {', '.join(map(str, hi_alpha))}, and mean CORR on those blocks was {mlp.loc[hi_alpha,'FULL'].mean():+.3f}, against {mlp.loc[lo_alpha,'FULL'].mean():+.3f} on the rest. The baseline picked the same alpha on blocks {', '.join(map(str, hi_b0))} without collapsing ({mlp.loc[hi_b0,'B0'].mean():+.3f}). The fixed-config deficit, {d('mlp_ktz','corr','FULL')['delta']:+.4f} over 5 seeds, is the better number. The training panel was not run, so the mechanism is unexamined.</p>

<h2 id="blends">Blends</h2>
<p>Rank blend of B0 and ADDED_ONLY, with the weight picked on the other blocks and scored on the held-out block. No model gains. LightGBM's best in-sample payout weight does not hold up out of sample (t {BP.loc['lgb_ktz','t']:+.1f}).</p>
{table(['model','metric','best weight, all blocks','B0','best in-sample','held-out change','t','n'], blend_rows)}

<h2 id="stability">Stability</h2>
<h3>CORR change against B0 by block</h3>
{table(['model','arm'] + [str(i) for i in range(11)], stab, 'l2')}
<h3>CORR change against B0 by seed</h3>
{table(['model','arm'] + [str(i) for i in range(5)], seed_rows, 'l2')}

<h2 id="redundancy">Redundancy and single-feature signal</h2>
<p>Dev rows only. Correlations are per-date Spearman averaged over {RED['dates']} sampled dates. IC is the mean per-date rank correlation with the target. "Sig" counts columns above a Bonferroni |t| of 3.8.</p>
{table(['family','window','present','mid bin','mean IC','sig','max |corr| with base','max |corr| in block'], wrows, 'l2')}
<ul>
<li>{RED['pairs_gt_95']} pairs in the block exceed |corr| .95 and {RED['pairs_gt_80']} exceed .80. {RED['comps_for_90']} principal components reach 90% of variance and {RED['comps_for_99']} reach 99%, against 352 columns, so the block is correlated but far from a handful of repeated signals.</li>
<li>Only {int((PER['max_abs_corr_base']>.8).sum())} column exceeds |corr| .80 with a base feature, so the block is not a copy of the baseline.</li>
<li>{RED['sig_t2']} columns pass |t| &ge; 2 and {RED['bonf']} pass the Bonferroni bar on their own, mostly rank (negative, a reversal sign) and <code>tschg</code> (positive). This signal does not survive into models: the TSC arm is flat or harmful ({d('ridge','corr','TSC')['delta']:+.4f}, {d('lgb_ktz','corr','TSC')['delta']:+.4f}, {d('mlp_ktz','corr','TSC')['delta']:+.4f}). A single-feature IC is no evidence of incremental value when the 22 base features already sit beside it.</li>
</ul>

<h2 id="gaps">Gaps and limits</h2>
<ul>
<li>No arm or block is missing: all 127 prediction files were scored, on 1,430 dates.</li>
<li>Two LightGBM jobs (the last 3 SHORT seeds and the FULL retune) stopped at 08:29 with no traceback and were relaunched at 13:24 to produce only the missing files. The cause is unknown; quota was ruled out.</li>
<li>Payout and MMC cover 468 dates in 4 blocks, against 1,430 dates and 11 blocks for CORR.</li>
<li>Filling NaN with 0.5 at training time merges "no history" with "middle of the field". An indicator was not tested, because E131 showed how indicators can go wrong here.</li>
<li>Only the 130-date window was tested. With longer training windows the harm might shrink.</li>
<li>Catboost and the random forest were not run. E131's three model classes were used.</li>
</ul>
"""
    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           '<title>E133 Own-History Features</title><style>%s</style></head><body><main>%s</main><script>%s</script></body></html>' % (CSS, body, JS))
    with open(OUT, "w") as h:
        h.write(doc)
    print(OUT, "%.0f KB" % (os.path.getsize(OUT) / 1e3), "rows in coverage", nn)


if __name__ == "__main__":
    main()
