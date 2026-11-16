"""E147 report: group-only importance on crypto/v2.0-c2syn-c2ft. HTML, then python3 -m src.analysis.html_pdf <html>.

    python3 experiments/e147_report.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e133_report as W  # noqa: E402  CSS, table, tint, num
import e147_group_importance as X  # noqa: E402

os.chdir(REPO)
ROOT = X.ROOT
NAME = {"ridge": "ridge", "mlp_ktz": "MLP"}
LABEL = {"MOM": "momentum", "REV": "reversion", "VOLU": "volume", "VOLA": "volatility", "PRICE": "price level", "SIZE": "size",
         "C2COV": "c2cov (16 PCs)", "C2AR": "c2ar (16 PCs)", "C2FTCOV": "c2ftcov (16 PCs)", "C2FTAR": "c2ftar (16 PCs)",
         "C2COV_FULL": "c2cov-full (768)", "C2AR_FULL": "c2ar-full (768)", "C2FTCOV_FULL": "c2ftcov-full (768)", "C2FTAR_FULL": "c2ftar-full (768)",
         "ALL22": "all 22 shipped (reference)"}
NCOL = {"MOM": 4, "REV": 4, "VOLU": 4, "VOLA": 2, "PRICE": 4, "SIZE": 4, "ALL22": 22}
READING = json.load(open(os.path.join(ROOT, "reading.json"))) if os.path.exists(os.path.join(ROOT, "reading.json")) else {}

L = pd.read_csv(os.path.join(ROOT, "levels.csv"))
D = pd.read_csv(os.path.join(ROOT, "paired_deltas.csv"))
BB = pd.read_csv(os.path.join(ROOT, "by_block.csv"))
SD = pd.read_csv(os.path.join(ROOT, "by_seed.csv"))
PK = pd.read_csv(os.path.join(ROOT, "picks.csv"))
num = W.num


def lv(m, g, rs, sc):
    return L[(L.model == m) & (L.group == g) & (L.rows == rs) & (L.scope == sc)].iloc[0]


def dl(m, rs, a, b, metric, sc):
    r = D[(D.model == m) & (D.rows == rs) & (D.arm == a) & (D.vs == b) & (D.metric == metric) & (D.scope == sc)].iloc[0]
    return r


def order(rs="common"):
    """Groups by mean dev CORR across the two models, reference last."""
    s = {g: np.mean([lv(m, g, rs, "dev")["corr"] for m in X.MODELS]) for g in X.GROUPS[:-1]}
    return sorted(s, key=s.get, reverse=True) + ["ALL22"]


def ncols(g):
    return NCOL.get(g, 768 if g.endswith("_FULL") else 16)


def level_table(rs, sc):
    head = ["rank", "group", "cols", "model", "CORR", "t", "MMC", "c.sharpe", "payout", "p.sharpe", "dd(pay)", "calmar", "burn(d)", "n", "n(pay)"]
    rows = []
    for k, g in enumerate(order(rs)):
        for m in X.MODELS:
            r = lv(m, g, rs, sc)
            rows.append([str(k + 1) if g != "ALL22" else "ref", LABEL[g] if m == X.MODELS[0] else "", str(ncols(g)) if m == X.MODELS[0] else "", NAME[m],
                         num(r["corr"]), num(r.corr_t, 1, True), num(r.mmc, 4, True), num(r.c_sharpe, 2), num(r.payout, 4, True), num(r.p_sharpe, 2),
                         num(r.dd_pay, 2), num(r.calmar, 3, True), str(int(r.burn)), str(int(r.n)), str(int(r.n_pay))])
    return W.table(head, rows, "l2")


def dot_chart(rs):
    """CORR alone on dev, with bootstrap 95% CI: one row per group, ridge and MLP as two marks."""
    G = order(rs)
    rows = [(g, m, lv(m, g, rs, "dev")) for g in G for m in X.MODELS]
    lo = min(0, min(r.corr_lo for _, _, r in rows)) - 0.005
    hi = max(r.corr_hi for _, _, r in rows) + 0.005
    Wd, lab, rh, top = 760, 170, 22, 26
    H = top + rh * len(G) + 30
    x = lambda v: lab + (v - lo) / (hi - lo) * (Wd - lab - 20)
    col = {"ridge": "#2a78d6", "mlp_ktz": "#eb6834"}
    s = ['<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="CORR by group">' % (Wd, H)]
    for t in W.nice_ticks(lo, hi, 6):
        s.append('<line x1="%.1f" x2="%.1f" y1="%d" y2="%d" stroke="#ddd"/><text x="%.1f" y="%d" font-size="10" text-anchor="middle" fill="#666">%.2f</text>'
                 % (x(t), x(t), top - 6, H - 24, x(t), H - 10, t))
    s.append('<line x1="%.1f" x2="%.1f" y1="%d" y2="%d" stroke="#888"/>' % (x(0), x(0), top - 6, H - 24))
    for i, g in enumerate(G):
        y = top + rh * i + rh / 2
        s.append('<text x="%d" y="%.1f" font-size="11" text-anchor="end" fill="#333">%s</text>' % (lab - 8, y + 4, LABEL[g]))
        for j, m in enumerate(X.MODELS):
            r = lv(m, g, rs, "dev")
            yy = y + (j - 0.5) * 7
            s.append('<line x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f" stroke="%s" stroke-width="2"/>' % (x(r.corr_lo), x(r.corr_hi), yy, yy, col[m]))
            s.append('<circle cx="%.1f" cy="%.1f" r="4" fill="%s" stroke="#fff" stroke-width="1.5"><title>%s %s CORR %.4f [%.4f, %.4f]</title></circle>'
                     % (x(r["corr"]), yy, col[m], LABEL[g], NAME[m], r["corr"], r.corr_lo, r.corr_hi))
    s.append('<g font-size="11"><circle cx="%d" cy="12" r="4" fill="%s"/><text x="%d" y="16">ridge</text><circle cx="%d" cy="12" r="4" fill="%s"/><text x="%d" y="16">MLP</text></g>'
             % (lab, col["ridge"], lab + 8, lab + 70, col["mlp_ktz"], lab + 78))
    s.append("</svg>")
    return "".join(s)


def pair_table(rs, pairs):
    head = ["comparison", "model", "ΔCORR dev", "ΔCORR test", "Δpayout dev", "Δpayout test"]
    rows = []
    for a, b in pairs:
        for m in X.MODELS:
            c = [dl(m, rs, a, b, met, sc) for met in ("corr", "payout") for sc in ("dev", "test")]
            rows.append(["%s vs %s" % (LABEL[a], LABEL[b]) if m == X.MODELS[0] else "", NAME[m]] + [W.tint(r.delta, r.t) for r in c])
    return W.table(head, rows, "l2")


def block_table(rs):
    G = order(rs)
    blocks = sorted(BB.block.unique())
    rows = []
    for g in G:
        for m in X.MODELS:
            q = BB[(BB.model == m) & (BB.rows == rs) & (BB.group == g)].set_index("block")["corr"]
            rows.append([LABEL[g] if m == X.MODELS[0] else "", NAME[m]] + [num(q.get(b, np.nan), 3) if b in q else "–" for b in blocks])
    return W.table(["group", "model"] + [str(b) for b in blocks], rows, "l2")


def seed_table(rs):
    rows = []
    for g in order(rs):
        q = SD[(SD.model == "mlp_ktz") & (SD.group == g) & (SD.rows == rs)]
        a, b = q[q.scope == "dev"]["corr"], q[q.scope == "test"]["corr"]
        rows.append([LABEL[g], num(a.min()), num(a.max()), num(b.min()), num(b.max())])
    return W.table(["group", "dev CORR min seed", "dev max", "test min", "test max"], rows, "l2")


def pick_table():
    rows = []
    for g in X.GROUPS:
        r = PK[(PK.model == "ridge") & (PK.group == g)]
        p = PK[(PK.model == "mlp_ktz") & (PK.group == g)]
        rows.append([LABEL[g], " ".join(str(int(float(v))) for v in r["alpha"]), "%d of %d" % ((p["mlp__activation"] == "relu").sum(), len(p)),
                     " ".join("%g" % float(v) for v in p["mlp__alpha"])])
    return W.table(["group", "ridge alpha picked, blocks 0-11", "MLP relu picked", "MLP alpha picked, blocks 0-11"], rows, "l2")


def main():
    title = "How much does each feature group predict on its own? E147"
    sec = [("design", "Design"), ("rank", "Ranking"), ("all", "All keys"), ("test", "Final block"), ("pairs", "Contrasts"), ("ref", "Against all 22"),
           ("stab", "Stability"), ("picks", "Picked hyperparameters")]
    H = ["<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>E147 group importance</title><style>%s</style></head><body><main>" % W.CSS,
         "<nav>%s</nav>" % " · ".join('<a href="#%s">%s</a>' % s for s in sec),
         "<h1>%s</h1>" % title,
         "<p><code>crypto/v2.0-c2syn-c2ft</code>. A ridge and an MLP are each trained from scratch on ONE feature group at a time, 14 groups, plus the 22 shipped features as the reference. "
         "This measures what a group knows ALONE. It is not marginal value over v2.0, which E143, E144 and E146 measured (none found).</p>"]
    if READING:
        H += ['<div class="verdict"><b>Reading.</b> %s</div>' % READING["summary"], "<ul>%s</ul>" % "".join("<li>%s</li>" % x for x in READING["points"])]
    H += ['<h2 id="design">Design</h2><ul>',
          "<li><b>Groups.</b> momentum (momentum, sharpe_ratio; 4), reversion (RSI, bollinger; 4), volume (4), volatility (2), price level (close; 4), size (market cap; 4); "
          "c2cov / c2ar / c2ftcov / c2ftar as 16 PCs each (E143 zero-shot, E144 fine-tuned); the same four states as all 768 binned dims (E146). The six shipped groups partition the 22 features.</li>",
          "<li><b>From scratch.</b> Every (group, model, block) picks its own hyperparameters on the last 40 train dates after a 22-date purge, then refits on all 130 train dates. "
          "Ridge: alpha in {1, 10, 65.8, 316, 1000, 10000}. MLP (32, 16): activation {identity, relu} × alpha {0.01, 1} × learning rate {0.01, 0.05}, picked on seed 0 and refitted on 5 seeds. "
          "Alpha 100 is left out because it collapses the MLP to a constant (E137).</li>",
          "<li><b>Walk-forward.</b> 12 blocks of 130 predicted dates, 130 train, 22 purge; target_binned_return_20. Fine-tuned columns are each block&#39;s own (blocks.parquet). Missing values are 0.5.</li>",
          "<li><b>Two row sets.</b> <i>Common</i>: only coins whose fine-tuned state is observed that date, which implies every other group is observed too, so every group is scored on identical rows "
          "(blocks 2-11; 1,170 dev dates). This is the ranking table. <i>All</i>: every one of the 468,000 keys; a state group then also pays for its coverage gap (0.5 rows, and a constant prediction in blocks 0-1 for fine-tuned groups, scored CORR 0).</li>",
          "<li><b>Scores.</b> numerai_corr per date, proxy MMC against the lagged meta model (payout dates from May 2024 only), payout = 0.1×CORR + MMC per date. t is Newey-West at lag 20; intervals are block-bootstrap 95%. "
          "calmar is comparable only at equal n. Block 11 (2026-02-20 to 2026-08-20) is reported apart and selects nothing; it has been read by earlier studies, so it is not virgin.</li></ul>",
          '<h2 id="rank">Ranking: each group alone, common rows, dev</h2>',
          "<p>Groups ordered by mean dev CORR of the two models. Dots are mean CORR, bars the bootstrap 95% interval.</p>", dot_chart("common"),
          level_table("common", "dev"),
          '<h2 id="all">All keys, dev</h2><p>Same fits, scored on every key. The gap to the common table is what coverage costs a group.</p>', level_table("all", "dev"),
          '<h2 id="test">Final block (block 11), common rows</h2><p>Ordered by the dev ranking above; nothing is chosen here.</p>', level_table("common", "test"),
          '<h2 id="pairs">Contrasts within the Chronos-2 groups (common rows)</h2>',
          "<h3>768 dims against 16 PCs of the same state</h3>", pair_table("common", [(f.upper() + "_FULL", f.upper()) for f in X.FAMS]),
          "<h3>Fine-tuned against zero-shot</h3>", pair_table("common", [("C2FTCOV", "C2COV"), ("C2FTAR", "C2AR"), ("C2FTCOV_FULL", "C2COV_FULL"), ("C2FTAR_FULL", "C2AR_FULL")]),
          "<h3>With the 22 features as covariates against the label alone</h3>", pair_table("common", [("C2COV", "C2AR"), ("C2FTCOV", "C2FTAR"), ("C2COV_FULL", "C2AR_FULL"), ("C2FTCOV_FULL", "C2FTAR_FULL")]),
          '<h2 id="ref">Each group against all 22 shipped features (common rows)</h2><p>Negative means the group alone falls short of the full shipped set by that much.</p>',
          pair_table("common", [(g, "ALL22") for g in order("common")[:-1]]),
          '<h2 id="stab">Stability</h2><h3>Mean CORR by block, common rows (blocks 0-1 have no common rows)</h3>', block_table("common"),
          "<h3>MLP seed spread, common rows</h3>", seed_table("common"),
          '<h2 id="picks">Picked hyperparameters</h2><p>Block 0 of a zero-shot group and blocks 0-1 of a fine-tuned group have no state in the inner window, so every candidate scores nothing there and the first is kept.</p>', pick_table(),
          "</main></body></html>"]
    out = os.path.join(ROOT, "E147_group_importance.html")
    open(out, "w").write("\n".join(H))
    print(out, os.path.getsize(out))


if __name__ == "__main__":
    main()
