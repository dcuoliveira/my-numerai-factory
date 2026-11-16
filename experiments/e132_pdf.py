#!/usr/bin/env python3
"""E132 PDFs: one self-contained report per feature set, from e132_analysis.py's tables.

    python3 experiments/e132_blend_series.py     # once, for the blend curves
    python3 experiments/e132_pdf.py [L S C]

Every number comes from the CSVs and scores next to the HTML; nothing is rescored here, except
the indicator audit, which e132_analysis.py ran on the litmom columns only.
"""
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.pdfgen import canvas as pdfcanvas  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

ROOT = pathlib.Path("results/reports/crypto/e132_feed_fillind")
FIG = ROOT / "figures_pdf"
MODELS = ["ridge", "lgb_ktz", "mlp_ktz"]
MLAB = {"ridge": "ridge", "lgb_ktz": "LightGBM (lgb_ktz)", "mlp_ktz": "MLP (mlp_ktz, linear)"}
TEST0 = pd.Timestamp("2026-02-20")
BLUE, ORANGE, GREEN, DARK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#b0aea6"
INK, MUTED, LINE, ALT = "#1f1f1d", "#6b6a66", "#dcdad4", "#f4f3f0"

D = pd.read_csv(ROOT / "paired_deltas.csv")
L = pd.read_csv(ROOT / "summary_levels.csv")
TU = pd.read_csv(ROOT / "tuned_equal_budget.csv")
BL = pd.read_csv(ROOT / "blend_summary.csv")
BY = pd.read_csv(ROOT / "paired_deltas_by_block_year.csv")
SD = pd.read_csv(ROOT / "paired_deltas_by_seed.csv")
AU = json.load(open(ROOT / "audit_summary.json"))
AR = pd.read_csv(ROOT / "audit_dropped_rows.csv")
ARMS = json.load(open(ROOT / "preds" / "arms.json"))
S = pd.read_parquet(ROOT / "scores_by_date_seed.parquet")
BA = pd.read_parquet(ROOT / "blend_by_date.parquet")
A = S.groupby(["model", "arm", "date", "block", "kind"], as_index=False)[["corr", "payout"]].mean()

CFG = {
    "L": dict(name="litmom-feed-fillind", data="crypto/v2.0-litmom-feed-fillind", color=BLUE, nf=58,
              file="E132_litmom_feed_fillind_REJECT.pdf",
              comp=[("Raw v2.0 base features", 22), ("Literature momentum (feature_mom_*)", 17),
                    ("Momentum missing-data indicators", 17), ("Base missing-data indicators", 2)],
              arms={"L_B1": "base + base indicators", "L_M1": "base + 17 momentum", "L_M2": "L_M1 + momentum indicators",
                    "L_FULL": "the whole dataset", "L_ADDED": "momentum + momentum indicators, no base",
                    "L_no_ret": "L_M1 without the returns family", "L_no_maxvol": "L_M1 without the max/vol family",
                    "L_no_trend": "L_M1 without the trend family"},
              groups=["L_M1 - B0F", "L_M2 - L_M1", "L_B1 - B0F", "L_FULL - L_M1", "L_M1 - L_no_ret", "L_M1 - L_no_maxvol",
                      "L_M1 - L_no_trend"]),
    "S": dict(name="momswap-feed-fillind", data="crypto/v2.0-momswap-feed-fillind", color=ORANGE, nf=48,
              file="E132_momswap_feed_fillind_REJECT.pdf",
              comp=[("Base features kept (8 momentum proxies removed)", 14), ("Literature momentum (feature_mom_*)", 17),
                    ("Momentum missing-data indicators", 17)],
              arms={"S_S0": "the 14 base features only", "S_M1": "S_S0 + 17 momentum", "S_FULL": "the whole dataset",
                    "S_ADDED": "momentum + indicators, no base (same columns as L_ADDED)"},
              groups=["S_S0 - B0F", "S_M1 - S_S0", "S_FULL - S_M1"]),
    "C": dict(name="momswap-macro-carver-feed-fillind", data="crypto/v2.0-momswap-macro-carver-feed-fillind", color=GREEN, nf=96,
              file="E132_momswap_macro_carver_feed_fillind_REJECT.pdf",
              comp=[("Base features kept (8 momentum proxies removed)", 14), ("Literature momentum (feature_mom_*)", 17),
                    ("Macro and cross-asset (feature_macro_*)", 18), ("Carver/Clenow trend (feature_tf_*)", 15),
                    ("Momentum missing-data indicators", 17), ("Trend missing-data indicators", 15)],
              arms={"C_MACRO": "14 base + macro", "C_TF": "14 base + trend", "C_RAW": "14 base + momentum + macro + trend",
                    "C_FULL": "the whole dataset", "C_ADDED": "everything except the 14 base features",
                    "C_no_mom": "C_FULL without momentum and its indicators", "C_no_macro": "C_FULL without macro",
                    "C_no_tf": "C_FULL without trend and its indicators"},
              groups=["C_MACRO - S_S0", "C_TF - S_S0", "C_RAW - S_M1", "C_FULL - C_RAW", "C_FULL - C_no_mom",
                      "C_FULL - C_no_macro", "C_FULL - C_no_tf"]),
}


def num(x, nd=4, sign=True):
    if x is None or x != x:
        return "n/a"
    return (("%+.*f" if sign else "%.*f") % (nd, x)).replace("-", "−")


def row(model, comp, scope):
    return D[(D["model"] == model) & (D["comparison"] == comp) & (D["scope"] == scope)].iloc[0]


def dt(model, comp, scope):
    r = row(model, comp, scope)
    return "%s (t %s)" % (num(r["mean"]), num(r["nw_t"], 1))


def cell(model, comp, scope, mean="mean", t="nw_t"):
    r = row(model, comp, scope)
    return "%s (%s)" % (num(r[mean]), num(r[t], 1))


def bar(model, comp):
    """The house bar: |t| >= 2 on dev and the same sign on test."""
    d, t = row(model, comp, "dev"), row(model, comp, "test")
    if abs(d["mean"]) < 0.0005:
        return "negligible size"
    if abs(d["nw_t"]) >= 2:
        word = "harmful" if d["mean"] < 0 else "helpful"
        if np.sign(t["mean"]) == np.sign(d["mean"]):
            return word + " on both"
        return "dev only (%s), test flips" % word
    if abs(t["nw_t"]) >= 2:
        return "test only, dev null"
    return "null"


# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": LINE, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.grid": True,
                     "grid.color": "#ececE8", "grid.linewidth": .6, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.fontsize": 7, "lines.linewidth": 1.5, "xtick.major.size": 0, "ytick.major.size": 0,
                     "savefig.dpi": 200, "date.autoformatter.year": "%Y"})


def panel(h=2.35, sharey=True):
    fig, axs = plt.subplots(1, 3, figsize=(7.3, h), sharey=sharey)
    for ax, m in zip(axs, MODELS):
        ax.set_title(MLAB[m], loc="left")
    return fig, axs


def test_mark(ax):
    ax.axvline(TEST0, color=MUTED, ls=":", lw=.9)


def years(ax):
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))


def save(fig, k, name):
    p = FIG / k / name
    p.parent.mkdir(parents=True, exist_ok=True)
    h, l = fig.axes[0].get_legend_handles_labels()
    bottom = 0
    if l:
        fig.legend(h, l, loc="lower center", ncol=len(l))
        bottom = .24 / fig.get_figheight()
    fig.tight_layout(pad=.5, rect=(0, bottom, 1, 1))
    fig.savefig(p)
    plt.close(fig)
    return p


def series(model, arm, col="corr"):
    return A[(A["model"] == model) & (A["arm"] == arm)].sort_values("date").set_index("date")[col]


def figures(k):
    c = CFG[k]
    full, col = k + "_FULL", c["color"]
    style = {"B0": (DARK, "-"), "B0F": (GREY, "--"), full: (col, "-")}
    name = {"B0": "B0 raw v2.0", "B0F": "B0F raw v2.0, same rows", full: full}
    out = {}

    fig, axs = panel()
    for ax, m in zip(axs, MODELS):
        for a, (cl, ls) in style.items():
            s = series(m, a)
            ax.plot(s.index, s.cumsum(), color=cl, ls=ls, label=name[a])
        test_mark(ax); years(ax)
    axs[0].set_ylabel("cumulative CORR")
    out["cum"] = save(fig, k, "cum_corr.png")

    fig, axs = panel()
    for ax, m in zip(axs, MODELS):
        for a, (cl, ls) in style.items():
            s = series(m, a, "payout").dropna()
            ax.plot(s.index, s.cumsum(), color=cl, ls=ls, label=name[a])
        test_mark(ax)
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
        ax.tick_params(axis="x", labelsize=6.5)
    axs[0].set_ylabel("cumulative payout (proxy)")
    out["pay"] = save(fig, k, "cum_payout.png")

    fig, axs = panel()
    for ax, m in zip(axs, MODELS):
        b, f, x = series(m, "B0"), series(m, "B0F"), series(m, full)
        ax.plot((f - b).cumsum(), color=GREY, ls="--", label="row handling: B0F − B0")
        ax.plot((x - f).cumsum(), color=DARK, label="features: %s − B0F" % full)
        ax.plot((x - b).cumsum(), color=col, label="total: %s − B0" % full)
        ax.axhline(0, color=MUTED, lw=.7); test_mark(ax); years(ax)
    axs[0].set_ylabel("cumulative ΔCORR")
    out["delta"] = save(fig, k, "cum_delta.png")

    fig, axs = panel()
    for ax, m in zip(axs, MODELS):
        for a, (cl, ls) in style.items():
            s = series(m, a).rolling(60).mean()
            ax.plot(s.index, s, color=cl, ls=ls, lw=1.2, label=name[a])
        test_mark(ax); years(ax)
    axs[0].set_ylabel("60-date mean CORR")
    out["roll"] = save(fig, k, "rolling.png")

    fig, axs = panel(2.0)
    for ax, m in zip(axs, MODELS):
        d = (series(m, full) - series(m, "B0F"))
        d = d[d.index < TEST0]
        ax.hist(d, bins=50, color=col, rwidth=.9)
        ax.axvline(0, color=INK, lw=.8); ax.axvline(d.mean(), color=DARK, ls="--", lw=1.1)
        ax.set_xlabel("ΔCORR per date, dev; dashed = mean", fontsize=6.8)
    axs[0].set_ylabel("dates")
    out["dist"] = save(fig, k, "delta_distribution.png")

    fig, axs = plt.subplots(2, 3, figsize=(7.3, 3.9))
    for j, m in enumerate(MODELS):
        g = BY[(BY["by"] == "block") & (BY["model"] == m)]
        for i, (cmp_, cl) in enumerate(((full + " - B0", col), (full + " - B0F", DARK))):
            z = g[g["comparison"] == cmp_].assign(key=lambda f: f["key"].astype(int)).sort_values("key")
            ax = axs[i, j]
            ax.bar(z["key"], z["mean"], color=[cl if b != 11 else GREY for b in z["key"]], width=.75)
            ax.axhline(0, color=MUTED, lw=.7); ax.set_xticks(range(0, 12))
            ax.tick_params(axis="x", labelsize=6); ax.grid(axis="x", visible=False)
            ax.set_title(("%s: " % MLAB[m] if i == 0 else "") + cmp_.replace(" - ", " − "), loc="left", fontsize=7)
        axs[0, j].sharey(axs[0, 0]); axs[1, j].sharey(axs[1, 0])
    axs[1, 1].set_xlabel("block (grey bar, 11 = test)"); axs[0, 0].set_ylabel("mean ΔCORR"); axs[1, 0].set_ylabel("mean ΔCORR")
    out["block"] = save(fig, k, "by_block.png")

    arms = [a for a in ("B0N", "B0F", "P_DROP", "P_HIST") + tuple(c["arms"])]
    fig, axs = plt.subplots(1, 2, figsize=(7.3, .55 + .165 * len(arms)), sharey=True)
    vals = []
    for sc in ("dev", "test"):
        lv = L[L["scope"] == sc].pivot_table(index="model", columns="arm", values="mean")
        vals.append(lv.sub(lv["B0"], axis=0).loc[MODELS, arms].T)
    v = max(np.abs(x.values).max() for x in vals)
    for ax, x, sc in zip(axs, vals, ("dev", "test")):
        ax.imshow(x.values, cmap="RdBu", vmin=-v, vmax=v, aspect="auto"); ax.grid(False)
        ax.set_title("CORR − B0, %s" % sc, loc="left")
        ax.set_xticks(range(3)); ax.set_xticklabels(["ridge", "lgb_ktz", "mlp_ktz"]); ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms, fontsize=7)
        for i in range(x.shape[0]):
            for j in range(3):
                z = x.values[i, j]
                ax.text(j, i, num(z, 3), ha="center", va="center", fontsize=6.5, color="white" if abs(z) > .62 * v else INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
    out["heat"] = save(fig, k, "heatmap.png")

    fig, axs = panel(2.0)
    for ax, m in zip(axs, MODELS):
        g = BA[(BA["model"] == m) & (BA["dataset"] == k) & (BA["kind"] == "dev")].groupby("w")["corr"].mean()
        ax.plot(g.index, g.values - g.loc[0.0], color=col, marker="o", ms=4)
        ax.axhline(0, color=MUTED, lw=.7); ax.set_xlabel("w on %s_ADDED" % k)
    axs[0].set_ylabel("dev CORR − (w = 0)")
    out["blend"] = save(fig, k, "blend.png")

    comps = [full + " - B0", full + " - B0F", k + "_ADDED - B0F"] + c["groups"]
    fig, axs = plt.subplots(1, 3, figsize=(7.3, .7 + .24 * len(comps)), sharey=True, sharex=True)
    for ax, m in zip(axs, MODELS):
        for i, cp in enumerate(reversed(comps)):
            for sc, cl, off, mk in (("dev", col, .16, "o"), ("test", DARK, -.16, "s")):
                r = row(m, cp, sc)
                ax.errorbar(r["mean"], i + off, xerr=[[r["mean"] - r["nw_lo"]], [r["nw_hi"] - r["mean"]]], fmt=mk, color=cl, ms=3, lw=1,
                            label=sc if i == 0 else None)
        ax.axvline(0, color=INK, lw=.8); ax.set_title(MLAB[m], loc="left"); ax.set_xlabel("ΔCORR, NW 95% CI")
        ax.tick_params(axis="x", labelsize=6.5)
    axs[0].set_yticks(range(len(comps))); axs[0].set_yticklabels([x.replace(" - ", " − ") for x in reversed(comps)], fontsize=6.5)
    out["ci"] = save(fig, k, "ablation_ci.png")

    arms2 = ["B0F", "P_DROP"] + list(c["arms"])
    fig, axs = plt.subplots(1, 3, figsize=(7.3, .6 + .19 * len(arms2)), sharey=True, sharex=True)
    pc = L[L["scope"] == "dev"].pivot_table(index="arm", columns="model", values="predcorr_b0")
    for ax, m in zip(axs, MODELS):
        ax.barh(range(len(arms2)), pc.loc[arms2[::-1], m], color=col, height=.7)
        for i, a in enumerate(arms2[::-1]):
            ax.text(pc.loc[a, m] + .01, i, "%.2f" % pc.loc[a, m], va="center", fontsize=6.3, color=INK)
        ax.set_title(MLAB[m], loc="left"); ax.set_xlim(0, 1.12); ax.grid(axis="y", visible=False); ax.set_xlabel("rank corr with B0 (dev)")
    axs[0].set_yticks(range(len(arms2))); axs[0].set_yticklabels(arms2[::-1], fontsize=7)
    out["pred"] = save(fig, k, "predcorr.png")
    return out


def neutral_fig(k):
    keys = pd.read_parquet(ROOT / "preds" / "keys.parquet")
    m = keys.groupby("date")["no_feed"].mean()
    fig, ax = plt.subplots(figsize=(7.3, 1.8))
    ax.plot(m.index, m.values * 100, color=DARK)
    test_mark(ax); ax.set_ylabel("% of coins at 0.5"); ax.set_ylim(0)
    return save(fig, k, "neutral_share.png")


def indicator_audit(k):
    """Every indicator in THIS dataset, on its own rows, by scope."""
    keys = pd.read_parquet(ROOT / "preds" / "keys.parquet")
    f = pd.read_parquet("data/%s/train.parquet" % CFG[k]["data"]).reset_index()
    T20 = "target_binned_return_20"
    f = f[f["date"].isin(keys["date"])]
    rows = []
    for c in [c for c in f if c.startswith("feature_isna_")]:
        for sc, g in (("dev", f[f["date"] < TEST0]), ("test", f[f["date"] >= TEST0])):
            on = g[g[c] == 1]
            rows.append(dict(indicator=c.replace("feature_isna_", ""), scope=sc, share=(g[c] == 1).mean(),
                             target_on=on[T20].mean(), target_off=g.loc[g[c] == 0, T20].mean()))
    r = pd.DataFrame(rows)
    assert len(f) == 468000 - AU["dropped_rows"], "the datasets must drop the same rows"
    return r, len(f)


# ------------------------------------------------------------------ page furniture
for n, fn in (("DV", "DejaVuSans"), ("DV-B", "DejaVuSans-Bold"), ("DV-I", "DejaVuSans-Oblique"), ("DVM", "DejaVuSansMono")):
    pdfmetrics.registerFont(TTFont(n, os.path.join(matplotlib.get_data_path(), "fonts/ttf", fn + ".ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")
BODY = ParagraphStyle("body", fontName="DV", fontSize=8.8, leading=12.6, textColor=INK, spaceAfter=5, alignment=TA_LEFT)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.4, leading=10, textColor=colors.HexColor(MUTED))
H1 = ParagraphStyle("h1", parent=BODY, fontName="DV-B", fontSize=12.5, leading=16, spaceBefore=11, spaceAfter=5, keepWithNext=1)
H2 = ParagraphStyle("h2", parent=BODY, fontName="DV-B", fontSize=9.4, leading=12.5, spaceBefore=7, spaceAfter=3, keepWithNext=1)
TITLE = ParagraphStyle("title", parent=BODY, fontName="DV-B", fontSize=19, leading=24, spaceAfter=4)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=10, leading=14, textColor=colors.HexColor(MUTED), spaceAfter=9)
CAP = ParagraphStyle("cap", parent=SMALL, spaceBefore=2, spaceAfter=8)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=11, bulletIndent=1, spaceAfter=3)
CL = ParagraphStyle("cl", parent=BODY, fontSize=6.9, leading=8.8, spaceAfter=0)
VBOX = ParagraphStyle("v", parent=BODY, fontSize=9.4, leading=13.4, spaceAfter=0)


class Doc(SimpleDocTemplate):
    head = ""

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name == "h1":
            key = "h%d" % self.seq
            self.seq += 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(f.getPlainText(), key, 0, 0)
            self.notify("TOCEntry", (0, f.getPlainText(), self.page, key))


class Numbered(pdfcanvas.Canvas):
    head = ""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved)
        for st in self._saved:
            self.__dict__.update(st)
            self.setFont("DV", 6.8); self.setFillColor(colors.HexColor(MUTED))
            if self._pageNumber > 1:
                self.drawString(18 * mm, A4[1] - 11 * mm, self.head)
            self.drawString(18 * mm, 9 * mm, "Proxy MMC, not official. Nothing was uploaded, created or staked.")
            self.drawRightString(A4[0] - 18 * mm, 9 * mm, "%d / %d" % (self._pageNumber, n))
            super().showPage()
        super().save()


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(xs):
    return [Paragraph(x, BUL, bulletText="•") for x in xs]


def mono(s):
    return '<font name="DVM" size="8">%s</font>' % s


def table(rows, widths=None, size=6.9, left=1, group=None, note_col=None):
    """rows[0] is the header. Text columns wrap, numbers stay on one line."""
    body = [[Paragraph(str(x), CL) if (note_col is not None and j in note_col and i) else x for j, x in enumerate(r)] for i, r in enumerate(rows)]
    t = Table(body, colWidths=widths, repeatRows=1, hAlign="LEFT")
    st = [("FONT", (0, 0), (-1, -1), "DV", size), ("FONT", (0, 0), (-1, 0), "DV-B", size), ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
          ("ALIGN", (left, 0), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, 0), .7, colors.HexColor(DARK)),
          ("LINEBELOW", (0, 1), (-1, -1), .25, colors.HexColor(LINE)), ("TOPPADDING", (0, 0), (-1, -1), 2.1), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.1),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(ALT))]
    if note_col:
        st += [("ALIGN", (j, 0), (j, -1), "LEFT") for j in note_col]
    if group:
        st += [("LINEABOVE", (0, i), (-1, i), .6, colors.HexColor(MUTED)) for i in range(2, len(rows)) if rows[i][group - 1] != rows[i - 1][group - 1] and i > 1]
    t.setStyle(TableStyle(st))
    return t


def fig(path, cap=None):
    from PIL import Image as PI
    w, h = PI.open(path).size
    W = A4[0] - 36 * mm
    out = [Image(str(path), width=W, height=W * h / w)]
    if cap:
        out.append(P(cap, CAP))
    return KeepTogether(out)


# ------------------------------------------------------------------ dataset-specific reading
def reading(k):
    full = k + "_FULL"
    if k == "L":
        return [
            "<b>Raw momentum is what hurts.</b> Adding the 17 momentum columns to raw v2.0 costs ridge %s and the MLP %s on dev. "
            "Test is small and not significant (%s and %s). LightGBM is null, %s." % (
                dt("ridge", "L_M1 - B0F", "dev"), dt("mlp_ktz", "L_M1 - B0F", "dev"), dt("ridge", "L_M1 - B0F", "test"), dt("mlp_ktz", "L_M1 - B0F", "test"), dt("lgb_ktz", "L_M1 - B0F", "dev")),
            "<b>Indicators add a little more harm for ridge.</b> All indicators together cost ridge %s on dev, with test at %s. "
            "They are null for LightGBM and the MLP on dev." % (dt("ridge", "L_FULL - L_M1", "dev"), dt("ridge", "L_FULL - L_M1", "test")),
            "<b>The two base indicators hurt on test.</b> They are null on dev for every model, "
            "then cost LightGBM %s and the MLP %s on test. The dev null means this fails the house bar, but they should not be kept." % (
                dt("lgb_ktz", "L_B1 - B0F", "test"), dt("mlp_ktz", "L_B1 - B0F", "test")),
            "<b>Inside momentum,</b> the trend family is the only one with a dev effect, on ridge: %s, with test at %s. "
            "The returns family is not significant for any model on dev, and neither is max/vol." % (dt("ridge", "L_M1 - L_no_trend", "dev"), dt("ridge", "L_M1 - L_no_trend", "test")),
        ]
    if k == "S":
        return [
            "<b>Removing the 8 momentum proxies is about neutral.</b> Ridge gains %s on dev, below the bar, and LightGBM and the MLP do not move significantly. "
            "Test is %s, %s and %s." % (dt("ridge", "S_S0 - B0F", "dev"), num(row("ridge", "S_S0 - B0F", "test")["mean"]), num(row("lgb_ktz", "S_S0 - B0F", "test")["mean"]), num(row("mlp_ktz", "S_S0 - B0F", "test")["mean"])),
            "<b>Swapping in literature momentum is where the set loses on dev.</b> It costs ridge %s and the MLP %s." % (dt("ridge", "S_M1 - S_S0", "dev"), dt("mlp_ktz", "S_M1 - S_S0", "dev")),
            "<b>The same swap is the most interesting test result in this report, and it still fails.</b> It gains LightGBM %s and the MLP %s on test. "
            "LightGBM's dev effect is null (%s) and the MLP's is significantly negative, so neither has the dev support the bar asks for. "
            "This window was also seen by E65 and E131. It is worth watching and is not a win." % (dt("lgb_ktz", "S_M1 - S_S0", "test"), dt("mlp_ktz", "S_M1 - S_S0", "test"), dt("lgb_ktz", "S_M1 - S_S0", "dev")),
            "<b>Indicators</b> cost ridge %s on dev, test %s. They are null for LightGBM and the MLP on dev." % (dt("ridge", "S_FULL - S_M1", "dev"), dt("ridge", "S_FULL - S_M1", "test")),
        ]
    return [
        "<b>The carver indicators are the clearest dev harm in this set.</b> They cost ridge %s and the MLP %s, with LightGBM null. Test turns positive and not significant (%s and %s)." % (
            dt("ridge", "C_FULL - C_RAW", "dev"), dt("mlp_ktz", "C_FULL - C_RAW", "dev"), dt("ridge", "C_FULL - C_RAW", "test"), dt("mlp_ktz", "C_FULL - C_RAW", "test")),
        "<b>Momentum is harmful here too.</b> Taking it out helps ridge (%s is the value of keeping it) and the MLP (%s) on dev. The MLP's test sign flips, %s, so this is not a clean read." % (
            dt("ridge", "C_FULL - C_no_mom", "dev"), dt("mlp_ktz", "C_FULL - C_no_mom", "dev"), dt("mlp_ktz", "C_FULL - C_no_mom", "test")),
        "<b>Macro does nothing reliable.</b> Ridge's %s reaches the t threshold on a negligible effect size. The MLP's %s reverses on test, %s." % (
            dt("ridge", "C_MACRO - S_S0", "dev"), dt("mlp_ktz", "C_MACRO - S_S0", "dev"), dt("mlp_ktz", "C_MACRO - S_S0", "test")),
        "<b>Carver trend is null to mildly harmful on dev.</b> The MLP loses %s, and ridge and LightGBM are below the bar. LightGBM's %s on test is close to the bar with no dev support." % (
            dt("mlp_ktz", "C_TF - S_S0", "dev"), dt("lgb_ktz", "C_TF - S_S0", "test")),
        "<b>On payout, macro plus trend hurts the MLP on both windows:</b> %s dev (t %s), %s test (t %s), proxy MMC, with CORR null. It is the only group result in this report that clears the dev bar on payout and holds on test." % (
            num(row("mlp_ktz", "C_RAW - S_M1", "dev")["payout_mean"]), num(row("mlp_ktz", "C_RAW - S_M1", "dev")["payout_t"], 1),
            num(row("mlp_ktz", "C_RAW - S_M1", "test")["payout_mean"]), num(row("mlp_ktz", "C_RAW - S_M1", "test")["payout_t"], 1)),
    ]


# ------------------------------------------------------------------ the document
def build(k):
    c = CFG[k]
    full, added, col = k + "_FULL", k + "_ADDED", c["color"]
    figs = figures(k)
    figs["neutral"] = neutral_fig(k)
    ind, nrows = indicator_audit(k)
    b = BL[BL["dataset"] == k].set_index("model")
    tu = TU[TU["comparison"].str.startswith(k + " ")]
    st = []

    st += [P("E132: %s" % c["name"], TITLE), P("Does %s add value over crypto/v2.0 when coins without features are held at 0.5?" % mono(c["data"]), SUB)]
    ok = all(row(m, full + " - B0", "dev")["nw_t"] <= -2 and row(m, full + " - B0", "test")["mean"] < 0 for m in MODELS)
    assert ok
    harm = [m for m in MODELS if row(m, full + " - B0F", "dev")["nw_t"] <= -2]
    box = Table([[P("<b>Verdict: REJECT.</b> This dataset loses to raw v2.0 on all three models, on dev and on test, by %s to %s CORR on dev. "
                    "Retuning does not rescue it and no blend passes. Row handling, meaning the 0.5 rows plus refitting without those coins, accounts for about −0.0175 of the loss. "
                    "The features alone are %s, and not significant on test." % (
                        num(max(row(m, full + " - B0", "dev")["mean"] for m in MODELS)), num(min(row(m, full + " - B0", "dev")["mean"] for m in MODELS)),
                        "harmful on dev for ridge and the MLP, null for LightGBM" if len(harm) == 2 else "mixed"), VBOX)]], colWidths=[A4[0] - 36 * mm])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(ALT)), ("LINEBEFORE", (0, 0), (0, -1), 2.5, colors.HexColor(col)),
                             ("LEFTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    st += [box, Spacer(1, 8)]
    rows = [["model", "vs v2.0\ndev", "vs v2.0\ntest", "features alone\ndev", "features alone\ntest", "payout vs v2.0\ndev", "payout vs v2.0\ntest", "verdict\nvs v2.0", "verdict\nfeatures alone"]]
    for m in MODELS:
        fa = "REJECT" if row(m, full + " - B0F", "dev")["nw_t"] <= -2 else "NO IMPROVEMENT"
        rows.append([m, cell(m, full + " - B0", "dev"), cell(m, full + " - B0", "test"), cell(m, full + " - B0F", "dev"), cell(m, full + " - B0F", "test"),
                     cell(m, full + " - B0", "dev", "payout_mean", "payout_t"), cell(m, full + " - B0", "test", "payout_mean", "payout_t"), "REJECT", fa])
    st += [table(rows, size=6.3, left=1, note_col=[8]),
           P("ΔCORR, with the Newey-West t at lag 20 in brackets. \"Features alone\" compares with raw v2.0 on the same rows and the same 0.5 treatment. "
             "Payout is 0.1 × CORR + proxy MMC, built per period.", CAP)]
    st += [P("Contents", H2), TableOfContents(), PageBreak()]
    st[-2].levelStyles = [ParagraphStyle("toc", parent=BODY, fontSize=8.4, leading=11.5, spaceAfter=0)]

    # 1
    st += [P("1. What this feature set contains", H1)]
    st += [table([["block", "columns"]] + [[a, b_] for a, b_ in c["comp"]] + [["total", c["nf"]]], widths=[300, 60])]
    assert sum(x[1] for x in c["comp"]) == c["nf"] == len(ARMS[full]["cols"])
    st += [Spacer(1, 6), P("Arms fitted for this dataset", H2)]
    st += [table([["arm", "columns", "what it is"]] + [[a, len(ARMS[a]["cols"]), d] for a, d in c["arms"].items()], widths=[70, 45, 300], note_col=[2])]
    if k == "C":
        st += [P("The macro and trend comparisons use S_S0 and S_M1 from the momswap set as their reference, because those are the same columns on the same rows. "
                 "They appear in the tables below for that reason.", CAP)]
    st += [P("Baselines and controls, shared by all three reports", H2),
           table([["arm", "columns", "what it is"],
                  ["B0", 22, "raw v2.0, fitted and scored on every key. The baseline to beat."],
                  ["B0F", 22, "raw v2.0 fitted on the feed rows, 0.5 elsewhere. Separates row handling from features."],
                  ["B0N", 22, "B0's own predictions with the dropped rows set to 0.5, no refit."],
                  ["P_DROP", 23, "raw v2.0 + the no-feed flag on every key. A look-ahead placebo."],
                  ["P_HIST", 23, "raw v2.0 + causal days in panel, on the feed rows."]], widths=[70, 45, 300], note_col=[2])]

    # 2
    st += [P("2. Design, and where it departs from the protocol", H1)]
    st += bullets([
        "Protocol of E131: 513,496 keys, 12 walk-forward blocks (train 130, purge 22, predict 130), 11 dev blocks and a test block of 2026-02-20 to 2026-08-20, on target_binned_return_20. Ridge is deterministic. LightGBM and the MLP use 5 seeds.",
        "Scoring uses 468,000 keys. Neutral rule, as requested: each arm's predictions become a per-date percentile, and a key with no prediction takes 0.5. A coin with no features is scored at the middle.",
        "Models are ridge, lgb_ktz and mlp_ktz at E131's fixed configs, searched on raw v2.0. There is no ridge_ktz. mlp_ktz uses identity activation, so it is linear and this study has no nonlinear neural arm.",
        "<b>The test block is not virgin:</b> E65 and E131 scored it before. It was not used for any choice here.",
        "MMC is a proxy against the historical meta model and is not official. Payout rows use only the dates that have a meta model, so calmar and burn(d) compare only between rows with the same n(P).",
    ])

    # 3
    st += [P("3. Point-in-time audit", H1),
           P("<b>The row set is look-ahead.</b> %s of 468,000 scored keys (%.2f%%) sit at 0.5, on %d symbols. A coin is dropped because no price feed exists <i>today</i>. "
             "Dropped rows average a target of %.3f against %.3f for kept rows, and only %d of the %d symbols are still in the panel on the last date. "
             "So the row set knows which coins die, and fitting on it (B0F) is not causal either. All three datasets drop the same rows." % (
                 format(AU["dropped_rows"], ","), AU["share"] * 100, AU["symbols"], AU["target_dropped"], AU["target_kept"], AU["symbols_alive_on_last_date"], AU["symbols"]))]
    rows = [["year", "keys", "share at 0.5", "target, kept", "target, dropped"]] + [[int(r.year), format(int(r.rows), ","), "%.1f%%" % (r.dropped * 100), "%.3f" % r.target_kept, "%.3f" % r.target_dropped] for r in AR.itertuples()]
    st += [KeepTogether(table(rows, widths=[50, 60, 70, 70, 80])), Spacer(1, 6), fig(figs["neutral"], "Share of coins held at 0.5 per date. The dotted line starts the test block.")]
    w = ind[ind["scope"] == "dev"]
    st += [P("Inside the kept rows, the %d missing-data indicators in this dataset mark weaker coins: on dev, the target with an indicator on averages %.3f against %.3f with it off, across indicators. "
             "Warm-up is known at the time, so this channel is causal. The full table is in Appendix C." % (ind["indicator"].nunique(), w["target_on"].mean(), w["target_off"].mean())),
           P("<b>Open data issue:</b> about 25 kept coins carry wrong-coin Yahoo prices, so some computed values describe the wrong token. This study did not correct them, which limits how far the features-alone numbers can be read as a clean test.")]

    # 4
    st += [P("4. Levels", H1)]
    lev = ["B0", "B0F", full]
    hdr = ["model", "arm", "scope", "CORR", "MMC", "c.sharpe", "dd(corr)", "payout", "p.sharpe", "dd(pay)", "calmar", "burn(d)", "tgt", "n(C)", "n(P)"]

    def lrow(r):
        return [r.model, r.arm, r.scope, num(r.mean, 4, False), num(r.mmc, 4, False), num(r.sharpe, 2, False), num(r.max_dd, 2, False), num(r.payout, 4, False), num(r.p_sharpe, 2, False),
                num(r.dd_pay, 2, False), num(r.calmar, 3, False), int(r.burn), "t20", int(r.n), int(r.window_n)]

    def ltable(arms):
        rows = [hdr]
        for m in MODELS:
            for a in arms:
                for sc in ("dev", "test"):
                    rows.append(lrow(next(L[(L["model"] == m) & (L["arm"] == a) & (L["scope"] == sc)].itertuples())))
        return table(rows, size=6.2, left=3, group=1)

    st += [ltable(lev), P("CORR, c.sharpe and dd(corr) cover n(C) dates. MMC (proxy), payout, p.sharpe, dd(pay), calmar and burn(d) cover the n(P) dates with a meta model. "
                          "Dev and test rows have different n, so compare calmar and burn(d) only within a scope. Every arm is in Appendix A.", CAP)]
    st += [fig(figs["cum"], "Cumulative CORR. The dotted line starts the test block."), fig(figs["pay"], "Cumulative payout with proxy MMC, on the dates with a meta model.")]

    # 5
    st += [P("5. Paired deltas: this dataset against the baselines", H1)]
    hdr2 = ["comparison", "model", "scope", "ΔCORR", "NW t", "NW 95% CI", "bootstrap 95% CI", "days +", "kept rows Δ (t)", "Δpayout (t)"]

    def pd_rows(comps):
        rows = [hdr2]
        for cp in comps:
            for m in MODELS:
                for sc in ("dev", "test"):
                    r = row(m, cp, sc)
                    rows.append([cp.replace(" - ", " − "), m, sc, num(r["mean"]), num(r["nw_t"], 1), "%s, %s" % (num(r["nw_lo"]), num(r["nw_hi"])), "%s, %s" % (num(r["boot_lo"]), num(r["boot_hi"])),
                                 "%.0f%%" % (r["pos_rate"] * 100), "%s (%s)" % (num(r["kept_mean"]), num(r["kept_t"], 1)), "%s (%s)" % (num(r["payout_mean"]), num(r["payout_t"], 1))])
        return table(rows, size=6.1, left=3, group=1)

    st += [pd_rows([full + " - B0", full + " - B0F", added + " - B0F"])]
    st += [P("Paired by date, seed-averaged. NW is Newey-West at lag 20. \"Kept rows\" rescores the same predictions on feed rows only.", CAP)]
    st += [fig(figs["delta"], "The total loss split into row handling and features. The two parts add to the total."), fig(figs["roll"], "60-date rolling mean CORR."),
           fig(figs["dist"], "%s − B0F, per date on dev. The MLP and ridge means sit left of zero; the spread is wide relative to the mean, which is why the overlap-robust t is used." % full)]

    # 6
    st += [P("6. Where the loss comes from: row handling", H1),
           P("These comparisons contain no dataset features, so they are the same in all three reports. Setting the rows to 0.5 costs about −0.013. Refitting on the feed rows costs about −0.004 more. "
             "v2.0 already ranks these coins low, and they are low, so a 0.5 throws that information away. On kept rows B0N − B0 is exactly zero, as it must be.")]
    st += [pd_rows(["B0N - B0", "B0F - B0N", "B0F - B0"])]

    # 7
    st += [P("7. Feature group ablations", H1)]
    st += bullets(reading(k))
    rows = [["group", "meaning", "model", "dev", "test", "Δpayout dev", "Δpayout test", "house bar (CORR)"]]
    for cp in c["groups"]:
        for m in MODELS:
            rows.append([cp.replace(" - ", " − "), D.loc[D["comparison"] == cp, "meaning"].iloc[0], m, cell(m, cp, "dev"), cell(m, cp, "test"),
                         cell(m, cp, "dev", "payout_mean", "payout_t"), cell(m, cp, "test", "payout_mean", "payout_t"), bar(m, cp)])
    st += [Spacer(1, 4), table(rows, size=6.1, left=3, group=1, note_col=[1, 7], widths=[68, 62, 34, 62, 62, 62, 62, 60]),
           P("Cells are Δ (NW t). The bar is |t| ≥ 2 on dev and the same sign on test. LOFO and LOBO rows read as the value of keeping that block. "
             "With this many cells, some |t| ≥ 2 results are expected by chance.", CAP)]
    st += [fig(figs["ci"], "Newey-West 95% intervals, dev and test."), fig(figs["heat"], "Every arm against B0, dev and test. Blue is better than B0, red is worse.")]

    # 8
    st += [P("8. Equal-budget retune", H1),
           P("Each model was re-searched with 50 draws on this dataset's rows, and raw v2.0 got the same budget. This rules out the fixed configs simply favouring v2.0. "
             "Candidates and B0F are completed with the neutral rule, and B0 is E131's search on all keys. Pre-test covers 2024-08 to 2026-02-19.")]
    rows = [["model", "vs", "scope", "candidate", "reference", "ΔCORR", "NW t", "bootstrap 95% CI", "n"]]
    for r in tu.itertuples():
        rows.append([r.model, r.comparison.split(" - ")[1], r.scope, num(r.arm, 4, False), num(r.ref, 4, False), num(r.mean), num(r.nw_t, 1), "%s, %s" % (num(r.boot_lo), num(r.boot_hi)), int(r.n)])
    st += [table(rows, widths=[46, 30, 44, 50, 50, 50, 36, 96, 28], left=3, group=1)]
    v = tu[tu["comparison"] == k + " - B0"]
    nneg, n2 = int((v["mean"] < 0).sum()), int((v["nw_t"].abs() >= 2).sum())
    txt = "Against v2.0, %d of %d cells are negative and %d reach |t| ≥ 2. Retuning does not rescue this dataset." % (nneg, len(v), n2)
    if k == "S":
        txt += " One cell is positive against B0F: the MLP on test, +0.0165 (t +2.1). Its pre-test is −0.0201 and its bootstrap interval includes zero, so it fails the same-sign bar."
    st += [Spacer(1, 3), P(txt)]

    # 9
    st += [P("9. Placebo and tenure controls", H1),
           P("Shared by all three reports. P_DROP is the look-ahead flag alone. P_HIST is causal days in panel. P_HIST weakens E131's tenure lead: E131 was positive on test for all three models, and here the test signs are mixed and none is significant.")]
    st += [pd_rows(["P_DROP - B0", "P_HIST - B0F"])]

    # 10
    st += [P("10. Blends", H1), P("(1 − w) × B0 + w × %s on completed percentiles. w is chosen on dev, then test is read once." % added)]
    rows = [["model", "w*", "dev Δ", "dev t", "test, w = 0", "test, w*", "test Δ", "test t", "w* in leave-one-block-out folds"]]
    for m in MODELS:
        r = b.loc[m]
        rows.append([m, "%.2f" % r["w_star"], num(r["dev_delta"]), num(r["dev_t"], 1), num(r["test_w0"], 4, False), num(r["test_wstar"], 4, False), num(r["test_delta"]), num(r["test_t"], 1),
                     ", ".join(sorted(set(r["lobo"].split(","))))])
    st += [table(rows, size=6.6, widths=[42, 26, 44, 32, 50, 44, 44, 32, 130], note_col=[8]), Spacer(1, 3)]
    if k in "LS":
        st += [P("Ridge and the MLP pick w = 0 in every fold, so no blend exists for them. E131's LightGBM lead does not repeat on these rows: w falls to 0.10 and the test gain is %s (t %s), not significant.%s" % (
            num(b.loc["lgb_ktz", "test_delta"]), num(b.loc["lgb_ktz", "test_t"], 1), " L_ADDED and S_ADDED are the same columns, so this table matches the litmom report exactly." if k == "S" else ""))]
    else:
        st += [P("Every model picks a small w, none clears the bar on dev, and test is %s for LightGBM, %s for ridge and %s for the MLP. No blend passes." % (
            num(b.loc["lgb_ktz", "test_delta"]), num(b.loc["ridge", "test_delta"]), num(b.loc["mlp_ktz", "test_delta"])))]
    st += [fig(figs["blend"], "Dev gain by blend weight. Test is shown only in the table, so it cannot influence the choice of w."), fig(figs["pred"], "Mean per-date rank correlation with B0 on dev. Lower means more diverse, which only pays if CORR holds up.")]

    # 11
    st += [KeepTogether([P("11. Stability by block, year and seed", H1), fig(figs["block"], "Mean ΔCORR by block.")])]

    def pivot(by):
        g = BY[(BY["by"] == by) & BY["comparison"].isin([full + " - B0", full + " - B0F"])].copy()
        g["key"] = g["key"].astype(int)
        p = g.pivot_table(index=["comparison", "model"], columns="key", values="mean", sort=False)
        rows = [["comparison", "model"] + [str(x) for x in p.columns]]
        for (cp, m), r in p.iterrows():
            rows.append([cp.replace(" - ", " − "), m] + [num(x, 3) for x in r.values])
        return table(rows, size=5.9 if by == "block" else 6.8, left=2, group=1)

    st += [P("By block (11 = test)", H2), pivot("block"), P("By year", H2), pivot("year"), P("By seed", H2)]
    rows = [["comparison", "model", "scope", "lowest seed", "highest seed", "seeds below 0", "seeds at |t| ≥ 2"]]
    for cp in (full + " - B0", full + " - B0F"):
        for m in MODELS[1:]:
            for sc in ("dev", "test"):
                g = SD[(SD["model"] == m) & (SD["comparison"] == cp) & (SD["scope"] == sc)]
                rows.append([cp.replace(" - ", " − "), m, sc, num(g["mean"].min()), num(g["mean"].max()), "%d of %d" % ((g["mean"] < 0).sum(), len(g)), "%d of %d" % ((g["nw_t"].abs() >= 2).sum(), len(g))])
    st += [table(rows, left=3, group=1), P("Ridge is deterministic, so it has one seed.", CAP)]

    # 12
    st += [P("12. Gaps", H1)]
    st += bullets(["MISS_ONLY and M1FB from E131 were not rerun.",
                   "All 12 retunes finished, after three were relaunched because systemd stopped them. None is missing.",
                   "The retune comparison against B0 uses E131's search on all keys, without neutral rows.",
                   "There is no catboost or forest arm. These three classes repeat E131's.",
                   "Wrong-coin prices were not corrected (section 3)."])
    st += [P("13. Recommended next steps", H1)]
    st += bullets(["Do not ship this dataset on these terms, and do not set no-feature coins to 0.5 in a live slot based on this study.",
                   "To judge the features alone, fix the about 25 wrong-coin price series first, then rerun only %s and B0F." % full,
                   "Stop work on the missing-data indicators here. E131 and E132 agree they do not help.",
                   "The tenure lead needs a causal test on all keys before it gets more time, because its test sign was mixed here."])

    # appendices
    st += [PageBreak(), P("Appendix A. Every arm, levels", H1), ltable(["B0", "B0N", "B0F", "P_DROP", "P_HIST"] + list(c["arms"]) + (["S_S0", "S_M1"] if k == "C" else [])),
           P("Appendix B. Group ablation deltas with intervals", H1), pd_rows(c["groups"]),
           P("Appendix C. Missing-data indicators in this dataset", H1)]
    p = ind.pivot_table(index="indicator", columns="scope", values=["share", "target_on", "target_off"], sort=False)
    rows = [["indicator", "share on, dev", "target on, dev", "target off, dev", "share on, test", "target on, test", "target off, test"]]
    for i, r in p.iterrows():
        rows.append([i] + ["%.1f%%" % (r[("share", "dev")] * 100), num(r[("target_on", "dev")], 3, False), num(r[("target_off", "dev")], 3, False),
                           "%.1f%%" % (r[("share", "test")] * 100), num(r[("target_on", "test")], 3, False), num(r[("target_off", "test")], 3, False)])
    st += [table(rows, size=6.5), P("On this dataset's own rows over the scored dates. n/a means the indicator never fired in that window.", CAP)]

    head = "E132 · %s · REJECT" % c["name"]
    Numbered.head = head
    path = ROOT / c["file"]
    doc = Doc(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm, bottomMargin=14 * mm,
              title="E132 %s: REJECT" % c["name"], author="my-numerai-agent", subject="Feature set evaluation against crypto/v2.0")
    doc.seq = 0

    class C(Numbered):
        pass
    C.head = head
    orig = doc.handle_documentBegin

    def begin():
        doc.seq = 0
        orig()
    doc.handle_documentBegin = begin
    doc.multiBuild(st, canvasmaker=C)
    print(path, os.path.getsize(path), "rows checked", nrows)


if __name__ == "__main__":
    for k in sys.argv[1:] or list(CFG):
        build(k)
