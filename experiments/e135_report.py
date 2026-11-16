#!/usr/bin/env python3
"""E135 PDF: crypto/v2.0-histfe under the live routing rule, from e135_route_histfe.py's tables.

    python3 experiments/e135_report.py
"""
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Spacer, Table, TableStyle

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e132_pdf as G  # noqa: E402  fonts, styles, table and figure helpers
from e132_pdf import ALT, CAP, DARK, GREY, H1, H2, MLAB, MODELS, MUTED, P, SUB, TITLE, VBOX, bullets, fig, mono, num, table  # noqa: E402

os.chdir(REPO)
ROOT = pathlib.Path("results/reports/crypto/e135_route_histfe")
D = pd.read_csv(ROOT / "paired_deltas.csv")
LV = pd.read_csv(ROOT / "summary_levels.csv")
TU = pd.read_csv(ROOT / "tuned_equal_budget.csv")
BY = pd.read_csv(ROOT / "paired_deltas_by_block_year.csv")
SD = pd.read_csv(ROOT / "paired_deltas_by_seed.csv")
VT = pd.read_csv(ROOT / "vs_tenure.csv")
E133 = pd.read_csv("results/reports/crypto/e133_histfe/paired_deltas.csv")
S = pd.read_parquet(ROOT / "scores_by_date_seed.parquet")
B = pd.read_parquet(ROOT / "scores_b0.parquet")
FILE = "E135_histfe_routed_REJECT.pdf"
HOR = ["SHORT", "LONG", "COMBO", "TSC"]
FAM = ["F_RANK", "F_MA", "F_TSMAX"]
COL = {"FULL": "#2a78d6", "F_RANK": "#eb6834", "TSC": "#1baf7a"}
FK = "../../e135_route_histfe/figures"


def row(m, a, sc, rule="R_SLOT"):
    return D[(D["model"] == m) & (D["arm"] == a) & (D["scope"] == sc) & (D["rule"] == rule)].iloc[0]


def cell(m, a, sc, mean="mean", t="nw_t", rule="R_SLOT"):
    r = row(m, a, sc, rule)
    return "%s (%s)" % (num(r[mean]), num(r[t], 1))


def verdict(m, a, rule="R_SLOT"):
    """E134's bar, unchanged."""
    d, t = row(m, a, "dev", rule), row(m, a, "test", rule)
    if d["nw_t"] <= -2:
        return "REJECT"
    if d["nw_t"] >= 2 and t["mean"] > 0 and d["payout_t"] > -2 and t["payout_t"] > -2:
        return "ADOPT"
    if t["nw_t"] >= 2:
        return "PROMISING"
    return "NO IMPROVEMENT"


def cum(m, a):
    return S[(S["model"] == m) & (S["arm"] == a) & (S["rule"] == "R_SLOT")].groupby("date")[["corr", "payout"]].mean()


def figures():
    out = {}
    fig_, axs = G.panel()
    for ax, m in zip(axs, MODELS):
        b = B[B["model"] == m].groupby("date")["corr"].mean()
        for a in COL:
            ax.plot((cum(m, a)["corr"] - b).cumsum(), color=COL[a], label="routed %s" % a)
        ax.plot((cum(m, "P_HIST")["corr"] - b).cumsum(), color=GREY, ls="--", label="tenure control")
        ax.axhline(0, color=MUTED, lw=.7); G.test_mark(ax); G.years(ax)
    axs[0].set_ylabel("cumulative ΔCORR vs B0")
    out["delta"] = G.save(fig_, FK, "cum_delta.png")
    fig_, axs = G.panel()
    for ax, m in zip(axs, MODELS):
        ax.plot(B[B["model"] == m].groupby("date")["payout"].mean().dropna().cumsum(), color=DARK, ls="--", label="B0 raw v2.0")
        for a in COL:
            ax.plot(cum(m, a)["payout"].dropna().cumsum(), color=COL[a], label="routed %s" % a)
        G.test_mark(ax)
        ax.xaxis.set_major_locator(G.matplotlib.dates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(G.matplotlib.dates.DateFormatter("%b %y"))
        ax.tick_params(axis="x", labelsize=6.5)
    axs[0].set_ylabel("cumulative payout (proxy)")
    out["pay"] = G.save(fig_, FK, "cum_payout.png")
    fig_, axs = plt.subplots(1, 3, figsize=(7.3, 2.0), sharey=True)
    for ax, m in zip(axs, MODELS):
        z = BY[(BY["by"] == "block") & (BY["model"] == m) & (BY["arm"] == "FULL") & (BY["rule"] == "R_SLOT")].assign(key=lambda f: f["key"].astype(int)).sort_values("key")
        ax.bar(z["key"], z["mean"], color=[COL["FULL"] if b != 11 else GREY for b in z["key"]], width=.75)
        ax.axhline(0, color=MUTED, lw=.7); ax.set_xticks(range(12)); ax.tick_params(axis="x", labelsize=6); ax.grid(axis="x", visible=False)
        ax.set_title(MLAB[m], loc="left")
    axs[0].set_ylabel("mean ΔCORR"); axs[1].set_xlabel("block (grey bar, 11 = test)")
    out["block"] = G.save(fig_, FK, "by_block.png")
    return out


def arm_table(arms, col_v=True):
    rows = [["arm", "model", "ΔCORR\ndev", "ΔCORR\ntest", "Δpayout\ndev", "Δpayout\ntest", "routed\ndev", "corr with\nB0, dev", "verdict"]]
    for a in arms:
        for m in MODELS:
            r = row(m, a, "dev")
            rows.append([a, m, cell(m, a, "dev"), cell(m, a, "test"), cell(m, a, "dev", "payout_mean", "payout_t"), cell(m, a, "test", "payout_mean", "payout_t"),
                         "%.1f%%" % (r["routed"] * 100), "%.2f" % r["predcorr_b0"], verdict(m, a)])
    return table(rows, left=2, group=1, note_col=[8])


def levels(arms):
    hdr = ["model", "arm", "scope", "CORR", "MMC", "c.sharpe", "dd(corr)", "payout", "p.sharpe", "dd(pay)", "calmar", "burn(d)", "tgt", "n(C)", "n(P)"]
    rows = [hdr]
    for m in MODELS:
        for a, r_ in arms:
            for sc in ("dev", "test"):
                r = LV[(LV["model"] == m) & (LV["arm"] == a) & (LV["rule"] == r_) & (LV["scope"] == sc)].iloc[0]
                rows.append([m, a, sc, num(r["corr"], 4, False), num(r["mmc"], 4, False), num(r["c_sharpe"], 2, False), num(r["dd_corr"], 2, False), num(r["payout"], 4, False), num(r["p_sharpe"], 2, False),
                             num(r["dd_pay"], 2, False), num(r["calmar"], 3, False), int(r["burn"]), "t20", int(r["n"]), int(r["n_pay"])])
    return table(rows, size=6.2, left=3, group=1)


def build():
    figs = figures()
    v = {m: verdict(m, "FULL") for m in MODELS}
    assert set(v.values()) == {"REJECT"} and all(verdict(m, "FULL", r) == "REJECT" for m in MODELS for r in ("R_POOL", "R_HALF"))
    assert verdict("lgb_ktz", "F_RANK") == "PROMISING" and verdict("lgb_ktz", "TSC") == "PROMISING"
    assert verdict("lgb_ktz", "F_RANK", "R_HALF") == "NO IMPROVEMENT"
    assert all(verdict(m, "P_HIST") == "ADOPT" for m in MODELS)
    test = D[(D["rule"] == "R_SLOT") & (D["scope"] == "test") & (D["arm"] != "P_HIST")]
    up, down = int((test["nw_t"] >= 2).sum()), int((test["nw_t"] <= -2).sum())
    st = [P("E135: crypto/v2.0-histfe, routed", TITLE),
          P("Do own-history features beat crypto/v2.0 under the rule we would run live?", SUB)]
    box = Table([[P("<b>Verdict: REJECT for deployment.</b> FULL, the whole 374-feature dataset, is REJECT on all three models under all three routing rules. Dev ΔCORR is %s on ridge, %s on LightGBM and %s on the MLP. "
                    "Two narrow LightGBM subsets reach PROMISING, and neither is a win: both are test-only readings from 24 cells." % (num(row("ridge", "FULL", "dev")["mean"]), num(row("lgb_ktz", "FULL", "dev")["mean"]), num(row("mlp_ktz", "FULL", "dev")["mean"])), VBOX)]], colWidths=[A4[0] - 36 * mm])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(ALT)), ("LINEBEFORE", (0, 0), (0, -1), 2.5, colors.HexColor(COL["FULL"])),
                             ("LEFTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    st += [box, Spacer(1, 8), arm_table(["FULL"]),
           P("Against B0 on all keys, rule R_SLOT. Cells are Δ, with the Newey-West t at lag 20 in brackets. Payout is 0.1 × CORR + proxy MMC, built per period.", CAP)]

    st += [P("1. What was tested", H1),
           P("E135 applies E134's rule and bar to the dataset E133 evaluated. The dev predictions are E133's. E133 never read a date from 2026-02-20, so one new fit covers the final block: train 130, purge 22, predict 130, at the same fixed configs. B0 on that block reproduces E132's to 1.2e-7 on all three models.")]
    st += bullets(["<b>What covered means here:</b> every coin has the 22 base features, so histfe has no feed mask. A coin lacks history features only in warm-up, which is known at the time. Coverage is read from each arm's own columns on that date.",
                   "<b>R_SLOT, the primary rule:</b> at least one of the arm's features is observed. v2.0 ranks every coin, and the candidate reorders covered coins inside the rank slots they already hold.",
                   "<b>R_POOL, sensitivity:</b> the same covered set, with percentiles merged and re-ranked.",
                   "<b>R_HALF, sensitivity:</b> at least half the arm's features are observed. This is the closest match to a new listing still in warm-up.",
                   "<b>Bar, unchanged from E134:</b> ADOPT needs dev CORR t ≥ +2, a positive test delta, and payout not significantly negative on either window. REJECT is dev t ≤ −2. PROMISING is test t ≥ +2 without a dev REJECT. Anything else is NO IMPROVEMENT.",
                   "<b>Primary cells:</b> FULL on three models. The other seven arms are secondary. ADDED_ONLY was left out because it drops the base features."])
    half = row("ridge", "FULL", "dev", "R_HALF")["routed"] * 100
    st += [P("<b>Routing has little to do here.</b> Under R_SLOT only %.1f%% of FULL's keys take the fallback, and %.1f%% under R_HALF. The E134 gain came from coins with no feed, and histfe has none." % (row("ridge", "FULL", "dev")["routed"] * 100, half)),
           P("2. What routing changed against E133", H1)]
    rows = [["model", "E133 plain\ndev", "E135 R_SLOT\ndev", "E135 R_HALF\ndev", "E135 R_SLOT\ntest", "E135 R_HALF\ntest"]]
    for m in MODELS:
        e = E133[(E133["model"] == m) & (E133["arm"] == "FULL") & (E133["vs"] == "B0") & (E133["metric"] == "corr")].iloc[0]
        rows.append([m, "%s (%s)" % (num(e["delta"]), num(e["t"], 1)), cell(m, "FULL", "dev"), cell(m, "FULL", "dev", rule="R_HALF"), cell(m, "FULL", "test"), cell(m, "FULL", "test", rule="R_HALF")])
    st += [table(rows, left=1), P("ΔCORR of FULL against B0 (NW t). E133's column has no test reading.", CAP), fig(figs["delta"], "Cumulative ΔCORR against B0, R_SLOT. The dotted line starts the test block.")]

    st += [P("3. Horizons and feature families", H1), arm_table(HOR + FAM),
           P("Same layout as above. These are secondary cells. Across the 24 non-control cells on test, %d reach t ≥ +2 and %d reach t ≤ −2." % (up, down), CAP)]
    fr_d, fr_t, tc_t = row("lgb_ktz", "F_RANK", "dev"), row("lgb_ktz", "F_RANK", "test"), row("lgb_ktz", "TSC", "test")
    lo = SD[(SD["model"] == "lgb_ktz") & (SD["arm"] == "F_RANK") & (SD["rule"] == "R_SLOT") & (SD["scope"] == "dev")]
    st += bullets(["<b>lgb F_RANK is PROMISING by a thin margin.</b> Its dev t is %s, just inside the REJECT line, and %d of 5 seeds cross −2 alone. Its test t is %s under R_SLOT and %s under R_HALF, which drops it to NO IMPROVEMENT." % (num(fr_d["nw_t"], 2), int((lo["nw_t"] <= -2).sum()), num(fr_t["nw_t"], 2), num(row("lgb_ktz", "F_RANK", "test", "R_HALF")["nw_t"], 2)),
                   "<b>lgb F_RANK's payout is the better reading.</b> It is %s on dev and %s on test, the same sign on both windows, and its gain comes through MMC rather than CORR." % (cell("lgb_ktz", "F_RANK", "dev", "payout_mean", "payout_t"), cell("lgb_ktz", "F_RANK", "test", "payout_mean", "payout_t")),
                   "<b>lgb TSC, days since change, is PROMISING on test only.</b> Dev is flat at %s, test is %s, and test payout is not significant at %s. The test delta is positive on all three models, but that is one six-month window, and ridge and the MLP are REJECT on dev." % (cell("lgb_ktz", "TSC", "dev"), num(tc_t["mean"]) + " (" + num(tc_t["nw_t"], 1) + ")", cell("lgb_ktz", "TSC", "test", "payout_mean", "payout_t")),
                   "<b>Neither clears multiplicity.</b> A Bonferroni bar over 24 cells is about |t| 3.1."])

    st += [P("4. The tenure control", H1), arm_table(["P_HIST"]),
           P("Tenure exists for every coin, so nothing is routed. It is a control, not part of histfe.", CAP)]
    st += [P("P_HIST meets the ADOPT bar by its letter on all three models, but do not read that as confirmed. Its dev gain is real, with t from +2.6 to +4.8, yet its test deltas are between +0.0006 and +0.0019, all with |t| below 0.5. The bar asks only for a positive sign on test, and that is close to no evidence. LightGBM's test payout is negative at %s, and its seeds split 3 to 2 on test." % cell("lgb_ktz", "P_HIST", "test", "payout_mean", "payout_t"))]
    rows = [["model", "arm", "ΔCORR dev", "ΔCORR test", "Δpayout dev", "Δpayout test"]]
    for m in MODELS:
        for a in ("FULL", "F_RANK"):
            g = VT[(VT["model"] == m) & (VT["arm"] == a) & (VT["rule"] == "R_SLOT")].set_index("scope")
            c = lambda sc, x, y: "%s (%s)" % (num(g.loc[sc, x]), num(g.loc[sc, y], 1))  # noqa: E731
            rows.append([m, a, c("dev", "mean", "nw_t"), c("test", "mean", "nw_t"), c("dev", "payout_mean", "payout_t"), c("test", "payout_mean", "payout_t")])
    st += [table(rows, left=2, group=1), P("Routed arm against P_HIST, R_SLOT (NW t). lgb F_RANK ties tenure on dev payout, as in E133, and is ahead on test without reaching t 2.", CAP)]

    st += [P("5. Levels", H1), levels([("B0", "-"), ("FULL", "R_SLOT"), ("F_RANK", "R_SLOT"), ("TSC", "R_SLOT"), ("P_HIST", "R_SLOT")]),
           P("MMC is a proxy, not official. MMC, payout, p.sharpe, dd(pay), calmar and burn(d) cover n(P) dates. Compare calmar and burn(d) only within a scope.", CAP)]
    lg = LV[LV["model"] == "lgb_ktz"].set_index(["arm", "rule", "scope"])
    st += [P("<b>One lead:</b> lgb F_RANK's calmar beats B0 on both windows, %.3f against %.3f on dev and %.3f against %.3f on test. E134's calmar lead failed on test, and this one does not. It is still no better than tenure on dev (%.3f), and calmar has no significance test here." % (
        lg.loc[("F_RANK", "R_SLOT", "dev"), "calmar"], lg.loc[("B0", "-", "dev"), "calmar"], lg.loc[("F_RANK", "R_SLOT", "test"), "calmar"], lg.loc[("B0", "-", "test"), "calmar"], lg.loc[("P_HIST", "R_SLOT", "dev"), "calmar"])),
           fig(figs["pay"], "Cumulative payout with proxy MMC, R_SLOT.")]

    st += [P("6. Sensitivity to the rule", H1)]
    for mean, t, cap in (("mean", "nw_t", "ΔCORR against B0 (NW t)."), ("payout_mean", "payout_t", "Δpayout against B0 with proxy MMC (NW t).")):
        rows = [["arm", "model", "R_SLOT dev", "R_SLOT test", "R_POOL dev", "R_POOL test", "R_HALF dev", "R_HALF test"]]
        for a in ("FULL", "F_RANK", "TSC"):
            for m in MODELS:
                rows.append([a, m] + [cell(m, a, sc, mean, t, r) for r in ("R_SLOT", "R_POOL", "R_HALF") for sc in ("dev", "test")])
        st += [table(rows, size=6.4, left=2, group=1), P(cap, CAP)]

    st += [P("7. Equal-budget retune", H1), P("B0 and FULL retuned under E133's grid, one seed, picked per block on the last 40 train dates after a 22-date purge. This is the small grid E133 used, not E134's 50 draws.")]
    rows = [["model", "scope", "routed", "tuned B0", "ΔCORR", "NW t", "bootstrap 95% CI", "Δpayout (t)", "n"]]
    for m in MODELS:
        for r in TU[TU["model"] == m].itertuples():
            rows.append([r.model, r.scope, num(r.arm_corr, 4, False), num(r.b0_corr, 4, False), num(r.mean), num(r.nw_t, 1), "%s, %s" % (num(r.boot_lo), num(r.boot_hi)), "%s (%s)" % (num(r.payout_mean), num(r.payout_t, 1)), r.n])
    st += [table(rows, left=2, group=1), P("Giving FULL its own hyper-parameters does not rescue it. Every ΔCORR is negative.", CAP)]

    st += [KeepTogether([P("8. Stability", H1), fig(figs["block"], "Mean ΔCORR of routed FULL against B0 by block.")])]
    g = BY[(BY["by"] == "year") & (BY["rule"] == "R_SLOT") & BY["arm"].isin(["FULL", "F_RANK", "TSC"])].assign(key=lambda f: f["key"].astype(int))
    p = g.pivot_table(index=["arm", "model"], columns="key", values="mean", sort=False)
    rows = [["arm", "model"] + [str(x) for x in p.columns]]
    for (a, m), r in p.iterrows():
        rows.append([a, m] + [num(x, 3) for x in r.values])
    st += [P("By year", H2), table(rows, left=2, group=1), P("By seed", H2)]
    rows = [["arm", "model", "scope", "lowest seed", "highest seed", "seeds below 0", "seeds at |t| ≥ 2"]]
    for a in ("FULL", "F_RANK", "TSC"):
        for m in MODELS[1:]:
            for sc in ("dev", "test"):
                z = SD[(SD["model"] == m) & (SD["arm"] == a) & (SD["scope"] == sc) & (SD["rule"] == "R_SLOT")]
                rows.append([a, m, sc, num(z["mean"].min()), num(z["mean"].max()), "%d of %d" % ((z["mean"] < 0).sum(), len(z)), "%d of %d" % ((z["nw_t"].abs() >= 2).sum(), len(z))])
    st += [table(rows, left=3, group=1)]

    st += [P("9. Caveats", H1)]
    st += bullets(["E133 kept the final block unread. E135 spent it, because E134's bar needs it. No histfe reading on these dates existed before, but E65, E131, E132 and E134 had scored the same dates on other features, so it is not virgin. No arm was chosen from it.",
                   "That was the last untouched histfe window. The two PROMISING cells cannot be confirmed on it, only on later dates.",
                   "Missingness here is causal warm-up, so E134's survivorship caveat does not apply to the routing.",
                   "Routed models were still trained with warm-up cells at 0.5. Training only on covered rows was not tested.",
                   "mlp_ktz is linear. MMC is a proxy. No catboost or forest arm was tested.",
                   "Blends of B0 with the added block were not re-scored. E133 covers them on dev only."])
    st += [P("10. Next steps", H1)]
    st += bullets(["Keep live slots on the existing book. Do not deploy crypto/v2.0-histfe.",
                   "Do not stake or promote lgb F_RANK or TSC on this evidence. If either is wanted, track it as an unstaked candidate on new dates.",
                   "Give tenure its own study, with the seeds and later dates it needs, instead of adopting it from a control arm."])

    path = ROOT / FILE

    class C(G.Numbered):
        pass
    C.head = "E135 · crypto/v2.0-histfe, routed · REJECT"
    doc = G.Doc(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm, bottomMargin=14 * mm, title="E135 histfe routed: REJECT", author="my-numerai-agent", subject="Feature set evaluation under the live routing rule")
    orig = doc.handle_documentBegin

    def begin():
        doc.seq = 0
        orig()
    doc.handle_documentBegin = begin
    doc.build(st, canvasmaker=C)
    print(path, os.path.getsize(path), up, down)


if __name__ == "__main__":
    build()
