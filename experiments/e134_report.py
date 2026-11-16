#!/usr/bin/env python3
"""E134 PDFs: one report per feature set, from e134_route_feature_sets.py's tables.

    python3 experiments/e134_report.py
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
import e132_pdf as G  # noqa: E402  fonts, styles, table and figure helpers, and E132's scores
from e132_pdf import ALT, CAP, DARK, GREY, H1, H2, MLAB, MODELS, MUTED, P, SUB, TITLE, VBOX, bullets, fig, mono, num, table  # noqa: E402

os.chdir(REPO)
ROOT = pathlib.Path("results/reports/crypto/e134_route")
FIG = ROOT / "figures"
D = pd.read_csv(ROOT / "paired_deltas.csv")
LV = pd.read_csv(ROOT / "summary_levels.csv")
TU = pd.read_csv(ROOT / "tuned_equal_budget.csv")
BY = pd.read_csv(ROOT / "paired_deltas_by_block_year.csv")
SD = pd.read_csv(ROOT / "paired_deltas_by_seed.csv")
S = pd.read_parquet(ROOT / "scores_by_date_seed.parquet")
B = pd.read_parquet(ROOT / "scores_b0.parquet")
NOIND = {"L": "L_M1", "S": "S_M1", "C": "C_RAW"}
FILE = {"L": "E134_litmom_feed_fillind_routed_REJECT.pdf", "S": "E134_momswap_feed_fillind_routed_REJECT.pdf",
        "C": "E134_momswap_macro_carver_feed_fillind_routed_REJECT.pdf"}


def row(m, a, sc, rule="R_SLOT"):
    return D[(D["model"] == m) & (D["arm"] == a) & (D["scope"] == sc) & (D["rule"] == rule)].iloc[0]


def cell(m, a, sc, mean="mean", t="nw_t", rule="R_SLOT"):
    r = row(m, a, sc, rule)
    return "%s (%s)" % (num(r[mean]), num(r[t], 1))


def verdict(m, a):
    """The bar registered before scoring."""
    d, t = row(m, a, "dev"), row(m, a, "test")
    if d["nw_t"] <= -2:
        return "REJECT"
    if d["nw_t"] >= 2 and t["mean"] > 0 and d["payout_t"] > -2 and t["payout_t"] > -2:
        return "ADOPT"
    if t["nw_t"] >= 2:
        return "PROMISING"
    return "NO IMPROVEMENT"


def cum(m, a):
    x = S[(S["model"] == m) & (S["arm"] == a) & (S["rule"] == "R_SLOT")].groupby("date")[["corr", "payout"]].mean()
    return x


def FK(k):
    return "../../e134_route/figures/" + k


def figures(k):
    full, ni, col = k + "_FULL", NOIND[k], G.CFG[k]["color"]
    out = {}
    fig_, axs = G.panel()
    for ax, m in zip(axs, MODELS):
        b = B[B["model"] == m].groupby("date")["corr"].mean()
        ax.plot((G.series(m, full) - b).cumsum(), color=GREY, ls="--", label="E132: %s, no-feed coins at 0.5" % full)
        ax.plot((cum(m, ni)["corr"] - b).cumsum(), color=DARK, label="routed %s" % ni)
        ax.plot((cum(m, full)["corr"] - b).cumsum(), color=col, label="routed %s" % full)
        ax.axhline(0, color=MUTED, lw=.7); G.test_mark(ax); G.years(ax)
    axs[0].set_ylabel("cumulative ΔCORR vs B0")
    out["delta"] = G.save(fig_, FK(k), "cum_delta.png")
    fig_, axs = G.panel()
    for ax, m in zip(axs, MODELS):
        b = B[B["model"] == m].groupby("date")["payout"].mean().dropna()
        ax.plot(b.cumsum(), color=DARK, ls="--", label="B0 raw v2.0")
        ax.plot(cum(m, full)["payout"].dropna().cumsum(), color=col, label="routed %s" % full)
        G.test_mark(ax)
        ax.xaxis.set_major_locator(G.matplotlib.dates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(G.matplotlib.dates.DateFormatter("%b %y"))
        ax.tick_params(axis="x", labelsize=6.5)
    axs[0].set_ylabel("cumulative payout (proxy)")
    out["pay"] = G.save(fig_, FK(k), "cum_payout.png")
    fig_, axs = plt.subplots(1, 3, figsize=(7.3, 2.0), sharey=True)
    for ax, m in zip(axs, MODELS):
        z = BY[(BY["by"] == "block") & (BY["model"] == m) & (BY["arm"] == full) & (BY["rule"] == "R_SLOT")].assign(key=lambda f: f["key"].astype(int)).sort_values("key")
        ax.bar(z["key"], z["mean"], color=[col if b != 11 else GREY for b in z["key"]], width=.75)
        ax.axhline(0, color=MUTED, lw=.7); ax.set_xticks(range(12)); ax.tick_params(axis="x", labelsize=6); ax.grid(axis="x", visible=False)
        ax.set_title(MLAB[m], loc="left")
    axs[0].set_ylabel("mean ΔCORR"); axs[1].set_xlabel("block (grey bar, 11 = test)")
    out["block"] = G.save(fig_, FK(k), "by_block.png")
    return out


def build(k):
    c = G.CFG[k]
    full, ni, col = k + "_FULL", NOIND[k], c["color"]
    arms = [full, ni]
    figs = figures(k)
    st = [P("E134: %s, routed" % c["name"], TITLE),
          P("Does %s beat crypto/v2.0 under the rule we would run live?" % mono(c["data"]), SUB)]
    v = {(m, a): verdict(m, a) for m in MODELS for a in arms}
    assert not any(x in ("ADOPT", "PROMISING") for x in v.values())
    best = max(row(m, a, "dev")["mean"] for m in MODELS for a in arms)
    worst = min(row(m, a, "dev")["mean"] for m in MODELS for a in arms)
    box = Table([[P("<b>Verdict: REJECT for deployment.</b> Routing removes much of E132's loss, but nothing here beats v2.0. Dev ΔCORR runs from %s to %s. "
                    "Ridge and the MLP are REJECT on dev, LightGBM is NO IMPROVEMENT, and no cell reaches the ADOPT or PROMISING bar." % (num(best), num(worst)), VBOX)]], colWidths=[A4[0] - 36 * mm])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(ALT)), ("LINEBEFORE", (0, 0), (0, -1), 2.5, colors.HexColor(col)),
                             ("LEFTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    st += [box, Spacer(1, 8)]
    rows = [["arm", "model", "ΔCORR\ndev", "ΔCORR\ntest", "Δpayout\ndev", "Δpayout\ntest", "routed\ncorr with B0, dev", "verdict"]]
    for a in arms:
        for m in MODELS:
            rows.append([a, m, cell(m, a, "dev"), cell(m, a, "test"), cell(m, a, "dev", "payout_mean", "payout_t"), cell(m, a, "test", "payout_mean", "payout_t"), "%.2f" % row(m, a, "dev")["predcorr_b0"], v[(m, a)]])
    st += [table(rows, left=2, group=1, note_col=[7]),
           P("Against B0 on all keys, rule R_SLOT. Cells are Δ, with the Newey-West t at lag 20 in brackets. Payout is 0.1 × CORR + proxy MMC, built per period. %s is the same model without missing-data indicators." % ni, CAP)]

    st += [P("1. What was tested", H1),
           P("E132 gave a coin with no features a 0.5 prediction. Live, the v2.0 model scores every coin anyway, so that coin would keep its v2.0 prediction. E134 scores that rule. No model was refitted: it reuses E132's saved predictions, paired by seed.")]
    st += bullets(["<b>R_SLOT, the primary rule:</b> v2.0 ranks every coin on each date. The richer model reorders only the coins in its dataset, inside the rank slots those coins already hold. A no-feed coin keeps its exact v2.0 rank, so the two arms can differ only on covered coins.",
                   "<b>R_POOL, sensitivity:</b> each model's percentiles are merged and re-ranked, so an uncovered coin can move.",
                   "<b>R_HALF, sensitivity:</b> R_SLOT, but a coin also goes to v2.0 if fewer than half the proposed features are observed that date. This is the closest match to a new listing with no history.",
                   "<b>Baseline:</b> B0, raw v2.0 on every key, which reproduces E132's B0 exactly.",
                   "<b>Bar, fixed before scoring:</b> ADOPT needs dev CORR t ≥ +2, a positive test delta, and payout not significantly negative on either window. REJECT is dev t ≤ −2. PROMISING is test t ≥ +2 without a dev REJECT. Anything else is NO IMPROVEMENT. There are 18 primary cells across the three reports, so one or two |t| ≥ 2 results are not surprising."])
    routed = D[(D["rule"] == "R_SLOT") & (D["scope"] == "dev")]["routed"].mean() * 100
    half = D[(D["rule"] == "R_HALF") & (D["scope"] == "dev")]["routed"].mean() * 100
    st += [P("On dev, %.1f%% of coins take the v2.0 fallback under R_SLOT and %.1f%% under R_HALF. Live feed coverage is 298 of 300, about 0.7%%, so the backtest routes more coins than today's live universe would." % (routed, half))]

    st += [P("2. What routing changed against E132", H1)]
    rows = [["model", "E132 at 0.5\ndev", "E134 routed\ndev", "E132 at 0.5\ntest", "E134 routed\ntest"]]
    for m in MODELS:
        rows.append([m, G.cell(m, full + " - B0", "dev"), cell(m, full, "dev"), G.cell(m, full + " - B0", "test"), cell(m, full, "test")])
    st += [table(rows, left=1), P("ΔCORR of %s against B0 (NW t)." % full, CAP), fig(figs["delta"], "Cumulative ΔCORR against B0. The dotted line starts the test block. Grey dashes are E132's rule.")]

    st += [P("3. Levels", H1)]
    hdr = ["model", "arm", "scope", "CORR", "MMC", "c.sharpe", "dd(corr)", "payout", "p.sharpe", "dd(pay)", "calmar", "burn(d)", "tgt", "n(C)", "n(P)"]
    rows = [hdr]
    for m in MODELS:
        for a, r_ in (("B0", "-"), (full, "R_SLOT"), (ni, "R_SLOT")):
            for sc in ("dev", "test"):
                r = LV[(LV["model"] == m) & (LV["arm"] == a) & (LV["rule"] == r_) & (LV["scope"] == sc)].iloc[0]
                rows.append([m, a, sc, num(r["corr"], 4, False), num(r["mmc"], 4, False), num(r["c_sharpe"], 2, False), num(r["dd_corr"], 2, False), num(r["payout"], 4, False), num(r["p_sharpe"], 2, False),
                             num(r["dd_pay"], 2, False), num(r["calmar"], 3, False), int(r["burn"]), "t20", int(r["n"]), int(r["n_pay"])])
    st += [table(rows, size=6.2, left=3, group=1), P("MMC is a proxy, not official. MMC, payout, p.sharpe, dd(pay), calmar and burn(d) cover n(P) dates. Compare calmar and burn(d) only within a scope.", CAP)]
    lg = LV[(LV["model"] == "lgb_ktz")].set_index(["arm", "rule", "scope"])
    a0, a1, t0, t1 = lg.loc[("B0", "-", "dev"), "calmar"], lg.loc[(full, "R_SLOT", "dev"), "calmar"], lg.loc[("B0", "-", "test"), "calmar"], lg.loc[(full, "R_SLOT", "test"), "calmar"]
    st += [P("<b>One lead, not a win:</b> LightGBM's %s has a better dev calmar than B0, %.3f against %.3f, because its payout drawdown is shallower. On test it is %.3f against %.3f, so this does not hold, and calmar has no significance test here." % (full, a1, a0, t1, t0)),
           fig(figs["pay"], "Cumulative payout with proxy MMC.")]

    st += [P("4. Do the features help on the coins they cover?", H1),
           P("This is the features-only question, on covered coins alone. \"vs B0\" uses the v2.0 model fitted on every coin. \"vs B0F\" uses v2.0 fitted on the feed rows only, so it removes the effect of training without the no-feed coins.")]
    rows = [["arm", "model", "vs B0, dev", "vs B0, test", "vs B0F, dev", "vs B0F, test"]]
    for a in arms:
        for m in MODELS:
            rows.append([a, m, cell(m, a, "dev", "cov_mean", "cov_t"), cell(m, a, "test", "cov_mean", "cov_t"), cell(m, a, "dev", "covf_mean", "covf_t"), cell(m, a, "test", "covf_mean", "covf_t")])
    st += [table(rows, left=2, group=1), P("ΔCORR on covered coins, same rows in both legs (NW t).", CAP)]

    st += [P("5. Sensitivity to the rule", H1)]
    rows = [["arm", "model", "R_SLOT dev", "R_SLOT test", "R_POOL dev", "R_POOL test", "R_HALF dev", "R_HALF test"]]
    for a in arms:
        for m in MODELS:
            rows.append([a, m] + [cell(m, a, sc, rule=r) for r in ("R_SLOT", "R_POOL", "R_HALF") for sc in ("dev", "test")])
    st += [table(rows, size=6.4, left=2, group=1), P("ΔCORR against B0 (NW t).", CAP)]
    rows = [["arm", "model", "R_SLOT dev", "R_SLOT test", "R_POOL dev", "R_POOL test", "R_HALF dev", "R_HALF test"]]
    for a in arms:
        for m in MODELS:
            rows.append([a, m] + [cell(m, a, sc, "payout_mean", "payout_t", r) for r in ("R_SLOT", "R_POOL", "R_HALF") for sc in ("dev", "test")])
    st += [table(rows, size=6.4, left=2, group=1), P("Δpayout against B0 with proxy MMC (NW t).", CAP)]

    st += [P("6. Equal-budget retune", H1), P("%s retuned with 50 draws and routed onto E131's tuned B0, which had the same budget. Pre-test covers 2024-08 to 2026-02-19." % full)]
    rows = [["model", "scope", "routed", "tuned B0", "ΔCORR", "NW t", "bootstrap 95% CI", "Δpayout (t)", "n"]]
    for m in MODELS:
        for r in TU[(TU["dataset"] == k) & (TU["model"] == m)].itertuples():
            rows.append([r.model, r.scope, num(r.arm_corr, 4, False), num(r.b0_corr, 4, False), num(r.mean), num(r.nw_t, 1), "%s, %s" % (num(r.boot_lo), num(r.boot_hi)), "%s (%s)" % (num(r.payout_mean), num(r.payout_t, 1)), r.n])
    st += [table(rows, left=2, group=1)]

    st += [KeepTogether([P("7. Stability", H1), fig(figs["block"], "Mean ΔCORR of routed %s against B0 by block." % full)])]
    g = BY[(BY["by"] == "year") & (BY["rule"] == "R_SLOT") & BY["arm"].isin(arms)].assign(key=lambda f: f["key"].astype(int))
    p = g.pivot_table(index=["arm", "model"], columns="key", values="mean", sort=False)
    rows = [["arm", "model"] + [str(x) for x in p.columns]]
    for (a, m), r in p.iterrows():
        rows.append([a, m] + [num(x, 3) for x in r.values])
    st += [P("By year", H2), table(rows, left=2, group=1), P("By seed", H2)]
    rows = [["arm", "model", "scope", "lowest seed", "highest seed", "seeds below 0", "seeds at |t| ≥ 2"]]
    for a in arms:
        for m in MODELS[1:]:
            for sc in ("dev", "test"):
                z = SD[(SD["model"] == m) & (SD["arm"] == a) & (SD["scope"] == sc) & (SD["rule"] == "R_SLOT")]
                rows.append([a, m, sc, num(z["mean"].min()), num(z["mean"].max()), "%d of %d" % ((z["mean"] < 0).sum(), len(z)), "%d of %d" % ((z["nw_t"].abs() >= 2).sum(), len(z))])
    st += [table(rows, left=3, group=1)]

    st += [P("8. Caveats", H1)]
    st += bullets(["The covered set is still chosen by today's feed, so it leans towards coins that survived. A no-feed coin cannot flatter the candidate under R_SLOT, but live the candidate would also face coins that later die, so these numbers may be generous to it.",
                   "The richer models were trained without the no-feed coins, as they would be live.",
                   "The test block has now been scored by E65, E131, E132 and this study. It chose nothing here, but it is not virgin.",
                   "About 25 kept coins still carry wrong-coin Yahoo prices.",
                   "mlp_ktz is linear. MMC is a proxy. No catboost or forest arm was tested.",
                   "Passing all-0.5 proposed features through the richer model was not tested, because the fitted models were not saved and it would need a refit.",
                   "Only the rule was evaluated. These feature sets are not wired into the live pipeline."])
    st += [P("9. Next steps", H1)]
    st += bullets(["Keep live slots on the existing book. Do not deploy this dataset.",
                   "Use routing as the standard rule for scoring any future dataset that cannot cover every coin.",
                   "Fix the wrong-coin prices before spending more time on this feature set."])

    path = ROOT / FILE[k]
    head = "E134 · %s, routed · REJECT" % c["name"]

    class C(G.Numbered):
        pass
    C.head = head
    doc = G.Doc(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm, bottomMargin=14 * mm, title="E134 %s routed: REJECT" % c["name"], author="my-numerai-agent", subject="Feature set evaluation under the live routing rule")
    orig = doc.handle_documentBegin

    def begin():
        doc.seq = 0
        orig()
    doc.handle_documentBegin = begin
    doc.build(st, canvasmaker=C)
    print(path, os.path.getsize(path), {x: list(v.values()).count(x) for x in set(v.values())})


if __name__ == "__main__":
    for k in sys.argv[1:] or list(G.CFG):
        build(k)
