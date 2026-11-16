#!/usr/bin/env python3
"""E138/E139/E140 report: one self-contained HTML file per study, then printed to PDF.

    python3 experiments/e138_report.py E138        # or E139, E140
    python3 -m src.analysis.html_pdf <the html it prints>

Every number comes from the CSVs the study's runner wrote. The verdict is computed, never typed: the
same bar as E134-E137 (REJECT dev CORR t <= -2; ADOPT dev t >= +2, test delta > 0, payout t > -2 on
both windows; PROMISING test t >= +2; else NO IMPROVEMENT), read on the ROUTED R_SLOT score for arms
that route and on the plain score for arms that cannot (ADDED_ONLY and the controls).
The study's own reading is passed in READING, written after the numbers were seen, and says so.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e131_analysis as E  # noqa: E402
import e133_report as W  # noqa: E402  CSS, crosshair script, table, tint, line_panels

os.chdir(REPO)
MODELS = ["ridge", "lgb_ktz", "mlp_ktz"]
NAME = {"ridge": "ridge", "lgb_ktz": "LightGBM", "mlp_ktz": "MLP (linear)"}
_table = W.table
W.table = lambda head, rows, cls="": _table(head, [[str(c) for c in r] for r in rows], cls)
HDR = ["arm", "model", "ΔCORR dev", "ΔCORR test", "Δpayout dev", "Δpayout test", "verdict"]

STUDIES = {
    "E138": dict(root="results/reports/crypto/e138_carver_macro", slug="carver_macro", dataset="crypto/v2.0-carver-macro", nfeat=55, parent=None,
                 title="Do Carver trend signals and macro features help on Numerai Crypto?",
                 what="all 22 shipped features + 15 Carver / Clenow trend signals (<code>feature_tf_</code>) + 18 shared macro features (<code>feature_macro_</code>)",
                 builder="src/data_mani/build_carver_macro.py", hosts="hamilton, hopper and lovelace",
                 blocks=["F_TF", "F_MACRO", "ADDED_ONLY"], ablate=["NO_EWMAC", "NO_BREAKOUT", "NO_BBZ", "NO_CLENOW", "NO_RATES", "NO_RISK"],
                 series=["FULL", "F_TF", "F_MACRO", "P_HIST"], fits=None),
    "E139": dict(root="results/reports/crypto/e139_carver_macro_infl", slug="carver_macro_infl", dataset="crypto/v2.0-carver-macro-infl", nfeat=67, parent="CM",
                 title="Does break-even inflation add to carver-macro on Numerai Crypto?",
                 what="E138&#39;s 55 features + 12 shared break-even inflation features (4 levels, 8 changes over 20/60 business days)",
                 builder="src/data_mani/build_inflation.py onto crypto/v2.0-carver-macro", hosts="dostoievski and tolstoi",
                 blocks=["CM", "F_INFL", "ADDED_ONLY"], ablate=["NO_BEILVL", "NO_BEICHG"], series=["FULL", "CM", "F_INFL", "P_HIST"], fits=None),
    "E140": dict(root="results/reports/crypto/e140_carver_macro_infl_xinteractions", slug="carver_macro_infl_xinteractions", dataset="crypto/v2.0-carver-macro-infl-xinteractions", nfeat=337, parent="CMI",
                 title="Do trend x macro interactions add to carver-macro-infl on Numerai Crypto?",
                 what="E139&#39;s 67 features + 270 interactions: each raw trend signal times each signed macro momentum (10 metals / S&amp;P / DXY returns, 8 break-even changes), binned per date after the product",
                 builder="src/data_mani/build_trend_xinteractions.py", hosts="deepfour and deepsix",
                 blocks=["CMI", "F_IX", "ADDED_ONLY"], ablate=["IX_RET", "IX_BEI"], series=["FULL", "CMI", "F_IX", "P_HIST"], fits=None),
    "E141": dict(root="results/reports/crypto/e141_macrobeta", slug="macrobeta", dataset="crypto/v2.0-macrobeta", nfeat=54, parent=None,
                 title="Do coins' rolling betas to macro and inflation factors help on Numerai Crypto?",
                 what="all 22 shipped features + 32 causal rolling betas (60 and 120 macro business days) of each coin&#39;s log return on the one-day change of 16 factors: gold, silver, S&amp;P, DXY, the 1/2/5/10/30-year yields, 3 curve slopes and 4 break-evens",
                 builder="src/data_mani/build_macro_betas.py", hosts="hamilton, hopper and lovelace", block="beta block",
                 blocks=["F_PRICE", "F_RATES", "F_INFL", "W60", "W120", "ADDED_ONLY"], ablate=["NO_PRICE", "NO_RATES", "NO_INFL"], series=["FULL", "F_PRICE", "F_RATES", "F_INFL", "P_HIST"], fits=None),
    "E143": dict(root="results/reports/crypto/e143_c2syn", slug="c2syn", dataset="crypto/v2.0-c2syn", nfeat=54, parent=None,
                 title="Do Chronos-2-Synth representations of each coin&#39;s history help on Numerai Crypto?",
                 what="all 22 shipped features + 32 PCs of the encoder state Chronos-2-Synth&#39;s forecasting head reads for date D: "
                      "16 from each coin&#39;s label series with its 22 features as covariates (cov), 16 from the label series alone (ar); "
                      "labels only from dates at least 35 days old, zero-shot, rank-binned per date",
                 builder="src/data_mani/chronos2_syn_repr.py + src/data_mani/build_c2syn.py", hosts="deepfive (GPU 1) and the CPU boxes",
                 block="representation block",
                 blocks=["F_COV", "F_AR", "ADDED_ONLY"], ablate=["F_COV", "F_AR"], series=["FULL", "F_COV", "F_AR", "P_HIST"], fits=None,
                 missing="A row has NaN in the new block when the coin has fewer than 32 resolved labels in its 256-step context, or falls in the first 60 dates that fitted the PCA basis: the file keeps the NaN and the loader fills 0.5. No indicator is added and no row is dropped. Nothing about a missing state depends on a feed fetched today: it is warm-up, known at the time.",
                 live_gap="The live path does not build this dataset: it needs the chronos2 venv and a GPU (or a slow CPU pass) at every round, neither wired into <code>src/live</code>."),
    "E144": dict(root="results/reports/crypto/e144_c2ft", slug="c2ft", dataset="crypto/v2.0-c2ft", nfeat=54, parent=None,
                 title="Do fine-tuned Chronos-2-Synth representations of each coin&#39;s history help on Numerai Crypto?",
                 what="all 22 shipped features + 32 PCs of the encoder state a LoRA fine-tuned Chronos-2-Synth&#39;s forecasting head reads for date D: "
                      "16 from the label with its 22 features as covariates (cov), 16 from the label alone (ar). One fine-tuned model pair per walk-forward block, "
                      "trained on the panel cut at the block&#39;s first train date, so every label it saw resolved before the block&#39;s training window",
                 builder="src/data_mani/chronos2_ft_repr.py + src/data_mani/build_c2ft.py", hosts="deepfive (GPU 1) and the CPU boxes",
                 block="representation block",
                 blocks=["F_COV", "F_AR", "ADDED_ONLY"], ablate=["F_COV", "F_AR"], series=["FULL", "F_COV", "F_AR", "P_HIST"], fits=None,
                 missing="The features differ by block: each block&#39;s train and test rows carry the PCs of that block&#39;s own fine-tuned models, on a PCA basis fitted on its train rows, so a row in block b&#39;s test window and block b+1&#39;s train window has two different values (data/crypto/v2.0-c2ft/blocks.parquet). Blocks 0 and 1 have no model: block 0 has no label before its train window, and block 1 (cut 2020-07-29) has about six months, short of the fine-tune&#39;s 134-step minimum, so no coin qualifies. Their rows are 0.5 and routed to v2.0, so the fine-tuned features start on 2021-08-27. Otherwise a row is NaN when the coin has fewer than 32 resolved labels in its context; the loader fills 0.5, no indicator, no row dropped.",
                 unobserved="warm-up, or blocks 0 and 1, which have no fine-tuned model",
                 live_gap="The live path does not build this dataset: it needs the chronos2 venv, a GPU and a LoRA fine-tune at every refit, none wired into <code>src/live</code>."),
    "E146": dict(root="results/reports/crypto/e146_c2dims", slug="c2dims", dataset="crypto/v2.0-c2syn-c2ft-dims", nfeat=3094, parent=None,
                 title="Do all 3,072 hidden dimensions of the Chronos-2-Synth states help on Numerai Crypto?",
                 what="all 22 shipped features + every one of the 768 dimensions of four Chronos-2-Synth encoder states, each rank-binned: "
                      "zero-shot with the 22 features as covariates (c2cov) and on the label alone (c2ar), and the same two from a LoRA fine-tune per walk-forward block (c2ftcov, c2ftar). "
                      "E143 and E144 kept only 16 PCs of each",
                 builder="src/data_mani/build_c2syn_c2ft.py --raw --no-pc", hosts="hamilton, hopper, lovelace, dostoievski, tolstoi, deepfour, deepfive and deepsix",
                 block="state block",
                 blocks=["F_ZS", "F_FT", "F_C2COV", "F_C2AR", "F_C2FTCOV", "F_C2FTAR", "ADDED_ONLY"], ablate=["F_ZS", "F_FT"], series=["FULL", "F_ZS", "F_FT", "P_HIST"], fits=None,
                 missing="Zero-shot dims are binned within date and are the same in every block; fine-tuned dims are binned within (block, date) and differ by block, so each block is fitted on its own values from blocks.parquet. Blocks 0 and 1 have no fine-tuned model (0.5, routed to v2.0). A row is NaN where the coin has too few resolved labels for a state; the loader fills 0.5, no indicator, no row dropped. The MLP&#39;s tuned B0 is E144&#39;s (same grid, same rows). Skipped and named at registration: FULL minus each shipped group and FULL minus each single state, each over 3,000 columns.",
                 unobserved="warm-up, or blocks 0 and 1 for the fine-tuned states",
                 live_gap="The live path does not build this dataset: it needs the chronos2 venv, a GPU and a LoRA fine-tune per refit, none wired into <code>src/live</code>; and 3,072 extra columns per round."),
    "E148": dict(root="results/reports/crypto/e148_c2resid", slug="c2resid", dataset="crypto/v2.0-c2resid", nfeat=28, parent=None,
                 title="Does a Chronos-2 forecast of the residual return path help on Numerai Crypto?",
                 what="all 22 shipped features + six forecasts of each coin&#39;s BTC+ETH beta-residual return over the target&#39;s span: a causal rolling ridge, 20-day residual momentum, and zero-shot Chronos-2 on the raw path, the residual path, the residual path with BTC / ETH / leave-one-out size-peer covariates, and grouped size-bucket context",
                 builder="src/data_mani/c2resid_paths.py, chronos2_resid.py, build_c2resid.py", hosts="deepfour (GPU forecasts) and hamilton, hopper, lovelace, dostoievski, tolstoi and deepsix (fits)",
                 block="residual signal",
                 blocks=["F_COV", "F_GRP", "F_RES", "F_RAW", "F_RIDGE", "F_RESMOM", "ADDED_ONLY"], ablate=["F_COV", "F_RIDGE"], series=["FULL", "F_COV", "F_RIDGE", "P_HIST"], fits=None,
                 missing="Each signal is rank-binned within date with Numerai&#39;s rule. A row has no signal when its coin has no spot feed, fewer than 64 days of residuals in the 256-day context, or is BTC or ETH (no residual); the loader fills 0.5, no indicator, no row dropped. Each arm is routed on its own signal&#39;s coverage.",
                 unobserved="no spot feed, short history, or BTC / ETH",
                 b0chk="1.6e-6", live_gap="The live path does not build this dataset: it needs the chronos2 venv and a GPU every round, and the spot feed for every coin; none is wired into <code>src/live</code>."),
    "E149": dict(root="results/reports/crypto/e149_vol_causal", slug="vol_causal", dataset="crypto/v2.0-vol-causal", nfeat=176, parent=None,
                 title="Does the proposed realised-volatility block help on Numerai Crypto?",
                 what="all 22 shipped features (including the shipped <code>feature_volatility_20d/60d</code>, which stay in the base) + 154 proposed <code>feature_vol_*</code> columns: "
                      "22 realised-volatility levels (close-to-close 5-120d, EWMA &lambda; .90-.99, Parkinson and Yang-Zhang 5-120d) and each level ranked against the coin&#39;s own 7-365-day history",
                 builder="src/data_mani/build_vol_causal.py", hosts="hamilton, hopper, lovelace and dostoievski",
                 block="volatility block", b0chk="1.9e-6",
                 blocks=["F_LVL", "F_TSR", "F_SQRET", "F_EWMA", "F_PARK", "F_YZ", "F_SHORT", "F_LONG", "F_SWAP", "ADDED_ONLY"], ablate=["F_LVL", "F_TSR"], series=["FULL", "F_LVL", "F_TSR", "P_HIST"], fits=None,
                 missing="Estimates are attached as-of D-1 (at most 4 days stale) behind a causal wrong-coin identity gate and binned per date over the coins that passed. A cell the block cannot fill stays NaN; the loader fills 0.5, no indicator, no row dropped. Every arm is routed on the coverage of its own columns.",
                 unobserved="no spot feed, a coin that has not passed the identity gate, or warm-up",
                 live_gap="The live path does not build this block: it needs the spot store refreshed daily for every coin and the identity gate run live; none is wired into <code>src/live</code>."),
    "E159": dict(root="results/reports/crypto/e159_volumealpha", slug="volumealpha", dataset="crypto/v2.0-volumealpha-causal", nfeat=92, parent=None,
                 title="Does the volume-alpha block help on Numerai Crypto?",
                 what="all 22 shipped features + 70 <code>feature_va_*</code> columns from E152&#39;s inferred reconstruction: the paper&#39;s 8 technical inputs (lagged means of return and log dollar volume), "
                      "13 liquidity, 17 fundamental (size, surprise, leverage, valuation, beta, age), 11 cross-asset, 11 out-of-fold volume forecasts and 10 cost / participation columns",
                 builder="src/data_mani/volumealpha_features.py (E152 build)", hosts="the src.cluster boxes",
                 block="volume-alpha block",
                 blocks=["F_TECH", "F_LIQ", "F_FUND", "F_XASSET", "F_FCST", "F_COST", "F_RAW", "F_MODEL", "ADDED_ONLY"], ablate=["F_RAW", "F_MODEL"], series=["FULL", "F_TECH", "F_FCST", "P_HIST"], fits=None,
                 missing="Every column is binned per date with Numerai&#39;s rule over the coins that have it, entering with a 2-day observation lag. A cell the block cannot fill stays NaN; the loader fills 0.5, no indicator, no row dropped. Every arm is routed on the coverage of its own columns.",
                 unobserved="no exchange or aggregator volume for the coin, an identity that is not inferred, or warm-up",
                 live_gap="The live path does not build this block: it needs the exchange archives, aggregator volume, fundamentals and the OOF volume models refreshed every round; none is wired into <code>src/live</code>."),
}
PARTS = {"E146": "e146_report_parts", "E148": "e148_report_parts", "E149": "e149_report_parts", "E159": "e159_report_parts"}   # studies with their own audit + extra sections
READING = {}   # filled per study after its numbers are read; keyed by study id


class Study:
    def __init__(self, sid):
        self.sid, self.c = sid, STUDIES[sid]
        r = self.c["root"]
        self.D = pd.read_csv(os.path.join(r, "paired_deltas.csv"))
        self.L = pd.read_csv(os.path.join(r, "summary_levels.csv"))
        self.RT = pd.read_csv(os.path.join(r, "routing.csv"))
        self.RL = pd.read_csv(os.path.join(r, "routing_levels.csv"))
        self.RS = pd.read_csv(os.path.join(r, "routing_by_seed.csv"))
        self.PW = pd.read_csv(os.path.join(r, "payout_window.csv"))
        self.BL = pd.read_csv(os.path.join(r, "blends.csv"))
        self.BY = pd.read_csv(os.path.join(r, "paired_deltas_by_block.csv"))
        self.AU = json.load(open(os.path.join(r, "audit.json")))
        self.S = pd.read_parquet(os.path.join(r, "scores_by_date_seed.parquet"), columns=["model", "arm", "seed", "date", "block", "kind", "corr", "mmc", "payout"])
        self.X = pd.read_parquet(os.path.join(r, "routed_by_date_seed.parquet"))
        self.routed = set(self.RT["arm"])

    # -------- cells
    def plain(self, m, a, v, metric, sc):
        q = self.D[(self.D["model"] == m) & (self.D["arm"] == a) & (self.D["vs"] == v) & (self.D["metric"] == metric) & (self.D["scope"] == sc)]
        r = q.iloc[0]
        return r["delta"], r["t"]

    def route(self, m, a, metric, sc, rule="R_SLOT"):
        r = self.RT[(self.RT["model"] == m) & (self.RT["arm"] == a) & (self.RT["rule"] == rule) & (self.RT["scope"] == sc)].iloc[0]
        return r[metric + "_delta"], r[metric + "_t"]

    def routed_vs(self, m, a, b, metric, sc, rule="R_SLOT"):
        """Routed a against routed b, seed-averaged per date."""
        x = self.X[(self.X["model"] == m) & (self.X["rule"] == rule)].groupby(["arm", "date", "kind"], as_index=False)[[metric]].mean()
        z = x[x["arm"] == a].merge(x[x["arm"] == b], on=["date", "kind"], suffixes=("", "_b"))
        z = z[z["kind"] == sc]
        r = E.describe((z[metric] - z[metric + "_b"]).dropna())
        return r["mean"], r["nw_t"]

    def cell(self, m, a, v, metric, sc, mode):
        if mode == "route":
            return self.route(m, a, metric, sc) if v == "B0" else self.routed_vs(m, a, v, metric, sc)
        return self.plain(m, a, v, metric, sc)

    def mode(self, a, v="B0"):
        return "route" if a in self.routed and (v == "B0" or v in self.routed) else "plain"

    def verdict(self, m, a, v="B0", mode=None):
        mode = mode or self.mode(a, v)
        dv, tt = self.cell(m, a, v, "corr", "dev", mode), self.cell(m, a, v, "corr", "test", mode)
        pd_, pt = self.cell(m, a, v, "payout", "dev", mode), self.cell(m, a, v, "payout", "test", mode)
        if dv[1] <= -2:
            return "REJECT"
        if dv[1] >= 2 and tt[0] > 0 and pd_[1] > -2 and pt[1] > -2:
            return "ADOPT"
        if tt[1] >= 2:
            return "PROMISING"
        return "NO IMPROVEMENT"

    def t4(self, m, a, v, metric, sc, mode=None):
        d, t = self.cell(m, a, v, metric, sc, mode or self.mode(a, v))
        return W.tint(d, t)

    def cite(self, m, a, v, metric, sc, mode=None):
        d, t = self.cell(m, a, v, metric, sc, mode or self.mode(a, v))
        return "%s (t %s)" % (W.num(d, 4, True), W.num(t, 1, True))

    def rows(self, pairs, verdicts=True, mode=None):
        out = []
        for a, v in pairs:
            md = mode or self.mode(a, v)
            for m in MODELS:
                r = [(a if v == "B0" else "%s vs %s" % (a, v)) + (" <small>routed</small>" if md == "route" else ""), m] + \
                    [self.t4(m, a, v, x, sc, md) for x in ("corr", "payout") for sc in ("dev", "test")]
                r = [r[0], r[1], r[2], r[3], r[4], r[5]]
                out.append(r + [self.verdict(m, a, v, md)] if verdicts else r)
        return out

    def level_rows(self, arms):
        rows = []
        for m in MODELS:
            for a in arms:
                for sc in ("dev", "test"):
                    if a in self.routed:
                        r = self.RL[(self.RL["model"] == m) & (self.RL["arm"] == a) & (self.RL["rule"] == "R_SLOT") & (self.RL["scope"] == sc)].iloc[0]
                        lab, pc = a + " (routed)", "&ndash;"
                    else:
                        r = self.L[(self.L["model"] == m) & (self.L["arm"] == a) & (self.L["scope"] == sc)].iloc[0]
                        lab, pc = a, W.num(r["predcorr_b0"], 2)
                    cal = W.num(r["calmar"], 3) if np.isfinite(r["calmar"]) else "no drawdown"
                    rows.append([m, lab, sc, W.num(r["corr"]), W.num(r["mmc"]), W.num(r["c_sharpe"], 2), W.num(r["dd_corr"], 2), W.num(r["payout"]), W.num(r["p_sharpe"], 2),
                                 W.num(r["dd_pay"], 2), cal, int(r["burn"]), "t20", int(r["n"]), int(r["n_pay"]), pc])
        return rows

    def curves(self, col, keys):
        A = self.S.groupby(["model", "arm", "date"], as_index=False)[[col]].mean().pivot_table(index=["model", "date"], columns="arm", values=col)
        R = self.X[self.X["rule"] == "R_SLOT"].groupby(["model", "arm", "date"], as_index=False)[[col]].mean().pivot_table(index=["model", "date"], columns="arm", values=col)
        out = {}
        for m in MODELS:
            x = A.xs(m, level="model").dropna(subset=["B0"])
            f = pd.DataFrame(index=x.index)
            for k in keys:
                src = R.xs(m, level="model")[k] if k in self.routed else x[k]
                f[k] = (src.reindex(x.index) - x["B0"]).fillna(0)
            out[m] = f.cumsum()
        return out


def render(sid):
    s = Study(sid)
    c = s.c
    blk_name = c.get("block", "trend block")
    parent = c["parent"]
    fulls = {m: s.verdict(m, "FULL") for m in MODELS}
    order = ["REJECT", "NO IMPROVEMENT", "PROMISING", "ADOPT"]
    overall = "ADOPT" if all(v == "ADOPT" for v in fulls.values()) else \
              "REJECT" if all(v == "REJECT" for v in fulls.values()) else \
              ("PROMISING BUT UNCONFIRMED" if any(v == "PROMISING" for v in fulls.values()) and not any(v == "REJECT" for v in fulls.values()) else "NO IMPROVEMENT")
    rejects = [NAME[m] for m in MODELS if fulls[m] == "REJECT"]
    head = overall + (" (REJECT on %s)" % ", ".join(rejects) if rejects and overall != "REJECT" else "")
    tag = head.split(" (")[0].replace(" ", "_")
    out = os.path.join(c["root"], "%s_does_%s_help_%s.html" % (sid, c["slug"], tag))
    W.SERIES.clear()
    W.SERIES.update({a: i + 1 for i, a in enumerate(c["series"])})
    W.ARM_NAME.update({a: a if a != "P_HIST" else "tenure control" for a in c["series"]})

    vt = [[NAME[m]] + [s.t4(m, "FULL", "B0", x, sc) for x in ("corr", "payout") for sc in ("dev", "test")] + [s.verdict(m, "FULL")] for m in MODELS]
    H = ['<h1>%s %s</h1>' % (c["title"], head),
         '<p class="sub"><code>%s</code> (%d features: %s) against <code>crypto/v2.0</code> (22). Fitted on %s. Registered before launch as %s.</p>' % (c["dataset"], c["nfeat"], c["what"], c["hosts"], sid),
         '<nav><a href="#design">Design</a><a href="#audit">Audit</a><a href="#levels">Levels</a><a href="#deltas">Deltas</a>%s<a href="#controls">Controls</a><a href="#retune">Retune</a><a href="#blends">Blends</a><a href="#stability">Stability</a><a href="#gaps">Gaps</a></nav>'
         % ('<a href="#parent">Against the parent</a>' if parent else ""),
         '<div class="verdict"><b>Verdict: %s.</b> %s</div>' % (head, READING.get(sid, {}).get("verdict", "")),
         '<h3>FULL against v2.0, routed (the deployment score)</h3>',
         W.table(["model", "ΔCORR dev", "ΔCORR test", "Δpayout dev", "Δpayout test", "verdict"], vt)]
    if parent:
        pv = [[NAME[m]] + [s.t4(m, "FULL", parent, x, sc) for x in ("corr", "payout") for sc in ("dev", "test")] + [s.verdict(m, "FULL", parent)] for m in MODELS]
        H += ['<h3>FULL against its parent %s, both routed: the new block&#39;s own margin</h3>' % parent,
              W.table(["model", "ΔCORR dev", "ΔCORR test", "Δpayout dev", "Δpayout test", "verdict"], pv)]
    H += ['<p>Cells are the change with the Newey-West t at lag 20, tinted only at |t| &ge; 2. <b>Routed</b>: v2.0&#39;s model ranks every coin, and the candidate reorders only the coins whose @BLK@ is observed that date, inside the rank slots they already hold (E134 R_SLOT); a coin without a feed keeps v2.0&#39;s exact rank. Payout is 0.1&times;CORR + MMC built per period, with proxy MMC against the historical meta model, never official. Dev has 1,430 CORR dates and 468 payout dates (payout starts May 2024); test has 130 and 120.</p>']
    if READING.get(sid, {}).get("leads"):
        H += ['<h3>Reading</h3><ul>%s</ul><p><small>Written after the numbers were seen. The verdicts above are computed from the bar, not from this text.</small></p>' % "".join("<li>%s</li>" % x for x in READING[sid]["leads"])]

    au = s.AU
    tr = au.get("truncation", [])
    H += ['<h2 id="design">Design and what deviated</h2><ul>',
          ('<li><b>Dataset:</b> built by <code>%s</code>. Every v2.0 row is kept and the 22 shipped features are untouched. ' % c["builder"] + c["missing"] + '</li>') if "missing" in c else
          '<li><b>Dataset:</b> built by <code>%s</code>. Every v2.0 row is kept and the 22 shipped features are untouched. A coin with no spot feed, or still in warm-up, has NaN in the new block: the file keeps the NaN, the loader fills 0.5, and no <code>feature_isna_*</code> indicator is added, because a flag built from today&#39;s feed marks coins that later died (E131). Rows are not dropped, because which coin has a feed is decided by today&#39;s coverage (E132).</li>' % c["builder"],
          '<li><b>Fits:</b> 12 walk-forward blocks: train 130, purge 22, predict 130. Target <code>target_binned_return_20</code>. Fixed configs from raw v2.0 (E133): ridge, lgb_ktz, mlp_ktz (identity activation, so linear). Five seeds for LightGBM and the MLP; ridge is deterministic.</li>',
          '<li><b>Rows:</b> every arm is fitted and scored on the same 468,000 keys, 300 coins per date, 2020-08-28 to 2026-08-20.</li>',
          '<li><b>Reused:</b> B0, tenure (P_HIST) and the ridge and LightGBM tuned B0 come from E133/E135. A ridge B0 refit on this dataset&#39;s rows matches them to %s.%s</li>' % (c.get("b0chk", "8.9e-7"), " The parent arm %s and the P_WARM / P_NM placebos are the earlier study&#39;s predictions, with the same columns on the same rows." % parent if parent else ""),
          '<li><b>Retune:</b> E133&#39;s small grid, one seed, picked per block on the last 40 train dates after a 22-date purge. The MLP&#39;s alpha grid is capped at 1, because alpha 100 collapsed the net to a constant in E137, so the MLP&#39;s tuned B0 is refitted on the capped grid.</li>',
          '<li><b>Test</b> (block 11, 2026-02-20 to 2026-08-20) selects nothing here, but earlier studies scored those dates on other features, so it is not virgin. The 28 labelled dates after it stay unread.</li>',
          '<li>A within-date-constant column (every macro and inflation feature) cannot reorder coins in a linear model: it only shifts all predictions on a date by the same amount. On ridge and the linear MLP, any effect of such a block comes through the other coefficients changing, not through the block itself.</li></ul>' if "missing" not in c else '</ul>']
    if sid in PARTS:
        import importlib
        X = importlib.import_module(PARTS[sid])
        H += X.audit(s, W)
        au = {}
    H += [] if sid in PARTS else ['<h2 id="audit">Point-in-time audit</h2><ul>',
          '<li><b>No later data is used:</b> the new block was rebuilt from inputs cut at %d dates (%s), and %s of %s cells differ from %s.</li>' % (len(dict.fromkeys(t["date"] for t in tr)), ", ".join(dict.fromkeys(t["date"] for t in tr)), "{:,}".format(sum(t["differ"] for t in tr)), "{:,}".format(sum(t["cells"] for t in tr)),
          "the same saved block model run on the uncut panel" if "stored_differ" in tr[0] else "the dataset")]
    if "redundancy" in au:
        cv, rd = au["coverage"], au["redundancy"]
        top = max(au["vs_base"].items(), key=lambda kv: kv[1]["rho"])
        ic = sorted(au["date_ic"].items(), key=lambda kv: -abs(kv[1]["t"]))
        H += ['<li><b>The cut runs:</b> for each date the extractor was rerun on the panel truncated at that date, with every label newer than 35 days blanked: what was knowable then. Rows with a state per audited date (%s); %s%s.</li>' % (", ".join("%s: %d" % (t["date"], t["states_cut"]) for t in tr if t["mode"] == "cov"),
              ("every one of the 768 values of every state is identical in both modes" if sum(t["differ"] for t in tr) == 0 and all(t["cut_rows_without_full_state"] == 0 for t in tr)
              else "<b>%s values differ</b> (largest %.3g)" % ("{:,}".format(sum(t["differ"] for t in tr)), max(t.get("max_abs_diff", 0) for t in tr))),
              ((". Each date used the model of the block whose test window holds it (%s), reloaded from its fp16 copy and run on both the cut and the uncut panel, so the comparison holds the weights fixed. "
                "Against the dataset&#39;s own states, which the in-memory fp32 merge produced, %s of %s values differ by at most %.3g (mean |value| %.2f, lowest cosine %.5f): the rounding of the stored weights, not information"
                % (", ".join("%s: block %d" % (t["date"], t["block"]) for t in tr if t["mode"] == "cov"), "{:,}".format(sum(t["stored_differ"] for t in tr)), "{:,}".format(sum(t["cells"] for t in tr)),
                   max(t["stored_max_abs_diff"] for t in tr), np.mean([t["stored_scale"] for t in tr]), min(t["stored_cosine_min"] for t in tr))) if "stored_differ" in tr[0] else "")),
              '<li><b>Coverage:</b> the @BLK@ is observed on %.1f%% of scored keys; the rest is %s. %.1f%% of keys belong to a coin that never has a state anywhere in the file (P_NM&#39;s flag).</li>' % (100 * cv["observed"], c.get("unobserved", "warm-up or the 60 burn-in dates"), 100 * cv["no_feed"]),
              '<li><b>Scale:</b> bin mass %s, the shipped 5/20/50/20/5 once the 0.5 fill is counted.</li>' % " / ".join("%s: %.3f" % (k, v) for k, v in au["mass"].items()),
              '<li><b>Redundancy:</b> mean |Spearman| among the 32 PCs is %.2f with %d pairs above 0.8; the most similar cov/ar pair reaches %.2f. The highest |Spearman| of any PC with a shipped feature is %.2f (%s with %s).</li>' % (rd["mean_abs_rho"], rd["pairs_over_08"], rd["cov_vs_ar_max"], top[1]["rho"], top[0], top[1]["base"]),
              '<li><b>Raw per-date IC with the label</b>, before any model: %d of 32 PCs reach |t| &ge; 2 (lag-20 scaling); the strongest are %s. Standalone IC is not marginal value: see the deltas.</li>' % (sum(abs(v["t"]) >= 2 for _, v in ic), ", ".join("%s %+.3f (t %+.1f)" % (k.replace("feature_", ""), v["mean"], v["t"]) for k, v in ic[:4]))]
        y = pd.DataFrame(au["coverage_by_year"])
        H += ['</ul>', W.table(["year", "scope", "observed", "never-observed coin"], [[int(r.date), r.kind, "%.1f%%" % (100 * r.observed), "%.1f%%" % (100 * r.no_feed)] for r in y.itertuples()]),
              '<p><small>Missingness over time on the scored keys.</small></p><ul>']
    elif "coverage" in au:
        cv = au["coverage"]
        H += ['<li><b>Coverage:</b> the @BLK@ is observed on %.1f%% of keys (%.1f%% with at least half its columns). %.1f%% of keys belong to a coin with no value in it anywhere in the file, and %.1f%% are warm-up on a coin that has a feed.</li>' % (100 * cv["tf_any"], 100 * cv["tf_half"], 100 * cv["no_feed"], 100 * cv["warm_given_feed"])]
        if "vs_momswap_macro_carver" in au:
            H += [
              '<li><b>Against the old carver set:</b> on the same keys, the macro block equals <code>v2.0-momswap-macro-carver</code>&#39;s cell for cell. The extended spot file gives the trend block %s more observed cells and loses none; %s cells differ in all, through re-binning over a larger cross-section.</li>' % ("{:,}".format(au["vs_momswap_macro_carver"]["gained"]), "{:,}".format(au["vs_momswap_macro_carver"]["differ"]))]
        H += ['<li><b>Scale:</b> the @BLK@ bins carry mass %s, the shipped 5/20/50/20/5.%s</li>' % (" / ".join("%.3f" % v for v in au["mass_tf"].values()),
              (" Macro columns take %d distinct value per date." % au["macro_distinct_per_date"]) if "beta_redundancy" not in au else "")]
        if "beta_redundancy" in au:
            br_, top = au["beta_redundancy"], max(au["beta_vs_base"].items(), key=lambda kv: kv[1]["rho"])
            H += ['<li><b>Redundancy:</b> mean |Spearman| among the 32 betas is %.2f, with %d pairs above 0.8. The highest |Spearman| of any beta with a shipped feature is %.2f (%s with %s): a beta scales with the coin&#39;s volatility, but is far from a copy of it.</li>' % (br_["mean_abs_rho"], br_["pairs_over_08"], top[1]["rho"], top[0], top[1]["base"])]
        y = pd.DataFrame(au["coverage_by_year"])
        H += ['</ul>', W.table(["year", "scope", "trend observed", "half observed", "no-feed coin"], [[int(r.date), r.kind, "%.1f%%" % (100 * r.tf_any), "%.1f%%" % (100 * r.tf_half), "%.1f%%" % (100 * r.no_feed)] for r in y.itertuples()]),
              '<p><small>Missingness over time on the scored keys.</small></p><ul>']
    if sid not in PARTS and "parent_differ" in au:
        H += ['<li><b>Parent cells:</b> %s of %s cells shared with E138&#39;s dataset differ. The block takes %d distinct value per date and is observed on %.1f%% of key cells.</li>' % ("{:,}".format(au["parent_differ"]), "{:,}".format(au["parent_cells"]), au["distinct_per_date"], 100 * au["observed"])]
    if "abs_spearman_ix_vs_own_trend" in au:
        q = au["abs_spearman_ix_vs_own_trend"]
        H += ['<li><b>Sign-flip identity:</b> on sampled dates an interaction&#39;s |Spearman| with its own trend signal averages %.3f (minimum %.3f, %d pairs). The macro magnitude washes out in the per-date rank, as the construction says it must: each interaction is its trend signal, reversed on dates when the macro signal is negative.</li>' % (q["mean"], q["min"], q["n"]),
              '<li><b>Coverage:</b> interactions are observed on %.1f%% of key cells against %.1f%% for the trend block.</li>' % (100 * au["observed"], 100 * au["tf_observed"])]
    H += [] if sid in PARTS else ['</ul>'] if "missing" in c else ['<li><b>Survivorship that routing cannot remove:</b> the spot file was fetched in 2026, so which coins have a feed at all is a today-snapshot. Routing keeps a no-feed coin on v2.0&#39;s rank, but the candidate is still judged mostly on coins that lived. P_NM below measures how much the no-feed flag itself is worth.</li></ul>']

    larms = ["B0", "FULL"] + ([parent] if parent else []) + [b for b in c["blocks"] if b not in (parent,)] + ["P_HIST"]
    H += ['<h2 id="levels">Levels</h2>', W.table(["model", "arm", "scope", "CORR", "MMC", "c.sharpe", "dd(corr)", "payout", "p.sharpe", "dd(pay)", "calmar", "burn(d)", "tgt", "n", "n(pay)", "corr B0"], s.level_rows(larms), "l3"),
          '<p><small>Routed rows use R_SLOT. MMC, payout, p.sharpe, dd(pay), calmar and burn(d) cover the n(pay) dates. Compare calmar and burn(d) only within one scope. corr B0 is the mean per-date rank correlation of an unrouted arm&#39;s predictions with B0&#39;s.</small></p>']

    H += ['<h2 id="deltas">Paired deltas against v2.0</h2><h3>The whole block and its parts</h3>',
          W.table(HDR, s.rows([(a, "B0") for a in ["FULL"] + c["blocks"]]), "l2"),
          '<h3>FULL with one family removed, or one family added to the parent</h3>', W.table(HDR, s.rows([(a, "B0") for a in c["ablate"]]), "l2"),
          '<h3>Unrouted, for reference: the same arms scored on all 300 coins, with a no-feed coin&#39;s 0.5 cells fed to the model</h3>',
          W.table(HDR, s.rows([(a, "B0") for a in ["FULL"] + [b for b in c["blocks"] if b != "ADDED_ONLY"]], mode="plain"), "l2")]
    fam_pairs = [("FULL", a) for a in c["ablate"]] if sid == "E138" else [(a, parent) for a in c["ablate"]] if sid == "E140" else [("FULL", a) for a in c["ablate"]]
    H += ['<h3>Each family&#39;s own margin</h3>', W.table(HDR, s.rows(fam_pairs), "l2")]
    if sid in ("E143", "E144"):
        H += ['<h3>With covariates against temporal only</h3>', W.table(HDR, s.rows([("F_COV", "F_AR")]), "l2")]
    if sid == "E138":
        H += ['<h3>Each block given the other</h3>', W.table(HDR, s.rows([("FULL", "F_MACRO"), ("FULL", "F_TF")]), "l2")]
    if sid in PARTS:
        H += X.extra(s, W, HDR)
    rr = [[NAME[m], "%.1f%%" % (100 * s.RT[(s.RT["model"] == m) & (s.RT["arm"] == "FULL") & (s.RT["rule"] == "R_SLOT") & (s.RT["scope"] == "test")]["routed"].iloc[0])]
          + [s.t4(m, "FULL", "B0", x, sc, "route") for x in ("corr",) for sc in ("dev", "test")]
          + [W.tint(*s.route(m, "FULL", x, sc, "R_HALF")) for x in ("corr",) for sc in ("dev", "test")] for m in MODELS]
    H += ['<h3>Routing rule sensitivity, FULL</h3>', W.table(["model", "coins routed to v2.0", "R_SLOT ΔCORR dev", "R_SLOT test", "R_HALF ΔCORR dev", "R_HALF test"], rr)]
    pw = [[NAME[m], a + (" vs " + v if v != "B0" else "")] + ["%s <small>t %s</small>" % (W.num(r[k], 4, True), W.num(r[k.replace("delta", "t")], 1, True)) for r in [s.PW[(s.PW["model"] == m) & (s.PW["arm"] == a) & (s.PW["vs"] == v) & (s.PW["scope"] == "dev")].iloc[0]] for k in ("corr_delta", "payout_delta")]
          for a, v in (("FULL", "B0"),) + (((("FULL", parent),)) if parent else ()) for m in MODELS]
    H += ['<h3>CORR on payout&#39;s own dev dates (May 2024 on), unrouted</h3><p>Dev CORR starts in 2020 and dev payout in May 2024, so "CORR down, payout up" can be two windows (E136). This puts both on the 468 payout dates.</p>',
          W.table(["model", "arm", "ΔCORR", "Δpayout"], pw)]

    if parent:
        H += ['<h2 id="parent">Against the parent %s</h2><p>The parent is the previous study&#39;s FULL, so this is what the new block adds to it. Both legs are routed on the same trend coverage.</p>' % parent,
              W.table(HDR, s.rows([("FULL", parent)] + [(a, parent) for a in c["ablate"] if sid == "E140"]), "l2"),
              W.table(HDR, s.rows([(parent, "B0")]), "l2")]

    H += ['<h2 id="controls">Tenure, warm-up and look-ahead controls</h2>',
          W.table(HDR, s.rows([("P_HIST", "B0"), ("FULL_H", "P_HIST"), ("P_WARM", "B0"), ("P_NM", "B0")]), "l2"),
          '<ul><li><b>P_HIST</b> (causal tenure) and <b>P_WARM</b> (binned share of the @BLK@ observed, causal) are controls, never candidates. A numeric ADOPT on one of them is a dev effect on a control.</li>',
          '<li><b>FULL_H vs P_HIST</b> asks whether the block adds anything once listing age is in the model.</li>',
          '<li><b>P_NM</b> adds a flag that is look-ahead by construction: the coin never has a value in the @BLK@ anywhere in the file, which only the whole file can tell. Whatever it earns is the survivorship channel a coverage indicator would buy, and why none is shipped.</li></ul>']

    H += ['<h2 id="retune">Equal-budget retune</h2>', W.table(HDR[:-1], s.rows([("tuneFULL", "tuneB0")], False, "plain"), "l2"),
          '<p>Unrouted. The same grid on both legs, so a block that only looks bad under configs tuned for v2.0 would recover here.</p>']

    A = pd.read_parquet(os.path.join(c["root"], "blend_by_date.parquet")).groupby(["model", "date", "kind", "w"], as_index=False)[["corr", "payout"]].mean()
    br = [[r.model, r.metric, W.num(r.w, 2), r.scope, W.num(r.b0), W.num(r.level)] + ([W.tint(r.delta, r.t), "%s, %s" % (W.num(r.lo, 4, True), W.num(r.hi, 4, True))] if r.w > 0 else ["no blend", "&ndash;"]) for r in s.BL.itertuples()]
    bt = []
    for m in MODELS:
        for sc in ("dev", "test"):
            q = A[(A["model"] == m) & (A["kind"] == sc)].groupby("w")[["corr", "payout"]].mean()
            bt += [[m, sc, "CORR"] + [W.num(v) for v in q["corr"]], [m, sc, "payout"] + [W.num(v) for v in q["payout"]]]
    H += ['<h2 id="blends">Blends</h2><p>Rank-blend of B0 and ADDED_ONLY, unrouted. The weight is picked on dev alone, once by CORR and once by payout; test then scores it once.</p>',
          W.table(["model", "picked on", "w", "scope", "B0", "blend", "Δ", "95% CI"], br, "l3"),
          W.table(["model", "scope", "metric"] + ["w = %.2f" % w for w in sorted(A["w"].unique())], bt, "l3")]

    blk = []
    for a in ["FULL"] + c["blocks"]:
        for m in MODELS:
            g = s.BY[(s.BY["model"] == m) & (s.BY["arm"] == a) & (s.BY["vs"] == "B0")].sort_values("block")
            if len(g):
                blk.append([a, m] + [W.num(x, 3, True) for x in g["mean"]])
    sd = s.RS[(s.RS["arm"] == "FULL") & (s.RS["rule"] == "R_SLOT") & (s.RS["model"] != "ridge")]
    seed = [[NAME[m], sc] + [W.num(v, 4, True) for v in sd[(sd["model"] == m) & (sd["scope"] == sc)].sort_values("seed")["corr_delta"]] for m in ("lgb_ktz", "mlp_ktz") for sc in ("dev", "test")]
    H += ['<h2 id="stability">Stability</h2><h3>By block, mean ΔCORR against v2.0, unrouted (block 11 is test)</h3>', W.table(["arm", "model"] + [str(i) for i in range(12)], blk, "l2"),
          '<h3>FULL routed, ΔCORR by seed</h3>', W.table(["model", "scope"] + ["seed %d" % i for i in range(5)], seed),
          W.line_panels(s.curves("corr", c["series"]), c["series"], lambda m: m + ": cumulative ΔCORR vs v2.0"),
          W.line_panels(s.curves("payout", c["series"]), c["series"], lambda m: m + ": cumulative Δpayout vs v2.0, proxy MMC"),
          '<p><small>Routed arms are drawn routed (R_SLOT); P_HIST is unrouted. Payout curves start in May 2024.</small></p>']

    H += ['<h2 id="gaps">Gaps and next steps</h2><ul>',
          '<li>No catboost or forest arm; mlp_ktz is linear, so only LightGBM can use a date-constant block through interactions with coin features. MMC is a proxy.</li>',
          '<li>The retune used one seed and a small grid.</li>',
          '<li>%s</li>' % c.get("live_gap", "The live path does not build this dataset: it needs the spot feed and the macro feed at D-1, neither wired into <code>src/live</code>."),
          '<li>The 28 labelled dates after 2026-08-20 stay unread; they have no meta model.</li>'] + ["<li>%s</li>" % x for x in READING.get(sid, {}).get("next", [])] + ['</ul>']

    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>%s evaluation</title><style>%s\nnav a{margin-right:1em}h3{font-size:1rem;margin:1.4em 0 .3em}</style></head><body><main>%s</main><script>%s</script></body></html>'
           % (sid, W.CSS, "\n".join(H), W.JS))
    doc = doc.replace("@BLK@", blk_name)
    open(out, "w").write(doc)
    print(out, os.path.getsize(out))
    return out, s


if __name__ == "__main__":
    import e138_readings  # noqa: E402,F401  the per-study READING text, written after the numbers
    READING.update(e138_readings.READING)
    render(sys.argv[1] if len(sys.argv) > 1 else "E138")
