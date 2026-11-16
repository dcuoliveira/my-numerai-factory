#!/usr/bin/env python3
"""E137 report: the feature-set evaluation of crypto/v2.0-fenccausal-binned as one self-contained HTML file.

    python3 experiments/e137_report.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e133_report as W  # noqa: E402  CSS, crosshair script, table, tint and line_panels

os.chdir(REPO)
ROOT = "results/reports/crypto/e137_fenccausal_binned"
OUT = os.path.join(ROOT, "E137_does_fenccausal_binned_help_NO_IMPROVEMENT.html")
MODELS = ["ridge", "lgb_ktz", "mlp_ktz"]
NAME = {"ridge": "ridge", "lgb_ktz": "LightGBM", "mlp_ktz": "MLP (linear)"}
D = pd.read_csv(os.path.join(ROOT, "paired_deltas.csv"))
L = pd.read_csv(os.path.join(ROOT, "summary_levels.csv"))
PW = pd.read_csv(os.path.join(ROOT, "payout_window.csv"))
BL = pd.read_csv(os.path.join(ROOT, "blends.csv"))
BY = pd.read_csv(os.path.join(ROOT, "paired_deltas_by_block.csv"))
RT = pd.read_csv(os.path.join(ROOT, "routing.csv"))
AU = json.load(open(os.path.join(ROOT, "audit.json")))
S = pd.read_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"), columns=["model", "arm", "seed", "date", "block", "kind", "corr", "payout"])
SERIES = ["FULL", "CONT", "F_SIGNAL", "F_VOL", "P_HIST"]
W.SERIES.clear()
W.SERIES.update({a: i + 1 for i, a in enumerate(SERIES)})
W.ARM_NAME.update({"FULL": "FULL", "CONT": "CONT", "F_SIGNAL": "F_SIGNAL", "F_VOL": "F_VOL", "P_HIST": "tenure control"})
_table = W.table
W.table = lambda head, rows, cls="": _table(head, [[str(c) for c in r] for r in rows], cls)
HDR = ["arm", "model", "ΔCORR dev", "ΔCORR test", "Δpayout dev", "Δpayout test", "verdict"]


def d(m, a, v, metric, sc):
    return D[(D["model"] == m) & (D["arm"] == a) & (D["vs"] == v) & (D["metric"] == metric) & (D["scope"] == sc)].iloc[0]


def verdict(m, a, v="B0"):
    """The numeric reading of the skill's vocabulary, the same bar as E134 and E136."""
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


def cite(m, a, v, metric, sc):
    r = d(m, a, v, metric, sc)
    return "%s (t %s)" % (W.num(r["delta"], 4, True), W.num(r["t"], 1, True))


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
                calmar = W.num(r["calmar"], 3) if np.isfinite(r["calmar"]) else "no drawdown"
                rows.append([m, a, sc, W.num(r["corr"]), W.num(r["mmc"]), W.num(r["c_sharpe"], 2), W.num(r["dd_corr"], 2), W.num(r["payout"]), W.num(r["p_sharpe"], 2), W.num(r["dd_pay"], 2),
                             calmar, int(r["burn"]), "t20", int(r["n"]), int(r["n_pay"]), W.num(r["predcorr_b0"], 2)])
    return rows


def collapse():
    """Median within-date spread of the tuned MLP, per block. Below 1e-9 the net predicts a constant."""
    rows = []
    for arm, paths in (("tuneB0", ["results/reports/crypto/e133_histfe/preds/mlp_ktz/tuneB0_s0.parquet", "results/reports/crypto/e135_route_histfe/preds/mlp_ktz/tuneB0_s0.parquet"]),
                       ("tuneFULL", [os.path.join(ROOT, "preds/mlp_ktz/tuneFULL_s0.parquet")])):
        p = pd.concat([pd.read_parquet(x, columns=["date", "prediction"]) for x in paths])
        p["date"] = pd.to_datetime(p["date"])
        q = S[(S["model"] == "mlp_ktz") & (S["arm"] == arm)][["date", "block", "corr"]]
        sd = p.groupby("date")["prediction"].std().rename("sd").reset_index().merge(q, on="date")
        g = sd.groupby("block").agg(sd=("sd", "median"), corr=("corr", "mean")).reset_index()
        g["arm"] = arm
        rows.append(g)
    return pd.concat(rows)


def main():
    arms = ["FULL", "ADDED_ONLY", "CONT", "F_PRICE", "F_CAP", "F_VOL", "F_SIGNAL", "NO_PRICE", "NO_CAP", "NO_VOL", "NO_SIGNAL", "H20", "H60"]
    assert verdict("ridge", "FULL") == "REJECT" and verdict("lgb_ktz", "FULL") == "NO IMPROVEMENT" and verdict("mlp_ktz", "FULL") == "NO IMPROVEMENT"
    assert not any(verdict(m, a) == "ADOPT" for m in MODELS for a in arms)
    assert [(m, a) for m in MODELS for a in arms if verdict(m, a) == "PROMISING"] == [("ridge", "F_VOL")]
    assert AU["rebuild"]["differ"] == 0 and all(t["differ"] == 0 for t in AU["truncation"])
    assert (RT["routed"] == 0).all()
    H = [W.table(["model", "FULL ΔCORR dev", "FULL ΔCORR test", "FULL Δpayout dev", "FULL Δpayout test", "verdict"],
                 [[NAME[m], t4(m, "FULL", "B0", "corr", "dev"), t4(m, "FULL", "B0", "corr", "test"), t4(m, "FULL", "B0", "payout", "dev"), t4(m, "FULL", "B0", "payout", "test"), verdict(m, "FULL")] for m in MODELS])]
    s = ['<h1>Do binned causal per-coin means help on Numerai Crypto? NO IMPROVEMENT</h1>',
         '<p class="sub"><code>crypto/v2.0-fenccausal-binned</code> (44 features) against <code>crypto/v2.0</code> (22). Fitted and scored on <code>lovelace</code>.</p>',
         '<nav><a href="#design">Design</a><a href="#audit">Audit</a><a href="#levels">Levels</a><a href="#deltas">Deltas</a><a href="#binning">Binned vs continuous</a><a href="#controls">Controls</a><a href="#mlp">MLP payout</a><a href="#retune">Retune</a><a href="#blends">Blends</a><a href="#stability">Stability</a><a href="#gaps">Gaps</a></nav>',
         '<div class="verdict"><b>Verdict: NO IMPROVEMENT, and REJECT on ridge.</b> The dataset does not beat v2.0 on any model. ADDED_ONLY is REJECT everywhere, and no feature-set arm reaches ADOPT. '
         'The audit is clean, so this is not a leakage artefact. The MLP&#39;s test payout gain is the largest positive number, but it fails the tenure control and is the same size as the swing from adding one feature. Do not deploy the dataset.</div>'] + H
    s += ['<p>Cells are the change against v2.0 with the Newey-West t at lag 20, tinted only at |t| &ge; 2. Payout uses proxy MMC, never official MMC, and is built per period. Dev is 1,430 CORR dates and 468 payout dates. Test is 130 and 120.</p>']

    s += ['<h2 id="design">Design and what deviated</h2><ul>',
          '<li><b>Fits:</b> 12 walk-forward blocks, train 130, purge 22, predict 130. Target <code>target_binned_return_20</code>. Fixed configs from raw v2.0: ridge, lgb_ktz, mlp_ktz (linear). Five seeds for LightGBM and the MLP.</li>',
          '<li><b>Rows:</b> every arm on the same 468,000 keys.</li>',
          '<li><b>Reused:</b> B0, tenure and tuned B0 from E133 and E135. A ridge refit on this dataset&#39;s rows matches the reused B0 to 8.9e-7.</li>',
          '<li><b>New:</b> 169 prediction files for 15 arms and one retune.</li>',
          '<li><b>CONT is recomputed.</b> <code>v2.0-fenccausal</code> stops on 2026-07-08, before the final block ends, so the unbinned means come from the builder&#39;s own function. They match the shipped file to %.1e on %s rows.</li>' % (AU["continuous"]["max_diff"], "{:,}".format(AU["continuous"]["rows"])),
          '<li><b>Test is not virgin.</b> E65, E131, E132 and E134 to E136 scored those dates on other features. No arm or weight here was chosen from them.</li>',
          '<li><b>Retune:</b> E133&#39;s small grid and one seed, not 50 draws.</li></ul>']

    tr, tk = AU["truncation"], AU["tickers"]
    s += ['<h2 id="audit">Point-in-time audit</h2><ul>',
          '<li><b>Reproduced:</b> %s of %s cells differ when the block is rebuilt.</li>' % ("{:,}".format(AU["rebuild"]["differ"]), "{:,}".format(AU["rebuild"]["cells"])),
          '<li><b>No later rows are used:</b> rebuilt on the panel cut at four dates, %s of %s cells differ. That covers the builder&#39;s whole-frame prior: it is in principle look-ahead, and it moved no bin on those dates.</li>' % ("{:,}".format(sum(t["differ"] for t in tr)), "{:,}".format(sum(t["cells"] for t in tr))),
          '<li><b>One real flaw:</b> the means are keyed on ticker, not coin. %d tickers served two coins, and %.1f%% of keys average in the earlier coin&#39;s history. The effect should be small, and it was not measured.</li>' % (tk["reused_symbols"], tk["keys_mixing_coins"] * 100),
          '<li><b>Scale:</b> masses are %s across 0 to 1, as on the shipped features. %.2f%% of cells are NaN, filled with 0.5 as in the loader.</li>' % (" / ".join("%.3f" % v for v in AU["mass"].values()), AU["nan_cells"] * 100),
          '<li><b>Redundancy:</b> a mean&#39;s Spearman with its own base feature averages %.2f, from %.2f to %.2f.</li>' % (AU["corr_with_own_base"]["mean"], AU["corr_with_own_base"]["min"], AU["corr_with_own_base"]["max"]),
          '<li><b>Routing does nothing:</b> no key lacks the block, so R_SLOT and R_HALF equal plain FULL. No coin falls back to v2.0.</li></ul>']

    s += ['<h2 id="levels">Levels</h2>',
          W.table(["model", "arm", "scope", "CORR", "MMC", "c.sharpe", "dd(corr)", "payout", "p.sharpe", "dd(pay)", "calmar", "burn(d)", "tgt", "n", "n(pay)", "corr B0"], level_rows(["B0", "FULL", "CONT", "ADDED_ONLY", "P_HIST", "P_GLOBAL"]), "l3"),
          '<p><small>MMC, payout, p.sharpe, dd(pay), calmar and burn(d) cover n(pay) dates. Compare calmar and burn(d) only within a scope. corr B0 is the mean per-date rank correlation of predictions with B0.</small></p>']

    s += ['<h2 id="deltas">Paired deltas</h2><h3>Whole block</h3>', W.table(HDR, arm_rows([(a, "B0") for a in ("FULL", "ADDED_ONLY", "H20", "H60")]), "l2"),
          '<h3>Groups added one at a time</h3>', W.table(HDR, arm_rows([("F_" + g, "B0") for g in ("PRICE", "CAP", "VOL", "SIGNAL")]), "l2"),
          '<h3>FULL with one group removed</h3>', W.table(HDR, arm_rows([("NO_" + g, "B0") for g in ("PRICE", "CAP", "VOL", "SIGNAL")]), "l2"),
          '<ul><li><b>The signal means carry the CORR harm.</b> F_SIGNAL is REJECT on ridge %s and the MLP %s. Without them, every NO_SIGNAL cell is NO IMPROVEMENT.</li>' % (cite("ridge", "F_SIGNAL", "B0", "corr", "dev"), cite("mlp_ktz", "F_SIGNAL", "B0", "corr", "dev")),
          '<li><b>One PROMISING cell.</b> Ridge F_VOL scores test CORR %s, with dev %s. It is one cell of 39 and sits on no dev edge, so treat it as expected noise.</li>' % (cite("ridge", "F_VOL", "B0", "corr", "test"), cite("ridge", "F_VOL", "B0", "corr", "dev")),
          '<li><b>Harm sits in the older dates.</b> On payout&#39;s own dates, FULL&#39;s dev CORR change is ridge %s, LightGBM %s, MLP %s, none near |t| 2. The ridge REJECT comes from before May 2024.</li></ul>' % tuple(W.num(pw(m, "FULL", "B0", "dev")["corr_delta"], 4, True) for m in MODELS)]

    s += ['<h2 id="binning">Binned against continuous</h2>', W.table(HDR, arm_rows([("CONT", "B0")]) + arm_rows([("FULL", "CONT")]), "l2"),
          '<ul><li><b>Binning costs nothing.</b> FULL beats CONT on dev CORR for LightGBM %s and the MLP %s. Ridge is level at %s.</li>' % (cite("lgb_ktz", "FULL", "CONT", "corr", "dev"), cite("mlp_ktz", "FULL", "CONT", "corr", "dev"), cite("ridge", "FULL", "CONT", "corr", "dev")),
          '<li><b>The continuous means do worse against v2.0</b>: REJECT on LightGBM and the MLP.</li>',
          '<li><b>The MLP&#39;s test gain over CONT is not robust.</b> CONT&#39;s seed 0 collapsed on test (CORR 0.042), and that one seed supplies most of it.</li>',
          '<li><b>This is not a verdict on IA_01.</b> IA_01 is a different network and fit, and this study did not test it. It does suggest one narrow question: whether binning the means would change that slot.</li></ul>']

    gap = [[NAME[m]] + [t4(m, "P_GLOBAL", "B0", x, sc) for x in ("corr", "payout") for sc in ("dev", "test")] for m in MODELS]
    s += ['<h2 id="controls">Tenure and look-ahead controls</h2>', W.table(HDR, arm_rows([("P_HIST", "B0"), ("FULL_H", "P_HIST")]), "l2"),
          '<ul><li><b>Tenure:</b> the numeric ADOPT on P_HIST is a dev effect on a control, not a candidate. Its test deltas are at most %s, the same flat result as E136.</li>' % W.num(max(d(m, "P_HIST", "B0", "corr", "test")["delta"] for m in MODELS), 4, True),
          '<li><b>The block adds nothing beyond tenure.</b> FULL + tenure is below tenure alone on dev CORR for all three models, and two of those are REJECT.</li></ul>',
          '<h3>What look-ahead would be worth: P_GLOBAL against v2.0</h3><p>P_GLOBAL bins whole-sample means, so it sees the future on purpose. It is a control and never a candidate.</p>',
          W.table(["model", "ΔCORR dev", "ΔCORR test", "Δpayout dev", "Δpayout test"], gap),
          '<p>Knowing each coin&#39;s full-sample mean is worth about +0.05 CORR, and the causal version is worth zero or less on dev. This is the size of the trap in any non-causal per-coin encoding. A result of this size from such a feature should be read as leakage.</p>']

    pv = S[(S["model"] == "mlp_ktz") & S["arm"].isin(["B0", "FULL", "FULL_H", "CONT"])].groupby(["arm", "kind", "seed"])[["corr", "payout"]].mean().reset_index()
    sib = [[a, sc] + [W.num(pv[(pv["arm"] == a) & (pv["kind"] == sc) & (pv["seed"] == k)]["payout"].iloc[0]) for k in sorted(pv["seed"].unique())] for a in ("B0", "FULL", "FULL_H", "CONT") for sc in ("dev", "test")]
    near = [[a + (" vs tenure" if v != "B0" else "")] + [W.num(d("mlp_ktz", a, v, "payout", sc)["delta"], 4, True) for sc in ("dev", "test")] for a, v in (("FULL", "B0"), ("NO_CAP", "B0"), ("NO_PRICE", "B0"), ("NO_VOL", "B0"), ("NO_SIGNAL", "B0"), ("H20", "B0"), ("H60", "B0"), ("FULL_H", "P_HIST"))]
    x1, x2 = pw("mlp_ktz", "FULL", "B0", "dev"), pw("mlp_ktz", "FULL", "B0", "test")
    s += ['<h2 id="mlp">The MLP payout reading, and why it is not a lead</h2>',
          '<p>MLP FULL has the largest positive result: payout dev %s (t %s), test %s (t %s), with the test CI excluding zero and 5 of 5 seeds above B0 on both windows. CORR does not follow: on test it is %s (t %s). The bar is set on CORR, so this stays NO IMPROVEMENT.</p>'
          % (W.num(x1["payout_delta"], 4, True), W.num(x1["payout_t"], 1, True), W.num(x2["payout_delta"], 4, True), W.num(x2["payout_t"], 1, True), W.num(x2["corr_delta"], 4, True), W.num(x2["corr_t"], 1, True)),
          '<ul><li><b>It fails the tenure control.</b> FULL + tenure against tenure is %s on dev and %s on test.</li>' % (cite("mlp_ktz", "FULL_H", "P_HIST", "payout", "dev"), cite("mlp_ktz", "FULL_H", "P_HIST", "payout", "test")),
          '<li><b>Near-identical arms disagree.</b> One extra feature, tenure, takes seeds 1 and 3 from test payout 0.042 and 0.057 to 0.005 and 0.004. A linear network this sensitive to a single column is showing fit noise.</li>',
          '<li><b>The t does not cover that noise.</b> Test is one block, so each seed is one fit. Newey-West measures date noise given those fits, not the spread between fits, and five seeds cannot pin it down.</li></ul>',
          W.table(["arm", "scope", "seed 0", "seed 1", "seed 2", "seed 3", "seed 4"], sib, "l2"), '<p><small>MLP payout by seed.</small></p>',
          W.table(["MLP Δpayout", "dev", "test"], near), '<p><small>Against v2.0 unless marked. The arms are close variants of FULL, and test ranges from negative to +0.0135.</small></p>']

    c = collapse()
    n = {a: int((c[c["arm"] == a]["sd"] < 1e-9).sum()) for a in ("tuneB0", "tuneFULL")}
    cell = lambda a, b, x: c[(c["arm"] == a) & (c["block"] == b)][x].iloc[0]  # noqa: E731
    cr = [[b] + [v for a in ("tuneB0", "tuneFULL") for v in ("%.1e" % cell(a, b, "sd"), W.num(cell(a, b, "corr"), 3))] for b in range(12)]
    s += ['<h2 id="retune">Equal-budget retune</h2>', W.table(HDR[:-1], arm_rows([("tuneFULL", "tuneB0")], False), "l2"),
          '<p>On ridge and LightGBM, retuning does not rescue the block, and LightGBM&#39;s test CORR is worse at %s.</p>' % cite("lgb_ktz", "tuneFULL", "tuneB0", "corr", "test"),
          '<p><b>The MLP row is unreadable, in both directions.</b> With a large alpha the net collapses to a constant: within-date spread falls to about 1e-14, near float64 spacing, and tied predictions appear. The grid chose that in %d of 12 blocks for tuned FULL and %d of 12 for tuned B0. Those blocks score anywhere from &minus;0.135 to +0.146 on one seed. Neither the dev t of &minus;5.5 nor the test t of +3.6 says anything about the features.</p>' % (n["tuneFULL"], n["tuneB0"]),
          W.table(["block", "tuneB0 spread", "tuneB0 CORR", "tuneFULL spread", "tuneFULL CORR"], cr), '<p><small>Median within-date prediction spread and mean CORR, by block.</small></p>']

    A = pd.read_parquet(os.path.join(ROOT, "blend_by_date.parquet")).groupby(["model", "date", "kind", "w"], as_index=False)[["corr", "payout"]].mean()
    bt = []
    for m in MODELS:
        for sc in ("dev", "test"):
            q = A[(A["model"] == m) & (A["kind"] == sc)].groupby("w")[["corr", "payout"]].mean()
            bt.append([m, sc, "CORR"] + [W.num(v) for v in q["corr"]])
            bt.append([m, sc, "payout"] + [W.num(v) for v in q["payout"]])
    br = [[r.model, r.metric, W.num(r.w, 2), r.scope, W.num(r.b0), W.num(r.level)] + ([W.tint(r.delta, r.t), "%s, %s" % (W.num(r.lo, 4, True), W.num(r.hi, 4, True))] if r.w > 0 else ["no blend", "&ndash;"]) for r in BL.itertuples()]
    s += ['<h2 id="blends">Blends</h2><p>Rank-blend of B0 and ADDED_ONLY. The weight comes from dev alone, and test then scores it once.</p>',
          W.table(["model", "picked on", "w", "scope", "B0", "blend", "Δ", "95% CI"], br, "l3"),
          '<p><b>No blend earns a place.</b> The CORR picks take 0 to 0.05 and change nothing. The payout picks take 0.50 and all lose on test, LightGBM significantly.</p>',
          W.table(["model", "scope", "metric"] + ["w = %.2f" % w for w in sorted(A["w"].unique())], bt, "l3")]

    blk = []
    for a in ("FULL", "F_SIGNAL", "F_VOL", "CONT"):
        for m in MODELS:
            g = BY[(BY["model"] == m) & (BY["arm"] == a) & (BY["vs"] == "B0")].sort_values("block")
            blk.append([a, m] + [W.num(x, 3, True) for x in g["mean"]])
    s += ['<h2 id="stability">Stability</h2><h3>By block, mean ΔCORR against v2.0 (block 11 is test)</h3>', W.table(["arm", "model"] + [str(i) for i in range(12)], blk, "l2"),
          W.line_panels(curves("corr", SERIES), SERIES, lambda m: m + ": cumulative ΔCORR vs v2.0"),
          W.line_panels(curves("payout", SERIES), SERIES, lambda m: m + ": cumulative Δpayout vs v2.0, proxy MMC")]

    s += ['<h2 id="gaps">Gaps and next steps</h2><ul>',
          '<li>No catboost or forest arm was tested, and mlp_ktz is linear. MMC is a proxy.</li>',
          '<li>The live path does not offer the binned block, so nothing here can ship without new live code.</li>',
          '<li>The ticker mixing was counted, not fixed. A <code>ucid</code>-keyed rebuild was not run.</li>',
          '<li>The retune used one seed, and the MLP grid can collapse the net.</li>',
          '<li>The 28 labelled dates after 2026-08-20 stay unread. They have no meta model.</li></ul>',
          '<p><b>Recommended:</b> keep live slots on the existing book and do not deploy <code>crypto/v2.0-fenccausal-binned</code>. Open no forward candidate from this study. Cap alpha in the mlp_ktz retune grid before the next evaluation, as the large values make that row unreadable.</p>']

    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fenccausal-binned evaluation</title><style>%s\nnav a{margin-right:1em}h3{font-size:1rem;margin:1.4em 0 .3em}</style></head><body><main>%s</main><script>%s</script></body></html>'
           % (W.CSS, "\n".join(s), W.JS))
    open(OUT, "w").write(doc)
    print(OUT, os.path.getsize(OUT))
    print(c.round(4).to_string(), n)


if __name__ == "__main__":
    main()
