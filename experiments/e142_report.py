#!/usr/bin/env python3
"""E142 report: the look-ahead positive control, one self-contained HTML file, then printed to PDF.

    python3 experiments/e142_report.py
    python3 -m src.analysis.html_pdf <the html it prints>

Every number comes from the CSVs experiments/e142_future_trends_evaluation.py wrote, read through
E138's report machinery. The check is computed, never typed: it PASSES when, for every model, FULL
beats B0 on dev and test CORR at t >= 5 and by more than the largest FULL-vs-B0 CORR gain any honest
block reached in E136-E141. The ordinary bar's verdict is shown too, because the point is that the
bar alone would call this ADOPT.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e133_report as W  # noqa: E402
import e138_report as R  # noqa: E402

os.chdir(REPO)
SID = "E142"
ROOT = "results/reports/crypto/e142_future_trends"
MODELS, NAME = R.MODELS, R.NAME
HONEST = {"E136": "e136_histfe_evaluation", "E137": "e137_fenccausal_binned", "E138": "e138_carver_macro",
          "E139": "e139_carver_macro_infl", "E140": "e140_carver_macro_infl_xinteractions", "E141": "e141_macrobeta"}
R.STUDIES[SID] = dict(root=ROOT, slug="future_trends", dataset="crypto/v2.0-future-trends", nfeat=23, parent=None,
                      title="", what="", builder="src/data_mani/build_future_trends.py", hosts="", blocks=["ADDED_ONLY"],
                      ablate=[], series=["FULL", "ADDED_ONLY", "P_HIST"], fits=None)


def honest_best():
    """Largest dev CORR gain of an unrouted FULL over B0 in each earlier evaluation, per model."""
    rows = []
    for sid, d in HONEST.items():
        f = os.path.join("results/reports/crypto", d, "paired_deltas.csv")
        if not os.path.exists(f):
            continue
        D = pd.read_csv(f)
        q = D[(D["arm"] == "FULL") & (D["vs"] == "B0") & (D["metric"] == "corr")]
        for m in MODELS:
            for sc in ("dev", "test"):
                r = q[(q["model"] == m) & (q["scope"] == sc)]
                if len(r):
                    rows.append(dict(study=sid, model=m, scope=sc, delta=float(r["delta"].iloc[0]), t=float(r["t"].iloc[0])))
    return pd.DataFrame(rows)


def main():
    s = R.Study(SID)
    hb = honest_best()
    best = hb.groupby(["model", "scope"])["delta"].max()
    checks, vt = {}, []
    for m in MODELS:
        dv, tt = s.cell(m, "FULL", "B0", "corr", "dev", "plain"), s.cell(m, "FULL", "B0", "corr", "test", "plain")
        ok = dv[1] >= 5 and tt[1] >= 5 and dv[0] > best.get((m, "dev"), -np.inf) and tt[0] > best.get((m, "test"), -np.inf)
        checks[m] = ok
        vt.append([NAME[m]] + [s.t4(m, "FULL", "B0", x, sc, "plain") for x in ("corr", "payout") for sc in ("dev", "test")]
                  + [W.num(best.get((m, "dev"), np.nan), 4, True), s.verdict(m, "FULL", "B0", "plain"), "PASS" if ok else "FAIL"])
    passed = all(checks.values())
    head = "PASS: leakage detected" if passed else "FAIL"
    au = json.load(open(os.path.join(ROOT, "audit.json")))
    out = os.path.join(ROOT, "%s_does_the_evaluation_catch_lookahead_%s.html" % (SID, "PASS" if passed else "FAIL"))
    W.SERIES.clear()
    W.SERIES.update({"FULL": 1, "ADDED_ONLY": 2, "P_HIST": 3})
    W.ARM_NAME.update({"FULL": "FULL", "ADDED_ONLY": "future return alone", "P_HIST": "tenure control"})
    lv = lambda m, a, sc: s.L[(s.L["model"] == m) & (s.L["arm"] == a) & (s.L["scope"] == sc)].iloc[0]

    H = ['<h1>Does the feature-set evaluation catch look-ahead? %s</h1>' % head,
         '<p class="sub"><code>crypto/v2.0-future-trends</code> (23 features: the 22 shipped + <code>feature_future_ret_20d</code>, each coin&#39;s spot return over the NEXT 20 calendar days) against <code>crypto/v2.0</code> (22). A positive control registered before launch as %s. The column knows the future and must never be deployed.</p>' % SID,
         '<div class="verdict"><b>Result: %s.</b> %s</div>' % (head,
             "Every model gains far more CORR from the future return than any honest block has reached in E136-E141, at t &ge; 5 on dev and test. The ordinary bar reads this as ADOPT, which is the point: the bar measures size and sign, not provenance, so the point-in-time audit and the placebo arms are what stop a leak." if passed else
             "At least one model failed the check; see the table."),
         W.table(["model", "ΔCORR dev", "ΔCORR test", "Δpayout dev", "Δpayout test", "best honest ΔCORR dev", "bar verdict", "check"], vt),
         '<p>Cells are the change against v2.0 with the Newey-West t at lag 20, unrouted, tinted at |t| &ge; 2. "Best honest" is the largest dev CORR gain of an unrouted FULL over B0 for that model in E136-E141. Payout is 0.1&times;CORR + proxy MMC against the historical meta model, built per period.</p>']

    H += ['<h2>Design</h2><ul>',
          '<li><b>Feature:</b> close(D+20 calendar days) / close(D) &minus; 1 from <code>data/crypto/spot/ohlcv.parquet</code> by symbol, nothing forward-filled, rank-binned per date with Numerai&#39;s exact rule (ties preserved). Missing on either end stays NaN and the loader fills 0.5. Builder <code>src/data_mani/build_future_trends.py</code>.</li>',
          '<li><b>Arms:</b> FULL (22 + the future return) and ADDED_ONLY (the future return alone), against B0 and the tenure control reused from E133/E135. No retune, ablation or placebo arms: there is one column and nothing to tune for.</li>',
          '<li><b>Fits:</b> E138&#39;s protocol: 12 walk-forward blocks (train 130, purge 22, predict 130), <code>target_binned_return_20</code>, fixed ridge / lgb_ktz / mlp_ktz configs, five seeds for LightGBM and the MLP. Same 468,000 keys, 300 coins per date, 2020-08-28 to 2026-08-20; block 11 is test.</li>',
          '<li><b>Reuse check:</b> a ridge B0 refit on this dataset matches the reused B0 (asserted before scoring).</li></ul>']
    sp = au["spearman_with_target"]
    H += ['<h2>What the column knows</h2><ul>',
          '<li><b>Rebuild:</b> recomputing it from the spot file reproduces %s of %s cells.</li>' % ("{:,}".format(au["rebuild_cells"] - au["rebuild_differ"]), "{:,}".format(au["rebuild_cells"])),
          '<li><b>Coverage:</b> observed on %.1f%% of scored keys (dev %.1f%%, test %.1f%%).</li>' % (100 * au["observed"], 100 * au["observed_by_kind"]["dev"], 100 * au["observed_by_kind"]["test"]),
          '<li><b>Overlap with the target:</b> per-date Spearman with <code>target_binned_return_20</code> averages %.3f (median %.3f, lowest %.3f; dev %.3f, test %.3f). It is not higher because the target is not this window: on the panel, the forward window that matches the target best starts 2 days after the row&#39;s date and ends 30 days after it (Spearman 0.88), a 20-trading-day return.</li>' % (sp["mean"], sp["median"], sp["min"], sp["dev"], sp["test"]),
          '<li><b>Scale:</b> bin mass %s. Largest |Spearman| with a shipped feature: %.2f.</li></ul>' % (" / ".join("%.3f" % v for v in au["mass"].values()), au["max_abs_spearman_with_base"])]

    H += ['<h2>Levels</h2>', W.table(["model", "arm", "scope", "CORR", "MMC", "c.sharpe", "dd(corr)", "payout", "p.sharpe", "dd(pay)", "calmar", "burn(d)", "tgt", "n", "n(pay)", "corr B0"],
                                      s.level_rows(["B0", "FULL", "ADDED_ONLY", "P_HIST"]), "l3"),
          '<p><small>The FULL row is routed (R_SLOT): coins without the future return keep v2.0&#39;s rank. MMC, payout, p.sharpe, dd(pay), calmar and burn(d) cover the n(pay) dates; compare calmar and burn(d) within one scope only.</small></p>',
          '<h2>Paired deltas against v2.0</h2>', W.table(R.HDR, s.rows([("FULL", "B0"), ("ADDED_ONLY", "B0"), ("P_HIST", "B0")], mode="plain"), "l2"),
          '<h3>Routed (the deployment score)</h3>', W.table(R.HDR, s.rows([("FULL", "B0")], mode="route"), "l2")]

    hr = [[r.study, NAME[r.model], r.scope, "%s <small>t %s</small>" % (W.num(r.delta, 4, True), W.num(r.t, 1, True))] for r in hb.itertuples()]
    H += ['<h2>Against every honest block</h2><p>FULL against B0, unrouted CORR, from each earlier evaluation&#39;s own paired_deltas.csv.</p>', W.table(["study", "model", "scope", "ΔCORR"], hr, "l2")]

    br = [[r.model, r.metric, W.num(r.w, 2), r.scope, W.num(r.b0), W.num(r.level)] + ([W.tint(r.delta, r.t)] if r.w > 0 else ["no blend"]) for r in s.BL.itertuples()]
    blk = []
    for a in ("FULL", "ADDED_ONLY"):
        for m in MODELS:
            g = s.BY[(s.BY["model"] == m) & (s.BY["arm"] == a) & (s.BY["vs"] == "B0")].sort_values("block")
            blk.append([a, m] + [W.num(x, 3, True) for x in g["mean"]])
    H += ['<h2>Blends</h2><p>Rank-blend of B0 and ADDED_ONLY, weight picked on dev.</p>', W.table(["model", "picked on", "w", "scope", "B0", "blend", "Δ"], br, "l3"),
          '<h2>Stability</h2><h3>By block, mean ΔCORR against v2.0, unrouted (block 11 is test)</h3>', W.table(["arm", "model"] + [str(i) for i in range(12)], blk, "l2"),
          W.line_panels(s.curves("corr", ["FULL", "ADDED_ONLY", "P_HIST"]), ["FULL", "ADDED_ONLY", "P_HIST"], lambda m: m + ": cumulative ΔCORR vs v2.0"),
          W.line_panels(s.curves("payout", ["FULL", "ADDED_ONLY", "P_HIST"]), ["FULL", "ADDED_ONLY", "P_HIST"], lambda m: m + ": cumulative Δpayout vs v2.0, proxy MMC"),
          '<h2>Limits</h2><ul><li>This shows the procedure can see a leak this blunt. It does not show it would catch a subtle one: a feature carrying a sliver of future information would sit inside the range of honest blocks, and only the point-in-time audit (rebuilding a block from inputs cut at T) and look-ahead placebos can find it.</li>',
          '<li>About 25 coins in the spot file carry another coin&#39;s Yahoo prices; on those the column is wrong, which only weakens the control.</li>',
          '<li>No catboost or forest arm. mlp_ktz is linear. MMC is a proxy.</li></ul>']
    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E142 look-ahead control</title><style>%s\nh3{font-size:1rem;margin:1.4em 0 .3em}</style></head><body><main>%s</main><script>%s</script></body></html>'
           % (W.CSS, "\n".join(H), W.JS))
    open(out, "w").write(doc)
    print(out)
    print(pd.DataFrame(vt, columns=["model", "dCORR dev", "dCORR test", "dpay dev", "dpay test", "best honest", "bar", "check"]).to_string())


if __name__ == "__main__":
    main()
