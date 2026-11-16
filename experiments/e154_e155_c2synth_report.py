"""Render the E154/E155 comparison as HTML (then print to PDF with src.analysis.html_pdf).

    python3 experiments/e154_e155_c2synth_report.py      # after e154_e155_c2synth_eval.py
"""
import base64
import io
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
from src.data_mani import c2synth_zeroshot as Z  # noqa: E402

ROOT = "results/reports/crypto/e154_e155_c2synth_comparison"


def tbl(df, fmt=4):
    return df.to_html(index=False, float_format=lambda x: ("%." + str(fmt) + "f") % x, na_rep="-", border=0, classes="t")


def curve():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(REPO, "experiments"))
    import e154_e155_c2synth_eval as V
    K = V.keys()
    A = pd.read_parquet(os.path.join(Z.OUT["return_rank"], "prediction_table.parquet")).rename(columns={"round": "date"})
    B = pd.read_parquet(os.path.join(Z.OUT["bin_forecast"], "prediction_table.parquet")).rename(columns={"round": "date"})
    F = K.merge(A[["symbol", "date", "raw_score", "coverage_flag"]].rename(columns={"raw_score": "A", "coverage_flag": "fa"}), on=["symbol", "date"])
    F = F.merge(B[["symbol", "date", "raw_score", "coverage_flag"]].rename(columns={"raw_score": "B", "coverage_flag": "fb"}), on=["symbol", "date"])
    F = F.merge(V.b0(), on=["symbol", "date"])
    C = F[(F.fa == 1) & (F.fb == 1)].copy()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for col, lab in (("A", "Arm A return_rank"), ("B", "Arm B bin_forecast"), ("b0", "B0 existing model")):
        s = V.per_round(C, col).set_index("date")["corr"].cumsum()
        ax.plot(s.index, s.values, label=lab)
    t0 = K.loc[K.split == "test", "date"].min()
    ax.axvline(t0, color="grey", ls="--", lw=1)
    ax.text(t0, ax.get_ylim()[1] * 0.92, " test", color="grey")
    ax.set_ylabel("cumulative per-round CORR")
    ax.set_title("Common token subset, selected context (256) for both arms")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=.3)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=130)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    L = pd.read_csv(os.path.join(ROOT, "levels.csv"))
    P = pd.read_csv(os.path.join(ROOT, "paired.csv"))
    D = json.load(open(os.path.join(ROOT, "decisions.json")))
    cs = pd.read_csv(os.path.join(ROOT, "coverage_summary.csv"))
    cfg = {a: json.load(open(os.path.join(Z.OUT[a], "config.json"))) for a in Z.OUT}
    aud = {a: json.load(open(os.path.join(Z.OUT[a], "audit_inputs.json"))) for a in Z.OUT}
    reasons = []
    for a in Z.OUT:
        t = pd.read_parquet(os.path.join(Z.OUT[a], "prediction_table.parquet"))
        r = t.groupby("split")["skip_reason"].value_counts().unstack(fill_value=0)
        r = (r / t.groupby("split").size().values[:, None]).reset_index()
        r.insert(0, "arm", a)
        reasons.append(r)
    R = pd.concat(reasons).fillna(0).rename(columns={"": "eligible"})
    lv = ["series", "split", "rounds", "mean_corr", "std_corr", "pos_frac", "corr_nw_t", "corr_boot_lo", "corr_boot_hi", "mean_mmc", "mean_tokens"]
    grid = L[L.role == "context grid"][lv]
    own = L[L.role.isin(["standalone, own universe", "reference"])][lv]
    com = L[L.role.isin(["common subset", "blend"])][lv]
    sl = L[L.role == "secondary existing model"][lv]
    pc = ["pair", "split", "corr_delta", "corr_nw_t", "corr_boot_lo", "corr_boot_hi", "corr_rounds", "mmc_delta", "mmc_nw_t", "mmc_boot_lo", "mmc_boot_hi"]
    test = lambda d: d[d.split == "test"]
    a_t = L[(L.series == "A return_rank (selected ctx)") & (L.split == "test")].iloc[0]
    b_t = L[(L.series == "B bin_forecast (selected ctx)") & (L.split == "test")].iloc[0]
    b0_t = L[(L.series == "B0 existing model, full universe") & (L.split == "test")].iloc[0]
    run_rows = pd.DataFrame([dict(registry=("numerai_crypto_c2synth_zero_shot_return_rank (E154)" if a == "return_rank" else
                                            "numerai_crypto_c2synth_zero_shot_bin_forecast (E155)"),
                                  server=c["host"], unit=c["unit"], job_invocation_id=c["job"], gpu=c["cuda_visible"],
                                  revision=c["revision"][:12], started=c["started"], minutes=c["minutes"], status="completed, exit 0",
                                  artifacts=Z.OUT[a]) for a, c in cfg.items()])
    html = """<!doctype html><html><head><meta charset="utf-8"><title>E154/E155 Chronos-2-Synth zero-shot</title>
<style>body{font-family:Helvetica,Arial,sans-serif;font-size:11px;margin:24px;color:#222}h1{font-size:18px}h2{font-size:14px;margin-top:22px;border-bottom:1px solid #ccc}
table.t{border-collapse:collapse;margin:6px 0;font-size:10px}table.t td,table.t th{padding:2px 6px;border-bottom:1px solid #eee;text-align:right}
table.t th{background:#f3f3f3}table.t td:first-child,table.t th:first-child{text-align:left}.k{background:#f7f7f2;padding:8px 12px;border-left:3px solid #998}
code{background:#f2f2f2;padding:0 3px}</style></head><body>
<h1>E154 / E155 &mdash; zero-shot Chronos-2-Synth on Numerai Crypto: return-rank vs bin-forecast</h1>
<p>Frozen <code>autogluon/chronos-2-synth</code> @ <code>3607918a</code>, fp32, per-token inference groups. Written 2026-11-13.</p>
<div class="k"><b>Result.</b> Neither arm is useful as a standalone signal. On the held-out test block (130 rounds) Arm A scores mean CORR
%.4f (NW t %.2f) and Arm B %.4f (t %.2f), against %.4f (t %.2f) for the existing model B0 on the full universe. On the common token
subset both arms trail B0 by about 0.09&ndash;0.10 CORR (A t %.2f, B t %.2f; block-bootstrap intervals in section 5), and both trail the
staked slot CURUPIRA_05 on the test rounds its backtest covers. Arm A beats its 28-day momentum baseline on validation
(+0.076, t +4.4) but not significantly on test (+0.031, t +1.0); Arm B does not beat the last-label or mean-of-five baselines on test.
The arms are only weakly rank-correlated (validation Spearman %.2f), but the predeclared 50/50 blend did not beat Arm A on validation
(%.4f vs %.4f), so by the rule it was <b>not</b> scored on test.</div>

<h2>1. Registry entries, jobs and artifacts</h2>%s
<p>Both units ran under <code>src.cluster</code> (<code>--gpu-slots 1</code>) on the two idle GPU boxes found by <code>probe</code>; each was collected by systemd
after a clean exit. Runner <code>src/data_mani/c2synth_zeroshot.py</code>; evaluator <code>experiments/e154_e155_c2synth_eval.py</code>; this report
<code>experiments/e154_e155_c2synth_report.py</code>. Prediction tables: <code>&lt;artifact dir&gt;/prediction_table.parquet</code> (round, symbol, raw score,
ranked signal in (0,1), coverage flag; Arm B adds last available label date and gap steps). All three contexts are kept in <code>predictions.parquet</code>.</p>

<h2>2. Shared setup (inspected, not assumed)</h2>
<ul>
<li><b>Rounds.</b> House split from <code>keys.parquet</code>: validation = blocks 9&ndash;10 (2025-02-21 &rarr; 2026-02-19, 260 rounds), test = block 11
(2026-02-20 &rarr; 2026-08-20, 130 rounds). Weekday calendar. The test block is <b>not virgin</b>: earlier feature studies scored it. Nothing here was selected on it.</li>
<li><b>Universe.</b> The panel's symbols on each round (300 per round), i.e. the historical universe of that round.</li>
<li><b>Feature cutoff.</b> Round C uses data through day C&minus;1 (the house one-day submit lag).</li>
<li><b>Target.</b> <code>target_binned_return_20</code>. The E142 probe locates it at close(C+2) &rarr; close(C+30), i.e. daily returns of days C+3..C+30
(Spearman 0.88 with that spot return, against 0.56 for C &rarr; C+20).</li>
<li><b>Label availability.</b> The label for date d needs close(d+30); Numerai serves labels about 31&ndash;32 days behind live (the frozen panel taken
2026-11-06 holds labels to 2026-10-06). Rule used: label d is available at round C iff d &le; C&minus;32 days, asserted on every round.</li>
<li><b>Context.</b> 64, 128 and 256 steps all computed; one chosen per arm by validation mean CORR. Eligibility is fixed on the 64-step window so the
three contexts cover identical tokens.</li>
<li><b>Point-in-time audit.</b> Inputs rebuilt from data truncated at each cutoff, at the first, middle and last round: %s cells differ (Arm A),
%s (Arm B). Arm A at a 256-day context reproduces E148's independent <code>raw</code> signal exactly (max abs diff 0.0 on 802 rows).</li>
<li><b>Existing model.</b> Primary: B0, lgb_ktz on the 22 v2.0 features, walk-forward 130/22/130, 5-seed rank average &mdash; the comparator of every
feature study, covering every round. Secondary: staked slot CURUPIRA_05's backtest, which ends 2026-07-08 (99 test rounds). No deployed slot's
backtest covers the whole test block.</li>
<li><b>Scoring.</b> Official <code>numerai_corr</code> and MMC (<code>correlation_contribution</code> against the historical meta model; meta covers 100%% of
validation and 92%% of test rows). Paired per-round differences with Newey&ndash;West lag 20 and a moving-block bootstrap, block 20 rounds, 2000 draws.</li>
</ul>

<h2>3. Arm definitions</h2>
<p><b>Arm A, return_rank.</b> Predeclared input: daily spot log returns (clipped to &plusmn;1) through C&minus;1. Forecast 31 steps (days C..C+30); raw score =
sum of the q50 over days C+3..C+30. The interval start close(C+2) is not known at C&minus;1, so it is forecast, not observed. Chronos-2's native point
forecast is the <b>median</b> (q50 of a 13-level quantile head; there is no mean), so the score is a sum of per-day medians. Eligible: &ge;32 finite returns in
the last 64 days. Baseline: 28-day momentum (sum of the last 28 log returns through C&minus;1).</p>
<p><b>Arm B, bin_forecast.</b> Each token's official labels on their original dates, on a daily calendar with gaps as missing. The context ends at the last
available label date L; the forecast runs from L to C and the step at C is kept (h = C&minus;L, mostly 32 or 34 days; tokens with h &gt; 64 are skipped as
stale). The continuous q50 is the score. Eligible: &ge;20 labels in the 64 days ending at L. Baselines (same availability): last available label; mean of the
last five.</p>

<h2>4. Coverage</h2>%s
<p>Share of the 300-token universe per round. Skip reasons (share of rows):</p>%s

<h2>5. Context selection (validation only) and standalone results</h2>
<p>Selected: Arm A ctx %d, Arm B ctx %d. Test rows of the grid are shown for transparency; they were not used.</p>%s
<p><b>Own eligible universe</b>, each arm with its baselines and B0 scored on the same rows:</p>%s

<h2>6. Paired comparison on the common token subset</h2>
<p>Rounds identical for every series; tokens = eligible in A and B (about %.0f of 300 per round), every series re-ranked within that subset.</p>%s
<p><b>Paired differences</b> (per-round CORR and MMC; NW t; 95%% block-bootstrap interval):</p>%s
<p><b>Secondary existing model</b> (CURUPIRA_05, test rounds to 2026-07-08, common subset with its tokens):</p>%s

<h2>7. Complementarity and the predeclared blend</h2>
<p>Rule: complementary iff the mean within-round Spearman between the arms on validation is &lt; 0.5 <b>and</b> the 50/50 rank average beats the
better arm's validation CORR. Mean rank correlation %.3f (passes); blend validation CORR %.4f vs Arm A %.4f and Arm B %.4f (fails). Not complementary
by the rule, so the blend was <b>not</b> evaluated on test.</p>

<h2>8. Cumulative CORR</h2><img style="width:100%%" src="data:image/png;base64,%s">

<h2>9. Caveats</h2>
<ul>
<li>~26 effective independent windows over 390 overlapping rounds: a null is the expected outcome and the intervals above are the finding.</li>
<li>On test, Arm A at a 64-step context scored CORR 0.016 (t +2.1), better than the selected 256. Choosing it would be selection on test; the
validation choice stands.</li>
<li>The arms show small positive standalone MMC (0.015&ndash;0.022) because they are nearly orthogonal to the meta model, but neither MMC difference against
B0 is significant on test.</li>
<li>Spot prices are a retrospective Yahoo/exchange reconstruction (about 10%% of round tokens have no spot series); historical price revisions cannot be
ruled out. Labels come from a frozen panel copy.</li>
<li>fp32 batch-invariance was measured at ~2e-6 in E145; each token is its own inference group, so group attention never mixes tokens.</li>
</ul></body></html>""" % (
        a_t.mean_corr, a_t.corr_nw_t, b_t.mean_corr, b_t.corr_nw_t, b0_t.mean_corr, b0_t.corr_nw_t,
        P[(P.pair == "A - B0 (common subset)") & (P.split == "test")].corr_nw_t.iloc[0],
        P[(P.pair == "B - B0 (common subset)") & (P.split == "test")].corr_nw_t.iloc[0],
        D["blend"]["mean_rank_corr"], D["blend"]["validation_corr_blend"], D["blend"]["validation_corr_A"],
        tbl(run_rows, 2),
        sum(r["differ"] for r in aud["return_rank"]), sum(r["differ"] for r in aud["bin_forecast"]),
        tbl(cs.rename(columns={"A": "A eligible", "B": "B eligible", "both": "both eligible"}), 3), tbl(R, 3),
        D["return_rank_context"]["selected"], D["bin_forecast_context"]["selected"], tbl(grid), tbl(own),
        D["common_subset_tokens_per_round"], tbl(com), tbl(P[pc]), tbl(sl),
        D["blend"]["mean_rank_corr"], D["blend"]["validation_corr_blend"], D["blend"]["validation_corr_A"], D["blend"]["validation_corr_B"],
        curve())
    out = os.path.join(ROOT, "report.html")
    open(out, "w").write(html)
    print(out)


if __name__ == "__main__":
    main()
