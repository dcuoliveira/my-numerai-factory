"""Wait for managed E150 batches, select on dev, score and produce the PDF."""
import html
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path("results/reports/crypto/e150_macrovol_observed_dev")
MODELS = ("ridge", "lgb_ktz", "mlp_ktz")
GROUPS = ("main", "sources", "gates", "controls")
CMD = [sys.executable, "experiments/e150_macrovol_evaluation.py", "--research-retrospective",
       "--dataset-dir", "data/crypto/v2.0-vol-causal-macrovol-retrospective", "--n-jobs", "1"]


def report():
    levels = pd.read_csv(ROOT / "levels.csv")
    paired = pd.read_csv(ROOT / "paired_deltas.csv")
    matched = paired[paired["vs"].str.endswith("__VC_COV") | paired["arm"].isin(["VC_COV", "tuneVC_COV"])]
    registration = json.loads((ROOT / "registration.json").read_text())
    selection = json.loads((ROOT / "parsim.json").read_text())
    headline = []
    for model in MODELS:
        c = paired[(paired.model == model) & (paired.arm == "tuneALL") &
                   (paired.vs == "tuneALL__VC_COV") & (paired.metric == "corr")].iloc[0]
        mmc = paired[(paired.model == model) & (paired.arm == "tuneALL") &
                     (paired.vs == "tuneALL__VC_COV") & (paired.metric == "mmc")].iloc[0]
        lev = levels[(levels.model == model) & (levels.arm == "tuneALL")].iloc[0]
        verdict = "REJECT on development" if c.t <= -2 else ("PROMISING, UNCONFIRMED" if c.t >= 2 else "NO IMPROVEMENT detected")
        headline.append(dict(model=model, verdict=verdict, corr_delta=c.delta, corr_t=c.t,
                             mmc_delta=mmc.delta, mmc_t=mmc.t, coverage=lev.covered,
                             worst_corr_block=c.worst_block, corr_win_rate=c.win,
                             payout_drawdown=lev.dd_pay, periods=int(c.n), mmc_periods=int(mmc.n)))
    headline = pd.DataFrame(headline)
    headline.to_csv(ROOT / "headline.csv", index=False)
    primary = ["F_" + k for k in ("GOLD", "SP500", "DXY", "Y10", "VIXLVL", "VIXVOV")] + ["ALL", "PARSIM", "NO_EXPO", "EXPO_ONLY"]
    primary += ["tune" + name for name in primary]
    regimes = pd.read_csv(ROOT / "regime_deltas.csv")
    regimes = regimes[regimes["arm"].isin(primary) & regimes["vs"].str.endswith("__VC_COV")
                      & regimes["regime"].isin(["high", "low"])]
    regimes = regimes[regimes.apply(lambda row: row["arm"].removeprefix("tune") in
                                   ("ALL", "PARSIM", "F_" + row["macro_source"].upper()), axis=1)]
    body = '''<h1>Does macro volatility improve causal crypto volatility?</h1>
    <p><b>E150 verdict: no macro family passed the frozen cross-model development selection rule.
    Do not adopt the combined dataset on this evidence.</b> The parsimonious selection is empty.
    The table below compares the searched all-macro model with the searched vol-causal model
    trained on exactly the same observed rows, after slot-preserving routing.</p>'''
    body += headline.to_html(index=False, float_format=lambda x: f"{x:.5g}")
    body += '''
    <p><b>Exploratory development comparison. No adoption decision or untouched-test claim.</b>
    Macro histories are retrospective. Gold is the legacy GLD ETF proxy rather than spot or futures.
    The inherited crypto universe is reconstructed. ALFRED gold/DXY coverage and a fresh future test remain outstanding.</p>
    <p>Ridge (requested ridge_ktz), LightGBM (requested gbm_ktz), and the repository's identity-activation MLP
    are reported separately. Fixed fits use five stochastic seeds; six-candidate inner searches use seed 0.
    Every candidate and its v2.0-only and vol-causal comparators use identical observed rows.
    Inactive high/low gates never enter fitting. Uncovered coins retain explicit baseline ranks.
    Training 130 dates, purge 22, evaluation 130; previously scored final block excluded.</p>
    <p>Paired statistics use Newey-West uncertainty and moving-block bootstrap intervals.
    MMC is a historical-meta-model proxy. Many comparisons are exploratory; no multiplicity-adjusted claim is made.
    Positive-scalar, circular and explicitly look-ahead placebos cannot be selected.
    High/low regime tables use each source's 20-day regime estimator.</p>'''
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scores = pd.read_parquet(ROOT / "scores_by_date_seed.parquet")
    daily = scores.groupby(["model", "arm", "date", "block"], as_index=False)[["corr", "mmc", "payout", "covered"]].mean()
    daily.groupby(["model", "arm", "block"], as_index=False)[["corr", "mmc", "payout", "covered"]].mean().to_csv(ROOT / "by_block.csv", index=False)
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        part = daily[daily.model.eq(model)].pivot(index="date", columns="arm", values="corr")
        for arm, reference, label in (("ALL", "ALL__VC_COV", "Fixed configurations (5 stochastic seeds)"),
                                      ("tuneALL", "tuneALL__VC_COV", "Searched configurations (seed 0)")):
            delta = (part[arm] - part[reference]).dropna()
            ax.plot(delta.index, delta.cumsum(), label=label, linewidth=1.1)
        ax.axhline(0, color="#444", linewidth=.6)
        ax.set_title({"ridge": "Ridge", "lgb_ktz": "LightGBM", "mlp_ktz": "MLP (identity)"}[model])
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Cumulative paired CORR difference")
    axes[-1].legend(fontsize=6)
    figure = figure_dir / "all_macro_cumulative_corr_delta.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    body += '<h2>Stability through development periods</h2><p>All-macro minus its matched vol-causal comparator. Downward movement means the macro features reduce CORR. These are accumulated correlation differences, not investment returns.</p>'
    body += '<img style="width:100%" src="data:image/png;base64,' + base64.b64encode(figure.read_bytes()).decode() + '">'
    tables = [("Matched marginal improvements", matched[matched["arm"].isin(primary)]),
              ("Levels and downside", levels[levels["arm"].isin(primary + ["B_ALL", "B_COV", "VC_COV"])]),
              ("Ablations, gates and controls (CORR)", paired[(paired["metric"] == "corr") &
                 (~paired["arm"].isin(primary))]),
              ("High/low regimes by macro source", regimes),
              ("Seed stability", pd.read_csv(ROOT / "seed_deltas.csv")),
              ("Coverage by year", pd.read_csv(ROOT / "coverage_by_year.csv"))]
    for title, frame in tables:
        if title == "Matched marginal improvements":
            frame = frame[["model", "arm", "metric", "delta", "se", "t", "lo", "hi", "win", "worst_block", "dd", "n"]]
        elif title == "Levels and downside":
            frame = frame[["model", "arm", "corr", "mmc", "payout", "c_sharpe", "p_sharpe", "dd_pay", "calmar", "burn", "covered", "n", "n_pay"]]
        elif title == "Ablations, gates and controls (CORR)":
            frame = frame[["model", "arm", "vs", "delta", "t", "lo", "hi", "win", "n"]]
        elif title == "High/low regimes by macro source":
            frame = frame[["model", "arm", "macro_source", "regime", "delta", "t", "win", "n"]]
        body += "<h2>" + title + "</h2>" + frame.to_html(index=False, float_format=lambda x: f"{x:.5g}")
    unavailable = [json.loads(p.read_text()) for p in (ROOT / "preds").rglob("*.unavailable.json")]
    if unavailable:
        body += "<h2>Arms with insufficient observed coverage</h2>" + pd.DataFrame(unavailable).to_html(index=False)
    body += "<h2>Development-only parsimonious selection</h2><pre>" + html.escape(json.dumps(selection, indent=2)) + "</pre>"
    body += "<h2>Frozen inputs and model rules</h2><pre>" + html.escape(json.dumps(registration, indent=2)) + "</pre>"
    css = "body{font:11px Arial;color:#18202a;background:white}table{border-collapse:collapse;font-size:9px;width:100%}td,th{padding:4px;border:1px solid #ddd;overflow-wrap:anywhere}h2{break-before:page}tr{break-inside:avoid}pre{white-space:pre-wrap}"
    path = ROOT / "E150_macrovol_EXPLORATORY_DEVELOPMENT.html"
    path.write_text('<!doctype html><html><head><meta charset="utf-8"><title>Macro volatility comparison</title><style>'
                    + css + '</style></head><body>' + body + '</body></html>')
    subprocess.run([sys.executable, "-m", "src.analysis.html_pdf", str(path)], check=True)


def main():
    expected = [ROOT / "batch_status" / (m + "_" + g + ".json") for m in MODELS for g in GROUPS]
    while True:
        statuses = [json.loads(p.read_text()) if p.exists() else dict(state="pending") for p in expected]
        failed = [str(p) for p, s in zip(expected, statuses) if s["state"] == "failed"]
        if failed:
            raise RuntimeError("Batch failure; report withheld: " + str(failed))
        complete = sum(s["state"] == "complete" for s in statuses)
        print("completed batches", complete, "/", len(expected), flush=True)
        if complete == len(expected):
            break
        time.sleep(30)
    os.environ["E150_WORKERS"] = "1"
    subprocess.run(CMD + ["--select"], check=True)
    if json.loads((ROOT / "parsim.json").read_text())["sources"]:
        for model in MODELS:
            for extra in ([], ["--tune"]):
                subprocess.run(CMD + ["--fit", "--model", model, "--arms", "PARSIM"] + extra, check=True)
    subprocess.run(CMD + ["--score"], check=True)
    report()
    (ROOT / "COMPLETE.json").write_text(json.dumps(dict(scope="exploratory development only", completed=time.time())))


if __name__ == "__main__":
    main()
