#!/usr/bin/env python3
"""E132 report: one self-contained HTML file built from e132_analysis.py's tables.

    python3 experiments/e132_report.py
"""
import base64
import json
import os
import pathlib

import markdown
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
ROOT = pathlib.Path("results/reports/crypto/e132_feed_fillind")
NAME = "E132_do_feed_fillind_datasets_help_REJECT.html"
M = {"ridge": "ridge", "lgb_ktz": "lgb_ktz", "mlp_ktz": "mlp_ktz"}
D = pd.read_csv(ROOT / "paired_deltas.csv")
L = pd.read_csv(ROOT / "summary_levels.csv")
X = pd.read_csv(ROOT / "cross_model.csv")
T = pd.read_csv(ROOT / "tuned_equal_budget.csv")
BL = pd.read_csv(ROOT / "blend_summary.csv")
BY = pd.read_csv(ROOT / "paired_deltas_by_block_year.csv")
SD = pd.read_csv(ROOT / "paired_deltas_by_seed.csv")
AU = json.load(open(ROOT / "audit_summary.json"))
AR = pd.read_csv(ROOT / "audit_dropped_rows.csv")
AI = pd.read_csv(ROOT / "audit_indicators.csv")


def md(df, nd=4):
    return df.round(nd).to_markdown(index=False)


def cell(model, comp, scope, mean="mean", t="nw_t"):
    r = D[(D["model"] == model) & (D["comparison"] == comp) & (D["scope"] == scope)].iloc[0]
    return "%+.4f (%+.1f)" % (r[mean], r[t])


def verdict_table():
    rows = []
    for k, n in (("L", "litmom-feed"), ("S", "momswap-feed"), ("C", "carver-feed")):
        for m in M:
            rows.append({"dataset": n, "model": m, "vs v2.0, dev": cell(m, k + "_FULL - B0", "dev"),
                         "vs v2.0, test": cell(m, k + "_FULL - B0", "test"),
                         "features alone, dev": cell(m, k + "_FULL - B0F", "dev"),
                         "features alone, test": cell(m, k + "_FULL - B0F", "test"),
                         "payout vs v2.0, dev": cell(m, k + "_FULL - B0", "dev", "payout_mean", "payout_t"),
                         "payout vs v2.0, test": cell(m, k + "_FULL - B0", "test", "payout_mean", "payout_t")})
    return pd.DataFrame(rows).to_markdown(index=False)


def deltas(comps):
    rows = []
    for c in comps:
        for m in M:
            rows.append({"comparison": c, "meaning": D.loc[D["comparison"] == c, "meaning"].iloc[0], "model": m,
                         "dev": cell(m, c, "dev"), "test": cell(m, c, "test"),
                         "dev kept rows": cell(m, c, "dev", "kept_mean", "kept_t")})
    return pd.DataFrame(rows).to_markdown(index=False)


LEV = L[L["arm"].isin(["B0", "B0N", "B0F", "P_DROP", "P_HIST", "L_FULL", "S_FULL", "C_FULL"])]
levels = LEV[["model", "arm", "scope", "mean", "sharpe", "mmc", "payout", "p_sharpe", "dd_pay", "calmar", "burn", "window_n", "n"]] \
    .rename(columns={"mean": "CORR", "sharpe": "c.sharpe", "mmc": "MMC proxy", "dd_pay": "dd(pay)", "burn": "burn(d)", "window_n": "n(payout)", "n": "n(CORR)"})
xm = X[["comparison", "meaning", "avg_mean", "avg_nw_t", "avg_boot_lo", "avg_boot_hi"]]
tu = T[["model", "comparison", "scope", "arm", "ref", "mean", "nw_t", "boot_lo", "boot_hi", "n"]].rename(columns={"arm": "candidate", "ref": "reference"})
bl = BL[["model", "dataset", "w_star", "lobo", "dev_delta", "dev_t", "test_w0", "test_wstar", "test_delta", "test_t"]]
blk = BY[(BY["by"] == "block") & BY["comparison"].isin(["B0F - B0", "L_FULL - B0F", "S_FULL - B0F", "C_FULL - B0F"])] \
    .pivot_table(index=["comparison", "model"], columns="key", values="mean").reset_index()
yr = BY[(BY["by"] == "year") & BY["comparison"].isin(["B0F - B0", "L_FULL - B0F", "S_FULL - B0F", "C_FULL - B0F"])] \
    .pivot_table(index=["comparison", "model"], columns="key", values="mean").reset_index()
yr.columns = [str(c) for c in yr.columns]
blk.columns = [str(c) for c in blk.columns]
seed = SD[SD["comparison"].isin(["L_FULL - B0F", "S_FULL - B0F", "C_FULL - B0F"]) & (SD["scope"] == "dev")] \
    .groupby(["model", "comparison"])["mean"].agg(["min", "max", "count"]).reset_index()
ind = AI[AI["indicator"].str.contains("ret_12_1|trend_240d|bollinger_60d")]

text = f"""# E132: do the feed-fillind datasets add value over crypto/v2.0?

## 1. Verdict

**REJECT for all three datasets, on all three models.** With no-feature coins held at a neutral 0.5,
`crypto/v2.0-litmom-feed-fillind`, `-momswap-feed-fillind` and `-momswap-macro-carver-feed-fillind` all lose
to raw `crypto/v2.0`, on dev and on the test block, and equal-budget retuning does not rescue them. Most of the
loss is the neutral rows themselves. The features add a second, smaller loss for ridge and the MLP on dev, and
are null for LightGBM. No blend passes.

Cells are ΔCORR (Newey-West t, lag 20). "vs v2.0" is the question asked. "features alone" compares with raw
v2.0 on the same rows and the same neutral treatment. Payout uses the proxy MMC.

{verdict_table()}

Verdicts, per dataset and model: **REJECT** against v2.0 on CORR (dev and test, every cell). Features alone:
**REJECT** on dev for ridge and the MLP, **NO IMPROVEMENT** for LightGBM. No cell clears the ADOPT bar. On the
test block, features alone are not significant anywhere in the fixed-config arms.

## 2. Design, and what differs from the spec

- Protocol of E131 and the `feature-set-evaluation` skill: the same 513,496 keys, 12 blocks (train 130 / purge 22 /
  predict 130), 11 dev blocks plus test 2026-02-20..08-20, target_binned_return_20, 5 seeds for LightGBM and the MLP.
- Models: `ridge`, `lgb_ktz`, `mlp_ktz` at E131's fixed configs, all searched on raw v2.0. `ridge_ktz` does not exist.
  `mlp_ktz` uses identity activation, so it is linear, and there is no nonlinear neural arm here.
- **Neutral rule (requested):** each arm's predictions become a per-date percentile, and a key with no prediction
  takes 0.5. A coin with no features is therefore scored, at the middle.
- Added arms: `B0F` (raw v2.0 fitted on the feed rows, neutral elsewhere), `B0N` (B0's own predictions with the same
  rows set to 0.5, no refit), `P_DROP` (raw v2.0 plus the look-ahead no-feed flag, on every key), `P_HIST` (causal
  tenure, on the feed rows).
- `L_ADDED` and `S_ADDED` are the same columns (17 momentum + 17 indicators), so their rows match by construction.
  `C_ADDED` differs. The carver and momswap sets have no base-indicator columns.
- **The test block is not untouched:** E65 and E131 already scored it. It was not used for any choice here.
- MMC is a proxy against the historical meta model and is not official. It exists on 468 dev and 120 test dates.
  Calmar and burn are only comparable between rows with the same n(payout).

## 3. Point-in-time audit

The three datasets keep the same keys, and every arm is scored on all 468,000. Of those, **{AU['dropped_rows']:,} ({AU['share']:.2%}) are neutral**, on {AU['symbols']} symbols.
(The 6.4% quoted while the study ran counted every date in the panel, not only the predicted ones.)

**The drop is look-ahead.** A coin is dropped because no price feed exists today. Dropped rows average target
{AU['target_dropped']:.3f}, kept rows {AU['target_kept']:.3f}, and only {AU['symbols_alive_on_last_date']} of the {AU['symbols']} symbols are still in the panel on the last date.
So the row set knows which coins die. Fitting on it (`B0F`) is not causal either.

{md(AR, 3)}

Inside the kept rows, the long-window indicators still mark weaker coins (warm-up is known at the time, so this
channel is causal):

{md(ind, 3)}

Open data issue, carried from the dataset build: about 25 kept coins carry wrong-coin Yahoo prices, so some momentum
and trend values are for the wrong token. This study did not correct them.

## 4. Levels

{md(levels)}

## 5. Paired deltas: row handling and controls

{deltas(['B0N - B0', 'B0F - B0N', 'B0F - B0', 'P_DROP - B0', 'P_HIST - B0F'])}

Setting these coins to 0.5 costs about -0.013 alone (`B0N - B0`) and refitting on the feed rows about another -0.004.
The cost is mechanical: v2.0 already ranks these coins low and they are low. On kept rows, `B0N - B0` is exactly zero.

## 6. Paired deltas: candidates

{deltas(['L_FULL - B0', 'S_FULL - B0', 'C_FULL - B0', 'L_FULL - B0F', 'S_FULL - B0F', 'C_FULL - B0F', 'L_ADDED - B0F', 'C_ADDED - B0F'])}

The bootstrap intervals are in `paired_deltas.csv`.

## 7. Group ablations

{deltas(['L_M1 - B0F', 'L_M2 - L_M1', 'L_B1 - B0F', 'L_FULL - L_M1', 'L_M1 - L_no_ret', 'L_M1 - L_no_maxvol', 'L_M1 - L_no_trend', 'S_S0 - B0F', 'S_M1 - S_S0', 'S_FULL - S_M1', 'C_MACRO - S_S0', 'C_TF - S_S0', 'C_RAW - S_M1', 'C_FULL - C_RAW', 'C_FULL - C_no_mom', 'C_FULL - C_no_macro', 'C_FULL - C_no_tf'])}

## 8. Average across models (dev)

{md(xm)}

Group reading: literature momentum is the harmful block for ridge and the MLP, as in E131. Indicators do not help.
Carver's indicators are significantly harmful on dev (average t -3.3). Macro and carver trend are null. Dropping the 8
base momentum proxies is null. The momentum swap is positive on test for LightGBM and the MLP (`S_M1 - S_S0`, t about
+2.3), but the MLP's dev sign is significantly negative and LightGBM's dev is null, so it fails the bar.

## 9. Equal-budget retune

Each model re-searched with 50 draws on each dataset's rows. `B0F` is the search on raw v2.0 feed rows, `B0` is E131's
search on all keys. Candidates and `B0F` are completed with the neutral rule. Pre-test covers 2024-08..2026-02-19.

{md(tu)}

Retuning does not rescue any dataset against v2.0: all 18 cells are negative, and 15 of them reach |t| ≥ 2,
including every test cell. The other three are LightGBM pre-test (t -1.5 to -2.0). One MLP cell, momswap vs `B0F` on test (+0.0165, t +2.1), is positive, but pre-test is
-0.0201 and its bootstrap interval includes zero, so it fails the same-sign bar.

## 10. Placebo and tenure

`P_DROP` averages +0.0042 on dev (t +2.05) and is mixed on test (ridge +0.0041, lgb -0.0024, mlp +0.0016, none significant).
That is the look-ahead flag's value alone. It does not collapse on test as E131's placebo did.
`P_HIST` averages +0.0088 on dev (t +3.8), but test is ridge +0.0029, lgb -0.0020, mlp -0.0021, none significant. E131's
tenure lead was positive on test for all three models, so this rerun weakens it.

## 11. Blends

`(1-w)·B0 + w·ADDED` on completed percentiles, w chosen on dev, then test once. The blend's ADDED leg is at 0.5 on neutral rows.

{md(bl)}

The E131 LightGBM lead does not repeat on the feed rows: w drops to 0.10 and test is +0.0030 (t +1.5) for litmom and
momswap, +0.0010 for carver. For ridge and the MLP on litmom and momswap, w = 0 wins in every leave-one-block-out fold.

## 12. Missingness over time

![](figures/08_missingness_time.png)

## 13. Stability by block, year and seed

ΔCORR by block (11 = test):

{md(blk, 3)}

By year:

{md(yr, 3)}

Seed range of dev ΔCORR, features alone (ridge is deterministic):

{md(seed)}

## 14. Figures

![](figures/01_cumulative_corr.png)

![](figures/02_cumulative_delta_vs_B0.png)

![](figures/03_rolling_corr.png)

![](figures/04_delta_distribution.png)

![](figures/05_by_fold_and_seed.png)

![](figures/06_model_feature_heatmap.png)

![](figures/07_blend_sensitivity.png)

![](figures/09_ablation_ci.png)

![](figures/10_predcorr_with_B0.png)

## 15. Gaps

- `MISS_ONLY` and `M1FB` from E131 were not rerun. Three of the 12 retunes needed relaunching after systemd stopped
  them. All 12 finished, and none is missing.
- The retune comparison against `B0` uses E131's search on all keys, and it is not re-scored with any neutral rows.
- No catboost or forest arm. The skill asks for at least three classes and these three repeat E131's.
- Wrong-coin prices were not corrected (section 3).

## 16. Recommended next steps

1. Do not ship these datasets on these terms, and do not set no-feature coins to 0.5 in a live slot based on this study.
2. If they are to be judged on features alone, fix the about 25 wrong-coin price series first, then rerun only `*_FULL` and `B0F`.
3. Drop further work on the missingness indicators. E131 and E132 agree they do not help.
4. The tenure lead needs a causal, all-keys test before more time goes into it, since this rerun's test sign was mixed.

No prediction was uploaded, no model created and no stake changed.
"""


def embed(m):
    p = ROOT / m[2]
    return "![%s](data:image/png;base64,%s)" % (m[1], base64.b64encode(p.read_bytes()).decode())


import re  # noqa: E402
text = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", embed, text)
text = text.replace("(+nan)", "(n/a)").replace(" nan ", " n/a ")
body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
css = """:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#ddd;--alt:#f6f6f6}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e6e6e6;--muted:#999;--line:#333;--alt:#202020}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:24px 16px}
table{border-collapse:collapse;margin:12px 0;font-size:13px;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}th{background:var(--alt)}
td:first-child,th:first-child{text-align:left}img{max-width:100%;background:#fff}
code{background:var(--alt);padding:1px 4px;border-radius:3px}h2{border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:2em}"""
html = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>E132 feed-fillind verdict</title><style>%s</style></head><body>%s</body></html>" % (css, body))
(ROOT / NAME).write_text(html)
print(ROOT / NAME, len(html))
