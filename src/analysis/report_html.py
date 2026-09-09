"""
The interactive HTML of a results report.

    python3 -m src.analysis.report_html --model lgb --runs e05_cumlag3

Writes one self-contained .html file -- no CDN, no network, no build step -- with
the same four sections as `report_pdf`: the ranked table, both correlation
matrices, screened ensembles, and cumulative backtested payout against the best
single submitted model. What it adds is the ability to interrogate them: sort any
column, hide the columns you are not reading, filter rows, narrow the date range,
and toggle individual series in the chart.

WHY THE NUMBERS ARE COMPUTED IN THE BROWSER, NOT BAKED IN. A date filter that
only moved the chart would leave the table, the score correlations and the
ensemble screen describing a different window than the picture above them -- the
reader would have four sections that quietly disagree. So this file embeds ONE
matrix, the per-period `corr` and `mmc` of every arm on the block's common dates,
and every figure on the page is derived from it live. Narrowing the range
re-ranks the table, re-screens the ensembles and re-bases the curves together.

The single exception is the PREDICTION correlation matrix, which needs the
row-level forecasts -- 11 MB per arm, far past what belongs in an HTML file. It
is computed once over the full window and carries a badge saying so. The SCORE
matrix, which the ensembling argument actually rests on, does follow the filter.

That leaves one risk worth naming: two implementations of the same arithmetic,
one in Python and one in JavaScript, can drift. `src/tests/test_report_html.py`
runs the embedded JavaScript under node and asserts it reproduces
`report_data`'s rows to 1e-9, so the drift fails a test rather than a decision.
For that to keep working the compute half must stay ES2015 -- no `??`, no `?.`,
no `flat()` -- since the node available here is old.
"""

import datetime
import json
import os

from src.analysis import slots as slotlib
from src.analysis.experiment_db import study_of
from src.analysis.report_data import (CALIBRATION, SCREEN_WARNING, build,
                                      common_parser, correlation_matrices,
                                      price_commands, render_block, short_label)
from src.analysis import report_data
from src.analysis.ensemble_screen import beat_their_best, rank, screen
from src.analysis.report_data import score_series_matrix
from src.analysis.results_table import short_target

#: The categorical slots, in the fixed order that clears the colourblind gates.
#: Assigned to an arm ONCE, by its position in the block, and never reassigned:
#: a reader who has learnt that e05_cumlag3 is blue must not find it repainted
#: because they hid an unrelated series. Past eight the hue repeats and the dash
#: pattern changes, which is a composite encoding rather than a ninth colour.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"]
DASHES = ["", "6 3", "2 3", "9 3 2 3"]

#: Correlation here is a MAGNITUDE -- "how redundant are these two" -- so it gets
#: a sequential single-hue ramp. The one case that is genuinely diverging is a
#: matrix holding a negative correlation, where zero means "no relationship";
#: that gets two opposed hues around a neutral gray.
RAMPS = {
    "seq": [[236, 244, 247], [199, 225, 232], [152, 204, 215], [96, 175, 193],
            [42, 140, 163], [16, 101, 122], [8, 63, 79]],
    "divNeg": [[42, 120, 214], [130, 175, 232], [196, 214, 235]],
    "divMid": [225, 225, 222],
    "divPos": [[240, 206, 200], [227, 133, 124], [196, 42, 40]],
}

#: One geometry for the chart, used by the server-rendered SVG and by the
#: interactive one that replaces it, so the page does not jump on load.
CHART = {"W": 1100, "H1": 320, "H2": 120, "padL": 58, "padR": 104,
         "padT": 30, "padB": 26, "gap": 46}


def ramp_rgb(stops, t):
    """One colour from a ramp."""
    t = max(0.0, min(1.0, t)) * (len(stops) - 1)
    i = min(len(stops) - 2, int(t))
    f = t - i
    return [int(round(stops[i][k] + f * (stops[i + 1][k] - stops[i][k])))
            for k in range(3)]


def heat_rgb(value, low, diverging):
    """The cell colour for a correlation."""
    if value != value:
        return [200, 200, 200]
    if diverging:
        if value >= 0:
            return ramp_rgb([RAMPS["divMid"]] + RAMPS["divPos"], value)
        return ramp_rgb([RAMPS["divMid"]] + RAMPS["divNeg"], -value)
    return ramp_rgb(RAMPS["seq"], (value - low) / max(1e-9, 1 - low))


def ink_on(rgb):
    """
    Text colour for a cell, taken from the cell rather than the theme.

    A fixed ink colour is unreadable on the dark end of any ramp.
    """
    luminance = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255
    return "#131820" if luminance > 0.55 else "#FFFFFF"

CSS = r"""
:root {
  --ground:#F6F7F9; --surface:#FFFFFF; --sunk:#EFF2F6;
  --ink:#131820; --muted:#5D6875; --faint:#8A94A2;
  --line:#E2E6EC; --line-strong:#CDD5DF;
  --accent:#0E7490; --accent-soft:#0E749022;
  --pos:#15803D; --pos-soft:#15803D18;
  --neg:#B4232F; --neg-soft:#B4232F18;
  --flag:#B45309; --flag-soft:#B4530918;
  --grid:#E2E6EC; --context:#B6BEC9;
  --shadow:0 1px 2px rgba(19,24,32,.05), 0 8px 24px -12px rgba(19,24,32,.18);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground:#0D1117; --surface:#151A21; --sunk:#1B222B;
    --ink:#E4E9F0; --muted:#94A0AF; --faint:#6B7889;
    --line:#232B35; --line-strong:#2F3945;
    --accent:#3AAFC9; --accent-soft:#3AAFC926;
    --pos:#4ADE80; --pos-soft:#4ADE8020;
    --neg:#F87171; --neg-soft:#F8717120;
    --flag:#FBBF24; --flag-soft:#FBBF2420;
    --grid:#232B35; --context:#4A5766;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
    --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --ground:#0D1117; --surface:#151A21; --sunk:#1B222B;
  --ink:#E4E9F0; --muted:#94A0AF; --faint:#6B7889;
  --line:#232B35; --line-strong:#2F3945;
  --accent:#3AAFC9; --accent-soft:#3AAFC926;
  --pos:#4ADE80; --pos-soft:#4ADE8020;
  --neg:#F87171; --neg-soft:#F8717120;
  --flag:#FBBF24; --flag-soft:#FBBF2420;
  --grid:#232B35; --context:#4A5766;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  color-scheme: dark;
}
* { box-sizing:border-box; }
body {
  margin:0; background:var(--ground); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased;
}
.wrap { max-width:1500px; margin:0 auto; padding:36px 24px 90px;
  display:flex; flex-direction:column; gap:26px; }
h1,h2,h3 { margin:0; text-wrap:balance; }
h1 { font-size:30px; font-weight:700; letter-spacing:-.02em; }
h2 { font-size:19px; font-weight:650; letter-spacing:-.01em; }
h3 { font-size:14px; font-weight:650; }
.mono,.num,code,pre { font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums; }
.eyebrow { font-size:11px; font-weight:700; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent); font-family:ui-monospace,monospace; }
.lede { color:var(--muted); max-width:78ch; margin:8px 0 0; font-size:14px; }
.head-row { display:flex; justify-content:space-between; align-items:flex-start; gap:20px; }
.card { background:var(--surface); border:1px solid var(--line); border-radius:12px;
  box-shadow:var(--shadow); overflow:hidden; }
.card-head { padding:14px 18px; border-bottom:1px solid var(--line);
  display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
.card-head .grow { flex:1; }
.card-body { padding:14px 18px 18px; }
.badge { font-size:11px; font-family:ui-monospace,monospace; padding:2px 8px;
  border-radius:999px; border:1px solid var(--line-strong); color:var(--muted);
  white-space:nowrap; }
.badge-warn { border-color:var(--flag); color:var(--flag); background:var(--flag-soft); }
.note { color:var(--muted); font-size:12.5px; margin:10px 0 0; }
.note code { font-size:12px; }

/* filters ---------------------------------------------------------------- */
.filters { position:sticky; top:0; z-index:30; background:var(--surface);
  border:1px solid var(--line); border-radius:12px; box-shadow:var(--shadow);
  padding:12px 16px; display:flex; flex-wrap:wrap; gap:14px 20px; align-items:center; }
.fgroup { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.fgroup > .lbl { font-size:11px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--faint); font-family:ui-monospace,monospace; }
button.chip, .chip { font:inherit; font-size:12px; padding:3px 10px; border-radius:999px;
  border:1px solid var(--line-strong); background:var(--sunk); color:var(--muted);
  cursor:pointer; font-family:ui-monospace,monospace; }
button.chip:hover { border-color:var(--accent); color:var(--ink); }
button.chip[aria-pressed="true"] { background:var(--accent-soft); border-color:var(--accent);
  color:var(--accent); font-weight:600; }
input[type=search], input[type=text] { font:inherit; font-size:13px; padding:5px 10px;
  border-radius:8px; border:1px solid var(--line-strong); background:var(--sunk);
  color:var(--ink); min-width:190px; }
input[type=range] { accent-color:var(--accent); width:160px; }
.range-readout { font-family:ui-monospace,monospace; font-size:12px; color:var(--muted); }
details.cols { position:relative; }
details.cols > summary { list-style:none; cursor:pointer; font-size:12px;
  font-family:ui-monospace,monospace; padding:3px 10px; border-radius:999px;
  border:1px solid var(--line-strong); background:var(--sunk); color:var(--muted); }
details.cols > summary::-webkit-details-marker { display:none; }
details.cols[open] > summary { border-color:var(--accent); color:var(--accent); }
.colmenu { position:absolute; top:32px; left:0; z-index:40; background:var(--surface);
  border:1px solid var(--line-strong); border-radius:10px; box-shadow:var(--shadow);
  padding:10px 12px; display:grid; grid-template-columns:repeat(2,minmax(120px,1fr));
  gap:2px 16px; min-width:300px; }
.colmenu label { font-size:12.5px; display:flex; gap:7px; align-items:center;
  cursor:pointer; white-space:nowrap; font-family:ui-monospace,monospace; }

/* tables ----------------------------------------------------------------- */
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:12.5px;
  font-family:ui-monospace,monospace; font-variant-numeric:tabular-nums; }
th,td { padding:5px 9px; text-align:right; white-space:nowrap;
  border-bottom:1px solid var(--line); }
th { position:sticky; top:0; background:var(--surface); z-index:2; cursor:pointer;
  font-weight:650; color:var(--muted); font-size:11px; letter-spacing:.04em;
  border-bottom:1px solid var(--line-strong); user-select:none; }
th:hover { color:var(--accent); }
th[aria-sort]::after { content:" \2195"; opacity:.35; }
th[aria-sort="ascending"]::after { content:" \25B2"; opacity:1; color:var(--accent); }
th[aria-sort="descending"]::after { content:" \25BC"; opacity:1; color:var(--accent); }
td.txt, th.txt { text-align:left; }
tbody tr:hover td { background:var(--sunk); }
tr.is-baseline td { background:var(--accent-soft); }
tr.is-baseline td.txt::before { content:"\25B8 "; color:var(--accent); }
.swatch { display:inline-block; width:10px; height:10px; border-radius:2px;
  margin-right:6px; vertical-align:-1px; }
.pos { color:var(--pos); } .neg { color:var(--neg); }
.dim { color:var(--faint); }
.tag { font-size:10.5px; padding:1px 6px; border-radius:4px; border:1px solid var(--line-strong);
  color:var(--muted); }
.tag-prop { border-color:var(--accent); color:var(--accent); background:var(--accent-soft); }
.tag-warn { border-color:var(--neg); color:var(--neg); background:var(--neg-soft); }
.tag-flag { border-color:var(--flag); color:var(--flag); background:var(--flag-soft); }
.count { font-size:12px; color:var(--muted); font-family:ui-monospace,monospace; }

/* heatmap ---------------------------------------------------------------- */
.heat { border-collapse:separate; border-spacing:2px; font-size:10.5px; }
.heat td,.heat th { border:none; padding:4px 5px; border-radius:3px; }
.heat th.rowlab { text-align:right; color:var(--muted); background:none; position:static;
  cursor:pointer; }
.heat th.collab { writing-mode:vertical-rl; transform:rotate(180deg); text-align:left;
  color:var(--muted); background:none; position:static; cursor:pointer; height:74px; }
.heat th.rowlab:hover,.heat th.collab:hover { color:var(--neg); text-decoration:line-through; }
.heat td { text-align:center; color:#131820; cursor:default; }
.heat td.dimmed { opacity:.16; }
.heat td.hl { outline:2px solid var(--ink); outline-offset:-2px; }
.matrices { display:flex; gap:26px; flex-wrap:wrap; align-items:flex-start; }
.matrix-block { flex:1 1 380px; min-width:0; }
.matrix-block h3 { margin-bottom:8px; }
.scale { display:flex; align-items:center; gap:8px; font-size:11px; color:var(--muted);
  font-family:ui-monospace,monospace; margin-top:10px; }
.ramp { height:9px; width:150px; border-radius:3px; border:1px solid var(--line);
  background:linear-gradient(90deg,#ecf4f7,#c7e1e8,#98ccd7,#60afc1,#2a8ca3,#10657a,#083f4f); }

/* chart ------------------------------------------------------------------ */
.chart-wrap { position:relative; }
svg.chart { display:block; width:100%; height:auto; overflow:visible; }
svg.chart .grid line { stroke:var(--grid); stroke-width:1; }
svg.chart .axis text { fill:var(--muted); font-size:10px;
  font-family:ui-monospace,monospace; }
svg.chart text.panel { fill:var(--faint); font-size:9.5px; letter-spacing:.12em; }
svg.chart .axis line { stroke:var(--line-strong); }
svg.chart path.series { fill:none; stroke-width:2; stroke-linejoin:round;
  stroke-linecap:round; }
svg.chart path.context { fill:none; stroke:var(--context); stroke-width:1;
  opacity:.55; }
svg.chart path.baseline { fill:none; stroke:var(--ink); stroke-width:3; }
svg.chart .zero { stroke:var(--line-strong); stroke-dasharray:3 3; }
svg.chart text.endlab { font-size:10.5px; font-family:ui-monospace,monospace;
  fill:var(--ink); }
.legend, .legend-static { display:flex; flex-wrap:wrap; gap:5px 8px; margin-top:12px; }
.legend-static .lg { cursor:default; }
.lg { display:flex; align-items:center; gap:7px; font-size:11.5px; padding:3px 9px;
  border-radius:999px; border:1px solid var(--line); background:var(--sunk);
  cursor:pointer; font-family:ui-monospace,monospace; color:var(--ink); }
.lg:hover { border-color:var(--accent); }
.lg.off { opacity:.4; }
.lg .key { width:16px; height:0; border-top-width:2.5px; border-top-style:solid;
  flex:none; }
.lg .stat { color:var(--muted); }
.tip { position:absolute; pointer-events:none; background:var(--surface);
  border:1px solid var(--line-strong); border-radius:8px; box-shadow:var(--shadow);
  padding:8px 10px; font-size:11.5px; font-family:ui-monospace,monospace;
  z-index:20; min-width:170px; opacity:0; transition:opacity .08s; }
.tip.on { opacity:1; }
.tip .tdate { color:var(--muted); font-size:10.5px; letter-spacing:.06em;
  text-transform:uppercase; margin-bottom:5px; }
.tip .trow { display:flex; align-items:center; gap:7px; justify-content:space-between; }
.tip .trow .k { width:14px; border-top-width:2.5px; border-top-style:solid; flex:none; }
.tip .trow .v { font-weight:650; margin-left:auto; }
.tip .trow .n { color:var(--muted); overflow:hidden; text-overflow:ellipsis;
  max-width:150px; }

pre.cmd { background:var(--sunk); border:1px solid var(--line); border-radius:8px;
  padding:10px 12px; overflow-x:auto; font-size:11.5px; margin:8px 0 0; }
.warnbox { border-left:3px solid var(--neg); background:var(--neg-soft);
  padding:10px 14px; border-radius:0 8px 8px 0; font-size:12.5px; margin-bottom:14px; }
.warnbox b { color:var(--neg); }
ul.plain { margin:6px 0 0; padding-left:18px; font-size:12.5px; color:var(--muted); }
ul.plain li { margin:3px 0; }
noscript .warnbox { display:block; margin-bottom:20px; }
/* While a coalesced re-render is pending the previous one is held at reduced
   opacity: no skeleton, no layout jump, and the reader can see the control
   took effect before the numbers catch up. */
body.stale .card-body { opacity:.55; transition:opacity .1s; }
.foot { color:var(--faint); font-size:12px; border-top:1px solid var(--line);
  padding-top:16px; }
"""

# --------------------------------------------------------------------------- #
# the compute half: pure functions, no DOM, ES2015 so node can unit-test it
# --------------------------------------------------------------------------- #

COMPUTE_JS = r"""
/* Every figure on the page comes from here, so the sections cannot disagree.
   Mirrors src/analysis/report_data.py and src/prediction/metrics.py exactly:
   std with ddof=0, drawdown POSITIVE on the CUMULATIVE series, payout rebuilt
   per period as 0.1*corr + mmc and summarised once. */

var CORR_WEIGHT = 0.1;

function summarize(values) {
  var n = values.length;
  if (!n) return { mean: NaN, std: NaN, sharpe: NaN, maxDrawdown: NaN, n: 0 };
  var i, sum = 0;
  for (i = 0; i < n; i++) sum += values[i];
  var mean = sum / n, acc = 0;
  for (i = 0; i < n; i++) acc += (values[i] - mean) * (values[i] - mean);
  var std = Math.sqrt(acc / n);                       /* ddof = 0 */
  var cum = 0, peak = -Infinity, dd = 0;
  for (i = 0; i < n; i++) {
    cum += values[i];
    if (cum > peak) peak = cum;
    if (peak - cum > dd) dd = peak - cum;             /* positive: bigger is worse */
  }
  return { mean: mean, std: std, sharpe: std > 0 ? mean / std : NaN,
           maxDrawdown: dd, n: n };
}

/* corr for one period, with a DEGENERATE period charged the 0 it earned.

   A null corr beside a scored mmc cannot mean "unscoreable": mmc came off the
   same rows against the reference, so the period had rows. It means the arm
   predicted one constant value and has no correlation. Dropping the period
   would score the arm only on the days it worked. Mirrors
   results_table.charged_corr -- the node test asserts the two agree. */
function chargedCorr(arm, i) {
  var c = arm.corr[i];
  if (c !== null) return c;
  if (!arm.mmc) return null;
  return arm.mmc[i] === null ? null : 0;
}

/* The payout series of one arm over [i0, i1). A period counts only when BOTH
   corr and mmc are scored -- the same rows pandas drops. */
function payoutSeries(arm, i0, i1) {
  var out = [], i, c, m;
  for (i = i0; i < i1; i++) {
    c = chargedCorr(arm, i); m = arm.mmc ? arm.mmc[i] : 0;
    if (c === null || m === null) continue;
    out.push(CORR_WEIGHT * c + m);
  }
  return out;
}

/* corr over [i0,i1) with degenerate periods charged 0. See chargedCorr. */
function chargedColumn(arm, i0, i1) {
  var out = [], i, v;
  for (i = i0; i < i1; i++) { v = chargedCorr(arm, i); if (v !== null) out.push(v); }
  return out;
}

function column(arm, key, i0, i1) {
  var out = [], i, v, src = arm[key];
  if (!src) return out;
  for (i = i0; i < i1; i++) { v = src[i]; if (v !== null) out.push(v); }
  return out;
}

/* One table row, matching report_data.row_from_frame column for column. */
function rowFor(arm, i0, i1) {
  var corr = summarize(chargedColumn(arm, i0, i1));
  var mmc = summarize(column(arm, "mmc", i0, i1));
  var pay = summarize(payoutSeries(arm, i0, i1));
  var calmar = pay.maxDrawdown > 0 ? pay.mean / pay.maxDrawdown : Infinity;
  return {
    run: arm.label, exp: arm.exp, kind: arm.kind, tgt: arm.tgt, n: pay.n,
    "n(own)": arm.nOwn,
    CORR: corr.mean, MMC: mmc.mean, "c.sharpe": corr.sharpe,
    "dd(corr)": corr.maxDrawdown, payout: pay.mean, "p.sharpe": pay.sharpe,
    "dd(pay)": pay.maxDrawdown, calmar: calmar, dpay: null, t: null,
    live: arm.live, staked: arm.staked, model: arm.model, runName: arm.run
  };
}

/* Newey-West, the ablation_report implementation: (mean, se, t). */
function neweyWest(x, lag) {
  var n = x.length, i, k, sum = 0;
  if (n < 3) return { mean: NaN, se: NaN, t: NaN };
  for (i = 0; i < n; i++) sum += x[i];
  var mean = sum / n, e = new Array(n);
  for (i = 0; i < n; i++) e[i] = x[i] - mean;
  var variance = 0;
  for (i = 0; i < n; i++) variance += e[i] * e[i];
  variance /= n;
  var top = Math.min(lag, n - 1), acc;
  for (k = 1; k <= top; k++) {
    acc = 0;
    for (i = k; i < n; i++) acc += e[i] * e[i - k];
    variance += 2 * (1 - k / (lag + 1)) * (acc / n);
  }
  var se = Math.sqrt(Math.max(variance, 0) / n);
  return { mean: mean, se: se, t: se > 0 ? mean / se : NaN };
}

/* The paired delta: an INNER join on periods, so a period either arm failed to
   score drops out of both rather than silently biasing one side. */
function pairedDelta(arm, base, i0, i1, lag) {
  var diff = [], i, ac, am, bc, bm;
  for (i = i0; i < i1; i++) {
    ac = chargedCorr(arm, i); am = arm.mmc ? arm.mmc[i] : 0;
    bc = chargedCorr(base, i); bm = base.mmc ? base.mmc[i] : 0;
    if (ac === null || am === null || bc === null || bm === null) continue;
    diff.push((CORR_WEIGHT * ac + am) - (CORR_WEIGHT * bc + bm));
  }
  return neweyWest(diff, lag);
}

function pearson(a, b) {
  var n = Math.min(a.length, b.length), i, sa = 0, sb = 0;
  if (n < 2) return NaN;
  for (i = 0; i < n; i++) { sa += a[i]; sb += b[i]; }
  var ma = sa / n, mb = sb / n, num = 0, da = 0, db = 0, x, y;
  for (i = 0; i < n; i++) {
    x = a[i] - ma; y = b[i] - mb;
    num += x * y; da += x * x; db += y * y;
  }
  return (da > 0 && db > 0) ? num / Math.sqrt(da * db) : NaN;
}

/* Correlation of the per-period CORR series -- the ensembling ceiling. Every
   cell uses the same periods: rows where any arm is unscored are dropped once,
   up front, rather than pairwise. */
function scoreMatrix(arms, i0, i1) {
  var cols = [], i, j, a, ok;
  var keep = [];
  for (i = i0; i < i1; i++) {
    ok = true;
    for (j = 0; j < arms.length; j++) if (arms[j].corr[i] === null) { ok = false; break; }
    if (ok) keep.push(i);
  }
  for (j = 0; j < arms.length; j++) {
    a = [];
    for (i = 0; i < keep.length; i++) a.push(arms[j].corr[keep[i]]);
    cols.push(a);
  }
  var m = [];
  for (i = 0; i < arms.length; i++) {
    m.push([]);
    for (j = 0; j < arms.length; j++) {
      m[i].push(i === j ? 1 : pearson(cols[i], cols[j]));
    }
  }
  return { matrix: m, n: keep.length };
}

function combinations(n, k) {
  var out = [], idx = [], i;
  for (i = 0; i < k; i++) idx.push(i);
  while (true) {
    out.push(idx.slice());
    i = k - 1;
    while (i >= 0 && idx[i] === n - k + i) i--;
    if (i < 0) break;
    idx[i]++;
    for (var j = i + 1; j < k; j++) idx[j] = idx[j - 1] + 1;
  }
  return out;
}

/* Equal-weight screening, scored against each combination's OWN BEST MEMBER.
   Mirrors ensemble_screen.screen: the question is "does combining beat just
   holding the best of these", and equal-weight averaging dilutes the mean by
   construction, so a global baseline would flatter every blend. */
/* Which arms are worth screening. Combinations grow as the cube of the arm
   count -- 37 arms is 8,436 of them, and the page shows 32. The other 8,404
   are mostly pairings nobody would deploy together, and computing them cost a
   third of a second per keystroke. Keep the proposals and the baseline first,
   spend the rest of the cap on the highest-paying slots, and say what was cut. */
function ensembleArms(arms, baseline, cap, i0, i1) {
  if (arms.length <= cap) return { arms: arms, cut: 0 };
  var scored = arms.map(function (a) {
    var p = payoutSeries(a, i0, i1);
    return { arm: a, mean: p.length ? summarize(p).mean : -Infinity };
  }).sort(function (x, y) { return y.mean - x.mean; });

  var others = scored.filter(function (s) { return s.arm.kind !== "prop"; });
  var keep = [], seen = {};
  function take(s) { keep.push(s.arm); seen[s.arm.label] = 1; }

  scored.forEach(function (s) {
    if (s.arm.label === baseline && !seen[s.arm.label]) take(s);
  });
  /* Half the room is reserved for the deployed book, mirroring
     report_data.matrix_arms: a study with twenty arms must not crowd out the
     slots it is being compared against. */
  var reserved = Math.min(others.length, Math.floor(cap / 2));
  scored.forEach(function (s) {
    if (s.arm.kind === "prop" && !seen[s.arm.label] && keep.length < cap - reserved) take(s);
  });
  others.forEach(function (s) { if (!seen[s.arm.label] && keep.length < cap) take(s); });
  scored.forEach(function (s) { if (!seen[s.arm.label] && keep.length < cap) take(s); });
  return { arms: keep, cut: arms.length - keep.length };
}

function screenEnsembles(arms, i0, i1, sizes, lag, cap) {
  var series = [], solo = [], i, j;
  var keep = [];
  for (i = i0; i < i1; i++) {
    var ok = true;
    for (j = 0; j < arms.length; j++) {
      if (chargedCorr(arms[j], i) === null || (arms[j].mmc && arms[j].mmc[i] === null)) {
        ok = false; break;
      }
    }
    if (ok) keep.push(i);
  }
  for (j = 0; j < arms.length; j++) {
    var s = new Array(keep.length);
    for (i = 0; i < keep.length; i++) {
      s[i] = CORR_WEIGHT * chargedCorr(arms[j], keep[i]) +
             (arms[j].mmc ? arms[j].mmc[keep[i]] : 0);
    }
    series.push(s);
    solo.push(summarize(s));
  }

  var rho = [];
  for (i = 0; i < arms.length; i++) {
    rho.push([]);
    for (j = 0; j < arms.length; j++) rho[i].push(i === j ? 1 : pearson(series[i], series[j]));
  }

  var rows = [], truncated = 0;
  /* Two scratch buffers reused across every combination. Allocating a pair of
     470-element arrays per combination was tens of megabytes of garbage per
     pass, which is what made a drag feel like a leak. */
  var blend = new Float64Array(keep.length), diff = new Float64Array(keep.length);
  for (var si = 0; si < sizes.length; si++) {
    var combos = combinations(arms.length, sizes[si]);
    for (var ci = 0; ci < combos.length; ci++) {
      if (rows.length >= cap) { truncated += combos.length - ci; break; }
      var c = combos[ci], best = c[0], r = 0, pairs = 0;
      for (i = 0; i < keep.length; i++) {
        var acc = 0;
        for (j = 0; j < c.length; j++) acc += series[c[j]][i];
        blend[i] = acc / c.length;
      }
      for (j = 1; j < c.length; j++) if (solo[c[j]].mean > solo[best].mean) best = c[j];
      for (i = 0; i < keep.length; i++) diff[i] = blend[i] - series[best][i];
      var st = summarize(blend), nw = neweyWest(diff, lag);
      for (i = 0; i < c.length; i++) {
        for (j = i + 1; j < c.length; j++) { r += rho[c[i]][c[j]]; pairs++; }
      }
      rows.push({
        members: c.map(function (k) { return arms[k].label; }),
        payout: st.mean, sharpe: st.sharpe, maxdd: st.maxDrawdown,
        best: arms[best].label, delta: nw.mean, t: nw.t, rho: pairs ? r / pairs : NaN
      });
    }
  }
  return { rows: rows, n: keep.length, truncated: truncated };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { summarize: summarize, rowFor: rowFor, neweyWest: neweyWest,
                     pairedDelta: pairedDelta, payoutSeries: payoutSeries,
                     scoreMatrix: scoreMatrix, screenEnsembles: screenEnsembles,
                     ensembleArms: ensembleArms, pearson: pearson };
}
"""

# --------------------------------------------------------------------------- #
# the UI half
# --------------------------------------------------------------------------- #

UI_JS = r"""
var R = window.REPORT;
var STATE = { i0: 0, i1: 0, search: "", kinds: {}, cols: {}, hidden: {},
              corrThreshold: 0, showContext: true, sort: {} };

function el(tag, cls, text) {
  var n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = String(text);
  return n;
}
function fmt(v, spec) {
  if (v === null || v === undefined || (typeof v === "number" && isNaN(v))) return "-";
  if (v === Infinity) return "inf";
  if (v === -Infinity) return "-inf";
  if (typeof v !== "number") return String(v);
  if (spec === "d") return String(Math.round(v));
  if (spec === "+.6f") return (v >= 0 ? "+" : "") + v.toFixed(6);
  if (spec === "+.2f") return (v >= 0 ? "+" : "") + v.toFixed(2);
  var d = parseInt(spec.replace(/[^0-9]/g, ""), 10);
  return v.toFixed(isNaN(d) ? 3 : d);
}
/* A CSS custom property rather than a literal: the token already carries the
   light and dark step of the same slot, so the theme toggle needs no repaint
   pass and the server-rendered SVG follows the theme too. Colour is keyed to
   the arm's fixed position, never to its current rank -- a reader who has
   learnt that cumlag3 is blue must not find it repainted because they hid an
   unrelated series. */
function seriesColor(i) { return "var(--s" + (i % 8 + 1) + ")"; }
function seriesDash(i) { return R.dashes[Math.floor(i / 8) % R.dashes.length]; }

/* ---- date window ------------------------------------------------------- */
function allDates() { return R.dates; }
function clampWindow() {
  if (STATE.i1 <= STATE.i0) STATE.i1 = Math.min(STATE.i0 + 2, R.dates.length);
}
function blockWindow(b) {
  /* Map the global date window onto this block's own date list. */
  var lo = R.dates[STATE.i0], hi = R.dates[STATE.i1 - 1], i0 = -1, i1 = 0;
  for (var i = 0; i < b.dates.length; i++) {
    if (b.dates[i] >= lo && b.dates[i] <= hi) { if (i0 < 0) i0 = i; i1 = i + 1; }
  }
  return i0 < 0 ? [0, 0] : [i0, i1];
}

/* ---- summary table ----------------------------------------------------- */
function visibleArms(b) {
  var q = STATE.search.toLowerCase();
  return b.arms.filter(function (a) {
    if (STATE.hidden[b.id + "|" + a.label]) return false;
    if (q && a.label.toLowerCase().indexOf(q) < 0) return false;
    var k = a.kind;
    if (STATE.kinds.prop === false && k === "prop") return false;
    if (STATE.kinds.sub === false && k === "sub") return false;
    if (STATE.kinds.blend === false && (k === "blend" || k === "slotblend")) return false;
    if (STATE.kinds.stakedOnly && !a.staked) return false;
    if (STATE.kinds.faultsOnly && (a.live === "ok" || a.live === "-")) return false;
    return true;
  });
}

function buildRows(b) {
  var w = blockWindow(b), arms = visibleArms(b);
  var base = null, i;
  for (i = 0; i < b.arms.length; i++) if (b.arms[i].label === b.baseline) base = b.arms[i];
  var rows = arms.map(function (a) {
    var r = rowFor(a, w[0], w[1]);
    if (base && a.label !== b.baseline) {
      var nw = pairedDelta(a, base, w[0], w[1], b.lag);
      r.dpay = nw.mean; r.t = nw.t;
    }
    return r;
  });
  var s = STATE.sort[b.id];
  if (s) {
    rows.sort(function (x, y) {
      var a = x[s.key], c = y[s.key];
      if (typeof a === "string" || typeof c === "string") {
        a = String(a); c = String(c);
        return s.dir * a.localeCompare(c);
      }
      if (a === null || a === undefined || isNaN(a)) a = -Infinity;
      if (c === null || c === undefined || isNaN(c)) c = -Infinity;
      return s.dir * (a - c);
    });
  }
  return { rows: rows, window: w, total: b.arms.length };
}

function renderTable(b) {
  var host = document.getElementById("tbl-" + b.id);
  var built = buildRows(b);
  host.textContent = "";
  var table = el("table"), thead = el("thead"), tr = el("tr");

  R.columns.forEach(function (c) {
    if (STATE.cols[c[0]] === false) return;
    var th = el("th", c[2] === null ? "txt" : "", c[0]);
    var s = STATE.sort[b.id];
    th.setAttribute("aria-sort", s && s.key === c[1]
      ? (s.dir === 1 ? "ascending" : "descending") : "none");
    th.addEventListener("click", function () {
      var cur = STATE.sort[b.id];
      STATE.sort[b.id] = (cur && cur.key === c[1] && cur.dir === -1)
        ? { key: c[1], dir: 1 } : { key: c[1], dir: -1 };
      renderTable(b);
    });
    tr.appendChild(th);
  });
  thead.appendChild(tr); table.appendChild(thead);

  var tbody = el("tbody");
  built.rows.forEach(function (r) {
    var row = el("tr");
    if (r.run === b.baseline) row.className = "is-baseline";
    R.columns.forEach(function (c) {
      if (STATE.cols[c[0]] === false) return;
      var key = c[1], td;
      if (key === "run") {
        td = el("td", "txt");
        var idx = b.order.indexOf(r.run);
        if (idx >= 0) {
          var sw = el("span", "swatch");
          sw.style.background = seriesColor(idx);
          td.appendChild(sw);
        }
        td.appendChild(document.createTextNode(r.run));
      } else if (key === "kind") {
        td = el("td", "txt");
        var tag = el("span", "tag" + (r.kind === "prop" ? " tag-prop" : ""), r.kind);
        td.appendChild(tag);
        if (r.staked) td.appendChild(el("span", "tag tag-flag", "$"));
      } else if (key === "live") {
        td = el("td", "txt");
        var bad = r.live !== "ok" && r.live !== "-";
        td.appendChild(el("span", "tag" + (bad ? " tag-warn" : ""), r.live));
      } else {
        td = el("td", null, fmt(r[key], c[2]));
        if (key === "dpay" || key === "t") {
          if (typeof r[key] === "number" && !isNaN(r[key])) {
            td.className = r[key] > 0 ? "pos" : "neg";
          }
        }
      }
      row.appendChild(td);
    });
    tbody.appendChild(row);
  });
  table.appendChild(tbody);
  host.appendChild(table);

  var w = built.window;
  var text = "showing " + built.rows.length + " of " + built.total + " arms  |  " +
    (w[1] - w[0]) + " periods, " + (b.dates[w[0]] || "-") + " to " +
    (b.dates[w[1] - 1] || "-");
  document.getElementById("cnt-" + b.id).textContent = text;
}

/* ---- correlation ------------------------------------------------------- */
/* Correlation here is a MAGNITUDE -- "how redundant are these two" -- so it gets
   a sequential single-hue ramp, light to dark. A multi-hue blue-yellow-red ramp
   would be a rainbow doing a magnitude's job: readers infer category boundaries
   at every hue change that are not in the data. The one case that is genuinely
   diverging is a matrix holding a negative correlation, where zero means "no
   relationship" and the sign matters; that gets two opposed hues around a
   neutral gray. */
var SEQ = R.ramps.seq, DIV_NEG = R.ramps.divNeg,
    DIV_MID = R.ramps.divMid, DIV_POS = R.ramps.divPos;

function ramp(stops, t) {
  var x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  var i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  return [0, 1, 2].map(function (k) {
    return Math.round(stops[i][k] + f * (stops[i + 1][k] - stops[i][k]));
  });
}
function heatRGB(v, lo, diverging) {
  if (isNaN(v)) return [200, 200, 200];
  if (diverging) {
    if (v >= 0) return ramp([DIV_MID].concat(DIV_POS), v);
    return ramp([DIV_MID].concat(DIV_NEG), -v);
  }
  return ramp(SEQ, (v - lo) / Math.max(1e-9, 1 - lo));
}
function heatColor(v, lo, diverging) {
  return "rgb(" + heatRGB(v, lo, diverging).join(",") + ")";
}
/* Cell text takes its colour from the cell it sits on, not from the theme:
   a fixed ink colour is unreadable on the dark end of any ramp. */
function inkOn(rgb) {
  var l = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255;
  return l > 0.55 ? "#131820" : "#FFFFFF";
}

function renderHeat(host, labels, matrix, caption) {
  host.textContent = "";
  if (!labels.length) { host.appendChild(el("p", "note", "nothing to correlate here.")); return; }
  var lo = 1, diverging = false;
  matrix.forEach(function (r) {
    r.forEach(function (v) { if (v < lo) lo = v; if (v < 0) diverging = true; });
  });
  lo = Math.min(lo, 0.95);

  var table = el("table", "heat"), thead = el("thead"), tr = el("tr");
  tr.appendChild(el("th"));
  labels.forEach(function (l) { tr.appendChild(el("th", "collab", l)); });
  thead.appendChild(tr); table.appendChild(thead);
  var tbody = el("tbody");
  labels.forEach(function (l, i) {
    var row = el("tr");
    row.appendChild(el("th", "rowlab", l));
    labels.forEach(function (l2, j) {
      var v = matrix[i][j];
      var td = el("td", null, isNaN(v) ? "-" : v.toFixed(3));
      var rgb = heatRGB(v, lo, diverging);
      td.style.background = "rgb(" + rgb.join(",") + ")";
      td.style.color = inkOn(rgb);
      if (Math.abs(v) < STATE.corrThreshold && i !== j) td.className = "dimmed";
      td.title = l + "  vs  " + l2 + " = " + (isNaN(v) ? "-" : v.toFixed(4));
      td.addEventListener("mouseenter", function () {
        table.querySelectorAll("td").forEach(function (c) { c.classList.remove("hl"); });
        for (var k = 0; k < labels.length; k++) {
          tbody.rows[i].cells[k + 1].classList.add("hl");
          tbody.rows[k].cells[j + 1].classList.add("hl");
        }
      });
      row.appendChild(td);
    });
    tbody.appendChild(row);
  });
  table.appendChild(tbody);
  var scroll = el("div", "scroll"); scroll.appendChild(table);
  host.appendChild(scroll);
  if (caption) host.appendChild(el("p", "note", caption));
}

function renderCorrelation(b) {
  if (!b.predCorr) return;
  var w = blockWindow(b);
  var arms = b.predCorr.labels.map(function (l) {
    for (var i = 0; i < b.arms.length; i++) if (b.arms[i].label === l) return b.arms[i];
    return null;
  }).filter(Boolean);
  var shown = arms.filter(function (a) { return !STATE.hidden[b.id + "|" + a.label]; });
  var idx = shown.map(function (a) { return b.predCorr.labels.indexOf(a.label); });
  var labels = shown.map(function (a) { return a.short; });

  var pred = idx.map(function (i) { return idx.map(function (j) { return b.predCorr.matrix[i][j]; }); });
  renderHeat(document.getElementById("pred-" + b.id), labels, pred,
    "Prediction correlation over the FULL window (" + b.predCorr.nDates +
    " dates). It needs the row-level forecasts, so the date filter does not " +
    "reach it. Click a label to drop an arm.");

  var sc = scoreMatrix(shown, w[0], w[1]);
  renderHeat(document.getElementById("score-" + b.id), labels, sc.matrix,
    "Score correlation on the selected window (n=" + sc.n + "). This is the one " +
    "that caps ensembling, and it does follow the date filter.");

  [["pred-", pred], ["score-", sc.matrix]].forEach(function (pair) {
    var host = document.getElementById(pair[0] + b.id);
    host.querySelectorAll("th.rowlab, th.collab").forEach(function (th, k) {
      var label = shown[k % shown.length].label;
      th.addEventListener("click", function () {
        STATE.hidden[b.id + "|" + label] = true;
        scheduleRender(0);
      });
    });
  });
}

/* ---- ensembles --------------------------------------------------------- */
function renderEnsembles(b) {
  var host = document.getElementById("ens-" + b.id);
  var w = blockWindow(b), arms = visibleArms(b);
  host.textContent = "";
  if (arms.length < 2) {
    host.appendChild(el("p", "note", "fewer than two arms selected -- nothing to screen."));
    return;
  }
  var picked = ensembleArms(arms, b.baseline, R.ensMaxArms, w[0], w[1]);
  var res = screenEnsembles(picked.arms, w[0], w[1], R.ensSizes, b.lag, R.ensCap);
  var rows = res.rows.slice();
  var key = (STATE.sort["ens-" + b.id] || { key: "sharpe", dir: -1 });
  rows.sort(function (x, y) {
    var a = x[key.key], c = y[key.key];
    if (typeof a === "string") return key.dir * a.localeCompare(c);
    if (isNaN(a)) a = -Infinity;
    if (isNaN(c)) c = -Infinity;
    return key.dir * (a - c);
  });
  var beat = res.rows.filter(function (r) { return r.t > 2; }).length;
  var shown = rows.slice(0, R.ensShow);

  var cols = [["ensemble", "members", null], ["screened payout", "payout", ".6f"],
              ["sharpe", "sharpe", ".4f"], ["maxdd", "maxdd", ".4f"],
              ["d vs best", "delta", "+.6f"], ["t", "t", "+.2f"], ["rho", "rho", ".2f"]];
  var table = el("table"), thead = el("thead"), tr = el("tr");
  cols.forEach(function (c) {
    var th = el("th", c[2] === null ? "txt" : "", c[0]);
    th.setAttribute("aria-sort", key.key === c[1]
      ? (key.dir === 1 ? "ascending" : "descending") : "none");
    th.addEventListener("click", function () {
      var cur = STATE.sort["ens-" + b.id];
      STATE.sort["ens-" + b.id] = (cur && cur.key === c[1] && cur.dir === -1)
        ? { key: c[1], dir: 1 } : { key: c[1], dir: -1 };
      renderEnsembles(b);
    });
    tr.appendChild(th);
  });
  thead.appendChild(tr); table.appendChild(thead);
  var tbody = el("tbody");
  shown.forEach(function (r) {
    var row = el("tr");
    cols.forEach(function (c) {
      var td;
      if (c[1] === "members") {
        td = el("td", "txt", r.members.join("  +  "));
      } else {
        td = el("td", null, fmt(r[c[1]], c[2]));
        if (c[1] === "delta" || c[1] === "t") td.className = r[c[1]] > 0 ? "pos" : "neg";
      }
      row.appendChild(td);
    });
    tbody.appendChild(row);
  });
  table.appendChild(tbody);
  var scroll = el("div", "scroll"); scroll.appendChild(table);
  host.appendChild(scroll);

  var line = beat + " of " + res.rows.length +
    " combinations beat their own best member on payout at |t| > 2.";
  if (!beat) {
    line += " None. Equal-weight averaging dilutes the mean by construction, so read" +
            " these as a RISK decision (sharpe, drawdown, correlation), not a level" +
            " one. That null is the finding, not a missing result.";
  }
  host.appendChild(el("p", "note", line));
  var foot = "showing top " + shown.length + " of " + res.rows.length +
    " over " + picked.arms.length + " arms, " + res.n + " periods.";
  if (picked.cut) {
    foot += "  " + picked.cut + " lower-paying arms were left out of the screen " +
      "(cap " + R.ensMaxArms + "); combinations grow as the cube of the arm count. " +
      "Raise it with --ens-max-arms.";
  }
  if (res.truncated) {
    foot += "  " + res.truncated + " combinations were not enumerated (cap " +
            R.ensCap + ") -- narrow the arm set to screen them all.";
  }
  host.appendChild(el("p", "note", foot));
}

/* ---- chart ------------------------------------------------------------- */
function chartSeries(b) {
  var w = blockWindow(b);
  var out = [];
  b.arms.forEach(function (a) {
    if (STATE.hidden[b.id + "|" + a.label]) return;
    if (!b.chartable[a.label]) return;
    var pay = payoutSeries(a, w[0], w[1]);
    if (!pay.length) return;
    var cum = [], run = 0, i, peak = -Infinity, under = [];
    for (i = 0; i < pay.length; i++) {
      run += pay[i]; cum.push(run);
      if (run > peak) peak = run;
      under.push(peak - run);
    }
    var st = summarize(pay);
    out.push({ label: a.label, kind: a.kind, cum: cum, under: under, stats: st,
               order: b.order.indexOf(a.label),
               isBaseline: a.label === b.baseline });
  });
  return { series: out, dates: b.dates.slice(w[0], w[1]) };
}

function path(values, x, y) {
  var d = "", i;
  for (i = 0; i < values.length; i++) {
    d += (i ? "L" : "M") + Math.round(x(i)) + " " + Math.round(y(values[i]));
  }
  return d;
}

function renderChart(b) {
  var host = document.getElementById("chart-" + b.id);
  var legendHost = document.getElementById("legend-" + b.id);
  var stale = host.parentNode.querySelector(".legend-static");
  if (stale) stale.parentNode.removeChild(stale);
  var data = chartSeries(b);
  host.textContent = ""; legendHost.textContent = "";
  if (!data.series.length) {
    host.appendChild(el("p", "note", "no series selected."));
    return;
  }

  var G = R.chart;
  var W = G.W, H1 = G.H1, H2 = G.H2, GAP = G.gap;
  var PAD = { l: G.padL, r: G.padR, t: G.padT, b: G.padB };
  var H = PAD.t + H1 + GAP + H2 + PAD.b;
  var n = data.dates.length;
  var lo = Infinity, hi = -Infinity, ulo = 0;
  data.series.forEach(function (s) {
    s.cum.forEach(function (v) { if (v < lo) lo = v; if (v > hi) hi = v; });
    s.under.forEach(function (v) { if (v > ulo) ulo = v; });
  });
  if (lo > 0) lo = 0;
  var span = (hi - lo) || 1;
  var x = function (i) { return PAD.l + (n < 2 ? 0 : (i / (n - 1)) * (W - PAD.l - PAD.r)); };
  var y1 = function (v) { return PAD.t + H1 - ((v - lo) / span) * H1; };
  var y2 = function (v) { return PAD.t + H1 + GAP + ((v) / (ulo || 1)) * H2; };

  var NS = "http://www.w3.org/2000/svg";
  var svg = document.createElementNS(NS, "svg");
  svg.setAttribute("class", "chart");
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  svg.setAttribute("role", "img");

  function add(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    (parent || svg).appendChild(e);
    return e;
  }

  var grid = add("g", { "class": "grid" });
  var axis = add("g", { "class": "axis" });
  var ticks = 5, i, v;
  for (i = 0; i <= ticks; i++) {
    v = lo + (span * i) / ticks;
    add("line", { x1: PAD.l, x2: W - PAD.r, y1: y1(v), y2: y1(v) }, grid);
    add("text", { x: PAD.l - 8, y: y1(v) + 3, "text-anchor": "end" }, axis)
      .textContent = v.toFixed(1);
  }
  for (i = 0; i <= 2; i++) {
    v = ((ulo || 1) * i) / 2;
    add("line", { x1: PAD.l, x2: W - PAD.r, y1: y2(v), y2: y2(v) }, grid);
    add("text", { x: PAD.l - 8, y: y2(v) + 3, "text-anchor": "end" }, axis)
      .textContent = v.toFixed(1);
  }
  var xt = Math.min(6, n);
  for (i = 0; i < xt; i++) {
    var di = Math.round((i / Math.max(xt - 1, 1)) * (n - 1));
    add("text", { x: x(di), y: H - 6, "text-anchor": i === 0 ? "start" :
                  (i === xt - 1 ? "end" : "middle") }, axis)
      .textContent = data.dates[di];
  }
  /* Panel titles sit ABOVE their plots, not inside them: at x=12 they collided
     with the topmost tick label of the axis they were naming. */
  add("text", { x: PAD.l - 50, y: PAD.t - 12, "text-anchor": "start",
                "class": "panel" }, axis).textContent = "CUMULATIVE PAYOUT";
  add("text", { x: PAD.l - 50, y: PAD.t + H1 + GAP - 12, "text-anchor": "start",
                "class": "panel" }, axis).textContent = "UNDERWATER";

  var ordered = data.series.slice().sort(function (a, c) {
    return (a.isBaseline ? 2 : a.kind === "prop" ? 1 : 0) -
           (c.isBaseline ? 2 : c.kind === "prop" ? 1 : 0);
  });
  var direct = data.series.filter(function (s) {
    return s.kind === "prop" || s.isBaseline;
  }).length <= 4;

  ordered.forEach(function (s) {
    var isContext = s.kind !== "prop" && !s.isBaseline;
    if (isContext && !STATE.showContext) return;
    var cls = s.isBaseline ? "baseline" : (isContext ? "context" : "series");
    var p = add("path", { "class": cls, d: path(s.cum, x, y1) });
    var u = add("path", { "class": cls, d: path(s.under, x, y2) });
    if (!isContext && !s.isBaseline) {
      p.setAttribute("stroke", seriesColor(s.order));
      u.setAttribute("stroke", seriesColor(s.order));
      var dash = seriesDash(s.order);
      if (dash) { p.setAttribute("stroke-dasharray", dash); u.setAttribute("stroke-dasharray", dash); }
    }
    if (s.isBaseline) u.setAttribute("stroke-width", "2");
    if (direct && !isContext) {
      var t = add("text", { x: x(n - 1) + 6, y: y1(s.cum[n - 1]) + 3, "class": "endlab" });
      t.textContent = s.label;
      if (!s.isBaseline) t.setAttribute("fill", seriesColor(s.order));
    }
  });

  var hair = add("line", { "class": "zero", y1: PAD.t, y2: PAD.t + H1 + GAP + H2,
                           x1: -10, x2: -10 });
  host.appendChild(svg);

  var tip = el("div", "tip");
  host.appendChild(tip);
  svg.addEventListener("mousemove", function (ev) {
    var box = svg.getBoundingClientRect();
    var px = ((ev.clientX - box.left) / box.width) * W;
    var idx = Math.round(((px - PAD.l) / Math.max(1, W - PAD.l - PAD.r)) * (n - 1));
    idx = Math.max(0, Math.min(n - 1, idx));
    hair.setAttribute("x1", x(idx)); hair.setAttribute("x2", x(idx));
    tip.textContent = "";
    tip.appendChild(el("div", "tdate", data.dates[idx]));
    data.series.slice().sort(function (a, c) { return c.cum[idx] - a.cum[idx]; })
      .slice(0, 12).forEach(function (s) {
        var row = el("div", "trow");
        var k = el("span", "k");
        k.style.borderTopColor = s.isBaseline ? "var(--ink)"
          : (s.kind === "prop" ? seriesColor(s.order) : "var(--context)");
        row.appendChild(k);
        row.appendChild(el("span", "n", s.label));
        row.appendChild(el("span", "v", s.cum[idx].toFixed(2)));
        tip.appendChild(row);
      });
    tip.classList.add("on");
    var left = ev.clientX - box.left + 16;
    if (left > box.width - 200) left = ev.clientX - box.left - 200;
    tip.style.left = left + "px";
    tip.style.top = Math.max(0, ev.clientY - box.top - 30) + "px";
  });
  svg.addEventListener("mouseleave", function () {
    tip.classList.remove("on"); hair.setAttribute("x1", -10); hair.setAttribute("x2", -10);
  });

  data.series.slice().sort(function (a, c) { return c.stats.mean - a.stats.mean; })
    .forEach(function (s) {
      var chip = el("button", "lg");
      chip.type = "button";
      var key = el("span", "key");
      key.style.borderTopColor = s.isBaseline ? "var(--ink)"
        : (s.kind === "prop" ? seriesColor(s.order) : "var(--context)");
      if (!s.isBaseline && s.kind === "prop" && seriesDash(s.order)) {
        key.style.borderTopStyle = "dashed";
      }
      chip.appendChild(key);
      chip.appendChild(el("span", null, s.label + (s.isBaseline ? " (baseline)" : "")));
      chip.appendChild(el("span", "stat", fmt(s.stats.mean, ".5f") + " / " +
        fmt(s.stats.sharpe, ".2f") + " / " + fmt(s.stats.maxDrawdown, ".2f")));
      chip.addEventListener("click", function () {
        STATE.hidden[b.id + "|" + s.label] = true;
        scheduleRender(0);
      });
      chip.title = "click to hide this series";
      legendHost.appendChild(chip);
    });

  var hiddenNames = Object.keys(STATE.hidden).filter(function (k) {
    return STATE.hidden[k] && k.indexOf(b.id + "|") === 0;
  });
  if (hiddenNames.length) {
    var restore = el("button", "lg off");
    restore.type = "button";
    restore.appendChild(el("span", null, "restore " + hiddenNames.length + " hidden"));
    restore.addEventListener("click", function () {
      hiddenNames.forEach(function (k) { delete STATE.hidden[k]; });
      scheduleRender(0);
    });
    legendHost.appendChild(restore);
  }
}

/* ---- wiring ------------------------------------------------------------ */
function readout() {
  clampWindow();
  document.getElementById("range-readout").textContent =
    R.dates[STATE.i0] + "  to  " + R.dates[STATE.i1 - 1] +
    "   (" + (STATE.i1 - STATE.i0) + " dates)";
}

function renderAll() {
  readout();
  R.blocks.forEach(function (b) {
    if (document.getElementById("tbl-" + b.id)) renderTable(b);
    if (b.predCorr && document.getElementById("pred-" + b.id)) renderCorrelation(b);
    if (document.getElementById("ens-" + b.id)) renderEnsembles(b);
    if (document.getElementById("chart-" + b.id)) renderChart(b);
  });
  document.body.classList.remove("stale");
}

/* A full re-render walks every block: rows, both matrices, the ensemble screen
   and two SVG panels. A range slider fires `input` on EVERY pixel of a drag, so
   calling this directly queued hundreds of them, the main thread never
   returned, and the tab took the machine down with it. Coalesce instead: mark
   the page stale immediately so the reader sees the control responding, and do
   the work once, after they stop moving. */
var pendingRender = null;
function scheduleRender(delay) {
  readout();
  document.body.classList.add("stale");
  if (pendingRender) clearTimeout(pendingRender);
  pendingRender = setTimeout(function () {
    pendingRender = null;
    renderAll();
  }, delay === undefined ? 140 : delay);
}

function wire() {
  STATE.i0 = 0; STATE.i1 = R.dates.length;

  var s0 = document.getElementById("r0"), s1 = document.getElementById("r1");
  s0.max = R.dates.length - 1; s1.max = R.dates.length; s1.value = R.dates.length;
  function onRange() {
    STATE.i0 = Math.min(parseInt(s0.value, 10), R.dates.length - 2);
    STATE.i1 = Math.max(parseInt(s1.value, 10), STATE.i0 + 2);
    scheduleRender();
  }
  s0.addEventListener("input", onRange);
  s1.addEventListener("input", onRange);

  document.querySelectorAll("[data-months]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var m = parseInt(btn.getAttribute("data-months"), 10);
      if (m <= 0) { STATE.i0 = 0; STATE.i1 = R.dates.length; }
      else {
        var end = new Date(R.dates[R.dates.length - 1]);
        var cut = new Date(end.getTime());
        cut.setMonth(cut.getMonth() - m);
        var iso = cut.toISOString().slice(0, 10), k = 0;
        while (k < R.dates.length - 2 && R.dates[k] < iso) k++;
        STATE.i0 = k; STATE.i1 = R.dates.length;
      }
      s0.value = STATE.i0; s1.value = STATE.i1;
      document.querySelectorAll("[data-months]").forEach(function (o) {
        o.setAttribute("aria-pressed", o === btn ? "true" : "false");
      });
      scheduleRender(0);
    });
  });

  var search = document.getElementById("search");
  search.addEventListener("input", function () {
    STATE.search = search.value.trim(); scheduleRender();
  });

  document.querySelectorAll("[data-kind]").forEach(function (btn) {
    var k = btn.getAttribute("data-kind");
    STATE.kinds[k] = btn.getAttribute("aria-pressed") === "true";
    if (k === "stakedOnly" || k === "faultsOnly") STATE.kinds[k] = false;
    btn.addEventListener("click", function () {
      var on = btn.getAttribute("aria-pressed") === "true";
      btn.setAttribute("aria-pressed", on ? "false" : "true");
      STATE.kinds[k] = !on;
      scheduleRender(0);
    });
  });

  R.columns.forEach(function (c) { STATE.cols[c[0]] = true; });
  var menu = document.getElementById("colmenu");
  R.columns.forEach(function (c) {
    var label = el("label");
    var box = document.createElement("input");
    box.type = "checkbox"; box.checked = true;
    box.addEventListener("change", function () {
      STATE.cols[c[0]] = box.checked;
      R.blocks.forEach(function (b) {
        if (document.getElementById("tbl-" + b.id)) renderTable(b);
      });
    });
    label.appendChild(box);
    label.appendChild(document.createTextNode(c[0]));
    menu.appendChild(label);
  });

  var thr = document.getElementById("corr-threshold");
  var thrPending = null;
  thr.addEventListener("input", function () {
    STATE.corrThreshold = parseFloat(thr.value);
    document.getElementById("corr-threshold-out").textContent =
      thr.value === "0" ? "off" : "\u2265 " + parseFloat(thr.value).toFixed(2);
    /* Only the matrices depend on this one, so it never pays for a full pass. */
    if (thrPending) clearTimeout(thrPending);
    thrPending = setTimeout(function () {
      thrPending = null;
      R.blocks.forEach(function (b) {
        if (b.predCorr && document.getElementById("pred-" + b.id)) renderCorrelation(b);
      });
    }, 60);
  });

  var ctx = document.getElementById("show-context");
  if (ctx) {
    ctx.addEventListener("click", function () {
      STATE.showContext = ctx.getAttribute("aria-pressed") !== "true";
      ctx.setAttribute("aria-pressed", STATE.showContext ? "true" : "false");
      R.blocks.forEach(function (b) {
        if (document.getElementById("chart-" + b.id)) renderChart(b);
      });
    });
  }

  var theme = document.getElementById("theme");
  theme.addEventListener("click", function () {
    var cur = document.documentElement.getAttribute("data-theme");
    var next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    theme.textContent = next === "dark" ? "light mode" : "dark mode";
    renderAll();
  });

  renderAll();
}

document.addEventListener("DOMContentLoaded", wire);
"""


# --------------------------------------------------------------------------- #
# payload
# --------------------------------------------------------------------------- #

def _study_label(run_name):
    """
    "E14", "E14s" for a scout, or "-" for a run belonging to no study.

    :param run_name: the run's name
    :rtype: str
    """
    study, is_scout = study_of(run_name)
    if study is None:
        return "-"
    return study + "s" if is_scout else study


def arm_payload(block, arm):
    """
    One arm as the browser needs it: its per-period corr and mmc on the window.

    Emitted per period rather than pre-summarised because every figure on the
    page is recomputed against whatever date range the reader picks, and a
    summary cannot be re-based after the fact.

    :param block: the block the arm belongs to
    :type block: report_data.Block
    :param arm: the arm
    :type arm: report_data.Arm
    :return: the JSON-ready record
    :rtype: dict
    """
    frame = block.slice(arm).reindex(block.index)

    def series(name):
        if name not in frame:
            return None
        # Twelve decimals, not six: the browser re-runs the cumulative sum over
        # up to 510 periods, and rounding error accumulates through it into the
        # drawdown. Twelve keeps the browser's numbers within 1e-9 of Python's,
        # which is what src/tests/test_report_html.py asserts.
        return [None if v != v else round(float(v), 12) for v in frame[name]]

    return {
        "label": arm.label, "model": arm.model, "run": arm.run,
        "short": short_label(arm.label), "kind": arm.kind,
        "tgt": short_target(arm.target), "staked": bool(arm.staked),
        "live": arm.live_status, "nOwn": int(len(arm.payout)),
        # Which numbered study wrote this run. Resolved in Python and shipped
        # with the arm rather than derived in the browser, so the two renderers
        # cannot disagree about it -- the same reason every other figure here
        # comes from report_data.
        "exp": _study_label(arm.run),
        "corr": series("corr"), "mmc": series("mmc"),
    }


def block_payload(args, block):
    """
    One target block, everything the page derives from.

    :param args: parsed arguments
    :type args: argparse.Namespace
    :param block: the block
    :type block: report_data.Block
    :return: (payload, [(label, why)] excluded)
    :rtype: (dict, [(str, str)])
    """
    dates = [str(d) for d in block.index]
    arms = [arm_payload(block, a) for a in block.arms]
    order = [a.label for a in block.arms]

    excluded = []
    predictions = None
    if not args.no_corr:
        matrix, _, n_dates, _, dropped = correlation_matrices(
            args.tournament, block, args.corr_max_arms, args.min_common_dates)
        excluded.extend(dropped)
        if matrix is not None:
            predictions = {
                "labels": list(matrix.index),
                "matrix": [[round(float(v), 6) for v in row]
                           for row in matrix.to_numpy()],
                "nDates": int(n_dates),
            }

    # Which arms may be drawn: the chart's window is the baseline's, so an arm
    # sharing too few dates with it would be a curve on a different denominator.
    chartable = {}
    if block.baseline is not None:
        base = block.payout(block.baseline)
        for arm in block.arms:
            shared = block.payout(arm).index.intersection(base.index)
            chartable[arm.label] = len(shared) >= args.min_common_dates
            if not chartable[arm.label]:
                excluded.append((arm.label, "shares only %d dates with the baseline, "
                                            "so it is not drawn" % len(shared)))

    commands = []
    if not args.no_ensemble:
        rows, _, _ = screen({a.label: block.payout(a) for a in block.arms},
                            tuple(args.ens_size), block.lag)
        commands = price_commands(block, rows, args.price_top)

    return {
        "id": short_target(block.target), "target": block.target,
        "tgt": short_target(block.target), "lag": block.lag,
        "dates": dates, "arms": arms, "order": order,
        "baseline": block.baseline.label if block.baseline else None,
        "rejected": [list(r) for r in block.rejected],
        "predCorr": predictions, "chartable": chartable,
        "commands": commands,
    }, excluded


def esc(text):
    """
    Escape text for HTML.

    Run names come out of the filesystem and slot names out of the registry, so
    they are data rather than markup even though nothing hostile is expected in
    them today.

    :param text: any value
    :type text: object
    :return: the escaped text
    :rtype: str
    """
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# --------------------------------------------------------------------------- #
# server-rendered sections
#
# Every section below is written out as ordinary HTML with the numbers already
# in it, and the script REPLACES it on load. That is not belt and braces: this
# file gets opened in mail clients, in sandboxed preview panes and behind
# content-security policies that refuse inline script, and a report whose tables
# are empty in half of those is not a report. Without JavaScript you get a
# complete static board on the full window; with it you get the same board you
# can sort, filter and re-range.
# --------------------------------------------------------------------------- #

def cell_text(value, spec):
    """One formatted cell, with the repo's missing and infinite conventions."""
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:
            return "-"
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
    if spec is None:
        return str(value)
    if spec == "d":
        return str(int(round(value)))
    return format(value, spec)


def static_table(payload, rows, columns):
    """
    The results table, already filled in.

    :param payload: the block payload
    :type payload: dict
    :param rows: rows from Block.rows()
    :type rows: [dict]
    :param columns: (header, key, format) triples
    :type columns: [(str, str, str)]
    :return: the table HTML
    :rtype: str
    """
    head = "".join(
        '<th class="%s" aria-sort="none">%s</th>'
        % ("txt" if spec is None else "", esc(header))
        for header, _, spec in columns)

    body = []
    for row in rows:
        cells = []
        for header, key, spec in columns:
            if key == "run":
                index = payload["order"].index(row["run"]) if row["run"] in payload["order"] else -1
                swatch = ('<span class="swatch" style="background:var(--s%d)"></span>'
                          % (index % 8 + 1)) if index >= 0 else ""
                cells.append('<td class="txt">%s%s</td>' % (swatch, esc(row["run"])))
            elif key == "kind":
                kind = row.get("kind", "")
                staked = kind.endswith("$")
                kind = kind.rstrip("$")
                cells.append(
                    '<td class="txt"><span class="tag%s">%s</span>%s</td>'
                    % (" tag-prop" if kind == "prop" else "", esc(kind),
                       '<span class="tag tag-flag">$</span>' if staked else ""))
            elif key == "live":
                live = row.get("live", "-")
                bad = live not in ("ok", "-")
                cells.append('<td class="txt"><span class="tag%s">%s</span></td>'
                             % (" tag-warn" if bad else "", esc(live)))
            elif key == "tgt":
                cells.append('<td>%s</td>' % esc(short_target(row.get("target"))))
            else:
                value = row.get(key)
                klass = ""
                if key in ("dpay", "t") and isinstance(value, float) and value == value:
                    klass = ' class="%s"' % ("pos" if value > 0 else "neg")
                cells.append("<td%s>%s</td>" % (klass, esc(cell_text(value, spec))))
        body.append('<tr%s>%s</tr>'
                    % (' class="is-baseline"' if row["run"] == payload["baseline"] else "",
                       "".join(cells)))
    return ("<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>"
            % (head, "".join(body)))


def static_heat(labels, matrix):
    """
    A correlation matrix, already coloured.

    :param labels: short arm labels
    :type labels: [str]
    :param matrix: square matrix of correlations
    :type matrix: [[float]]
    :return: the table HTML
    :rtype: str
    """
    if not labels:
        return '<p class="note">nothing to correlate here.</p>'
    values = [v for row in matrix for v in row if v == v]
    low = min(min(values), 0.95) if values else 0.0
    diverging = any(v < 0 for v in values)

    head = "".join('<th class="collab">%s</th>' % esc(l) for l in labels)
    body = []
    for i, label in enumerate(labels):
        cells = []
        for j in range(len(labels)):
            value = matrix[i][j]
            rgb = heat_rgb(value, low, diverging)
            cells.append(
                '<td style="background:rgb(%d,%d,%d);color:%s" title="%s vs %s">%s</td>'
                % (rgb[0], rgb[1], rgb[2], ink_on(rgb), esc(label), esc(labels[j]),
                   "-" if value != value else "%.3f" % value))
        body.append('<tr><th class="rowlab">%s</th>%s</tr>'
                    % (esc(label), "".join(cells)))
    return ('<div class="scroll"><table class="heat"><thead><tr><th></th>%s</tr>'
            '</thead><tbody>%s</tbody></table></div>'
            % (head, "".join(body)))


def static_ensembles(rows, periods, members, show, cut=()):
    """
    The screened ensembles, already ranked by sharpe.

    :param rows: rows from ensemble_screen.screen
    :type rows: [dict]
    :param periods: shared periods behind them
    :type periods: int
    :param members: how many arms were screened
    :type members: int
    :param show: how many rows to write out
    :type show: int
    :param cut: arms left out of the screen
    :type cut: [str]
    :return: the table HTML plus its footnotes
    :rtype: str
    """
    if not rows:
        return '<p class="note">fewer than two arms -- nothing to screen.</p>'
    columns = [("ensemble", "members", None), ("screened payout", "payout", ".6f"),
               ("sharpe", "sharpe", ".4f"), ("maxdd", "maxdd", ".4f"),
               ("d vs best", "delta", "+.6f"), ("t", "t", "+.2f"),
               ("rho", "rho", ".2f")]
    head = "".join('<th class="%s" aria-sort="%s">%s</th>'
                   % ("txt" if spec is None else "",
                      "descending" if key == "sharpe" else "none", esc(header))
                   for header, key, spec in columns)
    body = []
    for row in rank(rows, "sharpe")[:show]:
        cells = []
        for header, key, spec in columns:
            if key == "members":
                cells.append('<td class="txt">%s</td>'
                             % esc("  +  ".join(row["members"])))
            else:
                value = row[key]
                klass = ""
                if key in ("delta", "t") and value == value:
                    klass = ' class="%s"' % ("pos" if value > 0 else "neg")
                cells.append("<td%s>%s</td>" % (klass, esc(cell_text(value, spec))))
        body.append("<tr>%s</tr>" % "".join(cells))

    beat = len(beat_their_best(rows))
    line = ("%d of %d combinations beat their own best member on payout at |t| > 2."
            % (beat, len(rows)))
    if not beat:
        line += (" None. Equal-weight averaging dilutes the mean by construction, "
                 "so read these as a RISK decision (sharpe, drawdown, correlation), "
                 "not a level one. That null is the finding, not a missing result.")
    return ('<div class="scroll"><table><thead><tr>%s</tr></thead><tbody>%s</tbody>'
            '</table></div><p class="note">%s</p>'
            '<p class="note">showing top %d of %d over %d arms, %d periods.%s</p>'
            % (head, "".join(body), esc(line), min(show, len(rows)), len(rows),
               members, periods,
               esc("  %d lower-paying arms were left out of the screen: %s."
                   % (len(cut), ", ".join(cut))) if cut else ""))


def static_chart(payload, block):
    """
    The cumulative payout chart as inline SVG, on the full window.

    Same geometry as the interactive one so the page does not jump when the
    script replaces it. Series colours are CSS custom properties rather than
    literals, so this static picture still follows the reader's theme.

    :param payload: the block payload
    :type payload: dict
    :param block: the block
    :type block: report_data.Block
    :return: the SVG, plus a static legend
    :rtype: str
    """
    drawn = [a for a in block.arms if payload["chartable"].get(a.label)]
    if not drawn or block.baseline is None:
        return '<p class="note">no baseline in this block, so no chart.</p>'

    series = []
    for arm in drawn:
        payout = block.payout(arm)
        if not len(payout):
            continue
        cumulative, running, peak, under = [], 0.0, float("-inf"), []
        for value in payout:
            running += value
            peak = max(peak, running)
            cumulative.append(running)
            under.append(peak - running)
        series.append({
            "label": arm.label, "kind": arm.kind, "cum": cumulative, "under": under,
            "mean": float(payout.mean()),
            "sharpe": float(payout.mean() / payout.std(ddof=0)) if payout.std(ddof=0) else float("nan"),
            "dd": max(under), "order": payload["order"].index(arm.label),
            "base": arm.label == payload["baseline"],
        })
    if not series:
        return '<p class="note">no series to draw.</p>'

    n = len(series[0]["cum"])
    geometry = CHART
    width = geometry["W"]
    height = (geometry["padT"] + geometry["H1"] + geometry["gap"]
              + geometry["H2"] + geometry["padB"])
    low = min(0.0, min(v for s in series for v in s["cum"]))
    high = max(v for s in series for v in s["cum"])
    worst = max(max(s["under"]) for s in series) or 1.0
    span = (high - low) or 1.0

    def x(i):
        return geometry["padL"] + (0 if n < 2 else
                                   (i / (n - 1.0)) * (width - geometry["padL"] - geometry["padR"]))

    def y1(v):
        return geometry["padT"] + geometry["H1"] - ((v - low) / span) * geometry["H1"]

    def y2(v):
        return (geometry["padT"] + geometry["H1"] + geometry["gap"]
                + (v / worst) * geometry["H2"])

    def path(values, scale):
        return " ".join("%s%d %d" % ("M" if i == 0 else "L",
                                     round(x(i)), round(scale(v)))
                        for i, v in enumerate(values))

    parts = ['<svg class="chart" viewBox="0 0 %d %d" role="img" '
             'aria-label="cumulative backtested payout">' % (width, height)]
    parts.append('<g class="grid">')
    for i in range(6):
        parts.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>'
                     % (geometry["padL"], width - geometry["padR"],
                        y1(low + span * i / 5.0), y1(low + span * i / 5.0)))
    for i in range(3):
        parts.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>'
                     % (geometry["padL"], width - geometry["padR"],
                        y2(worst * i / 2.0), y2(worst * i / 2.0)))
    parts.append("</g><g class=\"axis\">")
    for i in range(6):
        value = low + span * i / 5.0
        parts.append('<text x="%d" y="%.1f" text-anchor="end">%.1f</text>'
                     % (geometry["padL"] - 8, y1(value) + 3, value))
    for i in range(3):
        value = worst * i / 2.0
        parts.append('<text x="%d" y="%.1f" text-anchor="end">%.1f</text>'
                     % (geometry["padL"] - 8, y2(value) + 3, value))
    dates = payload["dates"]
    for i in range(6):
        index = int(round(i / 5.0 * (n - 1)))
        anchor = "start" if i == 0 else ("end" if i == 5 else "middle")
        parts.append('<text x="%.1f" y="%d" text-anchor="%s">%s</text>'
                     % (x(index), height - 6, anchor, esc(dates[index])))
    parts.append('<text x="%d" y="%d" class="panel">CUMULATIVE PAYOUT</text>'
                 % (geometry["padL"] - 50, geometry["padT"] - 12))
    parts.append('<text x="%d" y="%d" class="panel">UNDERWATER</text>'
                 % (geometry["padL"] - 50,
                    geometry["padT"] + geometry["H1"] + geometry["gap"] - 12))
    parts.append("</g>")

    ordered = sorted(series, key=lambda s: (2 if s["base"] else
                                            1 if s["kind"] == "prop" else 0))
    for entry in ordered:
        context = entry["kind"] != "prop" and not entry["base"]
        klass = "baseline" if entry["base"] else ("context" if context else "series")
        stroke = ""
        if not context and not entry["base"]:
            stroke = ' stroke="var(--s%d)"' % (entry["order"] % 8 + 1)
            dash = DASHES[(entry["order"] // 8) % len(DASHES)]
            if dash:
                stroke += ' stroke-dasharray="%s"' % dash
        parts.append('<path class="%s"%s d="%s"/>' % (klass, stroke, path(entry["cum"], y1)))
        parts.append('<path class="%s"%s d="%s"%s/>'
                     % (klass, stroke, path(entry["under"], y2),
                        ' stroke-width="2"' if entry["base"] else ""))
    parts.append("</svg>")

    legend = []
    for entry in sorted(series, key=lambda s: -s["mean"]):
        colour = ("var(--ink)" if entry["base"] else
                  ("var(--s%d)" % (entry["order"] % 8 + 1)) if entry["kind"] == "prop"
                  else "var(--context)")
        legend.append(
            '<span class="lg"><span class="key" style="border-top-color:%s"></span>'
            '<span>%s</span><span class="stat">%s / %s / %s</span></span>'
            % (colour, esc(entry["label"] + (" (baseline)" if entry["base"] else "")),
               cell_text(entry["mean"], ".5f"), cell_text(entry["sharpe"], ".2f"),
               cell_text(entry["dd"], ".2f")))
    return "".join(parts) + '<div class="legend-static">%s</div>' % "".join(legend)


def block_html(payload, block, args, static):
    """
    The static shell of one block; the browser fills every number in.

    :param payload: this block's payload
    :type payload: dict
    :param block: the block
    :type block: report_data.Block
    :param args: parsed arguments, for the --no-* switches
    :type args: argparse.Namespace
    :param static: pre-rendered section HTML by name
    :type static: dict
    :return: the HTML
    :rtype: str
    """
    bid = esc(payload["id"])
    parts = ['<section class="blk"><h2>%s &mdash; %d arms</h2>' % (bid, len(block.arms))]

    rejected = "".join(
        "<li><code>%s</code> &mdash; %s</li>" % (esc(a), esc(b))
        for a, b in payload["rejected"][:8])
    parts.append(
        '<p class="lede">Baseline <b>%s</b>, Newey-West lag %d. '
        'Every figure below is recomputed on the date range you choose. '
        'CORR on this label is not comparable with the other block\'s.</p>'
        % (esc(payload["baseline"] or "none"), payload["lag"]))
    if rejected:
        parts.append('<details class="note"><summary>why this baseline, and what was '
                     'passed over</summary><ul class="plain">%s</ul></details>' % rejected)

    parts.append(
        '<div class="card"><div class="card-head"><h3 class="grow">Results</h3>'
        '<span class="count" id="cnt-%s">%s</span></div>'
        '<div class="card-body"><div class="scroll" id="tbl-%s">%s</div>'
        '<p class="note">Payout is rebuilt per period as <code>0.1*corr + mmc</code> '
        'and summarised once. <code>dd(corr)</code> belongs to the CORR series and '
        '<code>dd(pay)</code> to the payout series &mdash; different series, and the '
        'payout one runs 1.8&ndash;3.4&times; larger. <code>calmar</code>\'s '
        'denominator grows with series length, so it is only comparable across rows '
        'with the same <code>n</code>.</p></div></div>'
        % (bid, esc(static["count"]), bid, static["table"]))

    if payload["predCorr"]:
        parts.append(
            '<div class="card"><div class="card-head"><h3 class="grow">Correlation</h3>'
            '<span class="badge">click a label to drop an arm</span></div>'
            '<div class="card-body"><div class="matrices">'
            '<div class="matrix-block"><h3>Prediction &mdash; are these ranking the '
            'same coins the same way?</h3><div id="pred-%s">%s</div></div>'
            '<div class="matrix-block"><h3>Score &mdash; do they have good and bad '
            'periods together?</h3><div id="score-%s">%s</div></div></div>'
            '<div class="scale"><span>low</span><span class="ramp"></span>'
            '<span>1.000</span></div>'
            '<p class="note">What these numbers mean on this panel:</p>'
            '<ul class="plain">%s</ul>'
            '<p class="note">An arm at 0.9 with a better mean is a worse slot than an '
            'arm at 0.2 with a similar one.</p></div></div>'
            % (bid, static["pred"], bid, static["score"],
               "".join("<li><code>%s</code></li>" % esc(c) for c in CALIBRATION)))

    commands = "".join('<pre class="cmd">%s</pre>' % esc(c)
                       for c in payload["commands"] if c.strip().startswith("python3"))
    if not args.no_ensemble:
        parts.append(
            '<div class="card"><div class="card-head"><h3 class="grow">Ensembles</h3>'
            '<span class="badge badge-warn">screen, not a price</span></div>'
            '<div class="card-body"><div class="warnbox"><b>%s</b> %s</div>'
            '<div id="ens-%s">%s</div>%s</div></div>'
            % (esc(SCREEN_WARNING[0]), esc(" ".join(SCREEN_WARNING[1:])), bid,
               static["ensembles"],
               ('<p class="note">Price the shortlist &mdash; these are real blends, '
                'not screens:</p>' + commands) if commands else ""))

    if not args.no_chart:
        parts.append(
            '<div class="card"><div class="card-head">'
            '<h3 class="grow">Cumulative backtested payout</h3>'
            '<button class="chip" type="button" id="show-context" aria-pressed="true">'
            'other submitted slots</button></div>'
            '<div class="card-body"><div class="chart-wrap" id="chart-%s">%s</div>'
            '<div class="legend" id="legend-%s"></div>'
            '<p class="note">Cumulative payout re-based to the start of the selected '
            'range, with its own underwater curve below &mdash; the drawdown drawn '
            'here belongs to the curve directly above it. Legend reads '
            '<code>mean / p.sharpe / dd(pay)</code> and click hides a series.</p>'
            '</div></div>' % (bid, static["chart"], bid))

    parts.append("</section>")
    return "".join(parts)


def static_sections(args, block, payload, columns):
    """
    Every section of one block, pre-rendered on the full window.

    :param args: parsed arguments
    :type args: argparse.Namespace
    :param block: the block
    :type block: report_data.Block
    :param payload: its payload
    :type payload: dict
    :param columns: (header, key, format) triples
    :type columns: [(str, str, str)]
    :return: HTML by section name
    :rtype: dict
    """
    rows = block.rows()
    if args.sort_by != "none":
        rows = sorted(rows, key=lambda r: (r.get(args.sort_by)
                                           if isinstance(r.get(args.sort_by), (int, float))
                                           and r[args.sort_by] == r[args.sort_by]
                                           else float("-inf")), reverse=True)
    dates = payload["dates"]
    count = ("showing %d of %d arms  |  %d periods, %s to %s"
             % (len(rows), len(block.arms), len(dates),
                dates[0] if dates else "-", dates[-1] if dates else "-"))

    sections = {
        "count": count,
        "table": static_table(payload, rows, columns),
        "pred": "", "score": "", "ensembles": "", "chart": "",
    }

    if payload["predCorr"]:
        labels = [a["short"] for a in payload["arms"]
                  if a["label"] in payload["predCorr"]["labels"]]
        index = {l: i for i, l in enumerate(payload["predCorr"]["labels"])}
        order = [index[a["label"]] for a in payload["arms"]
                 if a["label"] in index]
        matrix = payload["predCorr"]["matrix"]
        sections["pred"] = static_heat(
            labels, [[matrix[i][j] for j in order] for i in order])
        kept = [a for a in block.arms if a.label in index]
        scores, _ = score_series_matrix(
            report_data.Block(block.target, kept, block.window, block.lag))
        scores = scores.loc[[a.label for a in kept], [a.label for a in kept]]
        sections["score"] = static_heat(
            [short_label(a.label) for a in kept],
            [[float(v) for v in row] for row in scores.to_numpy()])

    if not args.no_ensemble:
        picked, cut = report_data.matrix_arms(block, args.ens_max_arms)
        screened, _, periods = screen(
            {a.label: block.payout(a) for a in picked},
            tuple(args.ens_size), block.lag)
        sections["ensembles"] = static_ensembles(
            screened, periods, len(picked), args.ens_top * 4,
            [label for label, _ in cut])

    if not args.no_chart:
        sections["chart"] = static_chart(payload, block)
    return sections


def render(args, blocks, missing, dropped):
    """
    The whole document.

    :param args: parsed arguments
    :type args: argparse.Namespace
    :param blocks: the blocks
    :type blocks: [report_data.Block]
    :param missing: unmatched run patterns
    :type missing: [str]
    :param dropped: [(label, why)]
    :type dropped: [(str, str)]
    :return: the HTML
    :rtype: str
    """
    columns = [("run", "run", None), ("exp", "exp", None),
               ("kind", "kind", None), ("tgt", "tgt", None),
               ("n", "n", "d"), ("n(own)", "n(own)", "d"),
               ("CORR", "CORR", ".5f"), ("MMC", "MMC", ".5f"),
               ("c.sharpe", "c.sharpe", ".2f"), ("dd(corr)", "dd(corr)", ".2f"),
               ("payout", "payout", ".5f"), ("p.sharpe", "p.sharpe", ".2f"),
               ("dd(pay)", "dd(pay)", ".2f"), ("calmar", "calmar", ".3f"),
               ("dpay", "dpay", "+.6f"), ("t", "t", "+.2f"), ("live", "live", None)]

    payloads, statics, excluded = [], [], list(dropped)
    for block in blocks:
        payload, more = block_payload(args, block)
        payloads.append(payload)
        excluded.extend(more)
        statics.append(static_sections(args, block, payload, columns))

    dates = sorted({d for p in payloads for d in p["dates"]})
    round_number, _ = slotlib.latest_round_report()
    generated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")

    report = {
        "dates": dates, "blocks": payloads,
        "columns": [list(c) for c in columns],
        "dashes": DASHES, "ramps": RAMPS, "chart": CHART,
        "ensSizes": list(args.ens_size), "ensShow": args.ens_top * 4,
        "ensCap": 12000, "ensMaxArms": args.ens_max_arms,
    }
    data = json.dumps(report, allow_nan=False).replace("</", "<\\/")

    notes = []
    notes.append("<li>Unmatched run patterns: %s</li>"
                 % (esc(", ".join(missing)) if missing else "none"))
    seen, rows = set(), []
    for label, why in excluded:
        if (label, why) in seen:
            continue
        seen.add((label, why))
        rows.append("<li><code>%s</code> &mdash; %s</li>" % (esc(label), esc(why)))
    notes.append("<li>Excluded arms:<ul class=\"plain\">%s</ul></li>"
                 % ("".join(rows) or "<li>none</li>"))
    mism = []
    for slot in slotlib.slot_rows("all", args.tournament)[0]:
        if not slot.matches_registry:
            mism.append("<li><code>%s</code> &mdash; %s</li>"
                        % (esc(slot.name), esc(slot.note)))
    notes.append("<li>Slots whose recorded figure is not their series:<ul class=\"plain\">"
                 "%s</ul></li>" % ("".join(mism) or "<li>none</li>"))
    faults = ["<li><code>%s</code> &mdash; %s</li>" % (esc(s.name), esc(s.live_status))
              for s in slotlib.slot_rows("all", args.tournament)[0]
              if s.live_status not in ("ok", "-")]
    notes.append("<li>Live state, round %s:<ul class=\"plain\">%s</ul></li>"
                 % (esc(round_number or "none on disk"),
                    "".join(faults) or "<li>all slots submitted their own model</li>"))

    blocks_html = "".join(block_html(p, b, args, s)
                          for p, b, s in zip(payloads, blocks, statics))

    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%(title)s</title>
<style>%(css)s</style>
</head><body><div class="wrap">

<header class="head-row"><div>
<div class="eyebrow">Numerai crypto &mdash; backtest</div>
<h1>%(title)s</h1>
<p class="lede">Proposed runs against the deployed book. Generated %(generated)s UTC;
live state from round %(round)s. Runs: <code>%(runs)s</code>. Submitted book:
<code>%(submitted)s</code>. Baseline policy: <code>%(policy)s</code>.
Backtest artifacts are out-of-sample predictions for historical dates &mdash;
nothing here is a realised live result.</p>
</div><button class="chip" type="button" id="theme">dark mode</button></header>

<noscript><div class="warnbox"><b>Scripts are off in this viewer.</b>
Everything below is the full board on the whole window &mdash; every table, both
correlation matrices, the ensemble screen and the chart are already rendered.
What needs JavaScript is only the interrogation: sorting columns, hiding columns,
filtering rows, narrowing the date range and toggling series. Open the file in a
browser for that.</div></noscript>

<div class="filters">
  <div class="fgroup"><span class="lbl">Range</span>
    <button class="chip" type="button" data-months="0" aria-pressed="true">all</button>
    <button class="chip" type="button" data-months="24">24m</button>
    <button class="chip" type="button" data-months="12">12m</button>
    <button class="chip" type="button" data-months="6">6m</button>
    <button class="chip" type="button" data-months="3">3m</button>
    <input type="range" id="r0" min="0" value="0" step="1" aria-label="range start">
    <input type="range" id="r1" min="1" value="1" step="1" aria-label="range end">
    <span class="range-readout" id="range-readout"></span>
  </div>
  <div class="fgroup"><span class="lbl">Rows</span>
    <input type="search" id="search" placeholder="filter by name&hellip;"
           aria-label="filter rows by name">
    <button class="chip" type="button" data-kind="prop" aria-pressed="true">proposed</button>
    <button class="chip" type="button" data-kind="sub" aria-pressed="true">singles</button>
    <button class="chip" type="button" data-kind="blend" aria-pressed="true">blends</button>
    <button class="chip" type="button" data-kind="stakedOnly" aria-pressed="false">staked only</button>
    <button class="chip" type="button" data-kind="faultsOnly" aria-pressed="false">live faults only</button>
  </div>
  <div class="fgroup"><span class="lbl">Columns</span>
    <details class="cols"><summary>choose</summary>
      <div class="colmenu" id="colmenu"></div>
    </details>
  </div>
  <div class="fgroup"><span class="lbl">Dim rho below</span>
    <input type="range" id="corr-threshold" min="0" max="0.99" step="0.01" value="0"
           aria-label="dim correlations below">
    <span class="range-readout" id="corr-threshold-out">off</span>
  </div>
</div>

%(blocks)s

<section class="card"><div class="card-head"><h3 class="grow">Notes</h3>
<span class="badge">what is missing, and why</span></div>
<div class="card-body"><p class="note">A blank row reads as &ldquo;tested and
null&rdquo; when it means &ldquo;never run&rdquo;, so nothing is dropped
silently.</p><ul class="plain">%(notes)s</ul></div></section>

<p class="foot">Generated by <code>python3 -m src.analysis.report_html</code>.
The archival PDF of the same numbers is <code>python3 -m src.analysis.report_pdf</code>.
Every figure here is computed in the browser from one embedded per-period matrix,
so the table, the score correlations, the ensemble screen and the chart always
describe the same window.</p>

</div>
<script id="report-data" type="application/json">%(data)s</script>
<script>window.REPORT = JSON.parse(document.getElementById("report-data").textContent);</script>
<script>%(compute)s</script>
<script>%(ui)s</script>
</body></html>
""" % {
        "title": esc(args.title), "css": CSS, "generated": esc(generated),
        "round": esc(round_number or "none on disk"),
        "runs": esc(", ".join(args.runs) or "(none)"),
        "submitted": esc(args.submitted), "policy": esc(args.baseline_policy),
        "blocks": blocks_html, "notes": "".join(notes), "data": data,
        "compute": COMPUTE_JS, "ui": UI_JS,
    }


def parse_args(argv=None):
    """Parse the command line."""
    parser = common_parser("python3 -m src.analysis.report_html",
                           "The interactive HTML of a results report.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-html", action="store_true",
                        help="print the table and skip the document")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Build the report.

    :param argv: argument list, or None to read sys.argv
    :type argv: [str] or None
    :return: process exit code
    :rtype: int
    """
    args = parse_args(argv)
    if args.window == "own":
        # The page hangs every section off one shared date axis, which is what
        # lets the range control mean the same thing everywhere. Per-arm windows
        # have no such axis. report_pdf --window own is the tool for that.
        print("note: --window own has no shared date axis to filter on; "
              "building the HTML on the common window instead.")
        args.window = "common"

    blocks, missing, dropped = build(args)
    if not blocks:
        print("nothing to report: no run matched and no slot resolved.")
        if missing:
            print("unmatched: %s" % ", ".join(missing))
        return 1

    sort_by = None if args.sort_by == "none" else args.sort_by
    for block in blocks:
        header, body = render_block(block, sort_by)
        print("\n=== %s -- %d arms ===" % (short_target(block.target), len(block.arms)))
        print("\n".join(header + body))
    if missing:
        print("\nmissing (no artifacts): %s" % ", ".join(missing))
    for label, why in dropped:
        print("  excluded: %s -- %s" % (label, why))

    if args.no_html:
        return 0

    out = args.out or os.path.join(
        "results", "reports", args.tournament, "report_%s.html"
        % datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as handle:
        handle.write(render(args, blocks, missing, dropped))

    megabytes = os.path.getsize(out) / 1e6
    biggest = max(len(b.arms) for b in blocks)
    print("\nwrote %s  (%.1f MB, %d arms in the largest block)"
          % (out, megabytes, biggest))
    if megabytes > 3.0 or biggest > 40:
        # Weight here is arms x periods: every arm carries its per-period corr
        # and mmc so the browser can re-rank on any date range. Say which lever
        # to pull rather than leaving the reader to guess.
        print("  this is a heavy report. --submitted on-account drops the "
              "components, --no-corr skips the prediction matrix, and a "
              "narrower --runs is the biggest lever.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
