"""Render the LightGBM arm board. --leaf filters the feature-block section."""
import argparse, json, html, os

CAVEAT = {
 "fencglobal": "Lookahead - whole-sample feature mean. Upper bound, never deployable.",
 "allfeatppp": "Contains fenccausal, the worst standalone block.",
 "fenccausal": "Deployable twin of fencglobal; the gap between them is the leak.",
}
LABEL = {"raw":"raw","interact":"interact","spread":"spread","tenc":"tenc","onehot":"onehot",
 "tenc_onehot":"tenc + onehot","onehot_rollsel":"onehot + rollsel",
 "tenc_onehot_rollsel":"tenc + onehot + rollsel","onehot_interact":"onehot + interact",
 "allfeat":"allfeat","allfeatpp":"allfeat++","allfeatppp":"allfeat+++",
 "fenccausal":"fenccausal","fencglobal":"fencglobal"}
def lab(a): return LABEL.get(a, a.replace("grp_","group / ").replace("_"," "))

ap = argparse.ArgumentParser()
ap.add_argument("--arms", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--leaf", default=None, help="keep only this leaf size in the blocks section")
ap.add_argument("--ens", default=None,
                help="ens.json: screened ensembles to recommend")
ap.add_argument("--no-cadence", action="store_true",
                help="drop the cadence cross section and its arms from the board")
ap.add_argument("--corr", default=None,
                help="corr.json to render as a heatmap of the block table members")
ap.add_argument("--leaves", default=None,
                help="comma-separated leaf sizes to keep, rendered as ONE table "
                     "with a leaf column (e.g. 313,50)")
args = ap.parse_args()
arms = json.load(open(args.arms))
blocks = [a for a in arms if a["family"]=="blocks"]
if args.leaf:
    blocks = [a for a in blocks if a["leaf"]==float(args.leaf)]
KEEP = [float(x) for x in args.leaves.split(",")] if args.leaves else None
if KEEP: blocks = [a for a in blocks if a["leaf"] in KEEP]
cadence = [] if args.no_cadence else [a for a in arms if a["family"]=="cadence"]
reference = [a for a in arms if a["family"]=="reference"]
shown = blocks + cadence + reference
pmax = max(a["payout"] for a in shown)
sig = [a for a in shown if a["t"] and abs(a["t"])>2]
tested = [a for a in shown if a["t"] is not None]

def onehot_tag(a):
    """State the one-hot block's status on the arm itself.

    Two independent facts decide it, and the arm NAME reveals neither: whether
    the dataset carries the 740 ticker columns (allfeat, allfeat++ and
    allfeat+++ all do, though nothing in their names says so), and whether the
    leaf floor lets a split use them. A ticker holds at most 130 rows in a
    6-month block, so at min_child_samples 313 LightGBM discards the block
    before training and the columns are present but dead.
    """
    if not a.get("has_onehot"):
        return '<span class="oh oh-none" title="dataset carries no ticker one-hot columns">no 1-hot</span>'
    if a.get("onehot_live"):
        return ('<span class="oh oh-live" title="740 ticker columns present and usable '
                'at this leaf size">1-hot working</span>')
    return ('<span class="oh oh-dead" title="740 ticker columns are present but discarded '
            'before training -- no split can satisfy min_child_samples">1-hot dead</span>')


def chip(a):
    if a["is_base"]: return '<span class="chip chip-base">baseline</span>'
    if a["t"] is None: return '<span class="chip chip-none">&mdash;</span>'
    t=a["t"]
    if abs(t)<=2: return '<span class="chip chip-null">null&nbsp;%+.2f</span>'%t
    return '<span class="chip %s">%s&nbsp;%+.2f</span>'%("chip-pos" if t>0 else "chip-neg","helps" if t>0 else "hurts",t)
def row(a):
    bar=100.0*a["payout"]/pmax
    d="&mdash;" if a["delta"] is None else "%+.6f"%a["delta"]
    f=' <span class="flag" title="%s">&#9650;</span>'%html.escape(CAVEAT[a["arm"]]) if a["arm"] in CAVEAT else ""
    return (f'<tr><th scope="row">{html.escape(lab(a["arm"]))}{f}</th>'
            f'<td class="num dim">{a["feat"]}</td><td class="num">{a["corr"]:.6f}</td>'
            f'<td class="num">{a["mmc"]:.6f}</td><td class="num">{a["sharpe"]:.4f}</td>'
            f'<td class="num dd">{a["maxdd"]:.4f}</td><td class="num strong">{a["payout"]:.6f}</td>'
            f'<td class="bar"><span style="width:{bar:.1f}%"></span></td>'
            f'<td class="num dim">{d}</td><td>{chip(a)}</td></tr>')
HEAD=('<tr><th scope="col">arm</th><th scope="col" class="num">feat</th>'
 '<th scope="col" class="num">CORR</th><th scope="col" class="num">MMC</th>'
 '<th scope="col" class="num">sharpe</th><th scope="col" class="num">max dd</th>'
 '<th scope="col" class="num">payout</th><th scope="col"></th>'
 '<th scope="col" class="num">&Delta; vs base</th><th scope="col">paired t</th></tr>')

def leafchip(leaf):
    """The leaf size is the column that makes a combined table readable.

    At 313 a ticker's 130 rows cannot satisfy the split minimum, so LightGBM
    drops all 740 one-hot columns before training -- an arm marked `inert` is
    byte-identical to raw and is not a measurement of its block.
    """
    live = leaf < 130
    return ('<span class="leafchip %s" title="%s">%d</span>' % (
        "lc-live" if live else "lc-inert",
        "one-hot columns usable" if live else
        "one-hot columns discarded before training -- these arms equal raw",
        int(leaf)))

def combined_table(rows_):
    body = []
    for a in sorted(rows_, key=lambda x: -x["payout"]):
        bar = 100.0 * a["payout"] / pmax
        d = "&mdash;" if a["delta"] is None else "%+.6f" % a["delta"]
        f = (' <span class="flag" title="%s">&#9650;</span>'
             % html.escape(CAVEAT[a["arm"]])) if a["arm"] in CAVEAT else ""
        mark = " " + onehot_tag(a)
        body.append(
            f'<tr><th scope="row">{html.escape(lab(a["arm"]))}{f}{mark}</th>'
            f'<td>{leafchip(a["leaf"])}</td>'
            f'<td class="num dim">{a["feat"]}</td><td class="num">{a["corr"]:.6f}</td>'
            f'<td class="num">{a["mmc"]:.6f}</td><td class="num">{a["sharpe"]:.4f}</td>'
            f'<td class="num dd">{a["maxdd"]:.4f}</td>'
            f'<td class="num strong">{a["payout"]:.6f}</td>'
            f'<td class="bar"><span style="width:{bar:.1f}%"></span></td>'
            f'<td class="num dim">{d}</td><td>{chip(a)}</td></tr>')
    head = ('<tr><th scope="col">arm</th><th scope="col">leaf</th>'
            '<th scope="col" class="num">feat</th><th scope="col" class="num">CORR</th>'
            '<th scope="col" class="num">MMC</th><th scope="col" class="num">sharpe</th>'
            '<th scope="col" class="num">max dd</th><th scope="col" class="num">payout</th>'
            '<th scope="col"></th><th scope="col" class="num">&Delta; vs base</th>'
            '<th scope="col">paired t</th></tr>')
    return ('<div class="card"><div class="scroll"><table><thead>%s</thead><tbody>%s'
            '</tbody></table></div></div>' % (head, "".join(body)))

cards=[]
for leaf in sorted({a["leaf"] for a in blocks}, reverse=True):
    grp=sorted([a for a in blocks if a["leaf"]==leaf], key=lambda x:-x["payout"])
    live = leaf < 130
    badge=('<span class="badge badge-live">one-hot live</span>' if live
           else '<span class="badge badge-inert">one-hot inert</span>')
    note="" if live else ('<p class="note">A ticker holds at most 130 rows in a 6-month block, so no '
      'split can leave 313 rows on both sides. LightGBM discards all 740 one-hot columns at dataset '
      'construction &mdash; <code>onehot</code> here is byte-identical to <code>raw</code>.</p>')
    cards.append(f'<section class="card"><header class="card-head">'
      f'<h3>min_child_samples <span class="leafnum">{int(leaf)}</span></h3>{badge}</header>{note}'
      f'<div class="scroll"><table><thead>{HEAD}</thead><tbody>{"".join(row(a) for a in grp)}'
      '</tbody></table></div></section>')

matrix=""
if cadence:
  wins=["3m","6m","9m","12m"]; cells={}
  for a in cadence:
      w,r=a["arm"].replace("w","").split("_r"); cells[(w,r)]=a
  cmin=min(a["payout"] for a in cadence); cmax=max(a["payout"] for a in cadence)
  mrows=[]
  for w in wins:
    tds=[]
    for r in wins:
        a=cells.get((w,r))
        if not a: tds.append('<td class="mx empty"></td>'); continue
        f=(a["payout"]-cmin)/(cmax-cmin)
        cls=("mx"+(" mx-base" if a["is_base"] else "")+(" mx-bad" if (a["t"] and a["t"]<-2) else ""))
        t = "baseline" if a["is_base"] else ("t %+.2f"%a["t"] if a["t"] else "")
        tds.append(f'<td class="{cls}" style="--f:{f:.3f}"><span class="mx-pay">{a["payout"]:.4f}</span>'
                   f'<span class="mx-t">{t}</span></td>')
    mrows.append(f'<tr><th scope="row">{w}</th>{"".join(tds)}</tr>')
  matrix=('<div class="scroll"><table class="matrix"><thead><tr>'
   '<th scope="col" class="corner">window &#9586; refit</th>'
   + "".join('<th scope="col">%s</th>'%r for r in wins) + '</tr></thead><tbody>'
   + "".join(mrows) + '</tbody></table></div>')

sig_items="".join('<li><b>%s</b> at leaf %s <span class="mono">t %+.2f</span>%s</li>'%(
    html.escape(lab(a["arm"])), int(a["leaf"]) if a["leaf"] else "search", a["t"],
    ' <em>%s</em>'%html.escape(CAVEAT[a["arm"]]) if a["arm"] in CAVEAT else "")
    for a in sorted(sig,key=lambda y:-abs(y["t"]))) or "<li>No arm cleared |t| &gt; 2.</li>"
best=max((a for a in shown if a["arm"] not in CAVEAT), key=lambda a:a["payout"])
scope = ("feature blocks at min_child_samples&nbsp;%s" % int(float(args.leaf))) if args.leaf \
        else "every feature block, leaf size by leaf size"

SHORT = {"raw":"raw","tenc":"tenc","spread":"spread","interact":"inter",
 "onehot":"1hot","tenc_onehot":"t+1h","onehot_rollsel":"1h+roll",
 "tenc_onehot_rollsel":"t+1h+roll","onehot_interact":"1h+inter","allfeat":"allfeat",
 "allfeatpp":"allfeat++","allfeatppp":"allfeat+++","fenccausal":"fencC","fencglobal":"fencG"}

def heatmap(path):
    """Within-period rank correlation between the block table's members.

    The same statistic corr_heatmap.py uses: rank predictions inside each date,
    correlate, average over the 510 dates. It answers "do these two arms order
    the coins the same way", which is what decides whether a second arm adds
    anything to a book -- a question payout cannot answer.
    """
    d = json.load(open(path))
    labels, M = d["labels"], d["matrix"]
    n = len(labels)
    lo = min(M[i][j] for i in range(n) for j in range(n) if i != j)
    def cell(v, i, j):
        if i == j:
            return '<td class="hm hm-diag" title="%s"></td>' % html.escape(labels[i])
        f = (v - lo) / (1.0 - lo)
        ident = " hm-one" if v > 0.9999 else ""
        return ('<td class="hm%s" style="--f:%.3f" title="%s vs %s = %.3f">%s</td>'
                % (ident, f, html.escape(labels[i]), html.escape(labels[j]), v,
                   "%.2f" % v if n <= 14 else ""))
    def short(l):
        a, leaf = l.rsplit("@", 1)
        return '%s<span class="hm-leaf">%s</span>' % (SHORT.get(a, a), leaf)
    head = "".join('<th scope="col"><span>%s</span></th>' % short(l) for l in labels)
    body = "".join('<tr><th scope="row">%s</th>%s</tr>'
                   % (short(labels[i]), "".join(cell(M[i][j], i, j) for j in range(n)))
                   for i in range(n))
    return ('<div class="scroll"><table class="heat"><thead><tr><th class="corner"></th>%s</tr>'
            '</thead><tbody>%s</tbody></table></div>'
            '<div class="hm-key"><span>less alike</span>'
            '<span class="hm-ramp" style="--lo:%.3f"></span><span>identical</span>'
            '<span class="hm-note">rho %.2f &ndash; 1.00 &middot; outlined cells are '
            'byte-identical arms</span></div>' % (head, body, lo, lo))

def ens_table(path):
    """Screened 2- and 3-way equal-weight ensembles, ranked by sharpe.

    Screened on the PAYOUT SERIES, which is what run_ensemble's rank-average
    approximates and what a stake actually earns. The delta column is against
    the ensemble's own best member, not against a global baseline: the question
    an ensemble has to answer is "does combining beat just holding the best of
    these", and averaging with a weaker member mechanically dilutes the mean.
    """
    d = json.load(open(path))
    combos, solo = d["combos"], d["solo"]
    def short(k): return k.replace("lgb:", "").replace("cur:CURUPIRA_", "C")
    mixed = [c for c in combos if any(m.startswith("lgb:") for m in c["members"])
             and any(m.startswith("cur:") for m in c["members"])]
    lgbo  = [c for c in combos if all(m.startswith("lgb:") for m in c["members"])]
    best_solo = max(solo.items(), key=lambda kv: kv[1]["sharpe"])
    rows = []
    for title, group in (("LightGBM + CURUPIRA", mixed), ("LightGBM only", lgbo)):
        rows.append('<tr class="ens-head"><th colspan="7">%s</th></tr>' % title)
        for c in sorted(group, key=lambda x: -x["sharpe"])[:5]:
            rows.append(
                '<tr><th scope="row">%s</th><td class="num">%.6f</td>'
                '<td class="num strong">%.4f</td><td class="num dd">%.4f</td>'
                '<td class="num dim">%+.6f</td><td class="num dim">%+.2f</td>'
                '<td class="num dim">%.2f</td></tr>'
                % (" + ".join(short(m) for m in c["members"]), c["payout"],
                   c["sharpe"], c["maxdd"], c["d_vs_best"], c["t_vs_best"],
                   c["mean_rho"]))
    rows.append('<tr class="ens-head"><th colspan="7">best single arm, for reference</th></tr>')
    b = best_solo[1]
    rows.append('<tr><th scope="row">%s</th><td class="num">%.6f</td>'
                '<td class="num strong">%.4f</td><td class="num dd">%.4f</td>'
                '<td class="num dim">&mdash;</td><td class="num dim">&mdash;</td>'
                '<td class="num dim">&mdash;</td></tr>'
                % (short(best_solo[0]), b["payout"], b["sharpe"], b["maxdd"]))
    beat = sum(1 for c in mixed if c["t_vs_best"] > 2)
    return ('<div class="card"><div class="scroll"><table><thead><tr>'
            '<th scope="col">ensemble</th><th scope="col" class="num">payout</th>'
            '<th scope="col" class="num">sharpe</th><th scope="col" class="num">max dd</th>'
            '<th scope="col" class="num">&Delta; vs best member</th>'
            '<th scope="col" class="num">t</th><th scope="col" class="num">rho</th>'
            '</tr></thead><tbody>%s</tbody></table></div></div>'
            '<p class="lede" style="margin-top:14px"><b>%d of %d</b> LightGBM+CURUPIRA '
            'combinations beat their own best member on payout at |t| &gt; 2. Every delta is '
            'negative &mdash; equal-weight averaging dilutes the mean by construction. The case '
            'for these blends is entirely risk: the top pairing cuts max drawdown from '
            '%.2f to %.2f and lifts sharpe from %.2f to %.2f, on members correlating rho 0.03&ndash;0.08.</p>'
            % ("".join(rows), beat, len(mixed), b["maxdd"],
               min(c["maxdd"] for c in mixed), b["sharpe"],
               max(c["sharpe"] for c in mixed)))

ens_section = ("""<section>
    <h2>Ensemble candidates</h2>
    <p class="lede">Equal-weight combinations of the block arms above with the deployed CURUPIRA
    slots, screened over the 470 periods all members share. Ranked by sharpe, because payout
    cannot separate them.</p>
    <div style="margin-top:18px">%s</div>
  </section>""" % ens_table(args.ens)) if args.ens else ""

corr_section = ("""<section>
    <h2>Correlation between block arms</h2>
    <p class="lede">Within-period rank correlation across the %d members of the table above &mdash;
    predictions ranked inside each date, then averaged over all 510 dates. Two arms at 1.00 are
    the same model: at <span class="mono">min_child_samples 313</span> the one-hot block is
    discarded, so <span class="mono">onehot</span> collapses onto <span class="mono">raw</span>
    and <span class="mono">tenc+onehot</span> onto <span class="mono">tenc</span>. For a
    multi-slot book this matrix matters more than payout &mdash; an arm that scores well but
    ranks like one you already hold adds nothing.</p>
    <div class="card" style="margin-top:18px;padding:18px">%s</div>
  </section>""" % (len(json.load(open(args.corr))["labels"]), heatmap(args.corr))) if args.corr else ""

blocks_lede = ("One table across leaf sizes " + " and ".join(
    "<span class=\"mono\">%d</span>" % int(x) for x in sorted(KEEP, reverse=True)) +
    ". The leaf column is load-bearing: <span class=\"mono\">min_child_samples</span> decides "
    "whether the 740 ticker one-hot columns can be split on, and the arm name reveals "
    "neither fact &mdash; <span class=\"mono\">allfeat</span> and both "
    "<span class=\"mono\">allfeat+</span> variants carry all 740 without saying so. Each arm "
    "therefore states its own one-hot status. Every &Delta; is "
    "against <span class=\"mono\">raw</span> at that arm's own leaf size &mdash; comparing "
    "across leaf sizes would confound the columns with the regularisation."
) if KEEP else (
    "Grouped by <span class=\"mono\">min_child_samples</span>, because it decides whether the "
    "740 ticker one-hot columns exist at all. Each arm is compared to "
    "<span class=\"mono\">raw</span> at its own leaf size &mdash; crossing leaf sizes would "
    "confound the columns with the regularisation.")

cadence_section = "" if not cadence else """  <section>
    <h2>Cadence cross</h2>
    <p class="lede">Train window against refit cadence, model pinned, raw 22 features. Shading is
    payout; the outlined cell is the 6m/6m baseline every other cell is tested against.</p>
    <div class="card" style="margin-top:18px;padding:18px">%s</div>
  </section>""" % matrix

CSS = open(os.path.join(os.path.dirname(args.out), "board.css")).read()
open(args.out,"w").write(f"""<title>LightGBM Arm Board</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
  <header>
    <div class="eyebrow">Numerai Crypto &middot; walk-forward 510 periods</div>
    <h1>LightGBM Arm Board</h1>
    <p class="lede">LightGBM regression arms on this panel, covering {scope}. Deltas are paired
    against the matched control and corrected with Newey-West at lag&nbsp;20 &mdash; 510 daily
    periods at ~95% target overlap are roughly <b>26 effective independent windows</b>, so
    <span class="mono">|t| &lt; 2</span> is the expected outcome and a null is the finding.</p>
  </header>
  <div class="tiles">
    <div class="tile"><span class="k">arms</span><span class="v">{len(shown)}</span>
      <span class="s">{len(blocks)} feature blocks &middot; {len(reference)} reference</span></div>
    <div class="tile"><span class="k">cleared |t| &gt; 2</span><span class="v">{len(sig)}</span>
      <span class="s">of {len(tested)} arms with a matched control</span></div>
    <div class="tile"><span class="k">best deployable payout</span><span class="v">{best["payout"]:.4f}</span>
      <span class="s">{html.escape(lab(best["arm"]))} &middot; {best["feat"]} features</span></div>
    <div class="tile"><span class="k">max drawdown at leaf 50</span><span class="v">0.18&ndash;0.35</span>
      <span class="s">vs 0.42&ndash;1.17 for the same datasets at leaf 313</span></div>
  </div>
  <section>
    <h2>Feature blocks</h2>
    <p class="lede">{blocks_lede}</p>
    <div style="display:flex;flex-direction:column;gap:18px;margin-top:18px">{combined_table(blocks) if KEEP else "".join(cards)}</div>
  </section>
  {corr_section}
  {ens_section}
{cadence_section}
  <section>
    <h2>Reference arms</h2>
    <p class="lede">Runs that searched 50 draws per refit rather than using the pinned
    configuration, including the single-group arms. Not directly comparable to the pinned board
    above &mdash; the search is a second difference.</p>
    <div class="card" style="margin-top:18px"><div class="scroll"><table><thead>{HEAD}</thead>
    <tbody>{"".join(row(a) for a in sorted(reference,key=lambda x:-x["payout"]))}</tbody></table></div></div>
  </section>
  <section class="findings"><h2>What cleared significance</h2><ul>{sig_items}</ul></section>
  <div class="foot"><hr class="rule">
    <div><b>&#9650;</b> fencglobal is lookahead &mdash; a whole-sample feature mean encodes values
    you would not have live, and 62% of tokens do not survive three years, so the profile partly
    states whether the token survived. It is an upper bound, not a candidate.</div>
    <div>payout = 0.1 &times; CORR + MMC &middot; pinned config: 4327 trees, lr 0.002572, depth 8,
    255 leaves, colsample 0.0702, subsample 0.8162, reg_lambda 1.7575 &middot; seed 42</div>
  </div>
</div>""")
print("wrote %s: %d arms (%d blocks)" % (args.out, len(shown), len(blocks)))
