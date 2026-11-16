"""E148-only sections for e138_report.render: the audit, Part A (each signal alone, and the decomposition),
and the Part B contrasts that split a gain into residualising, peer information and C2 encoding."""
import json
import os

import pandas as pd

ROOT = "results/reports/crypto/e148_c2resid"
LAB = {"ridge": "RIDGE (causal rolling ridge)", "resmom": "RESMOM (20-day residual momentum)", "raw": "C2 RAW (raw path alone)",
       "res": "C2 RES (residual path alone)", "cov": "C2 COV (residual + BTC, ETH, peer index)", "grp": "C2 GRP (grouped size-bucket context)"}
ORDER = ["cov", "grp", "res", "raw", "ridge", "resmom"]


def assemble():
    """audit.json for Study: the GPU re-forecast audit, the CPU input audit and coverage."""
    gpu = json.load(open("/scratch/danielco/c2resid/audit.json"))
    cpu = json.load(open(os.path.join(ROOT, "audit_cpu.json")))
    o = pd.read_parquet(os.path.join(ROOT, "observed.parquet"))
    cov = {s: float(o["obs_" + s].mean()) for s in ORDER}
    y = o.assign(year=pd.to_datetime(o["date"]).dt.year).groupby("year")[["obs_" + s for s in ("cov", "raw", "ridge")]].mean().reset_index()
    out = dict(gpu=gpu, cpu=cpu, coverage=cov, no_feed=float(o["no_feed"].mean()), coverage_by_year=y.to_dict("records"))
    json.dump(out, open(os.path.join(ROOT, "audit.json"), "w"), indent=1)
    return out


def audit(s, W):
    au = s.AU
    g, c = au["gpu"], au["cpu"]
    dates = sorted({x["date"] for x in g})
    H = ['<h2 id="audit">Point-in-time audit</h2><ul>',
         '<li><b>The forecasts read nothing past D-1.</b> For %d dates (%s) the spot file was cut at D-1, every path rebuilt from it (returns, the 120-day betas, residuals, the peer index), and Chronos-2 rerun on date D for all four arms: '
         '%s of %s forecast summaries were recomputed; the largest absolute difference from the full run is %.2g, and %d rows exist in one run and not the other.</li>'
         % (len(dates), ", ".join(dates), "{:,}".format(sum(x["rows"] for x in g)), "{:,}".format(sum(x["rows"] for x in g)),
            max(x["max_abs_diff"] for x in g), sum(x["missing_in_full"] for x in g)),
         '<li><b>The ridge inputs, residual momentum and peer index</b> rebuilt the same way: %s of %s ridge input cells, %s of %s peer-index cells and %d residual-momentum values differ. '
         'The ridge itself is refitted every 22 panel dates on rows whose 30-day label resolved by R-1 (date &le; R - 31 days), with alpha chosen by leave-one-out on that window alone.</li>'
         % ("{:,}".format(sum(x["ridge_input_differ"] for x in c)), "{:,}".format(sum(x["ridge_input_cells"] for x in c)),
            "{:,}".format(sum(x["peer_differ"] for x in c)), "{:,}".format(sum(x["peer_cells"] for x in c)), sum(x["resmom_differ"] for x in c)),
         '<li><b>Coverage on the scored keys:</b> %s. %.1f%% of keys belong to a coin with no C2-COV signal anywhere in the file (P_NM&#39;s flag).</li>'
         % ("; ".join("%s %.1f%%" % (k.upper(), 100 * v) for k, v in au["coverage"].items()), 100 * au["no_feed"]),
         '<li><b>Survivorship that routing cannot remove:</b> the spot file was fetched in 2026, so which coins have a feed is a today-snapshot (E131/E132). Routing keeps a no-feed coin on v2.0&#39;s rank, but every signal is judged on coins with a feed.</li></ul>']
    y = pd.DataFrame(au["coverage_by_year"])
    H += [W.table(["year", "C2 signals observed", "C2 RAW observed", "RIDGE observed"], [[int(r.year), "%.1f%%" % (100 * r.obs_cov), "%.1f%%" % (100 * r.obs_raw), "%.1f%%" % (100 * r.obs_ridge)] for r in y.itertuples()])]
    return H


def standalone(W):
    L = pd.read_csv(os.path.join(ROOT, "standalone_levels.csv"))
    D = pd.read_csv(os.path.join(ROOT, "standalone_deltas.csv"))
    RC = pd.read_csv(os.path.join(ROOT, "signal_rank_corr.csv"), index_col=0)
    num = W.num
    head = ["signal", "scope", "CORR", "t", "MMC", "c.sharpe", "payout", "p.sharpe", "dd(pay)", "calmar", "burn(d)", "n", "n(pay)"]
    rows = []
    for sc in ("dev", "test"):
        for sg in ORDER:
            r = L[(L.signal == sg) & (L.rows == "common") & (L.scope == sc)].iloc[0]
            rows.append([LAB[sg], sc, num(r["corr"], 4, True), num(r.corr_t, 1, True), num(r.mmc, 4, True), num(r.c_sharpe, 2), num(r.payout, 4, True),
                         num(r.p_sharpe, 2), num(r.dd_pay, 2), num(r.calmar, 3, True), str(int(r.burn)), str(int(r.n)), str(int(r.n_pay))])
    drow = []
    for (a, b, lab), x in D.groupby(["arm", "vs", "label"], sort=False):
        cells = [W.tint(*x[(x.metric == m) & (x.scope == sc)][["delta", "t"]].iloc[0]) for m in ("corr", "payout") for sc in ("dev", "test")]
        drow.append([lab, "%s &minus; %s" % (a.upper(), b.upper())] + cells)
    rc = [[LAB[a].split(" (")[0]] + [num(RC.loc[a, b], 2) for b in ORDER] for a in ORDER]
    return ['<h2 id="partA">Part A: each signal alone</h2>',
            '<p>Each raw signal ranked within date and scored against <code>target_binned_return_20</code>, on the rows where all six exist (identical rows for every signal). '
            'A negative CORR means the forecast ranks coins the wrong way round: a model can flip it, so magnitude is what a feature can carry.</p>',
            W.table(head, rows, "l2"),
            '<h3>The decomposition, standalone</h3><p>Paired per date on the common rows, Newey-West t at lag 20.</p>',
            W.table(["question", "contrast", "ΔCORR dev", "ΔCORR test", "Δpayout dev", "Δpayout test"], drow, "l2"),
            '<h3>How similar the signals are</h3><p>Mean per-date Spearman between the raw signals.</p>',
            W.table([""] + [LAB[b].split(" (")[0] for b in ORDER], rc, "l2")]


def extra(s, W, HDR):
    return standalone(W) + [
        '<h2 id="decomp">Part B: where a marginal gain would come from (each signal added to v2.0, routed)</h2>',
        W.table(HDR, s.rows([("F_RES", "F_RAW"), ("F_COV", "F_RES"), ("F_GRP", "F_RES"), ("F_GRP", "F_COV"),
                             ("F_COV", "F_RIDGE"), ("F_GRP", "F_RIDGE"), ("F_RIDGE", "F_RESMOM")]), "l2"),
        '<p>Rows read: residualising the input (RES vs RAW), BTC / ETH / peer covariates (COV vs RES), grouped peers (GRP vs RES, GRP vs COV), '
        'C2 encoding against a ridge on the same inputs (COV and GRP vs RIDGE), and a fitted linear read against plain residual momentum.</p>']
