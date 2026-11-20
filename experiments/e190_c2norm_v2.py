#!/usr/bin/env python3
"""
E190 -- E189 with the ridge's information set = v2.0 (22 shipped features) + the transformed Chronos-2-Synth
state, for TS_MM, GRP_MM, POOL_MM and RAW (registered in experiments/EXPERIMENTS.md and
src/analysis/experiment_db.py before launch).

    python3 experiments/e190_c2norm_v2.py --fit        # V_N_TS_MM V_N_GRP_MM V_N_POOL_MM V_RAW
    python3 experiments/e190_c2norm_v2.py --score      # routed + covered scores (B_ALL/B_COV/E189 twins linked)
    python3 experiments/e190_c2norm_v2.py --report     # table, tests, HTML (print with src.analysis.html_pdf)

Everything except Features.build is E189's code path (experiments/e189_c2norm.py), imported and pointed at
this study's output directory.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e189_c2norm as M  # noqa: E402

E189_ROOT = M.ROOT
M.STUDY, M.ROOT, M.SCR = "E190", "results/reports/crypto/e190_c2norm_v2", "/scratch/danielco/e190"
INNER = ["N_TS_MM", "N_GRP_MM", "N_POOL_MM", "RAW"]
V_ARMS = ["V_" + a for a in INNER]
REUSED = ["B_ALL", "B_COV"] + INNER                       # E189 preds, same rows / schedule
_E189Features = M.Features


class Features(_E189Features):
    """V_<arm>: [22 v2.0 features | E189 <arm>'s transformed state] on covered rows."""

    def __init__(self, arm, df, emb, feats):
        if arm.startswith("V_"):
            super().__init__(arm[2:], df, emb, feats)
            self.arm = arm
        else:
            super().__init__(arm, df, emb, feats)

    def build(self, fit_rows, apply_rows):
        if not self.arm.startswith("V_"):
            return super().build(fit_rows, apply_rows)
        inner = self.arm[2:]
        self.arm = inner
        try:
            E, info = super().build(fit_rows, apply_rows)
        finally:
            self.arm = "V_" + inner
        V = self.df.loc[apply_rows, self.feats].to_numpy(np.float64)
        assert len(V) == len(E) and np.isfinite(V).all()
        return np.hstack([V, E]), info


M.Features = Features
M.ARMS = V_ARMS + REUSED
M.COVERED_ARMS = V_ARMS + REUSED[1:]


def link_reused():
    d = os.path.join(M.ROOT, "preds")
    os.makedirs(d, exist_ok=True)
    for a in REUSED:
        for ext in (".parquet", ".picks.json"):
            dst = os.path.join(d, a + ext)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(os.path.join(E189_ROOT, "preds", a + ext)), dst)


# ---------------------------------------------------------------------------------- report
def report():
    import e131_analysis as E
    S = pd.read_parquet(os.path.join(M.ROOT, "scores.parquet"))
    pay0 = S[S["mmc"].notna()]["date"].min()
    rows, tests = [], []
    for kind in ("dev", "test"):
        s = S[S["kind"] == kind]
        P = s.pivot(index="date", columns="arm", values="payout").dropna()
        C = s.pivot(index="date", columns="arm", values="corr")
        Mm = s.pivot(index="date", columns="arm", values="mmc").loc[P.index]
        for a in M.ARMS:
            dp = E.describe(P[a]) if hasattr(E, "describe") else {}
            cum = P[a].cumsum()
            rows.append(dict(kind=kind, arm=a, corr=C[a].mean(), mmc=Mm[a].mean(), payout=P[a].mean(),
                             pay_sharpe=P[a].mean() / P[a].std(ddof=0), pay_dd=float((cum.cummax() - cum).max()),
                             n_corr=int(C[a].notna().sum()), n_pay=len(P)))
        for a in V_ARMS:
            for ref in ("B_COV", "V_RAW", a[2:]):
                if ref == a:
                    continue
                for met, X in (("corr", C), ("payout", P)):
                    dlt = (X[a] - X[ref]).dropna()
                    m, t = E.newey_west(dlt.to_numpy())[:2]
                    tests.append(dict(kind=kind, arm=a, vs=ref, metric=met, delta=m, t=t, n=len(dlt),
                                      p=float(2 * (1 - __import__("scipy.stats").stats.norm.cdf(abs(t))))))
    T = pd.DataFrame(rows)
    K = pd.DataFrame(tests)
    for (kind, vs, met), g in K.groupby(["kind", "vs", "metric"]):    # Holm within family
        if vs in ("B_COV", "V_RAW"):
            o = g["p"].sort_values()
            adj = np.maximum.accumulate([min(1, p * (len(o) - i)) for i, p in enumerate(o)])
            K.loc[o.index, "holm"] = adj
    T.to_csv(os.path.join(M.ROOT, "table.csv"), index=False)
    K.to_csv(os.path.join(M.ROOT, "tests.csv"), index=False)
    fmt = lambda x: "%.4f" % x if isinstance(x, float) else x  # noqa: E731
    html = ["<html><head><meta charset='utf-8'><title>E190 v2.0 + Chronos-2 states</title><style>"
            "body{font-family:sans-serif;margin:24px;font-size:12px}table{border-collapse:collapse;margin:8px 0 18px}"
            "td,th{border:1px solid #ccc;padding:3px 7px;text-align:right}th{background:#eee}</style></head><body>",
            "<h1>E190: v2.0 + normalised Chronos-2-Synth states (ridge)</h1>",
            "<p>E189 design; covered-row information set = 22 v2.0 features + 768-d state. Routed R_SLOT vs B_ALL. "
            "CORR over all dates of the window; MMC/payout/sharpe/dd over dates with the meta model (from %s). "
            "Dev = 11 blocks 2020-08..2026-02 (selection); test = 2026-02-20..2026-08-20 (confirmation).</p>" % str(pay0)[:10]]
    for kind in ("dev", "test"):
        t = T[T["kind"] == kind].sort_values("payout", ascending=False).drop(columns="kind")
        html += ["<h2>%s, sorted by payout</h2>" % kind, t.to_html(index=False, float_format=lambda x: "%.4f" % x)]
        k = K[K["kind"] == kind].drop(columns="kind")
        html += ["<h3>%s paired Newey-West (lag 20); Holm within vs B_COV / vs V_RAW families</h3>" % kind,
                 k.to_html(index=False, float_format=lambda x: "%.4f" % x, na_rep="")]
    alphas = {a: [round(np.log10(p["alpha"]), 1) for p in json.load(open(os.path.join(M.ROOT, "preds", a + ".picks.json")))["picks"]]
              for a in M.ARMS}
    html += ["<h2>log10 alpha picked per block</h2>", pd.DataFrame(alphas).T.to_html()]
    rd = os.path.join(M.ROOT, "reading.md")
    if os.path.exists(rd):
        html += ["<h2>Reading</h2><pre style='white-space:pre-wrap'>%s</pre>" % open(rd).read()]
    html.append("</body></html>")
    open(os.path.join(M.ROOT, "E190_report.html"), "w").write("\n".join(html))
    for kind in ("dev", "test"):
        print(kind)
        print(T[T["kind"] == kind].sort_values("payout", ascending=False).to_string(index=False, float_format=lambda x: "%.4f" % x))
    print(K.to_string(index=False, float_format=lambda x: "%.4f" % x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    link_reused()
    if a.score:
        return M.score(a.workers)
    if a.report:
        return report()
    if a.fit:
        df, emb, feats = M.load()
        B = M.schedule(df)
        for arm in (a.arms or V_ARMS):
            assert arm in V_ARMS, arm
            M.fit_arm(arm, df, emb, feats, B)


if __name__ == "__main__":
    main()
