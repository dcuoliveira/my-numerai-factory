"""
Hyperparameter importance and sensitivity from a random search's per-candidate CV scores.

Input: one or more `results/tuning/<tournament>/<model>/<run>_cv.csv` written by
`run_forecast --save-cv-results` (one row per candidate per walk-forward refit).

Why these statistics
--------------------
A random search over a discrete grid draws every axis UNIFORMLY and
INDEPENDENTLY. Under that design the between-level variance of the score along
one axis, as a share of the total (eta^2), is an unbiased estimate of that axis'
MAIN EFFECT. That is exactly what fANOVA's first-order term estimates, but here
it comes without a surrogate model. Two things are layered on top:

  * a PERMUTATION NULL per axis (shuffle the score, recompute eta^2), because
    eta^2 of a many-level axis is positive even when it does nothing. Importance
    is reported as eta^2 minus the null mean, with a p-value;
  * STABILITY ACROSS REFITS. Each refit is a different training window, the
    walk-forward's own regimes. A setting that matters in one window and
    reverses in the next cannot be fixed ahead of time. Per axis: eta^2 in each
    refit, and Kendall's W on the level ranking across refits (1 = same order
    every window).

Scores are centred and scaled WITHIN each refit (z-score) before pooling, so a
refit whose whole level is high does not dominate the pooled ranking. A random
forest surrogate on the raw parameters is a cross-check: its permutation
importance captures interactions that eta^2 misses. Pairwise interaction
strength is eta^2(a x b) - eta^2(a) - eta^2(b) on the top axes.

The SUGGESTED GRID keeps, per axis, every level whose pooled mean z-score is
within one standard error of the best level. An axis with no detectable effect
(p >= 0.05 on both metrics) is marked "pin": it can be held at any kept level
to spend the walk-forward search budget on the axes that matter.

Usage:
    python3 -m src.analysis.hparam_importance \
        --cv results/tuning/crypto/tabnet_ktz/e154_raw_s42_cv.csv [...] \
        --label raw --out results/reports/crypto/e154_tabnet_ktz_hparams
Several --cv files with the same --label are pooled (e.g. two seeds); call the
tool with repeated `--group label=file1,file2` to report several arms in one
document.
"""

import argparse
import base64
import html
import io
import os

import numpy as np
import pandas as pd

METRICS = ("corr", "mmc")
N_PERM = 300


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load(paths):
    frames = []
    for i, p in enumerate(paths):
        d = pd.read_csv(p)
        d["source"] = os.path.basename(p)
        d["block"] = d["source"] + ":" + d["refit"].astype(str)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    params = [c for c in d.columns if c.startswith("param_")]
    # Drop axes that never vary and exact duplicates (n_a tied to n_d).
    keep, seen = [], {}
    for c in params:
        if d[c].nunique(dropna=False) < 2:
            continue
        sig = tuple(d[c].astype(str))
        if sig in seen:
            continue
        seen[sig] = c
        keep.append(c)
    return d, keep


def pretty(param):
    return param.replace("param_", "").replace("net__", "")


def is_numeric(series):
    return pd.to_numeric(series, errors="coerce").notna().all()


def level_key(series):
    """Sortable level labels: numeric axes sort by value, others alphabetically."""
    if is_numeric(series):
        return pd.to_numeric(series)
    return series.astype(str)


def zscore_within(d, col):
    g = d.groupby("block")[col]
    return (d[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def eta2(score, groups):
    ok = np.isfinite(score)
    s, g = score[ok], groups[ok]
    if s.size < 3:
        return np.nan
    total = ((s - s.mean()) ** 2).sum()
    if total <= 0:
        return np.nan
    df = pd.DataFrame({"s": s, "g": g})
    m = df.groupby("g")["s"].agg(["mean", "size"])
    between = (m["size"] * (m["mean"] - s.mean()) ** 2).sum()
    return float(between / total)


def eta2_with_null(score, groups, rng, n_perm=N_PERM):
    obs = eta2(score, groups)
    ok = np.isfinite(score)
    s, g = score[ok], groups[ok]
    null = np.array([eta2(rng.permutation(s), g) for _ in range(n_perm)])
    p = (1 + np.sum(null >= obs)) / (1 + n_perm)
    return obs, float(np.nanmean(null)), float(p)


def kendall_w(rank_matrix):
    """rank_matrix: refits x levels (ranks within each refit). NaN rows dropped."""
    r = np.asarray(rank_matrix, dtype=float)
    r = r[np.isfinite(r).all(axis=1)]
    m, n = r.shape if r.ndim == 2 else (0, 0)
    if m < 2 or n < 2:
        return np.nan
    R = r.sum(axis=0)
    S = ((R - R.mean()) ** 2).sum()
    return float(12 * S / (m ** 2 * (n ** 3 - n)))


def axis_table(d, params, metric, rng):
    zcol = "z_" + metric
    rows = []
    for p in params:
        groups = d[p].astype(str).to_numpy()
        obs, null, pval = eta2_with_null(d[zcol].to_numpy(), groups, rng)
        per_refit = []
        ranks = []
        levels = sorted(d[p].astype(str).unique(), key=str)
        for _, b in d.groupby("block"):
            e = eta2(b[zcol].to_numpy(), b[p].astype(str).to_numpy())
            per_refit.append(e)
            means = b.groupby(b[p].astype(str))[zcol].mean().reindex(levels)
            ranks.append(means.rank().to_numpy())
        rows.append({
            "axis": pretty(p), "param": p, "levels": len(levels),
            "eta2": obs, "null": null, "excess": obs - null, "p": pval,
            "eta2_refit_min": np.nanmin(per_refit) if per_refit else np.nan,
            "eta2_refit_max": np.nanmax(per_refit) if per_refit else np.nan,
            "kendall_w": kendall_w(ranks),
        })
    return pd.DataFrame(rows).sort_values("excess", ascending=False)


def level_table(d, p, metric):
    zcol = "z_" + metric
    g = d.groupby(d[p].astype(str))
    t = pd.DataFrame({
        "n": g.size(),
        "mean_z": g[zcol].mean(),
        "se_z": g[zcol].std() / np.sqrt(g.size()),
        "mean_raw": g["val_%s_mean" % metric].mean(),
        # Share of this level among the top-10% candidates (within refit), against
        # its share of all candidates: enrichment > 1 means over-represented.
        "top10_enrich": g["top10_" + metric].mean() / max(d["top10_" + metric].mean(), 1e-12),
    })
    order = level_key(pd.Series(t.index)).argsort().to_numpy()
    return t.iloc[order]


def interactions(d, top, metric, rng):
    zcol = "z_" + metric
    out = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            pair = (d[a].astype(str) + " | " + d[b].astype(str)).to_numpy()
            ea = eta2(d[zcol].to_numpy(), d[a].astype(str).to_numpy())
            eb = eta2(d[zcol].to_numpy(), d[b].astype(str).to_numpy())
            eab, null, pval = eta2_with_null(d[zcol].to_numpy(), pair, rng, n_perm=200)
            # What the pair explains beyond its two main effects. Read it against
            # pair_null: a many-cell pair picks up chance variance too.
            out.append({"pair": "%s x %s" % (pretty(a), pretty(b)),
                        "interaction": eab - ea - eb,
                        "pair_eta2": eab, "pair_null": null, "p_pair": pval})
    return pd.DataFrame(out).sort_values("interaction", ascending=False)


def surrogate_importance(d, params, metric, seed=0):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import KFold, cross_val_score
    zcol = "z_" + metric
    ok = d[zcol].notna()
    X = pd.DataFrame(index=d.index[ok])
    for p in params:
        s = d.loc[ok, p]
        if is_numeric(s):
            v = pd.to_numeric(s)
            X[pretty(p)] = np.log10(v) if (v > 0).all() and v.max() / v.min() >= 50 else v
        else:
            X[pretty(p)] = s.astype("category").cat.codes
    y = d.loc[ok, zcol].to_numpy()
    if len(y) < 30:
        return None, np.nan
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=5,
                               random_state=seed, n_jobs=4)
    r2 = cross_val_score(rf, X, y, cv=KFold(5, shuffle=True, random_state=seed),
                         scoring="r2").mean()
    rf.fit(X, y)
    pi = permutation_importance(rf, X, y, n_repeats=10, random_state=seed, n_jobs=4)
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
    return imp, float(r2)


def suggest_grid(levels_by_metric, axes_by_metric, params):
    rows = []
    for p in params:
        keep_sets, sig = [], False
        for m in METRICS:
            t = levels_by_metric[m].get(p)
            if t is None or t["mean_z"].isna().all():
                continue
            a = axes_by_metric[m].set_index("param").loc[p]
            sig = sig or (a["p"] < 0.05)
            best = t["mean_z"].idxmax()
            thr = t.loc[best, "mean_z"] - t.loc[best, "se_z"]
            keep_sets.append(set(t.index[t["mean_z"] >= thr]))
        if not keep_sets:
            continue
        keep = set.intersection(*keep_sets) or set.union(*keep_sets)
        order = sorted(keep, key=lambda v: (float(v) if _isfloat(v) else 0, v))
        best_corr = levels_by_metric["corr"][p]["mean_z"].idxmax()
        rows.append({"axis": pretty(p), "action": "search" if sig else "pin",
                     "keep levels": ", ".join(_fmt(v) for v in order),
                     "best (corr)": _fmt(best_corr),
                     "best (mmc)": _fmt(levels_by_metric["mmc"][p]["mean_z"].idxmax())
                     if p in levels_by_metric["mmc"] and levels_by_metric["mmc"][p]["mean_z"].notna().any() else "-"})
    return pd.DataFrame(rows)


def _isfloat(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


def _fmt(v):
    if _isfloat(v):
        f = float(v)
        return ("%.3g" % f) if (abs(f) < 1e-2 or abs(f) >= 1e4) and f != 0 else ("%g" % round(f, 4))
    return str(v)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def _fig_b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def marginal_figure(d, params, metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zcol = "z_" + metric
    n = len(params)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False)
    for ax, p in zip(axes.ravel(), params):
        levels = sorted(d[p].astype(str).unique(), key=lambda v: (float(v) if _isfloat(v) else 0, v))
        x = np.arange(len(levels))
        for _, b in d.groupby("block"):
            m = b.groupby(b[p].astype(str))[zcol].mean().reindex(levels)
            ax.plot(x, m.to_numpy(), color="0.75", lw=0.8)
        g = d.groupby(d[p].astype(str))[zcol]
        m = g.mean().reindex(levels)
        se = (g.std() / np.sqrt(g.size())).reindex(levels)
        ax.errorbar(x, m, yerr=se, color="C0", lw=2, marker="o", ms=3, capsize=2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(pretty(p), fontsize=9)
        step = max(1, len(levels) // 6)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([_fmt(v) for v in levels][::step], fontsize=7, rotation=30)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("%s: within-refit z-score by level (blue: pooled mean ± se; grey: each refit)"
                 % metric.upper(), fontsize=10)
    fig.tight_layout()
    return _fig_b64(fig)


def importance_figure(axes_by_metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = axes_by_metric["corr"].set_index("axis")["excess"]
    b = axes_by_metric["mmc"].set_index("axis")["excess"].reindex(a.index)
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(a) + 1))
    y = np.arange(len(a))
    ax.barh(y - 0.2, a.to_numpy(), height=0.4, label="CORR", color="C0")
    ax.barh(y + 0.2, b.to_numpy(), height=0.4, label="MMC", color="C1")
    ax.set_yticks(y)
    ax.set_yticklabels(a.index, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("eta² above permutation null (share of within-refit score variance)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_b64(fig)


def corr_mmc_figure(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(d["z_corr"], d["z_mmc"], s=6, alpha=0.4)
    ok = d[["z_corr", "z_mmc"]].dropna()
    rho = ok.corr(method="spearman").iloc[0, 1] if len(ok) > 2 else np.nan
    ax.set_xlabel("CV CORR (z within refit)")
    ax.set_ylabel("CV MMC (z within refit)")
    ax.set_title("candidates: Spearman %.2f" % rho, fontsize=9)
    fig.tight_layout()
    return _fig_b64(fig), rho


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def table_html(df, floats=4):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else ("%.*f" % (floats, v))
        return html.escape(str(v))
    head = "".join("<th>%s</th>" % html.escape(str(c)) for c in df.columns)
    body = "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % f(v) for v in r)
                   for r in df.itertuples(index=False))
    return "<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>" % (head, body)


def analyse(label, paths, seed=0):
    rng = np.random.default_rng(seed)
    d, params = load(paths)
    for m in METRICS:
        col = "val_%s_mean" % m
        if col not in d.columns:
            d[col] = np.nan
        d["z_" + m] = zscore_within(d, col)
        q = d.groupby("block")["z_" + m].transform(lambda s: s.quantile(0.9))
        d["top10_" + m] = (d["z_" + m] >= q).astype(float)
    axes = {m: axis_table(d, params, m, rng) for m in METRICS}
    levels = {m: {p: level_table(d, p, m) for p in params} for m in METRICS}
    top = list(axes["corr"]["param"].head(4))
    inter = {m: interactions(d, top, m, rng) for m in METRICS}
    surr = {m: surrogate_importance(d, params, m, seed) for m in METRICS}
    grid = suggest_grid(levels, axes, params)
    return dict(label=label, paths=paths, d=d, params=params, axes=axes,
                levels=levels, inter=inter, surr=surr, grid=grid)


def render(results, title, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    css = """body{font-family:Helvetica,Arial,sans-serif;margin:24px;color:#222;font-size:12px}
    h1{font-size:20px}h2{font-size:16px;border-bottom:1px solid #ccc;margin-top:28px}
    h3{font-size:13px}table{border-collapse:collapse;margin:6px 0 12px}
    td,th{border:1px solid #ccc;padding:2px 6px;text-align:right}th{background:#f0f0f0}
    td:first-child,th:first-child{text-align:left}img{max-width:100%}
    .note{color:#555;font-size:11px;max-width:900px}"""
    parts = ["<html><head><meta charset='utf-8'><title>%s</title><style>%s</style></head><body>"
             % (html.escape(title), css), "<h1>%s</h1>" % html.escape(title),
             "<p class='note'>Per-candidate inner-CV scores from a random search, "
             "one row per candidate per walk-forward refit. Scores are z-scored "
             "within each refit before pooling. <b>Importance</b> = eta² of the "
             "axis (between-level share of score variance) minus its permutation-null "
             "mean; p from %d permutations. <b>Kendall W</b>: agreement of the level "
             "ranking across refits (1 = identical order in every training window). "
             "These are CV (in-sample-window) scores, not walk-forward hold-out "
             "scores, which exist only for the winners.</p>" % N_PERM]
    for r in results:
        d = r["d"]
        parts.append("<h2>%s</h2>" % html.escape(r["label"]))
        parts.append("<p class='note'>%d candidates over %d refit-blocks from %s.</p>"
                     % (len(d), d["block"].nunique(),
                        html.escape(", ".join(os.path.basename(p) for p in r["paths"]))))
        img, rho = corr_mmc_figure(d)
        parts.append("<h3>Importance (main effects)</h3><img src='data:image/png;base64,%s'>"
                     % importance_figure(r["axes"]))
        for m in METRICS:
            t = r["axes"][m][["axis", "levels", "eta2", "null", "excess", "p",
                              "eta2_refit_min", "eta2_refit_max", "kendall_w"]]
            parts.append("<h3>%s: axis table</h3>%s" % (m.upper(), table_html(t)))
            imp, r2 = r["surr"][m]
            if imp is not None:
                parts.append("<p class='note'>RF surrogate (5-fold CV R² %.3f; low R² means "
                             "the space is mostly flat/noisy) permutation importance:</p>%s"
                             % (r2, table_html(imp.rename("importance").reset_index()
                                               .rename(columns={"index": "axis"}))))
            parts.append("<h3>%s: top interactions among the 4 strongest CORR axes</h3>%s"
                         % (m.upper(), table_html(r["inter"][m])))
        parts.append("<h3>CORR vs MMC across candidates</h3><img src='data:image/png;base64,%s'>"
                     % img)
        for m in METRICS:
            parts.append("<h3>%s: sensitivity curves</h3><img src='data:image/png;base64,%s'>"
                         % (m.upper(), marginal_figure(d, r["params"], m)))
        parts.append("<h3>Level tables</h3>")
        for p in r["axes"]["corr"]["param"]:
            for m in METRICS:
                t = r["levels"][m][p].reset_index().rename(columns={p: "level", "index": "level"})
                t.columns = ["level"] + list(t.columns[1:])
                t["level"] = t["level"].map(_fmt)
                parts.append("<p><b>%s</b> - %s</p>%s" % (html.escape(pretty(p)), m.upper(),
                                                          table_html(t)))
        parts.append("<h3>Suggested walk-forward search space</h3>"
                     "<p class='note'>Per axis, levels whose pooled mean z is within one "
                     "standard error of the best level, on CORR and on MMC (intersection; "
                     "the union if they disagree). 'pin' = no detectable main effect "
                     "(p ≥ 0.05 on both metrics): hold it at one kept level and spend the "
                     "budget elsewhere.</p>%s" % table_html(r["grid"]))
        r["grid"].to_csv(os.path.join(out_dir, "%s_suggested_grid.csv" % r["label"]), index=False)
        for m in METRICS:
            r["axes"][m].to_csv(os.path.join(out_dir, "%s_axes_%s.csv" % (r["label"], m)), index=False)
    parts.append("</body></html>")
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    return path


def print_summary(results):
    for r in results:
        print("=" * 80)
        print("%s  (%d candidates, %d refit-blocks)" % (r["label"], len(r["d"]), r["d"]["block"].nunique()))
        for m in METRICS:
            print("-- %s" % m.upper())
            print(r["axes"][m][["axis", "levels", "eta2", "excess", "p", "kendall_w"]]
                  .to_string(index=False, float_format=lambda x: "%.4f" % x))
        print("-- suggested grid")
        print(r["grid"].to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--group", action="append", required=True,
                    help="label=path1,path2 (several files pooled under one label)")
    ap.add_argument("--title", default="Hyperparameter importance")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    results = []
    for g in args.group:
        label, _, files = g.partition("=")
        results.append(analyse(label, [f for f in files.split(",") if f], args.seed))
    print_summary(results)
    path = render(results, args.title, args.out)
    print("html -> %s" % path)


if __name__ == "__main__":
    main()
