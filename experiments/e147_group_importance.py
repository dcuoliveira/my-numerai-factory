"""E147 -- group-only importance on crypto/v2.0-c2syn-c2ft: ridge and an MLP tuned from scratch on ONE feature group at a time.

    python3 experiments/e147_group_importance.py --fit --model ridge
    python3 experiments/e147_group_importance.py --fit --model mlp_ktz --groups C2COV_FULL --n-jobs 4
    E147_WORKERS=8 python3 experiments/e147_group_importance.py --score

Question: how much does each group predict ON ITS OWN? This is standalone importance, not marginal
value (E143/E144/E146 measured that, with the 22 shipped features underneath). A group can carry a lot
alone and add nothing to v2.0, because what it knows is already in v2.0.

Groups (columns):
  MOM    momentum, sharpe_ratio (20d, 60d)                       4
  REV    relative_strength_index, bollinger (20d, 60d)           4
  VOLU   volume_avg, volume_ewa (20d, 60d)                       4
  VOLA   volatility (20d, 60d)                                   2
  PRICE  close_avg, close_ewa (20d, 60d)                         4
  SIZE   market_cap_avg, market_cap_ewa (20d, 60d)               4
  C2COV / C2AR / C2FTCOV / C2FTAR              feature_<g>_pc01..16     16 each
  C2COV_FULL / C2AR_FULL / C2FTCOV_FULL / C2FTAR_FULL   feature_<g>_d000..767   768 each
  ALL22  the 22 shipped features, tuned the same way: the reference every group is read against.

From scratch: per block, each (group, model) picks its own hyperparameters on the last 40 train dates
after a 22-date purge (E133's inner split), then refits on the whole train window.
  ridge   alpha in {1, 10, 65.8, 316, 1000, 10000}                              one fit (deterministic)
  mlp     (32,16), activation {identity, relu} x alpha {0.01, 1} x lr {0.01, 0.05}   picked on seed 0,
          refit on seeds 0-4. alpha 100 is left out: it collapses the MLP to a constant (E137).
Fine-tuned columns differ by block and are swapped in from blocks.parquet (as E144/E146). Missing
states are 0.5. Blocks 0-1 have no fine-tuned model, so a fine-tuned group predicts a constant there.

Scoring on two row sets, both on E135's 468,000 keys:
  all      every key; a date where an arm predicts a constant scores CORR 0 (it ranks nothing).
  common   only coins whose fine-tuned state is observed that date (implies zero-shot observed and all
           shipped features), so every group is scored on IDENTICAL rows. Blocks 2-11. This is the
           ranking table: on 'all', a state group is also penalised for its coverage gap.
CORR per date (numerai_corr), proxy MMC against the lagged meta model, payout = 0.1*CORR + MMC per date.
Level t and every paired delta: Newey-West lag 20 + block bootstrap. Block 11 (2026-02-20..08-20) is
reported apart and selects nothing.
"""
import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e131_analysis as E  # noqa: E402
import e133_histfe_factorial as H  # noqa: E402
import e135_route_histfe as Q  # noqa: E402

os.chdir(REPO)

ROOT = "results/reports/crypto/e147_group_importance"
DIR = "data/crypto/v2.0-c2syn-c2ft"
MODELS = ["ridge", "mlp_ktz"]
SEEDS = {"ridge": [0], "mlp_ktz": list(range(5))}
SHIPPED = {"MOM": ("momentum_", "sharpe_ratio_"), "REV": ("relative_strength_index_", "bollinger_"),
           "VOLU": ("volume_",), "VOLA": ("volatility_",), "PRICE": ("close_",), "SIZE": ("market_cap_",)}
FAMS = ("c2cov", "c2ar", "c2ftcov", "c2ftar")
GROUPS = list(SHIPPED) + [f.upper() for f in FAMS] + [f.upper() + "_FULL" for f in FAMS] + ["ALL22"]
GRID = {"ridge": [dict(H.FIXED["ridge"], alpha=a) for a in H.GRID["ridge"]["alpha"]],
        "mlp_ktz": [dict(H.FIXED["mlp_ktz"], **{"mlp__activation": act, "mlp__alpha": al, "mlp__learning_rate_init": lr})
                    for act, al, lr in itertools.product(["identity", "relu"], [0.01, 1.0], [0.01, 0.05011872336272722])]}
COMPARE = ([(g, "ALL22", "group alone vs all 22 shipped") for g in GROUPS[:-1]]
           + [(f.upper() + "_FULL", f.upper(), "768 dims vs 16 PCs") for f in FAMS]
           + [("C2FTCOV", "C2COV", "fine-tuned vs zero-shot, cov PCs"), ("C2FTAR", "C2AR", "fine-tuned vs zero-shot, ar PCs"),
              ("C2FTCOV_FULL", "C2COV_FULL", "fine-tuned vs zero-shot, cov dims"), ("C2FTAR_FULL", "C2AR_FULL", "fine-tuned vs zero-shot, ar dims"),
              ("C2COV", "C2AR", "covariates vs label only, zero-shot PCs"), ("C2FTCOV", "C2FTAR", "covariates vs label only, fine-tuned PCs"),
              ("C2COV_FULL", "C2AR_FULL", "covariates vs label only, zero-shot dims"),
              ("C2FTCOV_FULL", "C2FTAR_FULL", "covariates vs label only, fine-tuned dims")])


def columns(cols):
    base = [c for c in cols if c.startswith("feature_") and not any(c.startswith("feature_%s_" % f) for f in FAMS)]
    assert len(base) == 22, len(base)
    G = {g: [c for c in base if c[len("feature_"):].startswith(k)] for g, k in SHIPPED.items()}
    for f in FAMS:
        G[f.upper()] = ["feature_%s_pc%02d" % (f, i) for i in range(1, 17)]
        G[f.upper() + "_FULL"] = ["feature_%s_d%03d" % (f, i) for i in range(768)]
    G["ALL22"] = base
    assert [len(G[g]) for g in GROUPS] == [4, 4, 4, 2, 4, 4] + [16] * 4 + [768] * 4 + [22], {g: len(v) for g, v in G.items()}
    assert sorted(sum((G[g] for g in SHIPPED), [])) == sorted(base)
    return G


def frames():
    """Per block: train + predict rows, with the block's own fine-tuned columns swapped in."""
    df = pd.read_parquet(os.path.join(DIR, "train.parquet")).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= Q.LAST].sort_values(["date", "symbol"]).reset_index(drop=True)
    G = columns(list(df))
    ft = [c for c in df if c.startswith("feature_c2ft")]
    keep = sorted(set(sum(G.values(), [])) - set(ft))
    df = df[["symbol", "date", H.T20] + keep]
    past = np.array(sorted(df.loc[df["date"] < H.TEST_START, "date"].unique()))
    k = pd.read_parquet(Q.KEYS, columns=["date", "kind"])
    last = pd.to_datetime(k.loc[k["kind"] == "test", "date"]).unique()
    B = H.blocks(past) + [(11, past[-(H.PURGE + H.TRAIN):-H.PURGE], last)]
    L = pd.read_parquet(os.path.join(DIR, "blocks.parquet"), columns=["block", "symbol", "date"] + ft)
    L["date"] = pd.to_datetime(L["date"])
    F = {}
    for i, tr, te in B:
        rows = df["date"].isin(tr) | df["date"].isin(te)
        d = df.loc[rows].merge(L.loc[L["block"] == i].drop(columns="block"), on=["symbol", "date"], how="left", validate="1:1")
        assert len(d) == rows.sum()
        print("block %d: %d rows, own fine-tuned state on %.1f%% of predict rows" % (
            i, len(d), 100 * d.loc[d["date"].isin(te), ft[0]].notna().mean()), flush=True)
        feats = [c for c in d if c.startswith("feature_")]
        d[feats] = d[feats].astype("float32").fillna(0.5)
        F[i] = d.reset_index(drop=True)
    return F, B, G


def pick(model, d, cols, train, n_jobs):
    """E133's inner split: fit on train minus the last 40 + 22 dates, score mean Spearman on the last 40."""
    fit_d, val_d = train[:-(H.INNER_VAL + H.PURGE)], train[-H.INNER_VAL:]
    fit, val = d[d["date"].isin(fit_d)], d[d["date"].isin(val_d)]
    Xf, yf, Xv = fit[cols].to_numpy(np.float32), fit[H.T20].to_numpy(), val[cols].to_numpy(np.float32)
    best, score, log = None, -np.inf, []
    for p in GRID[model]:
        pred = H.fit_predict(model, p, 0, n_jobs, Xf, yf, Xv)
        if np.std(pred) < 1e-12:                     # a constant ranks nothing
            s = -np.inf
        else:
            s = val.assign(p=pred).groupby("date")[["p", H.T20]].apply(lambda g: g["p"].corr(g[H.T20], method="spearman")).mean()
        log.append(float(s))
        if s > score:
            best, score = p, s
    return (best or GRID[model][0]), log


def fit(args):
    F, B, G = frames()
    names = args.groups or GROUPS
    for g in names:
        cols = G[g]
        todo = [s for s in SEEDS[args.model] if not os.path.exists(path(args.model, g, s))]
        if not todo:
            continue
        t1, picks, parts = time.time(), [], {s: [] for s in todo}
        for i, tr, te in B:
            d = F[i]
            trm, tem = d["date"].isin(tr), d["date"].isin(te)
            p, log = pick(args.model, d, cols, tr, args.n_jobs)
            picks.append(dict(block=i, val=log, **{k: str(p[k]) for k in ("alpha", "mlp__activation", "mlp__alpha", "mlp__learning_rate_init") if k in p}))
            Xtr, ytr, Xte = d.loc[trm, cols].to_numpy(np.float32), d.loc[trm, H.T20].to_numpy(), d.loc[tem, cols].to_numpy(np.float32)
            for s in todo:
                o = d.loc[tem, ["symbol", "date"]].copy()
                o["prediction"], o["block"] = H.fit_predict(args.model, p, s, args.n_jobs, Xtr, ytr, Xte), i
                parts[s].append(o)
        for s in todo:
            out = path(args.model, g, s)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            pd.concat(parts[s]).to_parquet(out + ".partial", index=False)
            os.replace(out + ".partial", out)
        json.dump(picks, open(os.path.join(ROOT, "preds", args.model, g + ".picks.json"), "w"))
        print("%s %-13s %4d cols  seeds %s  %.0fs" % (args.model, g, len(cols), todo, time.time() - t1), flush=True)


def path(model, g, s):
    return os.path.join(ROOT, "preds", model, "%s_s%d.parquet" % (g, s))


def keys():
    k = pd.read_parquet(Q.KEYS, columns=["symbol", "date", H.T20, "meta_model", "block", "kind"])
    k["date"] = pd.to_datetime(k["date"])
    c = pd.read_parquet(os.path.join(DIR, "blocks.parquet"), columns=["block", "role", "symbol", "date", "feature_c2ftcov_pc01"])
    c["date"] = pd.to_datetime(c["date"])
    c = c[c["role"] != "train"] if "role" in c else c
    k = k.merge(c[["block", "symbol", "date", "feature_c2ftcov_pc01"]], on=["block", "symbol", "date"], how="left", validate="1:1")
    k["common"] = k.pop("feature_c2ftcov_pc01").notna()
    return k


def priced(h):
    if h["prediction"].nunique() < 2:
        return 0.0, 0.0 if h["meta_model"].notna().sum() >= 10 else np.nan
    c = E.corr_one(h.set_index("symbol"))
    m = E.contrib_one(h.set_index("symbol"), "meta_model") if h["meta_model"].notna().sum() >= 10 else np.nan
    return c, m


def score_job(job):
    model, g, s = job
    p = pd.read_parquet(path(model, g, s))
    p["date"] = pd.to_datetime(p["date"])
    f = keys().merge(p.drop(columns="block"), on=["symbol", "date"], how="left", validate="1:1")
    assert f["prediction"].notna().all(), (model, g, s, f["prediction"].isna().sum())
    rows = []
    for d, h in f.groupby("date", sort=True):
        for rs, x in (("all", h), ("common", h[h["common"]])):
            if len(x) < 10:
                continue
            c, m = priced(x)
            rows.append(dict(model=model, group=g, seed=s, rows=rs, date=d, block=int(h["block"].iloc[0]), kind=h["kind"].iloc[0],
                             n_coins=len(x), corr=c, mmc=m, payout=0.1 * c + m if m == m else np.nan))
    return pd.DataFrame(rows)


def summarise(S):
    A = S.groupby(["model", "group", "rows", "date", "block", "kind"], as_index=False)[["corr", "mmc", "payout"]].mean()
    A.to_parquet(os.path.join(ROOT, "scores_by_date.parquet"), index=False)
    lev, dl, by, sd = [], [], [], []
    for (m, g, rs, sc), x in A.groupby(["model", "group", "rows", "kind"]):
        p = x.dropna(subset=["payout"])
        dd = E.max_dd(p["payout"].values)
        rc, rp = E.describe(x["corr"]), E.describe(p["payout"])
        lev.append(dict(model=m, group=g, rows=rs, scope=sc, corr=x["corr"].mean(), corr_t=rc["nw_t"], corr_lo=rc["boot_lo"], corr_hi=rc["boot_hi"],
                        mmc=p["mmc"].mean(), c_sharpe=x["corr"].mean() / x["corr"].std(ddof=0), dd_corr=E.max_dd(x["corr"].values),
                        payout=p["payout"].mean(), payout_t=rp["nw_t"], p_sharpe=p["payout"].mean() / p["payout"].std(ddof=0), dd_pay=dd,
                        calmar=p["payout"].mean() / dd if dd > 0 else np.nan, burn=int((p["payout"] < 0).sum()), n=len(x), n_pay=len(p)))
    for m in MODELS:
        for rs in ("all", "common"):
            w = A[(A["model"] == m) & (A["rows"] == rs)]
            for a, b, label in COMPARE:
                z = w[w["group"] == a].merge(w[w["group"] == b], on=["date", "kind", "block"], suffixes=("", "_b"))
                for metric in ("corr", "mmc", "payout"):
                    for sc in ("dev", "test"):
                        q = z[z["kind"] == sc]
                        r = E.describe((q[metric] - q[metric + "_b"]).dropna())
                        dl.append(dict(model=m, rows=rs, arm=a, vs=b, label=label, metric=metric, scope=sc, delta=r["mean"], t=r["nw_t"], lo=r["boot_lo"], hi=r["boot_hi"], n=r["n"]))
            for g, x in w.groupby("group"):
                for k, q in x.groupby("block"):
                    by.append(dict(model=m, rows=rs, group=g, block=k, corr=q["corr"].mean()))
    for (m, g, rs, s, sc), x in S.groupby(["model", "group", "rows", "seed", "kind"]):
        sd.append(dict(model=m, group=g, rows=rs, seed=s, scope=sc, corr=x["corr"].mean(), payout=x["payout"].mean()))
    pd.DataFrame(lev).to_csv(os.path.join(ROOT, "levels.csv"), index=False)
    pd.DataFrame(dl).to_csv(os.path.join(ROOT, "paired_deltas.csv"), index=False)
    pd.DataFrame(by).to_csv(os.path.join(ROOT, "by_block.csv"), index=False)
    pd.DataFrame(sd).to_csv(os.path.join(ROOT, "by_seed.csv"), index=False)
    picks = []
    for m in MODELS:
        for g in GROUPS:
            for r in json.load(open(os.path.join(ROOT, "preds", m, g + ".picks.json"))):
                picks.append(dict(model=m, group=g, **{k: v for k, v in r.items() if k != "val"}))
    pd.DataFrame(picks).to_csv(os.path.join(ROOT, "picks.csv"), index=False)


def score():
    jobs = [(m, g, s) for m in MODELS for g in GROUPS for s in SEEDS[m]]
    missing = [j for j in jobs if not os.path.exists(path(*j))]
    assert not missing, missing
    with ProcessPoolExecutor(int(os.environ.get("E147_WORKERS", "4"))) as ex:
        S = pd.concat(list(ex.map(score_job, jobs)), ignore_index=True)
    S.to_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"), index=False)
    summarise(S)
    print("wrote", ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--groups", nargs="*", choices=GROUPS)
    ap.add_argument("--n-jobs", type=int, default=1)
    a = ap.parse_args()
    if a.fit:
        fit(a)
    if a.score:
        score()


if __name__ == "__main__":
    main()
