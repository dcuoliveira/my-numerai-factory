#!/usr/bin/env python3
"""
E150 -- does the usefulness of each coin's causal volatility depend on macro volatility regimes?

    python3 experiments/e150_macrovol_evaluation.py --fit --model lgb_ktz --n-jobs 6 --arms VC_COV F_GOLD
    python3 experiments/e150_macrovol_evaluation.py --fit --model lgb_ktz --tune --n-jobs 6 --arms VC_COV ALL
    python3 experiments/e150_macrovol_evaluation.py --select        # dev only: writes parsim.json (PARSIM's families)
    python3 experiments/e150_macrovol_evaluation.py --score         # every arm, dev and the final block

Dataset crypto/v2.0-vol-causal-macrovol (src/data_mani/build_vol_macrovol.py): v2.0 + the vol-causal block
(22 levels + 132 own-history ranks) + 578 feature_mvx_ columns: per macro source (gold, S&P 500, DXY,
10-year yield, VIX level, VIX vol-of-vol) 4 interaction families with each crypto estimator (sgn, dist,
hi, lo) and coin-level exposures (cov, corr, beta, resvol, r2 over 60/120 trading days).

MATCHED COVERED SAMPLE. Every candidate arm, B_COV and VC_COV is trained and predicted ONLY on the rows of
one common mask M = cov_all & cov_tsr_all & cov_mvx_all, further restricted to complete observations
of that arm's actual features. High and low gates are separate arms; inactive gates never enter
training or inference. Each candidate gets B_COV and VC_COV comparators fitted on its identical rows.
Folds, seeds, fixed configs and six-candidate search budgets are identical across comparisons.

ROUTING (R_SLOT, the deployment rule). B_ALL is v2.0's model on every row (reused B0 of E133/E135). Each
arm's predictions reorder the covered coins inside the rank positions B_ALL gave them; every uncovered coin
keeps B_ALL's exact rank, so each scored date carries all of that date's coins. Comparisons are routed arm
against routed arm, so the covered sample cancels:
    VC_COV vs B_COV      what the vol-causal block adds on the covered sample
    F_<SRC> vs VC_COV    one macro source's margin       ALL vs VC_COV     all six
    ALL vs LOO_<SRC>     a source's margin given the rest PARSIM vs VC_COV  the dev-selected combination
    NO_EXPO / EXPO_ONLY  interactions without / only the coin-level exposures
    CTL_SCALAR           positive raw v x m products: rank-equivalent to v by construction, expected null
    PLC_CIRC             macro rotated by 504 trading days: autocorrelation kept, timing broken
    PLC_FUTURE           macro shifted 20 trading days INTO THE FUTURE: look-ahead, never used to select
B_COV vs B_ALL is also reported: what training on the covered sample alone does.

PARSIM is chosen on dev only (--select), by a rule frozen here: a source qualifies when its F_<SRC> beats
VC_COV on routed dev CORR at NW t >= 2 for at least 2 of the 3 models, with routed dev payout t > -2 for
all three. PARSIM = VC + every qualifying source. If none qualifies there is no PARSIM arm.

The legacy final block (2026-02-20..08-20) was scored by earlier studies, so this runner now fits and
scores development blocks ONLY. --research-retrospective explicitly permits the unverified source
histories for exploration; corrected fits use separate outputs and never reuse legacy E150 models.

THIN BLOCKS. The mask needs about a year of history (365-day own-history ranks, 120 prior estimates for
zc), so block 0 (2020) has 2 covered training rows. A block with fewer than MIN_TRAIN covered training
rows is not fitted for any arm and its dates are dropped from every routed comparison (thin_blocks()).
"""
import argparse
import hashlib
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
import e134_route_feature_sets as R  # noqa: E402
import e135_route_histfe as Q  # noqa: E402

os.chdir(REPO)
sys.path.insert(0, REPO)
from src.data_mani import build_vol_macrovol as MV  # noqa: E402
from src.data_mani.macrovol_contract import covered_rows  # noqa: E402

# New outputs must never reuse models trained with neutral-imputed gates.
ROOT = "results/reports/crypto/e150_macrovol_strict"
PRED = os.path.join(ROOT, "preds")
DIR = "data/crypto/v2.0-vol-causal-macrovol"
EXTRA = {"CTL_SCALAR": DIR + "-control-scalar", "PLC_CIRC": DIR + "-placebo-circular", "PLC_FUTURE": DIR + "-placebo-future-LOOKAHEAD"}
MODELS, SEEDS = E.MODELS, R.SEEDS
SOURCES = MV.SOURCES
MASK = ["cov_all", "cov_tsr_all", "cov_mvx_all"]
H.GRID["mlp_ktz"] = {"mlp__alpha": [0.01, 0.1, 1.0], "mlp__learning_rate_init": [0.01, 0.05011872336272722]}   # E137: alpha 100 collapses
TUNED = ["B_COV", "VC_COV", "ALL", "PARSIM"]
MIN_TRAIN = 5000            # a block with fewer covered training rows is neither fitted nor scored


def source_cols(k, mvx):
    """A source's interactions (incl. hi/lo) and, for the five change series, its exposures."""
    cols = [c for c in mvx if c.startswith("feature_mvx_%s_" % k)]
    if k == "vixvov":
        cols += [c for c in mvx if c.startswith("feature_mvx_vixchg_")]
    return cols


def is_expo(c):
    return c.rsplit("_", 2)[-2] in ("cov", "corr", "beta", "resvol", "r2")


def arms(cols, extra_cols):
    vol = [c for c in cols if c.startswith("feature_vol_")]
    mvx = [c for c in cols if c.startswith("feature_mvx_")]
    base = [c for c in cols if c.startswith("feature_") and c not in vol + mvx]
    assert len(base) == 22 and len(vol) == 154 and len(mvx) == 578, (len(base), len(vol), len(mvx))
    gated = [c for c in mvx if "_hi_" in c or "_lo_" in c]
    mvx = [c for c in mvx if c not in gated]
    VC = base + vol
    A = {"B_COV": base, "VC_COV": VC, "ALL": VC + mvx,
         "NO_EXPO": VC + [c for c in mvx if not is_expo(c)], "EXPO_ONLY": VC + [c for c in mvx if is_expo(c)]}
    for k in SOURCES:
        A["F_" + k.upper()] = VC + source_cols(k, mvx)
        A["LOO_" + k.upper()] = VC + [c for c in mvx if c not in set(source_cols(k, mvx))]
        for gate in ("hi", "lo"):
            # Opposite gates must be separate arms. Their inactive rows never
            # enter fitting or prediction, and each has matched comparators.
            A["GATE_%s_%s" % (k.upper(), gate.upper())] = VC + source_cols(k, mvx) + [
                c for c in gated if c.startswith("feature_mvx_%s_%s_" % (k, gate))]
    assert sorted(sum((source_cols(k, mvx) for k in SOURCES), [])) == sorted(mvx)
    for a, c in extra_cols.items():
        A[a] = VC + c
    sel = os.path.join(ROOT, "parsim.json")
    if os.path.exists(sel):
        picked = json.load(open(sel))["sources"]
        if picked:
            A["PARSIM"] = VC + sum((source_cols(k, mvx) for k in picked), [])
    return A


FIT_ARMS = ["B_COV", "VC_COV"] + ["F_" + k.upper() for k in SOURCES] + ["ALL", "NO_EXPO", "EXPO_ONLY"] + \
           ["LOO_" + k.upper() for k in SOURCES] + list(EXTRA) + \
           ["GATE_%s_%s" % (k.upper(), g) for k in SOURCES for g in ("HI", "LO")]


def load(need_extra=()):
    df = pd.read_parquet(os.path.join(DIR, "train.parquet")).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] < H.TEST_START].reset_index(drop=True)
    extra_cols = {}
    for a in need_extra:
        x = pd.read_parquet(os.path.join(EXTRA[a], "train.parquet")).reset_index()
        x["date"] = pd.to_datetime(x["date"])
        tag = {"CTL_SCALAR": "ctl", "PLC_CIRC": "circ", "PLC_FUTURE": "fut"}[a]
        new = [c for c in x if c.startswith("feature_mvx_")]
        x = x.rename(columns={c: c.replace("feature_mvx_", "feature_%s_" % tag) for c in new})
        df = df.merge(x, on=["symbol", "date"], how="left", validate="1:1")
        extra_cols[a] = [c.replace("feature_mvx_", "feature_%s_" % tag) for c in new]
    df["covered"] = df[MASK].all(axis=1)
    return df, extra_cols


def thin_blocks():
    """Walk-forward blocks with fewer than MIN_TRAIN covered training rows, from the mask alone."""
    d = pd.read_parquet(os.path.join(DIR, "train.parquet"), columns=["date"] + MASK).reset_index()
    d["date"] = pd.to_datetime(d["date"])
    d["c"] = d[MASK].all(axis=1)
    past = np.array(sorted(d.loc[d["date"] < H.TEST_START, "date"].unique()))
    return {i for i, tr, _ in H.blocks(past) if d.loc[d["date"].isin(tr), "c"].sum() < MIN_TRAIN}


def fit(args):
    need = [a for a in (args.arms or FIT_ARMS) if a in EXTRA]
    df, extra = load(need)
    A = arms([c for c in df if c.startswith("feature_") and not c.startswith(("feature_ctl_", "feature_circ_", "feature_fut_"))], extra)
    past = np.array(sorted(df.loc[df["date"] < H.TEST_START, "date"].unique()))
    # The old final block is already contaminated by earlier studies. Never
    # open it again here or describe it as an untouched test.
    B = H.blocks(past)
    names = args.arms or FIT_ARMS
    seeds = [0] if (args.tune or args.model == "ridge") else SEEDS[args.model]
    tag = "tune" if args.tune else ""
    for arm in names:
        if arm not in A:
            print("no arm", arm, "(PARSIM needs --select first, or nothing qualified)")
            continue
        cols = A[arm]
        cov = df[df["covered"] & covered_rows(df, cols)]
        # Every comparison uses the candidate's exact training/inference rows.
        specs = [(arm, cols)]
        if arm not in ("B_COV", "VC_COV"):
            specs += [(arm + "__B_COV", A["B_COV"]), (arm + "__VC_COV", A["VC_COV"])]
        fit_covered_specs(args, specs, cov, B, seeds, tag)


def fit_covered_specs(args, specs, cov, blocks, seeds, tag):
    for arm, cols in specs:
        digest = hashlib.sha256(pd.util.hash_pandas_object(cov[["symbol", "date"]], index=False).values.tobytes()
                                + "\0".join(cols).encode()).hexdigest()
        for s in seeds:
            path = os.path.join(PRED, args.model, "%s%s_s%d.parquet" % (tag, arm, s))
            if os.path.exists(path):
                print("skip", path)
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cache = os.path.join(PRED, args.model, "_matched_cache", "%s%s_s%d.parquet" % (tag, digest, s))
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            if os.path.exists(cache) and (not args.tune or os.path.exists(cache + ".picks.json")):
                os.link(cache, path)
                if args.tune:
                    os.link(cache + ".picks.json", path + ".picks.json")
                print("reuse identical covered sample", arm, "seed", s, flush=True)
                continue
            parts, picks, t1 = [], [], time.time()
            for i, tr, te in blocks:
                trm, tem = cov["date"].isin(tr), cov["date"].isin(te)
                if trm.sum() < MIN_TRAIN or not tem.any():
                    continue
                if args.tune:
                    inner_train, inner_val = tr[:-(H.INNER_VAL + H.PURGE)], tr[-H.INNER_VAL:]
                    if cov["date"].isin(inner_train).sum() < 100 or cov["date"].isin(inner_val).sum() < 100:
                        continue
                params = H.pick(args.model, cov[trm], cols, tr, s, args.n_jobs) if args.tune else H.FIXED[args.model]
                if params is None:      # every candidate scored NaN on the inner split (constant predictions)
                    params = dict(H.FIXED[args.model], fallback=True)
                picks.append({k: str(params.get(k)) for k in list(H.GRID[args.model]) + ["fallback"]})
                params = {k: v for k, v in params.items() if k != "fallback"}
                pred = H.fit_predict(args.model, params, s, args.n_jobs, cov.loc[trm, cols].to_numpy(np.float32),
                                     cov.loc[trm, H.T20].to_numpy(), cov.loc[tem, cols].to_numpy(np.float32))
                o = cov.loc[tem, ["symbol", "date"]].copy()
                o["prediction"], o["block"] = pred, i
                parts.append(o)
                print("%s %s seed %d block %d done" % (args.model, arm, s, i), flush=True)
            if not parts:
                print("no causally observed training coverage", arm, flush=True)
                json.dump(dict(arm=arm, model=args.model, seed=s, searched=args.tune,
                               reason="No fold has sufficient observed training and evaluation rows",
                               covered_rows=len(cov), minimum_training_rows=MIN_TRAIN),
                          open(path + ".unavailable.json", "w"), indent=2)
                continue
            tmp = path + ".partial"
            pd.concat(parts).to_parquet(tmp, index=False)
            os.replace(tmp, path)
            if args.tune:
                json.dump(picks, open(path + ".picks.json", "w"))
            try:
                os.link(path, cache)
                if args.tune:
                    os.link(path + ".picks.json", cache + ".picks.json")
            except FileExistsError:
                pass
            print("%s %-12s %s seed %d  %d cols  %.0fs" % (args.model, arm, tag, s, len(cols), time.time() - t1), flush=True)


# ------------------------------------------------------------------ scoring
def b_all(model, seed):
    return Q.preds(model, "B0_s%d" % seed).rename(columns={"prediction": "b0"})


def arm_preds(model, arm, seed):
    p = pd.read_parquet(os.path.join(PRED, model, "%s_s%d.parquet" % (arm, seed)), columns=["symbol", "date", "prediction", "block"])
    p = p[~p["block"].isin(THIN)].drop(columns="block")
    p["date"] = pd.to_datetime(p["date"])
    return p.rename(columns={"prediction": "cand"})


THIN = thin_blocks() if os.path.exists(os.path.join(DIR, "train.parquet")) else set()


def score_job(job):
    """Routed per-date CORR / MMC / payout of one arm (or B_ALL) for one model and seed."""
    model, arm, seed, kinds = job
    k = pd.read_parquet(Q.KEYS, columns=["symbol", "date", H.T20, "meta_model", "block", "kind"])
    k["date"] = pd.to_datetime(k["date"])
    k = k[k["kind"].isin(kinds) & ~k["block"].isin(THIN)]
    tseed = 0 if arm.startswith("tune") else seed
    f = k.merge(b_all(model, tseed), on=["symbol", "date"], how="left", validate="1:1")
    if arm != "B_ALL":
        f = f.merge(arm_preds(model, arm, seed), on=["symbol", "date"], how="left", validate="1:1")
    else:
        f["cand"] = np.nan
    assert f["b0"].notna().all()
    f["b0"] = f.groupby("date")["b0"].rank(pct=True)
    rows = []
    for d, g in f.groupby("date", sort=True):
        g = g.reset_index(drop=True)
        cov = g["cand"].notna().to_numpy()
        x = R.slot(g["b0"].to_numpy(), g["cand"].to_numpy(), cov) if cov.any() else g["b0"].to_numpy()
        assert np.array_equal(x[~cov], g["b0"].to_numpy()[~cov])
        assert np.isfinite(x).all(), "Routed predictions must contain explicit baseline fallbacks"
        h = g.assign(prediction=x).set_index("symbol")
        c = E.corr_one(h)
        m = E.contrib_one(h, "meta_model") if h["meta_model"].notna().sum() >= 10 else np.nan
        rows.append(dict(model=model, arm=arm, seed=seed, date=d, block=int(g["block"].iloc[0]), kind=g["kind"].iloc[0],
                         corr=c, mmc=m, payout=0.1 * c + m if m == m else np.nan, covered=cov.mean()))
    return pd.DataFrame(rows)


def jobs(arms_, kinds, tuned=False):
    out = []
    for m in MODELS:
        for a in arms_:
            for s in ([0] if tuned or m == "ridge" else SEEDS[m]):
                name = ("tune" + a) if tuned else a
                if a == "B_ALL" or os.path.exists(os.path.join(PRED, m, "%s_s%d.parquet" % (name, s))):
                    out.append((m, name, s, kinds))
    return out


def per_date(S):
    return S.groupby(["model", "arm", "date", "block", "kind"], as_index=False)[["corr", "mmc", "payout", "covered"]].mean()


def paired(A, a, b, m, sc, metric):
    x = A[(A["model"] == m) & (A["arm"] == a) & (A["kind"] == sc)].set_index("date")
    y = A[(A["model"] == m) & (A["arm"] == b) & (A["kind"] == sc)].set_index("date")
    z = x[[metric, "block"]].join(y[[metric]], rsuffix="_b", how="inner").dropna()
    d = z[metric] - z[metric + "_b"]
    if len(d) < 5:
        return None
    r = E.describe(d)
    worst = d.groupby(z["block"]).mean()
    return dict(model=m, arm=a, vs=b, scope=sc, metric=metric, delta=r["mean"], se=r["nw_se"],
                t=r["nw_t"], lo=r["boot_lo"], hi=r["boot_hi"], win=float((d > 0).mean()), worst_block=float(worst.min()),
                worst_block_id=int(worst.idxmin()), dd=E.max_dd(d.values), n=len(d))


PAIRS = [("VC_COV", "B_COV"), ("B_COV", "B_ALL"), ("VC_COV", "B_ALL")] + [("F_" + k.upper(), "VC_COV") for k in SOURCES] + \
        [("ALL", "VC_COV"), ("NO_EXPO", "VC_COV"), ("EXPO_ONLY", "VC_COV"), ("PARSIM", "VC_COV")] + \
        [("ALL", "LOO_" + k.upper()) for k in SOURCES] + [(a, "VC_COV") for a in EXTRA] + [("ALL", "PLC_CIRC"), ("ALL", "CTL_SCALAR")] + \
        [("tuneVC_COV", "tuneB_COV"), ("tuneALL", "tuneVC_COV"), ("tunePARSIM", "tuneVC_COV")]
PAIRS += [(prefix + a, prefix + a + "__VC_COV") for prefix in ("", "tune")
          for a in FIT_ARMS + ["PARSIM"] if a not in ("B_COV", "VC_COV")]
PAIRS += [(prefix + a + "__VC_COV", prefix + a + "__B_COV") for prefix in ("", "tune")
          for a in FIT_ARMS + ["PARSIM"] if a not in ("B_COV", "VC_COV")]


def regimes(source="vixlvl"):
    """Per panel date, the VIX-level percentile attached as-of D-1 (the same pm the hi/lo gates read)."""
    store = pd.read_parquet(MV.MACRO)
    ch, lvl = MV.macro_changes(store, "main")
    mv = MV.macro_vol(ch, lvl)
    f = mv[mv["source"] == source].drop(columns="source").assign(src=source)
    _, p, _ = MV.prior_scores(f, ["sqret_20d"], "src", MV.ZM_OBS, MV.ZM_MIN, calendar=False)
    g = pd.DataFrame({"date": f["date"].reset_index(drop=True), "pm": p["sqret_20d"].reset_index(drop=True), "symbol": "__m__"})
    k = pd.read_parquet(Q.KEYS, columns=["date"]).drop_duplicates()
    k["date"] = pd.to_datetime(k["date"])
    a = MV.attach_signals_asof(k.assign(symbol="__m__"), g, ["pm"], lag_days=MV.LAG, max_staleness_days=MV.STALE)
    a["regime"] = np.where(a["pm"] >= MV.HI, "high", np.where(a["pm"] <= MV.LO, "low", "mid"))
    return a[["date", "pm", "regime"]]


def summarise(S, kinds):
    A = per_date(S)
    rows = []
    for a, b in PAIRS:
        for m in MODELS:
            for sc in kinds:
                for metric in ("corr", "mmc", "payout"):
                    r = paired(A, a, b, m, sc, metric)
                    if r:
                        rows.append(r)
    D = pd.DataFrame(rows)
    lev = []
    for (m, a, sc), g in A.groupby(["model", "arm", "kind"]):
        p = g.dropna(subset=["payout"])
        dd = E.max_dd(p["payout"].values)
        lev.append(dict(model=m, arm=a, scope=sc, corr=g["corr"].mean(), mmc=p["mmc"].mean(), c_sharpe=g["corr"].mean() / g["corr"].std(ddof=0),
                        dd_corr=E.max_dd(g["corr"].values), payout=p["payout"].mean(), p_sharpe=p["payout"].mean() / p["payout"].std(ddof=0),
                        dd_pay=dd, calmar=p["payout"].mean() / dd if dd > 0 else np.nan, burn=int((p["payout"] < 0).sum()),
                        covered=g["covered"].mean(), n=len(g), n_pay=len(p)))
    return A, D, pd.DataFrame(lev)


def select():
    """Dev only. Fixed rule (module docstring); writes parsim.json. Never reads a test date."""
    names = [name for k in SOURCES for name in ("F_" + k.upper(), "F_" + k.upper() + "__VC_COV")]
    missing = [os.path.join(PRED, m, "tune%s_s0.parquet" % a) for m in MODELS for a in names
               if not os.path.exists(os.path.join(PRED, m, "tune%s_s0.parquet" % a))]
    if missing:
        raise RuntimeError("Selection needs searched candidates and matched comparators for all three models: " + str(missing))
    with ProcessPoolExecutor(int(os.environ.get("E150_WORKERS", "4"))) as pool:
        S = pd.concat(list(pool.map(score_job, jobs(names, ["dev"], tuned=True))), ignore_index=True)
    A = per_date(S)
    picked, why = [], {}
    for k in SOURCES:
        a = "tuneF_" + k.upper()
        c = [paired(A, a, a + "__VC_COV", m, "dev", "corr") for m in MODELS]
        p = [paired(A, a, a + "__VC_COV", m, "dev", "payout") for m in MODELS]
        ok = all(r is not None for r in c + p) and sum(r["t"] >= 2 and r["delta"] > 0 for r in c) >= 2 and all(r["t"] > -2 for r in p)
        why[k] = dict(corr_t={r["model"]: round(r["t"], 2) for r in c if r}, payout_t={r["model"]: round(r["t"], 2) for r in p if r}, qualifies=bool(ok))
        if ok:
            picked.append(k)
    json.dump(dict(rule="searched F_<SRC> vs matched VC_COV, routed dev CORR t >= 2 on >= 2 of 3 models and dev payout t > -2 on all", sources=picked, detail=why),
              open(os.path.join(ROOT, "parsim.json"), "w"), indent=1)
    print(json.dumps(why, indent=1), "\nPARSIM:", picked or "none qualifies")


def score():
    kinds = ["dev"]
    arms_ = ["B_ALL"] + FIT_ARMS + (["PARSIM"] if os.path.exists(os.path.join(ROOT, "parsim.json")) and json.load(open(os.path.join(ROOT, "parsim.json")))["sources"] else [])
    matched = [a + suffix for a in arms_ if a not in ("B_ALL", "B_COV", "VC_COV")
               for suffix in ("__B_COV", "__VC_COV")]
    J = jobs(arms_ + matched, kinds) + jobs(arms_[1:] + matched, kinds, tuned=True)
    with ProcessPoolExecutor(int(os.environ.get("E150_WORKERS", "16"))) as ex:
        S = pd.concat(list(ex.map(score_job, J, chunksize=2)), ignore_index=True)
    S.to_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"), index=False)
    A, D, L = summarise(S, kinds)
    D.to_csv(os.path.join(ROOT, "paired_deltas.csv"), index=False)
    L.to_csv(os.path.join(ROOT, "levels.csv"), index=False)
    rows = []
    for source in SOURCES:
        by_regime = A.merge(regimes(source), on="date", how="left")
        for a, b in PAIRS:
            for m in MODELS:
                for rg in ("high", "mid", "low"):
                    for sc in kinds:
                        r = paired(by_regime[by_regime["regime"] == rg], a, b, m, sc, "corr")
                        if r:
                            rows.append(dict(r, regime=rg, macro_source=source))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "regime_deltas.csv"), index=False)
    seed = []
    for (m, s), g in S[S["model"] != "ridge"].groupby(["model", "seed"]):
        A1 = per_date(g)
        for a, b in [("ALL", "VC_COV"), ("VC_COV", "B_COV")] + [("F_" + k.upper(), "VC_COV") for k in SOURCES]:
            for sc in kinds:
                r = paired(A1, a, b, m, sc, "corr")
                if r:
                    seed.append(dict(r, seed=s))
    pd.DataFrame(seed).to_csv(os.path.join(ROOT, "seed_deltas.csv"), index=False)
    cov = pd.read_parquet(os.path.join(DIR, "train.parquet"), columns=["date"] + MASK).reset_index()
    cov["date"] = pd.to_datetime(cov["date"])
    cov = cov[cov["date"] <= Q.LAST]
    cov["covered"] = cov[MASK].all(axis=1)
    k = pd.read_parquet(Q.KEYS, columns=["symbol", "date", "kind"])
    k["date"] = pd.to_datetime(k["date"])
    z = k.merge(cov, on=["symbol", "date"], how="left")
    z.groupby([z["date"].dt.year, "kind"])[MASK + ["covered"]].mean().round(4).reset_index().to_csv(os.path.join(ROOT, "coverage_by_year.csv"), index=False)
    print("wrote", ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--dataset-dir", default=DIR)
    ap.add_argument("--research-retrospective", action="store_true",
                    help="Explicit exploratory comparison on existing unverified datasets; development only")
    a = ap.parse_args()
    global ROOT, PRED, EXTRA, THIN
    dataset_dir = a.dataset_dir
    # DIR is kept on the module for the scorer and worker processes.
    globals()["DIR"] = dataset_dir
    stem = dataset_dir.removesuffix("-retrospective")
    suffix = "-retrospective" if dataset_dir.endswith("-retrospective") else ""
    EXTRA = {"CTL_SCALAR": stem + "-control-scalar" + suffix,
             "PLC_CIRC": stem + "-placebo-circular" + suffix,
             "PLC_FUTURE": stem + "-placebo-future-LOOKAHEAD" + suffix}
    if a.research_retrospective:
        ROOT = "results/reports/crypto/e150_macrovol_observed_dev"
        PRED = os.path.join(ROOT, "preds")
    manifest = json.load(open(os.path.join(DIR, "manifest.json")))
    if not manifest.get("macro_availability_verified", False) and not a.research_retrospective:
        raise SystemExit("Dataset is retrospective: rebuild with verified macro vintages before fitting or selection. "
                         "Existing E150 predictions are not compliant and will not be reused.")
    THIN = thin_blocks()
    os.makedirs(ROOT, exist_ok=True)
    if a.fit and not a.model:
        ap.error("--fit requires --model")
    registration = dict(scope="development only; no untouched test opened", retrospective=a.research_retrospective,
                        model_aliases=E.LABEL, missing="no imputation; inactive gates excluded from training and inference",
                        routing="slot preserving; explicit baseline ranks for uncovered rows",
                        inputs={p: MV.sha256(p) for p in [os.path.join(DIR, "train.parquet"),
                                                        os.path.join(DIR, "manifest.json"), MV.__file__]},
                        macro_source_inputs=manifest["inputs"],
                        runner_sha256=MV.sha256(__file__), search_grid=H.GRID, fixed=H.FIXED)
    frozen = os.path.join(ROOT, "registration.json")
    if os.path.exists(frozen):
        if json.load(open(frozen)) != json.loads(json.dumps(registration)):
            raise SystemExit("Inputs or experiment rules differ from registration; use a separate output directory")
    else:
        # Exclusive creation prevents two simultaneous launches replacing rules.
        try:
            with open(frozen, "x") as out:
                json.dump(registration, out, indent=2)
        except FileExistsError:
            raise SystemExit("Registration being created by another runner; retry")
    if a.fit:
        fit(a)
    elif a.select:
        select()
    elif a.score:
        score()


if __name__ == "__main__":
    main()
