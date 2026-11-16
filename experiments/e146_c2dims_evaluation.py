#!/usr/bin/env python3
"""
E146 -- the feature-set evaluation procedure for crypto/v2.0-c2syn-c2ft-dims.

    python3 experiments/e146_c2dims_evaluation.py --fit --model lgb_ktz --n-jobs 6 [--arms FULL F_ZS]
    python3 experiments/e146_c2dims_evaluation.py --fit --model lgb_ktz --tune --n-jobs 6
    python3 experiments/e146_c2dims_evaluation.py --audit
    python3 experiments/e146_c2dims_evaluation.py

22 shipped features + every dimension of the four 768-d Chronos-2-Synth encoder states the E143 / E144
PCs were taken from (src/data_mani/build_c2syn_c2ft.py --raw --no-pc), each rank-binned with
Numerai's rule: zero-shot dims within date, fine-tuned dims within (block, date).

  c2cov   feature_c2cov_d000..767    zero-shot, label + 22 features as covariates (E143 states)
  c2ar    feature_c2ar_d000..767     zero-shot, label alone
  c2ftcov feature_c2ftcov_d000..767  LoRA fine-tuned per walk-forward block, covariates (E144 states)
  c2ftar  feature_c2ftar_d000..767   fine-tuned, label alone

The fine-tuned dims DIFFER BY BLOCK, as in E144: fit() swaps each block's own values in from
blocks.parquet before fitting it. Blocks 0-1 have no fine-tuned model (0.5, routed to v2.0).

  FULL                    22 + all 3,072 dims
  ADDED_ONLY              the 3,072 dims alone
  F_C2COV F_C2AR F_C2FTCOV F_C2FTAR   22 + one 768-dim state
  F_ZS, F_FT              22 + both zero-shot / both fine-tuned states (each is also FULL minus the other)
  FULL_H                  FULL + tenure, against P_HIST
  P_WARM                  B0 + binned share of the four states observed (causal, per block)
  P_NM                    B0 + "this coin has no zero-shot state on any row of the file" (LOOK-AHEAD)
  NO_MOM NO_REV NO_VOLU NO_VOLA NO_PRICE NO_SIZE   v2.0 minus one shipped group, the scale of a shipped family
  PC_*                    E143 / E144 arms reused: the same states as 16 PCs, so F_x vs PC_x is what keeping all
                          768 dims buys over 16 PCs
  tuneFULL                equal-budget retune vs tuned B0 (mlp alpha capped at 1; mlp tuned B0 reused from E144)
Routing covers a coin when its arm's state is observed that date (fine-tuned arms: the fine-tuned state).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e133_histfe_factorial as H  # noqa: E402
import e135_route_histfe as Q  # noqa: E402
import e136_histfe_evaluation as N  # noqa: E402
import e138_carver_macro_evaluation as T  # noqa: E402
import e139_carver_macro_infl_evaluation as P  # noqa: E402

os.chdir(REPO)
from src.data_mani.history_features import bin_ties_preserved  # noqa: E402

ROOT = "results/reports/crypto/e146_c2dims"
DIR = "data/crypto/v2.0-c2syn-c2ft-dims"
DATA, BLOCKS = os.path.join(DIR, "train.parquet"), os.path.join(DIR, "blocks.parquet")
FAMS = ("c2cov", "c2ar", "c2ftcov", "c2ftar")
SHIPPED = {"MOM": ("momentum_", "sharpe_ratio_"), "REV": ("relative_strength_index_", "bollinger_"), "VOLU": ("volume_",),
           "VOLA": ("volatility_",), "PRICE": ("close_",), "SIZE": ("market_cap_",)}
E143, E144 = "results/reports/crypto/e143_c2syn/preds", "results/reports/crypto/e144_c2ft/preds"
P.STUDY, P.ROOT, P.DATA, P.META = "E146", ROOT, DATA, os.path.join(DIR, "historical_meta_models.parquet")
P.REUSE = {"PC_ZS": (E143, "FULL"), "PC_C2COV": (E143, "F_COV"), "PC_C2AR": (E143, "F_AR"),
           "PC_FT": (E144, "FULL"), "PC_C2FTCOV": (E144, "F_COV"), "PC_C2FTAR": (E144, "F_AR"),
           "tuneB0": (E144, "tuneB0")}                       # only read for mlp: ridge/lgb tuned B0 come from E133/E135
WIDE = ["FULL", "ADDED_ONLY", "F_C2COV", "F_C2AR", "F_C2FTCOV", "F_C2FTAR", "F_ZS", "F_FT", "FULL_H"]
P.FITS = WIDE + ["P_WARM", "P_NM"] + ["NO_" + g for g in SHIPPED]
P.SCORED = P.FITS + [a for a in P.REUSE if a != "tuneB0"]
P.ROUTED = ["FULL", "F_ZS", "F_FT", "F_C2COV", "F_C2AR", "F_C2FTCOV", "F_C2FTAR", "FULL_H"]
FT_ARMS = {"F_FT", "F_C2FTCOV", "F_C2FTAR"}
P.COMPARE = [(a, "B0", "against v2.0") for a in ["P_HIST"] + P.SCORED if a != "FULL_H"] + \
            [("FULL", "F_ZS", "fine-tuned dims, given zero-shot"), ("FULL", "F_FT", "zero-shot dims, given fine-tuned"),
             ("F_FT", "F_ZS", "fine-tuned against zero-shot"),
             ("F_C2COV", "F_C2AR", "zero-shot: covariates vs label only"), ("F_C2FTCOV", "F_C2FTAR", "fine-tuned: covariates vs label only"),
             ("F_ZS", "PC_ZS", "768 dims vs 16 PCs, zero-shot"), ("F_FT", "PC_FT", "768 dims vs 16 PCs, fine-tuned"),
             ("F_C2COV", "PC_C2COV", "dims vs PCs"), ("F_C2AR", "PC_C2AR", "dims vs PCs"),
             ("F_C2FTCOV", "PC_C2FTCOV", "dims vs PCs"), ("F_C2FTAR", "PC_C2FTAR", "dims vs PCs"),
             ("FULL_H", "P_HIST", "block beyond tenure"), ("FULL", "P_WARM", "block beyond warm-up"),
             ("tuneFULL", "tuneB0", "equal-budget retune")]
P.PAIRS = [("FULL", "B0"), ("F_ZS", "B0"), ("F_FT", "B0"), ("ADDED_ONLY", "B0"), ("F_FT", "F_ZS"), ("F_ZS", "PC_ZS"), ("F_FT", "PC_FT"),
           ("P_HIST", "B0"), ("P_WARM", "B0"), ("P_NM", "B0"), ("FULL_H", "P_HIST")]


def split(cols):
    fam = {f: [c for c in cols if c.startswith("feature_%s_d" % f)] for f in FAMS}
    base = [c for c in cols if c.startswith("feature_") and not any(c in v for v in fam.values())]
    assert len(base) == 22 and all(len(v) == 768 for v in fam.values()), (len(base), {f: len(v) for f, v in fam.items()})
    return base, fam


def load():
    raw = pd.read_parquet(DATA).reset_index()
    raw["date"] = pd.to_datetime(raw["date"])
    df = raw[raw["date"] <= Q.LAST].sort_values(["date", "symbol"]).reset_index(drop=True)
    base, fam = split(list(df))
    blk = [c for f in FAMS for c in fam[f]]
    for f in FAMS:                                            # a state is all 768 dims or none
        assert (df[fam[f][0]].notna() == df[fam[f][-1]].notna()).all(), f
    seen = {f: df[fam[f][0]].notna() for f in FAMS}
    tenure = df.groupby("ucid").cumcount().astype("float64").to_frame("feature_hist")
    df["feature_hist"] = bin_ties_preserved(tenure, df["date"])["feature_hist"]
    share = pd.concat(seen, axis=1).mean(axis=1).astype("float64").to_frame("feature_warm")
    df["feature_warm"] = bin_ties_preserved(share, df["date"])["feature_warm"]
    ever = raw[fam["c2cov"][0]].notna().groupby([raw["symbol"], raw["ucid"]]).any().rename("ever").reset_index()
    df = df.merge(ever, on=["symbol", "ucid"], how="left", validate="m:1")
    df["feature_nofeed"] = (~df.pop("ever")).astype("float32")
    del raw
    obs = df[["symbol", "date"]].copy()
    obs["obs_tf"] = seen["c2cov"].astype("float32").values        # zero-shot coverage, under T's column names
    obs["obs_macro"] = seen["c2ftcov"].astype("float32").values   # fine-tuned coverage (deployable view: the test-role block)
    obs["no_feed"] = df["feature_nofeed"]
    df[base + ["feature_hist", "feature_warm"]] = df[base + ["feature_hist", "feature_warm"]].fillna(0.5)
    df[blk] = df[blk].fillna(0.5)
    meta = pd.read_parquet(P.META)[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    return df.merge(meta, on=["symbol", "date"], how="left"), base, blk, [], obs


def arms(base, blk, _):
    fam = {f: [c for c in blk if c.startswith("feature_%s_d" % f)] for f in FAMS}
    A = {"B0": base, "B0CHK": base, "FULL": base + blk, "ADDED_ONLY": blk, "FULL_H": base + blk + ["feature_hist"],
         "F_ZS": base + fam["c2cov"] + fam["c2ar"], "F_FT": base + fam["c2ftcov"] + fam["c2ftar"],
         "P_WARM": base + ["feature_warm"], "P_NM": base + ["feature_nofeed"]}
    for f in FAMS:
        A["F_" + f.upper()] = base + fam[f]
    for g, keys in SHIPPED.items():
        A["NO_" + g] = [c for c in base if not c[len("feature_"):].startswith(keys)]
    assert [len(A[k]) for k in ("FULL", "ADDED_ONLY", "F_ZS", "F_FT", "F_C2COV", "F_C2FTAR")] == [3094, 3072, 1558, 1558, 790, 790]
    assert sorted(22 - len(A["NO_" + g]) for g in SHIPPED) == [2, 4, 4, 4, 4, 4]
    return A


def coverage(arm, o):
    return o["obs_macro"] if arm in FT_ARMS else o["obs_tf"]


def fit(args):
    df, base, blk, _, obs = load()
    A = arms(base, blk, None)
    past = np.array(sorted(df.loc[df["date"] < H.TEST_START, "date"].unique()))
    last = pd.read_parquet(Q.KEYS, columns=["date", "kind"])
    last = pd.to_datetime(last.loc[last["kind"] == "test", "date"]).unique()
    B = H.blocks(past) + [(11, past[-(H.PURGE + H.TRAIN):-H.PURGE], last)]
    ft = [c for c in blk if c.startswith("feature_c2ft")]
    zs_seen = obs.set_index(["symbol", "date"])["obs_tf"]
    L = pd.read_parquet(BLOCKS, columns=["block", "symbol", "date"] + ft)
    L["date"] = pd.to_datetime(L["date"])
    keep = ["symbol", "date", H.T20, "meta_model"]
    if args.tune:
        names, seeds = ["FULL"], [0]
    else:
        names = args.arms or (P.FITS + (["B0CHK"] if args.model == "ridge" else []))
        seeds = [0] if args.model == "ridge" else T.SEEDS[args.model]
    tag = "tune" if args.tune else ""
    todo = [(a, s) for a in names for s in seeds if not os.path.exists(os.path.join(ROOT, "preds", args.model, "%s%s_s%d.parquet" % (tag, a, s)))]
    print("%d prediction files to write" % len(todo), flush=True)
    F = {}
    for i, tr, te in B:                                       # each block on its own fine-tuned dims
        rows = df["date"].isin(tr) | df["date"].isin(te)
        d = df.loc[rows].drop(columns=ft + ["feature_warm"])
        own = L.loc[L["block"] == i].drop(columns="block")
        d = d.merge(own, on=["symbol", "date"], how="left", validate="1:1")
        assert len(d) == rows.sum()
        zs = zs_seen.reindex(pd.MultiIndex.from_frame(d[["symbol", "date"]])).to_numpy()
        fs = d[ft[0]].notna().to_numpy(np.float32)
        share = pd.DataFrame({"feature_warm": (zs + fs) / 2.0}, index=d.index)
        d["feature_warm"] = bin_ties_preserved(share, d["date"])["feature_warm"].fillna(0.5)
        print("block %d: %d rows, own fine-tuned state on %.1f%% of train / %.1f%% of test rows" % (
            i, len(d), 100 * fs[d["date"].isin(tr).to_numpy()].mean(), 100 * fs[d["date"].isin(te).to_numpy()].mean()), flush=True)
        d[ft] = d[ft].fillna(0.5)
        F[i] = d.reset_index(drop=True)
    del L, df
    for arm, s in todo:
        cols = A[arm]
        path = os.path.join(ROOT, "preds", args.model, "%s%s_s%d.parquet" % (tag, arm, s))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        parts, picks, t1 = [], [], time.time()
        for i, tr, te in B:
            d = F[i]
            trm, tem = d["date"].isin(tr), d["date"].isin(te)
            params = H.pick(args.model, d, cols, tr, s, args.n_jobs) if args.tune else H.FIXED[args.model]
            picks.append({k: str(params.get(k)) for k in H.GRID[args.model]})
            pred = H.fit_predict(args.model, params, s, args.n_jobs, d.loc[trm, cols].to_numpy(np.float32),
                                 d.loc[trm, H.T20].to_numpy(), d.loc[tem, cols].to_numpy(np.float32))
            o = d.loc[tem, keep].copy()
            o["prediction"], o["block"] = pred, i
            parts.append(o)
        tmp = path + ".partial"
        pd.concat(parts).to_parquet(tmp, index=False)
        os.replace(tmp, path)
        if args.tune:
            json.dump(picks, open(path + ".picks.json", "w"))
        print("%s %-12s %s seed %d  %d cols  %.0fs" % (args.model, arm, tag, s, len(cols), time.time() - t1), flush=True)


def audit():
    """Point-in-time (the raw states the dims are binned from), binning identity, coverage, mass, redundancy, IC."""
    from src.data_mani import build_c2ft, build_c2syn
    out = dict(truncation_zero_shot=build_c2syn.audit(), **{k + "_fine_tuned": v for k, v in build_c2ft.audit().items()})
    ds = pd.read_parquet(DATA).reset_index()
    ds["date"] = pd.to_datetime(ds["date"])
    base, fam = split(list(ds))
    # identity: re-bin the raw states and compare with the dataset
    skeys, sout = build_c2syn.states()
    skeys["date"] = pd.to_datetime(skeys["date"])
    ident = []
    for t in ("2021-06-30", "2023-06-30", "2025-12-31", "2026-08-20"):
        t = pd.Timestamp(t)
        k = skeys.reset_index(drop=True)
        sel = np.flatnonzero((k["date"] == t).to_numpy())
        day = ds.loc[ds["date"] == t].set_index("ucid")
        for m in ("cov", "ar"):
            x = pd.DataFrame(sout[m][sel])
            b = bin_ties_preserved(x, pd.Series([t] * len(sel))).to_numpy("float32")
            have = ~np.isnan(b[:, 0])
            u = k.iloc[sel]["ucid"].to_numpy()
            ref = day.reindex(u)[fam["c2" + m]].to_numpy("float32")
            eq = (b == ref) | (np.isnan(b) & np.isnan(ref))
            ident.append(dict(family="c2" + m, date=str(t.date()), states=int(have.sum()), cells=int(eq.size), differ=int((~eq).sum())))
            print(ident[-1], flush=True)
    del sout
    fkeys = build_c2ft.keys()
    fkeys["date"] = pd.to_datetime(fkeys["date"])
    L = pd.read_parquet(BLOCKS, columns=["block", "role", "ucid", "date"] + fam["c2ftcov"] + fam["c2ftar"])
    L["date"] = pd.to_datetime(L["date"])
    for b in (3, 7, 11):
        dates = np.sort(L.loc[(L["block"] == b) & (L["role"] == "test"), "date"].unique())
        t = pd.Timestamp(dates[len(dates) // 2])
        g = L[(L["block"] == b) & (L["date"] == t)]
        pos = g[["ucid", "date"]].merge(fkeys.reset_index(drop=True).reset_index(), on=["ucid", "date"], how="left")["index"].to_numpy()
        for m in ("cov", "ar"):
            ids, x = build_c2ft.block_states(b, m)
            full = np.full((len(fkeys), x.shape[1]), np.nan, "float32")
            full[ids] = x.astype("float16").astype("float32")        # the dataset was binned from float16 states
            v = bin_ties_preserved(pd.DataFrame(full[pos]), pd.Series([t] * len(pos))).to_numpy("float32")
            ref = g[fam["c2ft" + m]].to_numpy("float32")
            eq = (v == ref) | (np.isnan(v) & np.isnan(ref))
            ident.append(dict(family="c2ft" + m, block=b, date=str(t.date()), states=int((~np.isnan(v[:, 0])).sum()), cells=int(eq.size), differ=int((~eq).sum())))
            print(ident[-1], flush=True)
    out["identity"] = ident
    k = N.keys()
    o = pd.read_parquet(os.path.join(ROOT, "observed.parquet"))
    z = k.merge(o, on=["symbol", "date"], validate="1:1")
    out["coverage"] = dict(keys=len(z), zero_shot=float(z["obs_tf"].mean()), fine_tuned=float(z["obs_macro"].mean()), no_feed=float(z["no_feed"].mean()))
    out["coverage_by_year"] = z.groupby([z["date"].dt.year, "kind"]).agg(zero_shot=("obs_tf", "mean"), fine_tuned=("obs_macro", "mean")).reset_index().to_dict("records")
    q = ds.merge(k[["symbol", "date"]], on=["symbol", "date"])
    rng = np.random.default_rng(146)
    samp = {f: sorted(rng.choice(fam[f], 48, replace=False)) for f in FAMS}
    out["sample_dims_per_family"] = 48
    out["mass"] = {f: {str(a): float(b) for a, b in q[samp[f]].astype("float32").stack().value_counts(normalize=True).sort_index().items()} for f in FAMS}
    red, ic = {}, {}
    for f in FAMS:
        X = q[samp[f] + base + ["target_binned_return_20"]].astype("float32")
        C = X[samp[f]].corr(method="spearman").abs().to_numpy()
        np.fill_diagonal(C, np.nan)
        S = X[samp[f] + base].corr(method="spearman").loc[samp[f], base].abs()
        red[f] = dict(mean_abs_rho_within=float(np.nanmean(C)), pairs_over_08=int((C > .8).sum() // 2), max_rho_with_base=float(S.max().max()),
                      mean_max_rho_with_base=float(S.max(axis=1).mean()))
        Y = X.dropna(subset=["target_binned_return_20"])
        d = Y.groupby(q.loc[Y.index, "date"])[samp[f] + ["target_binned_return_20"]].corr(method="spearman").xs("target_binned_return_20", level=1)[samp[f]]
        t = d.mean() / d.std() * np.sqrt(len(d) / 20)
        ic[f] = dict(mean_abs_ic=float(d.mean().abs().mean()), max_abs_ic=float(d.mean().abs().max()), dims_t_over_2=int((t.abs() >= 2).sum()))
    out["redundancy"], out["date_ic"] = red, ic
    out["build"] = json.load(open(os.path.join(DIR, "build.json")))
    json.dump(out, open(os.path.join(ROOT, "audit.json"), "w"), indent=1, default=str)
    print(json.dumps({x: out[x] for x in ("coverage", "redundancy", "date_ic")}, default=str))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--model", choices=T.MODELS)
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--n-jobs", type=int, default=6)
    a = ap.parse_args()
    P.patch()
    os.makedirs(ROOT, exist_ok=True)
    if a.fit:
        fit(a)
        return
    obs_path = os.path.join(ROOT, "observed.parquet")
    if not os.path.exists(obs_path):
        k = pd.read_parquet(Q.KEYS, columns=["symbol", "date"])
        k["date"] = pd.to_datetime(k["date"])
        k.merge(load()[4], on=["symbol", "date"], how="left", validate="1:1").to_parquet(obs_path, index=False)
    if a.audit:
        audit()
        return
    P.score(int(os.environ.get("E146_WORKERS", "16")))


P.load, P.arms, P.coverage = load, arms, coverage
if __name__ == "__main__":
    main()
