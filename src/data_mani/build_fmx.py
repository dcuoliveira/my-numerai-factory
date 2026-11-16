"""
E118's builder: 34 representation blocks, merged onto the three baselines' datasets.

    python -m src.data_mani.build_fmx --base all

Reads every cache under data/crypto/fmrep_cache/ -- foundation_repr's E109 passes,
fmx_tabpfn's new TabPFN passes (x_*), fmx_models' source models (m_*) and the
TimesFM / Chronos-2 forecasts -- turns them into blocks `feature_xNN_*` (NN is the
row of the representation table; 21, distillation, is rows not columns), rank-bins
every column per date over the base panel, and writes

    ridge     crypto/v2.0-tenc-onehot-rollsel-spread-interact-fmx   (allfeat_pp; arms drop tenc)
    lgb       crypto/v2.0-tenc-onehot-fmx                           (tenc_onehot)
    catboost  crypto/v2.0-fenccausal-fmx                            (fenccausal)

One arm = the base plus ONE block, selected by
    --feature-exclude '^feature_x(?!NN_)\\d\\d_'
and build_fmx asserts each variant minus every block is its base, row for row, so
the reused baselines are the arms' true "without".

Causality is the caches' (see their docstrings). What this file adds is causal
too: the embedding and driver bases are fitted on the burn-in dates only (ending
before the first training date, checked), drift reads the previous date of the
same frozen block, rank uncertainty and the "gap" columns are within-date, and
binning is within-date.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.binning import bin_cross_sectional, bin_mass
from src.data_mani.build_fmrep import first_training_date, untie
from src.data_mani.foundation_repr import CACHE_DIR, PASSES
from src.data_mani.foundation_repr_ts import BURNIN, burnin_pca
from src.data_mani.utils import download_if_missing, get_api, local_path

BASES = {
    "ridge": "crypto/v2.0-tenc-onehot-rollsel-spread-interact",
    "lgb": "crypto/v2.0-tenc-onehot",
    "catboost": "crypto/v2.0-fenccausal",
}
N_PCS = 8
MC_DRAWS = 200


def cache(name, columns=None):
    return pd.read_parquet(os.path.join(CACHE_DIR, name + ".parquet"), columns=columns)


def block(df, nn, columns):
    """Rename `columns` of df to feature_xNN_<name>, keeping the keys."""
    keys = [c for c in ("symbol", "ucid", "date") if c in df.columns]
    return df[keys + list(columns)].rename(columns={c: "feature_x%02d_%s" % (nn, c) for c in columns})


def ranks(frame, column):
    return frame.groupby("date")[column].rank(pct=True)


def rank_uncertainty(main):
    """Monte Carlo ranks from the five bin probabilities, per date."""
    rng = np.random.default_rng(42)
    out = np.full((len(main), 3), np.nan)
    p = main[["tp_p%d" % k for k in range(5)]].to_numpy("float64")
    p = p / p.sum(axis=1, keepdims=True)
    cum = np.cumsum(p, axis=1)[:, :-1]
    for _, idx in main.groupby("date").indices.items():
        n = len(idx)
        u = rng.random((MC_DRAWS, n))
        bins = (u[:, :, None] > cum[idx][None]).sum(axis=2) + 0.1 * rng.random((MC_DRAWS, n))
        r = bins.argsort(axis=1).argsort(axis=1) / (n - 1)
        top, bottom = (r >= 0.95).mean(axis=0), (r <= 0.05).mean(axis=0)
        out[idx] = np.column_stack([r.mean(axis=0), r.std(axis=0), top - bottom])
    return out


def assemble():
    """Every block on (symbol, date) except x06, which is on (ucid, date)."""
    main = cache("main")
    dates = np.sort(main["date"].unique())
    burnin = dates[:BURNIN]
    out = main[["symbol", "date"]].copy()
    parts = []

    m = main.copy()
    m["tp_iqr"] = m["tp_q75"] - m["tp_q25"]
    m["tp_skew"] = (m["tp_q90"] + m["tp_q10"] - 2 * m["tp_q50"]) / (m["tp_q90"] - m["tp_q10"]).replace(0, np.nan)
    m["tp_disagree"] = m[[c for c in m.columns if c.startswith("tp_est")]].std(axis=1, ddof=0)
    emb = m[[c for c in m.columns if c.startswith("emb_")]].to_numpy("float64")
    for j, col in enumerate(burnin_pca(emb, m["date"].to_numpy(), burnin, N_PCS).T):
        m["emb_pc%d" % j] = col
    m[["rk_mean", "rk_sd", "rk_tail"]] = rank_uncertainty(m)
    prev = m.sort_values(["symbol", "date"]).groupby(["symbol", "block"])["tp_mean"].shift(1)
    m["drift"] = m["tp_mean"] - prev.reindex(m.index)
    parts += [block(m, 1, ["tp_mean", "tp_median", "tp_mode"]),
              block(m, 2, ["tp_std", "tp_iqr", "tp_skew"] + ["tp_p%d" % k for k in range(5)]),
              block(m, 3, ["tp_disagree"]),
              block(m, 4, ["emb_pc%d" % j for j in range(N_PCS)]),
              block(m, 8, ["rk_mean", "rk_sd", "rk_tail"]),
              block(m, 9, ["drift"])]

    heads = main[["symbol", "date"]].copy()
    for name in PASSES:
        if name != "main":
            heads = heads.merge(cache(name, ["symbol", "date", name]), on=["symbol", "date"], how="left")
    parts.append(block(heads, 5, [n for n in PASSES if n != "main"]))

    def x(name):
        return cache("x_" + name).drop(columns=["block"])
    a = x("attn")
    parts += [block(a, 7, ["att_label", "att_entropy", "att_max", "att_neff"]), block(a, 20, ["att_age"])]
    parts.append(block(x("nbr"), 10, ["nbr_label", "nbr_nearest", "nbr_proto"]))
    parts.append(block(x("resid"), 11, ["resid"]))
    for nn, name in ((12, "crowd"), (13, "t60"), (16, "self"), (17, "regime")):
        f = x(name).merge(main[["symbol", "date", "tp_mean"]], on=["symbol", "date"], how="left")
        f[name + "_gap"] = ranks(f, "tp_mean") - ranks(f, name)
        parts.append(block(f, nn, [name, name + "_gap"]))
    parts.append(block(x("extreme"), 14, ["extreme"]))
    nm = x("now_mom").merge(x("now_vol"), on=["symbol", "date"], how="outer")
    parts.append(block(nm, 15, ["now_mom", "now_vol"]))
    parts.append(block(x("pair"), 18, ["pair"]))
    d = x("drivers")
    delta = d[[c for c in d.columns if c.startswith("drv_")]].to_numpy("float64")
    d["drv_norm"] = np.sqrt((delta ** 2).sum(axis=1))
    d["drv_pc1"] = burnin_pca(delta, d["date"].to_numpy(), burnin, 1)[:, 0]
    parts.append(block(d, 19, ["drv_norm", "drv_pc1"]))

    rf, cb, lg, ml, bk = (cache("m_" + s) for s in ("rf", "catboost", "lgb", "mlp", "blocks"))
    parts += [block(rf, 22, ["pred"]), block(rf, 23, ["tree_std"]),
              block(rf, 24, ["leaf_pc%d" % j for j in range(N_PCS)]), block(rf, 25, ["leaf_support", "leaf_var"]),
              block(cb, 26, ["pred"]), block(cb, 27, ["ve_std"]),
              block(lg, 28, ["pred"]), block(lg, 29, ["early", "stage_sd"]), block(lg, 30, ["shap_norm", "shap_top"]),
              block(ml, 31, ["pred"]), block(ml, 32, ["hid_pc%d" % j for j in range(N_PCS)]), block(ml, 33, ["seed_std"])]
    bcols = [c for c in bk.columns if c.startswith("blk_")]
    parts.append(block(bk, 34, bcols))
    book = rf[["symbol", "date"]].copy()
    for tag, f in (("rf", rf), ("cb", cb), ("lgb", lg), ("mlp", ml)):
        book = book.merge(f[["symbol", "date", "pred"]].rename(columns={"pred": tag}), on=["symbol", "date"], how="outer")
    book = book.merge(bk.assign(blk=bk[bcols].mean(axis=1))[["symbol", "date", "blk"]], on=["symbol", "date"], how="outer")
    rk = pd.concat([ranks(book, c) for c in ("rf", "cb", "lgb", "mlp", "blk")], axis=1)
    book["member_sd"] = rk.std(axis=1)
    parts.append(block(book, 35, ["member_sd"]))

    for p in parts:
        out = out.merge(p, on=["symbol", "date"], how="outer")

    ts = None
    for model, tag in (("timesfm", "tsfm"), ("chronos2", "chr")):
        f = cache("ts_" + model)
        g = f[["ucid", "date"]].copy()
        g[tag + "_q50"] = f[tag + "_q50"]
        g[tag + "_spread"] = f[tag + "_q90"] - f[tag + "_q10"]
        g[tag + "_skew"] = (f[tag + "_q90"] + f[tag + "_q10"] - 2 * f[tag + "_q50"]) / (f[tag + "_q90"] - f[tag + "_q10"]).replace(0, np.nan)
        for j in range(N_PCS):
            if "%s_emb_pc%02d" % (tag, j) in f.columns:
                g["%s_emb_pc%d" % (tag, j)] = f["%s_emb_pc%02d" % (tag, j)]
        ts = g if ts is None else ts.merge(g, on=["ucid", "date"], how="outer")
    ts = block(ts, 6, [c for c in ts.columns if c not in ("ucid", "date")])
    return out, ts, burnin[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="all", help="ridge, lgb, catboost or all")
    args = parser.parse_args()
    bases = list(BASES) if args.base == "all" else args.base.split(",")
    blocks, ts, burnin_end = assemble()
    api = get_api("crypto")

    for base in bases:
        source = BASES[base]
        df = pd.read_parquet(download_if_missing(api, "crypto", source, "train.parquet")).reset_index()
        periods = np.sort(df["date"].unique())
        first_train = first_training_date(periods)
        if not burnin_end < first_train:
            raise SystemExit("burn-in ends %s, not before the first training date %s" % (burnin_end, first_train))
        merged = df.merge(blocks, on=["symbol", "date"], how="left").merge(ts, on=["ucid", "date"], how="left")
        cols = [c for c in merged.columns if c.startswith("feature_x") and c[9:11].isdigit()]
        merged, ties = untie(merged, cols, "feature_x01_tp_mean")
        merged = bin_cross_sectional(merged, cols, "date")
        pd.testing.assert_frame_equal(merged[df.columns], df)

        live = merged["date"] >= first_train
        print("%s: %s, %d dates; %d block columns; first training date %s, burn-in ends %s"
              % (base, source, len(periods), len(cols), pd.Timestamp(first_train).date(), pd.Timestamp(burnin_end).date()))
        for nn in sorted({c[9:11] for c in cols}):
            bc = [c for c in cols if c[9:11] == nn]
            cov = 100 * merged.loc[live, bc].notna().all(axis=1).mean()
            drift = max(np.abs(bin_mass(merged[c]) - [.05, .20, .50, .20, .05]).max() for c in bc)
            tied = [c[12:] for c in bc if c in ties]
            print("  x%s %2d cols  coverage %5.1f%%  mass drift %.4f%s" % (nn, len(bc), cov, drift,
                  "  untied: " + ", ".join(tied) if tied else ""))

        version = source + "-fmx"
        dest = local_path("crypto", version, "train.parquet")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        merged.set_index("symbol").to_parquet(dest)
        n_feat = sum(c.startswith("feature_") for c in merged.columns)
        print("  -> %s: %d features (%d base + %d block), %d rows, %.0f MB"
              % (dest, n_feat, n_feat - len(cols), len(cols), len(merged), os.path.getsize(dest) / 1e6))
        target = local_path("crypto", version, "historical_meta_models.parquet")
        if not os.path.exists(target):
            shutil.copy(download_if_missing(api, "crypto", source, "historical_meta_models.parquet"), target)


if __name__ == "__main__":
    main()
