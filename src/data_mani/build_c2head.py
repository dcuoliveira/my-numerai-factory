"""
crypto/v2.0-c2head (+ -bin): v2.0 + Chronos-2-Synth forecast-head summaries (E145).

    python -m src.data_mani.build_c2head              # after chronos2_head has written zs and every ft block
    python -m src.data_mani.build_c2head --audit      # point-in-time and fp16-adapter checks (writes audit.json)

Reads the q10/q50/q90 20-step forecasts chronos2_head wrote to /scratch/danielco/c2head and turns
each branch into five summaries:

  median_mean       mean of the 20 q50
  median_terminal   q50 at step 20
  median_slope      OLS slope of q50 on the horizon 1..20
  uncertainty_mean  mean of q90 - q10
  asymmetry_mean    mean of (q90 - q50) - (q50 - q10)

Branches zscov, zsar (zero-shot) and ftcov, ftar (fine-tuned, per block), plus zsdelta = zscov - zsar
and ftdelta = ftcov - ftar on the RAW summaries. Then, within each date (and, for ft, within the
block that produced the values), a continuous percentile rank (rank - 0.5) / n over the coins that
have a forecast -- the primary dataset -- or Numerai's five-bin rule (bin_ties_preserved) for the
-bin robustness copy. NaN stays NaN (the loader fills 0.5); no missingness indicator is added.

Files per dataset:
  train.parquet   frozen v2.0 (data/crypto/v2.0-asof-e145) + zs columns + each row's TEST-role ft
                  columns (the block that predicts it) + ctrl_ctx_share: the share of the row's
                  256-step context holding a resolved label (causal; the P_OBS control, not a feature)
  blocks.parquet  (block, role, ucid, date, ft columns): what the evaluation fits each block on.
Rows are never dropped; blocks 0-1 have no ft model, so their ft columns are NaN.
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani import chronos2_syn_repr as Z
from src.data_mani.binning import BIN_MASS, bin_mass
from src.data_mani.history_features import bin_ties_preserved

RAW = "/scratch/danielco/c2head"
SOURCE = os.path.join("data", "crypto", "v2.0-asof-e145")
DEST = {"rank": os.path.join("data", "crypto", "v2.0-c2head"), "bin": os.path.join("data", "crypto", "v2.0-c2head-bin")}
SUMS = ("median_mean", "median_terminal", "median_slope", "uncertainty_mean", "asymmetry_mean")
TRAIN, PURGE, BLOCK, FIRST = 130, 22, 130, 172
TEST_START, LAST = pd.Timestamp("2026-02-20"), pd.Timestamp("2026-08-20")


def blocks(grid):
    """The E133 windows (identical to chronos2_head.blocks, which needs only numpy/pandas but lives in the GPU module)."""
    past = grid[grid < np.datetime64(TEST_START)]
    n = len(past)
    starts = list(range(FIRST, n - BLOCK + 1, BLOCK))
    assert len(starts) == 11 and starts[-1] + BLOCK == n, (starts, n)
    out = [(i, past[b - PURGE - TRAIN:b - PURGE], past[b:b + BLOCK]) for i, b in enumerate(starts)]
    test = grid[(grid >= np.datetime64(TEST_START)) & (grid <= np.datetime64(LAST))]
    return out + [(11, past[-(PURGE + TRAIN):-PURGE], test)]


def summarise(q):
    """(n, 3, 20) q10/q50/q90 -> (n, 5) in SUMS order."""
    q10, q50, q90 = q[:, 0].astype("float64"), q[:, 1].astype("float64"), q[:, 2].astype("float64")
    h = np.arange(1, q.shape[-1] + 1, dtype="float64")
    hc = h - h.mean()
    slope = ((q50 - q50.mean(1, keepdims=True)) * hc).sum(1) / (hc ** 2).sum()
    return np.stack([q50.mean(1), q50[:, -1], slope, (q90 - q10).mean(1), ((q90 - q50) - (q50 - q10)).mean(1)], 1)


def load(path):
    z = np.load(path)
    return z["rows"], summarise(z["q"])


def raw_frame(n, fam, parts):
    """{branch: (n, 5) float64 NaN where no forecast} from {branch: (row ids, summaries)}; adds the deltas."""
    out = {}
    for name, (ids, s) in parts.items():
        assert len(np.unique(ids)) == len(ids), "row twice: %s" % name
        x = np.full((n, len(SUMS)), np.nan)
        x[ids] = s
        out[fam + name] = x
    out[fam + "delta"] = out[fam + "cov"] - out[fam + "ar"]
    return out


def cols(branch):
    return ["feature_%s_%s" % (branch, s) for s in SUMS]


def scale(frame, dates, how):
    """Within-date continuous percentile rank, or Numerai's five bins. NaN stays NaN."""
    if how == "bin":
        return bin_ties_preserved(frame, dates)
    r = frame.groupby(dates.to_numpy()).rank(method="average")
    n = frame.notna().groupby(dates.to_numpy()).transform("sum")
    return (r - 0.5) / n


def context_share(df, grid):
    """Share of each row's 256-step context (steps D-255..D) holding a resolved label, labels after D-35 not counted."""
    walls = Z.wall_steps(grid)
    out = np.zeros(len(df))
    steps = df["step"].to_numpy()
    y = df[Z.TARGET].to_numpy()
    for _, idx in df.groupby("ucid").indices.items():
        have = np.zeros(len(grid) + 1)
        have[steps[idx] + 1] = np.isfinite(y[idx])
        cs = np.cumsum(have)
        s = steps[idx]
        w = walls[s]
        lo = np.maximum(0, s - Z.CTX + 1)
        out[idx] = np.where(w >= 0, (cs[np.maximum(w, -1) + 1] - cs[lo]) / Z.CTX, 0.0)
    return out


def build(raw=RAW):
    keys = pd.read_parquet(os.path.join(raw, "rows.parquet")).reset_index(drop=True)
    keys["date"] = pd.to_datetime(keys["date"])
    df, feats, grid = Z.load_panel(os.path.join(SOURCE, "train.parquet"))
    assert (df[["ucid", "date"]].to_numpy() == keys[["ucid", "date"]].to_numpy()).all(), "panel changed since the GPU run"
    n = len(keys)
    zs = raw_frame(n, "zs", {m: load(os.path.join(raw, "zs_%s.npz" % m)) for m in ("cov", "ar")})
    rep = {"zs_rows": int(np.isfinite(zs["zscov"][:, 0]).sum()), "rows": n}
    # fine-tuned: per block, its train and test rows
    longs, info = [], {}
    for b, tr, te in blocks(grid):
        d = os.path.join(raw, "block_%02d" % b)
        if not os.path.exists(os.path.join(d, "ft_ar.npz")):
            print("block %d: no ft forecasts%s" % (b, "" if b <= 1 else "  <-- MISSING"))
            continue
        ft = raw_frame(n, "ft", {m: load(os.path.join(d, "ft_%s.npz" % m)) for m in ("cov", "ar")})
        rows = np.flatnonzero(keys["date"].isin(pd.to_datetime(np.concatenate([np.asarray(tr, "datetime64[ns]"), np.asarray(te, "datetime64[ns]")]))).to_numpy())
        L = keys.loc[rows, ["ucid", "date"]].copy()
        L["block"] = b
        L["role"] = np.where(L["date"].isin(pd.to_datetime(np.asarray(te, "datetime64[ns]"))), "test", "train")
        for br in ("ftcov", "ftar", "ftdelta"):
            L[cols(br)] = ft[br][rows]
        longs.append(L)
        info[b] = {m: {k: v for k, v in json.load(open(os.path.join(d, "%s_manifest.json" % m))).items() if k != "val_log"} for m in ("cov", "ar")}
        print("block %d: %d rows, ft forecast on %.1f%%" % (b, len(L), 100 * np.isfinite(L[cols("ftcov")[0]]).mean()), flush=True)
    long_raw = pd.concat(longs, ignore_index=True)
    base = pd.read_parquet(os.path.join(SOURCE, "train.parquet")).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    ctx = keys[["ucid", "date"]].assign(ctrl_ctx_share=context_share(df, grid))
    for how, dest in DEST.items():
        zcols = {}
        for br in ("zscov", "zsar", "zsdelta"):
            f = scale(pd.DataFrame(zs[br], columns=cols(br)), keys["date"], how)
            zcols.update({c: f[c].to_numpy("float32") for c in f})
        L = long_raw.copy()
        ftc = [c for br in ("ftcov", "ftar", "ftdelta") for c in cols(br)]
        g = L["block"].astype(str) + "|" + L["date"].astype(str)
        L[ftc] = scale(L[ftc], g, how).astype("float32")
        test = L[L["role"] == "test"].drop(columns=["block", "role"])
        block = keys[["ucid", "date"]].assign(**zcols).merge(test, on=["ucid", "date"], how="left", validate="1:1").merge(ctx, on=["ucid", "date"], validate="1:1")
        merged = base.merge(block, on=["ucid", "date"], how="left", validate="1:1")
        new = [c for c in merged if c.startswith("feature_") and c.split("_")[1] in ("zscov", "zsar", "zsdelta", "ftcov", "ftar", "ftdelta")]
        assert len(new) == 30, len(new)
        os.makedirs(dest, exist_ok=True)
        merged.set_index("symbol").to_parquet(os.path.join(dest, "train.parquet"))
        L.to_parquet(os.path.join(dest, "blocks.parquet"), index=False)
        shutil.copy(os.path.join(SOURCE, "historical_meta_models.parquet"), os.path.join(dest, "historical_meta_models.parquet"))
        drift = max(float(np.nanmax(np.abs(bin_mass(merged[c]) - BIN_MASS))) for c in new) if how == "bin" else None
        rep[how] = dict(rows=len(merged), features=sum(c.startswith("feature_") for c in merged), zs_observed=float(merged[cols("zscov")[0]].notna().mean()),
                        ft_observed=float(merged[cols("ftcov")[0]].notna().mean()), bin_mass_drift=drift)
        print("%s -> %s: %d rows, %d features, zs on %.1f%%, ft on %.1f%% of rows%s" % (
            how, dest, len(merged), rep[how]["features"], 100 * rep[how]["zs_observed"], 100 * rep[how]["ft_observed"],
            "" if drift is None else ", max bin-mass drift %.4f" % drift), flush=True)
    rep["ft_blocks"] = info
    for dest in DEST.values():
        json.dump(rep, open(os.path.join(dest, "build.json"), "w"), indent=1, default=str)


def audit(raw=RAW):
    """
    Each audit date T, recomputed by chronos2_head audit from the panel CUT at T:
      zs   cut vs stored (same frozen model): raw forecast cells and the final rank / bin values;
      ft   cut vs uncut with the same reloaded fp16 adapter (the leakage test proper), and the
           reloaded fp16 adapter vs the stored in-memory model, after the final transformation.
    """
    keys = pd.read_parquet(os.path.join(raw, "rows.parquet")).reset_index(drop=True)
    keys["date"] = pd.to_datetime(keys["date"])
    out = []
    for p in sorted(glob.glob(os.path.join(raw, "audit", "zs_cov_*_cut.npz"))):
        T = os.path.basename(p).split("_")[2]
        day = keys.index[keys["date"] == pd.Timestamp(T)].to_numpy()
        for fam in ("zs", "ft"):
            if not os.path.exists(os.path.join(raw, "audit", "%s_cov_%s_cut.npz" % (fam, T))):
                continue
            sums, stored, full = {}, {}, {}
            for m in ("cov", "ar"):
                c = np.load(os.path.join(raw, "audit", "%s_%s_%s_cut.npz" % (fam, m, T)))
                b = int(c["block"])
                src = os.path.join(raw, "zs_%s.npz" % m) if fam == "zs" else os.path.join(raw, "block_%02d" % b, "ft_%s.npz" % m)
                z = np.load(src)
                pos = dict(zip(z["rows"], range(len(z["rows"]))))
                sel = [r for r in day if r in pos]
                st = pd.Series(list(z["q"][[pos[r] for r in sel]]), index=keys.loc[sel, "ucid"].to_numpy())
                cu = pd.Series(list(c["q"]), index=c["ucid"])
                assert set(cu.index) == set(st.index), (T, fam, m, len(cu), len(st))
                cu = np.stack(cu.loc[st.index].to_numpy())
                st = np.stack(st.to_numpy())
                rec = dict(date=T, family=fam, mode=m, block=b, rows_on_date=len(day), forecasts=len(st),
                           q_cells=int(st.size), q_differ_cut_vs_stored=int((cu != st).sum()), q_maxdiff_cut_vs_stored=float(np.abs(cu - st).max()))
                if fam == "ft":
                    f = np.load(os.path.join(raw, "audit", "ft_%s_%s_full.npz" % (m, T)))
                    fu = np.stack(pd.Series(list(f["q"]), index=f["ucid"]).loc[keys.loc[sel, "ucid"].to_numpy()].to_numpy())
                    rec.update(q_differ_cut_vs_full=int((cu != fu).sum()), q_maxdiff_cut_vs_full=float(np.abs(cu - fu).max()))
                    full[m] = summarise(fu)
                sums[m], stored[m] = summarise(cu), summarise(st)
                out.append(rec)
            # after the final transformation: rank / bin within the date, reloaded vs stored
            def fin(x, how):
                d = pd.Series(np.zeros(len(x)))
                return scale(pd.DataFrame(x), d, how).to_numpy()
            for how in ("rank", "bin"):
                a = np.hstack([sums["cov"], sums["ar"], sums["cov"] - sums["ar"]])
                s = np.hstack([stored["cov"], stored["ar"], stored["cov"] - stored["ar"]])
                A, S = fin(a, how), fin(s, how)
                out.append(dict(date=T, family=fam, mode="final_" + how, block=b, cells=int(A.size),
                                differ=int((~np.isclose(A, S, rtol=0, atol=1e-9)).sum()), max_abs_diff=float(np.abs(A - S).max())))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    if a.audit:
        for r in audit():
            print(r)
    else:
        build()


if __name__ == "__main__":
    main()
