"""
crypto/v2.0-c2syn-c2ft: v2.0 + both Chronos-2-Synth hidden-state blocks in one file.

    python -m src.data_mani.build_c2syn_c2ft            # PCs only (86 features)
    python -m src.data_mani.build_c2syn_c2ft --raw      # then + all 768 dims of all four states, binned (3,158 features)
    python -m src.data_mani.build_c2syn_c2ft --no-pc    # then crypto/v2.0-c2syn-c2ft-dims: the same without the 64 PCs (3,094);
                                                        # E146 ran on it, deleted 2026-11-11, rebuild to reproduce

  feature_c2cov_pc01..16, feature_c2ar_pc01..16       E143, zero-shot (crypto/v2.0-c2syn)
  feature_c2ftcov_pc01..16, feature_c2ftar_pc01..16   E144, walk-forward LoRA fine-tuned (crypto/v2.0-c2ft)

86 features = 22 shipped + 32 + 32. Both parents carry the same 523,396 keys and identical shipped
features and targets (asserted cell for cell), so this is a column join, nothing recomputed.

The fine-tuned columns DIFFER BY BLOCK (each walk-forward block has its own models and PCA basis).
train.parquet carries, like v2.0-c2ft, each row's TEST-role values (the block that predicts it):
the deployable view. A walk-forward evaluation must fit block b on b's own values, which
blocks.parquet (copied from v2.0-c2ft) holds for every block's train and test rows. Blocks 0-1
have no fine-tuned model, so those columns are NaN there; the zero-shot columns are NaN on warm-up
and E143's 60 burn-in dates. NaN stays NaN (the loader fills 0.5).

--raw adds every dimension of the 768-d encoder states the PCs were taken from, 4 x 768 = 3,072 columns:

  feature_c2cov_d000..767, feature_c2ar_d000..767       zero-shot (E143, /scratch/danielco/c2syn)
  feature_c2ftcov_d000..767, feature_c2ftar_d000..767   fine-tuned (E144, /scratch/danielco/c2ft)

each rank-binned onto Numerai's {0, .25, .5, .75, 1} with the exact shipped rule
(history_features.bin_ties_preserved, 5/20/50/20/5): zero-shot dims within each date, fine-tuned
dims within each (block, date) -- every block has its own model, so its axes are its own. NaN
stays NaN (0.5 at load, now a true neutral bin). Stored as float16, which holds the five values
exactly. The unbinned states remain in /scratch/danielco/c2syn and c2ft. The zero-shot raw state
exists on E143's 60 burn-in dates too: the burn-in only blanked the PCs, whose basis saw those dates;
the states themselves are point-in-time. The fine-tuned raw columns follow the PCs' layout: each
row's test-role state in train.parquet, every block's own states in blocks.parquet.

The files are too large for the shared /home quota (which the live round also writes to), so the
dataset lives on /scratch and data/crypto/v2.0-c2syn-c2ft is a symlink to it.
"""

import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

SYN = os.path.join("data", "crypto", "v2.0-c2syn")
FT = os.path.join("data", "crypto", "v2.0-c2ft")
DEST = os.path.join("data", "crypto", "v2.0-c2syn-c2ft")
STORE = "/scratch/danielco/datasets/v2.0-c2syn-c2ft"     # where DEST points (the /home quota cannot hold the raw states)
RAW_SYN, RAW_FT = "/scratch/danielco/c2syn", "/scratch/danielco/c2ft"
DIM = 768
KEY = ["symbol", "ucid", "date"]


def same(a, b):
    a, b = a.to_numpy(), b.to_numpy()
    return int((~((a == b) | (pd.isna(a) & pd.isna(b)))).sum())


def build():
    a = pd.read_parquet(os.path.join(SYN, "train.parquet")).reset_index()
    b = pd.read_parquet(os.path.join(FT, "train.parquet")).reset_index()
    syn = [c for c in a if c.startswith("feature_c2") and not c.startswith("feature_c2ft")]
    ft = [c for c in b if c.startswith("feature_c2ft")]
    shared = [c for c in a if c not in syn and c not in KEY]
    assert len(syn) == 32 and len(ft) == 32 and sum(c.startswith("feature_") for c in shared) == 22
    m = a.merge(b[KEY + shared + ft], on=KEY, how="outer", suffixes=("", "_ft"), validate="1:1", indicator=True)
    assert (m["_merge"] == "both").all() and len(m) == len(a) == len(b)
    diff = {c: same(m[c], m[c + "_ft"]) for c in shared}
    assert not any(diff.values()), diff
    out = m[KEY + shared + syn + ft]
    os.makedirs(DEST, exist_ok=True)
    out.set_index("symbol").to_parquet(os.path.join(DEST, "train.parquet"))
    shutil.copy(os.path.join(FT, "blocks.parquet"), os.path.join(DEST, "blocks.parquet"))
    shutil.copy(os.path.join(SYN, "historical_meta_models.parquet"), os.path.join(DEST, "historical_meta_models.parquet"))
    info = dict(rows=len(out), features=sum(c.startswith("feature_") for c in out), shipped=22, c2syn=len(syn), c2ft=len(ft),
                observed=dict(c2syn=float(out[syn[0]].notna().mean()), c2ft=float(out[ft[0]].notna().mean()),
                              both=float((out[syn[0]].notna() & out[ft[0]].notna()).mean()),
                              either=float((out[syn[0]].notna() | out[ft[0]].notna()).mean())),
                parents=dict(c2syn=SYN, c2ft=FT), shared_cells_checked=int(len(m) * len(shared)))
    json.dump(info, open(os.path.join(DEST, "build.json"), "w"), indent=1)
    print("wrote %s: %d rows, %d features (22 shipped + %d c2syn + %d c2ft); observed c2syn %.1f%%, c2ft %.1f%%, both %.1f%%"
          % (DEST, info["rows"], info["features"], len(syn), len(ft), 100 * info["observed"]["c2syn"], 100 * info["observed"]["c2ft"],
             100 * info["observed"]["both"]))


def raw_names(fam, mode):
    return ["feature_%s%s_d%03d" % (fam, mode, j) for j in range(DIM)]


def index_on(keys, src):
    """Positions of `src` (ucid, date) rows in `keys`; -1 where absent."""
    k = keys[["ucid", "date"]].reset_index(drop=True).reset_index().rename(columns={"index": "i"})
    return src[["ucid", "date"]].merge(k, on=["ucid", "date"], how="left")["i"].fillna(-1).astype(int).to_numpy()


def binned(x, groups, chunk=64):
    """Rank-bin every column of (n, DIM) within `groups` with Numerai's rule; float16 (the five values are exact)."""
    from src.data_mani.history_features import bin_ties_preserved
    out = np.full(x.shape, np.nan, "float16")
    g = pd.Series(groups)
    for j in range(0, x.shape[1], chunk):
        f = pd.DataFrame(x[:, j:j + chunk].astype("float32"))
        out[:, j:j + chunk] = bin_ties_preserved(f, g).to_numpy("float16")
    return out


def build_raw():
    from src.data_mani import build_c2ft, build_c2syn
    t = pd.read_parquet(os.path.join(DEST, "train.parquet")).reset_index()
    t["date"] = pd.to_datetime(t["date"])
    assert not any(c.endswith("_d000") for c in t), "raw columns already present"
    n = len(t)
    new = {}
    # zero-shot: one state per panel row and mode
    skeys, sout = build_c2syn.states(RAW_SYN)
    skeys["date"] = pd.to_datetime(skeys["date"])
    pos = index_on(t, skeys)
    assert (pos >= 0).all()
    for m in ("cov", "ar"):
        x = np.full((n, DIM), np.nan, "float16")
        x[pos] = sout[m].astype("float16")
        new[("c2", m)] = binned(x, t["date"].to_numpy())
        print("c2 %s: state on %.1f%% of rows" % (m, 100 * np.isfinite(x[:, 0]).mean()), flush=True)
    del sout
    # fine-tuned: per block; test role -> train.parquet, every role -> blocks.parquet
    fkeys = build_c2ft.keys(RAW_FT)
    grid = np.sort(fkeys["date"].unique())
    L = pd.read_parquet(os.path.join(DEST, "blocks.parquet"), columns=["block", "role", "ucid", "date"])
    L["date"] = pd.to_datetime(L["date"])
    lraw = {m: np.full((len(L), DIM), np.nan, "float16") for m in ("cov", "ar")}
    tpos = index_on(t, fkeys)
    for b, tr, te in build_c2ft.blocks(grid):
        if not os.path.exists(os.path.join(RAW_FT, "block_%02d" % b, "ar_reps.npy")):
            continue
        sel = np.flatnonzero((L["block"] == b).to_numpy())
        lk = index_on(fkeys, L.iloc[sel])                    # blocks.parquet rows -> c2ft panel row ids
        assert (lk >= 0).all()
        test = np.isin(fkeys["date"].to_numpy(), np.asarray(te, "datetime64[ns]"))
        for m in ("cov", "ar"):
            ids, x = build_c2ft.block_states(b, m, RAW_FT)
            full = np.full((len(fkeys), DIM), np.nan, "float16")
            full[ids] = x.astype("float16")
            lraw[m][sel] = full[lk]
            if ("c2ft", m) not in new:
                new[("c2ft", m)] = np.full((n, DIM), np.nan, "float16")
            r = np.flatnonzero(test & np.isfinite(full[:, 0]))
            new[("c2ft", m)][tpos[r]] = full[r]
        print("block %d: raw fine-tuned states placed" % b, flush=True)
    groups = L["block"].astype(str).to_numpy() + "|" + L["date"].astype(str).to_numpy()
    test_rows = np.flatnonzero((L["role"] == "test").to_numpy())
    tl = index_on(t, L.iloc[test_rows])
    assert (tl >= 0).all()
    for m in ("cov", "ar"):
        lraw[m] = binned(lraw[m], groups)
        x = np.full((n, DIM), np.nan, "float16")
        x[tl] = lraw[m][test_rows]                           # train.parquet = each row's test-role bins, identical to blocks.parquet
        assert np.array_equal(np.isfinite(x[:, 0]), np.isfinite(new[("c2ft", m)][:, 0]))
        new[("c2ft", m)] = x
    for m in ("cov", "ar"):
        print("c2ft %s: test-role state on %.1f%% of rows" % (m, 100 * np.isfinite(new[("c2ft", m)][:, 0]).mean()), flush=True)
    # write to the store, then point DEST at it
    os.makedirs(STORE + ".tmp", exist_ok=True)
    raw = pd.DataFrame({c: v for (f, m), x in new.items() for c, v in zip(raw_names(f, m), x.T)}, index=t.index)
    pd.concat([t, raw], axis=1).set_index("symbol").to_parquet(os.path.join(STORE + ".tmp", "train.parquet"))
    del raw
    Lfull = pd.read_parquet(os.path.join(DEST, "blocks.parquet"))
    lr = pd.DataFrame({c: v for m in ("cov", "ar") for c, v in zip(raw_names("c2ft", m), lraw[m].T)}, index=Lfull.index)
    pd.concat([Lfull, lr], axis=1).to_parquet(os.path.join(STORE + ".tmp", "blocks.parquet"), index=False)
    shutil.copy(os.path.join(DEST, "historical_meta_models.parquet"), os.path.join(STORE + ".tmp", "historical_meta_models.parquet"))
    info = json.load(open(os.path.join(DEST, "build.json")))
    info.update(raw=dict(columns=4 * DIM, dtype="float16", scale="Numerai bins (bin_ties_preserved): c2 within date, c2ft within block x date",
                         bin_mass={"%s%s" % k: {str(a): round(float(b), 4) for a, b in pd.Series(v[:, :8].ravel()).dropna().value_counts(normalize=True).sort_index().items()} for k, v in new.items()},
                         observed={"%s%s" % k: float(np.isfinite(v[:, 0]).mean()) for k, v in new.items()},
                         sources=dict(c2syn=RAW_SYN, c2ft=RAW_FT)),
                features=info["features"] + 4 * DIM, store=STORE)
    json.dump(info, open(os.path.join(STORE + ".tmp", "build.json"), "w"), indent=1)
    if os.path.exists(STORE):
        shutil.rmtree(STORE)
    os.replace(STORE + ".tmp", STORE)
    if not os.path.islink(DEST):
        shutil.rmtree(DEST)
        os.symlink(STORE, DEST)
    print("wrote %s (-> %s): %d rows, %d features; train.parquet %.2f GB, blocks.parquet %.2f GB" % (
        DEST, STORE, n, info["features"], os.path.getsize(os.path.join(STORE, "train.parquet")) / 1e9,
        os.path.getsize(os.path.join(STORE, "blocks.parquet")) / 1e9))


NOPC_DEST = os.path.join("data", "crypto", "v2.0-c2syn-c2ft-dims")
NOPC_STORE = "/scratch/danielco/datasets/v2.0-c2syn-c2ft-dims"


def build_no_pc():
    """crypto/v2.0-c2syn-c2ft-dims: v2.0-c2syn-c2ft minus every _pcNN column, everything else byte for byte."""
    import pyarrow.parquet as pq
    os.makedirs(NOPC_STORE + ".tmp", exist_ok=True)
    out = {}
    for f in ("train.parquet", "blocks.parquet"):
        names = pq.ParquetFile(os.path.join(DEST, f)).schema_arrow.names
        assert any(c.endswith("_d000") for c in names), "run --raw first"
        keep = [c for c in names if "_pc" not in c]
        pq.write_table(pq.read_table(os.path.join(DEST, f), columns=keep), os.path.join(NOPC_STORE + ".tmp", f))
        out[f] = dict(columns=len(keep), dropped=len(names) - len(keep), features=sum(c.startswith("feature_") for c in keep))
    shutil.copy(os.path.join(DEST, "historical_meta_models.parquet"), os.path.join(NOPC_STORE + ".tmp", "historical_meta_models.parquet"))
    info = json.load(open(os.path.join(DEST, "build.json")))
    info.update(parent=DEST, files=out, features=out["train.parquet"]["features"], store=NOPC_STORE, c2syn=0, c2ft=0,
                note="v2.0-c2syn-c2ft without the 64 PCs: 22 shipped + 3,072 binned hidden dims")
    json.dump(info, open(os.path.join(NOPC_STORE + ".tmp", "build.json"), "w"), indent=1)
    if os.path.exists(NOPC_STORE):
        shutil.rmtree(NOPC_STORE)
    os.replace(NOPC_STORE + ".tmp", NOPC_STORE)
    if not os.path.islink(NOPC_DEST):
        os.symlink(NOPC_STORE, NOPC_DEST)
    print("wrote %s (-> %s): %s" % (NOPC_DEST, NOPC_STORE, out))


if __name__ == "__main__":
    if "--no-pc" in sys.argv:
        build_no_pc()
        sys.exit()
    build_raw() if "--raw" in sys.argv else build()
