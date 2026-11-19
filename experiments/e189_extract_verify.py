#!/usr/bin/env python3
"""
E189 validation steps 3-4: does the cached Chronos-2-Synth cov state reproduce from the frozen panel?

    CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/danielco/hf HF_HUB_DISABLE_XET=1 \
        /scratch/danielco/chronos2/venv/bin/python experiments/e189_extract_verify.py

Runs in the chronos2 venv (numpy/pandas/torch + chronos only). For each check date T:
  full  the extractor on the whole frozen panel (data/crypto/v2.0-asof-e189), date T's rows
  cut   the extractor on the panel cut at T with every label after T - 35 days blanked
        (chronos2_syn_repr --audit-cut semantics): what was knowable at T
and compares both with /scratch/danielco/c2syn (row ids index the panel sorted by date, ucid, which
the frozen panel shares with the cache's keys). Writes the states and a JSON summary to
/scratch/danielco/e189/verify/.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
from src.data_mani import chronos2_syn_repr as C  # noqa: E402

C.PANEL = "data/crypto/v2.0-asof-e189/train.parquet"
CACHE = "/scratch/danielco/c2syn"
OUT = "/scratch/danielco/e189/verify"
DATES = ["2021-06-30", "2023-06-30", "2025-03-14", "2026-04-15"]


def cache():
    ids, reps = [], []
    for f in sorted(glob.glob(os.path.join(CACHE, "part_*.npz"))):
        z = np.load(f)
        ids.append(z["cov_rows"])
        reps.append(z["cov_reps"])
    ids, reps = np.concatenate(ids), np.concatenate(reps)
    return pd.Series(np.arange(len(ids)), index=ids), reps


def compare(ids, got, pos, reps):
    """Max |diff|, share of exactly equal values, min cosine; and coverage agreement."""
    have = pos.reindex(ids)
    missing = int(have.isna().sum())
    ok = have.notna().to_numpy()
    a = got[ok].astype(np.float32)
    b = reps[have[ok].astype(int).to_numpy()].astype(np.float32)
    cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    return dict(rows=int(len(ids)), not_in_cache=missing, max_abs=float(np.abs(a - b).max()) if len(a) else None,
                equal_share=float((a == b).mean()) if len(a) else None, min_cos=float(cos.min()) if len(a) else None,
                width=int(got.shape[1]))


def main():
    os.makedirs(OUT, exist_ok=True)
    pos, reps = cache()
    assert reps.shape[1] == 768, reps.shape
    model = C.Extractor("cuda")
    full, feats, grid = C.load_panel()
    coins = C.coin_arrays(full, feats, len(grid))
    walls = C.wall_steps(grid)
    res = {"checkpoint": C.CKPT, "panel": C.PANEL, "dates": {}}
    for d in DATES:
        T = pd.Timestamp(grid[np.searchsorted(grid, np.datetime64(d), "right") - 1])
        s = int(np.searchsorted(grid, T.to_datetime64()))
        idx = np.flatnonzero(full["step"].to_numpy() == s)
        rows = list(zip(idx, full["ucid"].to_numpy()[idx]))
        got = C.date_reps(model, coins, rows, s, walls[s], len(feats))["cov"]
        cache_ids = pos.index[(pos.index >= idx.min()) & (pos.index <= idx.max())]
        r = {"full": compare(got[0], got[1], pos, reps), "cache_rows_on_date": int(len(cache_ids)),
             "same_covered_set": bool(set(got[0].tolist()) == set(cache_ids.tolist()))}
        np.savez(os.path.join(OUT, "full_%s.npz" % T.date()), ids=got[0], reps=got[1])
        cdf, cf, cg = C.load_panel(cut=T)
        assert cg[-1] == T.to_datetime64() and (cdf[["symbol", "ucid"]].to_numpy() == full.loc[:len(cdf) - 1, ["symbol", "ucid"]].to_numpy()).all()
        cc = C.coin_arrays(cdf, cf, len(cg))
        cs = len(cg) - 1
        cidx = np.flatnonzero(cdf["step"].to_numpy() == cs)
        cut = C.date_reps(model, cc, list(zip(cidx, cdf["ucid"].to_numpy()[cidx])), cs, C.wall_steps(cg)[cs], len(cf))["cov"]
        r["cut"] = compare(cut[0], cut[1], pos, reps)
        r["cut_same_covered_set"] = bool(set(cut[0].tolist()) == set(cache_ids.tolist()))
        np.savez(os.path.join(OUT, "cut_%s.npz" % T.date()), ids=cut[0], reps=cut[1])
        # reproducibility: the same forward pass twice
        again = C.date_reps(model, coins, rows, s, walls[s], len(feats))["cov"]
        r["rerun_identical"] = bool(np.array_equal(again[1], got[1]))
        r["horizon_steps"], r["patch"] = int(s - walls[s]), int((s - walls[s] - 1) // C.PATCH)
        res["dates"][str(T.date())] = r
        print(T.date(), json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(OUT, "verify.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
