"""
Frozen hidden states of other time-series foundation models on exactly E189's rows (E193-E197).

    # on a GPU box, in the model's own venv (imports only numpy/pandas/torch + the adapter's model code):
    CUDA_VISIBLE_DEVICES=0 <venv>/bin/python -m src.data_mani.tsfm_states --model timestral --shard 0 --shards 2
    ... --smoke          # 2 dates, timing + shapes + determinism, writes nothing
    ... --audit-cut T    # recompute date T from the panel cut at T (labels after T - 35d blanked)

Row set, context and point-in-time rule are E189's, unchanged (src/data_mani/chronos2_syn_repr.py):
the frozen panel data/crypto/v2.0-asof-e189, one series per coin on the global panel-date grid, the
label target_binned_return_20 as target, the 22 shipped features as covariates. Row (ucid, D) reads
labels only from grid steps <= wall(D) = last step with date <= D - 35 days, inside a 256-step
context, and is computed only if it is in E189's covered set (>= 32 resolved labels in that context;
the driver recomputes the rule and asserts it equals the Chronos-2 cache's row set). Features of the
h = step(D) - wall steps after the wall are known-future covariates (NaN -> 0.5, as in E189).

Each adapter (src/data_mani/tsfm_state_adapters/<model>.py) turns one row into ONE vector: the model's
last hidden state of the TARGET series at the forecast position holding D's step -- the tensor its
forecasting head reads for D. No pooling, PCA or quantiles. Output: /scratch/danielco/tsfmstates/<model>/
part_XXX.npz with cov_rows (row ids into the panel sorted by date, ucid) and cov_reps (float16), the same
layout as /scratch/danielco/c2syn so experiments/e189_c2norm.py reads it with --tsfm <model>.
"""
import argparse
import glob
import importlib
import json
import os
from time import time

import numpy as np
import pandas as pd

from src.data_mani import chronos2_syn_repr as C

PANEL = "data/crypto/v2.0-asof-e189/train.parquet"
C2CACHE = "/scratch/danielco/c2syn"
OUT = "/scratch/danielco/tsfmstates"
LAST = np.datetime64("2026-08-20")
PART = 50                           # grid steps per part file (same blocks as c2syn)
MODELS = ("toto", "tabpfnts", "tempopfn", "t0", "timestral")


def covered_ids():
    """E189's covered row ids (the Chronos-2 cov cache), as a set of panel row ids."""
    ids = [np.load(f)["cov_rows"] for f in sorted(glob.glob(os.path.join(C2CACHE, "part_*.npz")))]
    return np.unique(np.concatenate(ids))


def row_inputs(coins, rows, step, wall, last_date):
    """
    Model inputs for the rows of one grid step.

    :return: ids, y (b, CTX) target context left-padded with NaN, X (b, 22, CTX) covariate context,
             Xf (b, 22, h) known-future covariates (NaN -> 0.5), h, last_dates (b,) date of the wall step
    """
    lo = max(0, wall - C.CTX + 1)
    pad = C.CTX - (wall + 1 - lo)
    h = step - wall
    ids, Y, X, F = [], [], [], []
    for rid, ucid in rows:
        y, x = coins[ucid]
        if np.isfinite(y[lo:wall + 1]).sum() < C.MIN_OBS:
            continue
        yy = np.full(C.CTX, np.nan, "float32")
        yy[pad:] = y[lo:wall + 1]
        xx = np.full((x.shape[1], C.CTX), np.nan, "float32")
        xx[:, pad:] = x[lo:wall + 1].T
        f = x[wall + 1:step + 1].T
        ids.append(rid)
        Y.append(yy)
        X.append(xx)
        F.append(np.where(np.isfinite(f), f, 0.5).astype("float32"))
    if not ids:
        return np.zeros(0, "int64"), None, None, None, h, None
    return (np.asarray(ids, "int64"), np.stack(Y), np.stack(X), np.stack(F), h,
            np.full(len(ids), last_date, dtype="datetime64[D]"))


def load_adapter(name, device):
    mod = importlib.import_module("src.data_mani.tsfm_state_adapters." + name)
    return mod.Extractor(device)


def run_step(ext, coins, df, by_step, grid, walls, s):
    rows = list(zip(by_step[s], df["ucid"].to_numpy()[by_step[s]]))
    ids, y, X, F, h, ld = row_inputs(coins, rows, s, walls[s], grid[walls[s]])
    if not len(ids):
        return ids, np.zeros((0, ext.dim), "float16")
    reps = ext.states(y, X, F, h, ld)
    assert reps.shape == (len(ids), ext.dim) and np.isfinite(reps).all(), (reps.shape, s)
    return ids, reps.astype("float16")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--claim", action="store_true",
                    help="ignore --shard: take parts from the LAST one down, claiming each with an O_EXCL lock file "
                         "(extra workers next to running shards; a part both start is computed twice, written once)")
    ap.add_argument("--audit-cut", nargs="+", default=None)
    a = ap.parse_args()
    out = os.path.join(OUT, a.model)
    ext = load_adapter(a.model, a.device)

    if a.audit_cut:
        os.makedirs(os.path.join(out, "audit"), exist_ok=True)
        for cut in a.audit_cut:
            cut = pd.Timestamp(cut)
            df, feats, grid = C.load_panel(PANEL, cut=cut)
            coins = C.coin_arrays(df, feats, len(grid))
            walls = C.wall_steps(grid)
            s = len(grid) - 1
            ids, reps = run_step(ext, coins, df, df.groupby("step").indices, grid, walls, s)
            keys = df.loc[ids, ["symbol", "ucid", "date"]]
            np.savez(os.path.join(out, "audit", "cut_%s.npz" % cut.date()), reps=reps, ucid=keys["ucid"].to_numpy())
            print("audit cut %s: %d rows with states" % (cut.date(), len(ids)), flush=True)
        return

    df, feats, grid = C.load_panel(PANEL)
    coins = C.coin_arrays(df, feats, len(grid))
    walls = C.wall_steps(grid)
    by_step = df.groupby("step").indices
    steps = [s for s in range(len(grid)) if walls[s] >= 0 and grid[s] <= LAST]

    if a.smoke:
        cov = set(covered_ids().tolist())
        for s in (steps[200], steps[-40]):
            t0 = time()
            ids, reps = run_step(ext, coins, df, by_step, grid, walls, s)
            dt = time() - t0
            want = sorted(r for r in by_step[s] if r in cov)
            ids2, reps2 = run_step(ext, coins, df, by_step, grid, walls, s)
            r = reps.astype("float32")
            print("smoke %s: h=%d rows=%d states=%d (c2 covered %d, same set %s) dim=%d %.1fs (%.1f rows/s) "
                  "row-norm %.2f rerun max|d| %.2e"
                  % (str(grid[s])[:10], s - walls[s], len(by_step[s]), len(ids), len(want), list(ids) == want,
                     reps.shape[1], dt, len(ids) / dt, np.linalg.norm(r, axis=1).mean(),
                     np.abs(r - reps2.astype("float32")).max()), flush=True)
        return

    os.makedirs(out, exist_ok=True)
    cov = covered_ids()
    parts = [steps[i:i + PART] for i in range(0, len(steps), PART)]
    order = list(enumerate(parts))[::-1] if a.claim else list(enumerate(parts))
    for k, block in order:
        if not a.claim and k % a.shards != a.shard:
            continue
        path = os.path.join(out, "part_%03d.npz" % k)
        if os.path.exists(path):
            continue
        if a.claim:
            try:
                os.close(os.open(path + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
        t0 = time()
        I, R = [], []
        for s in block:
            ids, reps = run_step(ext, coins, df, by_step, grid, walls, s)
            I.append(ids)
            R.append(reps)
        I, R = np.concatenate(I), np.concatenate(R)
        lo, hi = by_step[block[0]].min(), by_step[block[-1]].max()
        want = cov[(cov >= lo) & (cov <= hi)]
        assert np.array_equal(np.sort(I), want), "row set differs from E189's covered set in part %d" % k
        np.savez(path + ".tmp.npz", cov_rows=I, cov_reps=R)
        os.replace(path + ".tmp.npz", path)
        json.dump({"model": a.model, "checkpoint": ext.checkpoint, "dim": ext.dim, "panel": PANEL,
                   "notes": getattr(ext, "notes", ""), "amp": os.environ.get("TSFM_AMP", "fp16")},
                  open(os.path.join(out, "manifest.json"), "w"), indent=1)
        print("part %d/%d: %s..%s, %d rows, %.1f min" % (k + 1, len(parts), str(grid[block[0]])[:10],
                                                   str(grid[block[-1]])[:10], len(I), (time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
