"""
Chronos-2-Synth representations of each coin's own panel history, taken where the forecasting head reads them.

    # on the GPU box, in the chronos2 venv (chronos-forecasting 2.3.2; the main env has no chronos):
    HF_HOME=/scratch/danielco/hf HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0 \\
        /scratch/danielco/chronos2/venv/bin/python -m src.data_mani.chronos2_syn_repr --shard 0 --shards 2
    ... --smoke                         # 2 dates, prints timing and shapes, writes nothing
    ... --audit-cut 2023-06-30          # recompute that date from a panel cut at it (see below)

The v2.0 panel is turned into one time series per coin (ucid) on the global panel-date grid: the
label target_binned_return_20 and the 22 shipped features, NaN where the coin has no row. Then
autogluon/chronos-2-synth (Chronos-2's 120M architecture, pretrained on synthetic series only) is
run zero-shot on every panel row (ucid, D) in two modes:

  cov  the coin's label series with its 22 features as covariates, one group of 23 variates, so
       group attention mixes them: temporal + covariate patterns.
  ar   the coin's label series alone, univariate: temporal patterns only.

What is kept is the encoder's last hidden state of the TARGET variate at the output patch that
holds date D's step -- exactly the 768-vector the forecasting head (output_patch_embedding) turns
into date D's quantiles. Nothing downstream of the encoder is used.

Point in time. A label dated t is a 20-day forward return that resolves about 30 days later
(E142: close(t+2) -> close(t+30)), and Numerai serves train.parquet 32 days behind live. So row D
reads labels only from dates <= D - WALL_DAYS (35). The features of dates <= D are known at D;
those after the label wall enter as KNOWN-FUTURE covariates, so the row's current features
condition the forecast while no label after the wall is read. The forecast horizon h is the
number of panel steps from the wall to D (35 on the daily grid, ~12 on the early 3-day grid),
and the representation is the patch holding step h. A row needs MIN_OBS resolved labels in its
CTX-step context, else it is left out (NaN, neutral at load). --audit-cut T recomputes date T
from rows <= T with every label after T - WALL_DAYS blanked -- what was knowable at T -- for
build_c2syn's point-in-time check.

Output: raw float16 representations on /scratch (they are large), one part file per block of
dates, so a job killed mid-run resumes where it stopped. build_c2syn reduces and bins them.

Imports only numpy/pandas/torch at module scope: it runs in a venv without the repo's stack.
"""

import argparse
import os
from time import time

import numpy as np
import pandas as pd

PANEL = os.path.join("data", "crypto", "v2.0", "train.parquet")
OUT = "/scratch/danielco/c2syn"
CKPT = "autogluon/chronos-2-synth"
TARGET = "target_binned_return_20"
WALL_DAYS = 35
CTX = 256                  # context steps, a multiple of the 16-step patch
MIN_OBS = 32               # resolved labels a row needs inside its context
PATCH = 16
CHUNK = 16                 # rows per forward pass; group attention is quadratic in the 16 x 23 variates
PART = 50                  # dates per part file
MODES = ("cov", "ar")


def load_panel(path=PANEL, cut=None):
    """Panel sorted by date, the global date grid, and each row's grid step."""
    df = pd.read_parquet(path).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    feats = sorted(c for c in df.columns if c.startswith("feature_"))
    assert len(feats) == 22, len(feats)
    if cut is not None:
        df = df[df["date"] <= cut].copy()
        df.loc[df["date"] > cut - pd.Timedelta(days=WALL_DAYS), TARGET] = np.nan
    df = df.sort_values(["date", "ucid"]).reset_index(drop=True)
    grid = np.sort(df["date"].unique())
    df["step"] = np.searchsorted(grid, df["date"].to_numpy())
    return df, feats, grid


def wall_steps(grid):
    """For each grid step, the last step whose date is <= date - WALL_DAYS (-1 if none)."""
    return np.searchsorted(grid, grid - np.timedelta64(WALL_DAYS, "D"), side="right") - 1


def coin_arrays(df, feats, n_steps):
    """{ucid: (y (n_steps,), X (n_steps, 22))}, NaN where the coin has no row."""
    out = {}
    y_all = df[TARGET].to_numpy("float32")
    x_all = df[feats].to_numpy("float32")
    steps = df["step"].to_numpy()
    for ucid, idx in df.groupby("ucid").indices.items():
        y = np.full(n_steps, np.nan, "float32")
        X = np.full((n_steps, len(feats)), np.nan, "float32")
        y[steps[idx]] = y_all[idx]
        X[steps[idx]] = x_all[idx]
        out[ucid] = (y, X)
    return out


class Extractor:
    def __init__(self, device):
        import torch
        from chronos import Chronos2Pipeline
        self.torch = torch
        pipe = Chronos2Pipeline.from_pretrained(CKPT, device_map=device, torch_dtype=torch.float32)
        self.model = pipe.model.eval()
        self.device = device

    def run(self, ctx, fut, fut_mask, groups, n_patches, take):
        """
        Encoder hidden states of the rows `take` at the n_patches output patches.

        Under fp16 autocast: twice as fast on the RTX 2070 as fp32, cosine to the fp32 states >= 0.99999
        on every row of a 300-coin date in both modes, no NaN.
        """
        t = self.torch
        with t.inference_mode(), t.autocast("cuda", dtype=t.float16):
            kw = dict(context=t.from_numpy(ctx).to(self.device), group_ids=t.from_numpy(groups).to(self.device),
                      num_output_patches=n_patches)
            if fut is not None:
                kw["future_covariates"] = t.from_numpy(fut).to(self.device)
                kw["future_covariates_mask"] = t.from_numpy(fut_mask).to(self.device)
            enc = self.model.encode(**kw)[0]
            hidden = enc[0] if isinstance(enc, tuple) else enc.last_hidden_state
            return hidden[take, -n_patches:]


def date_reps(model, coins, rows, step, wall, n_feats):
    """
    Both modes' representations for the rows of one date.

    :param rows: [(row id, ucid)] on grid step `step`
    :param wall: last grid step whose label is readable
    :return: {mode: (row ids, (n, 768) float16)}
    """
    h = step - wall
    n_patches = -(-h // PATCH)
    patch = (h - 1) // PATCH
    lo = max(0, wall - CTX + 1)
    keep = []
    for rid, ucid in rows:
        y, X = coins[ucid]
        if np.isfinite(y[lo:wall + 1]).sum() >= MIN_OBS:
            keep.append((rid, y, X))
    out = {m: ([], []) for m in MODES}
    v = n_feats + 1
    pad = CTX - (wall + 1 - lo)
    for i in range(0, len(keep), CHUNK):
        part = keep[i:i + CHUNK]
        b = len(part)
        ctx = np.full((b * v, CTX), np.nan, "float32")
        fut = np.full((b * v, h), np.nan, "float32")
        mask = np.zeros((b * v, h), "float32")
        for j, (_, y, X) in enumerate(part):
            ctx[j * v, pad:] = y[lo:wall + 1]
            ctx[j * v + 1:(j + 1) * v, pad:] = X[lo:wall + 1].T
            f = X[wall + 1:step + 1].T
            fut[j * v + 1:(j + 1) * v] = np.where(np.isfinite(f), f, 0.5)
            mask[j * v + 1:(j + 1) * v] = 1.0
        tgt = np.arange(b) * v
        groups = np.repeat(np.arange(b), v).astype("int64")
        cov = model.run(ctx, fut, mask, groups, n_patches, model.torch.from_numpy(tgt))
        ar = model.run(np.ascontiguousarray(ctx[tgt]), None, None, np.arange(b, dtype="int64"), n_patches, slice(None))
        ids = [p[0] for p in part]
        for m, rep in (("cov", cov), ("ar", ar)):
            out[m][0].extend(ids)
            out[m][1].append(rep[:, patch].float().cpu().numpy().astype("float16"))
    return {m: (np.asarray(ids, "int64"), np.concatenate(r) if r else np.zeros((0, 768), "float16"))
            for m, (ids, r) in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--audit-cut", nargs="+", default=None)
    a = ap.parse_args()
    model = Extractor(a.device)

    if a.audit_cut:
        os.makedirs(os.path.join(OUT, "audit"), exist_ok=True)
        for cut in a.audit_cut:
            cut = pd.Timestamp(cut)
            df, feats, grid = load_panel(cut=cut)
            coins = coin_arrays(df, feats, len(grid))
            s = len(grid) - 1
            rows = list(zip(df.index[df["step"] == s], df.loc[df["step"] == s, "ucid"]))
            got = date_reps(model, coins, rows, s, wall_steps(grid)[s], len(feats))
            keys = df.loc[df["step"] == s, ["symbol", "ucid", "date"]]
            for m, (ids, reps) in got.items():
                np.save(os.path.join(OUT, "audit", "%s_%s.npy" % (m, cut.date())), reps)
                keys.loc[ids].to_parquet(os.path.join(OUT, "audit", "%s_%s.parquet" % (m, cut.date())), index=False)
            print("audit cut %s: %d rows, %d with reps" % (cut.date(), len(rows), len(got["cov"][0])), flush=True)
        return

    df, feats, grid = load_panel()
    coins = coin_arrays(df, feats, len(grid))
    walls = wall_steps(grid)
    by_step = df.groupby("step").indices
    steps = [s for s in range(len(grid)) if walls[s] >= 0]
    if a.smoke:
        for s in (steps[200], steps[-40]):
            rows = list(zip(by_step[s], df["ucid"].to_numpy()[by_step[s]]))
            t0 = time()
            got = date_reps(model, coins, rows, s, walls[s], len(feats))
            print("smoke %s: h=%d, %d rows, %d with reps, cov %s ar %s, %.1fs"
                  % (str(grid[s])[:10], s - walls[s], len(rows), len(got["cov"][0]), got["cov"][1].shape,
                     got["ar"][1].shape, time() - t0), flush=True)
            r = got["cov"][1].astype("float32")
            print("  cov row-norm %.2f, cov-vs-ar mean cosine %.3f" % (
                np.linalg.norm(r, axis=1).mean(),
                np.mean(np.sum(r * got["ar"][1], 1) / np.linalg.norm(r, axis=1) / np.linalg.norm(got["ar"][1].astype("float32"), axis=1))))
        return

    os.makedirs(OUT, exist_ok=True)
    keys = df[["symbol", "ucid", "date"]]
    keys.to_parquet(os.path.join(OUT, "rows.parquet"))
    parts = [steps[i:i + PART] for i in range(0, len(steps), PART)]
    for k, block in enumerate(parts):
        if k % a.shards != a.shard:
            continue
        path = os.path.join(OUT, "part_%03d.npz" % k)
        if os.path.exists(path):
            continue
        t0 = time()
        acc = {m: ([], []) for m in MODES}
        for s in block:
            rows = list(zip(by_step[s], df["ucid"].to_numpy()[by_step[s]]))
            got = date_reps(model, coins, rows, s, walls[s], len(feats))
            for m in MODES:
                acc[m][0].append(got[m][0])
                acc[m][1].append(got[m][1])
        np.savez(path + ".tmp.npz", **{"%s_%s" % (m, w): np.concatenate(acc[m][i]) for m in MODES for i, w in ((0, "rows"), (1, "reps"))})
        os.replace(path + ".tmp.npz", path)
        print("part %d/%d: %s..%s, %d rows, %.1f min" % (k + 1, len(parts), str(grid[block[0]])[:10], str(grid[block[-1]])[:10],
                                                   sum(len(x) for x in acc["cov"][0]), (time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
