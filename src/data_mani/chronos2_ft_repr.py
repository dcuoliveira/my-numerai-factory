"""
Walk-forward LoRA fine-tuned Chronos-2-Synth representations, one model pair per evaluation block.

    # on the GPU box, in the chronos2 venv:
    HF_HOME=/scratch/danielco/hf HF_HUB_OFFLINE=1 PYTHONPATH=<repo> \\
        /scratch/danielco/chronos2/venv/bin/python -m src.data_mani.chronos2_ft_repr [--blocks 11 10 ...]
    ... --smoke                      # 30-step fine-tune of block 11, timing and memory, writes nothing
    ... --audit-cut 2026-03-31 ...   # recompute those dates from a panel cut at each date, block's own models

E143 ran autogluon/chronos-2-synth zero-shot. Here it is fine-tuned on the panel, which makes every
label it trains on in-sample, so a single fine-tune cannot be used for a backtest. Instead each
walk-forward block b of the E133 protocol (train 130 / purge 22 / predict 130) gets its own models:

  1. the panel is cut at block b's FIRST TRAIN DATE T_b exactly as chronos2_syn_repr --audit-cut
     cuts it (rows <= T_b, labels after T_b - 35 days blanked), so the newest label any model sees
     resolved before the block's training window starts;
  2. two LoRA fine-tunes (r 8, the library default targets) on that cut, prediction length 35 =
     the label wall on the daily grid, context 256:
       cov  each coin's label with its 22 features as known-future covariates (one group)
       ar   each coin's label alone
     early-stopped on each series' last 35 steps -- also inside the cut;
  3. both models represent every row of block b's train and test dates exactly as E143 did (label
     wall at D - 35, features after it as known-future covariates, the patch holding D's step).

Block 0's train window starts 2020-01-29, four weeks into the panel: there is no label before it,
so block 0 gets no model and no states (neutral 0.5, routed to v2.0). A row is represented once
per block whose train or test window holds it; build_c2ft keeps them apart.

Output, /scratch/danielco/c2ft/block_NN/: {mode}_rows.npy, {mode}_reps.npy (float16), fit.json.
"""

import argparse
import json
import os
import shutil
import sys
from time import time

import numpy as np
import pandas as pd

from src.data_mani import chronos2_syn_repr as Z

OUT = "/scratch/danielco/c2ft"
PRED_LEN = 35
STEPS = 1000
LR = 1e-5
BATCH = {"cov": 92, "ar": 64}        # series per batch: 4 cov groups of 23 variates; measured 2.4 / 2.0 GB peak on the shared 8 GB card
VAL_EVERY = 100

# E133 walk-forward geometry, on the panel's dates before the test start, plus the test block
TRAIN, PURGE, BLOCK, FIRST = 130, 22, 130, 172
TEST_START, LAST = pd.Timestamp("2026-02-20"), pd.Timestamp("2026-08-20")


def blocks(grid):
    """[(b, train dates, test dates)] -- the 12 windows of e133/e138, recomputed on the panel grid."""
    past = grid[grid < np.datetime64(TEST_START)]
    n = len(past)
    starts = list(range(FIRST, n - BLOCK + 1, BLOCK))
    assert len(starts) == 11 and starts[-1] + BLOCK == n, (starts, n)
    out = [(i, past[b - PURGE - TRAIN:b - PURGE], past[b:b + BLOCK]) for i, b in enumerate(starts)]
    test = grid[(grid >= np.datetime64(TEST_START)) & (grid <= np.datetime64(LAST))]
    return out + [(11, past[-(PURGE + TRAIN):-PURGE], test)]


def series(cut, mode):
    """Fine-tune inputs from the panel cut at `cut`: one dict per coin, NaN where no row."""
    df, feats, grid = Z.load_panel(cut=cut)
    newest = df.loc[df[Z.TARGET].notna(), "date"].max()
    assert newest <= cut - pd.Timedelta(days=Z.WALL_DAYS), (newest, cut)
    coins = Z.coin_arrays(df, feats, len(grid))
    train, val = [], []
    for y, X in coins.values():
        first = np.flatnonzero(np.isfinite(y) | np.isfinite(X).any(1))
        if not len(first):
            continue
        y, X = y[first[0]:], X[first[0]:]
        last = np.flatnonzero(np.isfinite(y))
        if len(last) < Z.MIN_OBS + PRED_LEN:
            continue
        end = last[-1] + 1                     # through the newest resolved label; later rows have none
        y, X = y[:end], X[:end]
        X = np.where(np.isfinite(X), X, 0.5).astype("float32")

        def item(n):
            d = {"target": y[:n].copy()}
            if mode == "cov":
                d["past_covariates"] = {f: X[:n, j].copy() for j, f in enumerate(feats)}
                d["future_covariates"] = {f: None for f in feats}
            return d
        if end - PRED_LEN >= 64 + PRED_LEN:          # fit's own min_past + prediction length
            train.append(item(end - PRED_LEN))
        val.append(item(end))
    return train, val, str(newest.date())


def finetune(base, cut, mode, path, steps=STEPS):
    """LoRA fine-tune of `base` on the cut panel; returns the merged torch model."""
    train, val, newest = series(cut, mode)
    t0 = time()
    ft = base.fit(train, prediction_length=PRED_LEN, validation_inputs=val, finetune_mode="lora",
                  context_length=Z.CTX, learning_rate=LR, num_steps=steps, batch_size=BATCH[mode],
                  min_past=64, output_dir=path, remove_printer_callback=True,
                  save_steps=VAL_EVERY, eval_steps=VAL_EVERY, logging_steps=VAL_EVERY, fp16=True)
    model = ft.model
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    info = dict(mode=mode, cut=str(cut.date()), newest_label=newest, series_train=len(train), series_val=len(val),
                steps=steps, minutes=round((time() - t0) / 60, 1))
    return model.eval(), info


class Model(Z.Extractor):
    """The E143 extractor around an already-built model."""

    def __init__(self, model, device):
        import torch
        self.torch, self.model, self.device = torch, model.to(device).float().eval(), device


def eval_log(path):
    """eval_loss by step from the trainer's last checkpoint state, if any."""
    states = sorted(glob_states(path))
    if not states:
        return []
    h = json.load(open(states[-1]))["log_history"]
    return [dict(step=x["step"], eval_loss=round(x["eval_loss"], 5)) for x in h if "eval_loss" in x]


def glob_states(path):
    import glob
    return glob.glob(os.path.join(path, "checkpoint-*", "trainer_state.json"))


def run_block(base, b, tr, te, df, coins, walls, by_step, grid, device, feats):
    d = os.path.join(OUT, "block_%02d" % b)
    os.makedirs(d, exist_ok=True)
    cut = pd.Timestamp(tr[0])
    steps = [s for s in np.searchsorted(grid, np.concatenate([tr, te])) if walls[s] >= 0]
    meta = json.load(open(os.path.join(d, "fit.json"))) if os.path.exists(os.path.join(d, "fit.json")) else {}
    for mode in Z.MODES:
        if os.path.exists(os.path.join(d, "%s_reps.npy" % mode)):
            continue
        work = os.path.join(d, "ft_" + mode)
        model, info = finetune(base, cut, mode, work)
        info["eval"] = eval_log(work)
        shutil.rmtree(work, ignore_errors=True)
        m = Model(model, device)
        t0, ids, reps = time(), [], []
        for s in steps:
            rows = list(zip(by_step[s], df["ucid"].to_numpy()[by_step[s]]))
            got = Z.date_reps(m, coins, rows, s, walls[s], len(feats))[mode]
            ids.append(got[0])
            reps.append(got[1])
        info["extract_minutes"] = round((time() - t0) / 60, 1)
        np.save(os.path.join(d, "%s_rows.npy" % mode), np.concatenate(ids))
        np.save(os.path.join(d, "%s_reps.tmp.npy" % mode), np.concatenate(reps))
        os.replace(os.path.join(d, "%s_reps.tmp.npy" % mode), os.path.join(d, "%s_reps.npy" % mode))
        torch_save(model, os.path.join(d, "%s_model.pt" % mode))
        meta[mode] = info
        json.dump(meta, open(os.path.join(d, "fit.json"), "w"), indent=1)
        print("block %d %s: cut %s, newest label %s, %d series, fit %.1f min, best eval %s, %d rows in %.1f min"
              % (b, mode, info["cut"], info["newest_label"], info["series_train"], info["minutes"],
                 min((e["eval_loss"] for e in info["eval"]), default=None), sum(len(x) for x in ids), info["extract_minutes"]), flush=True)
        del m, model


def torch_save(model, path):
    """The merged weights, so an audit re-extracts with exactly this model."""
    import torch
    torch.save({k: v.half() for k, v in model.state_dict().items()}, path)


def load_model(base, path):
    import torch
    from copy import deepcopy
    from chronos.chronos2.model import Chronos2Model
    m = Chronos2Model(deepcopy(base.model.config))
    m.load_state_dict({k: v.float() for k, v in torch.load(path).items()})
    return m


def main():
    import torch
    from chronos import Chronos2Pipeline
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blocks", type=int, nargs="+", default=list(range(11, 0, -1)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--audit-cut", nargs="+", default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    base = Chronos2Pipeline.from_pretrained(Z.CKPT, device_map=a.device, torch_dtype=torch.float32)
    df, feats, grid = Z.load_panel()
    B = {b: (tr, te) for b, tr, te in blocks(grid)}

    if a.smoke:
        tr, te = B[11]
        for mode in Z.MODES:
            t0 = time()
            model, info = finetune(base, pd.Timestamp(tr[0]), mode, "/scratch/danielco/c2ft_smoke_" + mode, steps=30)
            torch.cuda.synchronize()
            print("smoke %s: %d train series, 30 steps %.1fs, peak %.0f MiB, eval %s" % (
                mode, info["series_train"], time() - t0, torch.cuda.max_memory_allocated() / 2 ** 20,
                eval_log("/scratch/danielco/c2ft_smoke_" + mode)), flush=True)
            shutil.rmtree("/scratch/danielco/c2ft_smoke_" + mode, ignore_errors=True)
        return

    if a.audit_cut:
        # The block models are stored in fp16, so a reloaded model is not bit-identical to the one the
        # full run extracted with. Each date is therefore extracted twice with the SAME reloaded model:
        # from the panel cut at it ({mode}_{date}.npy) and from the full panel ({mode}_{date}_full.npy).
        # These two must agree exactly if nothing past the wall reaches a row's state.
        fcoins, fwalls = Z.coin_arrays(df, feats, len(grid)), Z.wall_steps(grid)
        for cut in map(pd.Timestamp, a.audit_cut):
            b = [k for k, (tr, te) in B.items() if np.datetime64(cut) in set(te)][0]
            cdf, cfeats, cgrid = Z.load_panel(cut=cut)
            coins = Z.coin_arrays(cdf, cfeats, len(cgrid))
            s = len(cgrid) - 1
            rows = list(zip(cdf.index[cdf["step"] == s], cdf.loc[cdf["step"] == s, "ucid"]))
            keys = cdf.loc[cdf["step"] == s, ["symbol", "ucid", "date"]]
            fs = int(np.searchsorted(grid, np.datetime64(cut)))
            assert grid[fs] == np.datetime64(cut)
            frows = list(zip(df.index[df["step"] == fs], df.loc[df["step"] == fs, "ucid"]))
            for mode in Z.MODES:
                m = Model(load_model(base, os.path.join(OUT, "block_%02d" % b, "%s_model.pt" % mode)), a.device)
                ids, reps = Z.date_reps(m, coins, rows, s, Z.wall_steps(cgrid)[s], len(cfeats))[mode]
                fids, freps = Z.date_reps(m, fcoins, frows, fs, fwalls[fs], len(feats))[mode]
                os.makedirs(os.path.join(OUT, "audit"), exist_ok=True)
                np.save(os.path.join(OUT, "audit", "%s_%s.npy" % (mode, cut.date())), reps)
                keys.loc[ids].assign(block=b).to_parquet(os.path.join(OUT, "audit", "%s_%s.parquet" % (mode, cut.date())), index=False)
                # full-panel states in the cut file's row order, NaN where the full run has none
                pos = dict(zip(df.loc[fids, "ucid"].to_numpy(), range(len(fids))))
                order = [pos.get(u, -1) for u in keys.loc[ids, "ucid"].to_numpy()]
                full = np.full(reps.shape, np.nan, "float32")
                full[[i for i, o in enumerate(order) if o >= 0]] = np.asarray(freps, "float32")[[o for o in order if o >= 0]]
                np.save(os.path.join(OUT, "audit", "%s_%s_full.npy" % (mode, cut.date())), full)
            print("audit cut %s (block %d): %d rows, %d with reps" % (cut.date(), b, len(rows), len(ids)), flush=True)
        return

    os.makedirs(OUT, exist_ok=True)
    df[["symbol", "ucid", "date"]].to_parquet(os.path.join(OUT, "rows.parquet"))
    coins = Z.coin_arrays(df, feats, len(grid))
    walls = Z.wall_steps(grid)
    by_step = df.groupby("step").indices
    for b in a.blocks:
        if b == 0:
            continue
        tr, te = B[b]
        run_block(base, b, tr, te, df, coins, walls, by_step, grid, a.device, feats)


if __name__ == "__main__":
    sys.exit(main())
