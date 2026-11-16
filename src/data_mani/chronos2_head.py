"""
Chronos-2-Synth FORECASTS (the head's quantiles) of each coin's label, zero-shot and walk-forward LoRA fine-tuned.

    # on the GPU box, in the chronos2 venv:
    HF_HOME=/scratch/danielco/hf HF_HUB_OFFLINE=1 PYTHONPATH=<repo> \\
        /scratch/danielco/chronos2/venv/bin/python -m src.data_mani.chronos2_head zs [--smoke]
    ... ft [--blocks 11 10 ...] [--smoke]
    ... audit --dates 2021-12-31 2023-06-30 ...     # cut-panel rebuilds, both families

E145. Unlike E143/E144 nothing upstream of the forecasting head is kept: every row gets the 20-step
quantile forecast itself, summarised later by build_c2head.

Input for row (coin, D), one per row, identical for every branch:
  * the coin's history on the panel-date grid, the CTX = 256 steps ending at D, NaN where it has no
    row (masked by Chronos, never filled from later dates);
  * target = target_binned_return_20, blanked after the label wall W(D) = the last step dated
    <= D - 35 days: a label dated t resolves about 30 days later, so only labels <= W(D) are known;
  * cov branches: the 22 v2.0 features through D as PAST-ONLY covariates (no future values given;
    Chronos forecasts them too, and those forecasts are discarded);
  * forecast the next HORIZON = 20 steps after D; keep q10, q50, q90 of the target.
  A row needs MIN_OBS = 32 resolved labels in its context, else it has no forecast (NaN downstream).

Branches: zs_cov / zs_ar = frozen autogluon/chronos-2-synth at revision REV; ft_cov / ft_ar = one LoRA
fine-tune per walk-forward block b = 2..11 (blocks 0-1 have too few resolved labels before their
cut) on the panel cut at b's first train date T_b. The fine-tune loop builds every training window
exactly like an inference row (wall 35 days before the origin, past-only covariates, 20-step
target), so training matches use. Its target labels all resolved by T_b - 35 days; early stopping
uses the last 35 resolved steps, also inside the cut. The block's train and test rows are then
forecast with the frozen model.

Output /scratch/danielco/c2head/: zs_{cov,ar}.npz and block_NN/ft_{cov,ar}.npz (rows, q [n,3,20]
float32), block_NN/{mode}_adapter.pt (LoRA weights, fp16), block_NN/{mode}_manifest.json, and
audit/. Imports only numpy/pandas/torch at module scope (the chronos venv has no repo stack).
"""

import argparse
import hashlib
import json
import os
import sys
from time import time

import numpy as np
import pandas as pd

from src.data_mani import chronos2_syn_repr as Z

OUT = "/scratch/danielco/c2head"
# data/crypto/v2.0 is rewritten by the live round every morning; every stage reads this frozen copy
# (byte-identical to the file the zs stage loaded, 2026-11-06 09:45) so row ids agree across stages.
PANEL = os.path.join("data", "crypto", "v2.0-asof-e145", "train.parquet")
REV = "3607918a9fd027d5c465d8213e46b98e2c041cea"     # autogluon/chronos-2-synth, the snapshot E143/E144 ran
HORIZON = 20
N_OUT = 2                                          # output patches of 16 steps; the first 20 are kept
QIDX = (2, 6, 10)                                  # q10, q50, q90 in the model's 13 quantiles
CHUNK = {"cov": 8, "ar": 128}                      # rows per inference pass (group attention is quadratic in the batch)
STEPS, LR, VAL_EVERY, SEED = 1000, 1e-5, 100, 145
TRAIN_BATCH = {"cov": 8, "ar": 64}                 # rows per fine-tune step (cov: x 23 variates)
N_VAL = 512
TRAIN, PURGE, BLOCK, FIRST = 130, 22, 130, 172
TEST_START, LAST = pd.Timestamp("2026-02-20"), pd.Timestamp("2026-08-20")
FT_BLOCKS = list(range(2, 12))
LORA = dict(r=8, lora_alpha=16, target_modules=["self_attention.q", "self_attention.v", "self_attention.k",
                                                  "self_attention.o", "output_patch_embedding.output_layer"])
MODES = ("cov", "ar")


def blocks(grid):
    """[(b, train dates, test dates)]: the 12 E133 windows on the panel grid (as chronos2_ft_repr)."""
    past = grid[grid < np.datetime64(TEST_START)]
    n = len(past)
    starts = list(range(FIRST, n - BLOCK + 1, BLOCK))
    assert len(starts) == 11 and starts[-1] + BLOCK == n, (starts, n)
    out = [(i, past[b - PURGE - TRAIN:b - PURGE], past[b:b + BLOCK]) for i, b in enumerate(starts)]
    test = grid[(grid >= np.datetime64(TEST_START)) & (grid <= np.datetime64(LAST))]
    return out + [(11, past[-(PURGE + TRAIN):-PURGE], test)]


# ------------------------------------------------------------------ one row's input
def window(y, X, s, wall):
    """(target ctx (CTX,), covariate ctx (22, CTX), resolved count): steps s-CTX+1..s, labels after wall blanked."""
    lo = s - Z.CTX + 1
    pad = max(0, -lo)
    lo = max(0, lo)
    t = np.full(Z.CTX, np.nan, "float32")
    c = np.full((X.shape[1], Z.CTX), np.nan, "float32")
    t[pad:] = y[lo:s + 1]
    if wall < s:
        t[Z.CTX - (s - wall):] = np.nan
    c[:, pad:] = X[lo:s + 1].T
    return t, c, int(np.isfinite(t).sum())


def batch(items, mode):
    """Stack (t, c) windows into the model's (context, group_ids) and the target rows' indices."""
    if mode == "ar":
        ctx = np.stack([t for t, _ in items])
        return ctx, np.arange(len(items), dtype="int64"), np.arange(len(items))
    v = items[0][1].shape[0] + 1
    ctx = np.empty((len(items) * v, Z.CTX), "float32")
    for j, (t, c) in enumerate(items):
        ctx[j * v] = t
        ctx[j * v + 1:(j + 1) * v] = c
    return ctx, np.repeat(np.arange(len(items)), v).astype("int64"), np.arange(len(items)) * v


class Head:
    """Frozen Chronos2Model -> q10/q50/q90 of the target's next HORIZON steps."""

    def __init__(self, model, device):
        import torch
        self.torch, self.model, self.device = torch, model.to(device).float().eval(), device

    def forecast(self, items, mode):
        t = self.torch
        ctx, groups, tgt = batch(items, mode)
        with t.inference_mode():       # fp32, TF32 off: batch-invariant to ~2e-6 (fp16 autocast moved q by up to 0.01 with batch composition)
            q = self.model(context=t.from_numpy(ctx).to(self.device), group_ids=t.from_numpy(groups).to(self.device),
                           num_output_patches=N_OUT).quantile_preds
        return q[t.from_numpy(tgt).to(self.device)][:, list(QIDX), :HORIZON].float().cpu().numpy()


def base_model(device):
    import torch
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(Z.CKPT, revision=REV, device_map=device, torch_dtype=torch.float32).model


def date_items(coins, rows, s, wall):
    keep = []
    for rid, ucid in rows:
        y, X = coins[ucid]
        t, c, n = window(y, X, s, wall)
        if n >= Z.MIN_OBS:
            keep.append((rid, t, c))
    return keep


def forecast_steps(head, df, coins, walls, steps, modes=MODES, log=None):
    """{mode: (row ids, q)} for every row on the given grid steps."""
    by_step = df.groupby("step").indices
    ucids = df["ucid"].to_numpy()
    out = {m: ([], []) for m in modes}
    t0 = time()
    for k, s in enumerate(steps):
        if walls[s] < 0:
            continue
        keep = date_items(coins, list(zip(by_step[s], ucids[by_step[s]])), s, walls[s])
        for m in modes:
            for i in range(0, len(keep), CHUNK[m]):
                part = keep[i:i + CHUNK[m]]
                out[m][0].append(np.array([p[0] for p in part], "int64"))
                out[m][1].append(head.forecast([(p[1], p[2]) for p in part], m))
        if log and (k + 1) % 100 == 0:
            print("  %s %d/%d dates, %.1f min" % (log, k + 1, len(steps), (time() - t0) / 60), flush=True)
    return {m: (np.concatenate(i) if i else np.zeros(0, "int64"), np.concatenate(q) if q else np.zeros((0, 3, HORIZON), "float32"))
            for m, (i, q) in out.items()}


def save(path, ids, q):
    np.savez(path + ".tmp.npz", rows=ids, q=q)
    os.replace(path + ".tmp.npz", path)


# ------------------------------------------------------------------ fine-tuning
def ft_windows(df, feats, grid):
    """Training and validation windows (ucid, origin step) from the panel already cut, built lazily by ft_items."""
    coins = Z.coin_arrays(df, feats, len(grid))
    walls = Z.wall_steps(grid)
    tr, va, keys = [], [], []
    newest = df.loc[df[Z.TARGET].notna(), "date"].max()
    for ucid, (y, X) in coins.items():
        fin = np.isfinite(y)
        res = np.flatnonzero(fin)
        if not len(res):
            continue
        last = res[-1]                       # newest resolved label of this coin, <= cut - 35 days
        cs = np.concatenate([[0], np.cumsum(fin)])
        for o in range(len(grid) - 1):
            if walls[o] < 0 or o + 1 > last:
                continue
            if cs[min(o + 1 + HORIZON, len(y))] - cs[o + 1] < HORIZON // 2:
                continue
            if cs[walls[o] + 1] - cs[max(0, o - Z.CTX + 1)] < Z.MIN_OBS:
                continue
            if o + HORIZON <= last - 35:
                tr.append((ucid, o))
                keys.append("%s:%s" % (ucid, str(grid[o])[:10]))
            elif o + 1 > last - 35:
                va.append((ucid, o))
                keys.append("v%s:%s" % (ucid, str(grid[o])[:10]))
    return coins, walls, tr, va, newest, hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def ft_items(coins, walls, pairs):
    out = []
    for ucid, o in pairs:
        y, X = coins[ucid]
        t, c, _ = window(y, X, o, walls[o])
        f = np.full(HORIZON, np.nan, "float32")
        fut = y[o + 1:o + 1 + HORIZON]
        f[:len(fut)] = fut
        out.append((t, c, f))
    return out


def ft_loss(model, items, mode, device):
    """The model's own quantile loss on the windows' 20-step targets (covariate rows unscored)."""
    import torch
    ctx, groups, tgt = batch([(t, c) for t, c, _ in items], mode)
    fut = np.full((len(ctx), N_OUT * 16), np.nan, "float32")
    fut[tgt, :HORIZON] = np.stack([f for _, _, f in items])
    with torch.autocast("cuda", dtype=torch.float16):
        return model(context=torch.from_numpy(ctx).to(device), group_ids=torch.from_numpy(groups).to(device),
                     num_output_patches=N_OUT, future_target=torch.from_numpy(fut).to(device)).loss


def finetune(base, cut, mode, device, steps=STEPS, seed=SEED):
    """LoRA fine-tune of a copy of `base` on the panel cut at `cut`; (merged model, adapter state, manifest)."""
    import peft
    import torch
    import transformers
    from copy import deepcopy
    from peft import LoraConfig, get_peft_model
    df, feats, grid = Z.load_panel(PANEL, cut=cut)
    coins, walls, tr, va, newest, khash = ft_windows(df, feats, grid)
    assert newest <= cut - pd.Timedelta(days=Z.WALL_DAYS), (newest, cut)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    va = ft_items(coins, walls, [va[i] for i in sorted(rng.choice(len(va), min(N_VAL, len(va)), replace=False))])
    model = get_peft_model(deepcopy(base).to(device).train(), LoraConfig(**LORA))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    scaler = torch.amp.GradScaler("cuda")
    vb = TRAIN_BATCH[mode] * 4

    def val():
        model.eval()
        with torch.no_grad():
            ls = [float(ft_loss(model, va[i:i + vb], mode, device)) * len(va[i:i + vb]) for i in range(0, len(va), vb)]
        model.train()
        return sum(ls) / len(va)

    t0, log = time(), [dict(step=0, val_loss=round(val(), 5))]
    best = (log[0]["val_loss"], 0, {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k})
    for step in range(1, steps + 1):
        idx = rng.choice(len(tr), TRAIN_BATCH[mode], replace=False)
        loss = ft_loss(model, ft_items(coins, walls, [tr[i] for i in idx]), mode, device)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % VAL_EVERY == 0:
            v = val()
            log.append(dict(step=step, val_loss=round(v, 5), train_loss=round(float(loss), 5)))
            if v < best[0]:
                best = (v, step, {k: x.detach().clone() for k, x in model.state_dict().items() if "lora_" in k})
    model.load_state_dict(best[2], strict=False)
    adapter = {k: v.half().cpu() for k, v in best[2].items()}
    merged = model.merge_and_unload().eval()
    manifest = dict(mode=mode, cut=str(cut.date()), max_feature_date=str(df["date"].max().date()), max_label_date=str(newest.date()),
                    train_windows=len(tr), val_windows=len(va), key_sha256=khash, model=Z.CKPT, revision=REV, adapter=LORA,
                    lr=LR, steps=steps, batch=TRAIN_BATCH[mode], seed=seed, best_step=best[1], val_log=log,
                    minutes=round((time() - t0) / 60, 1),
                    versions=dict(torch=torch.__version__, peft=peft.__version__, transformers=transformers.__version__,
                                  numpy=np.__version__, pandas=pd.__version__, chronos=_chronos_version()))
    return merged, adapter, manifest


def _chronos_version():
    from importlib.metadata import version
    return version("chronos-forecasting")


def merge_adapter(base, adapter, device):
    """base + a saved (fp16) adapter, merged: what an audit reloads."""
    import torch
    from copy import deepcopy
    from peft import LoraConfig, get_peft_model
    m = get_peft_model(deepcopy(base).to(device), LoraConfig(**LORA))
    miss = m.load_state_dict({k: v.float() for k, v in adapter.items()}, strict=False)
    assert not miss.unexpected_keys, miss.unexpected_keys[:3]
    return m.merge_and_unload().eval()


# ------------------------------------------------------------------ drivers
def run_zs(base, device, smoke=False):
    df, feats, grid = Z.load_panel(PANEL)
    coins, walls = Z.coin_arrays(df, feats, len(grid)), Z.wall_steps(grid)
    head = Head(base, device)
    os.makedirs(OUT, exist_ok=True)
    df[["symbol", "ucid", "date"]].to_parquet(os.path.join(OUT, "rows.parquet"))
    steps = [s for s in range(len(grid)) if walls[s] >= 0]
    if smoke:
        for s in (steps[300], steps[-40]):
            t0 = time()
            got = forecast_steps(head, df, coins, walls, [s])
            print("smoke %s: %d rows forecast, q %s, %.1fs; median q50 %.3f, mean q90-q10 %.3f" % (
                str(grid[s])[:10], len(got["cov"][0]), got["cov"][1].shape, time() - t0,
                float(np.median(got["cov"][1][:, 1])), float((got["cov"][1][:, 2] - got["cov"][1][:, 0]).mean())), flush=True)
        return
    for m in MODES:
        path = os.path.join(OUT, "zs_%s.npz" % m)
        if os.path.exists(path):
            continue
        got = forecast_steps(head, df, coins, walls, steps, modes=(m,), log="zs_" + m)[m]
        save(path, *got)
        print("zs_%s: %d rows" % (m, len(got[0])), flush=True)


def run_ft(base, device, which, smoke=False):
    import torch
    df, feats, grid = Z.load_panel(PANEL)
    coins, walls = Z.coin_arrays(df, feats, len(grid)), Z.wall_steps(grid)
    B = {b: (tr, te) for b, tr, te in blocks(grid)}
    if smoke:
        tr, te = B[11]
        for m in MODES:
            t0 = time()
            merged, adapter, man = finetune(base, pd.Timestamp(tr[0]), m, device, steps=30)
            print("smoke ft %s: %d train / %d val windows, 30 steps %.1fs, peak %.0f MiB, val %s" % (
                m, man["train_windows"], man["val_windows"], time() - t0, torch.cuda.max_memory_allocated() / 2 ** 20, man["val_log"]), flush=True)
        return
    for b in which:
        tr, te = B[b]
        d = os.path.join(OUT, "block_%02d" % b)
        os.makedirs(d, exist_ok=True)
        steps = list(np.searchsorted(grid, np.concatenate([np.asarray(tr, "datetime64[ns]"), np.asarray(te, "datetime64[ns]")])))
        for m in MODES:
            path = os.path.join(d, "ft_%s.npz" % m)
            if os.path.exists(path):
                continue
            merged, adapter, man = finetune(base, pd.Timestamp(tr[0]), m, device)
            torch.save(adapter, os.path.join(d, "%s_adapter.pt" % m))
            t0 = time()
            got = forecast_steps(Head(merged, device), df, coins, walls, steps, modes=(m,))[m]
            save(path, *got)
            man["forecast_minutes"], man["rows"] = round((time() - t0) / 60, 1), int(len(got[0]))
            json.dump(man, open(os.path.join(d, "%s_manifest.json" % m), "w"), indent=1)
            print("block %d %s: cut %s, newest label %s, %d train windows, best step %d (val %s -> %s), fit %.1f min, %d rows in %.1f min" % (
                b, m, man["cut"], man["max_label_date"], man["train_windows"], man["best_step"], man["val_log"][0]["val_loss"],
                min(x["val_loss"] for x in man["val_log"]), man["minutes"], man["rows"], man["forecast_minutes"]), flush=True)
            del merged


def run_audit(base, device, dates):
    """
    For each date T: zero-shot and the block's fine-tuned forecasts recomputed from the panel cut at T
    ({fam}_{mode}_{T}_cut.npz), and the fine-tuned ones also from the uncut panel with the same reloaded
    fp16 adapter ({..}_full.npz). build_c2head compares them with the stored (in-memory model) forecasts.
    """
    import torch
    df, feats, grid = Z.load_panel(PANEL)
    fcoins, fwalls = Z.coin_arrays(df, feats, len(grid)), Z.wall_steps(grid)
    B = {b: (tr, te) for b, tr, te in blocks(grid)}
    os.makedirs(os.path.join(OUT, "audit"), exist_ok=True)
    zs = Head(base, device)
    for T in map(pd.Timestamp, dates):
        b = [k for k, (tr, te) in B.items() if np.datetime64(T) in set(np.asarray(te, "datetime64[ns]"))][0]
        cdf, cfeats, cgrid = Z.load_panel(PANEL, cut=T)
        coins, walls = Z.coin_arrays(cdf, cfeats, len(cgrid)), Z.wall_steps(cgrid)
        s, fs = len(cgrid) - 1, int(np.searchsorted(grid, np.datetime64(T)))
        assert grid[fs] == np.datetime64(T)
        key = lambda d, ids: d.loc[ids, ["ucid"]].assign(date=str(T.date()))
        heads = [("zs", zs)]
        if b in FT_BLOCKS:
            heads += [("ft", None)]
        for fam, h in heads:
            for m in MODES:
                hh = h or Head(merge_adapter(base, torch.load(os.path.join(OUT, "block_%02d" % b, "%s_adapter.pt" % m)), device), device)
                ci, cq = forecast_steps(hh, cdf, coins, walls, [s], modes=(m,))[m]
                np.savez(os.path.join(OUT, "audit", "%s_%s_%s_cut.npz" % (fam, m, T.date())), ucid=cdf.loc[ci, "ucid"].to_numpy(), q=cq, block=b)
                if fam == "ft":
                    fi, fq = forecast_steps(hh, df, fcoins, fwalls, [fs], modes=(m,))[m]
                    np.savez(os.path.join(OUT, "audit", "%s_%s_%s_full.npz" % (fam, m, T.date())), ucid=df.loc[fi, "ucid"].to_numpy(), q=fq, block=b)
        print("audit %s (block %d) done" % (T.date(), b), flush=True)


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["zs", "ft", "audit"])
    ap.add_argument("--blocks", type=int, nargs="+", default=FT_BLOCKS[::-1])
    ap.add_argument("--dates", nargs="+")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    base = base_model(a.device)
    if a.what == "zs":
        run_zs(base, a.device, a.smoke)
    elif a.what == "ft":
        run_ft(base, a.device, a.blocks, a.smoke)
    else:
        run_audit(base, a.device, a.dates)


if __name__ == "__main__":
    sys.exit(main())
